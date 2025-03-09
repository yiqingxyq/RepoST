# input data: LLaMA-Factory/data/ExecEval_data.json
# call vllm
# post-processing (extract function)
# make an eval script for execution-based evaluation

import os
import argparse
import json 
import numpy as np 
from tqdm import tqdm 

from prompts import *

CODE_DIR = os.environ.get("CODE_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from utils import *


def postprocess_by_function(generation):
    # check whether generation has function def
    lines = generation.split("\n")
    
    if f"def {func_name}" in generation:
        for idx,line in enumerate(lines):
            if line.lstrip()[:3] == "def" and f"def {func_name}" in line:
                lines = lines[idx:]
                generation = "\n".join(lines)
                break
    
    if lines[0].lstrip()[:3] == "def" or lines[0].lstrip()[0] == "@":
        # skip until the lines ending with "):"
        for lid, line in enumerate(lines):
            if line.rstrip()[-1] == ":" or line.split("#")[0].rstrip()[-1] == ":":
                generation = "\n".join(lines[lid+1:])
                break
            
    elif lines[0].lstrip()[:6] == "import":
        for lid, line in enumerate(lines):
            if not line.strip():
                continue
            elif line.lstrip()[:6] == "import":
                continue 
            elif line.lstrip()[:3] == "def" or line.lstrip()[0] == "@":
                for func_lid, func_line in enumerate(lines[lid:]):
                    if func_line.rstrip()[-1] == ":" or func_line.split("#")[0].rstrip()[-1] == ":":
                        generation = "\n".join(lines[lid+func_lid+1:])
                        break
                break
            else:
                break
    
    first_token = generation.split()[0]
    func_body_indent = generation.split(first_token)[0]
    generation_lines = []
    for line in generation.split('\n'):
        if line.split() and line.split()[0]!='#':
            first_token = line.split()[0]
            indent = line.split(first_token)[0]
            if len(indent) < len(func_body_indent):
                break
            generation_lines.append(line)
        else:
            generation_lines.append(line)
    return "\n".join(generation_lines), func_body_indent


def make_eval_script(generation, eval_script, func_name, class_name=None, mode="new_implementation"):
    if generation[0] == "\n":
        generation = generation[1:]
    
    # extract reasoning
    reasoning = ""
    if "Reasoning: " in generation:
        reasoning = generation.split("Reasoning: ")[1].split("```")[0]
        
    # extract function
    func_code = extract_func(eval_script, func_name, class_name=class_name)
    func_start, func_end = get_function_line_idx(eval_script, func_name, class_name)
    func_body_start, func_body_end = get_func_body_line_idx(func_code, func_name)
    
    # get GT indent 
    gt_first_line = func_code.split("\n")[func_body_start]
    gt_first_token = gt_first_line.split()[0]
    gt_indent = gt_first_line.split(gt_first_token)[0]
    
    # post-process generation
    code_blocks = extract_code(generation)
    generation = code_blocks[0][1] if code_blocks else generation

    clean_generation, gen_indent = postprocess_by_function(generation)
    if gt_indent != gen_indent:
        # dedent generation
        clean_lines = []
        for line in clean_generation.split("\n"):
            if len(line) >= len(gen_indent):
                new_line = gt_indent + line[len(gen_indent):]
                clean_lines.append(new_line)
        clean_generation = "\n".join(clean_lines)

    # get generation function
    target = "\n".join(func_code.split("\n")[func_body_start:func_body_end+1])
    func_lines = func_code.split("\n")
    gen_func_code = "\n".join(func_lines[:func_body_start]) + "\n" + clean_generation
    
    if mode == "replace":
        lines = eval_script.split("\n")
        new_script = "\n".join(lines[:func_start] + [gen_func_code] + lines[func_end+1:])
        
    elif mode == "new_implementation":
        new_implementation = rename_func(gen_func_code, func_name, f"{func_name}_new_implementation")
        new_script = insert_new_func_after_exist_func(new_implementation, eval_script, func_name, class_name=class_name)
    
    return new_script, clean_generation, reasoning


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-11-20")
    parser.add_argument("--cache_save_name", type=str, default="cache")
    parser.add_argument("--backend", type=str, required=True, choices=["vllm", "api"])
    
    parser.add_argument("--prompt_data_file", type=str, default="../LLaMA-Factory/data/ExecCSN_train_sft_data.json")
    parser.add_argument("--data_file", type=str, default="../data/ExecCSN_train/checked_test_set_final.json")
    parser.add_argument("--output_file", type=str, default="results/generations.json")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    parser.add_argument("--max_seq_num", type=int, default=5)
    parser.add_argument("--max_debug_num", type=int, default=2)
    parser.add_argument("--num_per_example", type=int, default=2)
    parser.add_argument("--debug_mode", default=False, action='store_true')
    parser.add_argument("--use_CoT", default=False, action='store_true')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    prompt_data = json.load(open(args.prompt_data_file))
    data = json.load(open(args.data_file))
    idx2example = {str(example["idx"]):example for example in data}
    
    if args.end_idx != -1:
        prompt_data = prompt_data[: args.end_idx]
    prompt_data = prompt_data[args.start_idx : ]
    
    if args.debug_mode:
        idx2success_count = {}
        for prompt_example in prompt_data:
            idx = prompt_example["idx"].split("_")[0] if type(prompt_example["idx"]) == str else str(prompt_example["idx"])
            if idx not in idx2success_count:
                idx2success_count[idx] = 0
            if prompt_example["generation_script_exec_result"][0] == "success":
                idx2success_count[idx] += 1
                
        idx_to_skip = [x for x in idx2success_count if idx2success_count[idx] >= args.num_per_example]
    
    # Generation
    raw_output_file = args.output_file.replace(".json", "_raw.json")
    if not os.path.exists(raw_output_file):
        if args.backend == "vllm":
            from vlm import VLM
            from vllm import SamplingParams
            sampling_params = SamplingParams(n=args.max_seq_num, temperature=0.2, top_p=0.95, ignore_eos=True, logprobs=1, max_tokens=args.max_tokens)
            
            llm = VLM(args.model_name, sampling_params, max_length=args.max_length)
            all_messages = [x["instruction"] for x in prompt_data]
            generation_prompt_data = prompt_data
        
        elif args.backend == "api":
            from api_lm import API_LM, API_Sampling_Params
            sampling_params = API_Sampling_Params(n=args.max_seq_num, temperature=0.2, top_p=0.95, ignore_eos=True, logprobs=1, max_tokens=args.max_tokens)
            
            idx2debug_num = {}
            llm = API_LM(args.model_name, cache_name=args.cache_save_name, sampling_params=sampling_params)
            all_messages = []
            generation_prompt_data = []
            for prompt_example in prompt_data:
                idx = prompt_example["idx"].split("_")[0] if type(prompt_example["idx"]) == str else str(prompt_example["idx"])
                example = idx2example[idx]
                
                if args.debug_mode:
                    if idx not in idx2debug_num:
                        idx2debug_num[idx] = 0
                    
                    if prompt_example["generation_script_exec_result"][0] == "success":
                        continue
                    
                    if idx2debug_num[idx] >= args.max_debug_num:
                        continue
                    
                    example_func_name = "return_none"
                    example_context = "def return_none():\n"
                    example_buggy_solution = "    return 0"
                    example_err_msg = 'Traceback (most recent call last):\n  File "<stdin>", line 1, in <module>\n  File "<stdin>", line 2, in test_return_none\nAssertionError: Function should return None'
                    example_reasoning = "The return_none function returns 0, but the instruction specifies that it should return None."
                    example_solution = "    return None"
                    
                    if not args.use_CoT:
                        all_messages.append([
                            {"role": "system", "content": "Instruction: Debug the BUGGY IMPLEMENTATION of the given Python function based on the CONTEXT and ERROR MESSAGE."},
                            {"role": "system", "name": "example_user", "content": debug_prompt.format(func_name=example_func_name, context=example_context, buggy_solution=example_buggy_solution, error_msg=example_err_msg)},
                            {"role": "system", "name": "example_assistant", "content": debug_target.format(solution=example_solution)},
                            {"role": "user", "content": debug_prompt.format(func_name=example["func_name"], context=prompt_example["instruction"], buggy_solution=prompt_example["generation"], error_msg=prompt_example["generation_script_exec_result"][-1]) },
                        ])
                    else:
                        all_messages.append([
                            {"role": "system", "content": "Instruction: Debug the BUGGY IMPLEMENTATION of the given Python function based on the CONTEXT and ERROR MESSAGE.\n\nProvide your reasoning."},
                            {"role": "system", "name": "example_user", "content": debug_reasoning_prompt.format(func_name=example_func_name, context=example_context, buggy_solution=example_buggy_solution, error_msg=example_err_msg)},
                            {"role": "system", "name": "example_assistant", "content": debug_reasoning_target.format(reasoning=example_reasoning, solution=example_solution)},
                            {"role": "user", "content": debug_reasoning_prompt.format(func_name=example["func_name"], context=prompt_example["instruction"], buggy_solution=prompt_example["generation"], error_msg=prompt_example["generation_script_exec_result"][-1]) },
                        ])
                    idx2debug_num[idx] += 1
                    
                else:
                    if not args.use_CoT:
                        all_messages.append([
                            {"role": "system", "content": "Instruction: Continue writing the code. Provide your reasoning."},
                            {"role": "system", "name": "example_user", "content": "Continue writing the following code:\n\n```\ndef return_none():\n```"},
                            {"role": "system", "name": "example_assistant", "content": "```\n    return None\n```"},
                            {"role": "user", "content": "Continue writing the following code:\n\n```\n" + prompt_example["instruction"] + '\n```'},
                        ])
                    else:
                        all_messages.append([
                            {"role": "system", "content": "Instruction: Continue writing the code. Provide your reasoning."},
                            {"role": "system", "name": "example_user", "content": "Continue writing the following code. Provide your reasoning and code below SOLUTION:\n\n```\ndef return_none():\n```\n\nYour answer should follow the format below:\nReasoning: ...\n```python\n# Your Code. \n```\n\nSOLUTION:"},
                            {"role": "system", "name": "example_assistant", "content": "```\n    return None\n```"},
                            {"role": "user", "content": "Continue writing the following code and provide your reasoning below SOLUTION:\n\n```\n" + prompt_example["instruction"] + '\n```' + "\n\nYour answer should follow the format below:\nReasoning: ...\n```python\n# Your Code. \n```\n\nSOLUTION:"},
                        ])
                    
                generation_prompt_data.append(prompt_example)
    
        generations = llm.generate_all(all_messages)
        
        outputs = [{"idx":prompt_example["idx"], "instruction": prompt_example["instruction"], "generations": gen} for prompt_example,gen in zip(generation_prompt_data, generations)]
        
        raw_output_file = args.output_file.replace(".json", "_raw.json")
        json.dump(outputs, open(raw_output_file, "w"), indent=4)
    
    else:
        raw_output_file = args.output_file.replace(".json", "_raw.json")
        outputs = json.load(open(raw_output_file))
    
    gen_count, total_count = 0, 0
    generation_data = []
    for result in outputs:
        idx = result["idx"].split("_")[0] if type(result["idx"]) == str else str(result["idx"])
        instruction = result["instruction"]
        
        example = idx2example[idx]
        eval_script = example["eval_script"]
        class_name = example["func_name"].split(".")[0] if "." in example["func_name"] else None
        func_name = example["func_name"].split(".")[-1]
        
        for local_idx, generation in enumerate(result["generations"]):
            total_count += 1
            if not generation:
                continue 
            
            try:
                new_script, clean_generation, reasoning = make_eval_script(generation, eval_script, func_name, class_name)
                generation_data.append({
                    "idx": f"{result['idx']}_{local_idx}", 
                    "instruction": instruction,
                    "generation_script": new_script,
                    "generation": clean_generation,
                    "reasoning": reasoning,
                    "raw_generation": generation,
                })
                gen_count += 1
            except:
                pass
            
    print(f"Saving {gen_count}/{total_count} generations for evaluation..")
    json.dump(generation_data, open(args.output_file, "w"), indent=4)