import json
from tqdm import tqdm 
import os
import re
import time
import argparse

import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

CODE_DIR = os.environ.get("CODE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")
docker_CACHE_DIR = os.environ.get("docker_CACHE_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from RepoST.prompts import aggregate_and_sandbox_prompt_template, sandbox_prompt_template
from utils import *
from generation.api_lm import *


def get_code_block_by_path(context_blocks, path):
    for block in context_blocks:
        lines = [x for x in block.split("\n") if x and x[0] == "#"]
        if len(lines) == 0:
            return None 
        first_line = lines[0]
        if first_line:
            if path in first_line:
                return block 
    return None


def sanity_check(script, func_name, input_code, input_func):
    # check whether the new script contains the function
    class_name = func_name.split(".")[0] if "." in func_name else None
    func_name = func_name.split(".")[-1]
    start_line, end_line = get_function_line_idx(script, func_name, class_name=class_name)
    if start_line is None:
        print(f"function not exist: {func_name}")
        return False 
    
    # check the "omitting" keywords
    func_code = "\n".join(script.split("\n")[start_line:end_line+1])
    for omit_keyword in ["...", "remains unchanged"]:
        if omit_keyword not in input_code and omit_keyword in func_code:
            print(f"keyword error: {omit_keyword}")
            return False
        
    # compare length
    input_func_len, output_func_len = count_code_tokens(input_func), count_code_tokens(func_code)
    if output_func_len + 20 < input_func_len:
        print(f"Length error: input function {input_func_len}, output function {output_func_len}")
        return False
    
    input_len, output_len = count_code_tokens(input_code), count_code_tokens(script)
    if output_len + 50 < input_len:
        print(f"Length error: input {input_len}, output {output_len}")
        return False
    
    return True


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="cleaned_python_test.json")
    parser.add_argument("--model_name", type=str, default="claude-3-5-sonnet-20240620") # gpt-4o
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()
    
    input_data_file = os.path.join(dataset_generation_DIR, args.input_file)
    output_data_file = os.path.join(dataset_generation_DIR, "cleaned_python_test_sandboxed.json")
    if args.start_idx != 0 or args.end_idx != -1:
        output_data_file = os.path.join(dataset_generation_DIR, f"cleaned_python_test_sandboxed_{args.start_idx}_{args.end_idx}.json")
    
    examples = json.load(open(input_data_file, 'r'))
    if args.end_idx != -1:
        examples = examples[: args.end_idx]
    examples = examples[args.start_idx : ]

    if os.path.exists(output_data_file):
        new_examples = json.load(open(output_data_file, 'r'))
    else:
        new_examples = [
            {k:v for k,v in func_dict.items() if k in ["func_name", "idx", "func_code", "context", "repo_name", "func_path"]}
            for func_dict in examples
        ]
    
    max_tokens = 8192 if "claude" in args.model_name else 16384
    llm = API_LM(args.model_name, max_tokens=max_tokens)
    
    function_unchanged_count = 0
    sandboxed_count = 0
    compute_count = 0
    input_token_count, output_token_count = 0, 0
    for idx, (func_dict, new_func_dict) in enumerate(tqdm(zip(examples, new_examples), total=len(examples))):
        
        context_blocks = [x[1] for x in extract_code(func_dict["context"])]
        cur_context_block = get_code_block_by_path(context_blocks, func_dict["func_path"])
        if cur_context_block is None:
            continue 
        other_blocks = [x for x in context_blocks if x!=cur_context_block]
                
        if "sandboxed_script" in new_func_dict:
            function_unchanged_count += check_func_body_match(new_func_dict["sandboxed_script"], func_dict["func_code"])
            sandboxed_count += 1
            continue
                
        if len(other_blocks) == 0:
            prompt = sandbox_prompt_template.format(func_name=func_dict["func_name"], code=cur_context_block, docker_CACHE_DIR=docker_CACHE_DIR)
        else:
            context = "\n\n".join([f"```python\n{block}\n```" for block in other_blocks])
            prompt = aggregate_and_sandbox_prompt_template.format(func_name=func_dict["func_name"], code=cur_context_block, context=context, docker_CACHE_DIR=docker_CACHE_DIR)
            
        input_token_count += llm.get_token_num(prompt)
        
        response_text = llm.generate(messages=[{"role": "user", "content": prompt}])[0]

        output_token_count += llm.get_token_num(response_text)
        compute_count += 1
        
        try:
            script = extract_code(response_text)[0][1]
            
            if sanity_check(script, func_dict["func_name"], "\n".join(context_blocks), func_dict["func_code"]):
                new_func_dict["sandboxed_script"] = script
                function_unchanged_count += check_func_body_match(script, func_dict["func_code"])
                sandboxed_count += 1
        except:
            print("Error: Cannot extract function")
        
        if compute_count % 10 == 0 and compute_count > 0:
            json.dump(new_examples, open(output_data_file, 'w'), indent=4)
            print(f'Saved {function_unchanged_count}/{sandboxed_count}/{idx+1} examples..')
            
    json.dump(new_examples, open(output_data_file, 'w'), indent=4)    
    print(f'Saved {function_unchanged_count}/{sandboxed_count}/{idx+1} examples..')
    
    print(f"input: {input_token_count}, output: {output_token_count}")