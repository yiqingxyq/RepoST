import json
from tqdm import tqdm 
import os
import re
import time
import argparse

import tiktoken

CODE_DIR = os.environ.get("CODE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")
docker_CACHE_DIR = os.environ.get("docker_CACHE_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from RepoST.prompts import coverage_prompt_template, more_tests_prompt_template
from utils import *
from generation.api_lm import *
from RepoST.test_generation import sanity_check

COVERAGE_CUTOFF = 0.8 # 1.0 for the eval set

def reindent_func(func_code, indent):
    dedented_new_func = textwrap.dedent(func_code)
    return "\n".join([f"{indent}{line}" for line in dedented_new_func.split("\n")])

def replace_test_function(script, test_func_name, new_test_func):
    lines = script.split("\n")
    orig_test_func_start, orig_test_func_end = get_function_line_idx(script, test_func_name, class_name=None)
    if orig_test_func_start is None:
        return None
    
    content = lines[orig_test_func_start].lstrip()
    orig_indent = lines[orig_test_func_start].split(content)[0]
    reindented_new_func = reindent_func(new_test_func, orig_indent)
    
    return "\n".join(lines[:orig_test_func_start] + [reindented_new_func] + lines[orig_test_func_end+1:])


def make_code_block_str(lines, func_name, start_idx, end_idx):
    if start_idx == end_idx:
        title = f"# Line {start_idx}, in {func_name}"
    else:
        title = f"# Line {start_idx}-{end_idx}, in {func_name}"
    
    code_block = "\n".join(["```", title] + lines[start_idx:end_idx+1] + ["```"])
    return code_block

def get_missing_line_blocks(script, func_name, missing_line_list):
    # cov.coverage: start from Line 1
    missing_line_list = [x-1 for x in missing_line_list]
    
    lines = script.split("\n")
    code_blocks = []
    
    start_idx, end_idx = -1, -1
    for line_idx in missing_line_list:
        if start_idx == -1:
            start_idx = end_idx = line_idx
        else:
            if line_idx == end_idx+1:
                end_idx = line_idx
            else:
                # add a block
                code_blocks.append(make_code_block_str(lines, func_name, start_idx, end_idx))
                start_idx = end_idx = line_idx
                
    if start_idx != 0:
        code_blocks.append(make_code_block_str(lines, func_name, start_idx, end_idx))
    
    return "\n\n".join(code_blocks)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_round", type=int)
    parser.add_argument("--input_file", type=str, default="cleaned_python_test_debug_round0.json")
    parser.add_argument("--model_name", type=str, default="claude-3-5-sonnet-20240620") # gpt-4o
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()
    
    input_data_file = os.path.join(dataset_generation_DIR, args.input_file)
    
    if args.debug_round == 0:
        script_key = "full_script"
    else:
        script_key = f"full_script_debug_round{args.debug_round-1}"
    
    coverage_key = f"{script_key}_coverage_rate"
    missing_key = f"{script_key}_coverage_missing_lines"
    output_key = f"full_script_debug_round{args.debug_round}"
    examples = json.load(open(input_data_file, 'r'))
    if args.end_idx != -1:
        examples = examples[: args.end_idx]
    examples = examples[args.start_idx : ]
    
    output_data_file = os.path.join(dataset_generation_DIR, args.input_file.replace(".json", "coverage.json"))
    if args.start_idx != 0 or args.end_idx != -1:
        output_data_file = output_data_file.replace(".json", f"_{args.start_idx}_{args.end_idx}.json")
    
    if os.path.exists(output_data_file):
        new_examples = json.load(open(output_data_file, 'r'))
    else:
        new_examples = [{k:v for k,v in func_dict.items()} for func_dict in examples]
            
    max_tokens = 8192 if "claude" in args.model_name else 16384
    llm = API_LM(args.model_name, max_tokens=max_tokens)
    
    function_unchanged_count = 0
    test_regen_count = 0
    bad_test_count = 0
    good_test_count = 0
    compute_count = 0
    input_token_count, output_token_count = 0, 0
    for idx, (func_dict, new_func_dict) in enumerate(tqdm(zip(examples, new_examples), total=len(examples))):
        
        class_name = func_dict["func_name"].split(".")[0] if "." in func_dict["func_name"] else None
        func_name = func_dict["func_name"].split(".")[-1]
        test_func_name = f"test_{func_name}"
        
        if coverage_key not in func_dict:
            continue
        
        if func_dict[coverage_key] >= COVERAGE_CUTOFF or func_dict[coverage_key] == -1:
            new_func_dict[output_key] = func_dict[script_key]
            good_test_count += func_dict[coverage_key] > 0
            continue
        
        bad_test_count += 1
        
        # add new implementation
        func_script = extract_func(func_dict[script_key], func_name, class_name=class_name)
        new_implementation = rename_func(func_script, func_name, f"{func_name}_new_implementation")
        new_script = insert_new_func_after_exist_func(new_implementation, func_dict[script_key], func_name, class_name=class_name)

        # obtain missing blocks
        if func_dict[coverage_key] != 1.0:
            missing_code = get_missing_line_blocks(new_script, f"{func_name}_new_implementation", func_dict[missing_key])
            prompt = coverage_prompt_template.format(func_name=func_dict["func_name"], test_func_name=test_func_name, code=new_script, missing_code=missing_code, docker_CACHE_DIR=docker_CACHE_DIR)
        else:
            prompt = more_tests_prompt_template.format(func_name=func_dict["func_name"], test_func_name=test_func_name, code=new_script, docker_CACHE_DIR=docker_CACHE_DIR)
        
        input_token_count += llm.get_token_num(prompt)
        
        response_text = llm.generate(messages=[{"role": "user", "content": prompt}])[0]
        output_token_count += llm.get_token_num(response_text)
        compute_count += 1
        
        try:
            # output_script = new_func_dict[output_key]
            output_script = extract_code(response_text)[0][1]
            new_test_func = extract_func(output_script, test_func_name, class_name=None)
            
            script = replace_test_function(func_dict[script_key], test_func_name, new_test_func)
            
            # sanity check
            if sanity_check(script, func_dict["func_name"], func_dict[script_key], func_dict["func_code"]):
                new_func_dict[output_key] = script
                function_unchanged_count += check_func_body_match(script, func_dict["func_code"])
                test_regen_count += 1
            else:
                new_func_dict[output_key] = func_dict[script_key]
        except:
            print("Error: Cannot extract function")
            new_func_dict[output_key] = func_dict[script_key]
            function_unchanged_count += check_func_body_match(script, func_dict["func_code"])
            
        if compute_count % 10 == 0 and compute_count > 0:
            json.dump(new_examples, open(output_data_file, 'w'), indent=4)
            print(f'Saved {function_unchanged_count}/{test_regen_count}/{bad_test_count}/{idx+1} examples.. ({good_test_count} good tests)')
            
    json.dump(new_examples, open(output_data_file, 'w'), indent=4)
    print(f'Saved {function_unchanged_count}/{test_regen_count}/{bad_test_count}/{idx+1} examples.. ({good_test_count} good tests)')
    
    print(f"input: {input_token_count}, output: {output_token_count}")