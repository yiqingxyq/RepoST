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
from RepoST.prompts import debug_prompt_template
from utils import *
from generation.api_lm import *


MAIN_FUNC_STR = 'if __name__ == "__main__":'
MAIN_FUNC_STR2 = "if __name__ == '__main__':"
def sanity_check(script, func_name, input_code, input_func):
    # check whether the new script contains the function
    class_name = func_name.split(".")[0] if "." in func_name else None
    func_name = func_name.split(".")[-1]
    start_line, end_line = get_function_line_idx(script, func_name, class_name=class_name)
    if start_line is None:
        print(f"function not exist: {func_name}")
        return False 
    
    # check the "omitting" keywords
    for omit_keyword in ["...", "remains unchanged"]:
        if omit_keyword not in input_code and omit_keyword in script:
            print(f"keyword error: {omit_keyword}")
            return False
        
    # compare length
    func_code = "\n".join(script.split("\n")[start_line:end_line+1])
    input_func_len, output_func_len = count_code_tokens(input_func), count_code_tokens(func_code)
    if output_func_len + 20 < input_func_len:
        print(f"Length error: input function {input_func_len}, output function {output_func_len}")
        return False
    
    input_len, output_len = count_code_tokens(input_code), count_code_tokens(script)
    if output_len + 20 < input_len:
        print(f"Length error: input {input_len}, output {output_len}")
        return False
    
    # test functions
    test_func_name = f"test_{func_name}"
    start_line, end_line = get_function_line_idx(script, test_func_name)
    if start_line is None:
        print(f"test function not exist: {func_name}")
        return False 
    
    # asserts
    test_code = "\n".join(script.split("\n")[start_line:end_line+1])
    assert_count = test_code.lower().count('assert') # raise AssertionException also counts
    if assert_count < 3:
        print(f"error: not enough asserts (found {assert_count}, expected 3)")
        return False 
    
    # call the focal method at least once 
    if f"{func_name}" not in test_code.replace(test_func_name, ""):
        print(f"error: does not call the focal method")
        return False  
    
    # Have a __main__ function to call the tests
    if MAIN_FUNC_STR not in script and MAIN_FUNC_STR2 not in script:
        print(f"error: main function does not exist")
        return False
    
    return True


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_round", type=int)
    parser.add_argument("--input_file", type=str, default="cleaned_python_test_tests_exec.json")
    parser.add_argument("--model_name", type=str, default="claude-3-5-sonnet-20240620") # gpt-4o
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()
    
    if args.debug_round == 0:
        input_data_file = os.path.join(dataset_generation_DIR, args.input_file)
        script_key = "full_script"
    else:
        input_data_file = os.path.join(dataset_generation_DIR, args.input_file.replace("tests_exec.json", f"debug_round{args.debug_round-1}_exec.json"))
        script_key = f"full_script_debug_round{args.debug_round-1}"
        
    exec_key = f"{script_key}_exec_result"
    output_key = f"full_script_debug_round{args.debug_round}"
    sh_output_key = f"full_script_debug_round{args.debug_round}_sh"
    examples = json.load(open(input_data_file, 'r'))
    if args.end_idx != -1:
        examples = examples[: args.end_idx]
    examples = examples[args.start_idx : ]

    output_data_file = os.path.join(dataset_generation_DIR, args.input_file.replace("tests_exec.json", f"debug_round{args.debug_round}.json"))
    if args.start_idx != 0 or args.end_idx != -1:
        output_data_file = output_data_file.replace(".json", f"_{args.start_idx}_{args.end_idx}.json")
    
    if os.path.exists(output_data_file):
        new_examples = json.load(open(output_data_file, 'r'))
    else:
        new_examples = [{k:v for k,v in func_dict.items()} for func_dict in examples]
            
    max_tokens = 8192 if "claude" in args.model_name else 16384
    llm = API_LM(args.model_name, max_tokens=max_tokens)
        
    function_unchanged_count = 0
    debugged_count = 0
    bug_count = 0
    successful_count = 0
    compute_count = 0
    input_token_count, output_token_count = 0, 0
    for idx, (func_dict, new_func_dict) in enumerate(tqdm(zip(examples, new_examples), total=len(examples))):
        
        class_name = func_dict["func_name"].split(".")[0] if "." in func_dict["func_name"] else None
        func_name = func_dict["func_name"].split(".")[-1]
        test_func_name = f"test_{func_name}"
        
        if exec_key not in func_dict:
            continue
        
        if func_dict[exec_key][0] == "success":
            new_func_dict[output_key] = func_dict[script_key]
            function_unchanged_count += check_func_body_match(func_dict[script_key], func_dict["func_code"])
            successful_count += 1
            continue
        
        if output_key in new_func_dict:
            debugged_count += new_func_dict[output_key] != func_dict[script_key]
            continue
        
        bug_count += 1
        
        # add new implementation
        func_script = extract_func(func_dict[script_key], func_name, class_name=class_name)
        new_implementation = rename_func(func_script, func_name, f"{func_name}_new_implementation")
        new_script = insert_new_func_after_exist_func(new_implementation, func_dict[script_key], func_name, class_name=class_name)
        
        err_msg = func_dict[exec_key][2]
        prompt = debug_prompt_template.format(func_name=func_dict["func_name"], test_func_name=test_func_name, code=new_script, err_msg=err_msg, docker_CACHE_DIR=docker_CACHE_DIR)
        
        input_token_count += llm.get_token_num(prompt)
        
        response_text = llm.generate(messages=[{"role": "user", "content": prompt}])[0]
        output_token_count += llm.get_token_num(response_text)
        compute_count += 1
        
        try:
            code_blocks = extract_code(response_text)
            script = code_blocks[0][1]
            sh_script = "\n".join([x[1] for x in code_blocks if x[0]=="bash"])
            
            # remove the newly implemented function
            new_func_name = f"{func_name}_new_implementation"
            script = remove_function_if_exist(script, new_func_name, class_name=class_name)
            
            if sanity_check(script, func_dict["func_name"], func_dict[script_key], func_dict["func_code"]):
                new_func_dict[output_key] = script
                function_unchanged_count += check_func_body_match(script, func_dict["func_code"])
                
                if sh_script:
                    new_func_dict[sh_output_key] = sh_script
                
                debugged_count += 1
            else:
                new_func_dict[output_key] = func_dict[script_key]
        except:
            print("Error: Cannot extract function")
            new_func_dict[output_key] = func_dict[script_key]
        
        if compute_count % 10 == 0 and compute_count > 0:
            json.dump(new_examples, open(output_data_file, 'w'), indent=4)
            print(f'Saved {function_unchanged_count}/{debugged_count}/{bug_count}/{idx+1} examples.. ({successful_count} successful)')
            
    json.dump(new_examples, open(output_data_file, 'w'), indent=4) 
    print(f'Saved {function_unchanged_count}/{debugged_count}/{bug_count}/{idx+1} examples.. ({successful_count} successful)')
    
    print(f"input: {input_token_count}, output: {output_token_count}")