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
from RepoST.prompts import test_generation_prompt_template
from utils import *
from generation.api_lm import *


def sanity_check(script, func_name, input_code, input_func):
    MAIN_FUNC_STR = 'if __name__ == "__main__":'
    MAIN_FUNC_STR2 = "if __name__ == '__main__':"
    
    # check whether the new script contains the function
    class_name = func_name.split(".")[0] if "." in func_name else None
    func_name = func_name.split(".")[-1]
    start_line, end_line = get_function_line_idx(script, func_name, class_name=class_name)
    if start_line is None:
        print(f"function not exist: {func_name}")
        return False 
    
    # The newly implemented focal method does not exist
    new_func_name = f"{func_name}_new_implementation"
    new_start_line, new_end_line = get_function_line_idx(script, new_func_name, class_name=class_name)
    if new_start_line is not None:
        print(f"error! the newly implemented function should not exist: {new_func_name}")
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
    if output_len < input_len:
        print(f"Length error: input {input_len}, output {output_len}")
        return False
    
    # test functions
    test_func_name = f"test_{func_name}"
    test_start_line, test_end_line = get_function_line_idx(script, test_func_name)
    if test_start_line is None:
        print(f"test function not exist: {func_name}")
        return False 
    
    # asserts
    test_code = "\n".join(script.split("\n")[test_start_line:test_end_line+1])
    assert_count = test_code.lower().count('assert') # raise AssertionException also counts
    if assert_count < 3:
        print(f"error: not enough asserts (found {assert_count}, expected 3)")
        return False 
    
    # call the focal method at least once 
    if f"{func_name}(" not in test_code:
        print(f"error: does not call the focal method")
        return False  
    
    # call the newly implemented focal method at least once 
    if f"{func_name}_new_implementation(" not in test_code:
        print(f"error: does not call the new implementation of the focal method")
        return False  
    
    # Have a __main__ function to call the tests
    if MAIN_FUNC_STR not in script and MAIN_FUNC_STR2 not in script:
        print(f"error: main function does not exist")
        return False
    
    return True


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="cleaned_python_test_sandboxed.json")
    parser.add_argument("--model_name", type=str, default="claude-3-5-sonnet-20240620") # gpt-4o
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()
    
    input_data_file = os.path.join(dataset_generation_DIR, args.input_file)
    output_data_file = os.path.join(dataset_generation_DIR, "cleaned_python_test_tests.json")
    if args.start_idx != 0 or args.end_idx != -1:
        output_data_file = os.path.join(dataset_generation_DIR, f"cleaned_python_test_tests_{args.start_idx}_{args.end_idx}.json")
    
    examples = json.load(open(input_data_file, 'r'))
    if args.end_idx != -1:
        examples = examples[: args.end_idx]
    examples = examples[args.start_idx : ]

    if os.path.exists(output_data_file):
        new_examples = json.load(open(output_data_file, 'r'))
    else:
        new_examples = [{k:v for k,v in func_dict.items()} for func_dict in examples]
            
    max_tokens = 8192 if "claude" in args.model_name else 16384
    llm = API_LM(args.model_name, max_tokens=max_tokens)
    
    function_unchanged_count = 0
    test_generated_count = 0
    sandboxed_count = 0
    compute_count = 0
    input_token_count, output_token_count = 0, 0
    for idx, (func_dict, new_func_dict) in enumerate(tqdm(zip(examples, new_examples), total=len(examples))):
        
        if "sandboxed_script" not in func_dict:
            continue
        
        sandboxed_script = remove_main(func_dict["sandboxed_script"])
        new_func_dict["sandboxed_script"] = sandboxed_script
                
        test_func_name = "test_" + func_dict["func_name"].split(".")[-1]
        prompt = test_generation_prompt_template.format(func_name=func_dict["func_name"], test_func_name=test_func_name, code=sandboxed_script, docker_CACHE_DIR=docker_CACHE_DIR)
        
        if "full_script" in new_func_dict:
            function_unchanged_count += check_func_body_match(new_func_dict["full_script"], func_dict["func_code"])
            test_generated_count += 1
            sandboxed_count += 1
            continue
        
        sandboxed_script_len = llm.get_token_num(sandboxed_script)
        if sandboxed_script_len > max_tokens:
            print(f"Skip this example. Script length: {sandboxed_script_len}")
            continue
            
        input_token_count += llm.get_token_num(prompt)
        
        response_text = llm.generate(messages=[{"role": "user", "content": prompt}])[0]

        output_token_count += llm.get_token_num(response_text)
        compute_count += 1
        sandboxed_count += 1
        
        try:
            script = extract_code(response_text)[0][1]
            
            # remove the newly implemented function
            class_name = func_dict["func_name"].split(".")[0] if "." in func_dict["func_name"] else None
            func_name = func_dict["func_name"].split(".")[-1]
            new_func_name = f"{func_name}_new_implementation"
            script = remove_function_if_exist(script, new_func_name, class_name=class_name)
            
            # sanity check
            if sanity_check(script, func_dict["func_name"], sandboxed_script, func_dict["func_code"]):
                new_func_dict["full_script"] = script
                function_unchanged_count += check_func_body_match(script, func_dict["func_code"])
                test_generated_count += 1
        except:
            print("Error: Cannot extract function")
        
        if compute_count % 10 == 0 and compute_count > 0:
            json.dump(new_examples, open(output_data_file, 'w'), indent=4)
            print(f'Saved {function_unchanged_count}/{test_generated_count}/{sandboxed_count}/{idx+1} examples..')
            
    json.dump(new_examples, open(output_data_file, 'w'), indent=4)
    print(f'Saved {function_unchanged_count}/{test_generated_count}/{sandboxed_count}/{idx+1} examples..')
    
    print(f"input: {input_token_count}, output: {output_token_count}")