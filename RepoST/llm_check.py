import json
from tqdm import tqdm 
import os
import re
import time
import argparse

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from utils import *
from RepoST.prompts import sandbox_check_prompt_template, test_check_prompt_template
from generation.api_lm import *


def extract_answer(text):
    if not text or "REASONING:" not in text or "ANSWER:" not in text:
        return None, None

    reasoning = text.split("ANSWER")[0].strip()
    answer_text = text.split("ANSWER")[-1].lower().strip()
    
    label2pos = {label:answer_text.find(label) for label in ["same", "yes", "no"]} 
    # label2pos = {label:answer_text.find(label) for label in ["same", "minor", "major"]}
    label2pos = {k:v for k,v in label2pos.items() if v>0}
    answer = min(label2pos, key=lambda k:label2pos[k]) if label2pos else None
    
    return reasoning, answer


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='test_set_final.json')
    parser.add_argument("--model_name", type=str, default="gpt-4o") # claude-3-5-sonnet-20240620
    parser.add_argument("--mode", type=str, choices=["sandbox", "test", "combine"], required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()
    
    input_file = os.path.join(dataset_generation_DIR, args.input_file)
    
    if args.mode == "combine":
        output_file = os.path.join(dataset_generation_DIR, f"checked_{args.input_file}")
        examples = json.load(open(input_file, 'r'))
        for mode in ["sandbox", "test"]:
            fname = os.path.join(dataset_generation_DIR, f"{mode}_check_{args.input_file}")
            new_examples = json.load(open(fname))
            for example, new_example in zip(examples, new_examples):
                example.update(new_example)
        
        checked_examples = []
        for x in examples:
            if x["sandbox_functionality_check"]["answer"] in ["same", "yes"] or x["sandbox_ast_check"]:
                if x["test_correctness_check"]["answer"] == "yes":
                    checked_examples.append(x)
                    
        repo_set = {x["repo_name"] for x in checked_examples}
        
        json.dump(checked_examples, open(output_file, "w"), indent=4)
        print(f"Saved {len(checked_examples)}/{len(examples)} = {len(checked_examples)/len(examples)} checked examples")
        print(f"Covering {len(repo_set)} repos")
        exit()
    
    output_file = os.path.join(dataset_generation_DIR, f"{args.mode}_check_{args.input_file}")
    if args.start_idx != 0 or args.end_idx != -1:
        output_file = output_file.replace(".json", f"_{args.start_idx}_{args.end_idx}.json")
    
    if os.path.exists(output_file):
        examples = json.load(open(output_file, 'r'))
    else:
        examples = json.load(open(input_file, 'r'))
        if args.end_idx != -1:
            examples = examples[: args.end_idx]
        examples = examples[args.start_idx : ]
    
    max_tokens = 1024
    llm = API_LM(args.model_name, max_tokens=max_tokens)
    
    compute_count = 0
    successful_count = 0
    almost_successful_count = 0
    same_count = 0
    for idx,func_dict in enumerate(tqdm(examples)):
        
        class_name = func_dict["func_name"].split(".")[0] if "." in func_dict["func_name"] else None
        func_name = func_dict["func_name"].split(".")[-1]
        test_func_name = f"test_{func_name}"
        
        code = func_dict["eval_script"]
        orig_func = func_dict["orig_func"]
        
        if args.mode == "sandbox":
            prompt = sandbox_check_prompt_template.format(func_name=func_name, orig_func=orig_func, new_code=code)
            saved_key = "sandbox_functionality_check"
            
            if check_func_ast_match(code, orig_func, func_name, class_name):
                func_dict["sandbox_ast_check"] = True 
            else:
                func_dict["sandbox_ast_check"] = False
            
        elif args.mode == "test":
            saved_key = "test_correctness_check"
            
            try:
                # add new implementation
                func_script = extract_func(code, func_name, class_name=class_name)
                new_implementation_script = rename_func(func_script, func_name, f"{func_name}_new_implementation")
                
                func_body_indent = get_func_body_rel_indent(func_script, func_name)
                assert func_body_indent is not None
                new_implementation = mask_func_body(new_implementation_script, func_name, docstring=f"\n{func_body_indent}The actual implementation is hidden.\n{func_body_indent}", add_pass=True)
                
                new_script = insert_new_func_after_exist_func(new_implementation, code, func_name, class_name=class_name)
            except:
                print(f"Parsing error, skip this example")
                func_dict[saved_key] = {"reasoning": "Parsing error", "answer": "no"}
                continue
            
            prompt = test_check_prompt_template.format(func_name=func_name, test_func_name=test_func_name, code=new_script)
            
            
        if saved_key in func_dict:
            continue
        
        response_text = llm.generate(messages=[{"role": "user", "content": prompt}])[0]
        
        reasoning, answer = extract_answer(response_text)
        if answer:
            func_dict[saved_key] = {"reasoning": reasoning, "answer": answer}
            
            same_count += answer in ["same"]
            successful_count += answer in ["same", "yes"]
            almost_successful_count += answer in ["same", "yes", "minor"]
            compute_count += 1
        else:
            print("Generation error! Cannot extract answer")
            print(response_text)
        
        if compute_count % 10 == 0 and compute_count > 0:
            print(f"{same_count}/{successful_count}/{almost_successful_count}/{compute_count} examples are successful")
            json.dump(examples, open(output_file, 'w'), indent=4)
            print(f'Saved {compute_count}/{(idx+1)} examples to file..')
            
    print(f"{same_count}/{successful_count}/{almost_successful_count}/{compute_count} examples are successful")
    json.dump(examples, open(output_file, 'w'), indent=4)