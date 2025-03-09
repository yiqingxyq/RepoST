import json
from tqdm import tqdm 
import os
import re
import time
import argparse
import numpy as np

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from utils import *

COVERAGE_CUTOFF = 0.8 # 1.0 for the eval set

def get_stats(data, clean_data):
    repos, final_repos = set(), set()
    coverage_rates = []
    ast_same_count = 0
    for example in data:
        repos.add(example["repo_name"])
            
    for example in tqdm(clean_data):
        final_repos.add(example["repo_name"])
        if example["coverage_report"] != "No Branch, Coverage Rate = 100%.":
            coverage_rates.append(example["coverage_rate"])
            
        class_name = example["func_name"].split(".")[0] if "." in example["func_name"] else None
        func_name = example["func_name"].split(".")[-1]
        ast_same_count += check_func_ast_match(example["eval_script"], example["orig_func"], func_name, class_name)
    
    print(f"Num examples with AST match: {ast_same_count}/{len(clean_data)}={ast_same_count/len(clean_data)}")
    print(f"Num repos: {len(final_repos)}/{len(repos)}={len(final_repos)/len(repos)}")
    print(f"Num examples with multiple branches: {len(coverage_rates)}/{len(clean_data)}={len(coverage_rates)/len(clean_data)}")
    print(f"Average branch coverage rate: {np.mean(coverage_rates)}")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="cleaned_python_test_debug_round0_exec.json")
    parser.add_argument("--output_file", type=str, default="test_set_final.json")
    parser.add_argument("--script_key", type=str, default="full_script_debug_round0")
    args = parser.parse_args()

    exec_result_key = f"{args.script_key}_exec_result"
    coverage_result_key = f"{args.script_key}_coverage_rate"
    coverage_report_key = f"{args.script_key}_coverage_report"

    orig_data = json.load(open( os.path.join(dataset_generation_DIR, args.input_file) ))
    data = [x for x in orig_data if exec_result_key in x and x[exec_result_key][0] == "success" and x[coverage_result_key] >= COVERAGE_CUTOFF]

    clean_data = []
    for example in data:
        new_example = {k:v for k,v in example.items() 
            if k in ["func_name", "idx", "repo_name", "func_path"]
        }
        new_example["orig_func"] = example["func_code"]
        new_example["orig_context"] = example["context"]
        new_example["eval_script"] = example[args.script_key]
        new_example["coverage_rate"] = example[coverage_result_key]
        new_example["coverage_report"] = example[coverage_report_key]
        clean_data.append(new_example)
        
    get_stats(orig_data, clean_data)
        
    print(f"Saving {len(clean_data)} examples..")
    json.dump(clean_data, open(os.path.join(dataset_generation_DIR, args.output_file), "w"), indent=4)