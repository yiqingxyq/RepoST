import subprocess
import json
import re 
import os 
import argparse
import datetime
import time

import numpy as np
from tqdm import tqdm 

import coverage
import inspect
from io import StringIO
import importlib

CODE_DIR = os.environ.get("docker_CODE_DIR")
CACHE_DIR = os.environ.get("docker_CACHE_DIR")
EXEC_FILE_PREFIX = os.path.join(CACHE_DIR, "execution_test")
assert CACHE_DIR is not None

COVERAGE_CUTOFF = 0.8 # 1.0 for the eval set

import sys 
sys.path.insert(0,CODE_DIR)
from utils import *
from utils_exec import *

sys.path.insert(0, CACHE_DIR)

def branch_coverage_check(file_name, func_name, class_name):
    code = open(file_name).read()
    start_idx, end_idx = get_function_line_idx(code, func_name, class_name)
    func_code = "\n".join( code.split("\n")[start_idx:end_idx+1] )
    branch_count = count_code_branches(func_code)
    if branch_count == 0:
        return 1.0, "No Branch, Coverage Rate = 100%.", []
    
    cov = coverage.Coverage(branch=True, source=["execution_test"], data_file=os.path.join(CACHE_DIR, ".coverage"))
    cov.start()
    
    import execution_test
    importlib.reload(execution_test)
    import execution_test
    test_exist_flag = False
    for name, func in inspect.getmembers(execution_test, inspect.isfunction):
        if name == f"test_{func_name}":  # Test functions typically start with "test_"
            print(f"Running test: {name}")
            func()  # Call the test function
            test_exist_flag = True
            break
    
    cov.stop()
    cov.save()
    
    report_buffer = StringIO()
    cov.report(file=report_buffer)  # Redirect the report to the buffer
    full_report = report_buffer.getvalue()  # Save the full report as a string
    report_buffer.close()
    
    analysis = cov.analysis2(file_name)
    statements, executed, excluded, missing, formatted_missing = analysis
        
    report = cov.report()
    coverage_rate = report / 100
    
    cov.erase()
    
    return coverage_rate, full_report, missing


def skip_non_target_functions(code, func_name, class_name):
    start_idx, end_idx = get_function_line_idx(code, func_name, class_name)
    parent_line_idx_list = get_function_parent_line_idx(code, start_idx)
    code_lines = code.split("\n")
    new_lines = []
    for line_idx, line in enumerate(code_lines):
        if line_idx < start_idx or line_idx > end_idx and line_idx not in parent_line_idx_list:
            if line.strip() and line.strip()[-1] != "\\":
                line = line + " # pragma: no cover"
        new_lines.append(line)
    
    return "\n".join(new_lines)
    

def insert_gt_implementation(script, full_func_name):
    class_name = full_func_name.split(".")[0] if "." in full_func_name else None
    func_name = full_func_name.split(".")[-1]
    
    func_script = extract_func(script, func_name, class_name=class_name)
    new_implementation = rename_func(func_script, func_name, f"{func_name}_new_implementation")
    new_script = insert_new_func_after_exist_func(new_implementation, script, func_name, class_name=class_name)
    return new_script

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='cleaned_python_test_tests.json')
    parser.add_argument("--script_key", type=str, required=True) #e.g., full_script, eval_script
    parser.add_argument("--run_sh_commands", default=False, action='store_true')
    parser.add_argument("--test_gt", default=False, action='store_true')
    parser.add_argument("--check_coverage_mode", default=False, action='store_true')
    args = parser.parse_args()
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    output_file = args.input_file.replace(".json", "_exec.json")
    if args.check_coverage_mode:
        output_file = args.input_file.replace(".json", "_coverage.json")
    if os.path.exists(output_file):
        func_list = json.load(open(output_file))
    else:
        func_list = json.load(open(args.input_file))
    
    num_examples = len(func_list)
    
    sh_commands = []
    coverage_count = 0
    successful_count = 0
    execution_count = 0
    extraction_error_count = 0
    success_idx_set = set()
    for code_idx, func_dict in enumerate(tqdm(func_list)):
        if args.script_key not in func_dict:
            continue 
        
        if args.check_coverage_mode:
            # skip unsuccessful examples
            if f"{args.script_key}_exec_result" not in func_dict or func_dict[f"{args.script_key}_exec_result"][0] != "success":
                continue
            
        if args.check_coverage_mode or args.test_gt:
            class_name = func_dict["func_name"].split(".")[0] if "." in func_dict["func_name"] else None
            func_name = func_dict["func_name"].split(".")[-1]
        
        if args.test_gt:
            code = insert_gt_implementation(func_dict[args.script_key], func_dict["func_name"])
            processed_code = skip_non_target_functions(code, f"{func_name}_new_implementation", class_name) if args.check_coverage_mode else code
        else:
            code = func_dict[args.script_key]
            processed_code = skip_non_target_functions(code, func_name, class_name) if args.check_coverage_mode else code
        if not code or not processed_code:
            extraction_error_count += 1
            continue
            
        if args.check_coverage_mode:
            
            exec_file = f"{EXEC_FILE_PREFIX}.py"
            with open(exec_file, "w") as fout:
                fout.write(processed_code)
                
            execution_count += 1
            successful_count += 1
            
            try:
                cov_rate, full_report, missing = branch_coverage_check(exec_file, func_name, class_name)
                func_dict[f"{args.script_key}_coverage_rate"] = cov_rate
                func_dict[f"{args.script_key}_coverage_report"] = full_report
                func_dict[f"{args.script_key}_coverage_missing_lines"] = missing
                print(f"Coverage rate: {cov_rate}")
                
                if cov_rate > COVERAGE_CUTOFF:
                    coverage_count += 1
                
            except:
                print(f"Coverage check bug")
                func_dict[f"{args.script_key}_coverage_rate"] = -1
                
        else:
            exec_file = f"{EXEC_FILE_PREFIX}.py"
            with open(exec_file, "w") as fout:
                fout.write(code)
                
            # run python code
            res, idx = run_program(f"python {exec_file}", code_idx)
            
            # automatically install packages and re-execute the code
            if args.run_sh_commands:
                installed_list, res = automatic_package_install(f"python {exec_file}", code_idx, res)
                sh_commands.extend(installed_list)

                sh_script_key = f"{args.script_key}_sh"
                if sh_script_key in func_dict:
                    for cmd in func_dict[sh_script_key].split("\n"):
                        if cmd:
                            run_program(cmd, code_idx)
                            sh_commands.append(cmd)
                            
                    
            print([code_idx])
            print(res)
            res = ( str(res[0]), res[1].replace('"',"'"), res[2].replace('"',"'"))
            
            execution_count += 1
            func_dict[f"{args.script_key}_exec_result"] = res
            
            if res[0] == "success":
                success_idx_set.add(func_dict["idx"].split("_")[0])
                successful_count += 1
            
        # clean cache dir
        if execution_count % 500 == 0 and execution_count > 0:
            print(f"Saving results")
            json.dump(func_list, open(output_file, "w"), indent=4)
            
        if execution_count % 1 == 0 and execution_count > 0:
            print(f"Cleaning {CACHE_DIR}")
            run_program(f"rm -rf {CACHE_DIR}/*", code_idx, timeout=30)
            run_program(f"rm -rf {CACHE_DIR}/.[a-z]*", code_idx, timeout=30)
            
    
    print(f"Successfully execute: {coverage_count}/{successful_count}/{execution_count} extraction error count: {extraction_error_count}")
    print(f"Success rate: {successful_count/execution_count}")
    print(f"Covering {len(success_idx_set)} training examples")
    json.dump(func_list, open(output_file, "w"), indent=4)
    
    if args.run_sh_commands and not args.check_coverage_mode:
        new_sh_commands = []
        for c in sh_commands:
            if c not in new_sh_commands:
                new_sh_commands.append(c)
                
        sh_cmd_file = args.input_file.replace(".json", "sh_commands.json" + str(datetime.datetime.now()).replace(" ","."))
        json.dump(new_sh_commands, open(sh_cmd_file,'w'), indent=4)