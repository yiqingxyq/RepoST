import os
import json 
import argparse
import numpy as np

from r2e.paths import REPOS_DIR, EXTRACTION_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default='temp')
    parser.add_argument("--func_num", type=int, default=10)
    args = parser.parse_args()
    
    FUNC_NUM = args.func_num

    results_file_docstring = str(EXTRACTION_DIR / f"{args.exp_id}_docstring_extracted.json")
    results_file_no_docstring = str(EXTRACTION_DIR / f"{args.exp_id}_no_docstring_extracted.json")
    results_docstring = json.load(open(results_file_docstring))
    if os.path.exists(results_file_no_docstring):
        results_no_docstring = json.load(open(results_file_no_docstring))
    else:
        results_no_docstring = []
    
    output_results_file = str(EXTRACTION_DIR / f"{args.exp_id}_sampled{args.func_num*2}_extracted.json")
    
    # sample k functions and k methods from each repo
    # prioritizing functions w/ docstrings 
    docstring_func_set = set()
    for func_dict in results_docstring:
        full_repo_name = func_dict["file"]["file_module"]["repo"]["repo_org"] + "/" + func_dict["file"]["file_module"]["repo"]["repo_name"]
        func_type = "function" if "function_id" in func_dict else "method"
        
        full_func_id = full_repo_name + "." + func_dict[f"{func_type}_id"]["identifier"]
        docstring_func_set.add(full_func_id)
        
        
    repo2func_list = {}
    covered_func_set = set()
    for func_dict in results_docstring + results_no_docstring:
        full_repo_name = func_dict["file"]["file_module"]["repo"]["repo_org"] + "/" + func_dict["file"]["file_module"]["repo"]["repo_name"]
        
        func_type = "function" if "function_id" in func_dict else "method"
        full_func_id = full_repo_name + "." + func_dict[f"{func_type}_id"]["identifier"]
        
        if full_repo_name not in repo2func_list:
            repo2func_list[full_repo_name] = []
        
        if full_func_id not in covered_func_set:
            func_dict["has_docstring_flag"] = full_func_id in docstring_func_set
            func_dict["type"] = func_type
            repo2func_list[full_repo_name].append(func_dict)
            covered_func_set.add(full_func_id)
            
            
    # randomly select k functions and k methods
    np.random.seed(1)
    
    sampled_results = []
    for repo_name, func_dict_list in repo2func_list.items():
        selected_funcs = []
        for func_type in ["function", "method"]:
            func_dict_list_type = [x for x in func_dict_list if x["type"]==func_type and '-' not in x[f"{func_type}_id"]["identifier"]]
            if len(func_dict_list_type) <= FUNC_NUM:
                selected_funcs.extend(func_dict_list_type)
            else:
                func_dict_list_docstring = [x for x in func_dict_list_type if x["has_docstring_flag"]]
                if len(func_dict_list_docstring) >= FUNC_NUM:
                    selected_funcs.extend( list(np.random.choice(func_dict_list_docstring, FUNC_NUM, replace=False)) )
                else:
                    selected_funcs.extend(func_dict_list_docstring)
                    
                    wo_func_num = FUNC_NUM - len(func_dict_list_docstring)
                    func_dict_list_without = [x for x in func_dict_list_type if not x["has_docstring_flag"]]
                    selected_funcs.extend( list(np.random.choice(func_dict_list_without, wo_func_num, replace=False)) )
            
        sampled_results.extend(selected_funcs)
        
    json.dump(sampled_results, open(output_results_file, 'w'), indent=4)
    
    # stats 
    num_docstring = len([x for x in sampled_results if x["has_docstring_flag"]])
    print(f"Sampled {len(sampled_results)} functions. {num_docstring}/{len(sampled_results)} have docstrings.")
    
    output_file = f"results/{args.exp_id}_concrete_examples.py"
    with open(output_file, 'w') as fout:
        for idx,x in enumerate(sampled_results):
            func_content = x[f"{x['type']}_code"]
            fout.write(f"[{idx}]\n")
            fout.write(func_content)
            fout.write("\n\n\n\n")    