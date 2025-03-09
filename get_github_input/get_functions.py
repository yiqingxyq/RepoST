import os
import json
import argparse
from tqdm import tqdm
import git

from check_repo_stats import check_repo_stats

from r2e.repo_builder.repo_args import RepoArgs
from r2e.repo_builder.extract_func_methods import build_functions_and_methods
from r2e.paths import REPOS_DIR, EXTRACTION_DIR


def extract_functions(disable_no_docstring=False, exp_id="temp"):
    repo_args = RepoArgs(
        overwrite_extracted=True, 
        disable_no_docstring=disable_no_docstring, 
        exp_id=exp_id
    )
    build_functions_and_methods(repo_args)
    

def print_stats(exp_id):
    repo_list = os.listdir(REPOS_DIR)
    results_file = str(EXTRACTION_DIR / f"{exp_id}_extracted.json")
    results = json.load(open(results_file))
    repo_list_w_function = list({x["file"]["file_module"]["repo"]["local_repo_path"] for x in results})
    print(f"Successfully extracted at least one functions from {len(repo_list_w_function)}/{len(repo_list)} repos")
    
    with open("results/error_repos.txt", 'w') as fout:
        for repo in repo_list:
            if repo not in repo_list_w_function:
                fout.write(repo+'\n')
                
    covered_repos = [repo.replace("___",'/') for repo in repo_list_w_function]
    no_docstring_id = exp_id if "no_docstring" in exp_id else exp_id.replace("docstring", "no_docstring")
    if os.path.exists(f"results/{no_docstring_id}_repo_stats_func_filter.json"):
        stats_by_repo_name = json.load(open(f"results/{no_docstring_id}_repo_stats_func_filter.json"))
    else:
        stats_by_repo_name = {}
    
    if "no_docstring" not in exp_id and len(stats_by_repo_name) >= 0:
        stats_by_repo_name = {k:v for k,v in stats_by_repo_name.items() if k in covered_repos}
    else:
        stats_by_repo_name = check_repo_stats(covered_repos, stats_by_repo_name)
        json.dump(stats_by_repo_name, open(f"results/{exp_id}_repo_stats_func_filter.json", 'w'), indent=4)
    
    num_setup = len([k for k,v in stats_by_repo_name.items() if v["setup_file"]])
    print(f"Num repos: {len(stats_by_repo_name)}/{len(covered_repos)}")
    print(f"Percent with setup files: {num_setup}/{len(stats_by_repo_name)}={num_setup/len(stats_by_repo_name)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default='temp')
    parser.add_argument('--disable_no_docstring', action='store_true')
    args = parser.parse_args()
    
    # get functions
    print(f"Getting functions..")
    extract_functions(disable_no_docstring=args.disable_no_docstring, exp_id=args.exp_id)
    
    print(f"Geting stats..")
    print_stats(args.exp_id)