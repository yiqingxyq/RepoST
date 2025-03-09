import os
import json
import glob
import subprocess

dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

data_dict = json.load(open( os.path.join(dataset_generation_DIR, "cleaned_python_test.json")  ))

repo_set = set()
for example in data_dict:
    repo_set.add(example["repo_name"])
    
print("Number of selected repos:", len(repo_set))
    
all_repos = glob.glob( os.path.join(dataset_generation_DIR, "repos/*") )
    
for repo_path in all_repos:
    repo_name = repo_path.split("/")[-1]
    if repo_name not in repo_set:
        return_code = subprocess.call(f"rm -r {repo_path}", shell=True)
        
subprocess.call(f"ls {dataset_generation_DIR}/repos | wc -l", shell=True)

all_repos = glob.glob( os.path.join(dataset_generation_DIR, "repos/*") )
print("Number of selected repos:", len(all_repos))