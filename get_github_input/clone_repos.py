import os
import json
import argparse
from tqdm import tqdm
import shutil
import git

from r2e.paths import REPOS_DIR, EXTRACTION_DIR


def clone_repo_from_url(full_repo_name: str):
    repo_url = f"https://github.com/{full_repo_name}.git"
    
    repo_username, repo_name = (
        repo_url.rstrip("/").removesuffix(".git").split("/")[-2:]
    )
    local_repo_clone_path = REPOS_DIR / f"{repo_username}___{repo_name}"

    if os.path.exists(local_repo_clone_path):
        # print(f"Repository {repo_url} already exists at {local_repo_clone_path}... skipping")
        return
    
    try:
        git.Repo.clone_from(f"{repo_url}", local_repo_clone_path)
        print(f"Successfully cloned repository {repo_url} to {local_repo_clone_path}")
    except:
        print(f"Error cloning from {repo_url}")
      
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_names_file", type=str, default='results/repo_names_sampled.txt')
    args = parser.parse_args()

    # load repo_list
    repo_list = open(args.repo_names_file).read().split('\n')
    repo_list = [x for x in repo_list if x]
    
    # clone repos
    print(f"Cloning {len(repo_list)} repos to {REPOS_DIR}..")
    for full_repo_name in tqdm(repo_list):
        clone_repo_from_url(full_repo_name)
                
    repos = os.listdir(REPOS_DIR)
    print(f"Successfully cloning {len(repos)} repos!")