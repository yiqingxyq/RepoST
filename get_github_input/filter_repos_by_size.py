import os
import json
import argparse
from tqdm import tqdm
import shutil
import git

from r2e.paths import REPOS_DIR, EXTRACTION_DIR


def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            # Skip if it's a broken symlink
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size / (1024 * 1024)  # Convert to megabytes
      
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_repo_size", type=float, default=2)
    args = parser.parse_args()
        
    # remove repos exceeding certain size
    repos = os.listdir(REPOS_DIR)
    for repo in repos:
        repo_path = os.path.join(REPOS_DIR, repo)
        dir_size = get_directory_size(repo_path)
        if dir_size > args.max_repo_size:
            print(f"Removing repo: {repo} with size {dir_size}..")
            shutil.rmtree(repo_path)
                
    repos = os.listdir(REPOS_DIR)
    print(f"Successfully cloning {len(repos)} repos!")