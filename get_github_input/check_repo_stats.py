import os 
import json

from github import Github, Auth
from tqdm import tqdm

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
auth = Auth.Token(GITHUB_TOKEN)
github_api = Github(auth=auth)

def check_repo_stats(repo_list, stats_by_repo_name={}):  
    for i, repo_name in tqdm(enumerate(repo_list), total=len(repo_list)):
        if repo_name not in stats_by_repo_name.keys():
            stats_by_repo_name[repo_name] = {
                'topics': [],
                'stars': -1,
                'contributors_count': -1,
                'setup_file': "",
            }
            try:
                repo = github_api.get_repo(repo_name)
                stats_by_repo_name[repo_name]['topics'] = list(repo.get_topics())
                stats_by_repo_name[repo_name]['stars'] = repo.stargazers_count
                stats_by_repo_name[repo_name]['contributors_count'] = len(list(repo.get_contributors()))
                contents = repo.get_contents("")
            except:
                print(f"Error when reading repos: {repo_name}")
                continue
            
            for content in contents:
                if content.type != "dir":
                    filename = content.path
                    if "setup.py" in filename or "pyproject.toml" in filename:
                        stats_by_repo_name[repo_name]["setup_file"] = filename
            
    return stats_by_repo_name


if __name__ == "__main__":
    repo_list = open("results/repo_names_sampled.txt").read().split("\n")
    repo_list = [x for x in repo_list if x]

    stats_by_repo_name = check_repo_stats(repo_list)
    json.dump(stats_by_repo_name, open("results/repo_stats.json", 'w'), indent=4)

    # stats_by_repo_name = json.load(open("results/repo_stats.json"))

    num_setup = len([k for k,v in stats_by_repo_name.items() if v["setup_file"]])
    print(f"Num repos: {len(stats_by_repo_name)}/{len(repo_list)}")
    print(f"Percent with setup files: {num_setup}/{len(stats_by_repo_name)}={num_setup/len(stats_by_repo_name)}")