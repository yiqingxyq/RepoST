import os
import argparse
import random

SEED=1

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='repo_names.txt')
    parser.add_argument("--output_file", type=str, default='repo_names_sampled.txt')
    parser.add_argument("--sampled_size", type=int, default=50)
    args = parser.parse_args()
    
    repo_list = open(args.input_file).read().split("\n")
    repo_list = list(set([x for x in repo_list if x]))
    random.Random(SEED).shuffle(repo_list)
    
    with open(args.output_file, 'w') as fout:
        for name in repo_list[:args.sampled_size]:
            fout.write(name+'\n')