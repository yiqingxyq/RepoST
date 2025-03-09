import json
from tqdm import tqdm 
import os
import re
import time
import argparse
import numpy as np

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
final_dataset_DIR = os.environ.get("final_dataset_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from utils import *


def replay(data, idx2gt_example):
    rej_idx_set = {x["idx"].split("_")[0] for x in data}
    
    replay_data = []
    for idx in idx2gt_example:
        if idx not in rej_idx_set:
            replay_data.append(idx2gt_example[idx])
            
    return replay_data


def format_target(target, eval_script, func_name):
    # remove empty targets (due to parsing error)
    lines = target.split("\n")
    empty_flag = True
    for line in lines:
        if line.lstrip() and line.strip()[0] != "#":
            empty_flag = False
    if empty_flag:
        print(f"empty answer:")
        print(target)
        return None
    
    # obtain GT function indent 
    gt_func_indent = ""
    lines = eval_script.split("\n")
    for line in lines[::-1]:
        if f"def {func_name}" in line:
            clean_code = line.lstrip()
            gt_func_indent = line.split(clean_code)[0]
            
    # obtain function indent (if any)
    func_indent = ""
    lines = target.split("\n")
    for line in target.split("\n"):
        if f"def {func_name}" in line:
            clean_code = line.lstrip()
            func_indent = line.split(clean_code)[0]
        
    # obtain function body indent    
    func_body_indent = func_indent
    for line in target.split("\n"):
        if line.lstrip() and line.strip()[0] != "#":
            clean_code = line.lstrip()
            line_indent = line.split(clean_code)[0]
            if len(line_indent.replace("\t", "    ")) > len(func_indent.replace("\t", "    ")):
                func_body_indent = line_indent
                break
            
    # remove comments with incorrect indent
    clean_lines = []
    for line in target.split("\n"):
        if line.lstrip():
            clean_code = line.lstrip()
            line_indent = line.split(clean_code)[0]
            if len(line_indent) > len(func_indent):
                clean_lines.append(line)
        else:
            clean_lines.append(line)
            
    # do not start with blank lines
    line_start_idx = 0
    for idx,line in enumerate(clean_lines):
        if line.strip():
            line_start_idx = idx
            break
    clean_lines = clean_lines[line_start_idx:]
            
    # remove docstring (if any)
    content_start_idx = 0
    if '"""' in clean_lines[0]:
        # has docstring 
        if clean_lines[0].count('"""') == 2:
            content_start_idx = 1
        else:
            for idx in range(1, len(clean_lines)):
                if '"""' in clean_lines[idx]:
                    # end of docstring 
                    content_start_idx = idx + 1
                    break
    
    clean_target = "\n".join(clean_lines[content_start_idx:])
    
    output_code = clean_target + f"\n{gt_func_indent}\n"

    return output_code

def sample_data(idx2examples, idx2gt_example, num_per_example, diff_first=True):
    sft_data = []
    for idx in idx2examples:
        gt = idx2gt_example[idx]["output"]
        
        if len(idx2examples[idx]) <= num_per_example:
            sft_data.extend(idx2examples[idx])
        else:
            if diff_first:
                # sort
                idx2examples[idx] = sorted(idx2examples[idx], key=lambda x:x["idx"])
                idx2examples[idx] = sorted(idx2examples[idx], key=lambda x:len(x["idx"]))
                
                # prioritize diverse examples
                clean_data = []
                for res in idx2examples[idx]:
                    if res["output"] not in clean_data and res["output"] != gt:
                        clean_data.append(res)
                idx2examples[idx] = clean_data + idx2examples[idx]
                
                sft_data.extend(idx2examples[idx][:num_per_example])
            else:
                sampled_ids = np.random.choice(range(len(idx2examples[idx])), num_per_example, replace=False)
                sft_data.extend([idx2examples[idx][i] for i in sampled_ids])
    
    return sft_data
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="checked_test_set_final.json")
    parser.add_argument("--sft_data_file", type=str, default="ExecTrain_exec_sft_data.json")
    parser.add_argument("--generation_files", type=str, default="rejection_sampling/ExecTrain_gpt4o_top5_exec.json")
    parser.add_argument("--output_file", type=str, default="ExecTrain_rejection_sampling_data.json")
    parser.add_argument("--num_per_example", type=int, default=2)
    parser.add_argument("--diff_first", default=False, action='store_true')
    parser.add_argument("--use_cot", default=False, action='store_true')
    parser.add_argument("--replay", default=False, action='store_true')
    args = parser.parse_args()
    
    input_file = os.path.join( final_dataset_DIR, args.input_file)
    sft_data_file = os.path.join( os.path.join(CODE_DIR, "LLaMA-Factory/data"), args.sft_data_file)
    output_file = os.path.join( os.path.join(CODE_DIR, "LLaMA-Factory/data"), args.output_file)
    
    idx2func_dict = {x["idx"]:x for x in json.load(open(input_file))}
    idx2gt_example = {x["idx"]:x for x in json.load(open(sft_data_file))}
    
    generation_data = []
    for gen_file in args.generation_files.split(","):
        generation_file = os.path.join(final_dataset_DIR, gen_file)
        data = json.load(open(generation_file))
        
        for example in data:
            example["source"] = gen_file
                
        generation_data.extend(data)
    
    idx2examples = {}
    for idx,example in enumerate(tqdm(generation_data)):
        if example["generation_script_exec_result"][0] != "success":
            continue
        
        idx = example["idx"].split("_")[0] if type(example["idx"]) == str else str(example["idx"])
        if idx not in idx2examples:
            idx2examples[idx] = []
            
        orig_example = idx2func_dict[idx]
        sft_example = idx2gt_example[idx]
        target = format_target(example["generation"], example["instruction"], orig_example["func_name"].split(".")[-1])
        
        if target:
            idx2examples[idx].append({
                "instruction": sft_example["instruction"],
                "input": "",
                "output": target,
                "idx": example["idx"],
            })
        
    sft_data = sample_data(idx2examples, idx2gt_example, args.num_per_example, diff_first=args.diff_first)
    
    if args.replay:
        # replay idx that only presents once
        replay_data = replay(sft_data, idx2gt_example)
        print(f"Replaying {len(replay_data)} examples..")
        sft_data = sft_data + replay_data
        
    print(f"Saving {len(sft_data)} training instances for {len(idx2examples)}/{len(idx2gt_example)} ({len(idx2examples)/len(idx2gt_example)}) examples")
    json.dump(sft_data, open(output_file, "w"), indent=4)