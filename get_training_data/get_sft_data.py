import json
from tqdm import tqdm 
import os
import re
import time
import argparse

CODE_DIR = os.environ.get("CODE_DIR")

import sys 
sys.path.insert(0,CODE_DIR)
from utils import *


def format_docstring(docstring, func_code, func_name):
    if not docstring:
        return None
    
    func_body_indent = get_func_body_rel_indent(func_code, func_name)
    docstring = docstring + "\n"
    return "\n" + "\n".join([func_body_indent + line.strip() for line in docstring.split("\n")])

    
def wrap_context(docs):
    """Builds the retrieved prompt based on a list of docs"""
    if len(docs) == 0:
        return ""
    
    start_line = "Here are some relevant code fragments from other files of the repo:"
    sep_line = "--------------------------------------------------"
    intro_line = "the below code fragment can be found in:"
    
    title_block = intro_line + "\n" + "__TITLE__" + "\n" + sep_line
    
    retrieved_prompt = start_line + "\n" + sep_line + "\n"
    for doc in docs:
        title, text = doc["title"], doc["text"]
        retrieved_prompt += title_block.replace("__TITLE__", title) + "\n"
        retrieved_prompt += doc["text"] + "\n" + sep_line + "\n"
    
    # add "# " to each line except for the last line
    retrieved_prompt = "\n".join(
        [ "# " + x for x in retrieved_prompt.split("\n")[:-1]]
    ) + "\n" 
    return retrieved_prompt

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/ExecTrain/checked_test_set_final.json")
    parser.add_argument("--output_file", type=str, default="ExecTrain_exec_sft_data.json")
    parser.add_argument("--max_example_num", type=int, default=10000)
    parser.add_argument('--use_instruction', action="store_true")
    parser.add_argument('--remove_docstring', action="store_true")
    parser.add_argument('--infile_mode', action="store_true")
    args = parser.parse_args()
    
    input_file = os.path.join(CODE_DIR, args.input_file)
    output_file = os.path.join( os.path.join(CODE_DIR, "LLaMA-Factory/data"), args.output_file)
    
    data = json.load(open(input_file))
    
    sft_data = []
    error_count = 0
    for idx,example in enumerate(tqdm(data)):
        class_name = example["func_name"].split(".")[0] if "." in example["func_name"] else None
        func_name = example["func_name"].split(".")[-1]
        
        context_blocks = [x[1] for x in extract_code(example["orig_context"])]
        cur_context_block = get_code_block_by_path(context_blocks, example["func_path"])
        if cur_context_block is None:
            error_count += 1
            continue
        
        other_blocks = [x for x in context_blocks if x!=cur_context_block]
        try:
            docs = [{
                "title": block.split("\n")[0].split(" ")[1],
                "text": "\n".join(block.split("\n")[1:]),
            } for block in other_blocks]
        except:
            print(f"Error: cannot extract title from context blocks")
            error_count += 1
            continue
            
        try:
            start_line, end_line = get_function_line_idx(cur_context_block, func_name, class_name=class_name)
            func_code = extract_func(cur_context_block, func_name, class_name=class_name)
            func_indent = get_func_indent(func_code, func_name)

            # get function body
            func_body_start, func_body_end = get_func_body_line_idx(func_code, func_name)
            target = "\n".join(func_code.split("\n")[func_body_start:func_body_end+1])
            target = target + "\n" + func_indent + "\n"
            
            # get docstring (if any)
            orig_docstring = get_docstring(func_code, func_name)
            if args.use_instruction:
                docstring = example["instruction"]
            elif args.remove_docstring:
                docstring = None
            else:
                docstring = orig_docstring if orig_docstring else None

            # mask function body (and add docstring)
            masked_func_code = mask_func_body(func_code, func_name, docstring=format_docstring(docstring, func_code, func_name), add_pass=False)
            
            # remove code after the target function
            new_cur_context_block = "\n".join(cur_context_block.split("\n")[:start_line]) + "\n" + masked_func_code + "\n"

        except:
            # print("Error extracting and masking the function from the evaluation script")
            if args.use_instruction:
                try:
                    docstring = orig_docstring if orig_docstring else None
                    masked_func_code = mask_func_body(func_code, func_name, docstring=format_docstring(docstring, func_code, func_name), add_pass=False)
                    new_cur_context_block = "\n".join(cur_context_block.split("\n")[:start_line]) + "\n" + masked_func_code + "\n"
                except:
                    error_count += 1
                    continue 
            else:
                error_count += 1
                continue
        
        context_text = wrap_context( docs )
        if args.infile_mode:
            context_text = ""
        
        input_prompt = context_text + "\n\n" + new_cur_context_block if context_text else new_cur_context_block
        
        sft_data.append({
            "instruction": input_prompt,
            "input": "",
            "output": target,
            "idx": example["idx"],
        })
        
    print(f"Error count: {error_count}/{len(data)} (Remaining: {len(sft_data)})")
        
    if len(sft_data) <= args.max_example_num:
        json.dump(sft_data, open(output_file, "w"), indent=4)
        
    else:
        split_num = ( len(sft_data) + args.max_example_num - 1 ) // args.max_example_num
        for split_idx in range(split_num):
            start_idx, end_idx = split_idx * args.max_example_num, (split_idx+1) * args.max_example_num
            json.dump(sft_data[start_idx : end_idx], open(output_file.replace(".json", f"_{split_idx}.json"), "w"), indent=4)