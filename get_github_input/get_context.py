import os
import json
import argparse
from tqdm import tqdm

from r2e.generators.testgen import TestGenArgs, R2ETestGenerator
from r2e.utils.data import load_functions
from r2e.paths import EXTRACTED_DATA_DIR
from r2e.utils.data import write_functions

dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

def write_chat_hist(chat_hist_list, chat_hist_file):
    json.dump(chat_hist_list, open(chat_hist_file, 'w'), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default='random50_sampled20')
    parser.add_argument("--context_type", type=str, default='sliced')
    parser.add_argument("--max_context_size", type=int, default=6000)
    args = parser.parse_args()
    
    # obtain context
    input_func_file = EXTRACTED_DATA_DIR / f"{args.exp_id}_extracted.json"
    functions = load_functions(input_func_file)
    
    SPLIT_SIZE = 1000
    block_num = (len(functions)+SPLIT_SIZE-1) // SPLIT_SIZE
    intermediate_dir = EXTRACTED_DATA_DIR / f"{args.exp_id}_context"
    os.makedirs(intermediate_dir, exist_ok=True)
    for block_idx in range(block_num):
        start_idx, end_idx = SPLIT_SIZE * block_idx, SPLIT_SIZE * (block_idx+1)
        
        output_func_file = intermediate_dir / f"{args.exp_id}_context_{block_idx}.json"
        chat_hist_file = intermediate_dir / f"{args.exp_id}_chat_hist_{block_idx}.json"
        
        if os.path.exists(output_func_file):
            continue
        
        test_gen_args = TestGenArgs(exp_id=args.exp_id, context_type="sliced", max_context_size=6000)
        tasks = R2ETestGenerator.prepare_tasks(test_gen_args, functions[start_idx:end_idx])
        
        # save files
        print(f"Saving results for block {block_idx}")
        functions_w_context = [x.func_meth for x in tasks]
        chat_hist_list = [x.chat_messages for x in tasks]
        write_functions(functions_w_context, output_func_file)
        write_chat_hist(chat_hist_list, chat_hist_file)
        
    # aggregate data
    data = []
    for block_idx in range(block_num):
        output_func_file = intermediate_dir / f"{args.exp_id}_context_{block_idx}.json"
        data.extend(json.load(open(output_func_file)))
        
    output_func_file = EXTRACTED_DATA_DIR / f"{args.exp_id}_context.json"
    json.dump(data, open(output_func_file, "w"), indent=4)
    
    # stats
    output_func_file = EXTRACTED_DATA_DIR / f"{args.exp_id}_context.json"
    functions_w_context = json.load(open(output_func_file))
    multi_file_context_count = len([x for x in functions_w_context if x["context"]["context"].count("```python") > 1 ])
    print(f"Examples with multi-file context: {multi_file_context_count}/{len(functions_w_context)}")
    
    # process into RepoST's format
    os.makedirs(dataset_generation_DIR, exist_ok=True)
    RepoST_input_file = os.path.join(dataset_generation_DIR, "cleaned_python_test.json")
    
    RepoST_inputs = []
    for idx,func_dict in enumerate(functions_w_context):
        if "function_id" in func_dict:
            func_name = func_dict["function_id"]["identifier"].split(".")[-1]
            func_code = func_dict["function_code"]
            
        elif "method_id" in func_dict:
            func_name = ".".join( func_dict["method_id"]["identifier"].split(".")[-2:] )
            func_code = func_dict["method_code"]
            
        assert "-" not in func_dict["file"]["file_module"]["module_id"]["identifier"]
        RepoST_inputs.append({
            "func_name":   func_name,
            "idx":         str(idx),
            "func_code":   func_code,
            "context":     func_dict["context"]["context"],
            "repo_name":   func_dict["file"]["file_module"]["repo"]["repo_id"],
            "func_path":   func_dict["file"]["file_module"]["module_id"]["identifier"].replace(".","/") + ".py",
        })
        
    print(f"Successfully obtained context for {len(RepoST_inputs)} examples!")
    json.dump(RepoST_inputs, open(RepoST_input_file, "w"), indent=4)