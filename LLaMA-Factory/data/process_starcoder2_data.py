import json

filename = "ExecEval_exec_sft_data.json"

data = json.load(open(filename))
for example in data:
    example["instruction"] = example["instruction"].rstrip()
    example["output"] = "\n" + example["output"]
    
output_filename = filename.replace(".json", "_starcoder2.json")
json.dump(data, open(output_filename, "w"), indent=4)