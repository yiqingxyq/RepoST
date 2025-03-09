# filter out context containing the GT answer 

import json

# retriever_name = "gist"
retriever_name = "openai_reranked"

input_file = f"repoeval-function-4k-{retriever_name}_nofilter.jsonl"
output_file = f"repoeval-function-4k-{retriever_name}_filter.jsonl"

data = [json.loads(line) for line in open(input_file)]

doc_num_before = 0
doc_num_after = 0
for example in data:
    filename = "-".join(example["metadata"]["fpath_tuple"][1:])
    
    new_docs = []
    for doc in example["docs"]:
        doc_num_before += 1
        if filename in doc["title"] and example["reference"] in doc["text"]:
            continue 
        
        new_docs.append(doc)
        doc_num_after += 1
        
    example["docs"] = new_docs
    
with open(output_file, "w") as fout:
    for example in data:
        fout.write(json.dumps(example) + "\n")
        
print(f"{doc_num_after}/{doc_num_before}")