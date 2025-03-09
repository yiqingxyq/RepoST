MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_SAVE_NAME="qwencoder"
PROMPT_NAME="gt_filter"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python main.py --task "repoeval-function" --model $MODEL_NAME --dataset_path "json" \
    --data_files_test "rag_results/repoeval-function-4k-${PROMPT_NAME}.jsonl" \
    --save_generations --save_generations_path results/repoeval-function_${PROMPT_NAME}_${MODEL_SAVE_NAME}_generations.json \
    --metric_output_path results/repoeval-function_${PROMPT_NAME}_${MODEL_SAVE_NAME}_evaluation_results.json \
    --max_length_input 7192 --max_length_generation 8192 --precision auto \
    --save_every_k_tasks 100 --ignore_eos --model_backend vllm --new_tokens_only --topk_docs 5

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python main.py --task "repoeval-function" --model $MODEL_NAME --dataset_path "json" \
    --data_files_test "rag_results/repoeval-function-4k-${PROMPT_NAME}.jsonl" \
    --load_generations_path results/repoeval-function_${PROMPT_NAME}_${MODEL_SAVE_NAME}_generations.json \
    --metric_output_path results/repoeval-function_${PROMPT_NAME}_${MODEL_SAVE_NAME}_evaluation_results.json \
    --max_length_input 7192 --max_length_generation 8192 --precision auto \
    --save_every_k_tasks 100 --ignore_eos --model_backend vllm --new_tokens_only --topk_docs 5 \
    --allow_code_execution