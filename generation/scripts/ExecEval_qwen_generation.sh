MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_SAVE_NAME="qwencoder"
top_k=1

CUDA_VISIBLE_DEVICES=0  PYTHONPATH=./ python generate.py \
    --model_name $MODEL_NAME --backend vllm \
    --data_file "../data/ExecEval/checked_test_set_final.json" \
    --prompt_data_file "../LLaMA-Factory/data/ExecEval_data.json" \
    --output_file "../data/ExecEval/results/ExecEval_${MODEL_SAVE_NAME}_top${top_k}.json" \
    --max_seq_num $top_k --max_tokens 1024 \
    --cache_save_name ExecEval
