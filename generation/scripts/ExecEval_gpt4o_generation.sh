MODEL_NAME="gpt-4o-2024-11-20"
MODEL_SAVE_NAME="gpt4o"
top_k=1

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python generate.py \
    --model_name $MODEL_NAME --backend api \
    --data_file "../data/ExecEval/checked_test_set_final.json" \
    --prompt_data_file "../LLaMA-Factory/data/ExecEval_data.json" \
    --output_file "../data/ExecEval/results/ExecEval_${MODEL_SAVE_NAME}_context_top${top_k}.json" --max_seq_num $top_k \
    --cache_save_name ExecEval