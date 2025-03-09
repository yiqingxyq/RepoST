MODEL_NAME="claude-3-5-sonnet-20240620"
top_k=5

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python generate.py \
    --model_name $MODEL_NAME --backend api \
    --data_file "../data/ExecTrain/checked_test_set_final.json" \
    --prompt_data_file "../LLaMA-Factory/data/ExecTrain_exec_sft_data.json" \
    --output_file "../data/ExecTrain/rejection_sampling/ExecTrain_claude_top${top_k}.json" --max_seq_num $top_k \
    --cache_save_name ExecTrain_reasoning