MODEL_NAME="claude-3-5-sonnet-20240620"
top_k=5

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python generate.py \
    --model_name $MODEL_NAME --backend api --debug_mode \
    --data_file "../data/ExecTrain/checked_test_set_final.json" \
    --prompt_data_file "../data/ExecTrain/rejection_sampling/ExecTrain_claude_top${top_k}_exec.json" \
    --output_file "../data/ExecTrain/rejection_sampling/ExecTrain_claude_top${top_k}_debug.json" --max_seq_num 1 \
    --cache_save_name ExecTrain
