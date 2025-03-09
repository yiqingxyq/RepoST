MODEL_NAME="../LLaMA-Factory/saves/qwen/full/ExecTrain_exec_sft"
MODEL_SAVE_NAME="qwencoder_ExecTrain_exec_sft"

CUDA_VISIBLE_DEVICES=0 python main.py --task "humaneval-unstripped" \
    --model $MODEL_NAME \
    --dataset_path "openai_humaneval" \
    --save_generations --save_generations_path results/humaneval_${MODEL_SAVE_NAME}_generations.json \
    --save_references --save_references_path results/humaneval_${MODEL_SAVE_NAME}_references.json \
    --metric_output_path results/humaneval_${MODEL_SAVE_NAME}_evaluation_results.json \
    --allow_code_execution --new_tokens_only --add_linebreak