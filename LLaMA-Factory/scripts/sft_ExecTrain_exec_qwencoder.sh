CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/qwencoder_ExecTrain_exec_sft.yaml
rm -r saves/qwen/full/ExecTrain_exec_sft/checkpoint*