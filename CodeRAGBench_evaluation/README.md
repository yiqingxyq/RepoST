# RepoEval evaluation

## Set up Environment for Code Generation
```
conda env create -n repost python=3.11 -y
conda activate repost

cd CodeRAGBench_evaluation
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Download Repos in RepoEval
```
cd repoeval_evaluation
mkdir repos
cd repos
wget https://github.com/yiqingxyq/repoeval_debug/raw/main/function_level.zip
```

## Build a Separate Environment for RepoEval
```
conda env create --file eval/tasks/custom_metrics/repoeval_environment.yml -n repoeval
```

First, set up the repositories and check whether all the tests can run successfully:
```
PYTHONPATH=./ python eval/tasks/custom_metrics/repoeval_execution.py
```

To solve the torch import bug, you may need to set:
```
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Evaluation
If all the tests are passed, you can run evaluation using Huggingface or local checkpoints. We provide two example scripts (using the ground truth context) here:
```
bash scripts/eval_repoeval_8k_qwen.sh

bash scripts/eval_repoeval_8k_ExecTrain_exec_sft.sh
```
