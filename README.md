# RepoST: Scalable Repository-Level Coding Environment Construction with Sandbox Testing

<p align="left">
  <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/license-MIT-blue"></a>
  <a href="https://arxiv.org/abs/2503.07358"><img src="https://img.shields.io/badge/arXiv-2503.07358-b31b1b.svg"></a>
</p>

Code for "RepoST: Scalable Repository-Level Coding Environment Construction with Sandbox Testing" ([Arxiv](https://arxiv.org/abs/2503.07358), [Website](https://repost-code-gen.github.io/))


<img width="1015" alt="image" src="https://github.com/user-attachments/assets/90b77667-a44c-4f9c-ad0e-c9f47cec4333" />



If you find our paper or code useful, please cite the paper:
```
@misc{xie2025repost,
      title={RepoST: Scalable Repository-LevelCoding Environment Construction with Sandbox Testing}, 
      author={Yiqing Xie and Alex Xie and Divyanshu Sheth and Pengfei Liu and Daniel Fried and Carolyn Rose},
      year={2025},
      eprint={2503.07358},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.07358}, 
}
```



<br><br>
## Organization
* [Format of the RepoST-Train and RepoST-Eval data](#data): The structure of our datasets.

* [Evaluation on RepoST-Eval](#repost-eval): conducting evaluation on RepoST-Eval. The code includes:
  * (1) loading the RepoST-Eval docker image for execution,
  * (2) running code generation for RepoST-Eval examples, and
  * (3) executing our evaluation scripts in the docker to compute Pass@k scores.
  
  We follow [R2E](https://r2e.dev/) and provide the "Ground-Truth" code context, which is constructed by extracting the modules directly or indirectly called by the target function. In principle, you can also clone the repos and retrieve context or apply agent frameworks.
  
* [Training with RepoST-Train (Vanilla SFT)](#repost-train-sft): the process of setting up our training environment, RepoST-Train, and using it to construct training data. We provide the code to run vanilla SFT (with ground truth functions as targets) using the [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory/tree/main) framework, where you do not need to set up dockers.

* [Training with RepoST-Train (Rejection Sampling)](#repost-train-rs): We also provide the code to run rejection sampling, which includes
  * (1) loading the RepoST-Train docker image,
  * (2) generating candidate solutions on training examples,
  * (3) executing the candidate solutions,
  * (4, optional) debugging failed candidate solutions, and
  * (5) running model training using LlamaFactory.

* [Evaluation on Public Datasets](#eval-public): After training the model, we provide the code to run evaluation on HumanEval and RepoEval-func.

* [Build Your Own Executable Code Generation Environments](#repost): You can also build your own executable environments using RepoST. The steps include:
  * (1) Curating GitHub Repos, Functions, and Context
  * (2) Sandboxing
  * (3) Test Generation
  * (4) Iterative Execution & Debugging
  * (5) Post-Checking
  



<br><br><br>
<a id="data"></a>
## Dataset Structure
The RepoST-Train and RepoST-Eval datasets have the same structure. They both consist of (1) a set of GitHub repos (with specific commit id), (2) a set of target functions inside each repo, and (3) an evaluation script for each target function that can be executed in our docker environment.

We provide the data files in `data/ExecTrain/checked_test_set_final.json` (and `data/ExecEval/checked_test_set_final.json`). The file is a list of dicts. Each dict represents one example and is in the following format:
```
{
  "func_name": "VNetwork.forward",                                                 # the (class name and) function name of the target function
  "idx": "708",                                                                    # the idx of the example
  "repo_name": "seakers___sacTransformerEOS",                                      # the repo name
  "commit_id": "669fb18ee47ae5947b578e392be25c1517f6d73b",                         # the commit id
  "func_path": "scripts/sac.py",                                                   # the local file path that contains the target function
  "orig_context": "```python\n## scripts/sac.py\nimport torch ...",                # the "ground truth" code context extracted from the repo
  "eval_script": "## scripts/sac.py\nimport torch\n\nimport torch.nn as nn ...",   # the evaluation script generated in the RepoST framework, used for execution-based evaluation
  "coverage_rate": 1.0,                                                            # coverage rate
  "sandbox_ast_check": true,                                                       # the following three are post-check results
  "sandbox_functionality_check": ...,
  "test_correctness_check": ..., 
}
```

<br><br><br>
<a id="repost-eval"></a>
## Evaluation on RepoST-Eval

### Setup and Installation
Create a new environment:
```
conda env create -n repost python=3.11 -y
conda activate repost
```

Set up the environment for code generation and RepoST environment construction (Note that you probably need a separate environment for model training):
```
pip install -r requirements.txt
```

And specify code and data paths:
```
source setup_eval.sh
```

### Load the RepoST-Eval Docker Image for Execution
We provide the [RepoST-Eval docker image](https://drive.google.com/file/d/1whs__889_1zlW6D24WnlEL4PhoZzNRym/view?usp=sharing).

You can use the image by 
```
docker load -i repost_eval_image.tar
```

and create a docker container that links the data folders (specified in setup_eval.sh)
```
docker run --name=execeval -v ${final_dataset_DIR}:${docker_final_dataset_DIR} -it repost_eval:v0
```

### Run Generation
We provide the code for running inference under `generation/`.
We use [vllm](https://github.com/vllm-project/vllm) to run code generation for open-source models and use [litellm](https://www.litellm.ai/) for API-based models.

Here are the example scripts, where we evaluate Pass@1 for GPT-4o and QwenCoder-7B.
```
cd generation
bash scripts/ExecEval_gpt4o_generation.sh
bash scripts/ExecEval_qwen_generation.sh
```

The results will be saved under `data/ExecEval/results/`, for example, `data/ExecEval/results/ExecEval_gpt4o_context_top1.json`. 
Our code will automatically copy the generated function to the evaluation script (with test cases) to check the correctness of the generation results (e.g., `generation_script`)


### Execution-based Evaluation in the Docker
After the previous step, you should see the generation file under ${docker_final_dataset_DIR} (specified in setup_eval.sh) inside the docker container.

Assume your generation file is `data/ExecEval/results/ExecEval_gpt4o_context_top1.json` and the dict key that stores the scripts is `generation_script`, you can run execution with:
```
# in the docker container
cd RepoST
source setup_eval.sh
python RepoST/execution.py --input_file ../ExecEval/results/ExecEval_gpt4o_context_top1.json --script_key generation_script
```

The execution results for each evaluation example will be stored in `data/ExecEval/results/ExecEval_gpt4o_context_top1_exec.json`.




<br><br><br>
<a id="repost-train-sft"></a>
## Training with RepoST-Train (Vanilla SFT)
We provide the vanilla sft data we used in our paper in `LLaMA-Factory/data/ExecTrain_exec_sft_data.json` and provide the code for model training under `LLaMA-Factory`.

If you want to process the data on your own, we provide the code for processing training data under `get_training_data`, which processes `data/ExecTrain/checked_test_set_final.json` into the training data format.

### Setup and Installation for model training
Create a new environment:
```
conda env create -n llama python=3.11 -y
conda activate llama
```

Set up the environment for model training (Note that you probably need a separate environment for code generation and RepoST environment construction):
```
cd LLaMA-Factory
pip install -e .
```

And specify code and data paths:
```
source setup_train.sh
```

### Data Processing
We provide the vanilla sft data in `LLaMA-Factory/data/ExecTrain_exec_sft_data.json`
You can also process the data again by running:
```
cd get_training_data
bash scripts/get_full_sft_data.sh
```
The data file will be generated under `LLaMA-Factory/data`.


### Model Training
Before training, you first need to add the data information in `LLaMA-Factory/data/dataset_info.json`. For example, adding an entry:
```
  "ExecTrain_exec_sft": {
    "file_name": "ExecTrain_exec_sft_data.json"
  },
```

After that, write a config file. We provide an example in `LLaMA-Factory/examples/train_full/qwencoder_ExecTrain_exec_sft.yaml`.

Finally, run training by calling `llamafactory-cli train`. We provide an example here:
```
cd LLaMA-Factory
bash scripts/sft_ExecTrain_exec_qwencoder.sh
```





<br><br><br>
<a id="repost-train-rs"></a>
## Training with RepoST-Train (Rejection Sampling)
Similar to vanilla SFT, we provide the rejection sampling (distill) data we used in our paper in `LLaMA-Factory/data/ExecTrain_rejection_sampling_claudegpt_debug_replay_data.json` and provide the code for model training under `LLaMA-Factory`. Please directly refer to the [Model Training Instructions](#model-training-rs) if you want to train with the provided data.

If you want to run rejection sampling on your own, we provide the code for generating candidate solutions under `generation/`, the code for execution in `RepoST/execution.py`, and the code to process training data under `get_training_data/`.

### Setup and Installation for both code generation and model training
We recommend using different environments for code generation and for model training. Specifically:

For code generation:
```
conda env create -n repost python=3.11 -y
conda activate repost

pip install -r requirements.txt
```

For model training:
```
conda env create -n llama python=3.11 -y
conda activate llama

cd LLaMA-Factory
pip install -e .
```

Finally, specify code and data paths:
```
source setup_train.sh
```

### Load the RepoST-Train Docker Image for Execution
We provide the [RepoST-Train docker image](https://drive.google.com/file/d/1ya5ZMyW3irOaSnNB4qja9N9uOCzytrQB/view?usp=sharing).

You can use the image by 
```
docker load -i repost_train_image.tar
```

and create a docker container that links the data folders (specified in setup_train.sh)
```
docker run --name=exectrain -v ${final_dataset_DIR}:${docker_final_dataset_DIR} -it repost_train:v0
```

### Generate Candidate Solutions on RepoST-Train
We provide the code for generating candidate solutions under `generation/`.
We use [vllm](https://github.com/vllm-project/vllm) to run code generation for open-source models and use [litellm](https://www.litellm.ai/) for API-based models.

Here are the example scripts, where we generate 5 candidate solutions for each training example using GPT-4o.
```
cd generation
bash ExecTrain_gpt4o_generation.sh
```

The results will be saved under `data/ExecTrain/results/`, for example, `data/ExecTrain/rejection_sampling/ExecTrain_gpt4o_top5.json`. 
Our code will automatically copy the generated function to the evaluation script (with test cases) to check the correctness of the generation results (e.g., `generation_script`)


### Executing the Candidate Solutions in the Docker 
After the previous step, you should see the generation file under ${docker_final_dataset_DIR} (specified in setup_train.sh) inside the docker container.

Assume your generation file is `data/ExecTrain/rejection_sampling/ExecTrain_gpt4o_top5.json` and the dict key that stores the scripts is `generation_script`, you can run execution with:
```
# in the docker container
cd RepoST
source setup_train.sh
python RepoST/execution.py --input_file ../ExecTrain/rejection_sampling/ExecTrain_gpt4o_top5.json --script_key generation_script
```

The execution results for each evaluation example will be stored in `data/ExecTrain/rejection_sampling/ExecTrain_gpt4o_top5_exec.json`.


### (Optional) Debugging Failed Candidate Solutions to Obtain More Training Instances
To obtain more training instances, you can also call the model to debug failed solutions.

Here is an example script, where we prompt GPT-4o to generate one debugged solution for each failed candidate solution.
```
cd generation
bash scripts/ExecTrain_gpt4o_debug.sh
```

### Process Data
Finally, process the training data under `get_training_data`.

We provide the example scripts here:
```
cd get_training_data
bash scripts/get_rej_sampling_data.sh
```

and 

```
cd get_training_data
bash scripts/get_rej_sampling_debug_data.sh
```

The processed training data files will be saved to `LLaMA-Factory/data/`.

<a id="model-training-rs"></a>
### Run Model Training
Before training, you first need to add the data information in `LLaMA-Factory/data/dataset_info.json`. For example, adding an entry:
```
  "ExecTrain_gpt_debug_replay": {
    "file_name": "ExecTrain_rejection_sampling_gpt_debug_replay_data.json"
  },
```

After that, write a config file with the data information. 
For example, create `LLaMA-Factory/examples/train_full/qwencoder_ExecTrain_exec_rej_sampling_gpt_debug_replay.yaml`.

Finally, run training by calling `llamafactory-cli train` with your config. For example:
```
cd LLaMA-Factory
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/qwencoder_ExecTrain_exec_rej_sampling_gpt_debug_replay.yaml
```




<br><br>
<a id="eval-public"></a>
## Evaluation on Public Datasets
We provide the code for evaluating on public benchmarks under `CodeRAGBench_evaluation`, where we borrow and modify the code of [CodeRAGBench](https://github.com/code-rag-bench/code-rag-bench).

Please refer to the instructions in [CodeRAGBench_evaluation/README.md](CodeRAGBench_evaluation/README.md).




<br><br>
<a id="repost"></a>
## Build Your Own Executable Code Generation Environments
We provide the code for scraping GitHub repos, functions, and context under `get_github_input` and the code for constructing the evaluation scripts under `RepoST`.

You will need to build a docker to execute the evaluation scripts.

### Setup and Installation
Create a new environment:
```
conda env create -n repost python=3.11 -y
conda activate repost
```

Set up the environment for code generation and RepoST environment construction (Note that you probably need a separate environment for model training):
```
pip install -r requirements.txt
```

Finally, specify the code and data paths in your own setup file. We provide `get_github_input/r2e/ExecEval_config.yml` as an example.

Load the setup file with:
```
source setup_eval.sh
```

### Docker Setup
Start with creating an empty docker image based on `Dockerfile`:
```
docker build --tag exec_eval .
```

And run a docker container:
```
docker run --name=exec_eval_test -v ${dataset_generation_DIR}:${docker_dataset_generation_DIR} -it exec_eval
```

Inside the docker container, you may want to create a restricted user that can only access the data file and the tmp file:
```
adduser --disabled-password --gecos "" restricted_user
chown restricted_user:restricted_user /home/user/ExecEval
chown restricted_user:restricted_user /home/user/tmp
```


### Step 1: Sample GitHub Repos, Functions, and Context
You first need to write a config file under `get_github_input/`, where we provide two examples: `get_github_input/r2e/ExecTrain_config.yml` and `get_github_input/r2e/ExecEval_config.yml`.

After that, specify the path to the config file in `get_github_input/setup.sh`.

To scrape GitHub repos, you can follow the example scripts file that is used for creating RepoST-Eval: `get_github_input/scripts/ExecEval_scrape_repos.sh`.
In this script, we scrape 1,000 repos created between 2024-09-01 and 2024-12-31, and randomly sample 300. We clone the selected repos and remove the ones with storage sizes larger than 10M.
```
cd get_github_input
bash scripts/ExecEval_scrape_repos.sh
```

To sample functions and extract ground truth context by call graph, you can follow the example scripts file that is used for creating RepoST-Eval: `get_github_input/scripts/ExecEval_obtain_functions.sh`.
In this script, we extract all the functions with docstrings, sample at most 15 standalone functions and 15 class methods for each repo, and extract context with a maximum block size of 8,000 tokens.
```
cd get_github_input
bash scripts/ExecEval_obtain_functions.sh
```

After running this script, the file with initial repo, function, and context will be saved under `data/`. For example, `data/ExecEval/cleaned_python_test.json`.


### Step 2: Sandboxing
The second step is to call an LLM to aggregate the target function and its context (both infile and cross-file ones) to a separate script.
```
cd RepoST
python sandboxing.py --input_file cleaned_python_test.json --model_name "claude-3-5-sonnet-20240620"
```

If you use our default setup files, the results will be saved to `data/ExecEval/cleaned_python_test_sandboxed.json`.


### Step 3: Test Generation
The third step is to call an LLM to create tests for the target function in the sandboxed script. 
```
cd RepoST
python test_generation.py --input_file cleaned_python_test_sandboxed.json --model_name "claude-3-5-sonnet-20240620"
```

If you use our default setup files, the results will be saved to `data/ExecEval/cleaned_python_test_tests.json`.

### Step 4: Iterative Execution & Debugging
In principle, the ground truth target function should pass all the generated tests in the sandboxed script with no errors.
We hence execute the sandboxed script and use an LLM to debug the script (if needed):

Run execution inside the docker:
```
# inside the docker
python RepoST/RepoST/execution.py --input_file /home/user/ExecEval/cleaned_python_test_tests.json --script_key full_script --test_gt
python RepoST/RepoST/execution.py --input_file /home/user/ExecEval/cleaned_python_test_tests_exec.json --script_key full_script --test_gt --check_coverage_mode
```

Alternatively, you can run execution under the restricted_user:
```
# inside the docker
su -c "python RepoST/RepoST/execution.py --input_file /home/user/ExecEval/cleaned_python_test_tests.json --script_key full_script --test_gt" restricted_user
su -c "python RepoST/RepoST/execution.py --input_file /home/user/ExecEval/cleaned_python_test_tests_exec.json --script_key full_script --test_gt --check_coverage_mode" restricted_user
```
If you use our default setup files, the execution results will be saved to `data/ExecEval/cleaned_python_test_tests_exec_coverage.json`.

Debug with an LLM:
```
cd RepoST
python debug.py --input_file cleaned_python_test_tests_exec_coverage.json --model_name "claude-3-5-sonnet-20240620"
```
If you use our default setup files, the results will be saved to `data/ExecEval/cleaned_python_test_debug_round0.json`.

You can iteratively run execution and debugging to improve the success rate of the RepoST method.

We also provide the code to improve the test coverage, if the generated tests are correct, but miss some important branches:
```
cd RepoST
python improve_coverage.py --input_file cleaned_python_test_tests.json --model_name "claude-3-5-sonnet-20240620"
```

### Step 5: Post-Checking
Assuming the result file you finally obtain (with execution results) is called `data/ExecEval/cleaned_python_test_debug_round0_exec_coverage.json`, you can run post-processing with:
```
cd RepoST
python post_processing.py --input_file cleaned_python_test_debug_round0_exec_coverage.json --script_key full_script_debug_round0 --output_file test_set_final.json
```

Then you can call LLM-checkers to check (1) the functionality equivalence of the sandboxed target function and the original function, and (2) the correctness of the test cases:
```
cd RepoST
python llm_check.py --input_file test_set_final.json --model_name gpt-4o --mode sandbox
python llm_check.py --input_file test_set_final.json --model_name gpt-4o --mode test
python llm_check.py --input_file test_set_final.json --model_name gpt-4o --mode combine
```

If you use our default setup files, the file with examples that pass all tests will be saved as `data/ExecEval/checked_test_set_final.json`.


### Resulting Files

Please refer to [here](#data) to see the format of the resulting data file.


