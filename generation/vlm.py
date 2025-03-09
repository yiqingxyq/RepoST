import os 
import pickle
import time

import torch
from vllm import LLM, SamplingParams

class VLM(object):
    def __init__(self, model_dir, sampling_params, max_length=8192):
        self.model_dir = model_dir
        self.model = None
        self.max_length = max_length
        self.sampling_params = sampling_params
        # SamplingParams(n=self.max_seq_num, temperature=0.2, top_p=0.95, ignore_eos=True, logprobs=1, max_tokens=512)

    def load_model(self):
        print("Loading vllm model..")
        n_gpus = torch.cuda.device_count()
        print("Num GPUs:", n_gpus)
        if n_gpus > 1:
            self.model = LLM(model=self.model_dir, max_model_len=self.max_length, dtype=torch.bfloat16, tensor_parallel_size=n_gpus)
        else:
            self.model = LLM(model=self.model_dir, max_model_len=self.max_length, dtype=torch.bfloat16)
        self.tokenizer = self.model.get_tokenizer()


    def generate_all(self, prompts, generation_prefix_list=None, check_prompt_len=True):
        if len(prompts) == 0:
            return []
        
        if self.model is None:
            self.load_model()

        # check prompt len 
        if check_prompt_len:
            start = time.time()
            print(f"Preprocessing infile prompts ..")
            tokenizer = self.model.get_tokenizer()
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.truncation_side = "left"
            tokenized_prompts = tokenizer(prompts, truncation=True, padding=True, max_length=self.max_length - self.sampling_params.max_tokens)
            prompts = tokenizer.batch_decode(tokenized_prompts.input_ids, skip_special_tokens=True)
            end = time.time()
            print(f"finished preprocessing with {end-start}s!")

        if generation_prefix_list is not None:
            assert type(generation_prefix_list) == list 
            assert len(generation_prefix_list) == len(prompts)
            prompts = [prompt+prefix for prompt,prefix in zip(prompts, generation_prefix_list)]

        vllm_out = self.model.generate(prompts, self.sampling_params)
        model_outputs = [[output.text for output in x.outputs] for x in vllm_out]

        return model_outputs
