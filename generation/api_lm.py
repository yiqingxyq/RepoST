import os
import json
import time
import tiktoken
import pickle
from tqdm import tqdm
from math import ceil
from typing import List, Optional

import litellm
from litellm import completion
litellm.drop_params=True

class API_Sampling_Params(object):
    def __init__(self, n=1, temperature=0.2, top_p=0.95, ignore_eos=True, logprobs=1, max_tokens=1024):
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.ignore_eos = ignore_eos
        self.logprobs = logprobs
        self.max_tokens = max_tokens


class API_LM(object):
    def __init__(self, model_name, cache_name="cache", sampling_params=None, max_tokens=None):
        self.model_name = model_name
        self.sampling_params = sampling_params if sampling_params is not None else API_Sampling_Params()
        if max_tokens is not None: 
            self.sampling_params.max_tokens = max_tokens
        # SamplingParams(n=self.max_seq_num, temperature=0.2, top_p=0.95, ignore_eos=True, logprobs=1, max_tokens=512)
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.cache_file = f".cache/{model_name}_{cache_name}.pkl"
        os.makedirs(".cache", exist_ok=True)
        self.cache = None # only activated in `generate_all`
        
    def load_cache(self):
        if os.path.exists(self.cache_file):
            self.cache = pickle.load(open(self.cache_file, "rb"))
        else:
            self.cache = {}
            
    def save_cache(self):
        pickle.dump(self.cache, open(self.cache_file, "wb"))
        
    def get_token_num(self, prompt):
        try:
            return len(self.tokenizer.encode(prompt, disallowed_special=()))
        except:
            return 0
    
    def get_response(self, messages: str, **kwargs) -> list[str]:
        i_iters = 0
        n_iters, sleep = 2, 3
        response = ""
        while i_iters < n_iters:
            i_iters += 1
            try:
                response = completion(
                    model=self.model_name, 
                    messages=messages,
                    **kwargs
                )
                return [c.message.content for c in response.choices]
            except Exception as error:
                print(f"Generation error: {error}")
                time.sleep(i_iters * sleep)
        return [response]
    
    def generate(self, messages):
        if "o3" in self.model_name:
            gen_kwargs = {
                "n": self.sampling_params.n,
                "max_completion_tokens": self.sampling_params.max_tokens,
                "temperature": self.sampling_params.temperature,
                "top_p": self.sampling_params.top_p,
            }
        else:
            gen_kwargs = {
                "n": self.sampling_params.n,
                "max_tokens": self.sampling_params.max_tokens,
                "temperature": self.sampling_params.temperature,
                "top_p": self.sampling_params.top_p,
            }
        
        # return self.get_response(messages=messages, **gen_kwargs)
        
        if "claude" in self.model_name:
            res = []
            for i in range(self.sampling_params.n):
                resi = self.get_response(messages=messages, **gen_kwargs)
                res.extend(resi)
        else:
            res = self.get_response(messages=messages, **gen_kwargs) # list[str]
        
        return res
    
    
    def generate_all(self, all_messages):
        if self.cache is None:
            self.load_cache()
        
        if "o3" in self.model_name:
            gen_kwargs = {
                "n": self.sampling_params.n,
                "max_completion_tokens": self.sampling_params.max_tokens,
            }
        else:
            gen_kwargs = {
                "n": self.sampling_params.n,
                "max_tokens": self.sampling_params.max_tokens,
                "temperature": self.sampling_params.temperature,
                "top_p": self.sampling_params.top_p,
            }
        
        if "claude" in self.model_name:
            del gen_kwargs["n"]
        
        generation_count = 0
        generations = []
        for idx, messages in enumerate(tqdm(all_messages)):
            if str(messages) in self.cache and len(self.cache[str(messages)]) >= self.sampling_params.n:
                generations.append(self.cache[str(messages)][:self.sampling_params.n])
                continue
                
            if "claude" in self.model_name:
                res = []
                for i in range(self.sampling_params.n):
                    resi = self.get_response(messages=messages, **gen_kwargs)
                    res.extend(resi)
            else:
                res = self.get_response(messages=messages, **gen_kwargs) # list[str]
            generations.append(res)
            generation_count += 1
            
            if str(messages) not in self.cache or len(self.cache[str(messages)]) < len(res):
                self.cache[str(messages)] = res
                
            if generation_count % 50 == 0 and generation_count > 0:
                self.save_cache()
                print(f"Saved cache for {generation_count} examples..")
                
        if generation_count > 0:
            self.save_cache()
            print(f"Saved cache for {generation_count} examples..")
            
        return generations
    
    
if __name__ == "__main__":
    prompts = [
        "## pyt/__init__.py\nimport os\n\ndef get_counts():\n    \"\"\"return test counts that are set via pyt environment variables when pyt \n    runs the test\n\n    :returns: dict, 3 keys (classes, tests, modules) and how many tests of each\n        were found by pyt\n    \"\"\"\n    counts = {}\n    ks = [\n        ('PYT_TEST_CLASS_COUNT', \"classes\"),\n        ('PYT_TEST_COUNT', \"tests\"),\n        ('PYT_TEST_MODULE_COUNT', \"modules\"),\n    ]\n\n    for ek, cn in ks:\n        counts[cn] = int(os.environ.get(ek, 0))\n\n    return counts\n\ndef is_single_class():\n    \"\"\"Returns True if only a single class is being run or some tests within a single class\"\"\"\n    ret = False\n    counts = get_counts()\n    if counts[\"classes\"] < 1 and counts[\"modules\"] < 1:\n        ret = counts[\"tests\"] > 0\n    else:\n        ret = counts[\"classes\"] <= 1 and counts[\"modules\"] <= 1\n    return ret\n\ndef is_single_module():\n    \"\"\"\n    Returns True if only a module is being run\n    \"\"\"\n",
        "## qiskit/pulse/channels/pulse_channels.py\nclass LoRange:\n    \"\"\"Range of LO frequency.\"\"\"\n\n    def __init__(self, lower_bound: float, upper_bound: float):\n        self._lb = lower_bound\n        self._ub = upper_bound\n\n    def includes(self, lo_freq: float) -> bool:\n        \"\"\"\n        Whether `lo_freq` is within the `LoRange`.\n    \n        Args:\n        lo_freq: LO frequency to be checked\n    \n        Returns:\n        bool: True if lo_freq is included in this range, otherwise False\n        \"\"\"\n",
    ]
    
    sampling_params = API_Sampling_Params(n=1, temperature=0.2, top_p=0.95, ignore_eos=True, logprobs=1, max_tokens=1024)
    api_lm = API_LM("gpt-4o-2024-11-20", sampling_params)
    
    generations = api_lm.generate_all(prompts)
    
    print(generations)