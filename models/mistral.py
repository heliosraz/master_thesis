from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, QuantoConfig
from torch import nn
import torch

class Mistral(nn.Module):
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", device: str = "cuda"):
        super(Mistral, self).__init__()
        quant_config = QuantoConfig(weights="int4")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = quant_config, device_map=device, trust_remote_code=True)
        self.device = self.model.device
        self.model_id = model_id
        
    def forward(self, prompts: List[str]):
        messages = []
        prompt = ""
        with torch.no_grad():
            for p in prompts["prompt"]:
                messages.append({"role": "user", "content": p})
                prompt += p + "\n"
                input_prompt = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(
                    input_prompt.input_ids.to(self.device), 
                    attention_mask=input_prompt.attention_mask.to(self.device),
                    do_sample=True, 
                    top_k=50, 
                    top_p=0.95,
                    max_new_tokens = 100, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    )
                response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]
                prompt += "Answer: "+response +"\n"
                messages.append({"role": "assistant", "content": response})
        return messages
    
    def __str__(self):
        return self.model_id.split("/")[-1]
# from typing import List
# from torch import nn
# from vllm import LLM
# from vllm import LLM
# from vllm.sampling_params import SamplingParams

# import os
# os.environ["VLLM_USE_V1"] = "0"

# import requests
# import json
# from huggingface_hub import hf_hub_download
# from datetime import datetime, timedelta

# url = "http://localhost:8000/v1/chat/completions"
# headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}
# model = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


# def load_system_prompt(repo_id: str, filename: str) -> str:
#     file_path = hf_hub_download(repo_id=repo_id, filename=filename)
#     with open(file_path, "r") as file:
#         system_prompt = file.read()
#     today = datetime.today().strftime("%Y-%m-%d")
#     yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
#     model_name = repo_id.split("/")[-1]
#     return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


# SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")


# class Mistral(nn.Module):
#     def __init__(self):
#         super(Mistral, self).__init__()
#         self.model = LLM(model = "mistralai/Mistral-Small-3.1-24B-Instruct-2503", tokenizer_mode="mistral")
        
#     def forward(self, prompts: List[str]):
#         messages = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#         ]
#         for prompt in prompts["prompt"]:
#             messages.append({"role": "user", "content": prompt})
#             data = {"model": model, "messages": messages, "temperature": 0}
#             response = requests.post(url, headers=headers, data=json.dumps(data))
#             messages.append({"role": "assistant", "content": response.json()["choices"][0]["message"]["content"]})
#         return messages
    
#     def __str__(self):
#         return self.__name__
    
# if __name__ == "__main__":
#     model = Mistral()
#     output = model.forward(["What is the capital of France?"])
#     print(output)