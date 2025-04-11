# from typing import List
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, QuantoConfig
# from torch import nn
# import torch

# class DeepSeek(nn.Module):
#     def __init__(self, model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device: str = "cuda"):
#         super(DeepSeek, self).__init__()
#         quant_config = QuantoConfig(weights="int4")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.tokenizer.padding_side = "left"
#         self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = quant_config, device_map=device, trust_remote_code=True)
#         self.device = self.model.device
#         self.model_id = model_id
        
#     def tokenize(self, text: str):
#         return self.tokenizer.tokenize(text, padding=True, truncation=True)
    
#     def encode(self, text: str):
#         return self.tokenizer.encode(text, return_tensors="pt")
        
#     def forward(self, prompts: List[str]):
#         messages = []
#         prompt = ""
#         with torch.no_grad():
#             for p in prompts["prompt"]:
#                 messages.append({"role": "user", "content": p})
#                 prompt += p + "\n"
#                 input_prompt = self.tokenizer(prompt, return_tensors="pt")
#                 outputs = self.model.generate(
#                     input_prompt.input_ids.to(self.device), 
#                     attention_mask=input_prompt.attention_mask.to(self.device),
#                     do_sample=True, 
#                     top_k=50, 
#                     top_p=0.95,
#                     max_new_tokens = 100, 
#                     pad_token_id=self.tokenizer.eos_token_id,
#                     )
#                 response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]
#                 prompt += "Answer: "+response +"\n"
#                 messages.append({"role": "assistant", "content": response})
#         return messages
    
#     def __str__(self):
#         return self.model_id.split("/")[-1]

from typing import List
from torch import nn
import torch
from vllm import LLM, SamplingParams

class DeepSeek(nn.Module):
    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device: str = "cuda"):
        super(DeepSeek, self).__init__()
        self.params = SamplingParams(
            top_k=50,
            top_p=0.95,
            max_tokens=100
        )
        self.llm = LLM(model=model_id, device=device)
        self.model_id = model_id

    # def tokenize(self, text: str):
    #     return self.tokenizer.tokenize(text, padding=True, truncation=True)

    # def encode(self, text: str):
    #     return self.tokenizer.encode(text, return_tensors="pt")

    def forward(self, prompts: List[str], use_tqdm = False):
        messages = []
        prompt = ""
        with torch.no_grad():
            for p in prompts["prompt"]:
                messages.append({"role": "user", "content": p})
                prompt += p + "\n"
                output = self.llm.generate(
                    prompt,
                    self.params,
                    use_tqdm = use_tqdm
                )
                response = output[0].outputs[0].text
                prompt += "Answer: "+response + "\n"
                messages.append({"role": "assistant", "content": response})
        return messages
    
    def __str__(self):
        return self.model_id.split("/")[-1]