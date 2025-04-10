from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from torch import nn
import torch

class Llama(nn.Module):
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B-Instruct", device: str = "cuda"):
        super(Llama, self).__init__()
        quant_config = QuantoConfig(weights="int4")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = quant_config, device_map=device, trust_remote_code=True)
        self.device = device
        self.model = torch.compile(self.model)
        
    def forward(self, prompts: List[str]):
        messages = []
        prompt = ""
        for p in prompts["prompt"]:
            messages.append({"role": "user", "content": p})
            prompt += p + "\n"
            input_prompt = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(input_prompt.input_ids.to(self.device), max_new_tokens = 100, attention_mask=input_prompt.attention_mask.to(self.device), pad_token_id = self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(outputs, skip_special_tokens=True)[:len(prompt)]
            print(response)
            prompt += "Answer: "+response +"\n"
            messages.append({"role": "assistant", "content": response})
        return messages
    
    def __str__(self):
        return "Llama-3.2-3B-Instruct"