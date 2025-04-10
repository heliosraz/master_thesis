from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, QuantoConfig
from torch import nn
import torch

class Llama(nn.Module):
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-3B-Instruct", device: str = "cuda"):
        super(Llama, self).__init__()
        quant_config = QuantoConfig(weights="int4")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = quant_config, device_map=device, trust_remote_code=True)
        self.device = device
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
        print(messages)
        return messages
    
    def __str__(self):
        return self.model_id.split("/")[-1]