from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, QuantoConfig
from torch import nn

class Llama(nn.Module):
    def __init__(self, model_id: str = "meta-llama/Llama-3.3-70B-Instruct", device: str = "cuda"):
        super(Llama, self).__init__()
        quant_config = QuantoConfig(weights="int4")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = quant_config, device_map=device, trust_remote_code=True)
        self.device = device
        
    def forward(self, prompts: List[str]):
        messages = []
        prompt = ""
        for p in prompts["prompt"]:
            messages.append({"role": "user", "content": p})
            prompt += p + "\n"
            input_prompt = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(input_prompt.input_ids.to(self.device), num_beams = 3, max_new_tokens = 100, attention_mask=input_prompt.attention_mask.to(self.device))
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][:len(prompt)]
            prompt += "Answer: "+response +"\n"
            messages.append({"role": "assistant", "content": response})
        return messages