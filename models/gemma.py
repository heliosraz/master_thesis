from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, QuantoConfig
from torch import nn

class Gemma(nn.Module):
    def __init__(self, model_id: str = "google/gemma-3-12b-it", device: str = "cuda"):
        super(Gemma, self).__init__()
        quant_config = QuantoConfig(weights="int4")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = quant_config, device_map=device, trust_remote_code=True)
        self.device = self.model.device
        self.model_id = model_id
        
    def forward(self, prompts: List[str]):
        messages = []
        prompt = ""
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
            response = self.tokenizer.decode(outputs, skip_special_tokens=True)[:len(prompt)]
            prompt += "Answer: "+response +"\n"
            messages.append({"role": "assistant", "content": response})
        return messages
    
    def __str__(self):
        return self.model_id.split("/")[-1]