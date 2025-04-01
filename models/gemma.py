from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, QuantoConfig
from torch import nn

class Gemma(nn.Module):
    def __init__(self, model_id: str = "google/gemma-3-27b-it", device: str = "cuda"):
        quant_config = QuantoConfig(weights="int4")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = quant_config, device_map=device, trust_remote_code=True)
        self.device = device
        
    def forward(self, prompt: str):
        input_prompt = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(input_prompt.input_ids.to(self.device), num_beams = 3, max_new_tokens = 100, attention_mask=input_prompt.attention_mask.to(self.device))
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][:len(prompt)]
        return response