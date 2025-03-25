from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from sys import argv
import json
from utils import hf_login

architectures = ["meta-llama/Llama-3.3-70B-Instruct", "deepseek-ai/DeepSeek-R1", "google/gemma-3-27b-it", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"] #distill version
data_address = "./data/task1.json"

def load_data(task:int):
    data_file = f"./data/task{task}.json"
    with open(data_file, "r") as fp:
        data = json.load(fp)
    return data

def run(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, data: dict):
    results = []
    for instance in data[0]:
        input_ids = tokenizer(instance["prompt"])
        outputs = model.generate(input_ids, do_sample=False, max_new_tokens=30)
        results.append({"word": instance["word"], "definition": instance["definition"], "sentence": instance["sentence"], "prompt": instance["prompt"], "output": tokenizer.batch_decode(outputs, skip_special_tokens=True)})
        

if __name__ == "__main__":
    if argv[1]:
        quant_config = QuantoConfig(weights="int4")
        model = AutoModelForCausalLM.from_pretrained(architectures[int(argv[1])], quantization_config = quant_config)
        tokenizer = AutoTokenizer.from_pretrained(architectures[int(argv[1])])
        data = load_data(int(argv[2]))

