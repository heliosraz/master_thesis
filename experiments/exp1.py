from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from sys import argv, path
import json
from utils import hf_login
import os
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

architectures = ["meta-llama/Llama-3.2-1B-Instruct", "deepseek-ai/DeepSeek-R1", "google/gemma-3-27b-it", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"] #distill version
data_address = "./data/tasks/task1.json"
path.append("..")

def load_data(task:int):
    data_file = f"./data/tasks/task{task}.json"
    data = []
    with open(data_file, "r") as fp:
        data = json.load(fp)
    return data

def run(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, data: List[dict]):
    results = []
    iteration = 0
    # while data:
    #     batch_size = 1000
    #     instances = [data.pop()["prompt"] for _ in range(batch_size)]
    #     prompts = ["" for _ in range(batch_size)]
    #     for iteration in range(len(instances[0])):
    #         prompts = [prompts[i]+instance[iteration]+"\n" for i, instance in enumerate(instances)]
    #         input_prompt = tokenizer(prompts, return_tensors="pt")
    #         outputs = model.generate(input_prompt.input_ids, max_new_tokens = 100, attention_mask=input_prompt.attention_mask)
    #         response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].replace(prompt, "")
    #         prompt += "Answer: "+response +"\n"
    for instance in tqdm(data):
        prompt = ""
        responses = []
        for p in instance["prompt"]:
            prompt += p + "\n"
            input_prompt = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(input_prompt.input_ids.to(model.device), max_new_tokens = 100, attention_mask=input_prompt.attention_mask.to(model.device))
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].replace(prompt, "")
            prompt += "Answer: "+response +"\n"
            responses.append(response)
        if "gold" in instance:
            results.append({"word": instance["word"], "definition": instance["definition"], "gold": instance["gold"], "sentence": instance["sentence"], "prompt": instance["prompt"], "output": responses})
        else:
            results.append({"word": instance["word"], "definition": instance["definition"], "sentence": instance["sentence"], "prompt": instance["prompt"], "output": responses})
        if iteration % 10 == 9:
            with open(f"./results/{architectures[arch].split("/")[1]}-task{i}.json", "w") as fp:
                json.dump(results, fp, indent=4)
        iteration += 1
    return results

if __name__ == "__main__":
    if len(argv)>1:
        arch = int(argv[1])
        i = int(argv[2])
        quant_config = QuantoConfig(weights="int4")
        model = AutoModelForCausalLM.from_pretrained(architectures[arch], quantization_config = quant_config, device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(architectures[arch])
        data = load_data(int(argv[2]))
        results = run(model, tokenizer, data)
        with open(f"./results/{architectures[arch].split("/")[1]}-task{i}.json", "w") as fp:
            json.dump(results, fp, indent=4)
    else:
        for i in range(5):
            for arch in range(len(architectures)):
                quant_config = QuantoConfig(weights="int4")
                model = AutoModelForCausalLM.from_pretrained(architectures[arch], quantization_config = quant_config, device_map="cuda")
                tokenizer = AutoTokenizer.from_pretrained(architectures[arch])
                data = load_data(i)
                results = run(model, tokenizer, data)
                with open(f"./results/{architectures[arch].split("/")[1]}-task{i}.json", "w") as fp:
                    json.dump(results, fp, indent=4)
        

