from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
from sys import argv, path
import json
import os
from tqdm import tqdm
from models import Llama, Mistral, Gemma
from torch import nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

architectures = [Llama, Gemma, Mistral] #distill version
data_address = "./data/tasks/task1.json"
path.append("..")

def load_data(task:int):
    data_file = f"./data/tasks/task{task}.json"
    data = []
    with open(data_file, "r") as fp:
        data = json.load(fp)
    return data

def checkpoint(arch: int, i: int, results: List[dict]):
    with open(f"./results/{str(architectures[arch])}-task{i}.json", "w") as fp:
        json.dump(results, fp, indent=4)

def run(model:nn.Module, data: List[dict]):
    results = []
    iteration = 0
    for instance in tqdm(data):
        responses = model.forward(instance["prompt"])
        if "gold" in instance:
            results.append({"word": instance["word"], "definition": instance["definition"], "gold": instance["gold"], "sentence": instance["sentence"], "prompt": instance["prompt"], "output": responses})
        else:
            results.append({"word": instance["word"], "definition": instance["definition"], "sentence": instance["sentence"], "prompt": instance["prompt"], "output": responses})
        if iteration % 10 == 9:
            checkpoint(arch, i, results)
        iteration += 1
    return results

if __name__ == "__main__":
    arches = [int(argv[1])]
    if len(argv)>2:
        tasks = [int(argv[2])]
    else:
        tasks = [1, 2, 3, 4]
    for arch in arches:
        for task in tasks:
            model = architectures[arch]()
            data = load_data(int(argv[2]))
            results = run(model, data)
            checkpoint(arch, task, results)
        

