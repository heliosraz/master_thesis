from typing import List
from sys import argv, path
path.append("~/master_thesis/")
import json
import os
from tqdm import tqdm
from models import Llama, Mistral, Gemma
from torch import nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

architectures = [Llama, Gemma, Mistral] #distill version
data_address = ""


def load_data(task:int):
    data_file = f"~/master_thesis/data/tasks/task{task}.json"
    data = []
    with open(data_file, "r") as fp:
        data = json.load(fp)
    return data

def checkpoint(arch: int, i: int, results: List[dict]):
    with open(f"../results/{str(architectures[arch])}-task{i}.json", "w") as fp:
        json.dump(results, fp, indent=4)

def run(model:nn.Module, data: List[dict], task: int):
    results = []
    iteration = 0
    for instance in tqdm(data):
        responses = model.forward(instance)
        if "gold" in instance:
            results.append({"word": instance["word"], "definition": instance["definition"], "gold": instance["gold"], "sentence": instance["sentence"], "prompt": instance["prompt"], "output": responses})
        else:
            results.append({"word": instance["word"], "definition": instance["definition"], "sentence": instance["sentence"], "prompt": instance["prompt"], "output": responses})
        if iteration % 10 == 9:
            checkpoint(arch, task, results)
        iteration += 1
    return results

if __name__ == "__main__":
    print("#### Test ####")
    arches = [int(argv[1])]
    if len(argv)>2:
        tasks = [int(argv[2])]
    else:
        tasks = [1, 2, 3, 4]
    for arch in arches:
        for task in tasks:
            print(f"Running architecture {arch} on task {task}")
            model = architectures[arch]()
            data = load_data(task)
            results = run(model, data, task)
            checkpoint(arch, task, results)
        

