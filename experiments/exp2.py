from typing import List
from sys import argv, path
import json
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
path.append(os.path.join(script_dir, ".."))
from tqdm import tqdm
from models import Llama, Mistral, Gemma, DeepSeek
from torch import nn




os.environ["TOKENIZERS_PARALLELISM"] = "true"

architectures = [Llama, Gemma, Mistral, DeepSeek] #distill version


def load_data(filename: str):
    data_path = os.path.join(script_dir, "..", "results", filename)
    data = []
    with open(data_path, "r") as fp:
        data = json.load(fp)
    return data

def cluster():
    pass

def run(model:nn.Module, data: List[dict], task: int):
    results = []
    for instance in tqdm(data):
        print(model.encode(instance['prompt']), model.tokenizer(instance["prompt"]))
    return results

if __name__ == "__main__":
    for root, dirs, files in os.walk(os.path.join(script_dir, "..", "results")):
        arch = ""
        data = []
        for f in files:
            if arch and arch == f.split("-")[0]:
                print(f"Running architecture {arch}")
                model = architectures[arch](device="mps")
                data = load_data(f)
                results = run(model, data, task)
        

