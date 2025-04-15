from torch import nn
import os
from sys import argv, path
script_dir = os.path.dirname(os.path.abspath(__file__))
path.append(os.path.join(script_dir, ".."))
from models import Llama, Mistral, Gemma, DeepSeek
from tqdm import tqdm
from typing import List, Set
import json
import re

os.makedirs(os.path.join(script_dir, "..", "data", "judgement"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "..", "results", "judgement"), exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

architectures = [Llama, Gemma, Mistral, DeepSeek]  # distill version


def load_data(data_file: str):
    data_path = os.path.join(script_dir, "..", "data", "judgement", data_file)
    data = []
    with open(data_path, "r") as fp:
        data = json.load(fp)
    return data


def checkpoint(model: nn.Module, results: List[dict]):
    result_path = os.path.join(
        script_dir, "..", "results", "judgement", str(model)+"-judgements.json")
    data = []
    with open(result_path, "r") as fp:
        data = json.load(fp)
    with open(result_path, "w") as fp:
        json.dump(data+results, fp, indent=4)


def run(model: nn.Module, data: List[dict], data_file: str, batch_size: int = 128):
    results = []
    use_tqdm = False
    assistant = "-".join(data_file.split("-")[:-1])
    task = int(data_file.split("-")[-1][4])
    for start in tqdm(range(0, len(data), batch_size), desc="Processing batches:"):
        end = start + batch_size
        instances = data[start:end]
        responses = model.forward([instance['prompt']
                                  for instance in instances], use_tqdm=use_tqdm)
        for response, instance in zip(responses, instances):
            instance.update({"task": task, "assistant": assistant,
                            "judge": str(model), "response": response})
            
            # min_length == 16
            results.append(instance)
        checkpoint(model, results)
    return results

def evaluate():
    record = {task: {} for task in range(1, 5)}
    result_path = os.path.join(
        script_dir, "..", "results", "judgement")
    for root, dirs, files in os.walk(result_path):
        for fp in files:
            task = int(fp.split("-")[-1][4])
            assistant = "-".join(fp.split("-")[:-1])
            if assistant not in record[task]:
                record[task][assistant] = {}
            with open(fp, "r") as f:
                data = json.load(f)
                data = [(instance["word"], int(re.search("\d+",instance["score"]).group())) for instance in data]
                for word, score in data:
                    if word not in record[task][assistant]:
                        record[task][assistant][word] = (score, 1)
                    else:
                        record[task][assistant][word] = (record[task][assistant][word][0]+score, record[task][assistant][word][1]+1)
    return record


def main(arches: List[int], tasks: Set[int]):
    data_path = os.path.join(script_dir, "..", "data", "judgement")
    for arch in arches:
        model = architectures[arch]()
        for root, dirs, files in os.walk(data_path):
            for data_file in tqdm(files):
                task = int(data_file.split("-")[-1][4])
                if task in tasks:
                    print(f"Running architecture {arch} on file {data_file}")
                    data = load_data(data_file)
                    if task == 1:
                        batch_size = 512 #256
                    else:
                        batch_size = 256
                    print("Starting inference...")
                    run(model, data, data_file, batch_size=batch_size)


if __name__ == "__main__":
    if len(argv) == 3:
        arches = [int(argv[1])]
        tasks = set([int(argv[2])])
    elif len(argv) == 2:
        arches = [int(argv[1])]
        tasks = set([1,2,3,4])
    else:
        arches = [0, 1, 2]
        tasks = set([1,2,3,4])
        
    main(arches, tasks)
