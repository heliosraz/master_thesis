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


def checkpoint(model: nn.Module, task: int, results: List[dict]):
    result_path = os.path.join(
        script_dir, "..", "results", "judgement", str(model)+f"-taskt{task}-judgements.json")
    data = []
    if os.path.isfile(result_path):
        with open(result_path, "r+") as fp:
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
        try:
            responses = model.forward([instance['prompt']
                                    for instance in instances], use_tqdm=use_tqdm)
        except Exception as e:
                print(f"Error during model.forward: {e}")
                continue
        for response, instance in zip(responses, instances):
            instance.update({"task": task, "assistant": assistant,
                            "judge": str(model), "response": response})
            
            # min_length == 16
            results.append(instance)
        checkpoint(model, task, results)
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

def sort_tasks():
    categories = {i: [] for i in range(1, 5)}
    data_path = os.path.join(script_dir, "..", "data", "judgement")
    for root, dirs, files in os.walk(data_path):
        for file in files:
            task = int(file.split("-")[-1][4])
            categories[int(task)].append(file)
    # print(categories)
    return categories
        

def main(arches: List[int], tasks: List[int]=[-1], data_path: str = ""):
    data_files = sort_tasks()
    for arch in arches:
        model = architectures[arch]()
        for task in tasks:
            if task in data_files:
                files = data_files[task]
                for data_file in tqdm(files):
                    print(f"Running architecture {arch} on file {data_file}")
                    data = load_data(data_file)
                    if task == 1:
                        batch_size = 64
                    else:
                        batch_size = 32
                    print("Starting inference...")
                    run(model, data, data_file, batch_size=batch_size)
            elif os.path.isfile(data_path):
                print(f"Running architecture {arch} on file {data_path}")
                data = load_data(data_path)
                if task == 1:
                    batch_size = 64
                else:
                    batch_size = 32
                print("Starting inference...")
                run(model, data, data_path, batch_size=batch_size)


if __name__ == "__main__":
    arches = [0, 1, 2, 3]
    tasks = [-1]
    file = ""
    if len(argv) == 3:
        arches = [int(argv[1])]
        if len(argv[2])>1:
            file = argv[2]
        else:
            tasks = [int(argv[2])]
    elif len(argv) == 2:
        arches = [int(argv[1])]
    
    # print(arches)
    # print(tasks)
    main(arches, tasks, file)
