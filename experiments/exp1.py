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

architectures = [Llama, Gemma, Mistral, DeepSeek]  # distill version
os.makedirs(os.path.join(script_dir, "..", "results", "task"), exist_ok=True)


def load_data(task: int):
    data_path = os.path.join(script_dir, "..", "data", "tasks", f"task{task}.json")
    data = []
    with open(data_path, "r") as fp:
        data = json.load(fp)
    return data


def checkpoint(model: nn.Module, i: int, results: List[dict]):
    data_path = os.path.join(
        script_dir, "..", "results", "task", f"{str(model)}-task{i}.json"
    )
    with open(data_path, "w") as fp:
        json.dump(results, fp, indent=4)


def run(model: nn.Module, data: List[dict], task: int, batch_size: int = 128):
    results = []
    iteration = 0
    # use_tqdm = batch_size > 1
    use_tqdm = False
    for start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        end = start + batch_size
        instances = data[start:end]
        responses = model.forward(
            [instance["prompt"] for instance in instances], use_tqdm=use_tqdm
        )
        for response, instance in zip(responses, instances):
            if "gold" in instance:
                results.append(
                    {
                        "task": task,
                        "word": instance["word"],
                        "definition": instance["definition"],
                        "gold": instance["gold"],
                        "sentence": instance["sentence"],
                        "prompt": instance["prompt"],
                        "output": response,
                    }
                )
            else:
                results.append(
                    {
                        "task": task,
                        "word": instance["word"],
                        "definition": instance["definition"],
                        "gold": "",
                        "sentence": instance["sentence"],
                        "prompt": instance["prompt"],
                        "output": response,
                    }
                )
        if iteration % 10 == 9:
            checkpoint(model, task, results)
        iteration += 1
    return results


if __name__ == "__main__":
    if len(argv) == 1:
        arches = [0, 1, 2, 3]
    else:
        arches = [int(argv[1])]
    if len(argv) > 2:
        tasks = [int(argv[2])]
    else:
        tasks = [1, 2, 3, 4]
    for arch in arches:
        for task in tasks:
            print(f"Running architecture {arch} on task {task}")
            model = architectures[arch]()
            data = load_data(task)
            batch_size = 32
            print("Starting inference...")
            results = run(model, data, task, batch_size=batch_size)
            checkpoint(model, task, results)
