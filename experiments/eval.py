from torch import nn
import os
from sys import argv, path
script_dir = os.path.dirname(os.path.abspath(__file__))
path.append(os.path.join(script_dir, ".."))
from models import Llama, Mistral, Gemma, DeepSeek
from tqdm import tqdm
from typing import List
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


def checkpoint(model: nn.Module, data_file: str, results: List[dict]):
    result_path = os.path.join(
        script_dir, "..", "results", "judgement", data_file)
    with open(result_path, "w") as fp:
        json.dump(results, fp, indent=4)


def run(model: nn.Module, data: List[dict], data_file: str, batch_size: int = 128):
    results = []
    iteration = 0
    use_tqdm = False
    assistant = "-".join(data_file.split("-")[:-1])
    task = int(data_file.split("-")[-1][4])
    for start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        end = start + batch_size
        instances = data[start:end]
        responses = model.forward([instance['prompt']
                                  for instance in instances], use_tqdm=use_tqdm)
        for response, instance in zip(responses, instances):
            instance.update({"task": task, "assistant": assistant,
                            "judge": model, "response": response, "score": re.search("\[\[\d+\]\]",response).group()})
            results.append(instance)
        if iteration % 10 == 9:
            checkpoint(model, data_file, results)
        iteration += 1
    return results


def main(arches: List[int]):
    data_path = os.path.join(script_dir, "..", "data", "judgement")
    model = architectures[arch]()
    for arch in arches:
        for root, dirs, files in os.walk(data_path):
            for data_file in tqdm(files):
                print(f"Running architecture {arch} on file {data_file}")
                data = load_data(data_file)
                batch_size = 64
                print("Starting inference...")
                results = run(model, data, data_file, batch_size=batch_size)
                checkpoint(model, data_file, results)


if __name__ == "__main__":
    if len(argv) == 1:
        arches = [0, 1, 2]
    else:
        arches = [int(argv[1])]
    main(arches)
