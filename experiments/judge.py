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
import eval

os.makedirs(os.path.join(script_dir, "..", "data", "judgement"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "..", "results", "judgement"), exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

architectures = [Llama, Gemma, Mistral, DeepSeek]  # distill version


def load_data(file_name: str):
    data_path = os.path.join(script_dir, "..", "data", "judgement", file_name)
    data = []
    with open(data_path, "r") as fp:
        data = json.load(fp)
    return data


def checkpoint(model: nn.Module, results: List[dict], task: int, assist: str):
    result_path = os.path.join(
        script_dir,
        "..",
        "results",
        "judgement",
        f"{str(model)}-task{task}-judgements-{assist.split('.')[0]}.json",
    )
    data = []
    if os.path.isfile(result_path):
        with open(result_path, "r+") as fp:
            data = json.load(fp)
    with open(result_path, "w") as fp:
        json.dump(data + results, fp, indent=4)


def run(model: nn.Module, data: List[dict], file_name: str, batch_size: int = 128):
    results = []
    use_tqdm = False
    assistant = "-".join(file_name.split("-")[:-1])
    task = int(file_name.split("-")[-1][4])
    progress_bar = tqdm(total=len(data), desc="Processing Instances:")
    while data:
        instances = data[:batch_size]
        data = data[batch_size:]
        try:
            responses = model.forward(
                [instance["prompt"] for instance in instances], use_tqdm=use_tqdm
            )
        except Exception as e:
            print(f"Error during model.forward: {e}")
            continue
        for response, instance in zip(responses, instances):
            if "repeat" not in instance:
                instance["repeat"] = 0
            rating = eval.find_score(response[-1]["content"])
            if rating or instance["repeat"] >= 5:
                instance.update(
                    {
                        "task": task,
                        "assistant": assistant,
                        "judge": str(model),
                        "response": response,
                    }
                )
                progress_bar.update(1)
                results.append(instance)
            else:
                instance["repeat"] += 1
                data.append(instance)
    checkpoint(model, results, task, file_name)
    return results


def sort_tasks():
    categories = {i: [] for i in range(1, 5)}
    data_path = os.path.join(script_dir, "..", "data", "judgement")
    for root, dirs, files in os.walk(data_path):
        for file in files:
            task = int(file.split("-")[-1][4])
            categories[int(task)].append(file)
    # print(categories)
    return categories


def main(arches: List[int], tasks: List[int] = [-1], file_name: str = ""):
    data_files = sort_tasks()
    file_path = os.path.join(script_dir, "..", "data", "judgement", file_name)
    for arch in arches:
        model = architectures[arch]()
        for task in tasks:
            if task in data_files:
                files = data_files[task]
                for file in tqdm(files):
                    print(f"Running architecture {arch} on file {file}")
                    data = load_data(file)
                    if task == 1:
                        batch_size = 64
                    else:
                        batch_size = 32
                    print("Starting inference...")
                    run(model, data, file, batch_size=batch_size)
            elif os.path.isfile(file_path):
                print(f"Running architecture {arch} on file {file_name}")
                data = load_data(file_name)
                if task == 1:
                    batch_size = 64
                else:
                    batch_size = 32
                print("Starting inference...")
                run(model, data, file_name, batch_size=batch_size)


if __name__ == "__main__":
    arches = [0, 1, 2, 3]
    tasks = [-1]
    file = ""
    if len(argv) == 3:
        arches = [int(argv[1])]
        if len(argv[2]) > 1:
            file = argv[2]
        else:
            tasks = [int(argv[2])]
    elif len(argv) == 2:
        arches = [int(argv[1])]

    print(arches)
    print(tasks)
    main(arches, tasks, file)
