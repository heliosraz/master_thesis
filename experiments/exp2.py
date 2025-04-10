from typing import List
from sys import argv, path
path.append("~/master_thesis/")
import json
import os
from tqdm import tqdm
from models import Llama, Mistral, Gemma
from torch import nn


if __name__ == "__main__":
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