from typing import List
from sys import argv, path
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
path.append(os.path.join(script_dir, ".."))
from tqdm import tqdm
from models import Llama, Mistral, Gemma, DeepSeek
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re


os.environ["TOKENIZERS_PARALLELISM"] = "true"
architectures = [Llama, Gemma, Mistral, DeepSeek]
model_ids = {
    0: "meta-llama/Llama-3.2-3B-Instruct",
    1: "google/gemma-3-4b-it",
    2: "mistralai/Mistral-7B-Instruct-v0.3",
    3: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}
os.makedirs(os.path.join(script_dir, "..", "results", "embed"), exist_ok=True)
device = "cuda"


def load_data(root, filename: str):
    file_path = os.path.join(root, filename)
    data = []
    with open(file_path, "r") as fp:
        data = json.load(fp)
    return data


def checkpoint(model: str, data: List[dict], task: str):
    data_path = os.path.join(
        script_dir, "..", "results", "embed", f"{model}-{task}-embeds.json"
    )
    with open(data_path, "w") as fp:
        json.dump(data, fp, indent=4)


def run(
    model: nn.Module,
    data: List[dict],
    tokenizer=None,
    batch_size: int = 32,
    tasks: List[str] = ["token", "definition", "response", "prompt"],
    vias: List[str] = ["prompt", "sentence"],
    device="auto",
):
    for start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        end = start + batch_size
        instances = data[start:end]
        for task in tasks:
            for i, via in enumerate(vias):
                if task == "token":
                    # print(f"Starting {via} token embedding...")
                    if via == "prompt":
                        batch = [instance[via][0] for instance in instances]
                    else:
                        batch = [instance[via] for instance in instances]
                elif task == "definition":
                    if i == 1:
                        continue
                    # print("Starting definition embedding...")
                    batch = [instance["definition"] for instance in instances]
                elif task == "response":
                    if i == 1:
                        continue
                    # print("Starting response embedding...")
                    batch = [instance["output"][1]["content"] for instance in instances]
                elif task == "prompt":
                    if i == 1:
                        continue
                    # print("Starting prompt embedding...")
                    batch = [instance["prompt"][0] for instance in instances]

                encodings = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    return_offsets_mapping=True,
                )

                inputs = {
                    k: v.to(device)
                    for k, v in encodings.items()
                    if k != "offset_mapping"
                }

                offsets = []
                for encoding in encodings["offset_mapping"]:
                    offsets.append(encoding)

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                batch_contextual_embed = outputs.hidden_states[-1]

                for instance, context, embeddings, offset in zip(
                    instances, batch, batch_contextual_embed, offsets
                ):
                    target = instance["word"]
                    if task == "token":
                        added = False
                        for embedding, (start, end) in zip(embeddings, offset):
                            token = context[start:end]
                            if token in target or target in token:
                                instance.update(
                                    {f"{task}_{via}_embedding": embedding.tolist()}
                                )
                                added = True
                                break
                        if not added:
                            print([context[start:end] for start, end in offset])
                            raise Exception("Didn't get embedding")
                    else:
                        instance.update(
                            {f"{task}_embedding": embeddings.mean(dim=0).tolist()}
                        )


if __name__ == "__main__":
    # torch.cuda.empty_cache()
    # arches = [3]
    # mode = "results"
    if len(argv) == 1:
        arches = [0, 1, 2, 3]
        mode = "general"
    else:
        print(argv)
        arches = [int(argv[1])]
        mode = argv[2]
    if mode == "general":
        for root, dirs, files in os.walk(
            os.path.join(script_dir, "..", "data", "tasks")
        ):
            data = []
            for arch in arches:
                model = AutoModelForCausalLM.from_pretrained(model_ids[arch]).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_ids[arch])
                tokenizer.pad_token = tokenizer.eos_token
                for fn in tqdm(files):
                    print(f"Running architecture {arch} on {fn}...")
                    data = load_data(root, fn)
                    batch_size = 16
                    run(
                        model,
                        tokenizer=tokenizer,
                        data=data,
                        batch_size=batch_size,
                        device=device,
                        tasks=["token", "definition", "prompt"],
                    )
                    checkpoint(
                        model_ids[arch].split("/")[-1], data, task=fn.split(".")[0]
                    )
    elif mode == "results":
        for root, dirs, files in os.walk(
            os.path.join(script_dir, "..", "results", "task")
        ):
            data = []
            for arch in arches:
                model = AutoModelForCausalLM.from_pretrained(model_ids[arch]).to(device)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_ids[arch], model_max_length=300
                )
                tokenizer.pad_token = tokenizer.eos_token

                for fn in tqdm(files):
                    if "-".join(fn.split("-")[:-1]) == model_ids[arch].split("/")[-1]:
                        print(
                            f"Embedding results from {fn} with architecture {arch} ..."
                        )
                        data = load_data(root, fn)
                        batch_size = 16
                        run(
                            model,
                            tokenizer=tokenizer,
                            data=data,
                            batch_size=batch_size,
                            device=device,
                            tasks=["response"],
                            vias=["none"],
                        )
                        checkpoint(
                            model_ids[arch].split("/")[-1],
                            data,
                            task=f"response_{fn.split('-')[-1].split('.')[0]}",
                        )
