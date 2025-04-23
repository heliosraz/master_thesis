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
model_ids = {0: "meta-llama/Llama-3.2-3B-Instruct", 1: "google/gemma-3-4b-it", 2: "mistralai/Mistral-7B-Instruct-v0.3", 3: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
os.makedirs(os.path.join(script_dir, "..", "results", "embed"), exist_ok=True)

def load_data(root, filename: str):
    file_path = os.path.join(root, filename)
    data = []
    with open(file_path, "r") as fp:
        data = json.load(fp)
    return data

def checkpoint(model:nn.Module, data: List[dict], task: str):
    data_path = os.path.join(script_dir, "..", "results", "embed", f"{str(model)}-{task}-embeds.json")
    with open(data_path, "w") as fp:
        json.dump(data, fp, indent=4)

def run(model:nn.Module, data: List[dict], tokenizer=None, batch_size: int = 32, task = "token", device = "auto", via: str = "sentence"):
    for start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        end = start + batch_size
        instances = data[start:end]
        if task == "token":
            batch = [instance[via] for instance in instances]
        elif task == "definition":
            batch = [instance["definition"] for instance in instances]
        elif task == "response":
            batch = [instance["output"][1]["content"] for instance in instances]
        elif task == "prompt":
            batch = [instance["prompt"][0] for instance in instances]
        
        encodings = tokenizer(batch, return_tensors="pt", padding=True, return_offsets_mapping=True)
        
        inputs = {k: v.to(device) for k, v in encodings.items() if k != "offset_mapping"}
        
        offsets = []
        for encoding in encodings["offset_mapping"]:
            offsets.append(encoding)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        
        batch_contextual_embed = outputs.hidden_states[-1]
        
        for instance, embeddings, offset in zip(instances, batch_contextual_embed, offsets):
            added = False
            if task == "token":
                for embedding, (start, end) in zip(embeddings, offset):
                        token = instance["prompt"][0][start:end]
                        if token in instance["word"] or instance["word"] in token:
                            instance.update({f"{task}_{via}_embedding": embedding.tolist()})
                            added = True
                            break
            else:
                instance.update({f"{task}_embedding": embeddings.mean(dim=1).tolist()})
                
            if not added:
                print([instance["prompt"][0][start:end] for start, end in offset])
                raise Exception("Didn't get embedding")

if __name__ == "__main__":
    device = "cpu"
    if len(argv) == 1:
        arches = [0, 1, 2, 3]
    else:
        arches = [int(argv[1])]
    print(arches)
    for root, dirs, files in os.walk(os.path.join(script_dir, "..", "data", "tasks")):
        data = []
        for arch in arches:
            model = AutoModelForCausalLM.from_pretrained(model_ids[arch]).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_ids[arch])
            for fn in files:
                print(f"Running architecture {arch}...")
                data = load_data(root, fn)
                batch_size = 32
                print("Starting definition token embedding...")
                run(model, tokenizer=tokenizer, data=data, batch_size=batch_size, device = device, task = "token", via = "definition")
                print("Starting sentence token embedding...")
                run(model, tokenizer=tokenizer, data=data, batch_size=batch_size, device = device, task = "token", via = "sentence")
                print("Starting definition embedding...")
                run(model, tokenizer=tokenizer, data=data, batch_size=batch_size, device = device, task = "definition")
                print("Starting prompt embedding...")
                run(model, tokenizer=tokenizer, data=data, batch_size=batch_size, device = device, task = "prompt")
                print("Starting response embedding...")
                run(model, tokenizer=tokenizer, data=data, batch_size=batch_size, device = device, task = "response")
                checkpoint(model, data, task = fn.split(".")[1])
        

