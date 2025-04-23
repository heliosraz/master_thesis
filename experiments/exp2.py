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

def checkpoint(model:nn.Module, data: List[dict]):
    data_path = os.path.join(script_dir, "..", "results", "embed", f"{str(model)}-embeds.json")
    with open(data_path, "w") as fp:
        json.dump(data, fp, indent=4)

def run(model:nn.Module, tokenizer, data: List[dict], batch_size: int = 32, task = "token", device = "auto"):
    for start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        end = start + batch_size
        instances = data[start:end]
        if task == "token":
            encodings = tokenizer([instance["prompt"][0] for instance in instances], return_tensors="pt", padding=True, return_offsets_mapping=True)
            
            inputs = {k: v.to(device) for k, v in encodings.items() if k != "offset_mapping"}
            
            offsets = []
            for encoding in encodings["offset_mapping"]:
                offsets.append(encoding)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            
            batch_contextual_embed = outputs.hidden_states[-1]
            
            for instance, embeddings, offset in zip(instances, batch_contextual_embed, offsets):
                added = False
                for embedding, (start, end) in zip(embeddings, offset):
                        token = instance["prompt"][0][start:end]
                        if token in instance["word"] or instance["word"] in token:
                            instance.update({"token_embedding": embedding.tolist()})
                            added = True
                            break
                if not added:
                    print([instance["prompt"][0][start:end] for start, end in offset])
                    raise Exception("Didn't get embedding")
            
        elif task == "definition":
            embeddings = model.encode([instance[task] for instance in instances], use_tqdm = False)
            for instance, embedding in zip(instances, embeddings):
                instance.update({"definition_embedding": embedding.outputs.embedding})
        elif task == "response":
            embeddings = model.encode([instance["output"][1]["content"] for instance in instances], use_tqdm = False)
            for instance, embedding in zip(instances, embeddings):
                instance.update({"response_embedding": embedding.outputs.embedding})
        elif task == "prompt":
            embeddings = model.encode([instance["prompt"][0] for instance in instances], use_tqdm = False)
            for instance, embedding in zip(instances, embeddings):
                instance.update({"prompt_embedding": embedding.outputs.embedding})

if __name__ == "__main__":
    device = "cpu"
    if len(argv) == 1:
        arches = [0, 1, 2, 3]
    else:
        arches = [int(argv[1])]
    print(arches)
    token_embeddings = {arch: {} for arch in arches}
    
    for root, dirs, files in os.walk(os.path.join(script_dir, "..", "data", "embed")):
        for arch in arches:
            result_path = os.path.join(script_dir, "..", "results", "embed", f"{model_ids[arch].split('/')[-1]}-embeds.json")
            if os.path.isfile(result_path):
                with open(result_path, "r+") as fp:
                    data = json.load(fp)
                    token_embeddings[arch] = {instance["prompt"][0]: instance["token_embedding"] for instance in data}
            else:
                tok_model = AutoModelForCausalLM.from_pretrained(model_ids[arch]).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_ids[arch])
                tokenizer.pad_token = tokenizer.eos_token
                for fn in files:
                    print(f"Running architecture {arch}...")
                    data = load_data(root, fn)
                    batch_size = 32
                    print("Starting embedding...")
                    run(tok_model, tokenizer, data, batch_size=batch_size, device = device)
                    token_embeddings[arch] = {instance["prompt"][0]: instance["token_embedding"] for instance in data}
                    checkpoint(model_ids[arch].split("/")[-1], data)
    
    for root, dirs, files in os.walk(os.path.join(script_dir, "..", "results", "task")):
        data = []
        for arch in arches:
            embed_model = architectures[arch](device = "auto", task = "embed")
            tokenizer.pad_token = tokenizer.eos_token
            for fn in files:
                print(f"Running architecture {arch}...")
                data = load_data(root, fn)
                batch_size = 32
                print("Starting embedding...")
                # run(word_model, tokenizer, data, batch_size=batch_size)
                run(embed_model.llm, tokenizer, data, batch_size=batch_size, task = "definition")
                run(embed_model.llm, tokenizer, data, batch_size=batch_size, task = "prompt")
                run(embed_model.llm, tokenizer, data, batch_size=batch_size, task = "response")
                for instance in data:
                    instance["token_embedding"] = token_embeddings[arch][instance["sentence"]]
                checkpoint(embed_model, data)
        

