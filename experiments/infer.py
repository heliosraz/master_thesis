from tqdm.asyncio import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field
from utils.data_processing import load_system, load_data, prompt_template

import sys
import os
import json
from typing import List

model_ids = {"0":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
             "1":"google/gemma-3-4b-it",
             "2":"meta-llama/Llama-3.2-3B-Instruct",
             "3":"mistralai/Mistral-7B-Instruct-v0.3"}


script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.makedirs(os.path.join(script_dir, "..", "results", "task"), exist_ok=True)

import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:800{sys.argv[1]}/v1"
)
model = model_ids[sys.argv[1]]
task = sys.argv[2]
status = 0

def load_data(task: int):
    data_path = os.path.join(script_dir, "..", "data", "tasks", f"task{task}.json")
    data = []
    with open(data_path, "r") as fp:
        data = json.load(fp)
    return data
        

async def run_async(instance, system_prompt):
    prompt = instance["prompt"]
    messages = [
                {"role": "system", "content": system_prompt},
            ]
    for message in prompt:
        messages.append({"role": "user", "content": message})
        response = await client.chat.completions.create(
            model=model,
            messages=messages
        )
        messages.append({"role": "assistant", "content": response})
    instance["output"] = response
    if "gold" not in instance:
        instance["gold"] = ""
    return instance

async def process_batch(batch, system_prompt):
    tasks = [run_async(inst, system_prompt) for inst in batch]
    batch_results = []
    for f in tqdm.as_completed(tasks, 
                            total=len(tasks), 
                            desc="Processing tasks"):
        result = await f
        batch_results.append(result)
    
    return batch_results

def main(instances: List[dict], task: int, batch_size: int = 128):
    results = []
    system_prompt = load_system(task)
    for i in range(0, len(instances), batch_size=32):
        batch = instances[i:i+batch_size]
        results.extend(asyncio.run(process_batch(batch, system_prompt)))
    return results

if __name__ == "__main__":
    if len(sys.argv)>1:
        data = load_data(sys.argv[2])
        prompt = load_system("data/prompts/system_prompt.jsonl", sys.argv[2])
    else:
        data = load_data(1)
        prompt = load_system("data/prompts/system_prompt.jsonl", "1")
    instances = list(data.items())
    
    results = main(instances, sys.argv[1])
    with open("results/predictions.json", "a") as fp:
        for r in tqdm(results):
            json.dump(r, fp)
            fp.write("\n")
