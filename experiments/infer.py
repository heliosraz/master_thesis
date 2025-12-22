import sys
import os
import json
from typing import List
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, ".."))

from tqdm.asyncio import tqdm
from openai import OpenAI, BadRequestError
from pydantic import BaseModel, Field
from utils import load_system_prompt, load_data



model_ids = {"0":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
             "1":"google/gemma-3-4b-it",
             "2":"meta-llama/Llama-3.2-3B-Instruct",
             "3":"mistralai/Mistral-7B-Instruct-v0.3"}

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.makedirs(os.path.join(script_dir, "..", "results", "task"), exist_ok=True)

import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:800{sys.argv[1]}/v1"
)
print(f"http://localhost:800{sys.argv[1]}/v1")
status = 0

def load_data(task: int):
    data_path = os.path.join(script_dir, "..", "data", "tasks", f"task{task}.json")
    data = []
    with open(data_path, "r") as fp:
        data = json.load(fp)
    return data

def checkpoint(results: List[dict]):
    mname = model_ids[model].split("/")[-1].split("-")[0]
    mname[0] = mname[0].lower()
    data_path = os.path.join(
        script_dir, "..", "results", "task", f"{mname}-task{task}.json"
    )
    with open(data_path, "a") as fp:
        json.dump(results, fp, indent=4)
        

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
        messages.append(response.choices[0].message)
    instance["output"] = response.choices[0].message.content
    print(response.choices[0].message.content)
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

async def main(instances: List[dict], task: int, batch_size: int = 128):
    system_prompt = load_system_prompt(task)
    for i in tqdm(range(0, len(instances), batch_size)):
        batch = instances[i:i+batch_size]
        checkpoint(await process_batch(batch, system_prompt))

if __name__ == "__main__":
    if len(sys.argv)>1:
        model = model_ids[sys.argv[1]]
        task = sys.argv[2]
    else:
        raise ValueError("Desired prompt id doesn't exist.")
    instances = load_data(task)
    
    results = asyncio.run(main(instances, task))
    # with open("results/predictions.json", "a") as fp:
    #     for r in tqdm(results):
    #         json.dump(r, fp)
    #         fp.write("\n")
