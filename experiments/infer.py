from tqdm.asyncio import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field
from data_processing import load_system, load_data, prompt_template

import sys
import os
import json
from typing import List


script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
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

import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)
model = sys.argv[1]
status = 0

async def run_async(instance, system_prompt, schema, model):
    prompt = prompt_template(instance[1]['precontext'],
                    instance[1]['sentence'],
                    instance[1]['ending'])
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format=schema
    )
    
    result = json.loads(response.choices[0].message.content)
    return {"id": instance[0], "prediction": result["score"]}

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
