import json

def load_data(task: str, path: str="../data/tasks"):
    fp = f"{path}/task{task}.json"
    with open(fp, "r") as f:
        data = json.load(f)
    return data

def load_system_prompt(id: str, path: str = "data/prompts/system_prompt.jsonl"):
    data = []
    with open(path, "r") as f:
        for line in f:
            prompt = json.load(line)
            if prompt["id"]==id:
                return prompt["prompt"]
    raise KeyError("Desired prompt id doesn't exist.")
    