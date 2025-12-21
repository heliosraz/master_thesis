import json

def load_data(task: str, path: str="../data/tasks"):
    fp = f"{path}/task{task}.json"
    with open(fp, "r") as f:
        data = json.load(f)
    return data

def load_system_prompt(id: str, path: str = "data/prompts/system_prompt.jsonl"):
    with open(path, "r") as fp:
        data = json.load(fp)

    for inst in data:
        if inst["id"]==id:
            return inst
    else:
        raise KeyError("Desired prompt id doesn't exist.")
    