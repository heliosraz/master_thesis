import json

def load_data(task: str, path: str="../data/tasks"):
    fp = f"{path}/task{task}.json"
    with open(fp, "r") as f:
        data = json.load(f)
    return data

def load_system_prompt(id: str, path: str = "data/prompts/system_prompt.jsonl"):
    with open(path, "r") as f:
        for line in f:
            prompt = json.loads(line)
            if prompt["id"]==id:
                return prompt["SYSTEM_PROMPT"]
    raise KeyError("Desired prompt id doesn't exist.")

if __name__ == "__main__":
    print(load_system_prompt("1"))
    