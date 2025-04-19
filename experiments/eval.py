import os
from sys import argv, path
script_dir = os.path.dirname(os.path.abspath(__file__))
path.append(os.path.join(script_dir, ".."))
from tqdm import tqdm
import json
import re
import sys
models = {'Llama-3.2-3B-Instruct', 'Mistral-7B-Instruct-v0.3', 'gemma-3-4b-it', 'gemma-3-12b-it', 'DeepSeek-R1-Distill-Llama-8B'}

def find_score(response: str):
    rating_match = re.search(r"Rating:\s*\[*\s*(\d{1,2})\s*\]*", response)
    if rating_match:
        rating = rating_match.group(1)
    else:
        fallback = re.search(r"\[*\s*(\d{1,2})\s*\]*", response)
        rating = fallback.group(1) if fallback else None
    return rating

def evaluate(file_name: str = ""):
    fail_count = 0
    record = {task: {judge: {assist: {} for assist in models} for judge in models} for task in range(1, 5)}
    fails = {judge: 0 for judge in models}
    if file_name:
        fp = os.path.join(
            script_dir, "..", "results", "judgement", file_name)
        with open(fp, "r") as f:
            data = json.load(f)
            for instance in data:
                task = int(instance["task"])
                judge = instance["judge"]
                assistant = instance["assistant"]
                word = instance["word"]
                rating = find_score(instance["response"][-1]["content"])
                if not rating:
                    fail_count += 1
                if word not in record[task][judge][assistant]:
                    record[task][judge][assistant][word] = (0, 0)
                if rating:
                    record[task][judge][assistant][word] = (record[task][judge][assistant][word][0]+int(rating), record[task][judge][assistant][word][1]+1)
                else:
                    fails[judge] += 1
    else:
        result_path = os.path.join(
            script_dir, "..", "results", "judgement")
        for root, dirs, files in os.walk(result_path):
            for fp in tqdm(files):
                if not fp.endswith(".json"):
                    continue
                fp = os.path.join(result_path, fp)
                with open(fp, "r") as f:
                    data = json.load(f)
                    for instance in data:
                        task = int(instance["task"])
                        judge = instance["judge"]
                        assistant = instance["assistant"]
                        word = instance["word"]
                        rating = find_score(instance["response"][-1]["content"])
                        if not rating:
                            fail_count += 1
                        if word not in record[task][judge][assistant]:
                            record[task][judge][assistant][word] = (0, 0)
                        if rating:
                            record[task][judge][assistant][word] = (record[task][judge][assistant][word][0]+int(rating), record[task][judge][assistant][word][1]+1)
                        else:
                            fails[judge] += 1
    # print(record)
    # print(fails)
    return record

if __name__ == "__main__":
    if len(sys.argv)>1:
        evaluate(sys.argv[1])
    else:
        evaluate()