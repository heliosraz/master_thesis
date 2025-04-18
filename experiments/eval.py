import os
from sys import argv, path
script_dir = os.path.dirname(os.path.abspath(__file__))
path.append(os.path.join(script_dir, ".."))
from tqdm import tqdm
import json
import re

def find_score(response: str):
    score = 0
    if re.findall(r"\[*\s*(\d{1,2})\s*\]*", response):
        print(str(re.findall(r"\[*\s*(\d{1,2})\s*\]*", response)))
    else:
        print(response)
    print("#####################")
    
    # if re.search("[+.*\d{1,2}.*]+", response)
    # if re.search("Rating:", response):
    #     m = re.search("Rating:", response)
    #     print(m.string[m.start():m.end()+10])
        
    return score

def evaluate():
    record = {task: {} for task in range(1, 5)}
    result_path = os.path.join(
        script_dir, "..", "results", "judgement")
    for root, dirs, files in os.walk(result_path):
        for fp in tqdm(files):
            if not fp.endswith(".json"):
                continue
            print("------------------------------")
            print(f"File: {fp}")
            fp = os.path.join(result_path, fp)
            with open(fp, "r") as f:
                data = json.load(f)
                for instance in data:
                    task = int(instance["task"])
                    assistant = instance["assistant"]
                    if assistant not in record[task]:
                        record[task][assistant] = {}
                    score = find_score(instance["response"][-1]["content"])
                # for word, score in data:
                #     if word not in record[task][assistant]:
                #         record[task][assistant][word] = (score, 1)
                #     else:
                #         record[task][assistant][word] = (record[task][assistant][word][0]+score, record[task][assistant][word][1]+1)
    return record

if __name__ == "__main__":
    evaluate()