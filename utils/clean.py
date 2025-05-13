import json
from sys import argv
from msgspec.json import decode
import os
import random
from tqdm import tqdm
import random

script_dir = os.path.dirname(os.path.abspath(__file__))


def merge(root_path, sub_file_path):
    data = []
    for path in [root_path, sub_file_path]:
        with open(path, "r") as f:
            data += json.load(f)
    random.shuffle(data)
    with open(root_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print(len(data))


def load_json(file_path: str):
    with open(file_path, "r") as f:
        data = decode(f.read(), type=list[dict])


# root_file = argv[1]
# sub_file = argv[2]
# merge(root_file, sub_file)
# with open(root_file, "r") as fp:
#         print(len(json.load(fp)))

result_dir = os.path.join(script_dir, "..", "results", "task")
output_dir = os.path.join(script_dir, "..", "data", "annotation")
os.makedirs(output_dir, exist_ok=True)
for root, dirs, files in os.walk(result_dir):
    for fn in tqdm(files):
        print(fn)
        assistant = "-".join(fn.split("-")[:3]).upper()
        with open(os.path.join(result_dir, fn), "r") as f:
            data = decode(f.read(), type=list[dict])
            subset = random.sample(data, 5)
            for i, instance in enumerate(subset):
                messages = [
                    f"<p>{message['role'].upper()}: {message['content']}</p>"
                    for message in instance["output"]
                ]
                subset[i] = {
                    "data": {
                        "messages": "".join(messages),
                        "assistant": assistant,
                    }
                }
        
        with open(os.path.join(output_dir, fn), "w") as f:
            json.dump(subset, fp=f)
            
for i in range(1, 5):
    for root, dirs, files in os.walk(output_dir):
        fp = os.path.join(output_dir, f"task{i}.json")
        if not os.path.isfile(fp):
            with open(fp, "w") as f:
                print("[]", file = f)
        for data in files:
            if len(data.split("-"))>1 and data.split("-")[-1]==f"task{i}.json":
                dp = os.path.join(root, data)
                merge(fp, dp)
                
            
        
