import json
from huggingface_hub import login

with open("./secrets.json", "r") as fp:
    secrets = json.load(fp)
    login(token=secrets["huggingface"])
