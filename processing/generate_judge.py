from typing import Dict, List
import json
import os
import sys
import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, ".."))

print(os.getcwd())
script_dir = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT = """[System] Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". \n"""

QA_TEMPLATE = """
[Question]
{question}

[The Start of Assistant’s Answer]
{answer} 
[The End of Assistant’s Answer]\n\n
"""


class DataInstance():
    def __init__(self, word, definition, example):
        self.word = word
        self.definition = definition
        self.example = example


def format_task(messages: List[Dict[str, str]]):
    prompt = SYSTEM_PROMPT
    for question, answer in zip(messages[:-1:2], messages[1::2]):
        prompt += QA_TEMPLATE.format(
            question=question["content"].replace("Question:", ""), answer=answer["content"].replace("Answer:", ""))
    return prompt


def generate_task(file_path: str):
    results = []
    with open(file_path, "r") as f:
        data = json.load(f)
        for instance in data:
            messages = instance["output"]
            instance.pop("output")
            instance["prompt"] = format_task(messages)
            results.append(instance)
    return results

def save(model_task, prompts):
    with open(os.path.join(script_dir, "..", "data", "judgement", model_task+".json"),"w") as f:
        json.dump(prompts, fp = f, indent = 4)


def main(data_file: str = ""):
    if data_file:
        generate_task(data_file)
    else:
        for root, dirs, files in os.walk(os.path.join(script_dir, "..", "results")):
            for f in tqdm(files):
                print(f)
                fp = os.path.join(script_dir, "..", "results", f)
                prompts = generate_task(file_path=fp)
                save(model_task = f, prompts = prompts)


if __name__ == "__main__":
    main()
