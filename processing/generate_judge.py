from typing import Dict, List
import json
import os
import sys
from tqdm import tqdm

# """[System] Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by rating the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". \n"""

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, ".."))
os.makedirs(os.path.join(script_dir, "..", "data", "judgement"), exist_ok=True)

print(os.getcwd())
script_dir = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT = """[System] Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Please rate each evaluation on a scale of 1 to 10. On this scale, 1 is for a response that completely does not match and 10 is for a response that is perfect. The beginning of your evaluation must be your rating by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". After the rating, provide a short explanation. Be as objective and concise as possible, using as little sentences as possible. Please follow the format exactly and make sure your response include your rating above all else.

For example, your answer should look like this: \n\n Rating: [[6]] \n\nExplanation: \nThe response provides a list of possible substitutions for \"minister\" but lacks a clear explanation for why each option is incorrect. The best answer is missing (answer \"C\"). While the final answer is somewhat relevant to the context, it may not necessarily provide the most accurate or helpful substitution for \"minister\", and the reasoning behind the other options is lacking."""

QA_TEMPLATE = """
[Question]
{question}

[The Start of Assistant’s Answer]
{answer} 
[The End of Assistant’s Answer]\n\n
"""


class DataInstance:
    def __init__(self, word, definition, example):
        self.word = word
        self.definition = definition
        self.example = example


def format_task(messages: List[Dict[str, str]]):
    prompt = SYSTEM_PROMPT
    for question, answer in zip(messages[:-1:2], messages[1::2]):
        prompt += QA_TEMPLATE.format(
            question=question["content"].replace("Question:", ""),
            answer=answer["content"].replace("Answer:", ""),
        )
    return [prompt]


def generate_task(file_path: str):
    results = []
    with open(file_path, "r") as f:
        data = json.load(f)
        for instance in data:
            messages = instance["output"]
            instance["messages"] = instance["output"]
            instance.pop("output")
            instance["prompt"] = format_task(messages)
            results.append(instance)
    return results


def save(model_task, prompts):
    with open(
        os.path.join(script_dir, "..", "data", "judgement", model_task), "w"
    ) as f:
        json.dump(prompts, fp=f, indent=4)


def main(data_file: str = ""):
    if data_file:
        fp = os.path.join(script_dir, "..", "results", data_file)
        generate_task(fp)
    else:
        for f in tqdm(os.listdir(os.path.join(script_dir, "..", "results"))):
            if f.endswith(".json"):
                fp = os.path.join(script_dir, "..", "results", f)
                prompts = generate_task(file_path=fp)
                save(model_task=f, prompts=prompts)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
