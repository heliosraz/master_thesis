import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import re
import json
from tqdm import tqdm
import os
from sys import argv, path
import pandas as pd
from msgspec.json import decode
from typing import List, Set
script_dir = os.path.dirname(os.path.abspath(__file__))
path.append(os.path.join(script_dir, ".."))
judges = [
    "Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-3-4b-it",
    "DeepSeek-R1-Distill-Llama-8B",
]



def find_score(response: str):
    rating_match = re.search(
        r"Rating:\s*(?<!\d)\[*\s*(10|[1-9])\s*\]*(?!\d|\.)", response
    )
    if rating_match:
        if len(rating_match.groups()) > 1:
            print(rating_match.groups())
        rating = rating_match.group(1)
    else:
        fallback = re.search(r"(?<!\d)\[*\s*(10|[1-9])\s*\]*(?!\d|\.)", response)
        if fallback and len(fallback.groups()) > 1:
            print(fallback.groups())
        rating = fallback.group(1) if fallback else None
    return rating


def evaluate(file_name: str = "", test="2"):
    """_summary_

    Args:
        file_name (str, optional): _description_. Defaults to "".
        test (str, optional): _description_. Defaults to "2".

    Returns:
        results (dict[int, dict[str, dict[str, dict]]]): a dictionary split up by {task {judge: {assistant: {word}}}}
    """
    assistants = [
        "Llama-3.2-3B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "DeepSeek-R1-Distill-Llama-8B",
    ]
    fail_count = 0
    record = {
        task: {
            judge: {
                assist: {
                    } for assist in assistants
                } for judge in judges}
        for task in range(1, 5)
    }
    keys = {1:["definition", "sentence"], 2:["sentence"], 3:["definition"], 4:["sentence"]}
    fails = {judge: 0 for judge in judges}
    result_path = os.path.join(script_dir, "..", "results", "judgment", test)
    for root, dirs, files in os.walk(result_path):
        for fp in tqdm(files):
            # if not fp.endswith(".json"):
            #     continue
            fp = os.path.join(result_path, fp)
            with open(fp, "r") as f:
                data = json.load(f)
                for instance in data:
                    task = int(instance["task"])
                    judge = instance["judge"]
                    assistant = instance["assistant"]
                    word = instance["word"]
                    rating = find_score(instance["response"][-1]["content"])
                    if word not in record[task][judge][assistant]:
                        record[task][judge][assistant][word] = {"total": 0,
                                                                "count": 0}
                    if not rating:
                        fail_count += 1
                        fails[judge] += 1
                    else:
                        key = (instance[p] for p in keys[task])
                        record[task][judge][assistant][word][key] = int(rating) / 10
                        record[task][judge][assistant][word]["count"] = record[task][judge][assistant][word]["count"] + 1
                        record[task][judge][assistant][word]["total"] = record[task][judge][assistant][word]["total"] + int(rating) / 10

    # print(record)
    print(fails)
    return record


def plot_1(
    results: dict[int, dict[str, dict[str, dict]]],
    assistants=[
        "Llama-3.2-3B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "gemma-3-4b-it",
        "DeepSeek-R1-Distill-Llama-8B",
    ],
):
    palette = sns.color_palette("tab10", len(assistants))
    color_map = dict(zip(assistants, palette))
    fit = {
        task: {
            judge: {
                assist: {
                    "m": 0, "b": 0
                    } for assist in assistants
                } for judge in judges
        } for task in range(1, 5)
    }
    fig, axs = plt.subplots(len(results), len(results[1]))
    for row, task in enumerate(results):
        fname = "task" + str(task + 1) + ".jpg"
        for col, judge in enumerate(judges):
            table_lines = []
            for assistant in assistants:
                ax = axs[row][col]
                x = [score["count"] for score in results[task][judge][assistant].values()]
                y = [score["total"] for score in results[task][judge][assistant].values()]
                if x and y:
                    limit = max(x + y)
                    ax.plot(
                        [0, limit],
                        [0, limit],
                        "--",
                        color="gray",
                        linewidth=1,
                        zorder=0,
                    )
                    ax.scatter(
                        x,
                        y,
                        color=color_map[assistant],
                        s=1,
                        label="-".join(assistant.split("-")[:3]).capitalize(),
                    )
                    m, b, r_value, p_value, std_err = scipy.stats.linregress(x, y)(x, y, 1)

                    ax.plot(
                        x,
                        [m * coord + b for coord in x],
                        "--",
                        color=color_map[assistant],
                        linewidth=1.0,
                    )

                    if row == len(results[task]) - 1:
                        ax.set_xlabel(
                            f"Judge:\n{'-'.join(judge.split('-')[:3]).capitalize()}",
                            fontsize=9,
                        )
                    if col == 0:
                        ax.set_ylabel(f"Task:\n{task}", fontsize=9)

                    fit[task][judge][assistant].update({"m": m, "b": b, "r_value": r_value, "p_value": p_value, "std_err": std_err})
                    table_lines.append((assistant, m, b))
                    
            for i, line in enumerate(table_lines):
                assistant, m, b = line
                label = f"$m$={m:.2f}, $b$={b:.2f}"
                ax.text(0.02, 0.98 - i*0.12, label,
                        transform=ax.transAxes,
                        fontsize=8, va='top', color=color_map[assistant])

    all_handles_labels = []
    for row in range(len(results)):
        for col in range(len(judges)):
            handles, labels = axs[row][col].get_legend_handles_labels()
            all_handles_labels.extend([(l, h) for h, l in zip(handles, labels)])
    all_handles_labels = dict(all_handles_labels)
    fig.legend(
        [handle for handle in all_handles_labels.values()],
        [label for label in all_handles_labels],
        ncol=3,
        loc="upper center",
        fontsize=12,
        markerscale=10,
    )
    plt.show()
    return fit


def plot_2(
    results: dict[int, dict[str, dict[str, dict]]],
    assistants=[
        "Llama-3.2-3B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "gemma-3-4b-it",
        "DeepSeek-R1-Distill-Llama-8B",
    ],
    tasks=[2, 3, 4],
):
    palette = sns.color_palette("tab10", len(tasks))
    color_map = dict(zip(tasks, palette))
    fit = {
        assist: {
            judge : {
                task: {
                    "m": 0, "b": 0
                    } for task in range(1, 5)
                } for judge in judges
            } for assist in assistants
    }
    fig, axs = plt.subplots(len(assistants), len(judges))
    for row, assistant in enumerate(assistants):
        for col, judge in enumerate(judges):
            table_lines = []
            for task in tasks:
                ax = axs[row][col]
                x = [score["count"] for score in results[task][judge][assistant].values()]
                y = [score["total"] for score in results[task][judge][assistant].values()]
                if x and y:
                    limit = max(x + y)
                    ax.plot(
                        [0, limit],
                        [0, limit],
                        "--",
                        color="gray",
                        linewidth=1,
                        zorder=0,
                    )
                    ax.scatter(
                        x,
                        y,
                        color=color_map[task],
                        s=1,
                        label=f"Task {task}" if (row == 0 and col == 0) else "",
                    )
                    m, b, r_value, p_value, std_err = scipy.stats.linregress(x, y)(x, y, 1)

                    ax.plot(
                        x,
                        [m * coord + b for coord in x],
                        "--",
                        color=color_map[task],
                        linewidth=1.0,
                    )

                    if row == len(assistants) - 1:
                        ax.set_xlabel(
                            f"Judge:\n{'-'.join(judge.split('-')[:3]).capitalize()}",
                            fontsize=9,
                        )
                    if col == 0:
                        ax.set_ylabel(
                            f"Assistant:\n{'-'.join(assistant.split('-')[:3]).capitalize()}",
                            fontsize=9,
                        )
                    fit[task][judge][assistant].update({"m": m, "b": b, "r_value": r_value, "p_value": p_value, "std_err": std_err})
                    table_lines.append((task, m, b))
            for i, line in enumerate(table_lines):
                task, m, b = line
                label = f"$m$={m:.2f}, $b$={b:.2f}"
                ax.text(0.02, 0.98 - i*0.12, label,
                        transform=ax.transAxes,
                        fontsize=8, va='top', color=color_map[task])

    all_handles_labels = []
    for row in range(len(results)):
        for col in range(len(judges)):
            handles, labels = axs[row][col].get_legend_handles_labels()
            all_handles_labels.extend([(l, h) for h, l in zip(handles, labels)])
    all_handles_labels = dict(all_handles_labels)
    fig.legend(
        [handle for handle in all_handles_labels.values()],
        [label for label in all_handles_labels],
        ncol=3,
        loc="upper center",
        fontsize=12,
        markerscale=10,
    )
    plt.show()
    return fit


def cat_plot(
    results: dict[int, dict[str, dict[str, dict]]],
    selected: Set[str] = {"book", "system"},
    ):
    """_summary_

    Args:
        results (dict[int, dict[str, dict[str, dict]]]): a dictionary split up by {task {judge: {assistant: {word}}}}
        assistants (list, optional): _description_. Defaults to [ "Llama-3.2-3B-Instruct", "Mistral-7B-Instruct-v0.3", "gemma-3-4b-it", "DeepSeek-R1-Distill-Llama-8B", ].
        words (List[str], optional): _description_. Defaults to [].
    """ 
    results = [{"task": task, 
                "judge": judge, 
                "assistant": assistant, 
                "word": word, 
                "key": key,
                "score": score}
                        for task, judges in results.items()
                        for judge, assistants in judges.items()
                        for assistant, words in assistants.items() 
                        for word, keys in words.items()
                        for key, score in keys.items()
                        if key not in {"count", "total"}
                        ]
    results = pd.DataFrame(results, columns=["task", "judge", "assistant", "word", "key", "score"])
    words = results["word"].unique()
    greys = iter(sns.color_palette("Greys", len(words)))
    palette = sns.color_palette("hls", len(words))
    pal = {word: palette[i] if word in selected else next(greys) for i, word in enumerate(words)}

    g = sns.FacetGrid(results, col="assistant")
    print("Plotting violin plot...")
    g.map_dataframe(sns.violinplot, "task", "score", order = ["1","2","3","4"], color=".9", inner=None)
    print("Plotting swarm plot...")
    g.map_dataframe(sns.swarmplot,  "task", "score", hue="word", palette=pal, order = ["1","2","3","4"], data = results)
    print("Rendering...")
    plt.show()
    
def annot_eval():
    annotation_path = os.path.join(script_dir, "..", "data", "annotation")
    judgment_path = os.path.join(script_dir, "..", "results", "judgment", "2")
    record = {}
    for root, _, files in os.walk(judgment_path):
        for fn in tqdm(files):
            with open(os.path.join(root, fn), "r") as f:
                data = decode(f.read(), type=list[dict])
                print("Loaded", fn)
                for instance in tqdm(data):
                    response = instance["messages"][1]["content"]
                    judgment = instance["response"][-1]["content"]
                    assistant = instance["assistant"]
                    score = find_score(judgment) if find_score(judgment) else -1
                    judge = instance["judge"]
                    if response not in record:
                        record[response] = []
                    record[response].append({"score": score,
                                            "judge": judge,
                                            "assistant": assistant,
                                            "judgment": judgment})
    result = []
    for root, _, files in os.walk(annotation_path):
        for fn in tqdm(files):
            with open(os.path.join(root, fn), "r") as f:
                data = decode(f.read(), type=list[dict])
                for instance in tqdm(data):
                    response = instance["data"]["messages"]\
                            .split("<p>")[2][11:-4]
                    score = [r["value"].values() for r in instance["annotations"][0]["result"]]
                    judge = fn
                    curr = {"response": response, "human_score": str(score)}
                    for judge in record[response]:
                        result.append({**curr, **judge})
    with open(os.path.join(annotation_path, "annotation.json"), "w") as f:
        json.dump(result, f)
        
def judge_annot():
    annotation_path = os.path.join(script_dir, "..", "data", "annotation")
    judgment_path = os.path.join(script_dir, "..", "results", "judgment", "2")
    result = []
    for root, _, files in os.walk(judgment_path):
        for fn in tqdm(files):
            with open(os.path.join(root, fn), "r") as f:
                data = decode(f.read(), type=list[dict])
                print("Loaded", fn)
                for instance in tqdm(data):
                    response = instance["messages"][1]["content"]
                    judgment = instance["response"][-1]["content"]
                    assistant = instance["assistant"]
                    score = find_score(judgment) if find_score(judgment) else "-1"
                    judge = instance["judge"]
                    if int(score) < 4:
                        result.append({"score": score,
                                            "judge": judge,
                                            "assistant": assistant,
                                            "judgment": judgment,
                                            "response": response})
    print(len(result))
    with open(os.path.join(annotation_path, "judgments.json"), "w") as f:
        json.dump(result, f, indent=4)
    

from collections import Counter
from nltk.tokenize import word_tokenize
def word_count(data:str):
    words = word_tokenize(data.lower())
    return Counter(words)

def jist(data_root: str = "results/judgment/2"):
    path = os.path.join(script_dir, "..", data_root)
    results = {judge: Counter() for judge in judges}
    for root, _, files in os.walk(path):
        for fn in tqdm(files):
            with open(os.path.join(root, fn), "r") as f:
                data = decode(f.read(), type=list[dict])
                for instance in data:
                    results[instance["judge"]].update(word_count(instance["messages"][-1]["content"]))
                    
#     micolumns = pd.MultiIndex.from_tuples(
#     [(judge, "word"), (judge, "count") for judge in judges], names=["lvl0", "lvl1"]
# )
    table = pd.DataFrame(columns = judges)\
        .from_dict({judge: [word[0] for word in counter.most_common(1000)] for judge, counter in results.items()})
    table.style.to_latex("test.tex")

if __name__ == "__main__":
    # annot_eval()
    # judge_annot()
    if len(sys.argv) > 1:
        results = evaluate(sys.argv[1])
    else:
        results = evaluate()
    # test = {
    #     task: {
    #         judge: {
    #             assist: { 
    #                 "word": {
    #                     "test1": 0,
    #                     "test2": 143,
    #                     "count": 1,
    #                     "total": 1
    #                 }
    #                 } for assist in ['test']
    #             } for judge in ['test']}
    #     for task in range(1, 5)
    # }
        
    cat_plot(results)
    # task_lines = {}
    # assistant_lines = {}

    # # assistant_lines.update(plot_1(results))
    # # assistant_lines.update(
    # #     plot_1(results, assistants=["gemma-3-4b-it", "gemma-3-12b-it"])
    # # )

    # # plot_2(results)
    # plot_2(results, tasks=[1,2,3,4])
    
    # jist()
