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
import scipy
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


def evaluate(file_name: str = "", test="3"):
    """_summary_

    Args:
        file_name (str, optional): if specified, the evaluation is only done on one file.
        test (str, optional): defines which iteration of the judgement data to use.

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
    total = {judge: 0 for judge in judges}
    fails = {judge: 0 for judge in judges}
    result_path = os.path.join(script_dir, "..", "results", "judgement", test, file_name)
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
                    total[judge] += 1
                    if not rating:
                        fail_count += 1
                        fails[judge] += 1
                    else:
                        key = tuple([instance[p] for p in keys[task]])
                        record[task][judge][assistant][word][key] = int(rating) / 10
                        record[task][judge][assistant][word]["count"] = record[task][judge][assistant][word]["count"] + 1
                        record[task][judge][assistant][word]["total"] = record[task][judge][assistant][word]["total"] + int(rating) / 10

    # print(record)
    print(fails)
    print(total)
    print({judge: fails[judge]/total[judge] for judge in fails})
    return record

def statistics(results: dict[int, dict[str, dict[str, dict]]]) -> pd.DataFrame:
    stats = {(task, judge, assistant): values
        for task, judges in results.items()
        for judge, assistants in judges.items()
        for assistant, values in assistants.items()}
    return pd.DataFrame.from_dict(stats, orient="index")

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
                    limit = 4480
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
                    m, b, r_value, p_value, std_err = scipy.stats.linregress(x, y)

                    ax.plot(
                        np.arange(limit),
                        [m * coord + b for coord in np.arange(limit)],
                        "-",
                        color=color_map[assistant],
                        linewidth=1.0,
                    )

                    if row == len(results[task]) - 1:
                        ax.set_xlabel(
                            f"Judge:\n{'-'.join(judge.split('-')[:3]).capitalize()}",
                            fontsize=12,
                        )
                    if col == 0:
                        ax.set_ylabel(f"Task:\n{task}", fontsize=12)

                    fit[task][judge][assistant].update({"m": m, "b": b, "r_value": r_value, "p_value": p_value, "std_err": std_err})
                    table_lines.append((assistant, m, b))
                    
            for i, line in enumerate(table_lines):
                assistant, m, b = line
                label = f"$m$={m:.2f}, $b$={b:.2f}"
                ax.text(0.02, 0.98 - i*0.12, label,
                        transform=ax.transAxes,
                        fontsize=12, va='top', color=color_map[assistant])

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
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.savefig(os.path.join(
        script_dir,
        "..",
        "results",
        'judgement',
        "plot1.png"
    ), dpi = 200)
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
                    limit = 4480
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
                    m, b, r_value, p_value, std_err = scipy.stats.linregress(x, y)

                    ax.plot(
                        np.arange(limit),
                        [m * coord + b for coord in np.arange(limit)],
                        "-",
                        color=color_map[task],
                        linewidth=1.0,
                    )

                    if row == len(assistants) - 1:
                        ax.set_xlabel(
                            f"Judge:\n{'-'.join(judge.split('-')[:3]).capitalize()}",
                            fontsize=12,
                        )
                    if col == 0:
                        ax.set_ylabel(
                            f"Assistant:\n{'-'.join(assistant.split('-')[:3]).capitalize()}",
                            fontsize=12,
                        )
                    fit[assistant][judge][task].update({"m": m, "b": b, "r_value": r_value, "p_value": p_value, "std_err": std_err})
                    table_lines.append((task, m, b))
            for i, line in enumerate(table_lines):
                task, m, b = line
                label = f"$m$={m:.2f}, $b$={b:.2f}"
                ax.text(0.02, 0.98 - i*0.12, label,
                        transform=ax.transAxes,
                        fontsize=12, va='top', color=color_map[task])

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
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.savefig(os.path.join(
        script_dir,
        "..",
        "results",
        'judgement',
        "plot2.png"
    ), dpi = 200)
    return fit


def cat_plot(
    results: dict[int, dict[str, dict[str, dict]]],
    selected_words: Set[str] = {"organization", "system", "administration"},
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
                        if key not in {"count", "total"} and assistant != "gemma-3-12b-it" and score >= 0
                        ]
    results = pd.DataFrame(results, columns=["task", "judge", "assistant", "word", "key", "score"])
    words = results["word"].unique()
    greys = iter(sns.color_palette("Greys", len(words)))
    palette = sns.color_palette("tab10", len(words))
    pal = {word: palette[i] if word in selected_words else next(greys) for i, word in enumerate(words)}
    g = sns.FacetGrid(results, col="assistant")
    g.map_dataframe(sns.boxplot, "task", "score", order = ["1","2","3","4"], fill=False, dodge=True)
    print("Plotting violin plots...")
    for word in selected_words:
        filtered = results[results["word"].isin({word})]
        for ax, (name, subdata) in zip(g.axes.flat, filtered.groupby("assistant")):
            sns.violinplot(x="task", y="score", order = ["1","2","3","4"], hue="word", palette=pal, inner=None, data=subdata, ax=ax, alpha=0.3)
            ax.legend_.remove()
        # print("Plotting strip plot...")
        # g.map_dataframe(sns.scatterplot, "task", "score", data=df_counts, hue="word", palette=pal, alpha=0.3, size="count")
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    g.add_legend({label: handle for label, handle in zip(labels, handles)}, title="Word")
    print("Rendering...")
    # plt.show()
    plt.savefig(os.path.join(
        script_dir, 
        "..",
        "results",
        "judgement",
        "cat_plot.png"
    ))
    
def annot_eval():
    annotation_path = os.path.join(script_dir, "..", "data", "annotation")
    judgement_path = os.path.join(script_dir, "..", "results", "judgement", "2")
    record = {}
    for root, _, files in os.walk(judgement_path):
        for fn in tqdm(files):
            with open(os.path.join(root, fn), "r") as f:
                data = decode(f.read(), type=list[dict])
                print("Loaded", fn)
                for instance in tqdm(data):
                    response = instance["messages"][1]["content"]
                    judgement = instance["response"][-1]["content"]
                    assistant = instance["assistant"]
                    score = find_score(judgement) if find_score(judgement) else -1
                    judge = instance["judge"]
                    if response not in record:
                        record[response] = []
                    record[response].append({"score": score,
                                            "judge": judge,
                                            "assistant": assistant,
                                            "judgement": judgement})
    result = []
    for root, _, files in os.walk(annotation_path):
        for fn in tqdm(files):
            if fn in {"1.json", "2.json", "3.json"}:
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
    judgement_path = os.path.join(script_dir, "..", "results", "judgement", "2")
    result = []
    for root, _, files in os.walk(judgement_path):
        for fn in tqdm(files):
            with open(os.path.join(root, fn), "r") as f:
                data = decode(f.read(), type=list[dict])
                print("Loaded", fn)
                for instance in tqdm(data):
                    response = instance["messages"][1]["content"]
                    judgement = instance["response"][-1]["content"]
                    assistant = instance["assistant"]
                    score = find_score(judgement) if find_score(judgement) else -1
                    judge = instance["judge"]
                    if int(score) < 4:
                        result.append({"score": score,
                                            "judge": judge,
                                            "assistant": assistant,
                                            "judgement": judgement,
                                            "response": response})
    print(len(result))
    with open(os.path.join(annotation_path, "judgements.json"), "w") as f:
        json.dump(result, f, indent=4)
    

from collections import Counter
from nltk.tokenize import word_tokenize
def word_count(data:str):
    words = word_tokenize(data.lower())
    return Counter(words)

def jist(data_root: str = "results/judgement/2"):
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
        
    # cat_plot(results)

    # statistics(plot_1(results)).to_latex("plot1.tex")
    # # plot_1(results, assistants=["gemma-3-4b-it", "gemma-3-12b-it"])

    # statistics(plot_2(results, tasks=[1,2,3,4])).to_latex("plot2.tex")
    
    # jist()
