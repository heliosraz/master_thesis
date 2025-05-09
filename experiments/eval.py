import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import re
import json
from tqdm import tqdm
import os
from sys import argv, path

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
    assistants = [
        "Llama-3.2-3B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "DeepSeek-R1-Distill-Llama-8B",
    ]
    fail_count = 0
    record = {
        task: {judge: {assist: {} for assist in assistants} for judge in judges}
        for task in range(1, 5)
    }
    fails = {judge: 0 for judge in judges}
    if file_name:
        fp = os.path.join(script_dir, "..", "results", "judgement", test, file_name)
        with open(fp, "r") as f:
            data = json.load(f)
            for instance in data:
                task = int(instance["task"])
                judge = instance["judge"]
                assistant = instance["assistant"]
                word = instance["word"]
                rating = find_score(instance["response"][-1]["content"])
                if word not in record[task][judge][assistant]:
                    record[task][judge][assistant][word] = (0, 0)
                if not rating:
                    fail_count += 1
                    fails[judge] += 1
                else:
                    record[task][judge][assistant][word] = (
                        record[task][judge][assistant][word][0] + int(rating) / 10,
                        record[task][judge][assistant][word][1] + 1,
                    )
    else:
        result_path = os.path.join(script_dir, "..", "results", "judgement", test)
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
                        if word not in record[task][judge][assistant]:
                            record[task][judge][assistant][word] = (0, 0)
                        if not rating:
                            fail_count += 1
                            fails[judge] += 1
                        else:
                            record[task][judge][assistant][word] = (
                                record[task][judge][assistant][word][0] + 1,
                                record[task][judge][assistant][word][1]
                                + int(rating) / 10,
                            )

    # print(record)
    print(fails)
    return record


def plot_1(
    results,
    assistants=[
        "Llama-3.2-3B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "gemma-3-4b-it",
        "DeepSeek-R1-Distill-Llama-8B",
    ],
):
    palette = sns.color_palette("tab10", len(assistants))
    color_map = dict(zip(assistants, palette))
    slopes = {
        task: {
            judge: {assist: {"m": 0, "b": 0} for assist in assistants}
            for judge in judges
        }
        for task in range(1, 5)
    }
    fig, axs = plt.subplots(len(results), len(results[1]))
    for row, task in enumerate(results):
        fname = "task" + str(task + 1) + ".jpg"
        for col, judge in enumerate(judges):
            x = []
            y = []
            for assistant in assistants:
                ax = axs[row][col]
                x = [score[0] for score in results[task][judge][assistant].values()]
                y = [score[1] for score in results[task][judge][assistant].values()]
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
                    m, b = np.polyfit(x, y, 1)

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

                    slopes[task][judge][assistant].update({"m": m, "b": b})

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
    return slopes


def plot_2(
    results,
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
    slopes = {
        task: {
            judge: {assist: {"m": 0, "b": 0} for assist in assistants}
            for judge in judges
        }
        for task in range(1, 5)
    }
    fig, axs = plt.subplots(len(assistants), len(judges))
    result_path = os.path.join(script_dir, "..", "results")
    for task in results:
        if task in tasks:
            fname = "task" + str(task) + ".jpg"
            for col, judge in enumerate(judges):
                x = []
                y = []
                for row, assistant in enumerate(assistants):
                    ax = axs[row][col]
                    x = [score[0] for score in results[task][judge][assistant].values()]
                    y = [score[1] for score in results[task][judge][assistant].values()]
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
                        m, b = np.polyfit(x, y, 1)

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

                        slopes[task][judge][assistant].update({"m": m, "b": b})

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
    return slopes


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results = evaluate(sys.argv[1])
    else:
        results = evaluate()
    task_lines = {}
    assistant_lines = {}

    assistant_lines.update(plot_1(results))
    assistant_lines.update(
        plot_1(results, assistants=["gemma-3-4b-it", "gemma-3-12b-it"])
    )

    task_lines.update(plot_2(results))
    task_lines.update(plot_2(results, tasks=[1]))
