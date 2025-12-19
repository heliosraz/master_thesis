import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
from sys import path
import ijson
import plotly.express as px
import plotly.graph_objects as go

script_dir = os.path.dirname(os.path.abspath(__file__))
path.append(os.path.join(script_dir, ".."))
os.makedirs(os.path.join(script_dir, "..", "results", "cluster"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "..", "data", "cluster"), exist_ok=True)
import json
from tqdm import tqdm
import torch
import pandas as pd

from msgspec.json import decode
import time


def test_msgspec():
    print("starting import...")
    s = time.process_time()
    result_path = os.path.join(script_dir, "..", "results", "embed")
    with open(
        os.path.join(result_path, "DeepSeek-R1-Distill-Llama-8B-task1-embeds.json"), "r"
    ) as f:
        data = decode(f.read(), type=list[dict])
    df = pd.DataFrame.from_dict(data)
    e = time.process_time()
    print(e - s, "seconds")


def get_parquet(
    model: str,
    embedding_types=[
        "token_prompt",
        "token_sentence",
        "definition",
        "prompt",
        "response",
    ],
):
    if not os.path.exists(os.path.join(script_dir, "..", "data", "cluster", f"{model}.parquet")):
        result_path = os.path.join(script_dir, "..", "results", "embed")
        for _, _, files in os.walk(result_path):
            rows = []
            for fn in tqdm(files):
                if fn.split("-")[0] == model:
                    print(fn)
                    task = fn.split("task")[-1].split("-")[0]
                    for instance in tqdm(get_data(fn)):
                        for embedding_type in embedding_types:
                            if f"{embedding_type}_embedding" in instance:
                                embedding = instance[f"{embedding_type}_embedding"]
                                word = instance["word"]
                                if embedding_type == "response":
                                    label = instance["output"][1]["content"]
                                else:
                                    label = instance[embedding_type.split("_")[-1]][0]
                                if type(label) == list:
                                    label = label[0]
                                embedding = np.array(embedding, dtype=np.float32)
                                rows.append(
                                    [model, word, embedding_type, task, label, embedding, -1, -1]
                                )  # adding a row
        df = pd.DataFrame(
            rows,
            columns=[
                "model",
                "word",
                "embedding_type",
                "task",
                "label",
                "embedding",
                "cluster",
                "center",
            ],
        )
        df.to_parquet(os.path.join(script_dir, "..", "data", "cluster", f"{model}.parquet"))


def cluster(X, pca, kmeans):
    X = pca.fit_transform(X)
    clustering = kmeans.fit(X)
    return clustering


def elbow(X):
    inertias = []
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    k_range = range(1, 13)
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=0)
        model.fit(X)
        inertias.append(model.inertia_)

    return k_range, inertias


def get_data(file_name: str):
    result_path = os.path.join(script_dir, "..", "results", "embed")
    file_stats = os.stat(os.path.join(result_path, file_name))
    if file_stats.st_size/(1023**3)>15:
        print(f"**FILE TOO LARGE: {file_stats.st_size/(1023**3)} GB **\n Streaming File...")
        with open(os.path.join(result_path, file_name), "rb") as f:
            for dic in ijson.items(f, "item"):
                yield dic
    else:
        print(f"** {file_stats.st_size/(1023**3)} GB **\n Reading File...")
        with open(os.path.join(result_path, file_name), "r") as f:
            data = decode(f.read(), type=list[dict])
        return data


if __name__ == "__main__":
    embedding_types = [
        "token_prompt",
        "token_sentence",
        "definition",
        "prompt",
        "response",
    ]
    selected_word = "car"
    print("Getting parquet...")
    for model in ["DeepSeek", "gemma", "Llama", "Mistral"]:
        get_parquet(model)
    kmeans = KMeans(n_clusters=5, random_state=0)
    tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3)
    pca = PCA(n_components=2)

    
    for model in ["DeepSeek", "gemma", "Llama", "Mistral"]: #["DeepSeek", "gemma", "Llama", "Mistral"]
        print(f"Loading {model}...")
        df = pd.read_parquet(
            os.path.join(script_dir, "..", "data", "cluster", model + ".parquet")
        )
        df["label"] = df["label"]\
                            .str\
                            .wrap(30)\
                            .apply(lambda x: x.replace('\n', '<br>'))
        print(f"Clustering {model}...")
        for embedding_type in embedding_types:
            for task in ["1", "2", "3", "4"]:
                if not os.path.exists(os.path.join(
                        script_dir,
                        "..",
                        "results",
                        "cluster",
                        f"{model}_{embedding_type}_{task}_clustering.png",
                    )) and embedding_type in df["embedding_type"].values:
                    mask = (df["embedding_type"] == embedding_type) & (df["task"] == task)
                    df_task_embed = df.loc[mask].reset_index()
                    # Clustering all embeddings
                    X = df_task_embed["embedding"].values
                    X = torch.tensor(np.vstack(X), dtype=torch.float32)
                    clustering = cluster(X, pca, kmeans)
                    tsne_X = tsne.fit_transform(X)
                    # Logging wrs
                    print("Logging...")
                    df.loc[mask ,"cluster"] = clustering.labels_
                    
                    
                    # Visualizing 500 embeddings per cluster
                    filtered = df_task_embed\
                        .groupby("cluster")[['word', 'embedding', 'label']]\
                        .apply(lambda x: x.sample(min(200, len(x))))
                    filtered_indices = [i[1] for i in filtered.index]
                    selected = df_task_embed[
                        (df_task_embed["word"] == selected_word)]
                    selected_indices = [i for i in selected.index]
                    clustering_labels = df_task_embed["cluster"]
                    
                    
                    
                    df_filtered =  pd.DataFrame({
                        'x': tsne_X[filtered_indices, 0],
                        'y': tsne_X[filtered_indices, 1],
                        'label': filtered.reset_index()['label'],
                        'cluster': filtered.reset_index()['cluster'],
                        "word": filtered.reset_index()["word"]
                        })                
                    df_selected = pd.DataFrame({
                        'x': tsne_X[selected_indices, 0],
                        'y': tsne_X[selected_indices, 1],
                        'label': selected['label'],
                        'cluster': selected['cluster'],
                        "word": selected['word']
                        })
                    
                    # plot clusters
                    print("Plotting...")
                    # fig = px.scatter(df_filtered, x = "x", y = "y",
                    #                 hover_data = "label",
                    #                 color = "cluster")
                    # # fig.add_scatter(df_selected,
                    # #                 hover_data = "label",
                    # #                 color = "darkorange")
                    
                    # fig.add_trace(go.Scatter(
                    #             x=df_selected['x'],
                    #             y=df_selected['y'],
                    #             mode='markers',
                    #             marker=dict(color='darkorange', size=10),
                    #             name='Selected',
                    #             text=df_selected['label'],
                    #             hoverinfo='text'
                    #         ))
                    # fig.show()
                    
                    
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    ax.scatter(df_filtered["x"],
                            df_filtered["y"],
                            c=df_filtered["cluster"],
                            marker = "+")

                    # plot the selected word 
                    ax.scatter(df_selected["x"],
                            df_selected["y"],
                            c="#e00b9f")
                    
                    ax.set_title(f"Clusters for {embedding_type} - {model} - Task {task}")
                    fig.show()

                    # print("Saving plot...")
                    plt.savefig(
                        os.path.join(
                            script_dir,
                            "..",
                            "results",
                            "cluster",
                            f"{model}_{embedding_type}_{task}_clustering.png",
                        )
                    )
                    plt.close(fig)
        df.to_parquet(
            os.path.join(script_dir, "..", "data", "cluster", model + ".parquet")
        )
    # with open(os.path.join(script_dir, "..", "results", "cluster", "clusters.json"), "w") as fp:
    #     json.dump(Xs, fp, indent=4)
    # with open(os.path.join(script_dir, "..", "results", "cluster", "labels.json"), "w") as fp:
    #     json.dump(ys, fp, indent=4)

    # cluster token_definition regardless of task
    # cluster token_sentence regardless of task
    # cluster definition regardless of task
    # plot token_definition and token_sentence clusters together
    # plot definitions and token_definition clusters together
    # cluster responses based on task
