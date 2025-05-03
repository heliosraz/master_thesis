import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
from sys import path
script_dir = os.path.dirname(os.path.abspath(__file__))
path.append(os.path.join(script_dir, ".."))
os.makedirs(os.path.join(script_dir, "..", "results", "cluster"), exist_ok=True)
import json
from tqdm import tqdm
import torch

def collect_embeddings(data_path, type, model = "all"):
    embeddings = {}
    with open(data_path, "w") as f:
        data = json.load(f)
    model_id = "-".join(data_path.split("/").split("-")[:-1])
    if model_id not in embeddings:
        embeddings[model_id] = []
    for instance in data:
        embeddings[model_id] = instance[f"{type}_embedding"]
        
def get_embeddings(embedding_types = ["token_prompt",
                                    "token_sentence",
                                    "definition",
                                    "prompt",
                                    "response"]):
    result_path = os.path.join(script_dir, "..", "results", "embed")
    embeddings = {et: {} for et in embedding_types}
    labels = {et: {} for et in embedding_types}
    # plt.xlabel('k')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method For Optimal k')
    # xticks = np.arange(1, 21, 1)
    # plt.xticks(xticks)
    for root, dirs, files in os.walk(result_path):
        for fn in tqdm(files):
        # for fn in tqdm(["DeepSeek-R1-Distill-Llama-8B-response_task1-embeds.json"]):
            model = "-".join(fn.split("-")[:4])
            task = fn.split("task")[-1].split("-")[0]
            with open(os.path.join(root, fn), "r") as f:
                data = json.load(f)
                for embedding_type in embedding_types:
                    for instance in data:
                        if model not in embeddings[embedding_type]:
                            embeddings[embedding_type][model]= {i: [] for i in range(1,5)}
                        if model not in labels[embedding_type]:
                            labels[embedding_type][model]= {i: [] for i in range(1,5)}
                        if f"{embedding_type}_embedding" in instance:
                            embeddings[embedding_type][model][int(task)].append(instance[f"{embedding_type}_embedding"])
                            if embedding_type=="response":
                                labels[embedding_type][model][int(task)].append(instance["output"][1]["content"])
                            else:
                                labels[embedding_type][model][int(task)].append(instance[embedding_type.split("_")[-1]])
                    # TESTING how many cluster to use:
                    # if f"{embedding_type}_embedding" in instance:           
                    #     k_range, inertias = elbow(embeddings[embedding_type][model][int(task)])
                    #     plt.plot(k_range, inertias, '-', linewidth=1)
                    #     # plt.savefig(os.path.join(script_dir, "..", "results", "cluster", f"{model}_{embedding_type}_{task}_elbow.png"))
    # plt.savefig(os.path.join(script_dir, "..", "results", "cluster", "elbow.png"))
    # plt.show()
    return embeddings, labels

def cluster(X, pca, kmeans):
    X = pca.fit_transform(X)
    clustering = kmeans.fit(X)
    return clustering.labels_, clustering.cluster_centers_
            
        
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

if __name__ == "__main__":
    embedding_types = ["token_prompt", "token_sentence", "definition", "prompt", "response"]
    cluster_labels = {et: {} for et in embedding_types}
    Xs, ys = get_embeddings()
    kmeans = KMeans(n_clusters=5, random_state=0)
    tsne = TSNE(n_components=2, learning_rate='auto',
                init='random', perplexity=3)
    pca = PCA(n_components=2)
    for i, (model_X, model_y) in enumerate(zip(Xs.values(), ys.values())):
        if model_X:
            for task_X, task_y in zip(model_X.values(), model_y.values()):
                for X, y in tqdm(zip(task_X.values(), task_y.values()),total = len(task_X), desc="Clustering"):
                    if X:
                        X = torch.Tensor(X)
                        task_X["cluster_labels"], task_X["cluster_centers"]= cluster(X, pca, kmeans)
                        # tsne_X = tsne.fit_transform(X)
                        # # plot clusters
                        # print("Plotting...")
                        # fig = plt.figure()
                        # ax = fig.add_subplot()
                        # ax.scatter(tsne_X[:, 0], tsne_X[:, 1], c = task_X["cluster_labels"])
                        # ax.set_title(f'Clusters for {embedding_types[i]} - {model_X}')
                        # fig.show()
                        # print("Saving plot...")
                        # plt.savefig(os.path.join(script_dir, "..", "results", "cluster", f"{model_X}_{embedding_types[i]}_clustering.png"))
    with open(os.path.join(script_dir, "..", "results", "cluster", "clusters.json"), "w") as fp:
        json.dump(Xs, fp, indent=4)
    with open(os.path.join(script_dir, "..", "results", "cluster", "labels.json"), "w") as fp:
        json.dump(ys, fp, indent=4)
        
        
        # cluster token_definition regardless of task
        # cluster token_sentence regardless of task
        # cluster definition regardless of task
        # plot token_definition and token_sentence clusters together
        # plot definitions and token_definition clusters together
        # cluster responses based on task
    