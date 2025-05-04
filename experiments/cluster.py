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
import pandas as pd
        
def get_parquet(model: str, embedding_types = ["token_prompt",
                                    "token_sentence",
                                    "definition",
                                    "prompt",
                                    "response"]):
    result_path = os.path.join(script_dir, "..", "results", "embed")
    df = pd.DataFrame(columns = ["model", "embedding_type", "task", "label","embedding", "cluster", "center"])
    for root, dirs, files in os.walk(result_path):
        for fn in tqdm(files):   
            if fn.split("-")[0]==model:
                task = fn.split("task")[-1].split("-")[0]
                with open(os.path.join(result_path, fn), "r") as f:
                    data = json.load(f)
                    for embedding_type in embedding_types:
                        for instance in data:
                            if f"{embedding_type}_embedding" in instance:
                                embedding = instance[f"{embedding_type}_embedding"]
                                
                                if embedding_type=="response":
                                    label = instance["output"][1]["content"]
                                else:
                                    label = instance[embedding_type.split("_")[-1]]
                                df.loc[-1] = [model, embedding_type, task, label, embedding, -1, -1]  # adding a row
                                df.index = df.index + 1  # shifting index
                                df = df.sort_index()  # sorting by index
    df.to_parquet(os.path.join(script_dir, "..", "data", "cluster", f"{model}.parquet"))
    return df

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

def get_data(file_name:str):
    return pd.read_parquet(os.path.join(script_dir, "..", "data", "cluster", file_name))
    

if __name__ == "__main__":
    embedding_types = ["token_prompt", "token_sentence", "definition", "prompt", "response"]
    for model in ["DeepSeek", "gemma", "Llama", "Mistral"]:
        cluster_labels = {et: {} for et in embedding_types}
        get_parquet(model)
    # kmeans = KMeans(n_clusters=5, random_state=0)
    # tsne = TSNE(n_components=2, learning_rate='auto',
    #             init='random', perplexity=3)
    # pca = PCA(n_components=2)
    # for model in ["DeepSeek", "gemma", "Llama", "Mistral"]:
    #     df = get_data(model+".parquet")
    #     for embedding_type in embedding_types:
    #         for task in ["1","2","3","4"]:
    #             X = df.loc[(df["embedding_type"]==embedding_type)&(df["task"]==task)]["embedding"].values
    #             X = torch.tensor(np.vstack(X), dtype=torch.float32)
    #             clustering = cluster(X, pca, kmeans)
    #             df.loc((df["embedding_type"]==embedding_type)&(df["task"]==task))["cluster"] = pd.Series(clustering.labels_)
                
    #             tsne_X = tsne.fit_transform(X)
    #             # plot clusters
    #             print("Plotting...")
    #             fig = plt.figure()
    #             ax = fig.add_subplot()
    #             ax.scatter(tsne_X[:, 0], tsne_X[:, 1], c = clustering.labels_)
    #             ax.set_title(f'Clusters for {embedding_type} - {model} - Task {task}')
    #             # fig.show()
                
    #             # print("Saving plot...")
    #             plt.savefig(os.path.join(script_dir, "..", "results", "cluster", f"{model}_{embedding_type}_{task}_clustering.png"))
    #     df.to_parquet(os.path.join(script_dir, "..", "data", "cluster", model+".parquet"))
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
    