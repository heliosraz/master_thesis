from typing import Dict, Tuple, Iterable
from nltk.corpus import wordnet as wn
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
import itertools
import pandas as pd
from sys import argv, path

path.append(os.path.join(script_dir, ".."))

print(os.getcwd())
os.makedirs(os.path.join(script_dir, "..", "data", "corpora"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "..", "data", "embed"), exist_ok=True)


class DataInstance:
    def __init__(self, word, definition, example):
        self.word = word
        self.definition = definition
        self.example = example


def generate_examples(word: str, pairs: Iterable[Tuple[str, str]]):
    examples = []
    for defn, sentences in pairs:
        filter_sentence = [s for s in sentences if word in s]
        if defn and filter_sentence:
            examples.append(
                {
                    "word": word,
                    "definition": defn,
                    "sentence": filter_sentence[0],
                    "prompt": filter_sentence,
                }
            )
    return examples


def task_helper(word: str):
    instances = []
    defns = [synset.definition() for synset in wn.synsets(word)]
    sentences = {
        tuple(
            ex.replace(
                synset.name().split(".")[0].replace("_", " "), word.replace("_", " ")
            )
            for ex in synset.examples()
        ): synset.definition()
        for synset in wn.synsets(word)
    }
    word = word.replace("_", " ")
    instances += generate_examples(word, zip(defns, sentences))
    return instances if instances else []


def process_nltk():
    with open("./data/corpora/COCA_WordFrequency.csv", "r") as fp:
        data = pd.read_csv(fp)
        data = data[data.PoS == "n"].lemma[:1000]
    with open("./data/corpora/most_common.json", "w") as fp:
        json.dump(data.to_list(), fp=fp)
    with open("./data/corpora/most_common.json", "r") as fp:
        most_common = set(json.load(fp))

    nouns = set(
        [
            n.name().split(".")[0]
            for n in list(wn.all_synsets("n"))
            if n.name().split(".")[0] in most_common and "_" not in n.name()
        ]
    )
    with open("./data/corpora/nouns.json", "w") as fp:
        json.dump(list(nouns), fp=fp)


def main():

    process_nltk()

    with open("./data/corpora/nouns.json", "r") as fp:
        nouns = json.load(fp)

    result = []
    for noun in nouns:
        examples = task_helper(noun)
        if examples:
            result += examples
    print(len(result))
    with open(f"./data/embed/embed_examples.json", "w") as fp:
        json.dump(result, fp, indent=4)


if __name__ == "__main__":
    main()
