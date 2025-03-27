from typing import Dict, Tuple, Iterable
from nltk.corpus import wordnet as wn
import json
import os
import itertools
import pandas as pd

print(os.getcwd())

class DataInstance():
    def __init__(self, word, definition, example):
        self.word = word
        self.definition = definition
        self.example = example

def format_example(example:DataInstance, task:int):
    if task == 1:
        prompt = [(f"Question: How well does \"{example.definition}\" describe \"{example.word}\" in this sentence? \n"
                f"Sentence: {example.example}"),
                "Question: Can you provide justification for your answer?"]
        # ranking the senses for this word
            # how well does this definition fit the word in this sentence?
    elif task == 2:
        prompt = [(f"Instruction: Give a definition that matches \"{example.word}\" in \"{example.example}\"?")]
    elif task == 3:
        prompt = [(f"Question: What words in \"{example.definition}\" are the key words in defining \"{example.word}\""),(f"Question: How do each of these keywords contribute to the definition?")]
    elif task == 4:
        prompt = [(f"Instruction: Replace \"{example.word}\" with a word that matches the meaning the closest in the sentence: {example.example}")]
    elif task == 5:
        prompt = [(f"Question: A dog means an informal term for a man. An example of the sentence is:\nyou lucky dog.\n"
            f"A {example.word} means {example.definition}. An example of the sentence is: ")]
    return prompt

def generate_examples(word:str, pairs: Iterable[Tuple[str, str]], task:int, gold: Dict[tuple, str]=None):
    examples = []
    for defn, sentence in pairs:
        filter_sentence = [s for s in sentence if word in s]
        if defn and filter_sentence:
            example = DataInstance(word, defn, filter_sentence[0])
            prompt = format_example(example, task)
            if gold:
                examples.append({'word': word, 'definition': defn, 'gold': gold[sentence], 'sentence': sentence[0], 'prompt': prompt})
            else: examples.append({'word': word, 'definition': defn , 'sentence': sentence[0], 'prompt': prompt})
    return examples

def task_helper(word:str, task:int):
    instances = []
    defns = [synset.definition() for synset in wn.synsets(word)]
    sentences = {tuple(ex.replace(synset.name().split(".")[0].replace("_", " "), word.replace("_", " ")) 
                    for ex in synset.examples()):synset.definition() 
                    for synset in wn.synsets(word)}
    word = word.replace("_", " ")
    # Task 1:
    if task == 1:
        instances += generate_examples(word, itertools.product(defns, sentences), task, gold = sentences)
    # Task 2:
    elif task == 2:
        instances += generate_examples(word, zip(defns, sentences), task, gold = sentences)
    # Task 3:
    elif task == 3:
        instances += generate_examples(word, zip(defns, sentences), task)
    # Task 4:
    elif task == 4:
        instances += generate_examples(word, zip(defns, sentences), task)
    return instances if instances else []

if __name__ == "__main__":
    with open("./data/corpora/COCA_WordFrequency.csv",'r') as fp:
        data = pd.read_csv(fp)
        data = data[data.PoS == "n"].lemma[:1000]
    with open("./data/corpora/most_common.json", "w") as fp:
        json.dump(data.to_list(), fp=fp)
    with open("./data/corpora/most_common.json", "r") as fp:
        most_common = set(json.load(fp))
    tasks = [1,2,3,4]
    nouns = set([n.name().split(".")[0] for n in list(wn.all_synsets('n')) 
                    if n.name().split(".")[0] in most_common and "_" not in n.name()])
    with open("./data/corpora/nouns.txt", "w") as fp:
        print(nouns, file=fp)
    for task in tasks:
        result = []
        for noun in nouns:
            examples = task_helper(noun, task)
            if examples:
                result += examples
        print(len(result))
        with open(f"./data/tasks/task{task}.json", "w") as fp:
            json.dump(result, fp, indent=4)
