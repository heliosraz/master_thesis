from nltk.corpus import wordnet as wn
import json
import os
import itertools

print(os.getcwd())

class DataInstance():
    def __init__(self, word, definition, example, choices=[]):
        self.word = word
        self.definition = definition
        self.example = example
        self.choices = choices

def format_example(example:DataInstance, task:int):
    if task == 1:
        prompt = (f"How well does \"{example.definition}\" describe {example.word} in this sentence? \n"
                f"Sentence: {example.example}")
        # why does it fit?
        # ask about the different words in the definition contribute to this word sense?
        # ranking the senses for this word
            # how well does this definition fit the word in this sentence?
    elif task == 2:
        prompt = (f"Which definition matches {example.word} in \"{example.example}\"?\n")
        for i, choice in enumerate(example.choices):
            prompt += f"{i}. {choice}\n"
            
    elif task == 3:
        prompt = 
    return prompt

def generate_examples(word:str, task:int):
    instances = []
    defns = [synset.definition() for synset in wn.synsets(word)]
    sentences = [[ex.replace(synset.name().split(".")[0].replace("_", " "), word.replace("_", " ")) for ex in synset.examples() if word.replace("_", " ") in ex.replace(synset.name().split(".")[0].replace("_", " "), word.replace("_", " "))] for synset in wn.synsets(word)]
    # sentences = [synset.examples() for synset in wn.synsets(word)]
    word = word.replace("_", " ")
    # Task 1:
    if task ==1:
        for defn, sentence in itertools.product(defns, sentences):
            if defn and sentence:
                example = DataInstance(word, defn, sentence[0])
                prompt = format_example(example, task)
                instances.append({'word': word, 'definition': defn, 'sentence': sentence[0], 'prompt': prompt})
    # Task 2:
    elif task == 2:
        for defn, sentence in zip(defns, sentences):
            if defn and sentence:
                example = DataInstance(word, defn, sentence[0], defns)
                prompt = format_example(example, task)
                instances.append({'word': word, 'definition': defn, 'sentence': sentence[0], 'prompt': prompt, 'choices': defns})
    # Task 3:
    elif task == 3:
        pass
    return instances if instances else []

if __name__ == "__main__":
    task = 1
    nouns = set([n.name().split(".")[0] for n in list(wn.all_synsets('n'))])
    result = []
    for noun in nouns:
        examples = generate_examples(noun, task)
        if examples:
            result += examples
    print(len(result))
    with open(f"./data/task{task}.json", "w") as fp:
        json.dump(result, fp, indent=4)
