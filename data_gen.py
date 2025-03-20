'''
import nltk
nltk.download('wordnet')
'''
from nltk.corpus import wordnet as wn
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import setup

class Example():
    def __init__(self, word, name, definition, sentence):
        self.word = word
        self.synset_name = name
        self.definition = definition
        self.sentence = sentence

def format_example(example):
    prompt = (f"Instruction: Generate a word sentence that contains the word of the given definition.\n"
              f"word: {example.word}\n"
              f"definition: {example.definition}\n"
              f"An example of this is: {example.sentence}")
    return prompt

def generate_examples(words):
    result = []
    for word in words:
        for lemma in wn.lemmas(word):
            syn = lemma.synset().name()
            defn = wn.synset(syn).definition()
            sentence = wn.synset(syn).examples()
            
            example = Example(lemma.name(), syn, defn, sentence)
            prompt = format_example(example)
            examples = [pipe(prompt)[0]["generated_text"] for _ in range(5)]
            print(examples)
            result.append({'word': syn, "word.token": word, "examples": examples, "definition": defn})
    return result
                
    
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
nouns = set([n.name().split(".")[0] for n in list(wn.all_synsets('n'))][:10])
# verbs = set([v.name().split(".")[0] for v in list(wn.all_synsets('v'))])
with open("data.json", "a") as data_file:
    json.dumps(generate_examples(nouns), fp=data_file, indent=4)
# generate_examples(verbs[:10])