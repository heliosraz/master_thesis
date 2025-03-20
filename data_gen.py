'''
import nltk
nltk.download('wordnet')
'''
from nltk.corpus import wordnet as wn
import itertools

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import json

with open("secrets.json", "r") as fp:
    secrets = json.load(fp)
    login(token=secrets["huggingface"])

class Example():
    def __init__(self, word, name, definition):
        self.word = word
        self.synset_name = name
        self.definition = definition

def format_example(example):
    prompt = (f"Instruction: Generate a word sentence that contains the word of the given definition.\n"
              f"word: {example.word}\n"
              f"definition: {example.definition}")
    return prompt

def generate_examples(words):
    result = []
    for word in words:
        for lemma in wn.lemmas(word):
            syn = lemma.synset().name()
            d = wn.synset(syn).definition()
            
            example = Example(lemma.name(), syn, d)
            prompt = format_example(example)
            examples = [pipe(prompt)[0]["generated_text"] for _ in range(5)]
            print(examples)
            result.append({'word': syn, "word.token": word, "examples": examples, "definition": d})
    return result
                
    
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
nouns = set([n.name().split(".")[0] for n in list(wn.all_synsets('n'))][:10])
# verbs = set([v.name().split(".")[0] for v in list(wn.all_synsets('v'))])
with open("data.json", "a") as data_file:
    json.dumps(generate_examples(nouns), fp=data_file, indent=4)
# generate_examples(verbs[:10])