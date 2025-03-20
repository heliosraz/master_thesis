from transformers import AutoTokenizer, AutoModelForCausalLM
import setup

def format_example(example):
    prompt = (f"Question: What is a sentence that where word1 and word2 could go together?"
              f"word1: {example.word1}"
              f"word2: {example.word2}")
    return prompt
    
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")



