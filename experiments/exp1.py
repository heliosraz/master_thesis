from transformers import AutoTokenizer, AutoModelForCausalLM
import setup

llms = ["meta-llama/Llama-3.3-1B", "DeepSeek_r1", "Gemma", "Mistral"] #distill version
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")



