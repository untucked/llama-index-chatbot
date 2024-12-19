from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
from dotenv import load_dotenv
import os
from llama_index.core import set_global_tokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
LLM_Directory = os.getenv("LLM_Directory")
sys.path.append(r'%s'%(LLM_Directory))

# Define your model configuration
model_config = {
    'path': "Qwen/Qwen2.5-1.5B"
}

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_config['path'],
    # token=huggingfacehub_api_token
).encode

# Set the global tokenizer
set_global_tokenizer(tokenizer)

# Initialize the HuggingFaceLLM
# llm = HuggingFaceLLM(
#     model_name=model_config['path'],
#     tokenizer_name=model_config['path'],
#     context_window=4096,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7, "do_sample": False},
#     device_map="auto")
llm = HuggingFaceLLM(
    model_name=model_config['path'],
    tokenizer_name=model_config['path'])

print("Tokenizer and LLM initialized successfully.")
