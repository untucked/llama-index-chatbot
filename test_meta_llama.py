from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM

from dotenv import load_dotenv
import os, sys
load_dotenv()
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
LLM_Directory = os.getenv("LLM_Directory")
sys.path.append(r'%s'%(LLM_Directory))

model_name = "meta-llama/Llama-2-7b-chat-hf"

# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name).encode #,  token=huggingfacehub_api_token)
#     print('Tokenizer is loaded')
#     model = AutoModelForCausalLM.from_pretrained(model_name) #,  token=huggingfacehub_api_token)
#     print("Model and tokenizer loaded successfully.")
# except Exception as e:
#     print(f"Error: {e}")

# try:
#     tokenizer = LlamaTokenizer.from_pretrained(model_name)
#     print('Tokenizer is loaded')
#     model = LlamaForCausalLM.from_pretrained(model_name)
#     print("Model and tokenizer loaded successfully.")
# except Exception as e:
#     print(f"Error: {e}")

try:
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    print('Tokenizer is loaded')
    model = HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=model_name,
        device_map="auto",
        max_new_tokens=256,
        generate_kwargs={
            "temperature": 0.7,  # Set 'do_sample=True' if using sampling
            "do_sample": True,    # Enabled sampling to utilize 'temperature'
            "top_p": 0.9           # Include 'top_p' if desired
        },
    )
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error: {e}")