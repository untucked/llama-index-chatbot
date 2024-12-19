# ai_support.py

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI  # Ensure this import is correct
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer

def get_embedding(emb_option='stella_lite'):
    model_kwargs = {'device': 'cpu'}

    if emb_option == 'bge-base-en':
        model_name = "BAAI/bge-base-en-v1.5"        
        embeddings = HuggingFaceEmbedding(
            model_name=model_name,
        )
    elif emb_option == 'stella':
        model_name = "blevlabs/stella_en_v5"
        embeddings = HuggingFaceEmbedding(
            model_name=model_name,
        )
    elif emb_option == 'stella_lite':
        model_name = 'dunzhang/stella_en_400M_v5'
        model_kwargs["trust_remote_code"] = True  # Add this to the model_kwargs
        embeddings = HuggingFaceEmbedding(
            model_name=model_name,
        )
    elif emb_option == 'MiniLM-L6':
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        model_kwargs["trust_remote_code"] = True  # Add this to the model_kwargs
        embeddings = HuggingFaceEmbedding(
            model_name=model_name,
        )
    elif emb_option == 'Ollama':
        embeddings = OllamaEmbedding(
            model_name="llama2",
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )
    elif emb_option == 'OpenAI':
        # https://docs.llamaindex.ai/en/stable/examples/embeddings/OpenAI/
        # model="text-embedding-3-large", "text-embedding-3-small"
        embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    else:
        raise ValueError("Embeddings option must be provided.")

    # if emb_option not in ('Ollama', 'OpenAI'):
    #     print('Applying LangchainEmbedding()')
    #     embeddings = LangchainEmbedding(embeddings)

    return embeddings

def get_LLM(llm_option='Ollama3.2', env_vars=None):
    huggingfacehub_api_token = env_vars.get("HUGGINGFACEHUB_API_TOKEN", "")
    MODEL_CONFIG = {
        'OpenAI': {'type': 'OpenAI', 'path': None},
        'meta-llama': {'type': 'MetaLlama', 'path': "meta-llama/Llama-2-7b-chat-hf"},
        'Qwen': {'type': 'HuggingFace', 'path': "Qwen/Qwen2.5-1.5B"},
        'MiniChat': {'type': 'HuggingFace', 'path': "GeneZC/MiniChat-1.5-3B"},
        'sentence-transformers': {'type': 'HuggingFace', 'path': "sentence-transformers/all-MiniLM-L6-v2"},
        'Ollama3.2': {'type': 'Ollama', 'path': 'llama3.2'},
        'Ollama3': {'type': 'Ollama', 'path': 'llama3'}
    }
    model_config = MODEL_CONFIG.get(llm_option)
    if model_config is None:
        raise ValueError(f"Unsupported chat_ai: {llm_option}")
    # Initialize the appropriate LLM based on model type
    if model_config['type'] == 'OpenAI':       
        # models: "gpt-3.5-turbo", "gpt-4o-mini"
        llm = OpenAI(model="gpt-3.5-turbo",
                     api_key=env_vars.get("OPENAI_API_KEY", ""))
    elif model_config['type'] == 'HuggingFace': 
        tokenizer = AutoTokenizer.from_pretrained(model_config['path']).encode
        set_global_tokenizer(tokenizer)
        llm = HuggingFaceLLM(model_name=model_config['path'],
                             tokenizer_name=model_config['path'] )
    elif model_config['type'] == 'MetaLlama':  
        tokenizer = AutoTokenizer.from_pretrained(model_config['path']).encode 
        set_global_tokenizer(tokenizer)
        llm = HuggingFaceLLM(
                model_name=model_config['path'],
                tokenizer_name=model_config['path'] ,
                device_map="auto",
                context_window=2048,       # Reduced from 4096 to 2048
                max_new_tokens=128,        # Reduced from 256 to 128
                max_new_tokens=256,
                generate_kwargs={
                    "temperature": 0.7,
                    "do_sample": True,      # Enabled sampling to utilize 'temperature'
                    "top_p": 0.9            # Added 'top_p' for nucleus sampling
                },        
            )
    elif model_config['type'] == 'Ollama':
        llm = Ollama(model=model_config['path'], request_timeout=360.0)
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")
    return llm
