# main.py
# streamlit run main.py
import streamlit as st
import sys
from dotenv import load_dotenv
import os
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
import shutil
# Local imports
# Load environment variables
load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
LLM_Directory = os.getenv("LLM_Directory")
sys.path.append(r'%s'%(LLM_Directory))
from ai_support import *
from htmlTemplates import *
from chat_support import *

# Disable Hugging Face symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MAX_HISTORY = 10  # Adjust this number as needed
DEBUG = False

# Define embedding and LLM options
embedding_options = ['bge-base-en', 'stella', 'stella_lite', 'MiniLM-L6', 'OpenAI']
llm_options = ['OpenAI', 'meta-llama', 'Qwen', 'MiniChat',\
                'sentence-transformers', 'Ollama3.2', 'Ollama3']

# Initialize session state variables
if 'chat_engine' not in st.session_state:
    st.session_state.chat_engine = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

st.title("📄 Document-Based Chatbot")
st.sidebar.header("Configuration")

# Directory input
directory = st.sidebar.text_input(
    "Enter the directory path containing your documents:",
    value='./data'
)

STORAGE_DIR = os.path.join(directory, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)
config_path = os.path.join(STORAGE_DIR, "config.json")

##############################
# Create config.json
############################
# Initialize session state variables for embedding and LLM options
if not os.path.exists(config_path):
    # Default values if the config file doesn't exist
    st.session_state.selected_emb_option = 'MiniLM-L6'
    st.session_state.selected_llm_option = 'Ollama3.2'
    # Create default config file
    with open(config_path, "w") as f:
        json.dump({
            "embedding_option": st.session_state.selected_emb_option,
            "llm_option": st.session_state.selected_llm_option
        }, f)
    print('Reading default embedding option: %s and LLM: %s'%(st.session_state.selected_emb_option, st.session_state.selected_llm_option))
else:
    # Load options from the config file
    with open(config_path, "r") as f:
        config = json.load(f)
        st.session_state.selected_emb_option = config.get("embedding_option")
        st.session_state.selected_llm_option = config.get("llm_option")
    print('Reading previous embedding option: %s and LLM: %s'%(st.session_state.selected_emb_option, st.session_state.selected_llm_option))

##############################
# Integrating Chat Memory
##############################

# Attempt to load a persisted chat store if it exists:
chat_store_path = os.path.join(STORAGE_DIR, "chat_store.json")
if os.path.exists(chat_store_path):
    try:
        chat_store = SimpleChatStore.from_persist_path(persist_path=chat_store_path)
        st.success("Chat store loaded successfully.")
    except Exception as e:
        print('Chat JSON exists but is empty or the wrong format')
        chat_store = SimpleChatStore()
        chat_store.persist(persist_path=chat_store_path)
else:
    print('Chat JSON does not exist')
    chat_store = SimpleChatStore()
    chat_store.persist(persist_path=chat_store_path)

# Create the chat memory buffer
# The chat_store_key can uniquely identify the user or session
# if 'chat_memory' not in st.session_state or st.session_state.chat_memory is None:
st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)

# Dropdown for Embedding Model Selection
emb_option = st.sidebar.selectbox(
    "Select Embedding Model:",
    options=embedding_options,
    index=embedding_options.index(st.session_state.selected_emb_option)
)

# Dropdown for LLM Model Selection
llm_option = st.sidebar.selectbox(
    "Select LLM Model:",
    options=llm_options,
    index=llm_options.index(st.session_state.selected_llm_option)
)

# Buttons in the sidebar
load_docs = st.sidebar.button("Load Documents")
clear_history = st.sidebar.button("Clear Chat History")
clear_index = st.sidebar.button("Clear Index and Chat History")

if clear_history:
    chat_store = SimpleChatStore()
    st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user1",
    )
    chat_store.persist(persist_path=chat_store_path)
    st.success("Chat history cleared.")

if clear_index:
    # Step 1: Clear the existing index by deleting the storage directory
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
        st.success("Index cleared successfully.")

    # Step 2: Clear the chat history
    chat_store = SimpleChatStore()
    st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user1",
    )
    chat_store.persist(persist_path=chat_store_path)
    st.success("Chat history cleared successfully.")

    # Step 3: Reset session state variables
    st.session_state.index = None
    st.session_state.chat_engine = None
    st.session_state.documents_loaded = False
    st.rerun()


if emb_option != st.session_state.selected_emb_option or llm_option != st.session_state.selected_llm_option:
    st.warning("Embedding or LLM option changed. Resetting index and chat history...")

    # Step 1: Clear the existing index by deleting the storage directory
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
        st.success("Index cleared successfully.")

    # Step 2: Clear the chat history
    chat_store = SimpleChatStore()
    st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user1",
    )
    chat_store.persist(persist_path=chat_store_path)
    st.success("Chat history cleared successfully.")

    # Step 3: Reset session state variables
    st.session_state.index = None
    st.session_state.chat_engine = None
    st.session_state.documents_loaded = False

    # Step 4: Update session state with new options
    st.session_state.selected_emb_option = emb_option
    st.session_state.selected_llm_option = llm_option

    # Step 5: Save new options to config.json
    with open(config_path, "w") as f:
        json.dump({
            "embedding_option": emb_option,
            "llm_option": llm_option
        }, f)
    st.success("New configuration saved successfully.")
    st.rerun()


if load_docs:
    with st.spinner("Loading documents and setting up the index..."):
        try:
            load_user_documents(directory, emb_option, llm_option, STORAGE_DIR, DEBUG=DEBUG)
            if st.session_state.index is not None:
                # Ensure chat_engine is initialized with chat_memory
                st.session_state.chat_engine = st.session_state.index.as_chat_engine(
                    memory=st.session_state.chat_memory
                )
                st.success("Chat engine initialized with chat_memory.")
        except Exception as e:
            st.error(f"Error loading documents: {e}")
        

if st.session_state.documents_loaded:
    st.header("💬 Chat with Your Documents")
    # Inject CSS for chat styling
    st.markdown(css, unsafe_allow_html=True)
    # Use a form to handle user input and submission
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", "")
        submit_button = st.form_submit_button(label="Send")
    if submit_button and user_input:
        with st.spinner("Generating response..."):
            try:
                conversation_html(user_input, chat_store_path, DEBUG=DEBUG)
            except Exception as e:
                st.error(f"Error generating response: {e}")
    update_html()
else:
    st.info("Please load your documents to start chatting.")

