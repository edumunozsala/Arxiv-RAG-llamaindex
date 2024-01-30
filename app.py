import streamlit as st
from streamlit_chat import message
#from PIL import Image
#from utils.load_config import LoadConfig
#from utils.app_utils import load_data, RAG, delete_data
from arxiv_utils import delete_papers, download_papers
from rag_query_pipeline import load_data, create_or_rebuild_index, RAG_pipeline

#import subprocess
import os
import yaml
from dotenv import load_dotenv

def load_config():
    """
    Load configuration from config.yaml file
    """
    # Load the secrets in the .env file
    load_dotenv("../.env")    
    
    with open("config.yml") as f:
        app_config = yaml.load(f, Loader=yaml.FullLoader)
        
    return app_config

config = load_config()

# ===================================
# Setting page title and header
# ===================================
#im = Image.open("images/maestro.png")
#os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

st.set_page_config(page_title="RAG Arxiv Papers", layout="wide") # page_icon=im
st.markdown(
    "<h1 style='text-align: center;'>RAG Arxiv Papers </h1>",
    unsafe_allow_html=True,
)
st.divider()
st.markdown(
        "<center><i>RAG Arxiv Papers is an up-to-date LLM assistant designed to provide clear and concise explanations of scientific concepts <b>and relevant papers</b>. As a Q&A bot, it does not keep track of your conversation and will treat each input independently.  Do not hesitate to clear the conversation once in a while! Hoping that RAG-Maestro will help get quick answers and expand your scientific knowledge.</center>",
        unsafe_allow_html=True,
    )
st.divider()

# ===================================
# Initialise session state variables
# ===================================
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# ==================================
# Sidebar:
# ==================================
counter_placeholder = st.sidebar.empty()
with st.sidebar:
    st.markdown(
        "<h3 style='text-align: center;'>Ask anything you need to brush up on!</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><b>Example: </b></center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><i>What is GPT4?</i></center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><i>Explain me Mixture of Models (MoE)</i></center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><i>How does RAG works?</i></center>",
        unsafe_allow_html=True,
    )
    # st.sidebar.title("An agent that read and summarizethe the news for you")
    #st.sidebar.image("images/maestro.png", use_column_width=True)
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    #st.markdown(
    #"<a style='display: block; text-align: center;' href='https://aymenkallala.github.io/' target='_blank'> Aymen Kallala</a>",
    #unsafe_allow_html=True,
    #)
    
# ==================================
# Reset everything (Clear button)
if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    delete_papers(config["data_path"])

response_container = st.container()  # container for message display

if query := st.chat_input(
    "What do you need to know? I will explain it and point you out interesting readings."
):
    st.session_state["past"].append(query)
    try:
        with st.spinner("Browsing the best papers..."):
            # Search and doenload the relevant papers based on the query
            metadata = download_papers(query, config["articles_to_search"], config["data_path"])

        with st.spinner("Reading them..."):
            #data = load_data()
            print("Reading the papers")
            docs = load_data(config["data_path"])
            #index = RAG(APPCFG, _docs=data)
            # Create or rebuild the index
            index = create_or_rebuild_index(docs, chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"])
            
            #query_engine = index.as_query_engine(
            #    response_mode="tree_summarize",
            #    verbose=True,
            #    similarity_top_k=APPCFG.similarity_top_k,
            #)
            # Define the RAG pipeline
            print("Defining the RAG pipeline")
            pipeline = RAG_pipeline(index, config["similarity_top_k"], config["gpt_model"])
            
        with st.spinner("Thinking..."):
            #response = query_engine.query(query + APPCFG.llm_format_output)
            # Run the pipeline
            output = pipeline.run(input=query)


        st.session_state["generated"].append(output.response)
        del index
        del pipeline

        with response_container:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True)

                message(st.session_state["generated"][i], is_user=False)

    except Exception as e:
        print(e)
        st.session_state["generated"].append(
            "An error occured with the paper search, please modify your query."
        )