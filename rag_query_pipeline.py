from dotenv import load_dotenv
from typing import List
import os
import pathlib
import argparse

from  arxiv_utils import download_papers

from llama_index.storage import StorageContext
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.text_splitter import SentenceSplitter

from llama_index.query_pipeline.query import QueryPipeline
from llama_index.llms import OpenAI

from llama_index.postprocessor import CohereRerank
from llama_index.response_synthesizers import TreeSummarize
from llama_index import ServiceContext
from llama_index.query_pipeline import InputComponent
from llama_index.embeddings import OpenAIEmbedding
from llama_index.node_parser import (
    SentenceWindowNodeParser,
)

def load_data(path: str):
    """Load data from directory

    Returns:
        list(Documents): list of documents loaded from directory
    """
    reader = SimpleDirectoryReader(input_dir=path, recursive=True)
    docs = reader.load_data()
    return docs

def create_or_rebuild_index_SentenceWindow(docs: List, storage_path: str = "storage", window_size: int = 3, chunk_size: int=256, chunk_overlap: int = 0):

    pathlib.Path().resolve()
    if not os.path.exists(storage_path):
        # Use a Sentence Splitter to split the text into chunks    
        text_splitter = SentenceSplitter(chunk_size= chunk_size, chunk_overlap=chunk_overlap)
        # create the sentence window node parser w/ default settings
        node_parser = SentenceWindowNodeParser.from_defaults(
            #sentence_splitter= text_splitter,
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
            include_metadata= False
        )        
        # Define a context
        print("Defining the Service Context")
        service_context = ServiceContext.from_defaults(
                embed_model=OpenAIEmbedding(
                            model="text-embedding-3-small",
                ),
                # embed_model="local:BAAI/bge-large-en-v1.5"
                node_parser=node_parser)
        # NO CHANGES
        # Build the index acording to the service
        print("Creating the index")
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        # save index to disk
        index.set_index_id("vector_index")
        print("Saving the index")
        index.storage_context.persist(os.path.join(pathlib.Path().resolve(), storage_path))
    else:
        print("Restoring the storage")
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        # load index
        print("Loading the index")
        index = load_index_from_storage(storage_context, index_id="vector_index")    
    
    return index

def create_or_rebuild_index(docs: List, storage_path: str = "storage", chunk_size: int=1024, chunk_overlap: int = 50):

    pathlib.Path().resolve()
    if not os.path.exists(storage_path):
        # Use a Sentence Splitter to split the text into chunks    
        text_splitter = SentenceSplitter(chunk_size= chunk_size, chunk_overlap=chunk_overlap)
        # Define a context
        print("Defining the Service Context")
        service_context = ServiceContext.from_defaults(text_splitter=text_splitter)
        # Build the index acording to the service
        print("Creating the index")
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        # save index to disk
        index.set_index_id("vector_index")
        print("Saving the index")
        index.storage_context.persist(os.path.join(pathlib.Path().resolve(), storage_path))
    else:
        print("Restoring the storage")
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        # load index
        print("Loading the index")
        index = load_index_from_storage(storage_context, index_id="vector_index")    
    
    return index

def RAG_pipeline(index, top_k: int = 3, model_name: str = "gpt-3.5-turbo", temperature: float = 0.3, max_tokens: int = 512):

    # Set the retriever
    retriever = index.as_retriever(similarity_top_k=top_k)
    # Define the summarizer
    summarizer = TreeSummarize(
        service_context=ServiceContext.from_defaults(
            llm=OpenAI(model=model_name,
                       temperature=temperature,
                       max_tokens=max_tokens)
        )
    )
    # Define the reranker
    reranker = CohereRerank()
    # Define the query pipeline
    p = QueryPipeline(verbose=True)
    p.add_modules(
        {
            "input": InputComponent(),
            "retriever": retriever,
            "reranker": reranker,
            "summarizer": summarizer,
        }
    )
    # Set the links between components
    p.add_link("input", "retriever")
    p.add_link("input", "reranker", dest_key="query_str")
    p.add_link("retriever", "reranker", dest_key="nodes")
    p.add_link("input", "summarizer", dest_key="query_str")
    p.add_link("reranker", "summarizer", dest_key="nodes")
    
    return p
   
def args_parser():
    parser = argparse.ArgumentParser(description='Search for information in Arxiv.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to return')
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo", help='Model name to use')
    parser.add_argument('--n_docs', type=int, default=3, help='Papers to download')
    parser.add_argument('--query', type=str, help='Query to solve')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Set the data path
    path="../data"
    # Read the arguments
    args=args_parser()
    # Set the OpenAI API key
    # Load the enviroment variables
    load_dotenv("../.env")    
    # Get the papers from Arxiv
    metadata= download_papers(args.query, args.n_docs, path)
    
    # Load the data
    docs = load_data(path)
    print("Docs loaded: ", len(docs))
    # Create or rebuild the index
    index = create_or_rebuild_index(docs)
    # Define the RAG pipeline
    pipeline = RAG_pipeline(index, args.top_k, args.model_name)
    # Run the pipeline
    # Test query: "what did the author do in YC"
    output = pipeline.run(input=args.query)
    print(type(output))
    print(output.response)
    print(output.get_formatted_sources(3))
   
