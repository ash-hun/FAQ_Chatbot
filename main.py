import pandas as pd
import torch
import os

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import chroma
from langchain.chains import RetrievalQA

from actions.model import getHFEmbedding


def load_data(file_path):
    dataLoader = CSVLoader(file_path=file_path)
    data = dataLoader.load()
    return data


if __name__ == "__main__":
    # Device Set
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Data Load
    data = load_data('./data/Documents.csv')
    
    # Check Model or Download
    getHFEmbedding(hf_model="BAAI/bge-large-en-v1.5", device=device)

    # LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview", max_tokens=1024)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    all_splits = text_splitter.split_documents(data)
    print(all_splits[0])
