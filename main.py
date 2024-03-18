import pandas as pd
import torch
import os

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
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
    embeddingModel = getHFEmbedding(hf_model="all-MiniLM-L6-v2", device=device)

    # LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview", max_tokens=1024)

    # text splitter
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    documents = text_splitter.split_documents(data)
    # print(f" ► Document Example : {documents[0]}") # log

    db = Chroma.from_documents(documents, embeddingModel, persist_directory="./data/DB/")

    query = "숏클립 복구 가능해요?"
    docs = db.similarity_search(query)
    print(docs[0].page_content)