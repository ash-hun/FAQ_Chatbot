import pandas as pd
import torch
import os

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    documents = text_splitter.split_documents(data)

    # db = Chroma.from_documents(documents, embeddingModel, persist_directory="./data/DB/")
    db = Chroma(persist_directory="./data/DB/", embedding_function=embeddingModel)

    query = "미성년자도 판매 회원 등록이 가능한가요?"
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    response = qa_chain.invoke({"query": query})
    print(response)