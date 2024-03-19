import pandas as pd
import torch
import os

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter

from actions.model import getHFEmbedding


def load_data(file_path):
    dataLoader = CSVLoader(file_path=file_path)
    data = dataLoader.load()
    return data

def document_Process(data):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10, length_function=len)
    documents = text_splitter.split_documents(data)
    return documents

def create_DB(embeddingModel, docs=None):
    # db = Chroma.from_documents(docs, embeddingModel, persist_directory="./data/DB_ko-sroberta-multitask/") ## Local DB 생성
    db = Chroma(persist_directory="./data/DB_ko-sroberta-multitask/", embedding_function=embeddingModel)
    return db

def create_chain(db):
    # LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview", max_tokens=1024)
    
    # retriever
    retriever=db.as_retriever(search_kwargs={'k': 1})
    
    # prompt
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks from documents.
        Use the following pieces of retrieved context to answer the question.
        If question doesn't about the SmartStore, just say that "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.".
        the others, answer the question from documents.
        
        Context : {context}
        Qusetion : {input}
        """)

    # stuff_chain
    stuff_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    
    # Retrieval Chain
    chain = create_retrieval_chain(
        retriever,
        stuff_chain
    )

    return chain

def chat(chain, user_input):
    response = chain.invoke({"input": user_input})
    print("===" * 20)
    print(f"[HUMAN]\n{user_input}\n")
    print(f"[AI]\n{response['answer']}")

if __name__ == "__main__":
    # Device Set
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Check Model or Download
    embeddingModel = getHFEmbedding(hf_model="jhgan/ko-sroberta-multitask", device=device)
    
    # data = load_data('./data/Documents.csv')
    # document_Process(data)

    db = create_DB(embeddingModel)
    QAchain = create_chain(db)

    # Memory
    # memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1024)
    # memory = ConversationBufferMemory(memory_key="chat_history")
    # memory.load_memory_variables({})

    while True:
        human_input = input()
        if human_input != 'exit':
            chat(QAchain, human_input)