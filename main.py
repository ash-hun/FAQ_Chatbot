import pandas as pd
import torch
import os

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

from actions.model import getHFEmbedding


def load_data(file_path):
    dataLoader = CSVLoader(file_path=file_path)
    data = dataLoader.load()
    return data

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    # Device Set
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Data Load
    data = load_data('./data/Documents.csv')
    
    # Check Model or Download
    embeddingModel = getHFEmbedding(hf_model="jhgan/ko-sroberta-multitask", device=device)

    # LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview", max_tokens=1024)

    # text splitter
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10, length_function=len)
    documents = text_splitter.split_documents(data)

    # db = Chroma.from_documents(data, embeddingModel, persist_directory="./data/DB_ko-sroberta-multitask/") ## Local DB 생성
    db = Chroma(persist_directory="./data/DB_ko-sroberta-multitask/", embedding_function=embeddingModel)

    # retriever
    retriever=db.as_retriever()

    # prompt
    prompt = hub.pull("rlm/rag-prompt")

    query = "미성년자도 판매 회원 등록이 가능한가요?"

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(query)

    print("===" * 20)
    print(f"[HUMAN]\n{query}\n")
    print(f"[AI]\n{response}")