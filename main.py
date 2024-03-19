import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
# from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from actions.embeddings import getHFEmbedding

# def document_Process(data):
#     text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10, length_function=len)
#     documents = text_splitter.split_documents(data)
#     return documents

def create_DB(embeddingModel):
    db = Chroma(persist_directory="./data/DB_ko-sroberta-multitask/", embedding_function=embeddingModel)
    return db

def create_chain(db):
    # LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview", max_tokens=1024)
    
    # retriever
    retriever=db.as_retriever(search_kwargs={'k': 1})
    
    # prompt
    system_prompt = """You are an assistant for question-answering tasks from documents.
        Use the following pieces of retrieved context to answer the question.
        If question doesn't about the SmartStore, just say that "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.".
        the others, answer the question from documents.
        
        Context : {context}
        """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ("human", "{input}")
    ])

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

def chat(chain, user_input, chat_history):
    response = chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response['answer']

if __name__ == "__main__":
    # Device Set
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Check Model or Download
    embeddingModel = getHFEmbedding(hf_model="jhgan/ko-sroberta-multitask", device=device)

    db = create_DB(embeddingModel)
    QAchain = create_chain(db)

    # Chatting History
    chat_history = []

    # --------------------------------------------------------------------------------
    # CLI
    print("FAQ Chatting입니다. 종료를 위해서는 exit를 입력해주세요.")
    while True:
        print("===" * 40)
        print()
        human_input = input("[HUMAN] : ")
        if human_input == 'exit':
            break
        else:
            chatbot_res = chat(QAchain, human_input, chat_history)
            chat_history.append(HumanMessage(content=human_input))
            chat_history.append(AIMessage(content=chatbot_res))
            print(f"[FAQ Bot] : {chatbot_res}")
            print()