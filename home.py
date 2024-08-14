import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from datetime import datetime
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import logging

#huggingface 로그인
from huggingface_hub import login

login()



#환경설정
ENV_PATH = './.env'
CACHE_DIR = './.cache'
FILES_DIR = os.path.join(CACHE_DIR, 'files')
EMBEDDINGS_DIR = os.path.join(CACHE_DIR, 'embeddings')
UPLOADS_DIR = "./uploads"
CACHE_DIR = "./cache"

# Ensure directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


#I. Streamlit 관련
st.title('RAG 시스템')

# 모델 선택 및 API Key 입력(필요시)
st.subheader('모델 및 파라미터 설정')
selected_model = st.sidebar.selectbox('LLM 모델을 선택하세요', ['Openai-GPT-4o', 'Google-Gemma-2-9b', 'Meta-Llama-3.1-8b'], key='selected_model')
if selected_model == 'Openai-GPT-4o':
    with st.sidebar:
        openai_api_key = st.text_input('Openai API Key를 입력해주세요')


#II. Langchain 관련
#Langsmith 설정
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "사내 LLM 구축 프로젝트"



#Prompt 생성
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

prompt =  ChatPromptTemplate.from_messages(
  [
    ("system", """당신은 느린 학습자 관련 교육 전문가입니다. 당신은 느린 학습자 관련 질문을 받고, 이에 대한 전문적인 답변을 제공합니다.
             아래 컨텍스트를 사용하여 질문에 답변하십시오. 답을 모르면 "모르겠습니다."라고 답변하세요. 답을 지어내지 마세요.
        컨텍스트: {context}"""),
    ("ai", "안녕하세요! 느린 학습자와 관련해서 어떤 사항이 궁금하신가요?"),
    ("human", "{question}"),
  ]
)


# Initialize the text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# Initialize the embeddings model
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

@st.cache_data(show_spinner="파일 로딩 중....")
def process_and_embed_file(file):
    # Save the uploaded file to disk
    file_path = os.path.join(UPLOADS_DIR, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.read())

    # Load and split the document
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split(text_splitter=text_splitter)

    # Embed the split documents
    embeddings = embeddings_model.embed_documents([doc.page_content for doc in documents])

    # Create FAISS index
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    index = faiss.IndexFlatL2(len(embeddings[0]))

    # Wrap FAISS in Langchain's FAISS vectorstore
    vector_store = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Add documents and embeddings to vector store
    vector_store.add_documents(documents=documents)

    # Save the FAISS index locally
    cache_path = os.path.join(CACHE_DIR, file.name.split('.')[0])
    vector_store.save_local(cache_path)

    return vector_store

def format_documents(docs):
    # Debugging: Log document content and type
    for doc in docs:
        print(f"Document Content: {doc.page_content}")
        
    return "\n\n".join([doc.page_content for doc in docs])
  
def load_vectorstore_from_cache(filename):
    # Load an existing FAISS vectorstore
    cache_path = os.path.join(CACHE_DIR, filename)
    vector_store = FAISS.load_local(cache_path, embeddings_model, allow_dangerous_deserialization=True)
    return vector_store


# Streamlit UI
with st.sidebar:
    st.title("PDF to FAISS Embedding")
    file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if file:
        vector_store = process_and_embed_file(file)
        st.success(f"File '{file.name}' has been processed and embedded successfully!")
    else:
        # Load an existing vectorstore if no file is uploaded
        if os.path.exists(os.path.join(CACHE_DIR, "test")):
            vector_store = load_vectorstore_from_cache("test")
            st.info("Loaded existing vectorstore from cache.")
        else:
            st.warning("No file uploaded and no existing vectorstore found.")
            vector_store = None

#LLM model

## Openai-GPT-4o
if selected_model == 'Openai-GPT-4o' and openai_api_key:    
    llm = ChatOpenAI(
      model="gpt-4o", 
      api_key=openai_api_key,
      verbose=True,
      max_tokens= 1500,
      )
    
    set_llm_cache(InMemoryCache()) 
    
    conversation = ConversationChain(
      llm=llm,
      memory=ConversationBufferWindowMemory(k=3),
      verbose=True
    )
    
            # User input handling
    if user_input := st.chat_input("질문을 입력하세요"):
        # Initialize retriever for the FAISS vectorstore
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
        
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(user_input)
        
        # Debugging: Check retrieved documents
        for doc in retrieved_docs:
            print(f"Retrieved Document Content: {doc.page_content}")
            
        # Format the retrieved documents as context
        context = format_documents(retrieved_docs)

        # Use the prompt template to prepare the LLM input
        inputs = {
            "context": context,
            "question": user_input
        }
        
        # Prepare the chain using prompt and LLM
        chain = prompt | llm
        
        # Invoke the chain with formatted input
        response = chain.invoke(inputs)
        st.write(response)

## Google-Gemma-2-9b
if selected_model == 'Google-Gemma-2-9b':
  #load the model using huggingface 
  from langchain_huggingface import HuggingFacePipeline
  llm = HuggingFacePipeline.from_model_id(
    model_id = "google/gemma-2-9b",
    task = "text-generation",
  )
  
  set_llm_cache(InMemoryCache())
  
  conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=3),
    verbose=True
  )
  
  # User input handling
  if user_input := st.text_input("질문을 입력하세요"):
    # Initialize retriever for the FAISS vectorstore
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2}) #TODO: similarity(default)/mmr/similarity_score_threshold 중 어느 것을 사용할지 테스팅 필요
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(user_input)
    
    # Debugging: Check retrieved documents
    for doc in retrieved_docs:
        print(f"Retrieved Document Content: {doc.page_content}")
        
    # Format the retrieved documents as context
    context = format_documents(retrieved_docs)

    # Use the prompt template to prepare the LLM input
    inputs = {
        "context": context,
        "question": user_input
    }
    
    # Prepare the chain using prompt and LLM
    chain = prompt | llm
    
    # Invoke the chain with formatted input
    response = chain.invoke(inputs)
    st.write(response)