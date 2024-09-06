import streamlit as st
from langchain.prompts import PromptTemplate
from datetime import datetime
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import logging
from langchain_community.document_loaders import PyPDFLoader
import uuid
from datetime import datetime
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage #추후 Gemma2용 프롬프트 작성에 테스트
from langchain.retrievers import BM25Retriever, EnsembleRetriever


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


#I. Streamlit 세션 관리
# Step 1: Generate a unique identifier for each user
if 'user_id' not in st.session_state:
  user_id = str(uuid.uuid4())
  st.session_state['user_id'] = user_id
else:
  user_id = st.session_state['user_id']
  
# Step 2: Initialize chat history and timestamp for the user if they don't exist
if 'chat_history' not in st.session_state:
  st.session_state['chat_history'] = {}

if 'last_activity' not in st.session_state:
  st.session_state['last_activity'] = {}

if user_id not in st.session_state['chat_history']:
  st.session_state['chat_history'][user_id] = []

if user_id not in st.session_state['last_activity']:
  st.session_state['last_activity'][user_id] = datetime.now()  # Initialize timestamp
  
#TODO : 주기적 세션 정리 필요

# Step 4: Functions to save and send messages
def save_message(message, role):
  st.session_state['chat_history'][user_id].append((role, message))
  st.session_state['last_activity'][user_id] = datetime.now()  # 메시지 보낼 경우, 마지막 활동 시간 업데이트
  
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if save:
            save_message(message, role)

def paint_history():
  for message in st.session_state['chat_history'][user_id]:
    send_message(message[1], message[0], save=False)
      
st.title('RAG 시스템')

# 모델 선택 및 API Key 입력(필요시)
st.subheader('느린학습자 관련 RAG')
selected_model = st.sidebar.selectbox('LLM 모델을 선택하세요', ['Openai-GPT-4o', 'Google-Gemma-2' ], key='selected_model')
if selected_model == 'Openai-GPT-4o':
    with st.sidebar:
        openai_api_key = st.text_input('Openai API Key를 입력해주세요')


#II. Langchain 관련
#Langsmith 설정
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "사내 LLM 구축 프로젝트"


#Prompt 생성
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
#TODO: 사용자의 Input을 받도록 유도하여 보고서를 작성하는 Prompt 작성 필요 -> 이 부분은 LangGraph 활용
prompt =  ChatPromptTemplate.from_messages( #TODO : 추후 퓨샷 템플릿으로 변경
  [
    ("system", """당신은 느린 학습자 관련 교육 전문가입니다. 당신은 느린 학습자 관련 질문을 받고, 이에 대한 전문적인 답변을 제공합니다.
             아래 컨텍스트를 참고하여 질문에 답변하십시오. 답을 모르면 "모르겠습니다. 느린 학습자에 관한 질문만 답변해주세요"라고 답변하세요. 답을 지어내지 마세요.
             또한 답변을 제공할 때에는 출처(source)를 함께 제공해주세요. 이를 통해 사용자가 답변의 신뢰성을 확인할 수 있습니다.
             
             사용자가 아동의 행동패턴을 제공하고 느린학습자에 해당하는지 질문한 경우에는, 해당 행동패턴을 언어, 기억력, 지각, 집중, 처리속도 특성으로 분류하여 답변해주세요.
             마지막은 '종합 판단 및 권장 사항'으로 분류하고, 보다 정확한 진단을 위해선 전문적인 검사 혹은 상담이 필요하다고 안내해주세요. 어떠한 검사 혹은 상담이 필요한지도 간략히 안내해주세요.
             
        컨텍스트: {context}"""),
    MessagesPlaceholder(variable_name="messages"), #채팅 히스토리를 컨텍스트로 넘겨주기 위해 사용
    ("ai", "안녕하세요, 느린 학습자에 대해 어떤 사항이 궁금하신가요?"),
    ("human", "{question}"),
  ]
)

# Initialize the text splitter
#TODO: 추후 보유 문서에 맞는 Chunking/Splitting 전략 구현 필요
from langchain_text_splitters import RecursiveCharacterTextSplitter 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
)

# Initialize the embeddings model
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    # model_kwargs={'device': 'cuda'},  #GPU를 사용한 임베딩 옵션. 현재 자원 제한으로 CPU 사용
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
    # multi_process=True, #GPU 사용시 멀티 프로세스 사용
    show_progress=True,
)


#OutputParer
#TODO : PydanticOutputParser를 사용한 출력 형식 지정 -> 추후 보고서 등 명확한 출력 형식이 필요한 경우 사용.
# from langchain.schema import BaseOutputParser
# class NewLineOutputParser(BaseOutputParser): 
#     def parse(self, output):
#         #'\n' -> '  \n'으로 변환
#         return output.replace('\n', '  \n')
# parser = NewLineOutputParser()

from langchain.schema import BaseOutputParser
import re

class LastPartOutputParser(BaseOutputParser):
    def parse(self, output: str) -> str:
        # Split the output by 'answer:' to find the last relevant part
        parts = output.split("answer:")
        if len(parts) > 1:
            # The last part is the valid answer
            final_answer = parts[-1].strip()
        else:
            # If there is no 'answer:', return the whole output
            final_answer = output.strip()

        # Replace '\n' with '  \n' in the final answer
        formatted_answer = final_answer.replace('\n', '  \n')

        # Use regex to find and replace the metadata pattern
        formatted_answer = self._format_metadata(formatted_answer)
        
        return formatted_answer

    def _format_metadata(self, text: str) -> str:
        # Regex pattern to match the metadata format and exclude './uploads/'
        metadata_pattern = r"\(source:\s*\.\/uploads\/([^|]+)\|\s*page:\s*(\d+)\)"
        
        # Function to replace the matched pattern with the desired format
        def replace_metadata(match):
            filename = match.group(1).strip()
            page_number = match.group(2).strip()
            return f"(출처 : {filename} | page: {page_number})"
        
        # Replace all occurrences in the text
        return re.sub(metadata_pattern, replace_metadata, text)

# Instantiate the parser
parser = LastPartOutputParser()


@st.cache_data(show_spinner="파일 임베딩 중...") #TODO: 파일이 임베딩 된 이후, streamlit에서 해당 파일을 제거하는 처리 필요
def process_and_embed_file(file) -> FAISS: 
  #TODO: FAISS indexing 방법 고려. 
  #방법 1. Store Embedded Data from Multiple Files Together. -> 가장 간편하지만, 데이터가 많아질 경우, 개별 파일에 대한 관리가 어려움.
  #방법 2. Separate Index for Each File, Unified Retriever -> FAISS는 multiple index searching을 지원하지 않기에, Aggregator 기능이 있는 custom retriever를 만들어야 함.
  # -> 일단 방법 1로 진행. 추후 방법 2로 변경 고려.
  
    # Load the existing FAISS index if it exists
    cache_path = os.path.join(CACHE_DIR, "combined_index") #combined_index라는 이름으로 인덱스 저장 및 관리
    if os.path.exists(cache_path):
      vector_store = FAISS.load_local(
          cache_path,
          embeddings_model,
          allow_dangerous_deserialization=True  # Pickle 파일을 FAISS Index로 deserialize하기 위해 필요
      )
    else:
        index = faiss.IndexFlatL2(len(embeddings_model.embed_query("test")))  # IndexFlatL2 : L2 distance를 사용하는 flat index. IndexFlatIP : Inner product를 사용하는 flat index
        vector_store = FAISS(
          embedding_function=embeddings_model, 
          index=index, 
          docstore=InMemoryDocstore(), 
          index_to_docstore_id={},
          # allow_dangerous_deserialization=True
          )
    
    # Process new file  TODO: 여러 종류의 Loader 구현 -> 현재는 PyPDFLoader만 지원
    file_path = os.path.join(UPLOADS_DIR, file.name)
    with open(file_path, 'wb') as f: #파일 저장
        f.write(file.read())
     
     #파일 형식에 따라 Loader 선택
     
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split(text_splitter=text_splitter)
    embeddings = embeddings_model.embed_documents([doc.page_content for doc in documents])

    # Add new embeddings to the existing vector store
    vector_store.add_documents(documents=documents)
    vector_store.save_local(cache_path)
    return vector_store

def format_documents(docs: list[str]) -> str:
    # Debugging: Log document content and type
    # for i, doc in enumerate(docs):
        # print(f"Document Content {i}: {doc.page_content}")
        
    return "\n\n".join([doc.page_content for doc in docs])
  
def load_vectorstore_from_cache(filename: str) -> FAISS:
    # Load an existing FAISS vectorstore
    cache_path = os.path.join(CACHE_DIR, filename)
    vector_store = FAISS.load_local(cache_path, embeddings_model, allow_dangerous_deserialization=True)
    return vector_store


# Streamlit UI
# with st.sidebar:
#     st.title("PDF to FAISS Embedding")
#     file = st.file_uploader("PDF 파일을 업로드 해주세요", type=["pdf"])

#     if file:
#       vectorstore = process_and_embed_file(file) #
#       st.success(f"업로드한 파일 '{file.name}' 이 성공적 임베딩 되었습니다. ")
#     else :
#       vectorstore = FAISS.load_local(os.path.join(CACHE_DIR, "combined_index"), embeddings_model, allow_dangerous_deserialization=True) #해당 인덱스가 로딩된 FAISS 객체를 반환
#       st.success(f"FAISS 인덱스가 성공적으로 로드되었습니다. ")
      
#     retriever =  vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
vectorstore = FAISS.load_local(os.path.join(CACHE_DIR, "combined_index"), embeddings_model, allow_dangerous_deserialization=True) #해당 인덱스가 로딩된 FAISS 객체를 반환
retriever =  vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    


#Callback Handler 설정
from langchain_core.callbacks.base import BaseCallbackHandler

class ChatCallbackHandler(BaseCallbackHandler):
  message = ""
  
  def on_llm_start(self, *args, **kwargs):
    self.message_box = st.empty()
    
  def on_llm_end(self, *args, **kwargs):
    save_message(self.message, 'ai')
    
  def on_llm_new_token(self, token, *args, **kwargs):
    # print(f"LLM 토큰 생성: {token}")
    self.message += token
    self.message_box.markdown(self.message)
  

#TODO : Reranker 추가 필요
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

#TODO : 싱글턴 방식의 llm 객체 생성 방식 변경 필요(완료)
class LLMManager:
  _instance = None

  @staticmethod
  def get_llm(model_name=None, model_params=None):
    if LLMManager._instance is None:
      if model_name == "Openai-GPT-4o":
        api_key = model_params.get("api_key")
        if not api_key:
          raise ValueError("API key is required for GPT-4o")
        LLMManager._instance = ChatOpenAI(
          model="gpt-4o",
          api_key=api_key,
          verbose=True,
          max_tokens=1500,
          streaming=True,
          callbacks=[ChatCallbackHandler()],
        )
      elif model_name == "Google-Gemma-2":
        LLMManager._instance = HuggingFacePipeline.from_model_id(
          model_id="google/gemma-2-2b",
          task="text-generation",
          verbose=True,
          # device_map='auto',
          device=None,
          device_map=None,
          model_kwargs={"device_map": "auto"},
          callbacks=[ChatCallbackHandler()],
          pipeline_kwargs={"max_new_tokens": 1000},
        ) 

        # Set LLM Cache
        set_llm_cache(InMemoryCache())
              
    return LLMManager._instance

#LLM 초기화
llm = None
if selected_model != "Openai-GPT-4o" or (selected_model == "Openai-GPT-4o" and 'openai_api_key' in globals()):
  llm = LLMManager.get_llm(
    model_name=selected_model, 
    model_params={"api_key": openai_api_key} if selected_model == "Openai-GPT-4o" else {}
  )
  
  #Mulit-Query Retriever용 LLM 객체 생성. TODO: Gemma-2 사용시 대응 필요
  
  # api_key = os.getenv("openai_api_key")
  # query_llm = ChatOpenAI(
  #   model="gpt-4o",
  #   api_key=api_key,
  # )
  
#Message History 구현 관련

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
  
config = {"configurable" : {"session_id": user_id}}

#TODO: trim_messages를 사용한 메시지 히스토리 관리 필요 -> trimmer의 체인 연결  https://python.langchain.com/v0.2/docs/how_to/trim_messages/
  
#TODO: Ensemble Retriever 구현
#1. BM25 Retriever 생성
  #1-1. document들을 로컬에 저장하도록 세팅
  #1-2. 어플리케이션 실행시, document들을 읽어들여 BM25 Retriever 생성
#2. FAISS Retriever 생성: 있는거 사용
#3. Ensemble Retriever 생성

# LLM integration with chat history
if 'llm' in globals() and llm:
    paint_history()
    if user_input := st.chat_input("질문을 입력하세요"):
        #add user input to the history
        user_inputs = st.session_state['chat_history'][user_id]
        # user_inputs.append(user_input)
      
        # Save and display user input
        send_message(user_input, 'user')
        
        # #Multi-Query Retriever 구현
        # from langchain.retrievers.multi_query import MultiQueryRetriever
        # mq_retriever = MultiQueryRetriever.from_llm(
        #   retriever =  retriever,
        #   llm = query_llm,
        # )

        # retrieved_docs = mq_retriever.invoke(user_input)
        # st.write(retrieved_docs) #디버깅용
        
        #일반 Retriever 사용
        # Retrieve documents based on user input
        # retrieved_docs = retriever.invoke(user_input)
        
        #Cross Encoder Reranker 사용
        retrieved_docs = compression_retriever.invoke(user_input)
        
        # # Format the retrieved documents as context
        context = format_documents(retrieved_docs)
        
        # Prepare inputs for the LLM
        inputs = {
            "context": context,
            "question": user_input,
            "messages": user_inputs,
        }
        
        #Create the Final Chain
        chain = prompt | llm | parser
        
        with_message_history = RunnableWithMessageHistory(
          chain,
          get_session_history,
          input_messages_key= "messages",
        )
        
        with st.chat_message("ai"):
            # response = chain.invoke(inputs)
            response =  with_message_history.invoke(inputs, config=config)
            # save_message(response, 'ai')
        # if selected_model == "Google-Gemma-2":
        #   send_message(response, 'ai')
          
        
# if __name__ == "main":
#   import os
#   os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#   import torch
#   import torch.multiprocessing as mp
#   device = 'cuda' if torch.cuda.is_available() else 'cpu'
#   mp.multiprocessing.set_start_method('spawn', force=True)