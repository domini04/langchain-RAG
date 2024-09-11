from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Header
from fastapi.responses import RedirectResponse
from langserve import add_routes
from dotenv import load_dotenv
from modules.llm import LLMManager
from modules.parser import LastPartOutputParser
from modules.retriever import load_vectorstore_from_cache, create_retriever, format_documents
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.messages import HumanMessage, AIMessage

from .models import QuestionRequest
from .db import questions_collection
from bson import ObjectId

import os
import uuid
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load the .env file located outside 'api-server'
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Load the .env file located inside 'api-server/app'
load_dotenv(os.path.join(os.path.dirname(__file__), './.env'))

# Apply LangChain tracing settings from .env
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "local-llm-langchain")

api_token = os.getenv("API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    version="0.1",
    description="Langserve을 활용한 FastAPI 기반 LLM API",
)

# Redirect root to docs
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


#LLM 관련
# Chat history store (using in-memory session)
store = {}

# Functions to manage session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Initialize retriever, LLMs, parser, and other components
vector_store = load_vectorstore_from_cache("combined_index")
retriever = create_retriever(vector_store)
parser = LastPartOutputParser()

# Reranker setup (if needed)
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

prompt =  ChatPromptTemplate.from_messages( 
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


# LLM initialization (use selected model)
selected_model = "Openai-GPT-4o"  # Change based on actual model being used
llm = LLMManager.get_llm(model_name=selected_model, model_params={"api_key": os.getenv("OPENAI_API_KEY")})

# Create the final chain
chain = prompt | llm | parser

import time
@app.middleware("http") # TODO: Middleware에서 진행해야 할 작업 고민
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

#DB 관련
# Background task for saving question to MongoDB
async def save_question_to_db(session_id: str, user_input: str, response: str):
    # Create a QuestionRequest object and insert it into MongoDB
    question_document = {
        "session_id": session_id,
        "question": user_input,
        "answer": response,
        "timestamp": datetime.utcnow()  # Optional timestamp
    }
    await questions_collection.insert_one(question_document)

#Request Body 정의
from pydantic import BaseModel

class LLMRequest(BaseModel):
    question: str
    session_id: str = str(uuid.uuid4())

# Define API to handle chain requests
@app.post("/api")
async def invoke_llm(request: LLMRequest, background_tasks: BackgroundTasks, api_key: str = Header(...)):
    # Validate API token
    print(f"API Key: {api_key}")
    print(f"API Token: {api_token}")
    if api_key != api_token:
        raise HTTPException(status_code=401, detail="Invalid API token")

    # Parse incoming request data (Pydantic version)
    user_input = request.question
    session_id = request.session_id or str(uuid.uuid4())

    # Handle session-based message history
    user_inputs = get_session_history(session_id)

    # Save the human message to the session history
    user_inputs.add_message(HumanMessage(content=user_input))

    # Cross Encoder Reranker retrieval
    retrieved_docs = compression_retriever.invoke(user_input)

    # Format retrieved documents as context
    context = format_documents(retrieved_docs)

    # Prepare inputs for the LLM
    inputs = {
        "context": context,
        "question": user_input,
        "messages": user_inputs.messages,  # Include chat history
    }

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )

    # Run the chain and return the response
    ai_response = with_message_history.invoke(inputs, config={"configurable": {"session_id": session_id}})

    # Save the AI message to the session history
    user_inputs.add_message(AIMessage(content=ai_response))

    # Background task to save question and response to the DB
    background_tasks.add_task(save_question_to_db, session_id, user_input, ai_response)

    # Validate the response using QuestionRequest model
    validated_response = QuestionRequest(session_id=session_id, question=user_input, answer=ai_response)

    return validated_response.dict()  # Return the validated response as a dictionary

from langchain.chat_models import ChatOpenAI
add_routes(
    app,
    ChatOpenAI(model="gpt-4o"),
    path="/openai",
)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
