from fastapi import FastAPI, Request
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

import os
import uuid
from datetime import datetime

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Apply LangChain tracing settings from .env
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "local-llm-langchain")

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

# Define the prompt template for the LLM chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """당신은 느린 학습자 관련 교육 전문가입니다. 
        당신은 느린 학습자 관련 질문을 받고, 이에 대한 전문적인 답변을 제공합니다.
        아래 컨텍스트를 참고하여 질문에 답변하십시오. 답을 모르면 "모르겠습니다. 
        느린 학습자에 관한 질문만 답변해주세요"라고 답변하세요. 답을 지어내지 마세요.
        또한 답변을 제공할 때에는 출처(source)를 함께 제공해주세요. 이를 통해 사용자가 
        답변의 신뢰성을 확인할 수 있습니다. 컨텍스트: {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("ai", "안녕하세요, 느린 학습자에 대해 어떤 사항이 궁금하신가요?"),
        ("human", "{question}")
    ]
)

# LLM initialization (use selected model)
selected_model = "Openai-GPT-4o"  # Change based on actual model being used
llm = LLMManager.get_llm(model_name=selected_model, model_params={"api_key": os.getenv("OPENAI_API_KEY")})

# Define API to handle chain requests
@app.post("/rag-system")
async def rag_system(request: Request):
    request_data = await request.json()
    
    # Parse incoming request data
    user_input = request_data.get("question", "")
    session_id = request_data.get("session_id", str(uuid.uuid4()))  # Generate a session if not provided

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
    
    # Create the final chain
    chain = prompt | llm | parser
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )
    
    # Run the chain and return the response
    response = with_message_history.invoke(inputs, config={"configurable": {"session_id": session_id}})
    
    # Save the AI message to the session history
    user_inputs.add_message(AIMessage(content=response))

    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
