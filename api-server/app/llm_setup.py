import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.chat_history import InMemoryChatMessageHistory
from modules.llm import LLMManager
from modules.parser import LastPartOutputParser
from modules.retriever import load_vectorstore_from_cache, create_retriever, format_documents

# Load environment variables for LLM setup (from config.py)
from config import OPENAI_API_KEY

# Function to initialize the LLM and chain
def initialize_llm_and_chain():
    # Initialize retriever
    vector_store = load_vectorstore_from_cache("combined_index")
    retriever = create_retriever(vector_store)

    # Initialize parser
    parser = LastPartOutputParser()

    # Initialize reranker
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    # Define the prompt template
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

    # Initialize LLM
    selected_model = "Openai-GPT-4o"  # Change this as needed
    llm = LLMManager.get_llm(model_name=selected_model, model_params={"api_key": OPENAI_API_KEY})

    # Create the final chain
    chain = prompt | llm | parser

    return chain, compression_retriever
