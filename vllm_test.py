#llm

from langchain_huggingface import HuggingFacePipeline
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
from langchain_community.document_loaders import PyPDFLoader
import uuid
from datetime import datetime, timedelta

llm = HuggingFacePipeline.from_model_id(
  model_id = "google/gemma-2-9b-it",
  task = "text-generation",
  verbose=True,
  device = 1,
  pipeline_kwargs = {
    "max_new_tokens": 1000,
  }
)

response = llm.invoke("Can you explain Paris to me?")
print(response)
  
  