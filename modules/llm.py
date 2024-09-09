# llm.py

import os
from langchain_openai import ChatOpenAI
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_community.llms import HuggingFacePipeline
from langchain_core.callbacks.base import BaseCallbackHandler
import streamlit as st
import datetime


# Callback Handler for LLM
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, 'ai')

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# LLM Manager Class for Singleton Pattern
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
                    model_kwargs={"device_map": "auto"},
                    callbacks=[ChatCallbackHandler()],
                    pipeline_kwargs={"max_new_tokens": 1000},
                )

                # Set LLM Cache
                set_llm_cache(InMemoryCache())

        return LLMManager._instance

