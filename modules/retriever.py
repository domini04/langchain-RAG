# retriever_module.py

import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Environment setup
CACHE_DIR = './cache'
UPLOADS_DIR = "./uploads"

# Ensure directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize the embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
    show_progress=True,
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
)

# Function to process and embed a file
def process_and_embed_file(file) -> FAISS:
    cache_path = os.path.join(CACHE_DIR, "combined_index")
    if os.path.exists(cache_path):
        vector_store = FAISS.load_local(
            cache_path,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
    else:
        index = faiss.IndexFlatL2(len(embeddings_model.embed_query("test")))
        vector_store = FAISS(
            embedding_function=embeddings_model, 
            index=index, 
            docstore=InMemoryDocstore(), 
            index_to_docstore_id={},
        )

    file_path = os.path.join(UPLOADS_DIR, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.read())

    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split(text_splitter=text_splitter)
    embeddings = embeddings_model.embed_documents([doc.page_content for doc in documents])

    vector_store.add_documents(documents=documents)
    vector_store.save_local(cache_path)
    return vector_store

# Function to load vectorstore from cache
def load_vectorstore_from_cache(filename: str) -> FAISS:
    cache_path = os.path.join(CACHE_DIR, filename)
    vector_store = FAISS.load_local(cache_path, embeddings_model, allow_dangerous_deserialization=True)
    return vector_store

# Function to format documents for display
def format_documents(docs: list) -> str:
    return "\n\n".join([doc.page_content for doc in docs])

# Function to create a retriever from FAISS
def create_retriever(vector_store: FAISS):
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
