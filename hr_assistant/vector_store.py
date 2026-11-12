"""Step 4: store chunk embeddings in FAISS so we can search them later."""

import os
from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model

# build_vector_store
def build_vector_store(chunks):
    """Enbed every chunk and build a searchable faiss inded in memory."""
    embeddings_model = get_embeddings_model()
    return FAISS.from_documents(chunks, embeddings_model)

# save vector store
def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH) -> None:
    """Save the faiss index to disk so we do not have to rebuild it everytime."""
    vector_store.save_local(path)

def load_vector_store(path:str = config.VECTOR_STORE_PATH):
    """Load a previously saved FAISS index from disk."""
    embedding_model = get_embeddings_model()
    return FAISS.load_local(path,embedding_model,allow_dangerous_deserialization=True)

def vector_store_exists(path :str = config.VECTOR_STORE_PATH) -> bool:
    """check if a saved faiss index already exists on disk."""
    return os.path.exists(os.path.join(path,"index.faiss"))

def get_retriever(vector_store, k:int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever that returns the top-k matching chunks."""
    return vector_store.as_retriever(search_kwargs = {"k":k})