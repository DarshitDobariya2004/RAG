"""Step 4: store chunk embeddings in FAISS so we can search them later."""

import os
from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

# build_vector_store
def build_vector_store(chunks):
    """Enbed every chunk and build a searchable faiss inded in memory."""
    logger.info("Embedding %d chunk(s) and building FAISS index....")
    embeddings_model = get_embeddings_model()
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    logger.info("FAISS index built in memory")
    return vector_store

# save vector store
def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH) -> None:
    """Save the faiss index to disk so we do not have to rebuild it everytime."""
    vector_store.save_local(path)
    logger.info("saved FAISS index to %s",path)

def load_vector_store(path:str = config.VECTOR_STORE_PATH):
    """Load a previously saved FAISS index from disk."""
    logger.info("Loading FAISS index from %s",path)
    embedding_model = get_embeddings_model()

    return FAISS.load_local(path,embedding_model,allow_dangerous_deserialization=True)

def vector_store_exists(path :str = config.VECTOR_STORE_PATH) -> bool:
    """check if a saved faiss index already exists on disk."""
    return os.path.exists(os.path.join(path,"index.faiss"))

def get_retriever(vector_store, k:int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever that returns the top-k matching chunks."""
    logger.info("creating retriever with top_k=%d",k)
    return vector_store.as_retriever(search_kwargs = {"k":k})