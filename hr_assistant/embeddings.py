"""Step 3:turn text into numbers (vectors) using jina."""

from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_embeddings_model():
    """Return a jina embeddings model. Reads JINA_API_KEY from the environment"""
    logger.info("Initilizing embeddings model %s", config.EMBEDDING_MODEL_NAME)
    return JinaEmbeddings(model = config.EMBEDDING_MODEL_NAME) 
