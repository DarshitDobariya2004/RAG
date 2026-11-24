"""Step 6b: route the LLM through the LLM gateway.

Instead of calling Groq directly, the main LLM call goes through LLM Gateway.
It stores the real Groq credentials - our code never sees the raw Groq key.
"""
from langchain_openai import ChatOpenAI

from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_gateway_llm():
    """Return a primary LLM routed through LLM Gateway with an automatic backup model."""
    logger.info(
        "Routing LLM calls through LLM Gateway (primary=%s, backup=%s)",
        config.LLM_MODEL_NAME,
        config.LLM_BACKUP_MODEL_NAME,
    )

    primary_llm = ChatOpenAI(
        api_key=config.LLM_GATEWAY_API_KEY,
        base_url=config.LLM_GATEWAY_URL,
        model=config.LLM_MODEL_NAME,          
        temperature=0,
        max_retries=1,
    )

    backup_llm = ChatOpenAI(
        api_key=config.LLM_GATEWAY_API_KEY,
        base_url=config.LLM_GATEWAY_URL,
        model= config.LLM_BACKUP_MODEL_NAME,
        temperature=0,
    )

    return primary_llm.with_fallbacks([backup_llm])