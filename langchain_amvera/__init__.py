"""LangChain интеграция для Amvera LLM."""

from langchain_amvera.amvera import AmveraLLM, create_amvera_chat_model

# Алиасы для совместимости
AmveraChatModel = AmveraLLM
ChatAmvera = AmveraLLM

__version__ = "0.3.1"
__all__ = [
    "AmveraLLM",
    "AmveraChatModel",
    "ChatAmvera",
    "create_amvera_chat_model",
]
