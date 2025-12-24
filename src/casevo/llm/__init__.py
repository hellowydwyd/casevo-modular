"""
Casevo LLM 模块

大语言模型接口和实现。
"""

from casevo.llm.interface import LLM_INTERFACE
from casevo.llm.openai import (
    OpenAILLM,
    OpenAIEmbeddingFunction,
    create_default_llm,
    create_llm,
    get_api_key_for_model,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MODEL_API_KEYS,
)

__all__ = [
    "LLM_INTERFACE",
    "OpenAILLM",
    "OpenAIEmbeddingFunction",
    "create_default_llm",
    "create_llm",
    "get_api_key_for_model",
    "DEFAULT_API_KEY",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "MODEL_API_KEYS",
]

