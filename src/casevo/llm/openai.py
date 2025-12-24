"""
OpenAI LLM 接口实现

支持自定义 API 基础地址，兼容 OpenAI 格式的 API。
"""

import json
import requests
from typing import List, Dict, Any

from casevo.llm.interface import LLM_INTERFACE


class OpenAIEmbeddingFunction:
    """
    OpenAI 嵌入函数
    
    兼容 ChromaDB 的嵌入函数接口。
    """
    
    def __init__(self, api_key: str, 
                 base_url: str = "https://api.openai.com",
                 model: str = "text-embedding-ada-002"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """获取文本嵌入"""
        url = f"{self.base_url}/v1/embeddings"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "input": input,
            "model": self.model
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            embeddings = sorted(result["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings]
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            return [[0.0] * 1536 for _ in input]
    
    def name(self) -> str:
        """返回嵌入函数名称，ChromaDB 需要此方法"""
        return f"openai_{self.model}"
    
    def embed_query(self, input: List[str]) -> List[List[float]]:
        """嵌入查询文本，ChromaDB query 操作需要此方法"""
        return self.__call__(input)
    
    def is_legacy(self) -> bool:
        """ChromaDB 需要此方法来判断是否是旧版嵌入函数"""
        return False
    
    def default_space(self) -> str:
        """ChromaDB 需要此方法来获取默认的向量空间类型"""
        return "cosine"
    
    def supported_spaces(self) -> List[str]:
        """ChromaDB 需要此方法来获取支持的向量空间类型列表"""
        return ["cosine", "l2", "ip"]


class OpenAILLM(LLM_INTERFACE):
    """
    OpenAI LLM 接口
    
    支持自定义 API 基础地址，兼容 OpenAI API 格式。
    """
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.openai.com",
                 model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-ada-002",
                 temperature: float = 0.7,
                 max_tokens: int = 2000,
                 timeout: int = 120):
        """
        初始化 OpenAI LLM 接口
        
        参数:
            api_key: API 密钥
            base_url: API 基础地址
            model: 聊天模型名称
            embedding_model: 嵌入模型名称
            temperature: 生成温度
            max_tokens: 最大生成令牌数
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self._embedding_function = OpenAIEmbeddingFunction(
            api_key=api_key,
            base_url=base_url,
            model=embedding_model
        )
    
    def send_message(self, prompt: str, json_flag: bool = False) -> str:
        """发送消息给 LLM"""
        url = f"{self.base_url}/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if json_flag:
            data["response_format"] = {"type": "json_object"}
        
        try:
            response = requests.post(
                url, headers=headers, json=data, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.Timeout:
            print("请求超时")
            return ""
        except requests.exceptions.HTTPError as e:
            print(f"HTTP 错误: {e}")
            return ""
        except Exception as e:
            print(f"发送消息失败: {e}")
            return ""
    
    def send_embedding(self, text_list: List[str]) -> List[List[float]]:
        """获取文本嵌入向量"""
        return self._embedding_function(text_list)
    
    def get_lang_embedding(self) -> OpenAIEmbeddingFunction:
        """获取嵌入函数"""
        return self._embedding_function
    
    def send_message_with_history(self, 
                                  messages: List[Dict[str, str]], 
                                  json_flag: bool = False) -> str:
        """发送带历史记录的消息"""
        url = f"{self.base_url}/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if json_flag:
            data["response_format"] = {"type": "json_object"}
        
        try:
            response = requests.post(
                url, headers=headers, json=data, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"发送消息失败: {e}")
            return ""


# API 密钥配置
MODEL_API_KEYS = {
    "gpt-4o": "sk-LxHyQAmGfoMXPLS2qxQs9lUWTjlkYJU48IHnFCk3VFtZ442I",
    "gpt-4o-2024-05-13": "sk-LxHyQAmGfoMXPLS2qxQs9lUWTjlkYJU48IHnFCk3VFtZ442I",
    "gpt-4o-2024-08-06": "sk-LxHyQAmGfoMXPLS2qxQs9lUWTjlkYJU48IHnFCk3VFtZ442I",
    "gpt-4o-mini": "sk-ABRLKFX1TiP8DUJOSXqmNQUU9khYM6Mjr7RP4R2RpTzySIi2",
    "gpt-4o-mini-2024-07-18": "sk-ABRLKFX1TiP8DUJOSXqmNQUU9khYM6Mjr7RP4R2RpTzySIi2",
    "gpt-3.5-turbo": "sk-ABRLKFX1TiP8DUJOSXqmNQUU9khYM6Mjr7RP4R2RpTzySIi2",
    "gpt-3.5-turbo-0125": "sk-ABRLKFX1TiP8DUJOSXqmNQUU9khYM6Mjr7RP4R2RpTzySIi2",
}

DEFAULT_BASE_URL = "https://api.whatai.cc"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_API_KEY = MODEL_API_KEYS[DEFAULT_MODEL]


def get_api_key_for_model(model: str) -> str:
    """获取指定模型对应的 API 密钥"""
    if model in MODEL_API_KEYS:
        return MODEL_API_KEYS[model]
    
    if model.startswith("gpt-4o-mini"):
        return MODEL_API_KEYS["gpt-4o-mini"]
    elif model.startswith("gpt-4o"):
        return MODEL_API_KEYS["gpt-4o"]
    elif model.startswith("gpt-3.5"):
        return MODEL_API_KEYS["gpt-3.5-turbo"]
    
    return DEFAULT_API_KEY


def create_default_llm(model: str = None) -> OpenAILLM:
    """创建默认配置的 LLM 实例"""
    model = model or DEFAULT_MODEL
    api_key = get_api_key_for_model(model)
    
    return OpenAILLM(
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
        model=model,
        temperature=0.7
    )


def create_llm(model: str, api_key: str = None) -> OpenAILLM:
    """创建指定模型的 LLM 实例"""
    if api_key is None:
        api_key = get_api_key_for_model(model)
    
    return OpenAILLM(
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
        model=model,
        temperature=0.7
    )

