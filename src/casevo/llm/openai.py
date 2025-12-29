"""
OpenAI LLM 接口实现

支持自定义 API 基础地址，兼容 OpenAI 格式的 API。
"""

import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Any

from casevo.llm.interface import LLM_INTERFACE


def create_session_with_retry(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """创建带有重试机制的 requests session"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# 全局详细日志开关
VERBOSE_LOGGING = True

def set_verbose_logging(enabled: bool):
    """设置是否启用详细日志"""
    global VERBOSE_LOGGING
    VERBOSE_LOGGING = enabled


class OpenAIEmbeddingFunction:
    """
    OpenAI 嵌入函数
    
    兼容 ChromaDB 的嵌入函数接口。
    """
    
    def __init__(self, api_key: str, 
                 base_url: str = "https://api.whatai.cc",
                 model: str = "text-embedding-ada-002"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._session = create_session_with_retry(retries=3, backoff_factor=1.0)
        self._last_request_time = 0
        self._min_interval = 0.2  # 最小请求间隔（秒）
        self._call_count = 0
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """获取文本嵌入"""
        self._call_count += 1
        
        # 请求限速
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        
        url = f"{self.base_url}/v1/embeddings"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 截断过长的文本
        truncated_input = [text[:500] + "..." if len(text) > 500 else text for text in input]
        
        data = {
            "input": input,
            "model": self.model
        }
        
        if VERBOSE_LOGGING:
            preview = truncated_input[0][:50] + "..." if len(truncated_input[0]) > 50 else truncated_input[0]
            print(f"        [嵌入 #{self._call_count}] {len(input)} 条文本, 预览: \"{preview}\"")
        
        try:
            start_time = time.time()
            self._last_request_time = time.time()
            response = self._session.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            embeddings = sorted(result["data"], key=lambda x: x["index"])
            elapsed_time = time.time() - start_time
            
            if VERBOSE_LOGGING:
                tokens = result.get("usage", {}).get("total_tokens", "?")
                print(f"        [嵌入 #{self._call_count}] ✓ 成功, {elapsed_time:.2f}s, {tokens} tokens")
            
            return [item["embedding"] for item in embeddings]
        except Exception as e:
            print(f"        [嵌入 #{self._call_count}] ✗ 失败: {e}")
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
                 base_url: str = "https://api.whatai.cc",
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
        
        # 创建带重试的 session
        self._session = create_session_with_retry(retries=3, backoff_factor=1.0)
        self._last_request_time = 0
        self._min_interval = 0.3  # 最小请求间隔（秒）
        self._call_count = 0
        
        self._embedding_function = OpenAIEmbeddingFunction(
            api_key=api_key,
            base_url=base_url,
            model=embedding_model
        )
    
    def send_message(self, prompt: str, json_flag: bool = False) -> str:
        """发送消息给 LLM"""
        self._call_count += 1
        
        # 请求限速
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        
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
        
        # 详细日志
        if VERBOSE_LOGGING:
            preview = prompt[:80].replace('\n', ' ') + "..." if len(prompt) > 80 else prompt.replace('\n', ' ')
            print(f"        [LLM #{self._call_count}] 发送请求, 预览: \"{preview}\"")
        
        try:
            start_time = time.time()
            self._last_request_time = time.time()
            response = self._session.post(
                url, headers=headers, json=data, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            elapsed_time = time.time() - start_time
            
            if VERBOSE_LOGGING:
                usage = result.get("usage", {})
                tokens_in = usage.get("prompt_tokens", "?")
                tokens_out = usage.get("completion_tokens", "?")
                resp_preview = content[:60].replace('\n', ' ') + "..." if len(content) > 60 else content.replace('\n', ' ')
                print(f"        [LLM #{self._call_count}] ✓ 成功, {elapsed_time:.2f}s, {tokens_in}→{tokens_out} tokens")
                print(f"        [LLM #{self._call_count}] 回复: \"{resp_preview}\"")
            
            return content
            
        except requests.exceptions.Timeout:
            print(f"        [LLM #{self._call_count}] ✗ 请求超时")
            return ""
        except requests.exceptions.HTTPError as e:
            print(f"        [LLM #{self._call_count}] ✗ HTTP 错误: {e}")
            return ""
        except Exception as e:
            print(f"        [LLM #{self._call_count}] ✗ 发送失败: {e}")
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


# API 配置 - 从环境变量读取
import os

DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")


def get_api_key_for_model(model: str) -> str:
    """获取 API 密钥（从环境变量）"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "未设置 OPENAI_API_KEY 环境变量。请在 .env 文件中设置或导出环境变量。\n"
            "例如: export OPENAI_API_KEY='your-api-key'"
        )
    return api_key


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

