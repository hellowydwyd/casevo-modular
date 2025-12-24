"""
LLM 接口测试

测试 LLM_INTERFACE, OpenAILLM, OpenAIEmbeddingFunction（使用 mock）
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from casevo.llm.interface import LLM_INTERFACE
from casevo.llm.openai import (
    OpenAILLM,
    OpenAIEmbeddingFunction,
    create_default_llm,
    create_llm,
    get_api_key_for_model,
    MODEL_API_KEYS,
    DEFAULT_MODEL,
    DEFAULT_BASE_URL
)


class TestLLMInterface:
    """LLM_INTERFACE 测试类"""
    
    def test_is_abstract(self):
        """测试接口是抽象类"""
        with pytest.raises(TypeError):
            LLM_INTERFACE()
    
    def test_concrete_implementation(self):
        """测试具体实现"""
        class ConcreteLLM(LLM_INTERFACE):
            def send_message(self, prompt, json_flag=False):
                return "response"
            
            def send_embedding(self, text_list):
                return [[0.1] * 10 for _ in text_list]
            
            def get_lang_embedding(self):
                return Mock()
        
        llm = ConcreteLLM()
        assert llm.send_message("test") == "response"


class TestOpenAILLM:
    """OpenAILLM 测试类"""
    
    def test_init(self):
        """测试初始化"""
        llm = OpenAILLM(
            api_key="test-key",
            base_url="https://api.test.com",
            model="gpt-4o-mini"
        )
        
        assert llm.api_key == "test-key"
        assert llm.base_url == "https://api.test.com"
        assert llm.model == "gpt-4o-mini"
    
    def test_init_strips_trailing_slash(self):
        """测试初始化时去除 base_url 末尾斜杠"""
        llm = OpenAILLM(
            api_key="test-key",
            base_url="https://api.test.com/",
            model="gpt-4o-mini"
        )
        
        assert llm.base_url == "https://api.test.com"
    
    def test_init_default_values(self):
        """测试默认值"""
        llm = OpenAILLM(api_key="test-key")
        
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2000
        assert llm.timeout == 120
        assert llm.model == "gpt-4o-mini"
    
    @patch('requests.post')
    def test_send_message_success(self, mock_post):
        """测试发送消息成功"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "测试响应"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        llm = OpenAILLM(api_key="test-key")
        result = llm.send_message("测试消息")
        
        assert result == "测试响应"
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_send_message_with_json_flag(self, mock_post):
        """测试发送消息并要求 JSON 响应"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"result": "ok"}'}}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        llm = OpenAILLM(api_key="test-key")
        result = llm.send_message("返回 JSON", json_flag=True)
        
        assert result == '{"result": "ok"}'
        
        # 验证请求包含 response_format
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['json']['response_format'] == {"type": "json_object"}
    
    @patch('requests.post')
    def test_send_message_timeout(self, mock_post):
        """测试发送消息超时"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        llm = OpenAILLM(api_key="test-key")
        result = llm.send_message("测试")
        
        assert result == ""
    
    @patch('requests.post')
    def test_send_message_http_error(self, mock_post):
        """测试 HTTP 错误"""
        import requests
        mock_post.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        
        llm = OpenAILLM(api_key="test-key")
        result = llm.send_message("测试")
        
        assert result == ""
    
    def test_get_lang_embedding(self):
        """测试获取嵌入函数"""
        llm = OpenAILLM(api_key="test-key")
        embedding_func = llm.get_lang_embedding()
        
        assert isinstance(embedding_func, OpenAIEmbeddingFunction)


class TestOpenAIEmbeddingFunction:
    """OpenAIEmbeddingFunction 测试类"""
    
    def test_init(self):
        """测试初始化"""
        func = OpenAIEmbeddingFunction(
            api_key="test-key",
            base_url="https://api.test.com",
            model="text-embedding-ada-002"
        )
        
        assert func.api_key == "test-key"
        assert func.base_url == "https://api.test.com"
        assert func.model == "text-embedding-ada-002"
    
    @patch('requests.post')
    def test_call_success(self, mock_post):
        """测试调用成功"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        func = OpenAIEmbeddingFunction(api_key="test-key")
        result = func(["text1", "text2"])
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
    
    @patch('requests.post')
    def test_call_failure_returns_zeros(self, mock_post):
        """测试调用失败返回零向量"""
        mock_post.side_effect = Exception("API Error")
        
        func = OpenAIEmbeddingFunction(api_key="test-key")
        result = func(["text1", "text2"])
        
        assert len(result) == 2
        assert len(result[0]) == 1536  # 默认向量维度
        assert all(v == 0.0 for v in result[0])


class TestHelperFunctions:
    """辅助函数测试类"""
    
    def test_get_api_key_for_known_model(self):
        """测试已知模型获取 API key"""
        key = get_api_key_for_model("gpt-4o")
        assert key == MODEL_API_KEYS["gpt-4o"]
    
    def test_get_api_key_for_prefix_match(self):
        """测试前缀匹配获取 API key"""
        key = get_api_key_for_model("gpt-4o-mini-test")
        assert key == MODEL_API_KEYS["gpt-4o-mini"]
        
        key = get_api_key_for_model("gpt-4o-test")
        assert key == MODEL_API_KEYS["gpt-4o"]
        
        key = get_api_key_for_model("gpt-3.5-turbo-test")
        assert key == MODEL_API_KEYS["gpt-3.5-turbo"]
    
    def test_get_api_key_for_unknown_model(self):
        """测试未知模型返回默认 key"""
        key = get_api_key_for_model("unknown-model")
        assert key == MODEL_API_KEYS[DEFAULT_MODEL]
    
    def test_create_default_llm(self):
        """测试创建默认 LLM"""
        llm = create_default_llm()
        
        assert isinstance(llm, OpenAILLM)
        assert llm.model == DEFAULT_MODEL
        assert llm.base_url == DEFAULT_BASE_URL
    
    def test_create_default_llm_with_model(self):
        """测试创建指定模型的默认 LLM"""
        llm = create_default_llm(model="gpt-4o")
        
        assert llm.model == "gpt-4o"
        assert llm.api_key == MODEL_API_KEYS["gpt-4o"]
    
    def test_create_llm(self):
        """测试创建 LLM"""
        llm = create_llm("gpt-4o")
        
        assert isinstance(llm, OpenAILLM)
        assert llm.model == "gpt-4o"
    
    def test_create_llm_with_custom_key(self):
        """测试使用自定义 key 创建 LLM"""
        llm = create_llm("gpt-4o", api_key="custom-key")
        
        assert llm.api_key == "custom-key"


class TestSendMessageWithHistory:
    """带历史记录发送消息测试"""
    
    @patch('requests.post')
    def test_send_with_history(self, mock_post):
        """测试发送带历史记录的消息"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "继续对话响应"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        llm = OpenAILLM(api_key="test-key")
        
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "继续对话"}
        ]
        
        result = llm.send_message_with_history(messages)
        
        assert result == "继续对话响应"
        
        # 验证请求包含所有消息
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['json']['messages'] == messages
    
    @patch('requests.post')
    def test_send_with_history_json_flag(self, mock_post):
        """测试带历史记录和 JSON 标志"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"status": "ok"}'}}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        llm = OpenAILLM(api_key="test-key")
        
        messages = [{"role": "user", "content": "返回 JSON"}]
        result = llm.send_message_with_history(messages, json_flag=True)
        
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['json']['response_format'] == {"type": "json_object"}


class TestMODEL_API_KEYS:
    """MODEL_API_KEYS 配置测试"""
    
    def test_keys_exist(self):
        """测试配置的 key 存在"""
        required_models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        
        for model in required_models:
            assert model in MODEL_API_KEYS
            assert len(MODEL_API_KEYS[model]) > 0
    
    def test_default_model_in_keys(self):
        """测试默认模型在配置中"""
        assert DEFAULT_MODEL in MODEL_API_KEYS

