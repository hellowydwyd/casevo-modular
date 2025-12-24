"""
提示模板测试

测试 Prompt, PromptFactory
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from casevo.prompt import Prompt, PromptFactory


class TestPromptFactory:
    """PromptFactory 测试类"""
    
    def test_init(self, mock_llm, prompt_dir):
        """测试初始化"""
        factory = PromptFactory(prompt_dir, mock_llm)
        
        assert factory.prompt_folder == prompt_dir
        assert factory.llm is mock_llm
        assert factory.env is not None
    
    def test_init_invalid_folder(self, mock_llm):
        """测试无效文件夹初始化"""
        with pytest.raises(Exception, match="prompt folder not exist"):
            PromptFactory("/nonexistent/path", mock_llm)
    
    def test_get_template(self, mock_llm, prompt_dir):
        """测试获取模板"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('test.txt')
        
        assert isinstance(prompt, Prompt)
        assert prompt.template is not None
        assert prompt.factory is factory
    
    def test_get_template_not_exist(self, mock_llm, prompt_dir):
        """测试获取不存在的模板"""
        factory = PromptFactory(prompt_dir, mock_llm)
        
        with pytest.raises(Exception, match="not exist"):
            factory.get_template('nonexistent.txt')
    
    def test_send_message(self, mock_llm, prompt_dir):
        """测试发送消息"""
        factory = PromptFactory(prompt_dir, mock_llm)
        
        response = factory.__send_message__("测试消息")
        
        assert response is not None
        assert mock_llm.call_history[-1]['prompt'] == "测试消息"


class TestPrompt:
    """Prompt 测试类"""
    
    def test_init(self, mock_llm, prompt_dir):
        """测试初始化"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('test.txt')
        
        assert prompt.template is not None
        assert prompt.factory is factory
    
    def test_get_prompt(self, mock_llm, prompt_dir):
        """测试获取渲染后的 prompt"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('test.txt')
        
        # 测试渲染
        result = prompt.__get_prompt__({
            "agent": {"description": "一位程序员"},
            "model": {},
            "extra": {"question": "你好吗？"}
        })
        
        assert "一位程序员" in result
        assert "你好吗？" in result
    
    def test_send_prompt_without_agent(self, mock_llm, prompt_dir):
        """测试不带 agent 发送 prompt"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('test.txt')
        
        response = prompt.send_prompt(ertra={"question": "测试问题"})
        
        assert response is not None
    
    def test_send_prompt_with_agent(self, mock_llm, prompt_dir, mock_agent):
        """测试带 agent 发送 prompt"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('test.txt')
        
        response = prompt.send_prompt(
            ertra={"question": "测试问题"},
            agent=mock_agent,
            model=mock_agent.model
        )
        
        assert response is not None
    
    def test_send_prompt_with_model(self, mock_llm, prompt_dir, mock_model):
        """测试带 model 发送 prompt"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('test.txt')
        
        response = prompt.send_prompt(
            ertra={"question": "测试问题"},
            model=mock_model
        )
        
        assert response is not None


class TestPromptTemplates:
    """Prompt 模板测试类"""
    
    def test_reflect_template(self, mock_llm, prompt_dir):
        """测试反思模板"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('reflect.txt')
        
        result = prompt.__get_prompt__({
            "agent": {},
            "model": {},
            "extra": {
                "long_memory": "我是一个程序员",
                "short_memory": "今天学习了新技术"
            }
        })
        
        assert "我是一个程序员" in result
        assert "今天学习了新技术" in result
    
    def test_choice_template(self, mock_llm, prompt_dir):
        """测试选择模板"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('choice.txt')
        
        result = prompt.__get_prompt__({
            "agent": {},
            "model": {},
            "extra": {"options": "A. 选项1, B. 选项2, C. 选项3"}
        })
        
        assert "A. 选项1" in result
    
    def test_score_template(self, mock_llm, prompt_dir):
        """测试评分模板"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('score.txt')
        
        result = prompt.__get_prompt__({
            "agent": {},
            "model": {},
            "extra": {"content": "这是需要评分的内容"}
        })
        
        assert "这是需要评分的内容" in result
    
    def test_json_template(self, mock_llm, prompt_dir):
        """测试 JSON 模板"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('json.txt')
        
        result = prompt.__get_prompt__({
            "agent": {},
            "model": {},
            "extra": {"request": "返回用户信息"}
        })
        
        assert "返回用户信息" in result


class TestPromptRendering:
    """Prompt 渲染测试类"""
    
    def test_render_with_complex_data(self, mock_llm, prompt_dir):
        """测试复杂数据渲染"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('test.txt')
        
        result = prompt.__get_prompt__({
            "agent": {
                "description": {
                    "general": "35岁程序员",
                    "character": "理性",
                    "issue": "关注技术"
                }
            },
            "model": {},
            "extra": {"question": "你的观点是什么？"}
        })
        
        assert "你的观点是什么？" in result
    
    def test_render_with_empty_extra(self, mock_llm, prompt_dir):
        """测试空 extra 渲染"""
        factory = PromptFactory(prompt_dir, mock_llm)
        prompt = factory.get_template('test.txt')
        
        # 应该不报错，只是 extra 部分为空
        result = prompt.__get_prompt__({
            "agent": {"description": "测试"},
            "model": {},
            "extra": {}
        })
        
        assert result is not None

