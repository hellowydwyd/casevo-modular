"""
Pytest 配置文件

提供所有测试共享的 fixtures 和 mock 对象。
"""

import sys
import os
import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import networkx as nx

# 添加 src 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================
# Mock LLM 相关
# ============================================================

class MockLLM:
    """Mock LLM 实现，用于测试时避免真实 API 调用"""
    
    def __init__(self, default_response="这是一个测试响应"):
        self.default_response = default_response
        self.call_history = []
        self._embedding_function = MockEmbeddingFunction()
    
    def send_message(self, prompt, json_flag=False):
        """模拟发送消息"""
        self.call_history.append({
            'method': 'send_message',
            'prompt': prompt,
            'json_flag': json_flag
        })
        if json_flag:
            return '{"status": "ok", "result": "test"}'
        return self.default_response
    
    def send_embedding(self, text_list):
        """模拟获取嵌入向量"""
        self.call_history.append({
            'method': 'send_embedding',
            'text_list': text_list
        })
        # 返回固定维度的模拟嵌入向量
        return [[0.1] * 1536 for _ in text_list]
    
    def get_lang_embedding(self):
        """返回模拟的嵌入函数"""
        return self._embedding_function


class MockEmbeddingFunction:
    """Mock 嵌入函数，兼容 ChromaDB"""
    
    def __call__(self, input):
        """返回模拟的嵌入向量"""
        if isinstance(input, str):
            input = [input]
        return [[0.1] * 1536 for _ in input]
    
    def name(self):
        """返回嵌入函数名称，ChromaDB 需要此方法"""
        return "mock_embedding_function"
    
    def embed_query(self, input):
        """嵌入查询文本，ChromaDB query 操作需要此方法"""
        if isinstance(input, str):
            input = [input]
        return [[0.1] * 1536 for _ in input]
    
    def is_legacy(self):
        """ChromaDB 需要此方法来判断是否是旧版嵌入函数"""
        return False
    
    def default_space(self):
        """ChromaDB 需要此方法来获取默认的向量空间类型"""
        return "cosine"
    
    def supported_spaces(self):
        """ChromaDB 需要此方法来获取支持的向量空间类型列表"""
        return ["cosine", "l2", "ip"]


@pytest.fixture
def mock_llm():
    """提供 Mock LLM 实例"""
    return MockLLM()


@pytest.fixture
def mock_llm_with_json():
    """提供返回 JSON 的 Mock LLM 实例"""
    llm = MockLLM()
    llm.default_response = '{"choice": "A", "score": 0.85, "reason": "测试原因"}'
    return llm


# ============================================================
# 网络图相关
# ============================================================

@pytest.fixture
def simple_graph():
    """创建简单的完全图（3 节点）"""
    return nx.complete_graph(3)


@pytest.fixture
def linear_graph():
    """创建线性图（5 节点）"""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    return G


@pytest.fixture
def star_graph():
    """创建星型图（中心 + 4 节点）"""
    return nx.star_graph(4)


# ============================================================
# 临时目录相关
# ============================================================

@pytest.fixture
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def prompt_dir(temp_dir):
    """创建包含测试 prompt 模板的临时目录"""
    prompt_folder = os.path.join(temp_dir, 'prompts')
    os.makedirs(prompt_folder, exist_ok=True)
    
    # 创建测试模板
    templates = {
        'test.txt': '你是 {{ agent.description }}，请回答：{{ extra.question }}',
        'reflect.txt': '根据以下记忆进行反思：\n长期记忆：{{ extra.long_memory }}\n短期记忆：{{ extra.short_memory }}',
        'choice.txt': '请从以下选项中选择：{{ extra.options }}',
        'score.txt': '请对以下内容评分（0-10）：{{ extra.content }}',
        'json.txt': '请以 JSON 格式返回结果：{{ extra.request }}',
    }
    
    for filename, content in templates.items():
        filepath = os.path.join(prompt_folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return prompt_folder


@pytest.fixture
def memory_dir(temp_dir):
    """创建记忆存储临时目录"""
    memory_folder = os.path.join(temp_dir, 'memory')
    os.makedirs(memory_folder, exist_ok=True)
    return memory_folder


# ============================================================
# Agent 描述数据
# ============================================================

@pytest.fixture
def agent_descriptions():
    """提供测试用的 agent 描述"""
    return [
        {
            "general": "一位 35 岁的程序员",
            "character": "理性、逻辑思维强、喜欢技术",
            "issue": "关注开源软件和 AI 发展"
        },
        {
            "general": "一位 50 岁的教师",
            "character": "耐心、善于沟通、关心教育",
            "issue": "关注教育公平和学生成长"
        },
        {
            "general": "一位 28 岁的创业者",
            "character": "冒险精神、创新思维、行动力强",
            "issue": "关注市场机会和商业模式"
        }
    ]


# ============================================================
# Mock Model 和 Agent
# ============================================================

@pytest.fixture
def mock_model(mock_llm, simple_graph, prompt_dir):
    """创建 Mock Model 实例"""
    from casevo import ModelBase
    
    model = ModelBase(
        tar_graph=simple_graph,
        llm=mock_llm,
        prompt_path=prompt_dir
    )
    return model


@pytest.fixture
def mock_agent(mock_model, agent_descriptions):
    """创建 Mock Agent 实例"""
    from casevo import AgentBase
    
    class SampleAgent(AgentBase):
        def step(self):
            pass
    
    agent = SampleAgent(
        unique_id=0,
        model=mock_model,
        description=agent_descriptions[0],
        context={"test": "context"}
    )
    return agent


# ============================================================
# Prompt 相关
# ============================================================

@pytest.fixture
def mock_prompt(mock_llm, prompt_dir):
    """创建 Mock Prompt 实例"""
    from casevo import PromptFactory
    
    factory = PromptFactory(prompt_dir, mock_llm)
    return factory.get_template('test.txt')


# ============================================================
# 清理
# ============================================================

@pytest.fixture(autouse=True)
def cleanup_chromadb():
    """每个测试后清理 ChromaDB"""
    yield
    # ChromaDB 使用临时目录时会自动清理

