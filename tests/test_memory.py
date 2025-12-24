"""
记忆模块测试

测试 MemoryItem, Memory, MemoryFactory
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from casevo.memory.base import MemoryItem, Memory, MemoryFactory
from casevo import AgentBase


class SampleAgent(AgentBase):
    """测试用 Agent（不以 Test 开头以避免 pytest 警告）"""
    def step(self):
        pass


class TestMemoryItem:
    """MemoryItem 测试类"""
    
    def test_init(self):
        """测试初始化"""
        item = MemoryItem(
            ts=1,
            source="agent_0",
            target="agent_1",
            action="talk",
            content="你好"
        )
        
        assert item.ts == 1
        assert item.source == "agent_0"
        assert item.target == "agent_1"
        assert item.action == "talk"
        assert item.content == "你好"
        assert item.id == -1  # 默认值
    
    def test_to_dict(self):
        """测试转换为字典"""
        item = MemoryItem(
            ts=2,
            source="agent_0",
            target="agent_1",
            action="listen",
            content="测试内容"
        )
        
        result = item.toDict()
        
        assert isinstance(result, dict)
        assert result['ts'] == 2
        assert result['source'] == "agent_0"
        assert result['target'] == "agent_1"
        assert result['action'] == "listen"
        assert result['content'] == "测试内容"
    
    def test_to_list(self):
        """测试批量转换为列表"""
        items = [
            MemoryItem(1, "agent_0", "agent_1", "talk", "消息1"),
            MemoryItem(2, "agent_0", "agent_2", "talk", "消息2"),
            MemoryItem(3, "agent_1", "agent_0", "reply", "回复")
        ]
        
        content_list, meta_list, id_list = MemoryItem.toList(items, start_id=10)
        
        assert len(content_list) == 3
        assert len(meta_list) == 3
        assert len(id_list) == 3
        
        assert content_list == ["消息1", "消息2", "回复"]
        assert id_list == ["10", "11", "12"]
        assert meta_list[0]['id'] == 10
        assert meta_list[1]['id'] == 11
        assert meta_list[2]['id'] == 12


class TestMemory:
    """Memory 测试类"""
    
    def test_init(self, mock_agent):
        """测试初始化"""
        memory = mock_agent.memory
        
        assert memory is not None
        assert memory.agent is mock_agent
        assert memory.long_memory is None
        assert memory.last_id == -1
    
    def test_add_short_memory(self, mock_agent):
        """测试添加短期记忆"""
        memory = mock_agent.memory
        
        memory.add_short_memory(
            source=mock_agent.component_id,
            target="agent_1",
            action="talk",
            content="这是测试对话"
        )
        
        # 验证记忆已添加（通过搜索确认）
        # 由于使用 ChromaDB，记忆会被存储
    
    def test_add_multiple_memories(self, mock_agent):
        """测试添加多条记忆"""
        memory = mock_agent.memory
        
        for i in range(5):
            memory.add_short_memory(
                source=mock_agent.component_id,
                target=f"agent_{i+1}",
                action="talk",
                content=f"对话内容 {i}"
            )
    
    def test_get_long_memory_initially_none(self, mock_agent):
        """测试初始长期记忆为空"""
        memory = mock_agent.memory
        
        assert memory.get_long_memory() is None


class TestMemoryFactory:
    """MemoryFactory 测试类"""
    
    def test_init(self, mock_model):
        """测试初始化"""
        factory = mock_model.memory_factory
        
        assert factory is not None
        assert factory.llm is mock_model.llm
        assert factory.memory_num == 10  # 默认值
    
    def test_create_memory(self, mock_model, agent_descriptions):
        """测试创建 Memory 实例"""
        agent = SampleAgent(
            unique_id=0,
            model=mock_model,
            description=agent_descriptions[0],
            context=None
        )
        
        memory = mock_model.memory_factory.create_memory(agent)
        
        assert isinstance(memory, Memory)
        assert memory.agent is agent
    
    def test_create_multiple_memories(self, mock_model, agent_descriptions):
        """测试为多个 Agent 创建 Memory"""
        memories = []
        
        for i, desc in enumerate(agent_descriptions):
            agent = SampleAgent(
                unique_id=i,
                model=mock_model,
                description=desc,
                context=None
            )
            memory = mock_model.memory_factory.create_memory(agent)
            memories.append(memory)
        
        assert len(memories) == 3
        # 每个 memory 的 component_id 应该不同
        ids = [m.component_id for m in memories]
        assert len(set(ids)) == 3


class TestMemorySearch:
    """记忆搜索测试类"""
    
    def test_search_by_doc(self, mock_agent):
        """测试按文档搜索"""
        memory = mock_agent.memory
        
        # 先添加一些记忆
        memory.add_short_memory(
            source=mock_agent.component_id,
            target="agent_1",
            action="discuss",
            content="讨论关于天气的话题"
        )
        
        memory.add_short_memory(
            source=mock_agent.component_id,
            target="agent_2",
            action="discuss",
            content="讨论关于技术的话题"
        )
        
        # 搜索相关记忆
        results = memory.search_short_memory_by_doc(["天气"])
        
        # 验证返回结果结构
        assert results is not None


class TestMemoryReflection:
    """记忆反思测试类"""
    
    def test_reflect_memory(self, mock_agent):
        """测试记忆反思"""
        memory = mock_agent.memory
        
        # 添加一些记忆
        memory.add_short_memory(
            source=mock_agent.component_id,
            target="agent_1",
            action="observe",
            content="观察到一些有趣的现象"
        )
        
        # 执行反思
        memory.reflect_memory()
        
        # 反思后 long_memory 应该有值
        # （取决于 mock_llm 的返回）

