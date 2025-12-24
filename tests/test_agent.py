"""
Agent 基类测试

测试 AgentBase 及其功能
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from casevo import AgentBase, ThoughtChain, BaseStep


class ConcreteAgent(AgentBase):
    """用于测试的具体 Agent 实现"""
    
    def __init__(self, unique_id, model, description, context):
        super().__init__(unique_id, model, description, context)
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        return self.step_count


class TestAgentBase:
    """AgentBase 测试类"""
    
    def test_init(self, mock_model, agent_descriptions):
        """测试 Agent 初始化"""
        agent = ConcreteAgent(
            unique_id=0,
            model=mock_model,
            description=agent_descriptions[0],
            context={"test": "context"}
        )
        
        assert agent.unique_id == 0
        assert agent.component_id == "agent_0"
        assert agent.description == agent_descriptions[0]
        assert agent.context == {"test": "context"}
        assert agent.memory is not None
    
    def test_multiple_agents(self, mock_model, agent_descriptions):
        """测试创建多个 Agent"""
        agents = []
        for i, desc in enumerate(agent_descriptions):
            agent = ConcreteAgent(
                unique_id=i,
                model=mock_model,
                description=desc,
                context=None
            )
            agents.append(agent)
        
        assert len(agents) == 3
        assert agents[0].component_id == "agent_0"
        assert agents[1].component_id == "agent_1"
        assert agents[2].component_id == "agent_2"
    
    def test_step_method(self, mock_model, agent_descriptions):
        """测试 step 方法"""
        agent = ConcreteAgent(
            unique_id=0,
            model=mock_model,
            description=agent_descriptions[0],
            context=None
        )
        
        assert agent.step_count == 0
        agent.step()
        assert agent.step_count == 1
        agent.step()
        assert agent.step_count == 2
    
    def test_setup_chain(self, mock_model, agent_descriptions, mock_prompt):
        """测试设置思维链"""
        agent = ConcreteAgent(
            unique_id=0,
            model=mock_model,
            description=agent_descriptions[0],
            context=None
        )
        
        # 创建测试步骤
        step1 = BaseStep(0, mock_prompt)
        step2 = BaseStep(1, mock_prompt)
        
        chain_dict = {
            'think': [step1],
            'act': [step1, step2]
        }
        
        agent.setup_chain(chain_dict)
        
        assert 'think' in agent.chains
        assert 'act' in agent.chains
        assert isinstance(agent.chains['think'], ThoughtChain)
        assert isinstance(agent.chains['act'], ThoughtChain)
    
    def test_memory_integration(self, mock_model, agent_descriptions):
        """测试 Agent 与 Memory 集成"""
        agent = ConcreteAgent(
            unique_id=0,
            model=mock_model,
            description=agent_descriptions[0],
            context=None
        )
        
        # Agent 应该有 memory 属性
        assert hasattr(agent, 'memory')
        assert agent.memory is not None
        
        # 测试添加记忆
        agent.memory.add_short_memory(
            source=agent.component_id,
            target="agent_1",
            action="talk",
            content="测试对话内容"
        )
    
    def test_agent_model_reference(self, mock_model, agent_descriptions):
        """测试 Agent 对 Model 的引用"""
        agent = ConcreteAgent(
            unique_id=0,
            model=mock_model,
            description=agent_descriptions[0],
            context=None
        )
        
        assert agent.model is mock_model
        assert agent.model.llm is not None


class TestAbstractAgentBase:
    """测试 AgentBase 抽象性"""
    
    def test_step_is_abstract_method(self):
        """测试 step 方法被标记为抽象方法"""
        import inspect
        from abc import abstractmethod
        
        # 检查 step 方法是否有 abstractmethod 装饰器
        assert hasattr(AgentBase.step, '__isabstractmethod__')
        assert AgentBase.step.__isabstractmethod__ is True

