"""
Model 基类测试

测试 ModelBase, VariableNetwork, OrderTypeActivation
"""

import pytest
import sys
import os
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from casevo import ModelBase, AgentBase, VariableNetwork, OrderTypeActivation


class SampleAgent(AgentBase):
    """测试用 Agent（不以 Test 开头以避免 pytest 警告）"""
    def step(self):
        pass


class TestModelBase:
    """ModelBase 测试类"""
    
    def test_init(self, mock_llm, simple_graph, prompt_dir):
        """测试 Model 初始化"""
        model = ModelBase(
            tar_graph=simple_graph,
            llm=mock_llm,
            prompt_path=prompt_dir
        )
        
        assert model.llm is mock_llm
        assert model.grid is not None
        assert model.schedule is not None
        assert model.prompt_factory is not None
        assert model.memory_factory is not None
        assert model.agent_list == []
    
    def test_init_with_context(self, mock_llm, simple_graph, prompt_dir):
        """测试带上下文的初始化"""
        context = {"scenario": "election", "round": 1}
        
        model = ModelBase(
            tar_graph=simple_graph,
            llm=mock_llm,
            context=context,
            prompt_path=prompt_dir
        )
        
        assert model.context == context
    
    def test_add_agent(self, mock_llm, simple_graph, prompt_dir, agent_descriptions):
        """测试添加 Agent"""
        model = ModelBase(
            tar_graph=simple_graph,
            llm=mock_llm,
            prompt_path=prompt_dir
        )
        
        agent = SampleAgent(
            unique_id=0,
            model=model,
            description=agent_descriptions[0],
            context=None
        )
        
        model.add_agent(agent, 0)
        
        assert len(model.agent_list) == 1
        assert model.agent_list[0] is agent
    
    def test_add_multiple_agents(self, mock_llm, simple_graph, prompt_dir, agent_descriptions):
        """测试添加多个 Agent"""
        model = ModelBase(
            tar_graph=simple_graph,
            llm=mock_llm,
            prompt_path=prompt_dir
        )
        
        for i, desc in enumerate(agent_descriptions):
            agent = SampleAgent(
                unique_id=i,
                model=model,
                description=desc,
                context=None
            )
            model.add_agent(agent, i)
        
        assert len(model.agent_list) == 3
    
    def test_step(self, mock_llm, simple_graph, prompt_dir, agent_descriptions):
        """测试 step 方法"""
        model = ModelBase(
            tar_graph=simple_graph,
            llm=mock_llm,
            prompt_path=prompt_dir
        )
        
        # 添加 agent
        for i, desc in enumerate(agent_descriptions):
            agent = SampleAgent(
                unique_id=i,
                model=model,
                description=desc,
                context=None
            )
            model.add_agent(agent, i)
        
        # 执行 step
        result = model.step()
        assert result == 0
    
    def test_type_schedule(self, mock_llm, simple_graph, prompt_dir):
        """测试类型调度器"""
        model = ModelBase(
            tar_graph=simple_graph,
            llm=mock_llm,
            prompt_path=prompt_dir,
            type_schedule=True
        )
        
        assert isinstance(model.schedule, OrderTypeActivation)


class TestVariableNetwork:
    """VariableNetwork 测试类"""
    
    def test_add_edge(self, simple_graph):
        """测试添加边"""
        network = VariableNetwork(simple_graph)
        
        # 创建新图并添加节点
        new_graph = nx.Graph()
        new_graph.add_nodes_from([0, 1, 2, 3])
        network = VariableNetwork(new_graph)
        
        # 添加边
        network.add_edge(0, 3)
        assert network.G.has_edge(0, 3)
    
    def test_del_edge(self, simple_graph):
        """测试删除边"""
        network = VariableNetwork(simple_graph)
        
        # 确认边存在
        assert network.G.has_edge(0, 1)
        
        # 删除边
        network.del_edge(0, 1)
        assert not network.G.has_edge(0, 1)


class TestOrderTypeActivation:
    """OrderTypeActivation 测试类"""
    
    def test_add_timestamp(self, mock_llm, simple_graph, prompt_dir):
        """测试时间戳增加"""
        model = ModelBase(
            tar_graph=simple_graph,
            llm=mock_llm,
            prompt_path=prompt_dir,
            type_schedule=True
        )
        
        initial_time = model.schedule.time
        model.schedule.add_timestamp()
        
        assert model.schedule.time == initial_time + 1

