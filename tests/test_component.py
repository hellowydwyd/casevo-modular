"""
组件基类测试

测试 BaseComponent, BaseAgentComponent, BaseModelComponent
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from casevo.core.component import (
    BaseComponent,
    BaseAgentComponent,
    BaseModelComponent
)


class TestBaseComponent:
    """BaseComponent 测试类"""
    
    def test_init(self):
        """测试初始化"""
        component = BaseComponent(
            component_id="test_comp_1",
            coponent_type="test_type",
            tar_context={"key": "value"}
        )
        
        assert component.component_id == "test_comp_1"
        assert component.componet_type == "test_type"
        assert component.context == {"key": "value"}
    
    def test_init_with_none_context(self):
        """测试空上下文初始化"""
        component = BaseComponent(
            component_id="test_comp_2",
            coponent_type="test_type",
            tar_context=None
        )
        
        assert component.context is None
    
    def test_component_id_uniqueness(self):
        """测试组件 ID 可以不同"""
        comp1 = BaseComponent("id_1", "type", None)
        comp2 = BaseComponent("id_2", "type", None)
        
        assert comp1.component_id != comp2.component_id


class TestBaseAgentComponent:
    """BaseAgentComponent 测试类"""
    
    def test_init_with_agent(self, mock_agent):
        """测试使用 agent 初始化"""
        component = BaseAgentComponent(
            component_id="agent_comp_1",
            coponent_type="agent_component",
            agent=mock_agent
        )
        
        assert component.component_id == "agent_comp_1"
        assert component.agent is mock_agent
        assert component.context == mock_agent.context
    
    def test_agent_reference(self, mock_agent):
        """测试 agent 引用正确"""
        component = BaseAgentComponent(
            component_id="agent_comp_2",
            coponent_type="test",
            agent=mock_agent
        )
        
        # 可以通过组件访问 agent 属性
        assert component.agent.unique_id == mock_agent.unique_id
        assert component.agent.description == mock_agent.description


class TestBaseModelComponent:
    """BaseModelComponent 测试类"""
    
    def test_init_with_model(self, mock_model):
        """测试使用 model 初始化"""
        component = BaseModelComponent(
            component_id="model_comp_1",
            coponent_type="model_component",
            model=mock_model
        )
        
        assert component.component_id == "model_comp_1"
        assert component.model is mock_model
        assert component.context == mock_model.context
    
    def test_model_reference(self, mock_model):
        """测试 model 引用正确"""
        component = BaseModelComponent(
            component_id="model_comp_2",
            coponent_type="test",
            model=mock_model
        )
        
        # 可以通过组件访问 model 属性
        assert component.model.llm is not None
        assert component.model.grid is not None







