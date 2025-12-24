"""
树状思维模块测试

测试 ToTNode, ToTStep, EvaluatorStep, TreeOfThought, SearchStrategy
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from casevo.reasoning.tree import (
    ToTNode,
    ToTStep,
    EvaluatorStep,
    TreeOfThought,
    SearchStrategy,
    AdaptiveToT
)
from casevo import AgentBase


class SampleAgent(AgentBase):
    """测试用 Agent（不以 Test 开头以避免 pytest 警告）"""
    def step(self):
        pass


class TestToTNode:
    """ToTNode 测试类"""
    
    def test_init(self):
        """测试初始化"""
        node = ToTNode(
            node_id=0,
            state={"problem": "测试问题"}
        )
        
        assert node.node_id == 0
        assert node.state == {"problem": "测试问题"}
        assert node.score == 0.0
        assert node.depth == 0
        assert node.parent is None
        assert node.children == []
        assert node.is_terminal is False
        assert node.is_pruned is False
    
    def test_node_comparison(self):
        """测试节点比较（按分数）"""
        node1 = ToTNode(node_id=0, state={}, score=0.5)
        node2 = ToTNode(node_id=1, state={}, score=0.8)
        
        # 分数高的节点优先
        assert node2 < node1
    
    def test_get_path_to_root(self):
        """测试获取到根节点的路径"""
        root = ToTNode(node_id=0, state={"level": 0})
        child1 = ToTNode(node_id=1, state={"level": 1}, depth=1, parent=root)
        child2 = ToTNode(node_id=2, state={"level": 2}, depth=2, parent=child1)
        
        root.children.append(child1)
        child1.children.append(child2)
        
        path = child2.get_path_to_root()
        
        assert len(path) == 3
        assert path[0].node_id == 0
        assert path[1].node_id == 1
        assert path[2].node_id == 2
    
    def test_get_full_reasoning(self):
        """测试获取完整推理路径"""
        root = ToTNode(node_id=0, state={}, reasoning_path="初始状态")
        child1 = ToTNode(node_id=1, state={}, depth=1, parent=root, reasoning_path="第一步推理")
        child2 = ToTNode(node_id=2, state={}, depth=2, parent=child1, reasoning_path="第二步推理")
        
        root.children.append(child1)
        child1.children.append(child2)
        
        reasoning = child2.get_full_reasoning()
        
        assert "初始状态" in reasoning
        assert "第一步推理" in reasoning
        assert "第二步推理" in reasoning


class TestToTStep:
    """ToTStep 测试类"""
    
    def test_init(self, mock_prompt):
        """测试初始化"""
        step = ToTStep(step_id="tot_step_1", tar_prompt=mock_prompt, num_branches=3)
        
        assert step.step_id == "tot_step_1"
        assert step.num_branches == 3
    
    def test_generate_branches(self, mock_prompt, mock_agent):
        """测试生成分支"""
        step = ToTStep(step_id="tot_step", tar_prompt=mock_prompt, num_branches=2)
        
        input_state = {"problem": "如何解决这个问题？"}
        branches = step.generate_branches(input_state, mock_agent, mock_agent.model)
        
        # 应该生成指定数量的分支
        assert len(branches) <= 2


class TestEvaluatorStep:
    """EvaluatorStep 测试类"""
    
    def test_init(self, mock_prompt):
        """测试初始化"""
        step = EvaluatorStep(step_id="eval_step", tar_prompt=mock_prompt)
        
        assert step.step_id == "eval_step"
        assert step.score_range == (0.0, 1.0)
    
    def test_init_custom_range(self, mock_prompt):
        """测试自定义评分范围"""
        step = EvaluatorStep(step_id="eval_step", tar_prompt=mock_prompt, score_range=(0, 10))
        
        assert step.score_range == (0, 10)
    
    def test_extract_score_from_response(self, mock_prompt):
        """测试从响应中提取分数"""
        step = EvaluatorStep(step_id="eval_step", tar_prompt=mock_prompt)
        
        # 测试各种格式
        assert step._extract_score("score: 0.75") == 0.75
        assert step._extract_score("评分: 0.8") == 0.8
        assert step._extract_score("7/10 分") == 7.0
        assert step._extract_score("这个方案的得分是 0.9") == 0.9
    
    def test_extract_score_no_match(self, mock_prompt):
        """测试无法提取分数时返回最小值"""
        step = EvaluatorStep(step_id="eval_step", tar_prompt=mock_prompt)
        
        result = step._extract_score("没有数字的响应")
        assert result == 0.0  # 返回最小值


class TestSearchStrategy:
    """SearchStrategy 测试类"""
    
    def test_strategy_values(self):
        """测试策略枚举值"""
        assert SearchStrategy.BFS.value == "breadth_first"
        assert SearchStrategy.DFS.value == "depth_first"
        assert SearchStrategy.BEAM.value == "beam_search"
        assert SearchStrategy.BEST_FIRST.value == "best_first"


class TestTreeOfThought:
    """TreeOfThought 测试类"""
    
    def test_init(self, mock_agent, mock_prompt):
        """测试初始化"""
        thought_step = ToTStep("tot", mock_prompt)
        
        tot = TreeOfThought(
            agent=mock_agent,
            thought_step=thought_step,
            max_depth=3,
            beam_width=2
        )
        
        assert tot.agent is mock_agent
        assert tot.thought_step is thought_step
        assert tot.max_depth == 3
        assert tot.beam_width == 2
        assert tot.status == 'init'
    
    def test_set_input(self, mock_agent, mock_prompt):
        """测试设置输入"""
        thought_step = ToTStep("tot", mock_prompt)
        tot = TreeOfThought(mock_agent, thought_step)
        
        initial_state = {"problem": "测试问题"}
        tot.set_input(initial_state)
        
        assert tot.status == 'ready'
        assert tot.initial_state == initial_state
        assert tot.node_counter == 0
        assert tot.all_nodes == []
    
    def test_set_input_error_on_running(self, mock_agent, mock_prompt):
        """测试运行中设置输入报错"""
        thought_step = ToTStep("tot", mock_prompt)
        tot = TreeOfThought(mock_agent, thought_step)
        tot.status = 'running'
        
        with pytest.raises(Exception, match="无法设置输入"):
            tot.set_input({"problem": "测试"})
    
    def test_create_node(self, mock_agent, mock_prompt):
        """测试创建节点"""
        thought_step = ToTStep("tot", mock_prompt)
        tot = TreeOfThought(mock_agent, thought_step)
        
        node = tot._create_node({"state": "test"}, reasoning="测试推理")
        
        assert node.node_id == 0
        assert node.state == {"state": "test"}
        assert node.reasoning_path == "测试推理"
        assert node in tot.all_nodes
    
    def test_create_child_node(self, mock_agent, mock_prompt):
        """测试创建子节点"""
        thought_step = ToTStep("tot", mock_prompt)
        tot = TreeOfThought(mock_agent, thought_step)
        
        parent = tot._create_node({"level": 0})
        child = tot._create_node({"level": 1}, parent=parent)
        
        assert child.parent is parent
        assert child.depth == 1
        assert child in parent.children
    
    def test_should_prune(self, mock_agent, mock_prompt):
        """测试剪枝判断"""
        thought_step = ToTStep("tot", mock_prompt)
        tot = TreeOfThought(mock_agent, thought_step, pruning_threshold=0.5)
        
        low_score_node = ToTNode(node_id=0, state={}, score=0.3)
        high_score_node = ToTNode(node_id=1, state={}, score=0.7)
        
        assert tot._should_prune(low_score_node) is True
        assert tot._should_prune(high_score_node) is False
    
    def test_is_terminal(self, mock_agent, mock_prompt):
        """测试终止判断"""
        thought_step = ToTStep("tot", mock_prompt)
        tot = TreeOfThought(mock_agent, thought_step, max_depth=3)
        
        normal_node = ToTNode(node_id=0, state={}, depth=1)
        deep_node = ToTNode(node_id=1, state={}, depth=3)
        final_node = ToTNode(node_id=2, state={"is_final": True}, depth=1)
        
        assert tot._is_terminal(normal_node) is False
        assert tot._is_terminal(deep_node) is True
        assert tot._is_terminal(final_node) is True
    
    def test_get_output_error_on_not_finish(self, mock_agent, mock_prompt):
        """测试未完成时获取输出报错"""
        thought_step = ToTStep("tot", mock_prompt)
        tot = TreeOfThought(mock_agent, thought_step)
        
        with pytest.raises(Exception, match="无法获取输出"):
            tot.get_output()
    
    def test_backtrack_to(self, mock_agent, mock_prompt):
        """测试回溯功能"""
        thought_step = ToTStep("tot", mock_prompt)
        tot = TreeOfThought(mock_agent, thought_step)
        
        root = tot._create_node({"level": 0})
        child1 = tot._create_node({"level": 1}, parent=root)
        child2 = tot._create_node({"level": 2}, parent=child1)
        
        # 回溯到 child1
        result = tot.backtrack_to(child1)
        
        assert result is True
        assert len(child1.children) == 0
        assert child2.is_pruned is True
    
    def test_backtrack_to_invalid_node(self, mock_agent, mock_prompt):
        """测试回溯到无效节点"""
        thought_step = ToTStep("tot", mock_prompt)
        tot = TreeOfThought(mock_agent, thought_step)
        
        fake_node = ToTNode(node_id=999, state={})
        result = tot.backtrack_to(fake_node)
        
        assert result is False


class TestAdaptiveToT:
    """AdaptiveToT 测试类"""
    
    def test_init(self, mock_agent, mock_prompt):
        """测试初始化"""
        thought_step = ToTStep("tot", mock_prompt)
        
        adaptive_tot = AdaptiveToT(mock_agent, thought_step)
        
        assert adaptive_tot.complexity_estimator is not None
    
    def test_default_complexity(self, mock_agent, mock_prompt):
        """测试默认复杂度估计"""
        thought_step = ToTStep("tot", mock_prompt)
        adaptive_tot = AdaptiveToT(mock_agent, thought_step)
        
        simple_state = {"x": 1}
        complex_state = {"content": "a" * 500}
        very_complex_state = {"content": "a" * 2000}
        
        assert adaptive_tot._default_complexity(simple_state) < 0.3
        assert 0.3 <= adaptive_tot._default_complexity(complex_state) < 0.7
        assert adaptive_tot._default_complexity(very_complex_state) == 1.0
    
    def test_adaptive_parameters(self, mock_agent, mock_prompt):
        """测试自适应参数调整"""
        thought_step = ToTStep("tot", mock_prompt)
        adaptive_tot = AdaptiveToT(mock_agent, thought_step)
        
        # 简单问题
        adaptive_tot.set_input({"x": 1})
        assert adaptive_tot.max_depth == 3
        assert adaptive_tot.search_strategy == SearchStrategy.DFS
        
        # 重置
        adaptive_tot.status = 'init'
        adaptive_tot.node_counter = 0
        adaptive_tot.all_nodes = []
        
        # 复杂问题
        adaptive_tot.set_input({"content": "a" * 2000})
        assert adaptive_tot.max_depth == 7
        assert adaptive_tot.search_strategy == SearchStrategy.BEST_FIRST

