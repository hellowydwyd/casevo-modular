"""
思维链模块测试

测试 BaseStep, ChoiceStep, ScoreStep, JsonStep, ThoughtChain, ChainPool
"""

import pytest
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from casevo.reasoning.chain import (
    BaseStep,
    ChoiceStep,
    ScoreStep,
    JsonStep,
    ToolStep,
    ThoughtChain,
    ChainPool
)
from casevo import AgentBase


class SampleAgent(AgentBase):
    """测试用 Agent（不以 Test 开头以避免 pytest 警告）"""
    def step(self):
        pass


class TestBaseStep:
    """BaseStep 测试类"""
    
    def test_init(self, mock_prompt):
        """测试初始化"""
        step = BaseStep(step_id=0, tar_prompt=mock_prompt)
        
        assert step.step_id == 0
        assert step.prompt is mock_prompt
    
    def test_get_id(self, mock_prompt):
        """测试获取 ID"""
        step = BaseStep(step_id=5, tar_prompt=mock_prompt)
        
        assert step.get_id() == 5
    
    def test_pre_process(self, mock_prompt):
        """测试预处理"""
        step = BaseStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"question": "测试问题"}
        
        result = step.pre_process(input_data)
        
        # 默认实现直接返回输入
        assert result == input_data
    
    def test_action(self, mock_prompt, mock_agent):
        """测试执行动作"""
        step = BaseStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"question": "测试问题"}
        
        result = step.action(input_data, mock_agent, mock_agent.model)
        
        # Mock LLM 应该返回默认响应
        assert result is not None
    
    def test_after_process(self, mock_prompt):
        """测试后处理"""
        step = BaseStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"question": "测试问题"}
        response = "这是回答"
        
        result = step.after_process(input_data, response)
        
        assert result['input'] == input_data
        assert result['last_response'] == response


class TestChoiceStep:
    """ChoiceStep 测试类"""
    
    def test_init(self, mock_prompt):
        """测试初始化"""
        step = ChoiceStep(step_id=0, tar_prompt=mock_prompt)
        
        assert step.step_id == 0
        assert step.answer_template is not None
    
    def test_init_with_custom_template(self, mock_prompt):
        """测试使用自定义模板初始化"""
        custom_pattern = re.compile(r"[1-5]")
        step = ChoiceStep(step_id=0, tar_prompt=mock_prompt, choice_template=custom_pattern)
        
        assert step.answer_template == custom_pattern
    
    def test_after_process_valid_choice(self, mock_prompt):
        """测试有效选择的后处理"""
        step = ChoiceStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"options": ["A", "B", "C"]}
        response = "我选择 B 选项"
        
        result = step.after_process(input_data, response)
        
        assert result['choice'] == 'B'
    
    def test_after_process_no_choice(self, mock_prompt):
        """测试无效选择的后处理"""
        step = ChoiceStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"options": ["A", "B", "C"]}
        response = "我不确定"
        
        with pytest.raises(Exception, match="No choice found"):
            step.after_process(input_data, response)


class TestScoreStep:
    """ScoreStep 测试类"""
    
    def test_init(self, mock_prompt):
        """测试初始化"""
        step = ScoreStep(step_id=0, tar_prompt=mock_prompt)
        
        assert step.step_id == 0
        assert step.answer_template is not None
    
    def test_after_process_valid_score(self, mock_prompt):
        """测试有效评分的后处理"""
        step = ScoreStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"content": "测试内容"}
        response = "我给这个内容打 8.5 分"
        
        result = step.after_process(input_data, response)
        
        assert result['score'] == 8.5
    
    def test_after_process_integer_score(self, mock_prompt):
        """测试整数评分的后处理"""
        step = ScoreStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"content": "测试内容"}
        response = "评分：7 分"
        
        result = step.after_process(input_data, response)
        
        assert result['score'] == 7.0
    
    def test_after_process_no_score(self, mock_prompt):
        """测试无评分的后处理"""
        step = ScoreStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"content": "测试内容"}
        response = "无法评分"
        
        with pytest.raises(Exception, match="No score found"):
            step.after_process(input_data, response)


class TestJsonStep:
    """JsonStep 测试类"""
    
    def test_init(self, mock_prompt):
        """测试初始化"""
        step = JsonStep(step_id=0, tar_prompt=mock_prompt)
        
        assert step.step_id == 0
        assert step.answer_template is not None
    
    def test_after_process_valid_json(self, mock_prompt):
        """测试有效 JSON 的后处理"""
        step = JsonStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"request": "返回用户信息"}
        response = '这是用户信息：{"name": "张三", "age": 25}'
        
        result = step.after_process(input_data, response)
        
        assert result['json'] == {"name": "张三", "age": 25}
    
    def test_after_process_nested_json(self, mock_prompt):
        """测试嵌套 JSON 的后处理"""
        step = JsonStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"request": "返回配置"}
        response = '{"config": {"debug": true, "level": 2}}'
        
        result = step.after_process(input_data, response)
        
        assert result['json']['config']['debug'] is True
        assert result['json']['config']['level'] == 2
    
    def test_after_process_no_json(self, mock_prompt):
        """测试无 JSON 的后处理"""
        step = JsonStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"request": "返回配置"}
        response = "没有 JSON 内容"
        
        with pytest.raises(Exception, match="No JSON found"):
            step.after_process(input_data, response)


class TestToolStep:
    """ToolStep 测试类"""
    
    def test_init(self, mock_prompt):
        """测试初始化"""
        callback = lambda x: f"处理结果: {x}"
        step = ToolStep(step_id=0, tar_prompt=mock_prompt, callback=callback)
        
        assert step.step_id == 0
        assert step.callback is callback
    
    def test_pre_process(self, mock_prompt):
        """测试预处理添加 arguments"""
        step = ToolStep(step_id=0, tar_prompt=mock_prompt)
        input_data = {"query": "测试"}
        
        result = step.pre_process(input_data)
        
        assert 'arguments' in result
        assert result['arguments'] is None
    
    def test_action(self, mock_prompt):
        """测试执行回调"""
        callback = lambda x: f"结果: {x}"
        step = ToolStep(step_id=0, tar_prompt=mock_prompt, callback=callback)
        input_data = {"arguments": "参数值"}
        
        result = step.action(input_data)
        
        assert result == "结果: 参数值"


class TestThoughtChain:
    """ThoughtChain 测试类"""
    
    def test_init(self, mock_agent, mock_prompt):
        """测试初始化"""
        steps = [BaseStep(0, mock_prompt), BaseStep(1, mock_prompt)]
        chain = ThoughtChain(mock_agent, steps)
        
        assert chain.status == 'init'
        assert chain.steps == steps
        assert chain.agent is mock_agent
    
    def test_set_input(self, mock_agent, mock_prompt):
        """测试设置输入"""
        steps = [BaseStep(0, mock_prompt)]
        chain = ThoughtChain(mock_agent, steps)
        
        chain.set_input({"question": "测试问题"})
        
        assert chain.status == 'ready'
        assert chain.input_content == {"question": "测试问题"}
    
    def test_set_input_error_on_running(self, mock_agent, mock_prompt):
        """测试运行中设置输入报错"""
        steps = [BaseStep(0, mock_prompt)]
        chain = ThoughtChain(mock_agent, steps)
        chain.status = 'running'
        
        with pytest.raises(Exception, match="set input error"):
            chain.set_input({"question": "测试"})
    
    def test_run_step(self, mock_agent, mock_prompt):
        """测试执行步骤"""
        steps = [BaseStep(0, mock_prompt)]
        chain = ThoughtChain(mock_agent, steps)
        
        chain.set_input({"question": "测试问题"})
        chain.run_step()
        
        assert chain.status == 'finish'
    
    def test_run_step_error_on_not_ready(self, mock_agent, mock_prompt):
        """测试未就绪时运行报错"""
        steps = [BaseStep(0, mock_prompt)]
        chain = ThoughtChain(mock_agent, steps)
        
        with pytest.raises(Exception, match="running status error"):
            chain.run_step()
    
    def test_get_output(self, mock_agent, mock_prompt):
        """测试获取输出"""
        steps = [BaseStep(0, mock_prompt)]
        chain = ThoughtChain(mock_agent, steps)
        
        chain.set_input({"question": "测试问题"})
        chain.run_step()
        
        output = chain.get_output()
        
        assert 'input' in output
        assert 'last_response' in output
    
    def test_get_output_error_on_not_finish(self, mock_agent, mock_prompt):
        """测试未完成时获取输出报错"""
        steps = [BaseStep(0, mock_prompt)]
        chain = ThoughtChain(mock_agent, steps)
        
        with pytest.raises(Exception, match="get output error"):
            chain.get_output()
    
    def test_get_history(self, mock_agent, mock_prompt):
        """测试获取历史"""
        steps = [BaseStep(0, mock_prompt), BaseStep(1, mock_prompt)]
        chain = ThoughtChain(mock_agent, steps)
        
        chain.set_input({"question": "测试问题"})
        chain.run_step()
        
        history = chain.get_history()
        
        assert len(history) == 2
        assert history[0]['id'] == 0
        assert history[1]['id'] == 1


class TestChainPool:
    """ChainPool 测试类"""
    
    def test_init(self):
        """测试初始化"""
        pool = ChainPool(thread_num=4)
        
        assert pool.status == 'init'
        assert pool.threads_num == 4
    
    def test_add_chains(self, mock_agent, mock_prompt):
        """测试添加链"""
        pool = ChainPool()
        
        chains = []
        for i in range(3):
            steps = [BaseStep(i, mock_prompt)]
            chain = ThoughtChain(mock_agent, steps)
            chain.set_input({"index": i})
            chains.append(chain)
        
        pool.add_chains(chains)
        
        assert pool.status == 'ready'

