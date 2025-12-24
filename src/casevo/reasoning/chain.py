"""
思维链模块 (Chain of Thought)

实现基础的链式推理机制。
"""

import re
import json
import threading
import queue
import time

from casevo.core.component import BaseAgentComponent


class BaseStep:
    """CoT 步骤基类"""
    prompt = None
    step_id = None

    def __init__(self, step_id, tar_prompt):
        self.prompt = tar_prompt
        self.step_id = step_id
    
    def pre_process(self, input, agent=None, model=None):
        """对输入数据进行预处理。"""
        return input

    def action(self, input, agent=None, model=None):
        """根据输入和上下文执行特定动作。"""
        response = self.prompt.send_prompt(input, agent, model)
        return response
    
    def after_process(self, input, response, agent=None, model=None):
        """处理对话后的回调函数。"""
        return {
            'input': input,
            'last_response': response
        }
    
    def get_id(self):
        return self.step_id


class ChoiceStep(BaseStep):
    """选择步骤类，用于处理需要用户进行选择的交互步骤。"""

    def __init__(self, step_id, tar_prompt, choice_template=None):
        super().__init__(step_id, tar_prompt)
        if choice_template:
            self.answer_template = choice_template
        else:
            self.answer_template = re.compile(r"[A-Z]")
    
    def after_process(self, input, response, agent=None, model=None):
        """处理对话后的后续操作。"""
        match = self.answer_template.search(response)
        if not match:
            raise Exception("No choice found")
        else:
            return {
                'input': input,
                'choice': match.group()
            }


class ScoreStep(BaseStep):
    """评分判断类，用于根据给定的步骤生成评分回答。"""

    def __init__(self, step_id, tar_prompt, score_template=None):
        super().__init__(step_id, tar_prompt)
        if score_template:
            self.answer_template = score_template
        else:
            self.answer_template = re.compile(r"(-?\d+)(\.\d+)?")
    
    def after_process(self, input, response, agent=None, model=None):
        """处理对话代理的响应后，提取答案得分。"""
        match = self.answer_template.search(response)
        if not match:
            raise Exception("No score found")
        else:
            return {
                'input': input,
                'score': float(match.group())
            }


class JsonStep(BaseStep):
    """JSON 步骤类"""

    def __init__(self, step_id, tar_prompt, json_template=None):
        super().__init__(step_id, tar_prompt)
        if json_template:
            self.answer_template = json_template
        else:
            self.answer_template = re.compile(r"\{[\s\S]*\}")
    
    def after_process(self, input, response, agent=None, model=None):
        """处理对话代理的响应，提取并返回解析后的 JSON 数据。"""
        match = self.answer_template.search(response)
        if not match:
            raise Exception("No JSON found")
        else:
            cur_json = json.loads(match.group())
            return {
                'input': input,
                'json': cur_json
            }


class ToolStep(BaseStep):
    """工具调用步骤"""

    def __init__(self, step_id, tar_prompt, callback=None):
        super().__init__(step_id, tar_prompt)
        self.callback = callback
    
    def pre_process(self, input, agent=None, model=None):
        input['arguments'] = None
        return input
    
    def action(self, input, agent=None, model=None):
        response = self.callback(input['arguments'])
        return response


class ThoughtChain(BaseAgentComponent):
    """思维链"""
    steps = None
    status = None
    input_content = None
    step_history = None
    output_content = None 
    
    def __init__(self, agent, step_list):   
        """初始化链式操作对象。"""
        super().__init__(agent.component_id + "_chain", 'chain', agent)
        self.steps = step_list
        self.status = 'init'
    
    def set_input(self, input):
        """设置输入内容并更新状态。"""
        if self.status != 'init' and self.status != 'finish':
            raise Exception("set input error")
        else:
            self.input_content = input
            self.step_history = []
            self.status = 'ready'
        
    def run_step(self):
        """执行流程。"""
        if self.status != 'ready':
            raise Exception("running status error")
        
        self.status = 'running'
        last_input = self.input_content
        for item in self.steps:
            
            error_flag = True
            for i in range(3):
                try:
                    cur_input = item.pre_process(last_input, self.agent, self.agent.model)
                    response = item.action(cur_input, self.agent, self.agent.model)
                    cur_output = item.after_process(cur_input, response, self.agent, self.agent.model)
                    error_flag = False
                    break
                except Exception as e:
                    print(e)
                    print("Thought Chain Retry..... %d" % i)
            if error_flag:
                self.status = 'ready'
                raise Exception("Thought Chain Retry Failed")
            
            self.step_history.append({
                'id': item.get_id(),
                'input': cur_input,
                'output': cur_output
            })
            last_input = cur_output
        
        self.output_content = self.step_history[-1]['output']
        self.status = 'finish'
    
    def get_output(self):
        """获取输出内容。"""
        if self.status != 'finish':
            raise Exception("get output error")
        else:
            return self.output_content
    
    def get_history(self):
        """获取步骤历史记录。"""
        if self.status != 'finish':
            raise Exception("get history error")
        else:
            return self.step_history


class ChainPool:
    """链池，支持并行执行多个 ThoughtChain"""

    def __init__(self, thread_num=8):
        self.status = 'init' 
        self.chain_list = queue.Queue()
        self.threads_num = thread_num

    def add_chains(self, chains):
        for item in chains:
            self.chain_list.put(item)
        self.status = 'ready'

    def worker(self):
        while True:
            tar_chain = self.chain_list.get()
            tar_chain.run_step()
            self.chain_list.task_done()
            if self.chain_list.empty():
                break      
            time.sleep(5)
    
    def start_pool(self):
        if self.status != 'ready':
            raise Exception("start pool error")
        self.threads = []
        for _ in range(self.threads_num):
            t = threading.Thread(target=self.worker)
            t.start()
            self.threads.append(t)
        self.chain_list.join()

