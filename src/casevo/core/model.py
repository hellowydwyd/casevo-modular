"""
Model 基类定义

模拟模型的基础类。
"""

import mesa

from casevo.memory.base import MemoryFactory
from casevo.prompt import PromptFactory


class OrderTypeActivation(mesa.time.RandomActivationByType):
    """按类型顺序激活的调度器"""
    
    def add_timestamp(self):
        self.time += 1
        self.steps += 1


class VariableNetwork(mesa.space.NetworkGrid):
    """可变网络结构"""
    
    def del_edge(self, source, target):
        self.G.remove_edge(source, target)
    
    def add_edge(self, source, target):
        self.G.add_edge(source, target)


class ModelBase(mesa.Model):
    """模型定义基类"""

    def __init__(self, tar_graph, llm, context=None, prompt_path='./prompt/', 
                 memory_path=None, memory_num=10, reflect_file='reflect.txt', 
                 type_schedule=False):
        """
        初始化模型。

        参数:
            tar_graph: 网络图结构。
            llm: 语言模型实例。
            context: 上下文信息。
            prompt_path: Prompt 模板路径。
            memory_path: 记忆存储路径。
            memory_num: 检索记忆数量。
            reflect_file: 反思 Prompt 文件名。
            type_schedule: 是否使用类型调度。
        """
        super().__init__()
        
        # 设置网络
        self.grid = VariableNetwork(tar_graph)
    
        # Agent 调度器
        if type_schedule:
            self.schedule = OrderTypeActivation(self)
        else:
            self.schedule = mesa.time.RandomActivation(self)
        
        # 上下文信息
        self.context = context
        
        # 设置基座模型
        self.llm = llm

        # 设置 prompt 工厂
        self.prompt_factory = PromptFactory(prompt_path, self.llm)
        
        # 反思 prompt
        reflect_prompt = self.prompt_factory.get_template(reflect_file)

        # 设置 memory 工厂
        self.memory_factory = MemoryFactory(
            self.llm, memory_num, reflect_prompt, self, memory_path
        )

        # 初始化 agent 列表
        self.agent_list = []
    
    def add_agent(self, tar_agent, node_id):
        """
        将一个新的代理添加到系统中。

        参数:
            tar_agent: 要添加的代理对象。
            node_id: 代理将被放置的节点 ID。
        """
        self.agent_list.append(tar_agent)
        self.schedule.add(tar_agent)
        self.grid.place_agent(tar_agent, node_id)
    
    def step(self):
        """
        执行模拟步骤。
        
        Returns:
            int: 始终返回 0。
        """
        self.schedule.step()
        return 0

