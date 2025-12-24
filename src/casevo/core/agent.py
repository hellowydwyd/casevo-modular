"""
Agent 基类定义

所有智能体的基础类。
"""

from abc import abstractmethod
import mesa

from casevo.reasoning.chain import ThoughtChain


class AgentBase(mesa.Agent):
    """用于构建 Agent 的基类"""

    def __init__(self, unique_id, model, description, context):
        """
        初始化代理类实例。

        参数:
            unique_id: 代理的唯一标识符。
            model: 代理所处的 model 环境。
            description: 代理对应的人设描述信息。
            context: agent 的上下文（用于 Prompt）。
        """
        super().__init__(unique_id, model)
        
        self.component_id = "agent_" + str(unique_id)
        self.description = description
        self.context = context
        self.memory = model.memory_factory.create_memory(self)

    def setup_chain(self, chain_dict):
        """
        初始化思考链集合。

        参数:
            chain_dict (dict): 思考链字典，键为标识符，值为步骤列表。
        """
        self.chains = {}
        for key, cur_chain in chain_dict.items():
            tmp_thought = ThoughtChain(self, cur_chain)
            self.chains[key] = tmp_thought

    @abstractmethod
    def step(self):
        """定义抽象方法，用于代理的每一步操作"""
        pass

