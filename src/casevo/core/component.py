"""
组件基类定义

所有 Casevo 组件的基础类。
"""


class BaseComponent:
    """所有组件的基类"""
    componet_type = ""
    component_id = ""
    context = None

    def __init__(self, component_id, coponent_type, tar_context):
        """
        初始化组件实例。

        参数:
            component_id (str): 组件的唯一标识符。
            coponent_type (str): 组件的类型。
            tar_context (object): 组件运行所需的上下文环境。
        """
        self.component_id = component_id
        self.componet_type = coponent_type
        self.context = tar_context


class BaseAgentComponent(BaseComponent):
    """Agent 组件的基类"""
    agent = None

    def __init__(self, component_id, coponent_type, agent):
        super().__init__(component_id, coponent_type, agent.context)
        self.agent = agent


class BaseModelComponent(BaseComponent):
    """Model 组件的基类"""
    model = None

    def __init__(self, component_id, coponent_type, model):
        super().__init__(component_id, coponent_type, model.context)
        self.model = model

