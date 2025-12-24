"""
LLM 接口基类

定义与大语言模型交互的标准接口。
"""

from abc import abstractmethod, ABCMeta


class LLM_INTERFACE(metaclass=ABCMeta):
    """LLM 接口基类"""

    @abstractmethod
    def send_message(self, prompt, json_flag=False):
        """
        发送 prompt 并获取响应。

        参数:
            prompt: 提示文本。
            json_flag: 是否要求 JSON 格式响应。

        返回:
            LLM 响应文本。
        """
        pass

    @abstractmethod
    def send_embedding(self, text_list):
        """
        获取文本嵌入向量。

        参数:
            text_list: 文本列表。

        返回:
            嵌入向量列表。
        """
        pass
    
    @abstractmethod
    def get_lang_embedding(self):
        """
        获取 LangChain/ChromaDB 兼容的嵌入函数。

        返回:
            嵌入函数实例。
        """
        pass

