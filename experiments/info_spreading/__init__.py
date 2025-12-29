"""
信息传播实验

研究虚假信息在社交网络中的传播动力学。
- 200 个节点的无标度网络
- 10% 初始虚假信息
- 传播抑制策略评估
"""

from experiments.info_spreading.scenario import (
    InfoSpreadingModel,
    InfoSpreadingAgent,
    InformationType,
    AgentType,
    run_info_spreading_experiment,
)

__all__ = [
    "InfoSpreadingModel",
    "InfoSpreadingAgent",
    "InformationType", 
    "AgentType",
    "run_info_spreading_experiment",
]
