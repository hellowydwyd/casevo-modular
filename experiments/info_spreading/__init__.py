"""
信息传播实验

研究虚假信息在社交网络中的传播动力学。

模块:
- scenario: 基础场景（规则型）
- with_llm: LLM 驱动版本
"""

from experiments.info_spreading.scenario import (
    InfoSpreadingModel,
    InfoAgent,
    InformationType,
    AgentType,
    run_info_spreading_experiment,
)

__all__ = [
    "InfoSpreadingModel",
    "InfoAgent",
    "InformationType", 
    "AgentType",
    "run_info_spreading_experiment",
]
