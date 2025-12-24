"""
资源分配实验

模拟多智能体资源协商分配过程。

模块:
- scenario: 基础场景（规则型）
- with_llm: LLM 驱动版本
"""

from experiments.resource.scenario import (
    ResourceModel,
    ResourceAgent,
    AgentPriority,
    run_resource_experiment,
)

__all__ = [
    "ResourceModel",
    "ResourceAgent",
    "AgentPriority",
    "run_resource_experiment",
]
