"""
资源分配实验

模拟多智能体资源协商分配过程。
- 50 个智能体
- 1000 单位总资源
- 最多 10 轮协商
"""

from experiments.resource.scenario import (
    ResourceAllocationModel,
    ResourceAgent,
    AgentPriority,
    run_resource_experiment,
)

__all__ = [
    "ResourceAllocationModel",
    "ResourceAgent",
    "AgentPriority",
    "run_resource_experiment",
]
