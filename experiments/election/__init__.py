"""
选举投票实验

模拟 2020 年美国总统大选投票过程。
- 101 个选民智能体
- 小世界网络拓扑
- 6 轮辩论事件
"""

from experiments.election.scenario import (
    ElectionModel,
    ElectionVoterAgent,
    VoteChoice,
    PoliticalLeaning,
    run_election_experiment,
)

__all__ = [
    "ElectionModel",
    "ElectionVoterAgent", 
    "VoteChoice",
    "PoliticalLeaning",
    "run_election_experiment",
]
