"""
选举投票实验

模拟 2020 年美国总统大选投票过程。

模块:
- scenario: 基础场景（规则型）
- with_llm: LLM 驱动版本
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
