"""
Casevo 推理模块

包含思维链 (CoT)、树状思维 (ToT) 和协同决策机制。
"""

from casevo.reasoning.chain import (
    ThoughtChain,
    BaseStep,
    ChoiceStep,
    ScoreStep,
    JsonStep,
    ToolStep,
    ChainPool,
)
from casevo.reasoning.tree import (
    TreeOfThought,
    ToTNode,
    ToTStep,
    EvaluatorStep,
    SearchStrategy,
    AdaptiveToT,
    ToTChainPool,
)
from casevo.reasoning.collaborative import (
    CollaborativeDecisionMaker,
    Message,
    StandardNegotiationProtocol,
    DistributedConsensus,
    CentralAggregator,
    DecisionMode,
    NegotiationStatus,
)
from casevo.reasoning.evaluator import (
    DecisionEvaluator,
    DecisionRecord,
    ConfidenceEstimator,
    MetaCognitionModule,
    EvaluationDimension,
)

__all__ = [
    # 思维链
    "ThoughtChain",
    "BaseStep",
    "ChoiceStep",
    "ScoreStep",
    "JsonStep",
    "ToolStep",
    "ChainPool",
    
    # 树状思维
    "TreeOfThought",
    "ToTNode",
    "ToTStep",
    "EvaluatorStep",
    "SearchStrategy",
    "AdaptiveToT",
    "ToTChainPool",
    
    # 协同决策
    "CollaborativeDecisionMaker",
    "Message",
    "StandardNegotiationProtocol",
    "DistributedConsensus",
    "CentralAggregator",
    "DecisionMode",
    "NegotiationStatus",
    
    # 决策评估
    "DecisionEvaluator",
    "DecisionRecord",
    "ConfidenceEstimator",
    "MetaCognitionModule",
    "EvaluationDimension",
]

