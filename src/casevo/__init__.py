"""
Casevo - Cognitive Agents and Social Evolution Simulator

基于大语言模型的多智能体社会模拟框架，支持：
- 树状思维 (Tree of Thought) 推理
- 上下文感知记忆检索
- 动态反思机制
- 多智能体协同决策

新模块结构 (v0.4.0+):
- casevo.core: 核心模块 (AgentBase, ModelBase)
- casevo.llm: LLM 接口
- casevo.memory: 记忆系统
- casevo.reasoning: 推理模块 (CoT, ToT, 协同决策)
"""

# ============================================================
# 核心模块 (casevo.core)
# ============================================================
from casevo.core import (
    AgentBase,
    ModelBase,
    VariableNetwork,
    OrderTypeActivation,
    BaseComponent,
    BaseAgentComponent,
    BaseModelComponent,
)

# ============================================================
# LLM 模块 (casevo.llm)
# ============================================================
from casevo.llm import (
    LLM_INTERFACE,
    OpenAILLM,
    OpenAIEmbeddingFunction,
    create_default_llm,
    create_llm,
    get_api_key_for_model,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MODEL_API_KEYS,
)

# ============================================================
# 记忆模块 (casevo.memory)
# ============================================================
from casevo.memory import (
    # 基础记忆
    Memory,
    MemoryItem,
    MemoryFactory,
    
    # 高级记忆
    AdvancedMemory,
    AdvancedMemoryFactory,
    AdvancedMemoryItem,
    ContextAwareRetriever,
    MemoryCompressor,
    MemoryType,
    ImportanceLevel,
    
    # 背景知识
    Background,
    BackgroundItem,
    BackgroundFactory,
)

# ============================================================
# 推理模块 (casevo.reasoning)
# ============================================================
from casevo.reasoning import (
    # 思维链
    ThoughtChain,
    BaseStep,
    ChoiceStep,
    ScoreStep,
    JsonStep,
    ToolStep,
    ChainPool,
    
    # 树状思维
    TreeOfThought,
    ToTNode,
    ToTStep,
    EvaluatorStep,
    SearchStrategy,
    AdaptiveToT,
    ToTChainPool,
    
    # 协同决策
    CollaborativeDecisionMaker,
    Message,
    StandardNegotiationProtocol,
    DistributedConsensus,
    CentralAggregator,
    DecisionMode,
    NegotiationStatus,
    
    # 决策评估
    DecisionEvaluator,
    DecisionRecord,
    ConfidenceEstimator,
    MetaCognitionModule,
    EvaluationDimension,
)

# ============================================================
# Prompt 模块
# ============================================================
from casevo.prompt import Prompt, PromptFactory

# ============================================================
# 工具模块 (casevo.utils)
# ============================================================
from casevo.utils import (
    MesaLog,
    TotLog,
    TotLogStream,
    RequestCache,
    ThreadSend,
    get_random_name,
)

# ============================================================
# 导出列表
# ============================================================
__all__ = [
    # 核心模块
    "AgentBase",
    "ModelBase",
    "VariableNetwork",
    "OrderTypeActivation",
    "BaseComponent",
    "BaseAgentComponent",
    "BaseModelComponent",
    
    # LLM 模块
    "LLM_INTERFACE",
    "OpenAILLM",
    "OpenAIEmbeddingFunction",
    "create_default_llm",
    "create_llm",
    "get_api_key_for_model",
    "DEFAULT_API_KEY",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "MODEL_API_KEYS",
    
    # 记忆模块
    "Memory",
    "MemoryItem",
    "MemoryFactory",
    "AdvancedMemory",
    "AdvancedMemoryFactory",
    "AdvancedMemoryItem",
    "ContextAwareRetriever",
    "MemoryCompressor",
    "MemoryType",
    "ImportanceLevel",
    "Background",
    "BackgroundItem",
    "BackgroundFactory",
    
    # 推理模块
    "ThoughtChain",
    "BaseStep",
    "ChoiceStep",
    "ScoreStep",
    "JsonStep",
    "ToolStep",
    "ChainPool",
    "TreeOfThought",
    "ToTNode",
    "ToTStep",
    "EvaluatorStep",
    "SearchStrategy",
    "AdaptiveToT",
    "ToTChainPool",
    "CollaborativeDecisionMaker",
    "Message",
    "StandardNegotiationProtocol",
    "DistributedConsensus",
    "CentralAggregator",
    "DecisionMode",
    "NegotiationStatus",
    "DecisionEvaluator",
    "DecisionRecord",
    "ConfidenceEstimator",
    "MetaCognitionModule",
    "EvaluationDimension",
    
    # Prompt 模块
    "Prompt",
    "PromptFactory",
    
    # 工具模块
    "MesaLog",
    "TotLog",
    "TotLogStream",
    "RequestCache",
    "ThreadSend",
    "get_random_name",
]

__version__ = "0.4.0"
