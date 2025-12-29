"""
实验工具模块

包含报告生成、数据处理、配置加载、评估指标等工具。
"""

from experiments.utils.report_generator import (
    generate_report_from_results,
    ReportGenerator,
)

from experiments.utils.config_loader import (
    ConfigLoader,
    get_config_loader,
    load_config,
    load_election_config,
    load_resource_config,
    load_info_spreading_config,
)

from experiments.utils.metrics import (
    # 性能追踪
    PerformanceTracker,
    LLMCallRecord,
    track_performance,
    # 推理能力指标
    ReasoningMetrics,
    ReasoningRecord,
    # 社会效应指标
    SocialEffectMetrics,
    # 决策质量指标
    DecisionQualityMetrics,
    DecisionQualityRecord,
    # 综合评估
    ExperimentMetrics,
    create_experiment_metrics,
    calculate_gini_coefficient,
)

__all__ = [
    # 报告生成
    "generate_report_from_results",
    "ReportGenerator",
    # 配置加载
    "ConfigLoader",
    "get_config_loader",
    "load_config",
    "load_election_config",
    "load_resource_config",
    "load_info_spreading_config",
    # 评估指标
    "PerformanceTracker",
    "LLMCallRecord",
    "track_performance",
    "ReasoningMetrics",
    "ReasoningRecord",
    "SocialEffectMetrics",
    "DecisionQualityMetrics",
    "DecisionQualityRecord",
    "ExperimentMetrics",
    "create_experiment_metrics",
    "calculate_gini_coefficient",
]

