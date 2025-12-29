"""
对比实验

各种方法和模型的对比实验（符合 Proposal 设计）。

模块:
- cot_vs_tot: CoT vs ToT 推理机制对比
- memory_optimization: 记忆系统优化对比
- full_optimization: 完整优化对比（三组对照实验）
"""

from experiments.comparisons.cot_vs_tot import run_cot_vs_tot_experiment
from experiments.comparisons.memory_optimization import run_memory_optimization_experiment
from experiments.comparisons.full_optimization import run_full_optimization_experiment

__all__ = [
    "run_cot_vs_tot_experiment",
    "run_memory_optimization_experiment",
    "run_full_optimization_experiment",
]
