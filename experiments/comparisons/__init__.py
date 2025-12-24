"""
对比实验

各种方法和模型的对比实验。

模块:
- cot_vs_tot: CoT vs ToT 推理机制对比
- baseline: 基线对比实验
"""

from experiments.comparisons.cot_vs_tot import run_cot_vs_tot_experiment
from experiments.comparisons.baseline import run_baseline_comparison

__all__ = [
    "run_cot_vs_tot_experiment",
    "run_baseline_comparison",
]
