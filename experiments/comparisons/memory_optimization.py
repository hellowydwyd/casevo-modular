"""
记忆系统优化对比实验

对比原始记忆检索与优化后的上下文感知记忆检索在决策质量上的差异。

实验设计（符合 Proposal 要求）：
- 基线组：原始 ChromaDB 向量检索
- 优化组：上下文感知记忆筛选 + 时间衰减因子
"""

import os
import sys
import random
import json
import statistics
from typing import Dict, Any, List
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from casevo import create_default_llm

from experiments.election.scenario import run_election_experiment
from experiments.resource.scenario import run_resource_experiment
from experiments.info_spreading.scenario import run_info_spreading_experiment


class MemoryOptimizationExperiment:
    """
    记忆系统优化对比实验
    
    对比原始记忆检索与优化后的记忆检索策略。
    """
    
    def __init__(self, output_dir: str = "experiments/results/comparisons", 
                 llm_interface=None):
        """
        初始化实验
        
        Args:
            output_dir: 输出目录
            llm_interface: LLM 接口（必须提供）
        """
        if llm_interface is None:
            raise RuntimeError(
                "记忆优化实验需要 LLM 接口！\n"
                "使用方法：\n"
                "  from casevo import create_default_llm\n"
                "  llm = create_default_llm()\n"
                "  exp = MemoryOptimizationExperiment(llm_interface=llm)"
            )
        
        self.output_dir = output_dir
        self.llm = llm_interface
        self.results: Dict[str, Any] = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_election_memory_comparison(self, num_runs: int = 10) -> Dict[str, Any]:
        """
        选举场景记忆优化对比
        
        Args:
            num_runs: 运行次数（Proposal 要求 10 次）
            
        Returns:
            对比结果
        """
        print("=" * 60)
        print("选举场景 - 记忆系统优化对比")
        print(f"运行次数: {num_runs}")
        print("=" * 60)
        
        configs = {
            'baseline_memory': {
                'num_voters': 101,
                'use_tot': False,
                'num_rounds': 6,
                'use_enhanced_memory': False,  # 基线：不使用增强记忆
                'use_dynamic_reflection': False,
                'use_collaborative': False
            },
            'optimized_memory': {
                'num_voters': 101,
                'use_tot': False,
                'num_rounds': 6,
                'use_enhanced_memory': True,   # 优化：启用增强记忆检索
                'use_dynamic_reflection': True,  # 启用动态反思
                'use_collaborative': False
            }
        }
        
        results = {}
        
        for config_name, config in configs.items():
            print(f"\n配置: {config_name}")
            run_results = []
            
            for i in range(num_runs):
                random.seed(42 + i)
                print(f"  运行 {i+1}/{num_runs}...")
                
                try:
                    result = run_election_experiment(config, llm_interface=self.llm)
                    run_results.append(result)
                except Exception as e:
                    print(f"    错误: {e}")
                    continue
            
            if run_results:
                aggregated = self._aggregate_election_results(run_results)
                results[config_name] = aggregated
                print(f"  平均 Biden 支持率: {aggregated['avg_biden_support']:.2f}")
                print(f"  平均 Undecided: {aggregated['avg_undecided']:.2f}")
        
        self.results['election_memory'] = results
        return results
    
    def _aggregate_election_results(self, run_results: List[Dict]) -> Dict[str, Any]:
        """聚合选举实验结果"""
        biden_supports = []
        trump_supports = []
        undecided_counts = []
        
        for result in run_results:
            final = result.get('final_results', {})
            biden_supports.append(final.get('biden', 0))
            trump_supports.append(final.get('trump', 0))
            undecided_counts.append(final.get('undecided', 0))
        
        return {
            'num_runs': len(run_results),
            'avg_biden_support': statistics.mean(biden_supports) if biden_supports else 0,
            'std_biden_support': statistics.stdev(biden_supports) if len(biden_supports) > 1 else 0,
            'avg_trump_support': statistics.mean(trump_supports) if trump_supports else 0,
            'std_trump_support': statistics.stdev(trump_supports) if len(trump_supports) > 1 else 0,
            'avg_undecided': statistics.mean(undecided_counts) if undecided_counts else 0,
            'std_undecided': statistics.stdev(undecided_counts) if len(undecided_counts) > 1 else 0,
        }
    
    def run_all_comparisons(self, num_runs: int = 10) -> Dict[str, Any]:
        """
        运行所有记忆优化对比实验
        
        Args:
            num_runs: 每个配置的运行次数
            
        Returns:
            所有对比结果
        """
        print("\n" + "#" * 70)
        print("# 记忆系统优化对比实验")
        print(f"# 每组运行 {num_runs} 次")
        print("#" * 70)
        
        self.run_election_memory_comparison(num_runs)
        
        # 生成汇总
        summary = self._generate_summary()
        
        # 保存结果
        self._save_results()
        
        return {
            'results': self.results,
            'summary': summary
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成汇总报告"""
        summary = {
            'timestamp': self.timestamp,
            'experiment_type': 'memory_optimization',
            'comparisons': {}
        }
        
        if 'election_memory' in self.results:
            baseline = self.results['election_memory'].get('baseline_memory', {})
            optimized = self.results['election_memory'].get('optimized_memory', {})
            
            summary['comparisons']['election'] = {
                'baseline_undecided': baseline.get('avg_undecided', 0),
                'optimized_undecided': optimized.get('avg_undecided', 0),
                'improvement': baseline.get('avg_undecided', 0) - optimized.get('avg_undecided', 0)
            }
        
        return summary
    
    def _save_results(self):
        """保存结果"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_file = os.path.join(
            self.output_dir,
            f"memory_optimization_{self.timestamp}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n结果已保存到: {output_file}")


def run_memory_optimization_experiment(num_runs: int = 10):
    """
    运行记忆系统优化对比实验
    
    Args:
        num_runs: 运行次数
    """
    print("初始化 LLM...")
    llm = create_default_llm()
    
    # 测试连接
    test_response = llm.send_message("回复 OK")
    if not test_response:
        print("LLM 连接失败")
        return None
    print("LLM 连接成功")
    
    exp = MemoryOptimizationExperiment(llm_interface=llm)
    return exp.run_all_comparisons(num_runs=num_runs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='记忆系统优化对比实验')
    parser.add_argument('--runs', type=int, default=10, help='每个配置的运行次数')
    
    args = parser.parse_args()
    
    run_memory_optimization_experiment(num_runs=args.runs)

