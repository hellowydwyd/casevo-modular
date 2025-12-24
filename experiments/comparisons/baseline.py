"""
基线对比实验

对比原始 CoT、ToT 和完整优化方案在三个实验场景中的表现。
"""

import json
import os
import sys
from typing import Dict, Any, List
from datetime import datetime
import statistics

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from election_scenario import run_election_experiment
from resource_allocation import run_resource_experiment
from info_spreading import run_info_spreading_experiment


class BaselineComparison:
    """
    基线对比实验运行器
    
    负责运行所有实验配置并收集对比数据。
    """
    
    def __init__(self, output_dir: str = "experiments/results"):
        """
        初始化对比实验
        
        Args:
            output_dir: 输出目录，默认为 experiments/results
        """
        self.output_dir = output_dir
        self.results: Dict[str, Dict[str, Any]] = {
            'election': {},
            'resource': {},
            'info_spreading': {}
        }
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_election_comparison(self, num_runs: int = 3) -> Dict[str, Any]:
        """
        运行选举实验对比
        
        Args:
            num_runs: 运行次数
            
        Returns:
            对比结果
        """
        print("=" * 60)
        print("运行选举投票实验对比")
        print("=" * 60)
        
        configs = {
            'baseline_cot': {
                'num_voters': 101,
                'use_tot': False,
                'num_rounds': 6
            },
            'optimized_tot': {
                'num_voters': 101,
                'use_tot': True,
                'num_rounds': 6
            }
        }
        
        results = {}
        
        for config_name, config in configs.items():
            print(f"\n运行配置: {config_name}")
            run_results = []
            
            for i in range(num_runs):
                print(f"  运行 {i+1}/{num_runs}...")
                result = run_election_experiment(config)
                run_results.append(result)
            
            # 聚合结果
            aggregated = self._aggregate_election_results(run_results)
            results[config_name] = aggregated
            
            print(f"  平均 Biden 支持率: {aggregated['avg_biden_support']:.2f}")
            print(f"  平均 Trump 支持率: {aggregated['avg_trump_support']:.2f}")
        
        self.results['election'] = results
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
            'avg_biden_support': statistics.mean(biden_supports),
            'std_biden_support': statistics.stdev(biden_supports) if len(biden_supports) > 1 else 0,
            'avg_trump_support': statistics.mean(trump_supports),
            'std_trump_support': statistics.stdev(trump_supports) if len(trump_supports) > 1 else 0,
            'avg_undecided': statistics.mean(undecided_counts),
            'individual_runs': run_results
        }
    
    def run_resource_comparison(self, num_runs: int = 3) -> Dict[str, Any]:
        """
        运行资源分配实验对比
        
        Args:
            num_runs: 运行次数
            
        Returns:
            对比结果
        """
        print("\n" + "=" * 60)
        print("运行资源分配实验对比")
        print("=" * 60)
        
        configs = {
            'baseline': {
                'num_agents': 50,
                'total_resources': 1000,
                'use_collaborative': False,
                'max_rounds': 10
            },
            'optimized_collaborative': {
                'num_agents': 50,
                'total_resources': 1000,
                'use_collaborative': True,
                'max_rounds': 10
            }
        }
        
        results = {}
        
        for config_name, config in configs.items():
            print(f"\n运行配置: {config_name}")
            run_results = []
            
            for i in range(num_runs):
                print(f"  运行 {i+1}/{num_runs}...")
                result = run_resource_experiment(config)
                run_results.append(result)
            
            # 聚合结果
            aggregated = self._aggregate_resource_results(run_results)
            results[config_name] = aggregated
            
            print(f"  平均协商轮次: {aggregated['avg_rounds']:.2f}")
            print(f"  平均基尼系数: {aggregated['avg_gini']:.4f}")
            print(f"  平均满意度: {aggregated['avg_satisfaction']:.4f}")
        
        self.results['resource'] = results
        return results
    
    def _aggregate_resource_results(self, run_results: List[Dict]) -> Dict[str, Any]:
        """聚合资源分配实验结果"""
        rounds_list = []
        gini_list = []
        satisfaction_list = []
        utilization_list = []
        
        for result in run_results:
            rounds_list.append(result.get('negotiation_rounds', 0))
            fairness = result.get('fairness_metrics', {})
            gini_list.append(fairness.get('gini_coefficient', 0))
            satisfaction_list.append(fairness.get('average_satisfaction', 0))
            utilization_list.append(fairness.get('utilization_rate', 0))
        
        return {
            'num_runs': len(run_results),
            'avg_rounds': statistics.mean(rounds_list),
            'std_rounds': statistics.stdev(rounds_list) if len(rounds_list) > 1 else 0,
            'avg_gini': statistics.mean(gini_list),
            'std_gini': statistics.stdev(gini_list) if len(gini_list) > 1 else 0,
            'avg_satisfaction': statistics.mean(satisfaction_list),
            'std_satisfaction': statistics.stdev(satisfaction_list) if len(satisfaction_list) > 1 else 0,
            'avg_utilization': statistics.mean(utilization_list),
            'individual_runs': run_results
        }
    
    def run_info_spreading_comparison(self, num_runs: int = 3) -> Dict[str, Any]:
        """
        运行信息传播实验对比
        
        Args:
            num_runs: 运行次数
            
        Returns:
            对比结果
        """
        print("\n" + "=" * 60)
        print("运行信息传播实验对比")
        print("=" * 60)
        
        configs = {
            'baseline': {
                'num_agents': 200,
                'use_enhanced_evaluation': False,
                'num_rounds': 20,
                'false_info_ratio': 0.3
            },
            'optimized_enhanced': {
                'num_agents': 200,
                'use_enhanced_evaluation': True,
                'num_rounds': 20,
                'false_info_ratio': 0.3
            }
        }
        
        results = {}
        
        for config_name, config in configs.items():
            print(f"\n运行配置: {config_name}")
            run_results = []
            
            for i in range(num_runs):
                print(f"  运行 {i+1}/{num_runs}...")
                result = run_info_spreading_experiment(config)
                run_results.append(result)
            
            # 聚合结果
            aggregated = self._aggregate_info_results(run_results)
            results[config_name] = aggregated
            
            print(f"  平均虚假信息接受率: {aggregated['avg_false_belief_ratio']:.4f}")
            print(f"  平均判断准确率: {aggregated['avg_accuracy']:.4f}")
        
        self.results['info_spreading'] = results
        return results
    
    def _aggregate_info_results(self, run_results: List[Dict]) -> Dict[str, Any]:
        """聚合信息传播实验结果"""
        false_belief_ratios = []
        accuracies = []
        spread_counts = []
        
        for result in run_results:
            final_stats = result.get('final_statistics', {})
            accuracy_stats = result.get('accuracy_statistics', {})
            
            false_belief_ratios.append(final_stats.get('false_belief_ratio', 0))
            accuracies.append(accuracy_stats.get('overall_accuracy', 0))
            spread_counts.append(final_stats.get('false_info_spread_count', 0))
        
        return {
            'num_runs': len(run_results),
            'avg_false_belief_ratio': statistics.mean(false_belief_ratios),
            'std_false_belief_ratio': statistics.stdev(false_belief_ratios) if len(false_belief_ratios) > 1 else 0,
            'avg_accuracy': statistics.mean(accuracies),
            'std_accuracy': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            'avg_spread_count': statistics.mean(spread_counts),
            'individual_runs': run_results
        }
    
    def run_all_comparisons(self, num_runs: int = 3) -> Dict[str, Any]:
        """
        运行所有对比实验
        
        Args:
            num_runs: 每个配置的运行次数
            
        Returns:
            所有对比结果
        """
        print("\n" + "#" * 70)
        print("# Casevo 决策优化方案 - 基线对比实验")
        print("#" * 70)
        
        # 运行各实验
        self.run_election_comparison(num_runs)
        self.run_resource_comparison(num_runs)
        self.run_info_spreading_comparison(num_runs)
        
        # 生成汇总报告
        summary = self.generate_summary()
        
        # 保存结果
        self.save_results()
        
        return {
            'results': self.results,
            'summary': summary
        }
    
    def generate_summary(self) -> Dict[str, Any]:
        """生成汇总报告"""
        summary = {
            'timestamp': self.timestamp,
            'experiments': {}
        }
        
        # 选举实验对比
        if self.results['election']:
            baseline = self.results['election'].get('baseline_cot', {})
            optimized = self.results['election'].get('optimized_tot', {})
            
            summary['experiments']['election'] = {
                'improvement': {
                    'undecided_reduction': baseline.get('avg_undecided', 0) - optimized.get('avg_undecided', 0)
                },
                'baseline_undecided': baseline.get('avg_undecided', 0),
                'optimized_undecided': optimized.get('avg_undecided', 0)
            }
        
        # 资源分配实验对比
        if self.results['resource']:
            baseline = self.results['resource'].get('baseline', {})
            optimized = self.results['resource'].get('optimized_collaborative', {})
            
            summary['experiments']['resource'] = {
                'improvement': {
                    'rounds_reduction': baseline.get('avg_rounds', 0) - optimized.get('avg_rounds', 0),
                    'satisfaction_increase': optimized.get('avg_satisfaction', 0) - baseline.get('avg_satisfaction', 0),
                    'gini_improvement': baseline.get('avg_gini', 0) - optimized.get('avg_gini', 0)
                },
                'baseline_gini': baseline.get('avg_gini', 0),
                'optimized_gini': optimized.get('avg_gini', 0)
            }
        
        # 信息传播实验对比
        if self.results['info_spreading']:
            baseline = self.results['info_spreading'].get('baseline', {})
            optimized = self.results['info_spreading'].get('optimized_enhanced', {})
            
            summary['experiments']['info_spreading'] = {
                'improvement': {
                    'false_belief_reduction': baseline.get('avg_false_belief_ratio', 0) - optimized.get('avg_false_belief_ratio', 0),
                    'accuracy_increase': optimized.get('avg_accuracy', 0) - baseline.get('avg_accuracy', 0)
                },
                'baseline_accuracy': baseline.get('avg_accuracy', 0),
                'optimized_accuracy': optimized.get('avg_accuracy', 0)
            }
        
        return summary
    
    def save_results(self):
        """保存结果到文件"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存完整结果
        output_file = os.path.join(
            self.output_dir, 
            f"baseline_comparison_{self.timestamp}.json"
        )
        
        # 转换结果为可序列化格式
        serializable_results = self._make_serializable(self.results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_file}")
        
        # 生成摘要文件
        summary_file = os.path.join(
            self.output_dir,
            f"comparison_summary_{self.timestamp}.txt"
        )
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self._format_summary())
        
        print(f"摘要已保存到: {summary_file}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _format_summary(self) -> str:
        """格式化摘要报告"""
        summary = self.generate_summary()
        
        lines = [
            "=" * 70,
            "Casevo 决策优化方案 - 基线对比实验摘要",
            "=" * 70,
            f"实验时间: {self.timestamp}",
            "",
            "-" * 70,
            "1. 选举投票实验",
            "-" * 70,
        ]
        
        if 'election' in summary.get('experiments', {}):
            exp = summary['experiments']['election']
            lines.extend([
                f"  基线未决票数: {exp.get('baseline_undecided', 'N/A')}",
                f"  优化未决票数: {exp.get('optimized_undecided', 'N/A')}",
                f"  未决票减少: {exp.get('improvement', {}).get('undecided_reduction', 'N/A')}",
            ])
        
        lines.extend([
            "",
            "-" * 70,
            "2. 资源分配实验",
            "-" * 70,
        ])
        
        if 'resource' in summary.get('experiments', {}):
            exp = summary['experiments']['resource']
            imp = exp.get('improvement', {})
            lines.extend([
                f"  基线基尼系数: {exp.get('baseline_gini', 'N/A'):.4f}",
                f"  优化基尼系数: {exp.get('optimized_gini', 'N/A'):.4f}",
                f"  协商轮次减少: {imp.get('rounds_reduction', 'N/A')}",
                f"  满意度提升: {imp.get('satisfaction_increase', 'N/A'):.4f}",
            ])
        
        lines.extend([
            "",
            "-" * 70,
            "3. 信息传播实验",
            "-" * 70,
        ])
        
        if 'info_spreading' in summary.get('experiments', {}):
            exp = summary['experiments']['info_spreading']
            imp = exp.get('improvement', {})
            lines.extend([
                f"  基线判断准确率: {exp.get('baseline_accuracy', 'N/A'):.4f}",
                f"  优化判断准确率: {exp.get('optimized_accuracy', 'N/A'):.4f}",
                f"  虚假信息接受率降低: {imp.get('false_belief_reduction', 'N/A'):.4f}",
                f"  准确率提升: {imp.get('accuracy_increase', 'N/A'):.4f}",
            ])
        
        lines.extend([
            "",
            "=" * 70,
            "实验结束",
            "=" * 70,
        ])
        
        return "\n".join(lines)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Casevo 基线对比实验')
    parser.add_argument('--runs', type=int, default=3, help='每个配置的运行次数')
    parser.add_argument('--output', type=str, default='experiments/results', help='输出目录')
    parser.add_argument('--experiment', type=str, choices=['all', 'election', 'resource', 'info'],
                       default='all', help='运行的实验类型')
    
    args = parser.parse_args()
    
    comparison = BaselineComparison(output_dir=args.output)
    
    if args.experiment == 'all':
        comparison.run_all_comparisons(num_runs=args.runs)
    elif args.experiment == 'election':
        comparison.run_election_comparison(num_runs=args.runs)
        comparison.save_results()
    elif args.experiment == 'resource':
        comparison.run_resource_comparison(num_runs=args.runs)
        comparison.save_results()
    elif args.experiment == 'info':
        comparison.run_info_spreading_comparison(num_runs=args.runs)
        comparison.save_results()
    
    print("\n实验完成！")


if __name__ == "__main__":
    main()

