"""
完整优化对比实验

对比实验方案（符合 Proposal 要求）：
- 基线组：原始 CoT 决策机制
- 优化组 A：仅 ToT 多层次推理
- 消融组 A1：ToT + 增强记忆
- 消融组 A2：ToT + 动态反思
- 优化组 B：全部优化（ToT + 增强记忆 + 动态反思 + 协同决策）

每组独立运行 N 次以排除随机因素，使用固定随机种子确保可复现。
"""

import os
import sys
import random
import json
import statistics
import numpy as np
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
from experiments.utils.thought_logger import ThoughtLogger, get_thought_logger


class FullOptimizationExperiment:
    """
    完整优化对比实验
    
    按照 Proposal 设计的对照实验：
    - 基线组：原始 CoT
    - 优化组 A：仅 ToT
    - 消融组 A1：ToT + 增强记忆
    - 消融组 A2：ToT + 动态反思
    - 优化组 B：全部优化
    """
    
    # 基础随机种子，确保可复现
    BASE_SEED = 42
    
    def __init__(self, output_dir: str = "experiments/results/comparisons",
                 llm_interface=None, base_seed: int = 42,
                 enable_persistent_memory: bool = True,
                 enable_thought_logging: bool = True):
        """
        初始化实验
        
        Args:
            output_dir: 输出目录
            llm_interface: LLM 接口（必须提供）
            base_seed: 基础随机种子
            enable_persistent_memory: 是否启用持久化记忆
            enable_thought_logging: 是否启用思考过程日志
        """
        if llm_interface is None:
            raise RuntimeError(
                "完整优化实验需要 LLM 接口！\n"
                "使用方法：\n"
                "  from casevo import create_default_llm\n"
                "  llm = create_default_llm()\n"
                "  exp = FullOptimizationExperiment(llm_interface=llm)"
            )
        
        self.output_dir = output_dir
        self.llm = llm_interface
        self.base_seed = base_seed
        self.enable_persistent_memory = enable_persistent_memory
        self.enable_thought_logging = enable_thought_logging
        self.results: Dict[str, Dict[str, Any]] = {
            'election': {},
            'resource': {},
            'info_spreading': {}
        }
        # 使用精确到毫秒的时间戳+进程ID，确保并行运行时唯一
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        
        # 创建持久化目录（包含进程ID，避免冲突）
        self.memory_base_dir = os.path.join(output_dir, "memories", self.timestamp)
        self.thoughts_dir = os.path.join(output_dir, "thoughts")
        if enable_persistent_memory:
            os.makedirs(self.memory_base_dir, exist_ok=True)
        if enable_thought_logging:
            os.makedirs(self.thoughts_dir, exist_ok=True)
        
        # 思考日志记录器（包含进程ID，避免冲突）
        self.thought_logger = None
        if enable_thought_logging:
            self.thought_logger = ThoughtLogger(
                output_dir=self.thoughts_dir,
                experiment_name=self.timestamp
            )
    
    def _set_seed(self, run_index: int):
        """设置随机种子确保可复现"""
        seed = self.base_seed + run_index
        random.seed(seed)
        np.random.seed(seed)
        return seed
    
    def run_election_comparison(self, num_runs: int = 10) -> Dict[str, Any]:
        """
        选举场景三组对比
        
        Args:
            num_runs: 运行次数（Proposal 要求 10 次）
        """
        print("=" * 60)
        print("选举投票场景 - 完整优化对比")
        print(f"运行次数: {num_runs}")
        print("=" * 60)
        
        # 对照配置（包含消融实验）
        configs = {
            'baseline_cot': {
                'num_voters': 30,
                'use_tot': False,
                'num_rounds': 6,
                'use_enhanced_memory': False,
                'use_dynamic_reflection': False,
                'use_collaborative': False,
                'description': '基线组：原始 CoT 决策机制'
            },
            'optimized_tot_only': {
                'num_voters': 30,
                'use_tot': True,
                'num_rounds': 6,
                'use_enhanced_memory': False,
                'use_dynamic_reflection': False,
                'use_collaborative': False,
                'description': '优化组 A：仅 ToT 多层次推理'
            },
            'ablation_tot_memory': {
                'num_voters': 30,
                'use_tot': True,
                'num_rounds': 6,
                'use_enhanced_memory': True,
                'use_dynamic_reflection': False,
                'use_collaborative': False,
                'description': '消融组 A1：ToT + 增强记忆'
            },
            'ablation_tot_reflection': {
                'num_voters': 30,
                'use_tot': True,
                'num_rounds': 6,
                'use_enhanced_memory': False,
                'use_dynamic_reflection': True,
                'use_collaborative': False,
                'description': '消融组 A2：ToT + 动态反思'
            },
            'optimized_full': {
                'num_voters': 30,
                'use_tot': True,
                'num_rounds': 6,
                'use_enhanced_memory': True,
                'use_dynamic_reflection': True,
                'use_collaborative': True,
                'description': '优化组 B：全部优化'
            }
        }
        
        results = {}
        
        for config_name, config in configs.items():
            print(f"\n【{config['description']}】")
            run_results = []
            
            for i in range(num_runs):
                seed = self._set_seed(i)
                print(f"  运行 {i+1}/{num_runs} (种子={seed})...")
                
                try:
                    result = run_election_experiment(config, llm_interface=self.llm)
                    result['random_seed'] = seed  # 记录使用的种子
                    run_results.append(result)
                except Exception as e:
                    print(f"    错误: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if run_results:
                aggregated = self._aggregate_election_results(run_results)
                results[config_name] = aggregated
                
                print(f"  ✓ Biden: {aggregated['avg_biden_support']:.1f} ± {aggregated['std_biden_support']:.1f}")
                print(f"  ✓ Trump: {aggregated['avg_trump_support']:.1f} ± {aggregated['std_trump_support']:.1f}")
                print(f"  ✓ Undecided: {aggregated['avg_undecided']:.1f} ± {aggregated['std_undecided']:.1f}")
                
                # 每组完成后保存
                self.results['election'] = results
                self._save_results()
                print(f"  [选举-{config_name} 数据已保存]")
        
        self.results['election'] = results
        return results
    
    def run_resource_comparison(self, num_runs: int = 10) -> Dict[str, Any]:
        """
        资源分配场景三组对比
        
        Args:
            num_runs: 运行次数
        """
        print("\n" + "=" * 60)
        print("资源分配场景 - 完整优化对比")
        print(f"运行次数: {num_runs}")
        print("=" * 60)
        
        configs = {
            'baseline_cot': {
                'num_agents': 20,
                'total_resources': 400,
                'max_rounds': 5,
                'use_collaborative': False,
                'use_tot': False,
                'use_enhanced_memory': False,
                'use_dynamic_reflection': False,
                'description': '基线组：原始 CoT 独立决策'
            },
            'optimized_tot_only': {
                'num_agents': 20,
                'total_resources': 400,
                'max_rounds': 5,
                'use_collaborative': False,
                'use_tot': True,
                'use_enhanced_memory': False,
                'use_dynamic_reflection': False,
                'description': '优化组 A：仅 ToT 多层次推理'
            },
            'ablation_tot_memory': {
                'num_agents': 20,
                'total_resources': 400,
                'max_rounds': 5,
                'use_collaborative': False,
                'use_tot': True,
                'use_enhanced_memory': True,
                'use_dynamic_reflection': False,
                'description': '消融组 A1：ToT + 增强记忆'
            },
            'ablation_tot_reflection': {
                'num_agents': 20,
                'total_resources': 400,
                'max_rounds': 5,
                'use_collaborative': False,
                'use_tot': True,
                'use_enhanced_memory': False,
                'use_dynamic_reflection': True,
                'description': '消融组 A2：ToT + 动态反思'
            },
            'optimized_full': {
                'num_agents': 20,
                'total_resources': 400,
                'max_rounds': 5,
                'use_collaborative': True,
                'use_tot': True,
                'use_enhanced_memory': True,
                'use_dynamic_reflection': True,
                'description': '优化组 B：全部优化'
            }
        }
        
        results = {}
        
        for config_name, config in configs.items():
            print(f"\n【{config['description']}】")
            run_results = []
            
            for i in range(num_runs):
                seed = self._set_seed(i)
                print(f"  运行 {i+1}/{num_runs} (种子={seed})...")
                
                try:
                    result = run_resource_experiment(config, llm_interface=self.llm)
                    result['random_seed'] = seed
                    run_results.append(result)
                except Exception as e:
                    print(f"    错误: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if run_results:
                aggregated = self._aggregate_resource_results(run_results)
                results[config_name] = aggregated
                
                print(f"  ✓ 协商轮次: {aggregated['avg_rounds']:.1f} ± {aggregated['std_rounds']:.1f}")
                print(f"  ✓ 基尼系数: {aggregated['avg_gini']:.4f} ± {aggregated['std_gini']:.4f}")
                print(f"  ✓ 满意度: {aggregated['avg_satisfaction']:.4f}")
                
                # 每组完成后保存
                self.results['resource'] = results
                self._save_results()
                print(f"  [资源-{config_name} 数据已保存]")
        
        self.results['resource'] = results
        return results
    
    def run_info_spreading_comparison(self, num_runs: int = 10) -> Dict[str, Any]:
        """
        信息传播场景对比
        
        Args:
            num_runs: 运行次数
        """
        print("\n" + "=" * 60)
        print("信息传播场景 - 优化对比")
        print(f"运行次数: {num_runs}")
        print("=" * 60)
        
        configs = {
            'baseline_cot': {
                'num_agents': 50,
                'use_enhanced_evaluation': False,
                'num_rounds': 10,
                'false_info_ratio': 0.3,
                'use_tot': False,
                'use_enhanced_memory': False,
                'use_dynamic_reflection': False,
                'description': '基线组：规则评估'
            },
            'optimized_tot_only': {
                'num_agents': 50,
                'use_enhanced_evaluation': True,
                'num_rounds': 10,
                'false_info_ratio': 0.3,
                'use_tot': True,
                'use_enhanced_memory': False,
                'use_dynamic_reflection': False,
                'description': '优化组 A：ToT 评估'
            },
            'ablation_tot_memory': {
                'num_agents': 50,
                'use_enhanced_evaluation': True,
                'num_rounds': 10,
                'false_info_ratio': 0.3,
                'use_tot': True,
                'use_enhanced_memory': True,
                'use_dynamic_reflection': False,
                'description': '消融组 A1：ToT + 增强记忆'
            },
            'ablation_tot_reflection': {
                'num_agents': 50,
                'use_enhanced_evaluation': True,
                'num_rounds': 10,
                'false_info_ratio': 0.3,
                'use_tot': True,
                'use_enhanced_memory': False,
                'use_dynamic_reflection': True,
                'description': '消融组 A2：ToT + 动态反思'
            },
            'optimized_full': {
                'num_agents': 50,
                'use_enhanced_evaluation': True,
                'num_rounds': 10,
                'false_info_ratio': 0.3,
                'use_tot': True,
                'use_enhanced_memory': True,
                'use_dynamic_reflection': True,
                'description': '优化组 B：全部优化'
            }
        }
        
        results = {}
        
        for config_name, config in configs.items():
            print(f"\n【{config['description']}】")
            run_results = []
            
            for i in range(num_runs):
                seed = self._set_seed(i)
                print(f"  运行 {i+1}/{num_runs} (种子={seed})...")
                
                try:
                    result = run_info_spreading_experiment(config, llm_interface=self.llm)
                    result['random_seed'] = seed
                    run_results.append(result)
                except Exception as e:
                    print(f"    错误: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if run_results:
                aggregated = self._aggregate_info_results(run_results)
                results[config_name] = aggregated
                
                print(f"  ✓ 虚假信息接受率: {aggregated['avg_false_belief_ratio']:.4f}")
                print(f"  ✓ 判断准确率: {aggregated['avg_accuracy']:.4f}")
                
                # 每组完成后保存
                self.results['info_spreading'] = results
                self._save_results()
                print(f"  [信息传播-{config_name} 数据已保存]")
        
        self.results['info_spreading'] = results
        return results
    
    def _aggregate_election_results(self, run_results: List[Dict]) -> Dict[str, Any]:
        """聚合选举实验结果"""
        biden = [r.get('final_results', {}).get('biden', 0) for r in run_results]
        trump = [r.get('final_results', {}).get('trump', 0) for r in run_results]
        undecided = [r.get('final_results', {}).get('undecided', 0) for r in run_results]
        
        return {
            'num_runs': len(run_results),
            'avg_biden_support': statistics.mean(biden),
            'std_biden_support': statistics.stdev(biden) if len(biden) > 1 else 0,
            'avg_trump_support': statistics.mean(trump),
            'std_trump_support': statistics.stdev(trump) if len(trump) > 1 else 0,
            'avg_undecided': statistics.mean(undecided),
            'std_undecided': statistics.stdev(undecided) if len(undecided) > 1 else 0,
            # 保留中间过程
            'voting_history': [r.get('voting_history', []) for r in run_results],
            'evaluation_metrics': [r.get('evaluation_metrics', {}) for r in run_results],
            'raw_results': run_results  # 保留完整原始结果
        }
    
    def _aggregate_resource_results(self, run_results: List[Dict]) -> Dict[str, Any]:
        """聚合资源分配实验结果"""
        rounds = [r.get('negotiation_rounds', 0) for r in run_results]
        gini = [r.get('fairness_metrics', {}).get('gini_coefficient', 0) for r in run_results]
        satisfaction = [r.get('fairness_metrics', {}).get('average_satisfaction', 0) for r in run_results]
        
        return {
            'num_runs': len(run_results),
            'avg_rounds': statistics.mean(rounds),
            'std_rounds': statistics.stdev(rounds) if len(rounds) > 1 else 0,
            'avg_gini': statistics.mean(gini),
            'std_gini': statistics.stdev(gini) if len(gini) > 1 else 0,
            'avg_satisfaction': statistics.mean(satisfaction),
            # 保留中间过程
            'allocation_history': [r.get('allocation_history', []) for r in run_results],
            'evaluation_metrics': [r.get('evaluation_metrics', {}) for r in run_results],
            'raw_results': run_results
        }
    
    def _aggregate_info_results(self, run_results: List[Dict]) -> Dict[str, Any]:
        """聚合信息传播实验结果"""
        false_belief = [r.get('final_statistics', {}).get('false_belief_ratio', 0) for r in run_results]
        accuracy = [r.get('accuracy_statistics', {}).get('overall_accuracy', 0) for r in run_results]
        
        return {
            'num_runs': len(run_results),
            'avg_false_belief_ratio': statistics.mean(false_belief),
            'avg_accuracy': statistics.mean(accuracy),
            # 保留中间过程
            'spread_history': [r.get('spread_history', []) for r in run_results],
            'evaluation_metrics': [r.get('evaluation_metrics', {}) for r in run_results],
            'raw_results': run_results
        }
    
    def run_single_group(self, experiment: str, group: str, num_runs: int = 3):
        """
        运行单个实验组（用于并行加速）
        
        Args:
            experiment: 场景类型 (election/resource/info)
            group: 实验组名称
            num_runs: 运行次数
        """
        print(f"\n运行单组实验: {experiment} - {group}")
        
        # 获取对应场景的配置
        if experiment == 'election':
            configs = self._get_election_configs()
            run_func = run_election_experiment
            aggregate_func = self._aggregate_election_results
        elif experiment == 'resource':
            configs = self._get_resource_configs()
            run_func = run_resource_experiment
            aggregate_func = self._aggregate_resource_results
        elif experiment == 'info':
            configs = self._get_info_configs()
            run_func = run_info_spreading_experiment
            aggregate_func = self._aggregate_info_results
        else:
            print(f"未知场景: {experiment}")
            return
        
        if group not in configs:
            print(f"未知实验组: {group}")
            print(f"可用组: {list(configs.keys())}")
            return
        
        config = configs[group]
        print(f"【{config['description']}】")
        run_results = []
        
        for i in range(num_runs):
            seed = self._set_seed(i)
            experiment_id = f"{experiment}_{group}_run{i+1}_{self.timestamp}"
            print(f"  运行 {i+1}/{num_runs} (种子={seed})...")
            
            # 准备持久化路径
            memory_path = None
            if self.enable_persistent_memory:
                memory_path = os.path.join(
                    self.memory_base_dir, 
                    f"{experiment}_{group}_run{i+1}"
                )
                os.makedirs(memory_path, exist_ok=True)
            
            try:
                # 传入持久化路径和思考日志
                result = run_func(
                    config, 
                    llm_interface=self.llm,
                    memory_path=memory_path,
                    thought_logger=self.thought_logger,
                    experiment_id=experiment_id
                )
                result['random_seed'] = seed
                result['memory_path'] = memory_path
                run_results.append(result)
            except Exception as e:
                print(f"    错误: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if run_results:
            aggregated = aggregate_func(run_results)
            # 映射实验名称到 results 字典的键
            result_key = 'info_spreading' if experiment == 'info' else experiment
            self.results[result_key][group] = aggregated
            self._save_single_group_results(experiment, group)
            
            # 保存思考日志
            if self.thought_logger:
                self.thought_logger.save_full_log(
                    f"thoughts_{experiment}_{group}_{self.timestamp}.json"
                )
            
            print(f"  ✓ {group} 完成，已保存")
    
    def _get_election_configs(self) -> Dict[str, Any]:
        """获取选举场景配置"""
        return {
            'baseline_cot': {
                'num_voters': 30, 'use_tot': False, 'num_rounds': 6,
                'use_enhanced_memory': False, 'use_dynamic_reflection': False,
                'use_collaborative': False, 'description': '基线组：原始 CoT 决策机制'
            },
            'optimized_tot_only': {
                'num_voters': 30, 'use_tot': True, 'num_rounds': 6,
                'use_enhanced_memory': False, 'use_dynamic_reflection': False,
                'use_collaborative': False, 'description': '优化组 A：仅 ToT 多层次推理'
            },
            'ablation_tot_memory': {
                'num_voters': 30, 'use_tot': True, 'num_rounds': 6,
                'use_enhanced_memory': True, 'use_dynamic_reflection': False,
                'use_collaborative': False, 'description': '消融组 A1：ToT + 增强记忆'
            },
            'ablation_tot_reflection': {
                'num_voters': 30, 'use_tot': True, 'num_rounds': 6,
                'use_enhanced_memory': False, 'use_dynamic_reflection': True,
                'use_collaborative': False, 'description': '消融组 A2：ToT + 动态反思'
            },
            'optimized_full': {
                'num_voters': 30, 'use_tot': True, 'num_rounds': 6,
                'use_enhanced_memory': True, 'use_dynamic_reflection': True,
                'use_collaborative': True, 'description': '优化组 B：全部优化'
            }
        }
    
    def _get_resource_configs(self) -> Dict[str, Any]:
        """获取资源分配场景配置"""
        return {
            'baseline_cot': {
                'num_agents': 20, 'total_resources': 400, 'max_rounds': 5,
                'use_collaborative': False, 'use_tot': False,
                'use_enhanced_memory': False, 'use_dynamic_reflection': False,
                'description': '基线组：原始 CoT 独立决策'
            },
            'optimized_tot_only': {
                'num_agents': 20, 'total_resources': 400, 'max_rounds': 5,
                'use_collaborative': False, 'use_tot': True,
                'use_enhanced_memory': False, 'use_dynamic_reflection': False,
                'description': '优化组 A：仅 ToT 多层次推理'
            },
            'ablation_tot_memory': {
                'num_agents': 20, 'total_resources': 400, 'max_rounds': 5,
                'use_collaborative': False, 'use_tot': True,
                'use_enhanced_memory': True, 'use_dynamic_reflection': False,
                'description': '消融组 A1：ToT + 增强记忆'
            },
            'ablation_tot_reflection': {
                'num_agents': 20, 'total_resources': 400, 'max_rounds': 5,
                'use_collaborative': False, 'use_tot': True,
                'use_enhanced_memory': False, 'use_dynamic_reflection': True,
                'description': '消融组 A2：ToT + 动态反思'
            },
            'optimized_full': {
                'num_agents': 20, 'total_resources': 400, 'max_rounds': 5,
                'use_collaborative': True, 'use_tot': True,
                'use_enhanced_memory': True, 'use_dynamic_reflection': True,
                'description': '优化组 B：全部优化'
            }
        }
    
    def _get_info_configs(self) -> Dict[str, Any]:
        """获取信息传播场景配置"""
        return {
            'baseline_cot': {
                'num_agents': 50, 'use_enhanced_evaluation': False, 'num_rounds': 10,
                'false_info_ratio': 0.3, 'use_tot': False,
                'use_enhanced_memory': False, 'use_dynamic_reflection': False,
                'description': '基线组：规则评估'
            },
            'optimized_tot_only': {
                'num_agents': 50, 'use_enhanced_evaluation': True, 'num_rounds': 10,
                'false_info_ratio': 0.3, 'use_tot': True,
                'use_enhanced_memory': False, 'use_dynamic_reflection': False,
                'description': '优化组 A：ToT 评估'
            },
            'ablation_tot_memory': {
                'num_agents': 50, 'use_enhanced_evaluation': True, 'num_rounds': 10,
                'false_info_ratio': 0.3, 'use_tot': True,
                'use_enhanced_memory': True, 'use_dynamic_reflection': False,
                'description': '消融组 A1：ToT + 增强记忆'
            },
            'ablation_tot_reflection': {
                'num_agents': 50, 'use_enhanced_evaluation': True, 'num_rounds': 10,
                'false_info_ratio': 0.3, 'use_tot': True,
                'use_enhanced_memory': False, 'use_dynamic_reflection': True,
                'description': '消融组 A2：ToT + 动态反思'
            },
            'optimized_full': {
                'num_agents': 50, 'use_enhanced_evaluation': True, 'num_rounds': 10,
                'false_info_ratio': 0.3, 'use_tot': True,
                'use_enhanced_memory': True, 'use_dynamic_reflection': True,
                'description': '优化组 B：全部优化'
            }
        }
    
    def _save_single_group_results(self, experiment: str, group: str):
        """保存单组实验结果"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_file = os.path.join(
            self.output_dir,
            f"{experiment}_{group}_{self.timestamp}.json"
        )
        
        # 映射实验名称到 results 字典的键
        result_key = 'info_spreading' if experiment == 'info' else experiment
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment': experiment,
                'group': group,
                'results': self.results[result_key].get(group, {}),
                'timestamp': self.timestamp,
                'base_seed': self.base_seed
            }, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"  结果已保存到: {output_file}")
    
    def run_all_experiments(self, num_runs: int = 3) -> Dict[str, Any]:
        """
        运行所有实验场景的完整对比
        
        Args:
            num_runs: 每个配置的运行次数（默认3次以确保统计有效性）
            
        Returns:
            所有实验结果
        """
        print("\n" + "#" * 70)
        print("# Casevo 决策优化方案 - 完整对比实验（含消融实验）")
        print(f"# 实验组：5组对照（基线+ToT+消融A1+消融A2+全优化）")
        print(f"# 每组运行 {num_runs} 次，基础种子={self.base_seed}")
        print("#" * 70)
        
        # 运行三个场景（每个场景完成后立即保存）
        self.run_election_comparison(num_runs)
        self._save_results()  # 选举完成后保存
        print("  [选举场景数据已保存]")
        
        self.run_resource_comparison(num_runs)
        self._save_results()  # 资源完成后保存
        print("  [资源场景数据已保存]")
        
        self.run_info_spreading_comparison(num_runs)
        self._save_results()  # 信息传播完成后保存
        print("  [信息传播场景数据已保存]")
        
        # 生成汇总报告
        summary = self._generate_summary()
        
        # 最终保存（包含汇总）
        self._save_results()
        
        # 打印汇总
        self._print_summary(summary)
        
        return {
            'results': self.results,
            'summary': summary
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成汇总报告"""
        summary = {
            'timestamp': self.timestamp,
            'experiment_type': 'full_optimization',
            'scenarios': {}
        }
        
        # 选举场景
        if self.results['election']:
            baseline = self.results['election'].get('baseline_cot', {})
            tot_only = self.results['election'].get('optimized_tot_only', {})
            full_opt = self.results['election'].get('optimized_full', {})
            
            summary['scenarios']['election'] = {
                'baseline': {
                    'undecided': baseline.get('avg_undecided', 0)
                },
                'tot_only': {
                    'undecided': tot_only.get('avg_undecided', 0),
                    'improvement_vs_baseline': baseline.get('avg_undecided', 0) - tot_only.get('avg_undecided', 0)
                },
                'full_optimization': {
                    'undecided': full_opt.get('avg_undecided', 0),
                    'improvement_vs_baseline': baseline.get('avg_undecided', 0) - full_opt.get('avg_undecided', 0)
                }
            }
        
        # 资源分配场景
        if self.results['resource']:
            baseline = self.results['resource'].get('baseline_cot', {})
            tot_only = self.results['resource'].get('optimized_tot_only', {})
            full_opt = self.results['resource'].get('optimized_full', {})
            
            summary['scenarios']['resource'] = {
                'baseline': {
                    'gini': baseline.get('avg_gini', 0),
                    'rounds': baseline.get('avg_rounds', 0),
                    'satisfaction': baseline.get('avg_satisfaction', 0)
                },
                'tot_only': {
                    'gini': tot_only.get('avg_gini', 0),
                    'rounds': tot_only.get('avg_rounds', 0),
                    'satisfaction': tot_only.get('avg_satisfaction', 0),
                    'gini_improvement': baseline.get('avg_gini', 0) - tot_only.get('avg_gini', 0)
                },
                'full_optimization': {
                    'gini': full_opt.get('avg_gini', 0),
                    'rounds': full_opt.get('avg_rounds', 0),
                    'satisfaction': full_opt.get('avg_satisfaction', 0),
                    'gini_improvement': baseline.get('avg_gini', 0) - full_opt.get('avg_gini', 0)
                }
            }
        
        # 信息传播场景
        if self.results['info_spreading']:
            baseline = self.results['info_spreading'].get('baseline_cot', {})
            tot_only = self.results['info_spreading'].get('optimized_tot_only', {})
            full_opt = self.results['info_spreading'].get('optimized_full', {})
            
            summary['scenarios']['info_spreading'] = {
                'baseline': {
                    'accuracy': baseline.get('avg_accuracy', 0),
                    'false_belief_ratio': baseline.get('avg_false_belief_ratio', 0)
                },
                'tot_only': {
                    'accuracy': tot_only.get('avg_accuracy', 0),
                    'false_belief_ratio': tot_only.get('avg_false_belief_ratio', 0),
                    'accuracy_improvement': tot_only.get('avg_accuracy', 0) - baseline.get('avg_accuracy', 0)
                },
                'full_optimization': {
                    'accuracy': full_opt.get('avg_accuracy', 0),
                    'false_belief_ratio': full_opt.get('avg_false_belief_ratio', 0),
                    'accuracy_improvement': full_opt.get('avg_accuracy', 0) - baseline.get('avg_accuracy', 0)
                }
            }
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """打印汇总报告"""
        print("\n" + "=" * 70)
        print("实验汇总报告")
        print("=" * 70)
        
        if 'election' in summary.get('scenarios', {}):
            e = summary['scenarios']['election']
            print("\n【选举投票场景】")
            print(f"  基线组 Undecided: {e['baseline']['undecided']:.1f}")
            print(f"  ToT优化组 Undecided: {e['tot_only']['undecided']:.1f} (改善 {e['tot_only']['improvement_vs_baseline']:.1f})")
            print(f"  完整优化组 Undecided: {e['full_optimization']['undecided']:.1f} (改善 {e['full_optimization']['improvement_vs_baseline']:.1f})")
        
        if 'resource' in summary.get('scenarios', {}):
            r = summary['scenarios']['resource']
            print("\n【资源分配场景】")
            print(f"  基线组基尼系数: {r['baseline']['gini']:.4f}, 满意度: {r['baseline']['satisfaction']:.4f}")
            print(f"  ToT优化组基尼系数: {r['tot_only']['gini']:.4f} (改善 {r['tot_only']['gini_improvement']:.4f})")
            print(f"  完整优化组基尼系数: {r['full_optimization']['gini']:.4f} (改善 {r['full_optimization']['gini_improvement']:.4f})")
        
        if 'info_spreading' in summary.get('scenarios', {}):
            i = summary['scenarios']['info_spreading']
            print("\n【信息传播场景】")
            print(f"  基线组准确率: {i['baseline']['accuracy']:.4f}")
            print(f"  ToT优化组准确率: {i['tot_only']['accuracy']:.4f} (改善 {i['tot_only']['accuracy_improvement']:.4f})")
            print(f"  完整优化组准确率: {i['full_optimization']['accuracy']:.4f} (改善 {i['full_optimization']['accuracy_improvement']:.4f})")
        
        print("\n" + "=" * 70)
    
    def _save_results(self):
        """保存结果"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_file = os.path.join(
            self.output_dir,
            f"full_optimization_{self.timestamp}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n结果已保存到: {output_file}")


def run_full_optimization_experiment(num_runs: int = 10, 
                                     experiment: str = 'all',
                                     api_key: str = None,
                                     group: str = None):
    """
    运行完整优化对比实验
    
    Args:
        num_runs: 运行次数（Proposal 要求 10 次）
        experiment: 运行的实验类型 (all/election/resource/info)
        api_key: 可选，指定API密钥（用于多key并行）
        group: 可选，指定单个实验组 (baseline_cot/optimized_tot_only/ablation_tot_memory/ablation_tot_reflection/optimized_full)
    """
    print("=" * 60)
    print("Casevo 完整优化对比实验")
    if group:
        print(f"单组模式: {group}")
    print("=" * 60)
    
    print("\n初始化 LLM...")
    try:
        from casevo.llm.openai import OpenAILLM
        if api_key:
            # 使用指定的API key
            llm = OpenAILLM(api_key=api_key)
            print(f"使用指定 API Key: {api_key[:8]}...")
        else:
            llm = create_default_llm()
        test = llm.send_message("回复 OK")
        if not test:
            raise Exception("LLM 无响应")
        print("LLM 连接成功")
    except Exception as e:
        print(f"LLM 连接失败: {e}")
        return None
    
    exp = FullOptimizationExperiment(llm_interface=llm)
    
    # 如果指定了单个组，只运行该组
    if group:
        exp.run_single_group(experiment=experiment, group=group, num_runs=num_runs)
        return exp.results
    
    if experiment == 'all':
        return exp.run_all_experiments(num_runs=num_runs)
    elif experiment == 'election':
        exp.run_election_comparison(num_runs=num_runs)
        exp._save_results()
        return exp.results
    elif experiment == 'resource':
        exp.run_resource_comparison(num_runs=num_runs)
        exp._save_results()
        return exp.results
    elif experiment == 'info':
        exp.run_info_spreading_comparison(num_runs=num_runs)
        exp._save_results()
        return exp.results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Casevo 完整优化对比实验')
    parser.add_argument('--runs', type=int, default=3, 
                        help='每个配置的运行次数（默认3次）')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'election', 'resource', 'info'],
                        help='运行的实验类型')
    parser.add_argument('--api-key', type=str, default=None,
                        help='指定API密钥（用于多key并行加速）')
    parser.add_argument('--group', type=str, default=None,
                        choices=['baseline_cot', 'optimized_tot_only', 
                                 'ablation_tot_memory', 'ablation_tot_reflection', 
                                 'optimized_full'],
                        help='只运行指定的实验组（用于并行加速）')
    
    args = parser.parse_args()
    
    run_full_optimization_experiment(
        num_runs=args.runs,
        experiment=args.experiment,
        api_key=args.api_key,
        group=args.group
    )

