"""
实验评估指标模块

提供完整的评估指标，符合 Proposal 要求的四个维度：
1. 决策质量：准确率、一致性、合理性
2. 推理能力：深度、多样性、连贯性
3. 计算效率：响应时间、内存占用、可扩展性
4. 社会效应：群体共识度、意见分化程度、社会稳定性
"""

import time
import statistics
import tracemalloc
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import threading


# ============================================================
# 1. 计算效率指标
# ============================================================

@dataclass
class LLMCallRecord:
    """LLM 调用记录"""
    call_id: str
    start_time: float
    end_time: float
    duration_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    success: bool = True
    error_message: str = ""


class PerformanceTracker:
    """
    性能追踪器
    
    追踪 LLM 调用时间、内存使用等计算效率指标。
    线程安全，支持并发场景。
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._call_records: List[LLMCallRecord] = []
        self._memory_snapshots: List[Dict[str, Any]] = []
        self._call_counter = 0
        self._start_time: Optional[float] = None
        self._tracking_memory = False
    
    def start_experiment(self):
        """开始实验计时"""
        self._start_time = time.time()
        self._call_records.clear()
        self._memory_snapshots.clear()
        self._call_counter = 0
    
    def stop_experiment(self) -> Dict[str, Any]:
        """
        停止实验计时并返回汇总
        
        Returns:
            性能汇总数据
        """
        end_time = time.time()
        total_duration = end_time - self._start_time if self._start_time else 0
        
        return {
            'total_duration_seconds': total_duration,
            'llm_call_stats': self.get_call_statistics(),
            'memory_stats': self.get_memory_statistics()
        }
    
    @contextmanager
    def track_llm_call(self, call_id: str = None):
        """
        上下文管理器：追踪单次 LLM 调用
        
        Usage:
            with tracker.track_llm_call("vote_decision"):
                response = llm.send_message(prompt)
        """
        with self._lock:
            self._call_counter += 1
            actual_call_id = call_id or f"call_{self._call_counter}"
        
        start_time = time.time()
        error_message = ""
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            record = LLMCallRecord(
                call_id=actual_call_id,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message
            )
            
            with self._lock:
                self._call_records.append(record)
    
    def record_llm_call(self, duration_ms: float, success: bool = True, 
                        call_id: str = None, tokens_in: int = 0, tokens_out: int = 0):
        """
        手动记录 LLM 调用
        
        Args:
            duration_ms: 调用耗时（毫秒）
            success: 是否成功
            call_id: 调用标识
            tokens_in: 输入 token 数
            tokens_out: 输出 token 数
        """
        with self._lock:
            self._call_counter += 1
            actual_call_id = call_id or f"call_{self._call_counter}"
            
            record = LLMCallRecord(
                call_id=actual_call_id,
                start_time=time.time() - duration_ms / 1000,
                end_time=time.time(),
                duration_ms=duration_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                success=success
            )
            self._call_records.append(record)
    
    def start_memory_tracking(self):
        """开始内存追踪"""
        if not self._tracking_memory:
            tracemalloc.start()
            self._tracking_memory = True
    
    def stop_memory_tracking(self):
        """停止内存追踪"""
        if self._tracking_memory:
            tracemalloc.stop()
            self._tracking_memory = False
    
    def snapshot_memory(self, label: str = ""):
        """记录当前内存快照"""
        if self._tracking_memory:
            current, peak = tracemalloc.get_traced_memory()
            snapshot = {
                'label': label,
                'timestamp': time.time(),
                'current_mb': current / 1024 / 1024,
                'peak_mb': peak / 1024 / 1024
            }
            with self._lock:
                self._memory_snapshots.append(snapshot)
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """
        获取 LLM 调用统计
        
        Returns:
            调用统计数据
        """
        with self._lock:
            records = list(self._call_records)
        
        if not records:
            return {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'avg_duration_ms': 0,
                'min_duration_ms': 0,
                'max_duration_ms': 0,
                'std_duration_ms': 0,
                'total_duration_ms': 0,
                'total_tokens_in': 0,
                'total_tokens_out': 0
            }
        
        durations = [r.duration_ms for r in records]
        successful = [r for r in records if r.success]
        failed = [r for r in records if not r.success]
        
        return {
            'total_calls': len(records),
            'successful_calls': len(successful),
            'failed_calls': len(failed),
            'avg_duration_ms': statistics.mean(durations),
            'min_duration_ms': min(durations),
            'max_duration_ms': max(durations),
            'std_duration_ms': statistics.stdev(durations) if len(durations) > 1 else 0,
            'total_duration_ms': sum(durations),
            'total_tokens_in': sum(r.tokens_in for r in records),
            'total_tokens_out': sum(r.tokens_out for r in records)
        }
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取内存统计"""
        with self._lock:
            snapshots = list(self._memory_snapshots)
        
        if not snapshots:
            return {
                'snapshots': 0,
                'avg_memory_mb': 0,
                'peak_memory_mb': 0
            }
        
        return {
            'snapshots': len(snapshots),
            'avg_memory_mb': statistics.mean(s['current_mb'] for s in snapshots),
            'peak_memory_mb': max(s['peak_mb'] for s in snapshots)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            'call_statistics': self.get_call_statistics(),
            'memory_statistics': self.get_memory_statistics(),
            'call_records': [
                {
                    'call_id': r.call_id,
                    'duration_ms': r.duration_ms,
                    'success': r.success
                }
                for r in self._call_records
            ]
        }


def track_performance(tracker: PerformanceTracker, call_id: str = None):
    """
    装饰器：追踪函数性能
    
    Usage:
        @track_performance(tracker, "my_function")
        def my_function():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracker.track_llm_call(call_id or func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================
# 2. 推理能力指标
# ============================================================

@dataclass
class ReasoningRecord:
    """推理记录"""
    agent_id: str
    decision_id: str
    reasoning_type: str  # "cot" or "tot"
    depth: int  # 推理深度
    branches_explored: int  # 探索的分支数
    pruned_branches: int  # 剪枝的分支数
    reasoning_steps: List[str] = field(default_factory=list)
    final_score: float = 0.0


class ReasoningMetrics:
    """
    推理能力指标
    
    量化推理的深度、多样性和连贯性。
    """
    
    def __init__(self):
        self._records: List[ReasoningRecord] = []
    
    def record_cot_reasoning(self, agent_id: str, decision_id: str,
                              steps: List[str], final_score: float = 0.0):
        """记录 CoT 推理"""
        record = ReasoningRecord(
            agent_id=agent_id,
            decision_id=decision_id,
            reasoning_type="cot",
            depth=len(steps),
            branches_explored=1,
            pruned_branches=0,
            reasoning_steps=steps,
            final_score=final_score
        )
        self._records.append(record)
    
    def record_tot_reasoning(self, agent_id: str, decision_id: str,
                              depth: int, branches_explored: int,
                              pruned_branches: int, 
                              reasoning_path: List[str],
                              final_score: float = 0.0):
        """记录 ToT 推理"""
        record = ReasoningRecord(
            agent_id=agent_id,
            decision_id=decision_id,
            reasoning_type="tot",
            depth=depth,
            branches_explored=branches_explored,
            pruned_branches=pruned_branches,
            reasoning_steps=reasoning_path,
            final_score=final_score
        )
        self._records.append(record)
    
    def get_depth_statistics(self) -> Dict[str, Any]:
        """
        获取推理深度统计
        
        Returns:
            深度统计数据
        """
        if not self._records:
            return {'avg_depth': 0, 'max_depth': 0, 'min_depth': 0}
        
        depths = [r.depth for r in self._records]
        return {
            'avg_depth': statistics.mean(depths),
            'max_depth': max(depths),
            'min_depth': min(depths),
            'std_depth': statistics.stdev(depths) if len(depths) > 1 else 0
        }
    
    def get_diversity_statistics(self) -> Dict[str, Any]:
        """
        获取推理多样性统计
        
        多样性 = 平均探索分支数 / 平均深度
        """
        if not self._records:
            return {'avg_branches': 0, 'diversity_index': 0}
        
        branches = [r.branches_explored for r in self._records]
        depths = [r.depth for r in self._records]
        
        avg_branches = statistics.mean(branches)
        avg_depth = statistics.mean(depths) if depths else 1
        
        # 多样性指数：分支数 / 深度
        diversity_index = avg_branches / avg_depth if avg_depth > 0 else 0
        
        return {
            'avg_branches': avg_branches,
            'max_branches': max(branches),
            'diversity_index': diversity_index,
            'pruning_rate': sum(r.pruned_branches for r in self._records) / sum(branches) if sum(branches) > 0 else 0
        }
    
    def get_coherence_score(self) -> float:
        """
        计算推理连贯性分数
        
        连贯性 = 成功完成的推理比例 × 平均最终得分
        """
        if not self._records:
            return 0.0
        
        completed = [r for r in self._records if r.depth > 0]
        if not completed:
            return 0.0
        
        completion_rate = len(completed) / len(self._records)
        avg_score = statistics.mean(r.final_score for r in completed)
        
        return completion_rate * avg_score
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取综合统计"""
        cot_records = [r for r in self._records if r.reasoning_type == "cot"]
        tot_records = [r for r in self._records if r.reasoning_type == "tot"]
        
        return {
            'total_reasoning_episodes': len(self._records),
            'cot_count': len(cot_records),
            'tot_count': len(tot_records),
            'depth_stats': self.get_depth_statistics(),
            'diversity_stats': self.get_diversity_statistics(),
            'coherence_score': self.get_coherence_score()
        }


# ============================================================
# 3. 社会效应指标
# ============================================================

class SocialEffectMetrics:
    """
    社会效应指标
    
    量化群体共识度、意见分化程度和社会稳定性。
    """
    
    @staticmethod
    def calculate_consensus_index(vote_distribution: Dict[str, int]) -> float:
        """
        计算群体共识度（基于信息熵）
        
        - 完全共识（只有一个选项有票）→ 熵=0 → 共识度=1.0
        - 完全分散（均匀分布）→ 熵=最大 → 共识度=0.0
        
        Args:
            vote_distribution: 投票分布 {选项: 票数}
            
        Returns:
            共识度指数 (0-1)
        """
        import math
        
        if not vote_distribution:
            return 0.0
        
        total = sum(vote_distribution.values())
        if total == 0:
            return 0.0
        
        # 计算信息熵
        entropy = 0.0
        for count in vote_distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # 归一化：最大熵 = log2(选项数)
        num_options = len(vote_distribution)
        if num_options <= 1:
            return 1.0
        
        max_entropy = math.log2(num_options)
        if max_entropy == 0:
            return 1.0
        
        # 共识度 = 1 - 归一化熵
        consensus = 1 - (entropy / max_entropy)
        return consensus
    
    @staticmethod
    def calculate_polarization_index(vote_distribution: Dict[str, int]) -> float:
        """
        计算极化指数（简化的 Esteban-Ray 指数）
        
        极化指数反映意见的两极分化程度。
        值越高表示越极化。
        
        Args:
            vote_distribution: 投票分布
            
        Returns:
            极化指数 (0-1)
        """
        if not vote_distribution or len(vote_distribution) < 2:
            return 0.0
        
        total = sum(vote_distribution.values())
        if total == 0:
            return 0.0
        
        shares = [count / total for count in vote_distribution.values()]
        
        # 简化极化指数：使用基尼系数变体
        # P = Σ Σ |pi - pj| * pi * pj
        polarization = 0.0
        for i, pi in enumerate(shares):
            for j, pj in enumerate(shares):
                if i != j:
                    polarization += abs(pi - pj) * pi * pj
        
        # 归一化到 0-1
        n = len(shares)
        max_polarization = (n - 1) / n if n > 1 else 1
        
        return min(1.0, polarization / max_polarization if max_polarization > 0 else 0)
    
    @staticmethod
    def calculate_opinion_stability(history: List[Dict[str, int]]) -> float:
        """
        计算意见稳定性
        
        稳定性 = 1 - 平均轮间变化率
        
        Args:
            history: 历史投票分布列表
            
        Returns:
            稳定性指数 (0-1)
        """
        if len(history) < 2:
            return 1.0
        
        total_change = 0.0
        total_comparisons = 0
        
        for i in range(1, len(history)):
            prev = history[i - 1]
            curr = history[i]
            
            all_keys = set(prev.keys()) | set(curr.keys())
            prev_total = sum(prev.values()) or 1
            curr_total = sum(curr.values()) or 1
            
            for key in all_keys:
                prev_share = prev.get(key, 0) / prev_total
                curr_share = curr.get(key, 0) / curr_total
                total_change += abs(curr_share - prev_share)
            
            total_comparisons += len(all_keys)
        
        avg_change = total_change / total_comparisons if total_comparisons > 0 else 0
        stability = 1 - min(1.0, avg_change)
        
        return stability
    
    @staticmethod
    def calculate_influence_spread(initial_believers: int, 
                                    final_believers: int,
                                    total_agents: int) -> Dict[str, float]:
        """
        计算信息影响力传播指标
        
        Args:
            initial_believers: 初始相信人数
            final_believers: 最终相信人数
            total_agents: 总智能体数
            
        Returns:
            影响力指标
        """
        if total_agents == 0:
            return {'spread_rate': 0, 'amplification': 0, 'penetration': 0}
        
        spread_rate = (final_believers - initial_believers) / total_agents
        amplification = final_believers / initial_believers if initial_believers > 0 else 0
        penetration = final_believers / total_agents
        
        return {
            'spread_rate': spread_rate,
            'amplification': amplification,
            'penetration': penetration
        }
    
    @staticmethod
    def calculate_misinformation_resistance(correct_rejections: int,
                                             total_false_info_exposures: int) -> float:
        """
        计算虚假信息抵抗力
        
        Args:
            correct_rejections: 正确拒绝虚假信息次数
            total_false_info_exposures: 总虚假信息暴露次数
            
        Returns:
            抵抗力 (0-1)
        """
        if total_false_info_exposures == 0:
            return 1.0
        
        return correct_rejections / total_false_info_exposures


# ============================================================
# 4. 决策质量指标
# ============================================================

@dataclass
class DecisionQualityRecord:
    """决策质量记录"""
    agent_id: str
    decision_id: str
    decision_content: str
    confidence: float
    is_correct: Optional[bool] = None
    reasoning_quality: float = 0.5  # 0-1
    consistency_with_history: float = 1.0  # 0-1


class DecisionQualityMetrics:
    """
    决策质量指标
    
    量化决策的准确率、一致性和合理性。
    """
    
    # 政治倾向与预期投票的映射
    EXPECTED_VOTES = {
        "Progressive Left": "Biden",
        "Establishment Liberal": "Biden", 
        "Democratic Mainstay": "Biden",
        "Outsider Left": "Biden",
        "Devout & Diverse": "Undecided",
        "Stressed Sideliner": "Undecided",
        "Ambivalent Right": "Undecided",
        "Committed Conservative": "Trump",
        "Populist Right": "Trump",
        "Faith & Flag": "Trump"
    }
    
    def __init__(self):
        self._records: List[DecisionQualityRecord] = []
        self._agent_histories: Dict[str, List[str]] = {}  # agent_id -> decision history
        self._agent_profiles: Dict[str, str] = {}  # agent_id -> political_leaning
    
    def register_agent_profile(self, agent_id: str, political_leaning: str):
        """注册智能体的政治倾向"""
        self._agent_profiles[agent_id] = political_leaning
    
    def evaluate_decision_alignment(self, agent_id: str, decision: str) -> float:
        """
        评估决策与政治倾向的一致性
        
        Args:
            agent_id: 智能体ID
            decision: 决策结果 (Biden/Trump/Undecided)
            
        Returns:
            一致性分数 (0-1)
        """
        leaning = self._agent_profiles.get(agent_id, "")
        expected = self.EXPECTED_VOTES.get(leaning, "Undecided")
        
        if decision == expected:
            return 1.0
        elif decision == "Undecided" or expected == "Undecided":
            return 0.5  # 未决定算部分合理
        else:
            return 0.0  # 完全相反
    
    def record_decision(self, agent_id: str, decision_id: str,
                        decision_content: str, confidence: float,
                        is_correct: Optional[bool] = None,
                        reasoning_quality: float = 0.5):
        """记录决策"""
        # 计算与历史决策的一致性
        history = self._agent_histories.get(agent_id, [])
        consistency = self._calculate_consistency(decision_content, history)
        
        record = DecisionQualityRecord(
            agent_id=agent_id,
            decision_id=decision_id,
            decision_content=decision_content,
            confidence=confidence,
            is_correct=is_correct,
            reasoning_quality=reasoning_quality,
            consistency_with_history=consistency
        )
        self._records.append(record)
        
        # 更新历史
        if agent_id not in self._agent_histories:
            self._agent_histories[agent_id] = []
        self._agent_histories[agent_id].append(decision_content)
    
    def _calculate_consistency(self, current: str, history: List[str]) -> float:
        """计算决策一致性"""
        if not history:
            return 1.0
        
        # 简单实现：检查最近决策是否相同
        recent = history[-3:]  # 最近 3 个决策
        same_count = sum(1 for h in recent if h == current)
        
        return same_count / len(recent) if recent else 1.0
    
    def get_accuracy(self) -> float:
        """
        获取决策准确率/合理性
        
        基于政治倾向评估决策是否合理。
        如果没有标注 is_correct，则使用政治倾向一致性评估。
        
        Returns:
            准确率 (0-1)
        """
        evaluated = [r for r in self._records if r.is_correct is not None]
        if evaluated:
            correct = sum(1 for r in evaluated if r.is_correct)
            return correct / len(evaluated)
        
        # 如果没有 ground truth，使用政治倾向一致性评估
        if not self._records:
            return 0.5
        
        alignment_scores = []
        for r in self._records:
            score = self.evaluate_decision_alignment(r.agent_id, r.decision_content)
            alignment_scores.append(score)
        
        return statistics.mean(alignment_scores) if alignment_scores else 0.5
    
    def get_consistency(self) -> float:
        """
        获取决策一致性
        
        Returns:
            平均一致性 (0-1)
        """
        if not self._records:
            return 1.0
        
        return statistics.mean(r.consistency_with_history for r in self._records)
    
    def get_confidence_calibration(self) -> float:
        """
        获取置信度校准度
        
        校准度 = 1 - |置信度 - 准确率|
        
        Returns:
            校准度 (0-1)
        """
        evaluated = [r for r in self._records if r.is_correct is not None]
        if not evaluated:
            return 0.5
        
        avg_confidence = statistics.mean(r.confidence for r in evaluated)
        accuracy = self.get_accuracy()
        
        calibration = 1 - abs(avg_confidence - accuracy)
        return calibration
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取综合统计"""
        return {
            'total_decisions': len(self._records),
            'unique_agents': len(self._agent_histories),
            'accuracy': self.get_accuracy(),
            'consistency': self.get_consistency(),
            'confidence_calibration': self.get_confidence_calibration(),
            'avg_confidence': statistics.mean(r.confidence for r in self._records) if self._records else 0,
            'avg_reasoning_quality': statistics.mean(r.reasoning_quality for r in self._records) if self._records else 0
        }


# ============================================================
# 5. 综合评估指标
# ============================================================

class ExperimentMetrics:
    """
    实验综合评估指标
    
    整合四个维度的所有指标，提供统一的评估接口。
    """
    
    def __init__(self):
        self.performance = PerformanceTracker()
        self.reasoning = ReasoningMetrics()
        self.social = SocialEffectMetrics
        self.decision_quality = DecisionQualityMetrics()
        
        # 投票历史（用于社会效应计算）
        self._vote_history: List[Dict[str, int]] = []
    
    def start_experiment(self, track_memory: bool = False):
        """开始实验"""
        self.performance.start_experiment()
        if track_memory:
            self.performance.start_memory_tracking()
    
    def stop_experiment(self) -> Dict[str, Any]:
        """停止实验并返回完整报告"""
        self.performance.stop_memory_tracking()
        
        return self.get_full_report()
    
    def record_vote_distribution(self, distribution: Dict[str, int]):
        """记录投票分布"""
        self._vote_history.append(distribution.copy())
    
    def get_full_report(self) -> Dict[str, Any]:
        """
        获取完整评估报告
        
        Returns:
            四个维度的完整评估数据
        """
        # 社会效应指标
        social_stats = {}
        if self._vote_history:
            latest = self._vote_history[-1]
            social_stats = {
                'consensus_index': self.social.calculate_consensus_index(latest),
                'polarization_index': self.social.calculate_polarization_index(latest),
                'opinion_stability': self.social.calculate_opinion_stability(self._vote_history)
            }
        
        return {
            'decision_quality': self.decision_quality.get_statistics(),
            'reasoning_ability': self.reasoning.get_statistics(),
            'computational_efficiency': self.performance.to_dict(),
            'social_effects': social_stats
        }
    
    def get_summary(self) -> Dict[str, float]:
        """
        获取简化摘要（每个维度一个分数）
        
        Returns:
            四个维度的摘要分数
        """
        report = self.get_full_report()
        
        # 决策质量分数
        dq = report['decision_quality']
        decision_score = (dq.get('accuracy', 0.5) + dq.get('consistency', 0.5) + 
                         dq.get('avg_reasoning_quality', 0.5)) / 3
        
        # 推理能力分数
        ra = report['reasoning_ability']
        depth_stats = ra.get('depth_stats', {})
        diversity_stats = ra.get('diversity_stats', {})
        reasoning_score = (
            min(1.0, depth_stats.get('avg_depth', 0) / 5) * 0.4 +
            min(1.0, diversity_stats.get('diversity_index', 0)) * 0.3 +
            ra.get('coherence_score', 0.5) * 0.3
        )
        
        # 计算效率分数（越快越好）
        ce = report['computational_efficiency']
        call_stats = ce.get('call_statistics', {})
        avg_duration = call_stats.get('avg_duration_ms', 1000)
        # 理想: 2000ms, 可接受: 10000ms, 最差: 60000ms+
        # CoT 通常 2-3s, ToT 通常 5-10s, 复杂推理可能更长
        if avg_duration <= 2000:
            efficiency_score = 1.0
        elif avg_duration <= 10000:
            efficiency_score = 1.0 - (avg_duration - 2000) / 16000  # 2s->1.0, 10s->0.5
        elif avg_duration <= 60000:
            efficiency_score = 0.5 - (avg_duration - 10000) / 100000  # 10s->0.5, 60s->0.0
        else:
            efficiency_score = 0.0
        
        # 社会效应分数
        se = report['social_effects']
        social_score = (
            se.get('consensus_index', 0.5) * 0.4 +
            (1 - se.get('polarization_index', 0.5)) * 0.3 +
            se.get('opinion_stability', 0.5) * 0.3
        )
        
        # 综合得分：不包含效率得分（LLM 场景效率指标不适合作为质量评估依据）
        # 仅基于决策质量、推理能力、社会效应三个维度
        overall_score = (decision_score + reasoning_score + social_score) / 3
        
        # 包含效率的综合得分（供参考）
        overall_score_with_efficiency = (decision_score + reasoning_score + efficiency_score + social_score) / 4
        
        return {
            'decision_quality_score': decision_score,
            'reasoning_ability_score': reasoning_score,
            'computational_efficiency_score': efficiency_score,
            'social_effects_score': social_score,
            'overall_score': overall_score,  # 不含效率
            'overall_score_with_efficiency': overall_score_with_efficiency  # 含效率（供参考）
        }


# ============================================================
# 便捷函数
# ============================================================

def create_experiment_metrics() -> ExperimentMetrics:
    """创建实验指标实例"""
    return ExperimentMetrics()


def calculate_gini_coefficient(values: List[float]) -> float:
    """
    计算基尼系数
    
    Args:
        values: 数值列表
        
    Returns:
        基尼系数 (0-1)
    """
    if not values or sum(values) == 0:
        return 0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_values))
    
    return (2 * cumsum) / (n * sum(sorted_values)) - (n + 1) / n

