"""
决策评估模块

实现决策质量的多维度评估。
"""

import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from casevo.core.component import BaseAgentComponent


class EvaluationDimension(Enum):
    """评估维度枚举"""
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    RATIONALITY = "rationality"
    CONFIDENCE = "confidence"
    DEPTH = "depth"
    DIVERSITY = "diversity"
    COHERENCE = "coherence"
    SOCIAL_CONSENSUS = "consensus"
    POLARIZATION = "polarization"


@dataclass
class DecisionRecord:
    """决策记录"""
    decision_id: str
    timestamp: int
    agent_id: str
    decision_content: str
    reasoning: str
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    evaluation_scores: Dict[str, float] = field(default_factory=dict)


class ConfidenceEstimator:
    """置信度估计器"""
    
    def __init__(self, 
                 reasoning_weight: float = 0.3,
                 consistency_weight: float = 0.25,
                 evidence_weight: float = 0.25,
                 uncertainty_weight: float = 0.2):
        self.reasoning_weight = reasoning_weight
        self.consistency_weight = consistency_weight
        self.evidence_weight = evidence_weight
        self.uncertainty_weight = uncertainty_weight
        
        self.uncertainty_markers = [
            '可能', '也许', '或许', '大概', '似乎', '应该',
            'maybe', 'perhaps', 'might', 'could', 'possibly',
            'uncertain', 'not sure', '不确定', '不太清楚'
        ]
        
        self.certainty_markers = [
            '一定', '必然', '肯定', '确定', '毫无疑问', '显然',
            'definitely', 'certainly', 'absolutely', 'clearly',
            'without doubt', 'sure', '必须', '绝对'
        ]
    
    def estimate(self, decision: DecisionRecord,
                historical_decisions: Optional[List[DecisionRecord]] = None,
                available_evidence: Optional[List[str]] = None) -> float:
        """估计决策的置信度"""
        scores = {}
        
        scores['reasoning'] = self._evaluate_reasoning_quality(decision.reasoning)
        
        if historical_decisions:
            scores['consistency'] = self._evaluate_consistency(
                decision, historical_decisions
            )
        else:
            scores['consistency'] = 0.5
        
        if available_evidence:
            scores['evidence'] = self._evaluate_evidence_support(
                decision.reasoning, available_evidence
            )
        else:
            scores['evidence'] = 0.5
        
        scores['uncertainty'] = self._evaluate_uncertainty_expression(
            decision.reasoning
        )
        
        confidence = (
            self.reasoning_weight * scores['reasoning'] +
            self.consistency_weight * scores['consistency'] +
            self.evidence_weight * scores['evidence'] +
            self.uncertainty_weight * scores['uncertainty']
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _evaluate_reasoning_quality(self, reasoning: str) -> float:
        """评估推理质量"""
        if not reasoning:
            return 0.2
        
        score = 0.5
        
        word_count = len(reasoning.split())
        if word_count > 50:
            score += 0.1
        if word_count > 100:
            score += 0.1
        
        logic_markers = [
            '因为', '所以', '因此', '由于', '但是', '然而', '首先', '其次', '最后',
            'because', 'therefore', 'thus', 'however', 'firstly', 'secondly',
            'in conclusion', 'as a result', '综上'
        ]
        logic_count = sum(1 for m in logic_markers if m in reasoning.lower())
        score += min(0.2, logic_count * 0.05)
        
        if any(c in reasoning for c in ['1.', '2.', '①', '②', '•', '-']):
            score += 0.1
        
        return min(1.0, score)
    
    def _evaluate_consistency(self, current: DecisionRecord,
                             historical: List[DecisionRecord]) -> float:
        """评估与历史决策的一致性"""
        if not historical:
            return 0.5
        
        recent_decisions = sorted(historical, 
                                  key=lambda x: x.timestamp, 
                                  reverse=True)[:5]
        
        consistency_scores = []
        for hist in recent_decisions:
            similarity = self._text_similarity(
                current.decision_content, hist.decision_content
            )
            consistency_scores.append(similarity)
        
        return statistics.mean(consistency_scores) if consistency_scores else 0.5
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _evaluate_evidence_support(self, reasoning: str,
                                  evidence: List[str]) -> float:
        """评估证据支持程度"""
        if not evidence:
            return 0.5
        
        reasoning_lower = reasoning.lower()
        supported_count = 0
        
        for ev in evidence:
            ev_words = set(ev.lower().split())
            reasoning_words = set(reasoning_lower.split())
            
            overlap = len(ev_words & reasoning_words)
            if overlap > len(ev_words) * 0.3:
                supported_count += 1
        
        return min(1.0, supported_count / len(evidence) + 0.3)
    
    def _evaluate_uncertainty_expression(self, reasoning: str) -> float:
        """评估不确定性表达的适当性"""
        reasoning_lower = reasoning.lower()
        
        uncertainty_count = sum(
            1 for m in self.uncertainty_markers if m in reasoning_lower
        )
        certainty_count = sum(
            1 for m in self.certainty_markers if m in reasoning_lower
        )
        
        if certainty_count > 3 and uncertainty_count == 0:
            return 0.4
        elif uncertainty_count > 3 and certainty_count == 0:
            return 0.5
        else:
            return 0.7


class ReflectionLevel(Enum):
    """反思层次枚举（符合 Proposal 多层次反思设计）"""
    SHALLOW = "shallow"     # 浅层反思：关注具体决策的对错
    DEEP = "deep"           # 深层反思：关注决策背后的价值观和信念


@dataclass
class ReflectionResult:
    """反思结果"""
    level: ReflectionLevel
    trigger_reason: str
    original_decision: str
    reflection_content: str
    insights: List[str]
    confidence_adjustment: float
    belief_updates: Dict[str, Any] = field(default_factory=dict)
    timestamp: int = 0


class MultiLevelReflection:
    """
    多层次反思机制（符合 Proposal 要求）
    
    实现两个层次的反思：
    - 浅层反思：关注具体决策的对错，快速调整
    - 深层反思：关注决策背后的价值观和信念，形成稳定观点
    """
    
    def __init__(self, 
                 shallow_threshold: float = 0.5,
                 deep_threshold: float = 0.3,
                 deep_reflection_interval: int = 5):
        """
        初始化多层次反思
        
        Args:
            shallow_threshold: 浅层反思触发阈值（置信度低于此值）
            deep_threshold: 深层反思触发阈值（置信度低于此值）
            deep_reflection_interval: 深层反思间隔（每 N 次决策触发一次）
        """
        self.shallow_threshold = shallow_threshold
        self.deep_threshold = deep_threshold
        self.deep_reflection_interval = deep_reflection_interval
        
        self.reflection_history: List[ReflectionResult] = []
        self.decision_count = 0
        self.core_beliefs: Dict[str, float] = {}  # 核心信念及其强度
        self.value_priorities: List[str] = []      # 价值优先级
    
    def determine_reflection_level(self, confidence: float, 
                                   decision: DecisionRecord) -> Optional[ReflectionLevel]:
        """
        确定需要的反思层次
        
        Args:
            confidence: 当前决策置信度
            decision: 决策记录
            
        Returns:
            需要的反思层次，None 表示不需要反思
        """
        self.decision_count += 1
        
        # 深层反思条件：置信度极低 或 达到间隔
        if confidence < self.deep_threshold:
            return ReflectionLevel.DEEP
        
        if self.decision_count % self.deep_reflection_interval == 0:
            return ReflectionLevel.DEEP
        
        # 浅层反思条件：置信度较低
        if confidence < self.shallow_threshold:
            return ReflectionLevel.SHALLOW
        
        return None
    
    def shallow_reflect(self, decision: DecisionRecord, 
                       llm=None) -> ReflectionResult:
        """
        浅层反思：关注具体决策的对错
        
        快速评估决策是否正确，进行即时调整。
        """
        if llm is None:
            # 无 LLM 时使用规则反思
            return self._rule_based_shallow_reflect(decision)
        
        prompt = f"""请对以下决策进行快速反思，评估其正确性。

## 决策内容
{decision.decision_content}

## 推理过程
{decision.reasoning}

## 当前置信度
{decision.confidence:.2f}

## 浅层反思任务
请快速评估：
1. 这个决策的逻辑是否正确？
2. 是否有明显的错误或遗漏？
3. 需要如何调整？

请按格式回答：
【逻辑评估】正确/有问题
【问题点】（如有问题，说明具体问题）
【调整建议】（1句话）
【调整后置信度】0.0-1.0
"""
        
        try:
            response = llm.send_message(prompt)
            
            import re
            match = re.search(r'【调整后置信度】\s*([0-9.]+)', response)
            new_confidence = float(match.group(1)) if match else decision.confidence
            
            match = re.search(r'【调整建议】\s*(.+?)(?:\n|$)', response)
            insight = match.group(1) if match else "保持当前决策"
            
            result = ReflectionResult(
                level=ReflectionLevel.SHALLOW,
                trigger_reason=f"置信度 {decision.confidence:.2f} 低于阈值",
                original_decision=decision.decision_content,
                reflection_content=response,
                insights=[insight],
                confidence_adjustment=new_confidence - decision.confidence,
                timestamp=decision.timestamp
            )
            self.reflection_history.append(result)
            return result
            
        except Exception as e:
            return self._rule_based_shallow_reflect(decision)
    
    def deep_reflect(self, decision: DecisionRecord,
                    recent_decisions: List[DecisionRecord],
                    llm=None) -> ReflectionResult:
        """
        深层反思：关注决策背后的价值观和信念
        
        审视自身的核心信念和价值观，形成更稳定的长期观点。
        """
        if llm is None:
            return self._rule_based_deep_reflect(decision, recent_decisions)
        
        # 构建决策历史摘要
        history_summary = "\n".join([
            f"- {d.decision_content} (置信度: {d.confidence:.2f})"
            for d in recent_decisions[-5:]
        ]) if recent_decisions else "无历史决策"
        
        # 当前核心信念
        beliefs_summary = "\n".join([
            f"- {belief}: 强度 {strength:.2f}"
            for belief, strength in self.core_beliefs.items()
        ]) if self.core_beliefs else "尚未形成明确信念"
        
        prompt = f"""请进行深层反思，审视决策背后的价值观和信念。

## 当前决策
{decision.decision_content}

## 推理过程
{decision.reasoning}

## 最近决策历史
{history_summary}

## 当前核心信念
{beliefs_summary}

## 深层反思任务
请深入思考：
1. 这个决策反映了什么价值观？
2. 与我的核心信念是否一致？
3. 是否需要调整我的信念或价值优先级？

请按格式回答：
【价值观分析】（这个决策体现的价值观）
【信念一致性】一致/存在冲突
【信念更新建议】（如需更新，说明具体内容）
【核心洞见】（1句话总结）
【长期立场调整】保持/微调/重大调整
"""
        
        try:
            response = llm.send_message(prompt)
            
            # 提取洞见
            import re
            match = re.search(r'【核心洞见】\s*(.+?)(?:\n|$)', response)
            core_insight = match.group(1) if match else "保持当前立场"
            
            match = re.search(r'【价值观分析】\s*(.+?)(?:\n|【)', response, re.DOTALL)
            value_analysis = match.group(1).strip() if match else ""
            
            # 更新核心信念
            belief_updates = {}
            if '信念更新建议】' in response:
                belief_section = response.split('【信念更新建议】')[-1].split('【')[0]
                if belief_section.strip() and '无需更新' not in belief_section:
                    # 简单提取关键词作为新信念
                    keywords = [w for w in belief_section.split() if len(w) > 2][:3]
                    for kw in keywords:
                        self.core_beliefs[kw] = 0.6
                        belief_updates[kw] = 0.6
            
            result = ReflectionResult(
                level=ReflectionLevel.DEEP,
                trigger_reason=f"深层反思（第 {self.decision_count} 次决策）",
                original_decision=decision.decision_content,
                reflection_content=response,
                insights=[core_insight, value_analysis] if value_analysis else [core_insight],
                confidence_adjustment=0.1,  # 深层反思通常略微提升置信度
                belief_updates=belief_updates,
                timestamp=decision.timestamp
            )
            self.reflection_history.append(result)
            return result
            
        except Exception as e:
            return self._rule_based_deep_reflect(decision, recent_decisions)
    
    def _rule_based_shallow_reflect(self, decision: DecisionRecord) -> ReflectionResult:
        """基于规则的浅层反思"""
        adjustment = 0.05 if decision.confidence < 0.4 else 0.0
        
        return ReflectionResult(
            level=ReflectionLevel.SHALLOW,
            trigger_reason="规则反思",
            original_decision=decision.decision_content,
            reflection_content="基于规则的快速评估",
            insights=["保持当前决策方向"],
            confidence_adjustment=adjustment,
            timestamp=decision.timestamp
        )
    
    def _rule_based_deep_reflect(self, decision: DecisionRecord,
                                 recent_decisions: List[DecisionRecord]) -> ReflectionResult:
        """基于规则的深层反思"""
        # 分析决策一致性
        consistency = 0.7
        if recent_decisions:
            same_direction = sum(
                1 for d in recent_decisions[-3:]
                if d.decision_content[:10] == decision.decision_content[:10]
            )
            consistency = same_direction / min(3, len(recent_decisions))
        
        insight = "决策方向稳定" if consistency > 0.5 else "决策方向波动，需要明确立场"
        
        return ReflectionResult(
            level=ReflectionLevel.DEEP,
            trigger_reason="周期性深层反思",
            original_decision=decision.decision_content,
            reflection_content=f"一致性分析: {consistency:.2f}",
            insights=[insight],
            confidence_adjustment=0.1 if consistency > 0.5 else -0.05,
            timestamp=decision.timestamp
        )
    
    def get_reflection_summary(self) -> Dict[str, Any]:
        """获取反思摘要"""
        shallow_count = sum(1 for r in self.reflection_history if r.level == ReflectionLevel.SHALLOW)
        deep_count = sum(1 for r in self.reflection_history if r.level == ReflectionLevel.DEEP)
        
        return {
            'total_reflections': len(self.reflection_history),
            'shallow_reflections': shallow_count,
            'deep_reflections': deep_count,
            'core_beliefs': self.core_beliefs,
            'value_priorities': self.value_priorities,
            'recent_insights': [r.insights[0] for r in self.reflection_history[-3:]] if self.reflection_history else []
        }


class MetaCognitionModule(BaseAgentComponent):
    """
    元认知模块
    
    支持多层次反思机制（符合 Proposal 要求）：
    - 浅层反思：快速评估具体决策的对错
    - 深层反思：审视决策背后的价值观和信念
    """
    
    def __init__(self, agent,
                 reflection_threshold: float = 0.6,
                 max_reflections_per_round: int = 3):
        super().__init__(agent.component_id + "_metacog", 'metacognition', agent)
        
        self.confidence_estimator = ConfidenceEstimator()
        self.reflection_threshold = reflection_threshold
        self.max_reflections = max_reflections_per_round
        
        self.reflections_this_round = 0
        self.decision_history: List[DecisionRecord] = []
        self.confidence_history: List[Tuple[int, float]] = []
        
        # 多层次反思模块
        self.multi_level_reflection = MultiLevelReflection(
            shallow_threshold=reflection_threshold,
            deep_threshold=reflection_threshold - 0.2,
            deep_reflection_interval=5
        )
    
    def evaluate_decision(self, decision: DecisionRecord) -> Dict[str, Any]:
        """评估决策并决定是否需要反思"""
        confidence = self.confidence_estimator.estimate(
            decision, self.decision_history
        )
        
        decision.confidence = confidence
        decision.evaluation_scores['confidence'] = confidence
        
        self.decision_history.append(decision)
        self.confidence_history.append((decision.timestamp, confidence))
        
        # 确定反思层次
        reflection_level = self.multi_level_reflection.determine_reflection_level(
            confidence, decision
        )
        needs_reflection = reflection_level is not None
        
        return {
            'confidence': confidence,
            'needs_reflection': needs_reflection,
            'reflection_level': reflection_level.value if reflection_level else None,
            'reflection_reason': self._get_reflection_reason(confidence) if needs_reflection else None,
            'decision_record': decision
        }
    
    def _should_trigger_reflection(self, confidence: float) -> bool:
        """判断是否应该触发反思"""
        if self.reflections_this_round >= self.max_reflections:
            return False
        
        if confidence < self.reflection_threshold:
            return True
        
        if len(self.confidence_history) >= 2:
            prev_confidence = self.confidence_history[-2][1]
            if confidence < prev_confidence - 0.2:
                return True
        
        return False
    
    def _get_reflection_reason(self, confidence: float) -> str:
        """获取反思原因"""
        reasons = []
        
        if confidence < self.reflection_threshold:
            reasons.append(f"置信度过低 ({confidence:.2f} < {self.reflection_threshold})")
        
        if len(self.confidence_history) >= 2:
            prev_confidence = self.confidence_history[-2][1]
            if confidence < prev_confidence - 0.2:
                reasons.append(f"置信度急剧下降 ({prev_confidence:.2f} -> {confidence:.2f})")
        
        return "; ".join(reasons) if reasons else "常规反思"
    
    def trigger_reflection(self, reflection_chain=None, llm=None) -> Optional[str]:
        """
        触发反思过程
        
        支持多层次反思：自动判断需要浅层还是深层反思
        """
        if self.reflections_this_round >= self.max_reflections:
            return None
        
        self.reflections_this_round += 1
        
        if not self.decision_history:
            return None
        
        latest_decision = self.decision_history[-1]
        confidence = latest_decision.confidence
        
        # 确定反思层次
        reflection_level = self.multi_level_reflection.determine_reflection_level(
            confidence, latest_decision
        )
        
        if reflection_level is None:
            return None
        
        # 执行对应层次的反思
        if reflection_level == ReflectionLevel.SHALLOW:
            result = self.multi_level_reflection.shallow_reflect(latest_decision, llm)
        else:
            result = self.multi_level_reflection.deep_reflect(
                latest_decision, self.decision_history, llm
            )
        
        # 应用置信度调整
        if result.confidence_adjustment != 0:
            new_confidence = max(0.1, min(1.0, confidence + result.confidence_adjustment))
            latest_decision.confidence = new_confidence
            self.confidence_history[-1] = (latest_decision.timestamp, new_confidence)
        
        return result.reflection_content
    
    def trigger_multi_level_reflection(self, llm=None) -> Optional[ReflectionResult]:
        """
        触发多层次反思并返回完整结果
        
        Args:
            llm: LLM 接口
            
        Returns:
            反思结果，包含层次、洞见和信念更新
        """
        if self.reflections_this_round >= self.max_reflections:
            return None
        
        if not self.decision_history:
            return None
        
        self.reflections_this_round += 1
        latest_decision = self.decision_history[-1]
        confidence = latest_decision.confidence
        
        reflection_level = self.multi_level_reflection.determine_reflection_level(
            confidence, latest_decision
        )
        
        if reflection_level is None:
            return None
        
        if reflection_level == ReflectionLevel.SHALLOW:
            return self.multi_level_reflection.shallow_reflect(latest_decision, llm)
        else:
            return self.multi_level_reflection.deep_reflect(
                latest_decision, self.decision_history, llm
            )
    
    def _analyze_confidence_trend(self) -> Dict[str, Any]:
        """分析置信度趋势"""
        if len(self.confidence_history) < 2:
            return {'trend': 'stable', 'change': 0}
        
        recent = [c for _, c in self.confidence_history[-5:]]
        
        if len(recent) < 2:
            return {'trend': 'stable', 'change': 0}
        
        avg_change = (recent[-1] - recent[0]) / len(recent)
        
        if avg_change > 0.05:
            trend = 'increasing'
        elif avg_change < -0.05:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change': avg_change,
            'recent_values': recent,
            'average': statistics.mean(recent)
        }
    
    def reset_round(self):
        """重置轮次计数器"""
        self.reflections_this_round = 0
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        """获取认知状态摘要"""
        base_summary = {
            'total_decisions': len(self.decision_history),
            'reflections_this_round': self.reflections_this_round,
            'confidence_trend': self._analyze_confidence_trend(),
            'average_confidence': statistics.mean(
                [c for _, c in self.confidence_history]
            ) if self.confidence_history else 0.5,
            'low_confidence_decisions': sum(
                1 for d in self.decision_history 
                if d.confidence < self.reflection_threshold
            )
        }
        
        # 添加多层次反思摘要
        base_summary['multi_level_reflection'] = self.multi_level_reflection.get_reflection_summary()
        
        return base_summary


class DecisionEvaluator:
    """决策评估器"""
    
    def __init__(self):
        self.confidence_estimator = ConfidenceEstimator()
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate(self, decision: DecisionRecord,
                historical: Optional[List[DecisionRecord]] = None,
                group_decisions: Optional[List[DecisionRecord]] = None) -> Dict[str, float]:
        """全面评估决策"""
        scores = {}
        
        scores[EvaluationDimension.CONFIDENCE.value] = self.confidence_estimator.estimate(
            decision, historical
        )
        
        scores[EvaluationDimension.CONSISTENCY.value] = self._evaluate_consistency(
            decision, historical
        )
        
        scores[EvaluationDimension.RATIONALITY.value] = self._evaluate_rationality(
            decision
        )
        
        scores[EvaluationDimension.DEPTH.value] = self._evaluate_depth(
            decision.reasoning
        )
        
        scores[EvaluationDimension.COHERENCE.value] = self._evaluate_coherence(
            decision
        )
        
        if group_decisions:
            scores[EvaluationDimension.SOCIAL_CONSENSUS.value] = self._evaluate_consensus(
                decision, group_decisions
            )
            scores[EvaluationDimension.POLARIZATION.value] = self._evaluate_polarization(
                group_decisions
            )
        
        self.evaluation_history.append({
            'decision_id': decision.decision_id,
            'timestamp': decision.timestamp,
            'scores': scores.copy()
        })
        
        return scores
    
    def _evaluate_consistency(self, current: DecisionRecord,
                             historical: Optional[List[DecisionRecord]]) -> float:
        """评估一致性"""
        if not historical:
            return 0.7
        
        recent = sorted(historical, key=lambda x: x.timestamp, reverse=True)[:5]
        
        similarities = []
        for hist in recent:
            sim = self._semantic_similarity(
                current.decision_content, hist.decision_content
            )
            similarities.append(sim)
        
        return statistics.mean(similarities) if similarities else 0.7
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.5
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        jaccard = intersection / union if union > 0 else 0
        return 0.5 + 0.5 * jaccard
    
    def _evaluate_rationality(self, decision: DecisionRecord) -> float:
        """评估合理性"""
        score = 0.5
        reasoning = decision.reasoning.lower()
        
        if decision.alternatives:
            score += 0.15
        
        logic_words = ['因为', '所以', '因此', 'because', 'therefore', 'thus']
        if any(w in reasoning for w in logic_words):
            score += 0.15
        
        if decision.context:
            context_refs = sum(
                1 for k, v in decision.context.items() 
                if str(v).lower() in reasoning
            )
            score += min(0.2, context_refs * 0.05)
        
        return min(1.0, score)
    
    def _evaluate_depth(self, reasoning: str) -> float:
        """评估推理深度"""
        if not reasoning:
            return 0.2
        
        word_count = len(reasoning.split())
        sentence_count = reasoning.count('.') + reasoning.count('。') + 1
        
        depth_score = 0.3
        
        if word_count > 30:
            depth_score += 0.1
        if word_count > 60:
            depth_score += 0.1
        if word_count > 100:
            depth_score += 0.1
        
        if sentence_count > 3:
            depth_score += 0.1
        if sentence_count > 5:
            depth_score += 0.1
        
        if any(c in reasoning for c in ['首先', '其次', '最后', 'firstly', 'secondly']):
            depth_score += 0.15
        
        return min(1.0, depth_score)
    
    def _evaluate_coherence(self, decision: DecisionRecord) -> float:
        """评估连贯性"""
        score = 0.5
        
        decision_words = set(decision.decision_content.lower().split())
        reasoning_words = set(decision.reasoning.lower().split())
        
        if decision_words and reasoning_words:
            overlap = len(decision_words & reasoning_words)
            coherence = overlap / min(len(decision_words), 10)
            score += min(0.3, coherence)
        
        conclusion_markers = ['结论', '因此', '所以', 'conclusion', 'therefore', '综上']
        if any(m in decision.reasoning.lower() for m in conclusion_markers):
            score += 0.1
        
        return min(1.0, score)
    
    def _evaluate_consensus(self, current: DecisionRecord,
                           group: List[DecisionRecord]) -> float:
        """评估与群体的共识程度"""
        if not group:
            return 0.5
        
        similarities = []
        for other in group:
            if other.decision_id != current.decision_id:
                sim = self._semantic_similarity(
                    current.decision_content, other.decision_content
                )
                similarities.append(sim)
        
        return statistics.mean(similarities) if similarities else 0.5
    
    def _evaluate_polarization(self, group: List[DecisionRecord]) -> float:
        """评估群体极化程度"""
        if len(group) < 2:
            return 0.0
        
        similarities = []
        for i, d1 in enumerate(group):
            for d2 in group[i+1:]:
                sim = self._semantic_similarity(
                    d1.decision_content, d2.decision_content
                )
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        variance = statistics.variance(similarities) if len(similarities) > 1 else 0
        return min(1.0, variance * 4)
    
    def get_aggregate_scores(self, 
                            decisions: List[DecisionRecord]) -> Dict[str, float]:
        """获取一组决策的聚合评分"""
        if not decisions:
            return {}
        
        all_scores: Dict[str, List[float]] = {}
        
        for decision in decisions:
            scores = self.evaluate(decision)
            for dim, score in scores.items():
                if dim not in all_scores:
                    all_scores[dim] = []
                all_scores[dim].append(score)
        
        return {
            dim: statistics.mean(scores)
            for dim, scores in all_scores.items()
        }
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """生成评估报告"""
        if not self.evaluation_history:
            return {'error': '暂无评估记录'}
        
        dimension_scores: Dict[str, List[float]] = {}
        for record in self.evaluation_history:
            for dim, score in record['scores'].items():
                if dim not in dimension_scores:
                    dimension_scores[dim] = []
                dimension_scores[dim].append(score)
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'dimension_averages': {
                dim: statistics.mean(scores)
                for dim, scores in dimension_scores.items()
            },
            'dimension_std': {
                dim: statistics.stdev(scores) if len(scores) > 1 else 0
                for dim, scores in dimension_scores.items()
            }
        }

