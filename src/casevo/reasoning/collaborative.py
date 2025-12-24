"""
协同决策模块

实现多智能体协同决策机制。
"""

import statistics
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from casevo.core.component import BaseModelComponent


class NegotiationStatus(Enum):
    """协商状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CONVERGED = "converged"
    DEADLOCKED = "deadlocked"
    COMPLETED = "completed"


class DecisionMode(Enum):
    """决策模式"""
    DISTRIBUTED = "distributed"
    CENTRALIZED = "centralized"
    HYBRID = "hybrid"


@dataclass
class Message:
    """标准化消息格式"""
    sender_id: str
    receiver_id: Optional[str]
    message_type: str
    timestamp: int
    round_num: int
    
    opinion: str
    reasoning: str
    confidence: float
    
    evidence: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    compromise_proposal: Optional[str] = None
    position_change: str = "unchanged"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type,
            'timestamp': self.timestamp,
            'round_num': self.round_num,
            'opinion': self.opinion,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'references': self.references,
            'compromise_proposal': self.compromise_proposal,
            'position_change': self.position_change
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建"""
        return cls(
            sender_id=data['sender_id'],
            receiver_id=data.get('receiver_id'),
            message_type=data.get('message_type', 'opinion'),
            timestamp=data.get('timestamp', 0),
            round_num=data.get('round_num', 0),
            opinion=data.get('opinion', ''),
            reasoning=data.get('reasoning', ''),
            confidence=data.get('confidence', 0.5),
            evidence=data.get('evidence', []),
            references=data.get('references', []),
            compromise_proposal=data.get('compromise_proposal'),
            position_change=data.get('position_change', 'unchanged')
        )


@dataclass
class NegotiationRound:
    """协商轮次记录"""
    round_num: int
    messages: List[Message]
    consensus_level: float
    main_disagreements: List[str]
    timestamp: int


class NegotiationProtocol(ABC):
    """协商协议抽象基类"""
    
    @abstractmethod
    def initialize(self, participants: List[str], topic: str):
        pass
    
    @abstractmethod
    def submit_position(self, agent_id: str, message: Message):
        pass
    
    @abstractmethod
    def get_others_positions(self, agent_id: str) -> List[Message]:
        pass
    
    @abstractmethod
    def advance_round(self) -> bool:
        pass
    
    @abstractmethod
    def check_convergence(self) -> Tuple[bool, float]:
        pass
    
    @abstractmethod
    def get_final_result(self) -> Dict[str, Any]:
        pass


class StandardNegotiationProtocol(NegotiationProtocol):
    """标准协商协议实现"""
    
    def __init__(self, max_rounds: int = 10,
                 convergence_threshold: float = 0.05,
                 convergence_rounds: int = 2):
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.convergence_rounds = convergence_rounds
        
        self.participants: List[str] = []
        self.topic: str = ""
        self.current_round: int = 0
        self.status: NegotiationStatus = NegotiationStatus.PENDING
        
        self.round_positions: Dict[int, Dict[str, Message]] = {}
        self.round_history: List[NegotiationRound] = []
        self.consensus_history: List[float] = []
    
    def initialize(self, participants: List[str], topic: str):
        """初始化协商"""
        self.participants = participants
        self.topic = topic
        self.current_round = 0
        self.status = NegotiationStatus.IN_PROGRESS
        self.round_positions = {}
        self.round_history = []
        self.consensus_history = []
    
    def submit_position(self, agent_id: str, message: Message):
        """提交立场"""
        if self.status != NegotiationStatus.IN_PROGRESS:
            raise Exception(f"协商未在进行中：{self.status}")
        
        if agent_id not in self.participants:
            raise Exception(f"非参与者：{agent_id}")
        
        if self.current_round not in self.round_positions:
            self.round_positions[self.current_round] = {}
        
        self.round_positions[self.current_round][agent_id] = message
    
    def get_others_positions(self, agent_id: str) -> List[Message]:
        """获取其他参与者的立场"""
        if self.current_round == 0:
            return []
        
        prev_round = self.current_round - 1
        if prev_round not in self.round_positions:
            return []
        
        return [
            msg for aid, msg in self.round_positions[prev_round].items()
            if aid != agent_id
        ]
    
    def advance_round(self) -> bool:
        """推进到下一轮"""
        current_submissions = self.round_positions.get(self.current_round, {})
        if set(current_submissions.keys()) != set(self.participants):
            missing = set(self.participants) - set(current_submissions.keys())
            raise Exception(f"以下参与者未提交立场：{missing}")
        
        consensus = self._calculate_consensus()
        self.consensus_history.append(consensus)
        
        disagreements = self._identify_disagreements()
        
        round_record = NegotiationRound(
            round_num=self.current_round,
            messages=list(current_submissions.values()),
            consensus_level=consensus,
            main_disagreements=disagreements,
            timestamp=max(m.timestamp for m in current_submissions.values())
        )
        self.round_history.append(round_record)
        
        converged, _ = self.check_convergence()
        if converged:
            self.status = NegotiationStatus.CONVERGED
            return False
        
        if self.current_round >= self.max_rounds - 1:
            self.status = NegotiationStatus.COMPLETED
            return False
        
        if self._is_deadlocked():
            self.status = NegotiationStatus.DEADLOCKED
            return False
        
        self.current_round += 1
        return True
    
    def check_convergence(self) -> Tuple[bool, float]:
        """检查是否收敛"""
        if len(self.consensus_history) < self.convergence_rounds:
            return False, self.consensus_history[-1] if self.consensus_history else 0
        
        recent = self.consensus_history[-self.convergence_rounds:]
        changes = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        
        if all(c < self.convergence_threshold for c in changes):
            return True, recent[-1]
        
        return False, recent[-1]
    
    def _calculate_consensus(self) -> float:
        """计算当前轮次的共识水平"""
        messages = list(self.round_positions.get(self.current_round, {}).values())
        
        if len(messages) < 2:
            return 1.0
        
        similarities = []
        for i, m1 in enumerate(messages):
            for m2 in messages[i+1:]:
                sim = self._opinion_similarity(m1.opinion, m2.opinion)
                weight = (m1.confidence + m2.confidence) / 2
                similarities.append(sim * weight)
        
        return statistics.mean(similarities) if similarities else 0.5
    
    def _opinion_similarity(self, op1: str, op2: str) -> float:
        """计算两个意见的相似度"""
        words1 = set(op1.lower().split())
        words2 = set(op2.lower().split())
        
        if not words1 or not words2:
            return 0.5
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.5
    
    def _identify_disagreements(self) -> List[str]:
        """识别主要分歧点"""
        messages = list(self.round_positions.get(self.current_round, {}).values())
        
        if len(messages) < 2:
            return []
        
        all_words: Dict[str, int] = {}
        for msg in messages:
            for word in msg.opinion.lower().split():
                all_words[word] = all_words.get(word, 0) + 1
        
        partial_consensus = [
            word for word, count in all_words.items()
            if 0 < count < len(messages)
        ]
        
        return partial_consensus[:5]
    
    def _is_deadlocked(self) -> bool:
        """检测是否陷入僵局"""
        if len(self.consensus_history) < 5:
            return False
        
        recent = self.consensus_history[-5:]
        variance = statistics.variance(recent) if len(recent) > 1 else 0
        
        return variance < 0.01 and recent[-1] < 0.3
    
    def get_final_result(self) -> Dict[str, Any]:
        """获取最终协商结果"""
        final_positions = self.round_positions.get(self.current_round, {})
        aggregated = self._aggregate_opinions(list(final_positions.values()))
        
        return {
            'status': self.status.value,
            'total_rounds': self.current_round + 1,
            'final_consensus': self.consensus_history[-1] if self.consensus_history else 0,
            'consensus_history': self.consensus_history,
            'final_positions': {
                aid: msg.to_dict() for aid, msg in final_positions.items()
            },
            'aggregated_opinion': aggregated,
            'round_history': [
                {
                    'round': r.round_num,
                    'consensus': r.consensus_level,
                    'disagreements': r.main_disagreements
                }
                for r in self.round_history
            ]
        }
    
    def _aggregate_opinions(self, messages: List[Message]) -> Dict[str, Any]:
        """聚合多个意见"""
        if not messages:
            return {'opinion': '', 'confidence': 0}
        
        weighted_opinions = [
            (msg.opinion, msg.confidence) for msg in messages
        ]
        
        best = max(weighted_opinions, key=lambda x: x[1])
        avg_confidence = statistics.mean(msg.confidence for msg in messages)
        
        return {
            'representative_opinion': best[0],
            'average_confidence': avg_confidence,
            'opinion_count': len(messages),
            'high_confidence_count': sum(1 for m in messages if m.confidence > 0.7)
        }


class DistributedConsensus:
    """分布式共识算法"""
    
    def __init__(self, influence_decay: float = 0.1, stubbornness: float = 0.3):
        self.influence_decay = influence_decay
        self.stubbornness = stubbornness
        
        self.agent_positions: Dict[str, Dict[str, Any]] = {}
        self.influence_network: Dict[str, Dict[str, float]] = {}
    
    def register_agent(self, agent_id: str, initial_position: Dict[str, Any]):
        """注册智能体及其初始立场"""
        self.agent_positions[agent_id] = {
            'opinion': initial_position.get('opinion', ''),
            'confidence': initial_position.get('confidence', 0.5),
            'reasoning': initial_position.get('reasoning', '')
        }
        self.influence_network[agent_id] = {}
    
    def set_influence(self, from_agent: str, to_agent: str, influence: float):
        """设置智能体间的影响力"""
        if from_agent not in self.influence_network:
            self.influence_network[from_agent] = {}
        self.influence_network[from_agent][to_agent] = influence
    
    def exchange_influence(self, agent_id: str, 
                          neighbor_messages: List[Message]) -> Dict[str, Any]:
        """与邻居交换影响"""
        if agent_id not in self.agent_positions:
            raise Exception(f"未注册的智能体：{agent_id}")
        
        current = self.agent_positions[agent_id]
        
        total_influence = 0
        weighted_confidence = 0
        
        for msg in neighbor_messages:
            neighbor_id = msg.sender_id
            influence = self.influence_network.get(agent_id, {}).get(neighbor_id, 0.5)
            
            similarity = self._opinion_similarity(current['opinion'], msg.opinion)
            
            effective_influence = influence * msg.confidence * (0.5 + 0.5 * similarity)
            total_influence += effective_influence
            weighted_confidence += msg.confidence * effective_influence
        
        if total_influence > 0:
            stubbornness_factor = self.stubbornness
            neighbor_factor = 1 - stubbornness_factor
            
            new_confidence = (
                stubbornness_factor * current['confidence'] +
                neighbor_factor * (weighted_confidence / total_influence)
            )
            
            current['confidence'] = max(0.1, min(1.0, new_confidence))
        
        return current
    
    def _opinion_similarity(self, op1: str, op2: str) -> float:
        """计算意见相似度"""
        words1 = set(op1.lower().split())
        words2 = set(op2.lower().split())
        
        if not words1 or not words2:
            return 0.5
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.5
    
    def get_global_consensus(self) -> float:
        """计算全局共识水平"""
        positions = list(self.agent_positions.values())
        
        if len(positions) < 2:
            return 1.0
        
        similarities = []
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                sim = self._opinion_similarity(p1['opinion'], p2['opinion'])
                similarities.append(sim)
        
        return statistics.mean(similarities) if similarities else 0.5


class CentralAggregator:
    """集中式结果聚合器"""
    
    def __init__(self, aggregation_method: str = "weighted_majority"):
        self.aggregation_method = aggregation_method
        self.collected_opinions: List[Message] = []
    
    def collect(self, message: Message):
        """收集意见"""
        self.collected_opinions.append(message)
    
    def clear(self):
        """清除收集的意见"""
        self.collected_opinions = []
    
    def aggregate(self) -> Dict[str, Any]:
        """执行聚合"""
        if not self.collected_opinions:
            return {'error': '没有收集到意见'}
        
        if self.aggregation_method == "majority":
            return self._simple_majority()
        elif self.aggregation_method == "weighted_majority":
            return self._weighted_majority()
        elif self.aggregation_method == "consensus":
            return self._find_consensus()
        else:
            return self._weighted_majority()
    
    def _simple_majority(self) -> Dict[str, Any]:
        """简单多数决"""
        opinion_counts: Dict[str, int] = {}
        for msg in self.collected_opinions:
            key = msg.opinion[:20] if len(msg.opinion) > 20 else msg.opinion
            opinion_counts[key] = opinion_counts.get(key, 0) + 1
        
        if not opinion_counts:
            return {'result': None, 'support': 0}
        
        winner = max(opinion_counts.items(), key=lambda x: x[1])
        
        return {
            'result': winner[0],
            'support_count': winner[1],
            'total_votes': len(self.collected_opinions),
            'support_ratio': winner[1] / len(self.collected_opinions)
        }
    
    def _weighted_majority(self) -> Dict[str, Any]:
        """置信度加权多数决"""
        opinion_weights: Dict[str, float] = {}
        opinion_full: Dict[str, str] = {}
        
        for msg in self.collected_opinions:
            key = msg.opinion[:20] if len(msg.opinion) > 20 else msg.opinion
            opinion_weights[key] = opinion_weights.get(key, 0) + msg.confidence
            if key not in opinion_full:
                opinion_full[key] = msg.opinion
        
        if not opinion_weights:
            return {'result': None, 'weighted_support': 0}
        
        winner_key = max(opinion_weights.items(), key=lambda x: x[1])[0]
        total_weight = sum(opinion_weights.values())
        
        return {
            'result': opinion_full.get(winner_key, winner_key),
            'weighted_support': opinion_weights[winner_key],
            'total_weight': total_weight,
            'support_ratio': opinion_weights[winner_key] / total_weight if total_weight > 0 else 0,
            'average_confidence': statistics.mean(m.confidence for m in self.collected_opinions)
        }
    
    def _find_consensus(self) -> Dict[str, Any]:
        """寻找共识点"""
        if len(self.collected_opinions) < 2:
            return self._weighted_majority()
        
        all_words: List[Set[str]] = [
            set(msg.opinion.lower().split()) for msg in self.collected_opinions
        ]
        
        word_freq: Dict[str, int] = {}
        for word_set in all_words:
            for word in word_set:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        threshold = len(self.collected_opinions) * 0.6
        consensus_words = [
            word for word, freq in word_freq.items()
            if freq >= threshold
        ]
        
        consensus_elements = ' '.join(consensus_words[:10])
        consensus_level = len(consensus_words) / max(len(word_freq), 1)
        
        return {
            'consensus_elements': consensus_elements,
            'consensus_level': consensus_level,
            'high_frequency_words': consensus_words[:10],
            'participating_agents': len(self.collected_opinions),
            'fallback_result': self._weighted_majority()
        }


class CollaborativeDecisionMaker(BaseModelComponent):
    """协同决策主类"""
    
    def __init__(self, model, 
                 decision_mode: DecisionMode = DecisionMode.HYBRID,
                 negotiation_protocol: Optional[NegotiationProtocol] = None):
        super().__init__("collaborative_decision", "decision_maker", model)
        
        self.decision_mode = decision_mode
        self.protocol = negotiation_protocol or StandardNegotiationProtocol()
        self.distributed = DistributedConsensus()
        self.aggregator = CentralAggregator()
        
        self.current_topic: str = ""
        self.participants: List[str] = []
        self.decision_history: List[Dict[str, Any]] = []
    
    def start_negotiation(self, topic: str, participant_ids: List[str]):
        """开始新的协商过程"""
        self.current_topic = topic
        self.participants = participant_ids
        self.protocol.initialize(participant_ids, topic)
        self.aggregator.clear()
        
        for pid in participant_ids:
            self.distributed.register_agent(pid, {
                'opinion': '',
                'confidence': 0.5
            })
    
    def submit_opinion(self, agent_id: str, 
                      opinion: str, 
                      reasoning: str,
                      confidence: float,
                      **kwargs) -> bool:
        """提交智能体意见"""
        try:
            message = Message(
                sender_id=agent_id,
                receiver_id=None,
                message_type='opinion',
                timestamp=self.model.schedule.time if hasattr(self.model, 'schedule') else 0,
                round_num=self.protocol.current_round,
                opinion=opinion,
                reasoning=reasoning,
                confidence=confidence,
                evidence=kwargs.get('evidence', []),
                references=kwargs.get('references', []),
                compromise_proposal=kwargs.get('compromise_proposal'),
                position_change=kwargs.get('position_change', 'unchanged')
            )
            
            self.protocol.submit_position(agent_id, message)
            self.aggregator.collect(message)
            
            return True
        except Exception as e:
            print(f"提交意见失败: {e}")
            return False
    
    def get_context_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体的协商上下文"""
        others = self.protocol.get_others_positions(agent_id)
        
        return {
            'topic': self.current_topic,
            'round': self.protocol.current_round,
            'max_rounds': self.protocol.max_rounds,
            'my_position': self.distributed.agent_positions.get(agent_id, {}),
            'others_positions': [m.to_dict() for m in others],
            'previous_rounds': [
                {
                    'round': r.round_num,
                    'consensus_level': r.consensus_level,
                    'main_disagreements': r.main_disagreements
                }
                for r in self.protocol.round_history
            ]
        }
    
    def advance_negotiation(self) -> Dict[str, Any]:
        """推进协商进程"""
        try:
            can_continue = self.protocol.advance_round()
            converged, consensus = self.protocol.check_convergence()
            
            return {
                'can_continue': can_continue,
                'current_round': self.protocol.current_round,
                'status': self.protocol.status.value,
                'consensus_level': consensus,
                'converged': converged
            }
        except Exception as e:
            return {
                'error': str(e),
                'can_continue': False
            }
    
    def finalize_decision(self) -> Dict[str, Any]:
        """完成决策并返回结果"""
        result = {}
        
        if self.decision_mode == DecisionMode.DISTRIBUTED:
            result = {
                'mode': 'distributed',
                'global_consensus': self.distributed.get_global_consensus(),
                'agent_positions': self.distributed.agent_positions
            }
        
        elif self.decision_mode == DecisionMode.CENTRALIZED:
            aggregated = self.aggregator.aggregate()
            result = {
                'mode': 'centralized',
                'aggregated_result': aggregated
            }
        
        else:  # HYBRID
            protocol_result = self.protocol.get_final_result()
            aggregated = self.aggregator.aggregate()
            
            result = {
                'mode': 'hybrid',
                'negotiation_result': protocol_result,
                'aggregated_result': aggregated,
                'global_consensus': self.distributed.get_global_consensus()
            }
        
        result['topic'] = self.current_topic
        result['participants'] = self.participants
        result['total_rounds'] = self.protocol.current_round + 1
        
        self.decision_history.append(result)
        
        return result
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """获取决策统计信息"""
        if not self.decision_history:
            return {'total_decisions': 0}
        
        avg_rounds = statistics.mean(
            d.get('total_rounds', 0) for d in self.decision_history
        )
        
        consensus_levels = [
            d.get('global_consensus', 0) for d in self.decision_history
            if 'global_consensus' in d
        ]
        
        return {
            'total_decisions': len(self.decision_history),
            'average_rounds': avg_rounds,
            'average_consensus': statistics.mean(consensus_levels) if consensus_levels else 0
        }

