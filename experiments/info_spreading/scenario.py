"""
信息传播实验场景

研究虚假信息在社交网络中的传播动力学：
- 200 个节点的无标度网络
- 虚假信息传播模拟
- 可信度判断机制
- 传播抑制策略评估
"""

import mesa
import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json
import math
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from casevo import (
    AgentBase,
    DecisionEvaluator, 
    DecisionRecord,
    ConfidenceEstimator
)


class InformationType(Enum):
    """信息类型"""
    TRUE = "true"           # 真实信息
    FALSE = "false"         # 虚假信息
    UNKNOWN = "unknown"     # 未知/未验证


class AgentType(Enum):
    """智能体类型"""
    NORMAL = "normal"           # 普通用户
    SKEPTIC = "skeptic"         # 怀疑者（高辨别力）
    GULLIBLE = "gullible"       # 易信者（低辨别力）
    INFLUENCER = "influencer"   # 影响者（高传播力）


@dataclass
class Information:
    """信息实体"""
    info_id: str
    content: str
    true_type: InformationType  # 实际类型
    source_credibility: float   # 来源可信度 (0-1)
    spread_count: int = 0       # 传播次数
    first_seen_time: int = 0    # 首次出现时间
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'info_id': self.info_id,
            'content': self.content,
            'true_type': self.true_type.value,
            'source_credibility': self.source_credibility,
            'spread_count': self.spread_count
        }


@dataclass
class BeliefState:
    """信念状态"""
    information: Information
    believed: bool                    # 是否相信
    confidence: float                 # 置信度
    decision_reasoning: str           # 决策理由
    timestamp: int


class InfoSpreadingAgent(AgentBase):
    """
    信息传播智能体
    
    能够接收、评估和传播信息的社交网络用户。
    """
    
    def __init__(self, unique_id: int, model: 'InfoSpreadingModel',
                 agent_type: AgentType, 
                 critical_thinking: float = 0.5,
                 spread_tendency: float = 0.5):
        """
        初始化信息传播智能体
        
        Args:
            unique_id: 唯一标识
            model: 模型实例
            agent_type: 智能体类型
            critical_thinking: 批判性思维能力 (0-1)
            spread_tendency: 传播倾向 (0-1)
        """
        description = self._generate_description(agent_type)
        context = {
            'agent_type': agent_type.value,
            'critical_thinking': critical_thinking,
            'spread_tendency': spread_tendency
        }
        
        super().__init__(unique_id, model, description, context)
        
        self.agent_type = agent_type
        self.critical_thinking = critical_thinking
        self.spread_tendency = spread_tendency
        
        # 信息状态
        self.received_info: Dict[str, Information] = {}
        self.believed_info: Dict[str, BeliefState] = {}
        self.spread_info: Set[str] = set()
        
        # 邻居（由网络设置）
        self.neighbors: List[int] = []
        
        # 决策评估
        self.evaluator = DecisionEvaluator()
        self.confidence_estimator = ConfidenceEstimator()
        
        # 统计
        self.correct_judgments = 0
        self.incorrect_judgments = 0
    
    def _generate_description(self, agent_type: AgentType) -> str:
        """生成智能体描述"""
        type_descriptions = {
            AgentType.NORMAL: "普通社交媒体用户，具有一般的信息辨别能力",
            AgentType.SKEPTIC: "怀疑型用户，对信息持谨慎态度，倾向于验证",
            AgentType.GULLIBLE: "易信型用户，较容易相信收到的信息",
            AgentType.INFLUENCER: "影响力用户，有大量关注者，传播能力强"
        }
        return type_descriptions.get(agent_type, "社交媒体用户")
    
    def receive_information(self, info: Information, from_agent_id: Optional[int] = None):
        """
        接收信息
        
        Args:
            info: 信息实体
            from_agent_id: 来源智能体ID
        """
        if info.info_id not in self.received_info:
            self.received_info[info.info_id] = info
            
            # 记录到记忆
            self.memory.add_short_memory(
                source=str(from_agent_id) if from_agent_id else "system",
                target=self.component_id,
                action="receive_info",
                content=f"收到信息: {info.content[:50]}..."
            )
    
    def evaluate_information(self, info: Information) -> Tuple[bool, float, str]:
        """
        评估信息可信度
        
        Args:
            info: 待评估的信息
            
        Returns:
            (是否相信, 置信度, 评估理由)
        """
        # 基础可信度评估
        base_credibility = info.source_credibility
        
        # 根据智能体类型调整
        if self.agent_type == AgentType.SKEPTIC:
            credibility_threshold = 0.7
            adjustment = -0.2
        elif self.agent_type == AgentType.GULLIBLE:
            credibility_threshold = 0.3
            adjustment = 0.2
        else:
            credibility_threshold = 0.5
            adjustment = 0.0
        
        # 批判性思维影响
        thinking_factor = self.critical_thinking
        
        # 综合评估
        perceived_credibility = base_credibility + adjustment
        perceived_credibility *= (0.5 + 0.5 * thinking_factor)
        
        # 检查逻辑一致性（简化：随机因素模拟）
        logic_check = random.random() * thinking_factor
        
        # 检查与已知信息的冲突
        conflict_penalty = self._check_conflicts(info)
        
        # 最终决策
        final_score = perceived_credibility + logic_check - conflict_penalty
        final_score = max(0, min(1, final_score))
        
        believed = final_score > credibility_threshold
        
        # 生成理由
        if believed:
            if final_score > 0.8:
                reasoning = "信息来源可靠，逻辑一致，选择相信"
            else:
                reasoning = "信息有一定可信度，暂时接受"
        else:
            if final_score < 0.3:
                reasoning = "信息来源不可靠，存在明显问题，不相信"
            else:
                reasoning = "信息可信度不足，保持怀疑"
        
        return believed, final_score, reasoning
    
    def _check_conflicts(self, info: Information) -> float:
        """检查与已有信息的冲突"""
        conflict_score = 0
        
        for existing_id, belief in self.believed_info.items():
            if belief.believed:
                # 简化：检查内容重叠度
                existing_words = set(belief.information.content.lower().split())
                new_words = set(info.content.lower().split())
                
                overlap = len(existing_words & new_words)
                if overlap > 3:
                    # 有重叠但类型不同，可能冲突
                    if belief.information.true_type != info.true_type:
                        conflict_score += 0.2
        
        return min(conflict_score, 0.5)
    
    def decide_to_spread(self, info: Information, belief: BeliefState) -> bool:
        """
        决定是否传播信息
        
        Args:
            info: 信息
            belief: 信念状态
            
        Returns:
            是否传播
        """
        if not belief.believed:
            return False
        
        # 传播概率受多因素影响
        spread_prob = self.spread_tendency * belief.confidence
        
        # 影响者更可能传播
        if self.agent_type == AgentType.INFLUENCER:
            spread_prob *= 1.5
        
        # 怀疑者不太传播
        if self.agent_type == AgentType.SKEPTIC:
            spread_prob *= 0.5
        
        # 已传播过的不再传播
        if info.info_id in self.spread_info:
            return False
        
        return random.random() < spread_prob
    
    def spread_to_neighbors(self, info: Information):
        """
        向邻居传播信息
        
        Args:
            info: 要传播的信息
        """
        for neighbor_id in self.neighbors:
            neighbor = self.model.schedule.agents[neighbor_id]
            if isinstance(neighbor, InfoSpreadingAgent):
                # 传播概率受边权重影响
                edge_weight = self.model.network[self.unique_id][neighbor_id].get('weight', 0.5)
                if random.random() < edge_weight:
                    neighbor.receive_information(info, self.unique_id)
        
        self.spread_info.add(info.info_id)
        info.spread_count += 1
    
    def process_received_info(self):
        """处理收到的信息"""
        for info_id, info in list(self.received_info.items()):
            if info_id not in self.believed_info:
                # 评估信息
                believed, confidence, reasoning = self.evaluate_information(info)
                
                belief = BeliefState(
                    information=info,
                    believed=believed,
                    confidence=confidence,
                    decision_reasoning=reasoning,
                    timestamp=self.model.schedule.time
                )
                
                self.believed_info[info_id] = belief
                
                # 记录决策
                decision = DecisionRecord(
                    decision_id=f"info_eval_{info_id}_{self.unique_id}",
                    timestamp=self.model.schedule.time,
                    agent_id=self.component_id,
                    decision_content=f"{'相信' if believed else '不相信'}信息",
                    reasoning=reasoning,
                    confidence=confidence
                )
                self.evaluator.evaluate(decision)
                
                # 统计准确性
                if believed and info.true_type == InformationType.TRUE:
                    self.correct_judgments += 1
                elif believed and info.true_type == InformationType.FALSE:
                    self.incorrect_judgments += 1
                elif not believed and info.true_type == InformationType.FALSE:
                    self.correct_judgments += 1
                elif not believed and info.true_type == InformationType.TRUE:
                    self.incorrect_judgments += 1
                
                # 决定是否传播
                if self.decide_to_spread(info, belief):
                    self.spread_to_neighbors(info)
    
    def get_judgment_accuracy(self) -> float:
        """获取判断准确率"""
        total = self.correct_judgments + self.incorrect_judgments
        if total == 0:
            return 0.5
        return self.correct_judgments / total
    
    def step(self):
        """每轮执行"""
        self.process_received_info()


class InfoSpreadingModel(mesa.Model):
    """
    信息传播模型
    
    管理信息在社交网络中的传播过程。
    """
    
    def __init__(self, num_agents: int = 200,
                 initial_infected: float = 0.1,
                 false_info_ratio: float = 0.3,
                 use_enhanced_evaluation: bool = True):
        """
        初始化信息传播模型
        
        Args:
            num_agents: 智能体数量
            initial_infected: 初始感染比例
            false_info_ratio: 虚假信息比例
            use_enhanced_evaluation: 是否使用增强评估
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.initial_infected = initial_infected
        self.false_info_ratio = false_info_ratio
        self.use_enhanced = use_enhanced_evaluation
        self.context = "社交网络信息传播"
        
        # 创建调度器
        self.schedule = mesa.time.RandomActivation(self)
        
        # 创建无标度网络
        self.network = nx.barabasi_albert_graph(num_agents, 3)
        
        # 为边添加权重
        for u, v in self.network.edges():
            self.network[u][v]['weight'] = random.uniform(0.3, 0.7)
        
        # 初始化记忆工厂（简化版）
        self.memory_factory = self._create_mock_memory_factory()
        
        # 创建智能体
        self._create_agents()
        
        # 信息库
        self.information_pool: Dict[str, Information] = {}
        self.info_counter = 0
        
        # 传播统计
        self.spread_history: List[Dict[str, Any]] = []
        
        # 初始化信息并传播
        self._initialize_information()
    
    def _create_mock_memory_factory(self):
        """创建模拟记忆工厂"""
        class MockMemory:
            def __init__(self, agent):
                self.short_memories = []
            
            def add_short_memory(self, source, target, action, content, ts=None):
                self.short_memories.append({
                    'source': source, 'target': target,
                    'action': action, 'content': content
                })
        
        class MockFactory:
            def create_memory(self, agent):
                return MockMemory(agent)
        
        return MockFactory()
    
    def _create_agents(self):
        """创建智能体"""
        # 类型分布
        type_distribution = {
            AgentType.NORMAL: 0.6,
            AgentType.SKEPTIC: 0.15,
            AgentType.GULLIBLE: 0.15,
            AgentType.INFLUENCER: 0.1
        }
        
        for i in range(self.num_agents):
            # 根据节点度数调整类型概率（高度数节点更可能是影响者）
            degree = self.network.degree(i)
            avg_degree = sum(dict(self.network.degree()).values()) / self.num_agents
            
            if degree > avg_degree * 2:
                # 高度数节点
                agent_type = random.choices(
                    [AgentType.INFLUENCER, AgentType.NORMAL],
                    weights=[0.4, 0.6]
                )[0]
            else:
                agent_type = random.choices(
                    list(type_distribution.keys()),
                    weights=list(type_distribution.values())
                )[0]
            
            # 根据类型设置属性
            if agent_type == AgentType.SKEPTIC:
                critical_thinking = random.uniform(0.7, 0.95)
                spread_tendency = random.uniform(0.2, 0.4)
            elif agent_type == AgentType.GULLIBLE:
                critical_thinking = random.uniform(0.2, 0.4)
                spread_tendency = random.uniform(0.5, 0.8)
            elif agent_type == AgentType.INFLUENCER:
                critical_thinking = random.uniform(0.4, 0.7)
                spread_tendency = random.uniform(0.7, 0.95)
            else:
                critical_thinking = random.uniform(0.4, 0.6)
                spread_tendency = random.uniform(0.4, 0.6)
            
            # 使用增强评估时提高批判性思维
            if self.use_enhanced:
                critical_thinking = min(1.0, critical_thinking + 0.1)
            
            agent = InfoSpreadingAgent(
                i, self, agent_type, critical_thinking, spread_tendency
            )
            self.schedule.add(agent)
            
            # 设置邻居
            agent.neighbors = list(self.network.neighbors(i))
    
    def _initialize_information(self):
        """初始化信息并传播给初始节点"""
        # 创建真实信息
        true_info = self._create_information(InformationType.TRUE)
        
        # 创建虚假信息
        false_info = self._create_information(InformationType.FALSE)
        
        # 选择初始感染节点
        num_initial = int(self.num_agents * self.initial_infected)
        initial_nodes = random.sample(range(self.num_agents), num_initial)
        
        for node_id in initial_nodes:
            agent = self.schedule.agents[node_id]
            if isinstance(agent, InfoSpreadingAgent):
                # 随机分配真假信息
                if random.random() < self.false_info_ratio:
                    agent.receive_information(false_info)
                else:
                    agent.receive_information(true_info)
    
    def _create_information(self, info_type: InformationType) -> Information:
        """创建信息"""
        self.info_counter += 1
        
        if info_type == InformationType.TRUE:
            content = f"真实信息_{self.info_counter}: 这是经过验证的事实..."
            credibility = random.uniform(0.6, 0.9)
        else:
            content = f"虚假信息_{self.info_counter}: 这是未经证实的说法..."
            credibility = random.uniform(0.3, 0.7)
        
        info = Information(
            info_id=f"info_{self.info_counter}",
            content=content,
            true_type=info_type,
            source_credibility=credibility,
            first_seen_time=self.schedule.time
        )
        
        self.information_pool[info.info_id] = info
        return info
    
    def inject_new_information(self, info_type: InformationType,
                               target_nodes: Optional[List[int]] = None):
        """
        注入新信息
        
        Args:
            info_type: 信息类型
            target_nodes: 目标节点（None则随机选择）
        """
        info = self._create_information(info_type)
        
        if target_nodes is None:
            # 随机选择一些节点
            num_targets = max(1, int(self.num_agents * 0.05))
            target_nodes = random.sample(range(self.num_agents), num_targets)
        
        for node_id in target_nodes:
            agent = self.schedule.agents[node_id]
            if isinstance(agent, InfoSpreadingAgent):
                agent.receive_information(info)
    
    def get_spread_statistics(self) -> Dict[str, Any]:
        """获取传播统计"""
        total_received = 0
        total_believed_true = 0
        total_believed_false = 0
        false_spread_count = 0
        
        for agent in self.schedule.agents:
            if isinstance(agent, InfoSpreadingAgent):
                total_received += len(agent.received_info)
                
                for info_id, belief in agent.believed_info.items():
                    if belief.believed:
                        if belief.information.true_type == InformationType.TRUE:
                            total_believed_true += 1
                        else:
                            total_believed_false += 1
        
        # 统计虚假信息传播次数
        for info in self.information_pool.values():
            if info.true_type == InformationType.FALSE:
                false_spread_count += info.spread_count
        
        return {
            'total_agents': self.num_agents,
            'total_info_received': total_received,
            'total_believed_true': total_believed_true,
            'total_believed_false': total_believed_false,
            'false_info_spread_count': false_spread_count,
            'false_belief_ratio': total_believed_false / (total_believed_true + total_believed_false + 1)
        }
    
    def get_agent_accuracy_stats(self) -> Dict[str, Any]:
        """获取智能体判断准确率统计"""
        accuracies = []
        type_accuracies: Dict[str, List[float]] = {
            t.value: [] for t in AgentType
        }
        
        for agent in self.schedule.agents:
            if isinstance(agent, InfoSpreadingAgent):
                acc = agent.get_judgment_accuracy()
                accuracies.append(acc)
                type_accuracies[agent.agent_type.value].append(acc)
        
        return {
            'overall_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'accuracy_by_type': {
                t: sum(accs) / len(accs) if accs else 0
                for t, accs in type_accuracies.items()
            },
            'min_accuracy': min(accuracies) if accuracies else 0,
            'max_accuracy': max(accuracies) if accuracies else 0
        }
    
    def step(self):
        """执行一轮模拟"""
        # 收集传播统计
        stats = self.get_spread_statistics()
        stats['round'] = self.schedule.time
        self.spread_history.append(stats)
        
        # 执行智能体步骤
        self.schedule.step()
        
        # 偶尔注入新信息
        if self.schedule.time % 3 == 0:
            info_type = InformationType.FALSE if random.random() < self.false_info_ratio else InformationType.TRUE
            self.inject_new_information(info_type)
    
    def run_simulation(self, rounds: int = 20) -> Dict[str, Any]:
        """运行完整模拟"""
        for _ in range(rounds):
            self.step()
        
        return {
            'spread_history': self.spread_history,
            'final_statistics': self.get_spread_statistics(),
            'accuracy_statistics': self.get_agent_accuracy_stats(),
            'network_stats': {
                'nodes': self.network.number_of_nodes(),
                'edges': self.network.number_of_edges(),
                'avg_degree': sum(dict(self.network.degree()).values()) / self.num_agents,
                'avg_clustering': nx.average_clustering(self.network)
            },
            'information_pool': {
                info_id: info.to_dict() 
                for info_id, info in self.information_pool.items()
            }
        }


def run_info_spreading_experiment(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    运行信息传播实验
    
    Args:
        config: 实验配置
        
    Returns:
        实验结果
    """
    if config is None:
        config = {
            'num_agents': 200,
            'initial_infected': 0.1,
            'false_info_ratio': 0.3,
            'use_enhanced_evaluation': True,
            'num_rounds': 20
        }
    
    model = InfoSpreadingModel(
        num_agents=config.get('num_agents', 200),
        initial_infected=config.get('initial_infected', 0.1),
        false_info_ratio=config.get('false_info_ratio', 0.3),
        use_enhanced_evaluation=config.get('use_enhanced_evaluation', True)
    )
    
    results = model.run_simulation(rounds=config.get('num_rounds', 20))
    results['config'] = config
    
    return results


if __name__ == "__main__":
    # 运行基线实验
    print("运行基线实验...")
    baseline_results = run_info_spreading_experiment({
        'use_enhanced_evaluation': False,
        'num_agents': 200,
        'num_rounds': 20
    })
    print(f"基线结果 - 虚假信息接受率: {baseline_results['final_statistics']['false_belief_ratio']:.3f}")
    print(f"基线判断准确率: {baseline_results['accuracy_statistics']['overall_accuracy']:.3f}")
    
    # 运行优化实验
    print("\n运行优化实验（增强评估）...")
    optimized_results = run_info_spreading_experiment({
        'use_enhanced_evaluation': True,
        'num_agents': 200,
        'num_rounds': 20
    })
    print(f"优化结果 - 虚假信息接受率: {optimized_results['final_statistics']['false_belief_ratio']:.3f}")
    print(f"优化判断准确率: {optimized_results['accuracy_statistics']['overall_accuracy']:.3f}")
    
    # 保存结果
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'info_spreading', 'baseline.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': baseline_results,
            'optimized': optimized_results
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n结果已保存到 experiments/results/info_spreading/baseline.json")

