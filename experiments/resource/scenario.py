"""
资源分配实验场景

模拟多智能体资源协商分配过程：
- 50 个智能体
- 固定总资源 1000 单位
- 多轮协商机制
- 公平性评估
"""

import mesa
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json
import statistics
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from casevo import (
    AgentBase,
    CollaborativeDecisionMaker, 
    Message, 
    StandardNegotiationProtocol,
    DecisionMode,
    DistributedConsensus,
    DecisionEvaluator, DecisionRecord
)


class AgentPriority(Enum):
    """智能体优先级"""
    CRITICAL = "critical"      # 关键需求
    HIGH = "high"              # 高优先级
    NORMAL = "normal"          # 普通
    LOW = "low"                # 低优先级


@dataclass
class ResourceNeed:
    """资源需求"""
    minimum: int           # 最低需求
    desired: int           # 期望需求
    maximum: int           # 最大可用量
    priority: AgentPriority
    justification: str     # 需求理由


@dataclass
class AllocationProposal:
    """分配提案"""
    agent_id: str
    requested_amount: int
    justification: str
    willingness_to_compromise: float  # 妥协意愿 (0-1)
    alternative_amounts: List[int]    # 可接受的替代数量


class ResourceAgent(AgentBase):
    """
    资源分配智能体
    
    参与资源协商，提出需求并与其他智能体协商。
    """
    
    def __init__(self, unique_id: int, model: 'ResourceAllocationModel',
                 need: ResourceNeed):
        """
        初始化资源智能体
        
        Args:
            unique_id: 唯一标识
            model: 模型实例
            need: 资源需求
        """
        description = self._generate_description(need)
        context = {
            'minimum_need': need.minimum,
            'desired_need': need.desired,
            'priority': need.priority.value
        }
        
        super().__init__(unique_id, model, description, context)
        
        self.need = need
        self.current_proposal: Optional[AllocationProposal] = None
        self.allocated_amount: int = 0
        self.satisfaction: float = 0.0
        self.negotiation_history: List[Dict[str, Any]] = []
        
        # 决策评估
        self.evaluator = DecisionEvaluator()
    
    def _generate_description(self, need: ResourceNeed) -> str:
        """生成智能体描述"""
        return f"""你是一个资源申请方，具有以下需求特征：
- 最低需求：{need.minimum} 单位
- 期望需求：{need.desired} 单位
- 优先级：{need.priority.value}
- 需求理由：{need.justification}

你需要在协商中争取足够的资源，同时考虑整体公平性。"""
    
    def create_initial_proposal(self) -> AllocationProposal:
        """创建初始提案"""
        # 根据需求和优先级确定请求量
        if self.need.priority == AgentPriority.CRITICAL:
            requested = self.need.desired
            compromise = 0.3
        elif self.need.priority == AgentPriority.HIGH:
            requested = int((self.need.minimum + self.need.desired) / 2 * 1.2)
            compromise = 0.5
        elif self.need.priority == AgentPriority.NORMAL:
            requested = self.need.desired
            compromise = 0.6
        else:
            requested = int((self.need.minimum + self.need.desired) / 2)
            compromise = 0.8
        
        # 生成可接受的替代数量
        alternatives = [
            int(requested * 0.9),
            int(requested * 0.8),
            self.need.minimum
        ]
        
        self.current_proposal = AllocationProposal(
            agent_id=self.component_id,
            requested_amount=requested,
            justification=self.need.justification,
            willingness_to_compromise=compromise,
            alternative_amounts=sorted(set(alternatives), reverse=True)
        )
        
        return self.current_proposal
    
    def respond_to_situation(self, total_requests: int, 
                            available_resources: int,
                            others_proposals: List[AllocationProposal]) -> AllocationProposal:
        """
        根据情况调整提案
        
        Args:
            total_requests: 总请求量
            available_resources: 可用资源
            others_proposals: 其他人的提案
            
        Returns:
            调整后的提案
        """
        # 计算资源紧张程度
        scarcity = total_requests / available_resources if available_resources > 0 else float('inf')
        
        if scarcity > 1.5:
            # 资源非常紧张，需要妥协
            adjustment_factor = 0.7
        elif scarcity > 1.2:
            # 资源略紧张
            adjustment_factor = 0.85
        else:
            # 资源充足
            adjustment_factor = 1.0
        
        # 根据优先级调整
        if self.need.priority == AgentPriority.CRITICAL:
            adjustment_factor = max(adjustment_factor, 0.9)
        
        new_request = max(
            self.need.minimum,
            int(self.current_proposal.requested_amount * adjustment_factor)
        )
        
        # 更新妥协意愿
        new_compromise = min(1.0, self.current_proposal.willingness_to_compromise + 0.1)
        
        self.current_proposal = AllocationProposal(
            agent_id=self.component_id,
            requested_amount=new_request,
            justification=f"考虑整体资源情况，调整请求至 {new_request}",
            willingness_to_compromise=new_compromise,
            alternative_amounts=[
                int(new_request * 0.9),
                int(new_request * 0.8),
                self.need.minimum
            ]
        )
        
        # 记录协商历史
        self.negotiation_history.append({
            'round': self.model.schedule.time,
            'request': new_request,
            'scarcity': scarcity,
            'adjustment': adjustment_factor
        })
        
        return self.current_proposal
    
    def receive_allocation(self, amount: int):
        """
        接收分配结果
        
        Args:
            amount: 分配的资源量
        """
        self.allocated_amount = amount
        
        # 计算满意度
        if amount >= self.need.desired:
            self.satisfaction = 1.0
        elif amount >= self.need.minimum:
            self.satisfaction = 0.5 + 0.5 * (amount - self.need.minimum) / (self.need.desired - self.need.minimum)
        else:
            self.satisfaction = 0.5 * amount / self.need.minimum if self.need.minimum > 0 else 0
        
        # 记录到记忆
        self.memory.add_short_memory(
            source="system",
            target=self.component_id,
            action="allocation",
            content=f"获得资源分配 {amount} 单位，满意度 {self.satisfaction:.2f}"
        )
    
    def step(self):
        """每轮执行"""
        pass  # 主要逻辑由模型协调


class ResourceAllocationModel(mesa.Model):
    """
    资源分配模型
    
    管理资源协商和分配过程。
    """
    
    def __init__(self, num_agents: int = 50,
                 total_resources: int = 1000,
                 max_negotiation_rounds: int = 10,
                 use_collaborative: bool = True):
        """
        初始化资源分配模型
        
        Args:
            num_agents: 智能体数量
            total_resources: 总资源量
            max_negotiation_rounds: 最大协商轮次
            use_collaborative: 是否使用协同决策模块
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.total_resources = total_resources
        self.max_rounds = max_negotiation_rounds
        self.use_collaborative = use_collaborative
        self.context = "资源分配协商"
        
        # 创建调度器
        self.schedule = mesa.time.BaseScheduler(self)
        
        # 初始化记忆工厂（简化版）
        self.memory_factory = self._create_mock_memory_factory()
        
        # 创建智能体
        self._create_agents()
        
        # 协商状态
        self.current_round = 0
        self.converged = False
        self.allocation_history: List[Dict[str, Any]] = []
        
        # 协同决策器
        if use_collaborative:
            self.decision_maker = CollaborativeDecisionMaker(
                self, DecisionMode.HYBRID
            )
    
    def _create_mock_memory_factory(self):
        """创建模拟记忆工厂"""
        class MockMemory:
            def __init__(self, agent):
                self.agent = agent
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
        """创建资源智能体"""
        # 优先级分布
        priority_distribution = {
            AgentPriority.CRITICAL: 0.1,
            AgentPriority.HIGH: 0.2,
            AgentPriority.NORMAL: 0.5,
            AgentPriority.LOW: 0.2
        }
        
        # 需求理由模板
        justifications = [
            "项目关键依赖",
            "日常运营需要",
            "扩展业务所需",
            "备用储备",
            "研发投入",
            "维护更新"
        ]
        
        for i in range(self.num_agents):
            # 随机选择优先级
            priority = random.choices(
                list(priority_distribution.keys()),
                weights=list(priority_distribution.values())
            )[0]
            
            # 根据优先级设置需求范围
            if priority == AgentPriority.CRITICAL:
                min_need = random.randint(25, 35)
                desired = random.randint(35, 50)
            elif priority == AgentPriority.HIGH:
                min_need = random.randint(20, 30)
                desired = random.randint(30, 40)
            elif priority == AgentPriority.NORMAL:
                min_need = random.randint(15, 25)
                desired = random.randint(25, 35)
            else:
                min_need = random.randint(10, 20)
                desired = random.randint(20, 30)
            
            need = ResourceNeed(
                minimum=min_need,
                desired=desired,
                maximum=desired + 10,
                priority=priority,
                justification=random.choice(justifications)
            )
            
            agent = ResourceAgent(i, self, need)
            self.schedule.add(agent)
    
    def collect_proposals(self) -> List[AllocationProposal]:
        """收集所有提案"""
        proposals = []
        for agent in self.schedule.agents:
            if isinstance(agent, ResourceAgent):
                if agent.current_proposal is None:
                    agent.create_initial_proposal()
                proposals.append(agent.current_proposal)
        return proposals
    
    def calculate_total_requests(self, proposals: List[AllocationProposal]) -> int:
        """计算总请求量"""
        return sum(p.requested_amount for p in proposals)
    
    def negotiate_round(self) -> Dict[str, Any]:
        """
        执行一轮协商
        
        Returns:
            本轮协商结果
        """
        proposals = self.collect_proposals()
        total_requests = self.calculate_total_requests(proposals)
        
        # 让每个智能体根据情况调整
        new_proposals = []
        for agent in self.schedule.agents:
            if isinstance(agent, ResourceAgent):
                adjusted = agent.respond_to_situation(
                    total_requests, self.total_resources,
                    [p for p in proposals if p.agent_id != agent.component_id]
                )
                new_proposals.append(adjusted)
        
        new_total = self.calculate_total_requests(new_proposals)
        
        # 检查是否收敛
        change = abs(new_total - total_requests) / max(total_requests, 1)
        self.converged = change < 0.05
        
        round_result = {
            'round': self.current_round,
            'total_requests': new_total,
            'available': self.total_resources,
            'scarcity_ratio': new_total / self.total_resources,
            'change_ratio': change,
            'converged': self.converged
        }
        
        self.allocation_history.append(round_result)
        self.current_round += 1
        
        return round_result
    
    def allocate_resources(self) -> Dict[str, int]:
        """
        执行最终资源分配
        
        Returns:
            分配结果字典
        """
        proposals = self.collect_proposals()
        total_requests = self.calculate_total_requests(proposals)
        
        allocations = {}
        
        if total_requests <= self.total_resources:
            # 资源充足，满足所有请求
            for p in proposals:
                allocations[p.agent_id] = p.requested_amount
        else:
            # 资源不足，按比例和优先级分配
            allocations = self._priority_weighted_allocation(proposals)
        
        # 分发分配结果
        for agent in self.schedule.agents:
            if isinstance(agent, ResourceAgent):
                amount = allocations.get(agent.component_id, 0)
                agent.receive_allocation(amount)
        
        return allocations
    
    def _priority_weighted_allocation(self, 
                                      proposals: List[AllocationProposal]) -> Dict[str, int]:
        """优先级加权分配"""
        # 优先级权重
        priority_weights = {
            AgentPriority.CRITICAL: 2.0,
            AgentPriority.HIGH: 1.5,
            AgentPriority.NORMAL: 1.0,
            AgentPriority.LOW: 0.7
        }
        
        # 计算加权请求
        weighted_requests = []
        for p in proposals:
            agent = self._get_agent_by_id(p.agent_id)
            if agent:
                weight = priority_weights.get(agent.need.priority, 1.0)
                weighted_requests.append({
                    'agent_id': p.agent_id,
                    'request': p.requested_amount,
                    'minimum': agent.need.minimum,
                    'weight': weight,
                    'weighted_request': p.requested_amount * weight
                })
        
        total_weighted = sum(wr['weighted_request'] for wr in weighted_requests)
        
        # 按加权比例分配
        allocations = {}
        remaining = self.total_resources
        
        # 首先确保最低需求
        for wr in weighted_requests:
            min_alloc = min(wr['minimum'], remaining)
            allocations[wr['agent_id']] = min_alloc
            remaining -= min_alloc
        
        # 分配剩余资源
        if remaining > 0:
            for wr in weighted_requests:
                additional_need = wr['request'] - allocations[wr['agent_id']]
                if additional_need > 0 and remaining > 0:
                    # 按权重比例分配额外资源
                    additional = min(
                        additional_need,
                        int(remaining * wr['weight'] / total_weighted * len(weighted_requests))
                    )
                    allocations[wr['agent_id']] += additional
                    remaining -= additional
        
        return allocations
    
    def _get_agent_by_id(self, agent_id: str) -> Optional[ResourceAgent]:
        """根据ID获取智能体"""
        for agent in self.schedule.agents:
            if isinstance(agent, ResourceAgent) and agent.component_id == agent_id:
                return agent
        return None
    
    def calculate_fairness_metrics(self) -> Dict[str, float]:
        """
        计算公平性指标
        
        Returns:
            公平性指标字典
        """
        satisfactions = []
        allocations = []
        needs = []
        
        for agent in self.schedule.agents:
            if isinstance(agent, ResourceAgent):
                satisfactions.append(agent.satisfaction)
                allocations.append(agent.allocated_amount)
                needs.append(agent.need.desired)
        
        if not allocations:
            return {}
        
        # 基尼系数
        gini = self._calculate_gini(allocations)
        
        # 满意度指标
        avg_satisfaction = statistics.mean(satisfactions)
        min_satisfaction = min(satisfactions)
        
        # 需求满足率
        fulfillment_rates = [
            a / n if n > 0 else 0 
            for a, n in zip(allocations, needs)
        ]
        avg_fulfillment = statistics.mean(fulfillment_rates)
        
        # 分配方差
        allocation_variance = statistics.variance(allocations) if len(allocations) > 1 else 0
        
        return {
            'gini_coefficient': gini,
            'average_satisfaction': avg_satisfaction,
            'minimum_satisfaction': min_satisfaction,
            'average_fulfillment_rate': avg_fulfillment,
            'allocation_variance': allocation_variance,
            'total_allocated': sum(allocations),
            'utilization_rate': sum(allocations) / self.total_resources
        }
    
    def _calculate_gini(self, values: List[float]) -> float:
        """计算基尼系数"""
        if not values or sum(values) == 0:
            return 0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = sum((i + 1) * v for i, v in enumerate(sorted_values))
        
        return (2 * cumsum) / (n * sum(sorted_values)) - (n + 1) / n
    
    def run_simulation(self) -> Dict[str, Any]:
        """运行完整模拟"""
        # 初始提案
        for agent in self.schedule.agents:
            if isinstance(agent, ResourceAgent):
                agent.create_initial_proposal()
        
        # 协商过程
        while self.current_round < self.max_rounds and not self.converged:
            self.negotiate_round()
        
        # 最终分配
        final_allocations = self.allocate_resources()
        
        # 计算公平性指标
        fairness = self.calculate_fairness_metrics()
        
        return {
            'negotiation_rounds': self.current_round,
            'converged': self.converged,
            'allocation_history': self.allocation_history,
            'final_allocations': final_allocations,
            'fairness_metrics': fairness
        }


def run_resource_experiment(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    运行资源分配实验
    
    Args:
        config: 实验配置
        
    Returns:
        实验结果
    """
    if config is None:
        config = {
            'num_agents': 50,
            'total_resources': 1000,
            'max_rounds': 10,
            'use_collaborative': True
        }
    
    model = ResourceAllocationModel(
        num_agents=config.get('num_agents', 50),
        total_resources=config.get('total_resources', 1000),
        max_negotiation_rounds=config.get('max_rounds', 10),
        use_collaborative=config.get('use_collaborative', True)
    )
    
    results = model.run_simulation()
    results['config'] = config
    
    return results


if __name__ == "__main__":
    # 运行基线实验
    print("运行基线实验...")
    baseline_results = run_resource_experiment({
        'use_collaborative': False,
        'num_agents': 50,
        'total_resources': 1000
    })
    print(f"基线结果 - 协商轮次: {baseline_results['negotiation_rounds']}")
    print(f"基线公平性: {baseline_results['fairness_metrics']}")
    
    # 运行优化实验
    print("\n运行优化实验（协同决策）...")
    optimized_results = run_resource_experiment({
        'use_collaborative': True,
        'num_agents': 50,
        'total_resources': 1000
    })
    print(f"优化结果 - 协商轮次: {optimized_results['negotiation_rounds']}")
    print(f"优化公平性: {optimized_results['fairness_metrics']}")
    
    # 保存结果
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'resource', 'baseline.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': baseline_results,
            'optimized': optimized_results
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n结果已保存到 experiments/results/resource/baseline.json")

