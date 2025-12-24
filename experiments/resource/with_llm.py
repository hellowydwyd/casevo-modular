"""
LLM 驱动的资源分配实验

使用 LLM 进行协商策略生成和公平性评估。
"""

import mesa
import networkx as nx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random
import json
import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from casevo import create_llm, create_default_llm


class AgentPriority(Enum):
    """智能体优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class ResourceNeed:
    """资源需求"""
    minimum: int
    desired: int
    maximum: int
    priority: AgentPriority


@dataclass
class AgentProfile:
    """智能体画像"""
    role: str           # 角色（医院、学校、工厂等）
    description: str    # 描述
    need: ResourceNeed  # 资源需求
    flexibility: float  # 灵活度 (0-1)


# 预设角色配置
ROLE_CONFIGS = {
    'hospital': {
        'description': '城市中心医院，负责急诊和重症救治，资源需求具有紧迫性和不可替代性',
        'need_range': (25, 35),
        'priority': AgentPriority.CRITICAL,
        'flexibility': 0.2,
        'distribution': 0.1
    },
    'school': {
        'description': '公立学校，负责基础教育，资源需求相对稳定但数量较大',
        'need_range': (20, 30),
        'priority': AgentPriority.HIGH,
        'flexibility': 0.4,
        'distribution': 0.2
    },
    'factory': {
        'description': '制造业工厂，资源需求与生产计划挂钩，有一定弹性空间',
        'need_range': (15, 25),
        'priority': AgentPriority.NORMAL,
        'flexibility': 0.6,
        'distribution': 0.25
    },
    'office': {
        'description': '商业办公楼，资源需求主要用于日常运营，弹性较大',
        'need_range': (10, 20),
        'priority': AgentPriority.NORMAL,
        'flexibility': 0.7,
        'distribution': 0.25
    },
    'residential': {
        'description': '居民社区，资源需求分散但涉及民生，需要基本保障',
        'need_range': (15, 25),
        'priority': AgentPriority.LOW,
        'flexibility': 0.5,
        'distribution': 0.2
    }
}


class LLMResourceAgent(mesa.Agent):
    """LLM 驱动的资源分配智能体"""
    
    def __init__(self, unique_id: int, model: 'LLMResourceModel', profile: AgentProfile):
        super().__init__(unique_id, model)
        self.profile = profile
        self.current_request = profile.need.desired
        self.allocated = 0
        self.negotiation_history: List[Dict] = []
        self.reasoning_history: List[str] = []
    
    def generate_negotiation_strategy(self, round_info: Dict) -> int:
        """使用 LLM 生成协商策略"""
        
        prompt = f"""你是 {self.profile.role}（{self.profile.description}）的资源管理者。

## 你的资源需求
- 最低需求: {self.profile.need.minimum} 单位
- 期望需求: {self.profile.need.desired} 单位
- 最大可用: {self.profile.need.maximum} 单位
- 优先级: {self.profile.need.priority.value}
- 当前申请: {self.current_request} 单位

## 当前协商情况
- 协商轮次: 第 {round_info['round'] + 1} 轮
- 总可用资源: {round_info['total_resources']} 单位
- 当前总申请: {round_info['total_requests']} 单位
- 资源缺口: {round_info['total_requests'] - round_info['total_resources']} 单位
- 你上轮获得: {self.allocated} 单位

## 其他参与者情况
{self._format_others_info(round_info)}

## 协商任务

作为 {self.profile.role} 的管理者，你需要决定本轮的资源申请量。

考虑因素：
1. 你的机构对社会的重要性和紧迫性
2. 当前资源供需比例
3. 其他参与者的需求合理性
4. 如何在保障自身需求的同时促进整体公平

请按以下格式回答：

【协商策略】简述你的考虑（1-2句话）
【申请数量】一个整数（范围：{self.profile.need.minimum}-{self.profile.need.maximum}）
"""
        
        try:
            response = self.model.llm.send_message(prompt)
            self.reasoning_history.append(response)
            
            # 解析申请数量
            new_request = self._parse_request(response)
            return new_request
            
        except Exception as e:
            print(f"  Agent {self.unique_id} LLM 调用失败: {e}")
            # 使用规则后备
            return self._rule_based_adjustment()
    
    def _format_others_info(self, round_info: Dict) -> str:
        """格式化其他参与者信息"""
        others = round_info.get('others_summary', {})
        if not others:
            return "（暂无其他参与者信息）"
        
        lines = []
        for role, info in others.items():
            lines.append(f"- {role}: 平均申请 {info['avg_request']:.0f} 单位, "
                        f"优先级 {info['priority']}")
        return '\n'.join(lines)
    
    def _parse_request(self, response: str) -> int:
        """解析 LLM 响应中的申请数量"""
        import re
        
        # 尝试从【申请数量】后提取数字
        match = re.search(r'【申请数量】\s*(\d+)', response)
        if match:
            value = int(match.group(1))
            # 确保在合理范围内
            return max(self.profile.need.minimum, 
                      min(self.profile.need.maximum, value))
        
        # 后备：保持当前申请
        return self.current_request
    
    def _rule_based_adjustment(self) -> int:
        """规则型后备调整"""
        scarcity = self.model.get_scarcity_ratio()
        if scarcity > 1.2:
            # 资源紧张，适当降低申请
            reduction = int(self.current_request * 0.1 * self.profile.flexibility)
            return max(self.profile.need.minimum, self.current_request - reduction)
        return self.current_request
    
    def step(self):
        """每轮执行"""
        pass  # 协商在模型层面统一控制


class LLMResourceModel(mesa.Model):
    """LLM 驱动的资源分配模型"""
    
    def __init__(self, num_agents: int = 50, total_resources: int = 1000,
                 llm=None, use_llm: bool = True):
        super().__init__()
        
        self.num_agents = num_agents
        self.total_resources = total_resources
        self.llm = llm or create_default_llm()
        self.use_llm = use_llm
        self.schedule = mesa.time.RandomActivation(self)
        
        self.current_round = 0
        self.allocation_history = []
        self.agents_dict: Dict[int, LLMResourceAgent] = {}
        
        # 创建智能体
        self._create_agents()
    
    def _create_agents(self):
        """创建智能体"""
        roles = list(ROLE_CONFIGS.keys())
        weights = [ROLE_CONFIGS[r]['distribution'] for r in roles]
        
        for i in range(self.num_agents):
            # 按分布选择角色
            role = random.choices(roles, weights=weights)[0]
            config = ROLE_CONFIGS[role]
            
            # 生成需求
            desired = random.randint(*config['need_range'])
            need = ResourceNeed(
                minimum=int(desired * 0.6),
                desired=desired,
                maximum=int(desired * 1.3),
                priority=config['priority']
            )
            
            profile = AgentProfile(
                role=role,
                description=config['description'],
                need=need,
                flexibility=config['flexibility']
            )
            
            agent = LLMResourceAgent(i, self, profile)
            self.schedule.add(agent)
            self.agents_dict[i] = agent
        
        # 打印分布
        role_dist = {}
        for agent in self.schedule.agents:
            role = agent.profile.role
            role_dist[role] = role_dist.get(role, 0) + 1
        
        print("\n智能体角色分布:")
        for role, count in sorted(role_dist.items()):
            print(f"  {role}: {count} ({count/self.num_agents*100:.0f}%)")
    
    def get_scarcity_ratio(self) -> float:
        """获取资源紧缺比例"""
        total_requests = sum(a.current_request for a in self.schedule.agents)
        return total_requests / self.total_resources if self.total_resources > 0 else float('inf')
    
    def get_round_info(self) -> Dict:
        """获取当前轮次信息"""
        total_requests = sum(a.current_request for a in self.schedule.agents)
        
        # 按角色汇总
        role_summary = {}
        for agent in self.schedule.agents:
            role = agent.profile.role
            if role not in role_summary:
                role_summary[role] = {
                    'requests': [],
                    'priority': agent.profile.need.priority.value
                }
            role_summary[role]['requests'].append(agent.current_request)
        
        others_summary = {}
        for role, data in role_summary.items():
            others_summary[role] = {
                'avg_request': sum(data['requests']) / len(data['requests']),
                'priority': data['priority']
            }
        
        return {
            'round': self.current_round,
            'total_resources': self.total_resources,
            'total_requests': total_requests,
            'others_summary': others_summary
        }
    
    def allocate_resources(self):
        """分配资源"""
        total_requests = sum(a.current_request for a in self.schedule.agents)
        
        if total_requests <= self.total_resources:
            # 资源充足，满足所有请求
            for agent in self.schedule.agents:
                agent.allocated = agent.current_request
        else:
            # 资源不足，按优先级和比例分配
            priority_weights = {
                AgentPriority.CRITICAL: 2.0,
                AgentPriority.HIGH: 1.5,
                AgentPriority.NORMAL: 1.0,
                AgentPriority.LOW: 0.7
            }
            
            # 计算加权请求
            weighted_requests = []
            for agent in self.schedule.agents:
                weight = priority_weights[agent.profile.need.priority]
                weighted_requests.append(agent.current_request * weight)
            
            total_weighted = sum(weighted_requests)
            
            # 按加权比例分配
            for i, agent in enumerate(self.schedule.agents):
                ratio = weighted_requests[i] / total_weighted
                agent.allocated = int(self.total_resources * ratio)
    
    def negotiate_round(self):
        """执行一轮协商"""
        print(f"\n--- 协商轮次 {self.current_round + 1} ---")
        
        round_info = self.get_round_info()
        print(f"总申请: {round_info['total_requests']}, 可用: {self.total_resources}, "
              f"缺口: {round_info['total_requests'] - self.total_resources}")
        
        # 每个智能体生成协商策略
        changes = 0
        for agent in self.schedule.agents:
            old_request = agent.current_request
            
            if self.use_llm:
                new_request = agent.generate_negotiation_strategy(round_info)
            else:
                new_request = agent._rule_based_adjustment()
            
            if new_request != old_request:
                changes += 1
                agent.negotiation_history.append({
                    'round': self.current_round,
                    'old': old_request,
                    'new': new_request
                })
            
            agent.current_request = new_request
            
            if self.use_llm:
                time.sleep(0.3)  # API 速率限制
        
        # 分配资源
        self.allocate_resources()
        
        # 记录历史
        new_total = sum(a.current_request for a in self.schedule.agents)
        self.allocation_history.append({
            'round': self.current_round,
            'total_requests': new_total,
            'changes': changes,
            'scarcity_ratio': new_total / self.total_resources
        })
        
        print(f"本轮调整: {changes} 个智能体修改了申请")
        
        self.current_round += 1
        
        return changes
    
    def run(self, max_rounds: int = 5, convergence_threshold: int = 2):
        """运行协商过程"""
        print(f"\n{'='*50}")
        print(f"开始资源分配协商 - {self.num_agents} 个智能体, {self.total_resources} 单位资源")
        print(f"使用 LLM: {self.use_llm}")
        print(f"{'='*50}")
        
        initial_requests = sum(a.current_request for a in self.schedule.agents)
        print(f"\n初始总申请: {initial_requests} (缺口: {initial_requests - self.total_resources})")
        
        no_change_rounds = 0
        
        for r in range(max_rounds):
            changes = self.negotiate_round()
            
            if changes == 0:
                no_change_rounds += 1
                if no_change_rounds >= convergence_threshold:
                    print(f"\n协商收敛（连续 {convergence_threshold} 轮无变化）")
                    break
            else:
                no_change_rounds = 0
        
        # 计算公平性指标
        fairness = self._calculate_fairness()
        
        print(f"\n{'='*50}")
        print("协商结束")
        print(f"{'='*50}")
        print(f"最终分配 - 基尼系数: {fairness['gini']:.3f}, "
              f"平均满足率: {fairness['avg_fulfillment']:.1%}")
        
        return self.get_results()
    
    def _calculate_fairness(self) -> Dict:
        """计算公平性指标"""
        allocations = [a.allocated for a in self.schedule.agents]
        desires = [a.profile.need.desired for a in self.schedule.agents]
        
        # 基尼系数
        n = len(allocations)
        if n == 0 or sum(allocations) == 0:
            gini = 0
        else:
            sorted_alloc = sorted(allocations)
            cumsum = sum((i + 1) * x for i, x in enumerate(sorted_alloc))
            gini = (2 * cumsum) / (n * sum(sorted_alloc)) - (n + 1) / n
        
        # 平均满足率
        fulfillments = [a / d if d > 0 else 0 for a, d in zip(allocations, desires)]
        avg_fulfillment = sum(fulfillments) / len(fulfillments) if fulfillments else 0
        
        # 最低满足率
        min_fulfillment = min(fulfillments) if fulfillments else 0
        
        return {
            'gini': gini,
            'avg_fulfillment': avg_fulfillment,
            'min_fulfillment': min_fulfillment,
            'total_allocated': sum(allocations)
        }
    
    def get_results(self) -> Dict:
        """获取完整结果"""
        fairness = self._calculate_fairness()
        
        return {
            'num_agents': self.num_agents,
            'total_resources': self.total_resources,
            'use_llm': self.use_llm,
            'rounds': self.current_round,
            'allocation_history': self.allocation_history,
            'fairness_metrics': fairness,
            'agent_details': [
                {
                    'id': a.unique_id,
                    'role': a.profile.role,
                    'priority': a.profile.need.priority.value,
                    'desired': a.profile.need.desired,
                    'final_request': a.current_request,
                    'allocated': a.allocated,
                    'fulfillment': a.allocated / a.profile.need.desired if a.profile.need.desired > 0 else 0,
                    'reasoning_sample': a.reasoning_history[-1] if a.reasoning_history else None
                }
                for a in self.schedule.agents
            ]
        }


def run_comparison_experiment(num_agents: int = 30, total_resources: int = 600,
                             max_rounds: int = 4):
    """运行对比实验（LLM vs 规则）"""
    
    print("\n" + "="*60)
    print("资源分配对比实验")
    print("="*60)
    
    results = {}
    
    # 固定随机种子
    random.seed(42)
    
    # 1. 规则型基线
    print("\n【基线实验 - 规则型】")
    model_baseline = LLMResourceModel(
        num_agents=num_agents,
        total_resources=total_resources,
        use_llm=False
    )
    results['baseline'] = model_baseline.run(max_rounds=max_rounds)
    
    # 重置随机种子
    random.seed(42)
    
    # 2. LLM 驱动
    print("\n【优化实验 - LLM驱动】")
    llm = create_default_llm()
    model_llm = LLMResourceModel(
        num_agents=num_agents,
        total_resources=total_resources,
        llm=llm,
        use_llm=True
    )
    results['llm'] = model_llm.run(max_rounds=max_rounds)
    
    # 输出对比
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    
    print(f"\n{'指标':<20} {'基线(规则)':<15} {'LLM驱动':<15}")
    print("-"*50)
    print(f"{'协商轮数':<20} {results['baseline']['rounds']:<15} {results['llm']['rounds']:<15}")
    print(f"{'基尼系数':<20} {results['baseline']['fairness_metrics']['gini']:<15.3f} {results['llm']['fairness_metrics']['gini']:<15.3f}")
    print(f"{'平均满足率':<20} {results['baseline']['fairness_metrics']['avg_fulfillment']:<15.1%} {results['llm']['fairness_metrics']['avg_fulfillment']:<15.1%}")
    print(f"{'最低满足率':<20} {results['baseline']['fairness_metrics']['min_fulfillment']:<15.1%} {results['llm']['fairness_metrics']['min_fulfillment']:<15.1%}")
    
    # 保存结果
    output_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'resource', 'llm_comparison.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM 资源分配实验')
    parser.add_argument('--agents', type=int, default=30, help='智能体数量')
    parser.add_argument('--resources', type=int, default=600, help='总资源量')
    parser.add_argument('--rounds', type=int, default=4, help='最大协商轮数')
    parser.add_argument('--compare', action='store_true', help='运行对比实验')
    
    args = parser.parse_args()
    
    if args.compare:
        run_comparison_experiment(
            num_agents=args.agents,
            total_resources=args.resources,
            max_rounds=args.rounds
        )
    else:
        # 单独运行 LLM 版本
        llm = create_default_llm()
        model = LLMResourceModel(
            num_agents=args.agents,
            total_resources=args.resources,
            llm=llm,
            use_llm=True
        )
        results = model.run(max_rounds=args.rounds)
        
        output_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'resource', 'llm.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n结果已保存到: {output_file}")

