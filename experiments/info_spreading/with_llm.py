"""
LLM 驱动的信息传播实验

使用 LLM 进行信息真伪判断和传播决策。
"""

import mesa
import networkx as nx
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import json
import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from casevo import create_llm, create_default_llm


class InformationType(Enum):
    """信息类型"""
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class AgentType(Enum):
    """智能体类型"""
    NORMAL = "normal"
    SKEPTIC = "skeptic"      # 怀疑者
    GULLIBLE = "gullible"    # 易信者
    INFLUENCER = "influencer"  # 影响者


@dataclass
class Information:
    """信息实体"""
    info_id: str
    content: str
    true_type: InformationType  # 真实类型
    source_credibility: float    # 来源可信度
    spread_count: int = 0


@dataclass
class AgentProfile:
    """智能体画像"""
    agent_type: AgentType
    education_level: str
    media_literacy: float   # 媒体素养 (0-1)
    trust_tendency: float   # 信任倾向 (0-1)
    influence_score: float  # 影响力分数


# 角色配置
AGENT_TYPE_CONFIG = {
    AgentType.NORMAL: {
        'distribution': 0.60,
        'education': ['高中', '大学', '研究生'],
        'media_literacy_range': (0.4, 0.7),
        'trust_range': (0.4, 0.6),
        'influence_range': (0.2, 0.5)
    },
    AgentType.SKEPTIC: {
        'distribution': 0.15,
        'education': ['大学', '研究生'],
        'media_literacy_range': (0.7, 0.95),
        'trust_range': (0.2, 0.4),
        'influence_range': (0.3, 0.6)
    },
    AgentType.GULLIBLE: {
        'distribution': 0.15,
        'education': ['初中', '高中'],
        'media_literacy_range': (0.1, 0.4),
        'trust_range': (0.6, 0.9),
        'influence_range': (0.1, 0.3)
    },
    AgentType.INFLUENCER: {
        'distribution': 0.10,
        'education': ['大学', '研究生'],
        'media_literacy_range': (0.5, 0.8),
        'trust_range': (0.3, 0.6),
        'influence_range': (0.7, 0.95)
    }
}

# 预设信息库
INFO_TEMPLATES = [
    {
        'content': '研究表明，每天喝8杯水对健康有益，有助于维持身体正常代谢。',
        'type': InformationType.TRUE,
        'credibility': 0.85
    },
    {
        'content': '最新科学发现：5G信号塔会导致新冠病毒传播，建议远离基站。',
        'type': InformationType.FALSE,
        'credibility': 0.3
    },
    {
        'content': '世界卫生组织确认：规律运动可以降低心血管疾病风险30%。',
        'type': InformationType.TRUE,
        'credibility': 0.9
    },
    {
        'content': '震惊！某知名饮料含有致癌物质，已有多人中毒住院！',
        'type': InformationType.FALSE,
        'credibility': 0.25
    },
    {
        'content': '气象部门预测：本周末将有大范围降温，请注意保暖。',
        'type': InformationType.TRUE,
        'credibility': 0.8
    },
    {
        'content': '重大发现：吃大蒜可以完全预防感冒，已被实验证实！',
        'type': InformationType.FALSE,
        'credibility': 0.35
    },
    {
        'content': '教育部通知：2024年高考时间确定为6月7日-8日。',
        'type': InformationType.TRUE,
        'credibility': 0.95
    },
    {
        'content': '紧急扩散：某银行系统崩溃，存款可能丢失，请立即取款！',
        'type': InformationType.FALSE,
        'credibility': 0.2
    }
]


class LLMInfoAgent(mesa.Agent):
    """LLM 驱动的信息传播智能体"""
    
    def __init__(self, unique_id: int, model: 'LLMInfoModel', profile: AgentProfile):
        super().__init__(unique_id, model)
        self.profile = profile
        self.received_info: Dict[str, Dict] = {}  # info_id -> {info, belief, source}
        self.spread_info: Set[str] = set()
        self.judgment_history: List[Dict] = []
        self.neighbors: List[int] = []
    
    def receive_information(self, info: Information, source_id: int):
        """接收信息"""
        if info.info_id in self.received_info:
            return  # 已接收过
        
        self.received_info[info.info_id] = {
            'info': info,
            'belief': None,  # 待判断
            'source': source_id
        }
    
    def judge_information(self, info: Information, source_id: int) -> Dict:
        """使用 LLM 判断信息真伪"""
        
        # 获取来源智能体信息
        source_info = "未知来源"
        if source_id >= 0:
            source_agent = self.model.agents_dict.get(source_id)
            if source_agent:
                source_info = f"{source_agent.profile.agent_type.value} 类型用户"
        
        prompt = f"""你是一个社交媒体用户，需要判断收到的信息是否真实。

## 你的背景
- 用户类型: {self.profile.agent_type.value}
- 教育程度: {self.profile.education_level}
- 媒体素养: {'高' if self.profile.media_literacy > 0.7 else '中等' if self.profile.media_literacy > 0.4 else '低'}
- 信任倾向: {'容易相信' if self.profile.trust_tendency > 0.6 else '中等' if self.profile.trust_tendency > 0.4 else '较为谨慎'}

## 收到的信息
内容: "{info.content}"
来源: {source_info}
来源可信度: {info.source_credibility:.0%}

## 判断任务

请根据你的背景和媒体素养，判断这条信息的真实性。

考虑因素：
1. 信息内容是否符合常识和科学原理
2. 来源的可信度如何
3. 是否存在耸人听闻或情绪化的表述
4. 是否需要进一步核实

请按以下格式回答：

【分析】简述你的判断依据（1-2句话）
【判断】真实 / 虚假 / 不确定（只写一个）
【传播】会转发 / 不转发（只写一个）
"""
        
        try:
            response = self.model.llm.send_message(prompt)
            
            # 解析判断结果
            belief = self._parse_belief(response)
            will_spread = self._parse_spread(response)
            
            result = {
                'info_id': info.info_id,
                'belief': belief,
                'will_spread': will_spread,
                'reasoning': response
            }
            
            self.judgment_history.append(result)
            return result
            
        except Exception as e:
            print(f"  Agent {self.unique_id} LLM 调用失败: {e}")
            # 使用规则后备
            return self._rule_based_judgment(info)
    
    def _parse_belief(self, response: str) -> InformationType:
        """解析判断结果"""
        response_lower = response.lower()
        
        if '【判断】' in response:
            judgment_part = response.split('【判断】')[-1].split('【')[0]
            if '真实' in judgment_part:
                return InformationType.TRUE
            elif '虚假' in judgment_part:
                return InformationType.FALSE
        
        return InformationType.UNKNOWN
    
    def _parse_spread(self, response: str) -> bool:
        """解析传播决策"""
        if '【传播】' in response:
            spread_part = response.split('【传播】')[-1].split('【')[0]
            if '会转发' in spread_part or '转发' in spread_part:
                return True
        return False
    
    def _rule_based_judgment(self, info: Information) -> Dict:
        """规则型后备判断"""
        # 基于媒体素养和来源可信度判断
        threshold = 0.5 + (self.profile.media_literacy - 0.5) * 0.3
        
        if info.source_credibility > threshold:
            belief = InformationType.TRUE
        elif info.source_credibility < threshold - 0.2:
            belief = InformationType.FALSE
        else:
            belief = InformationType.UNKNOWN
        
        # 传播决策
        will_spread = (
            belief == InformationType.TRUE and 
            random.random() < self.profile.influence_score
        )
        
        return {
            'info_id': info.info_id,
            'belief': belief,
            'will_spread': will_spread,
            'reasoning': 'rule-based'
        }
    
    def spread_to_neighbors(self, info: Information):
        """向邻居传播信息"""
        if info.info_id in self.spread_info:
            return  # 已传播过
        
        self.spread_info.add(info.info_id)
        info.spread_count += 1
        
        for neighbor_id in self.neighbors:
            neighbor = self.model.agents_dict.get(neighbor_id)
            if neighbor:
                neighbor.receive_information(info, self.unique_id)
    
    def step(self):
        """每轮执行"""
        pass  # 由模型统一控制


class LLMInfoModel(mesa.Model):
    """LLM 驱动的信息传播模型"""
    
    def __init__(self, num_agents: int = 100, llm=None, use_llm: bool = True):
        super().__init__()
        
        self.num_agents = num_agents
        self.llm = llm or create_default_llm()
        self.use_llm = use_llm
        self.schedule = mesa.time.RandomActivation(self)
        
        # 创建无标度网络
        self.network = nx.barabasi_albert_graph(num_agents, 3)
        
        self.current_round = 0
        self.spread_history = []
        self.agents_dict: Dict[int, LLMInfoAgent] = {}
        self.information_pool: Dict[str, Information] = {}
        
        # 创建智能体
        self._create_agents()
        
        # 初始化信息
        self._init_information()
    
    def _create_agents(self):
        """创建智能体"""
        types = list(AgentType)
        weights = [AGENT_TYPE_CONFIG[t]['distribution'] for t in types]
        
        for i in range(self.num_agents):
            agent_type = random.choices(types, weights=weights)[0]
            config = AGENT_TYPE_CONFIG[agent_type]
            
            profile = AgentProfile(
                agent_type=agent_type,
                education_level=random.choice(config['education']),
                media_literacy=random.uniform(*config['media_literacy_range']),
                trust_tendency=random.uniform(*config['trust_range']),
                influence_score=random.uniform(*config['influence_range'])
            )
            
            agent = LLMInfoAgent(i, self, profile)
            agent.neighbors = list(self.network.neighbors(i))
            self.schedule.add(agent)
            self.agents_dict[i] = agent
        
        # 打印分布
        type_dist = {}
        for agent in self.schedule.agents:
            t = agent.profile.agent_type.value
            type_dist[t] = type_dist.get(t, 0) + 1
        
        print("\n智能体类型分布:")
        for t, count in sorted(type_dist.items()):
            print(f"  {t}: {count} ({count/self.num_agents*100:.0f}%)")
    
    def _init_information(self):
        """初始化信息池"""
        for i, template in enumerate(INFO_TEMPLATES):
            info = Information(
                info_id=f"info_{i+1}",
                content=template['content'],
                true_type=template['type'],
                source_credibility=template['credibility']
            )
            self.information_pool[info.info_id] = info
        
        # 随机选择初始传播者
        initial_spreaders = random.sample(list(self.agents_dict.keys()), 
                                         min(10, self.num_agents // 10))
        
        for spreader_id in initial_spreaders:
            agent = self.agents_dict[spreader_id]
            # 随机分配一条信息
            info = random.choice(list(self.information_pool.values()))
            agent.received_info[info.info_id] = {
                'info': info,
                'belief': InformationType.TRUE,  # 初始传播者相信信息
                'source': -1  # 外部来源
            }
            agent.spread_info.add(info.info_id)
    
    def simulate_round(self):
        """模拟一轮传播"""
        print(f"\n--- 传播轮次 {self.current_round + 1} ---")
        
        # 收集需要处理的信息
        pending_judgments = []
        for agent in self.schedule.agents:
            for info_id, data in agent.received_info.items():
                if data['belief'] is None:
                    pending_judgments.append((agent, data['info'], data['source']))
        
        print(f"待判断信息: {len(pending_judgments)} 条")
        
        # 处理判断
        spread_decisions = []
        for agent, info, source_id in pending_judgments:
            if self.use_llm:
                result = agent.judge_information(info, source_id)
                time.sleep(0.2)  # API 速率限制
            else:
                result = agent._rule_based_judgment(info)
            
            agent.received_info[info.info_id]['belief'] = result['belief']
            
            if result['will_spread']:
                spread_decisions.append((agent, info))
        
        # 执行传播
        for agent, info in spread_decisions:
            agent.spread_to_neighbors(info)
        
        print(f"本轮传播: {len(spread_decisions)} 条")
        
        # 记录统计
        stats = self._calculate_stats()
        self.spread_history.append({
            'round': self.current_round,
            **stats
        })
        
        self.current_round += 1
        
        return len(spread_decisions)
    
    def _calculate_stats(self) -> Dict:
        """计算统计信息"""
        total_received = sum(len(a.received_info) for a in self.schedule.agents)
        
        believed_true = 0
        believed_false = 0
        
        for agent in self.schedule.agents:
            for info_id, data in agent.received_info.items():
                info = data['info']
                belief = data['belief']
                
                if belief == InformationType.TRUE:
                    if info.true_type == InformationType.FALSE:
                        believed_false += 1  # 错误相信虚假信息
                    else:
                        believed_true += 1
        
        false_belief_ratio = believed_false / total_received if total_received > 0 else 0
        
        return {
            'total_received': total_received,
            'believed_true': believed_true,
            'believed_false': believed_false,
            'false_belief_ratio': false_belief_ratio
        }
    
    def run(self, num_rounds: int = 10):
        """运行模拟"""
        print(f"\n{'='*50}")
        print(f"开始信息传播模拟 - {self.num_agents} 个智能体, {num_rounds} 轮")
        print(f"使用 LLM: {self.use_llm}")
        print(f"{'='*50}")
        
        for r in range(num_rounds):
            spreads = self.simulate_round()
            if spreads == 0 and r > 2:
                print("\n传播停止（无新传播）")
                break
        
        # 计算最终准确率
        accuracy = self._calculate_accuracy()
        
        print(f"\n{'='*50}")
        print("模拟结束")
        print(f"{'='*50}")
        print(f"整体判断准确率: {accuracy['overall']:.1%}")
        print(f"虚假信息识别率: {accuracy['false_detection']:.1%}")
        
        return self.get_results()
    
    def _calculate_accuracy(self) -> Dict:
        """计算判断准确率"""
        correct = 0
        total = 0
        false_correct = 0
        false_total = 0
        
        for agent in self.schedule.agents:
            for info_id, data in agent.received_info.items():
                if data['belief'] is None:
                    continue
                
                info = data['info']
                belief = data['belief']
                total += 1
                
                # 判断是否正确
                if info.true_type == InformationType.TRUE and belief == InformationType.TRUE:
                    correct += 1
                elif info.true_type == InformationType.FALSE and belief == InformationType.FALSE:
                    correct += 1
                    false_correct += 1
                
                if info.true_type == InformationType.FALSE:
                    false_total += 1
        
        return {
            'overall': correct / total if total > 0 else 0,
            'false_detection': false_correct / false_total if false_total > 0 else 0
        }
    
    def get_results(self) -> Dict:
        """获取完整结果"""
        accuracy = self._calculate_accuracy()
        
        return {
            'num_agents': self.num_agents,
            'use_llm': self.use_llm,
            'rounds': self.current_round,
            'spread_history': self.spread_history,
            'accuracy': accuracy,
            'network_stats': {
                'nodes': self.network.number_of_nodes(),
                'edges': self.network.number_of_edges(),
                'avg_degree': sum(dict(self.network.degree()).values()) / self.num_agents
            },
            'information_stats': [
                {
                    'info_id': info.info_id,
                    'type': info.true_type.value,
                    'spread_count': info.spread_count
                }
                for info in self.information_pool.values()
            ]
        }


def run_comparison_experiment(num_agents: int = 50, num_rounds: int = 8):
    """运行对比实验（LLM vs 规则）"""
    
    print("\n" + "="*60)
    print("信息传播对比实验")
    print("="*60)
    
    results = {}
    
    # 固定随机种子
    random.seed(42)
    
    # 1. 规则型基线
    print("\n【基线实验 - 规则型】")
    model_baseline = LLMInfoModel(
        num_agents=num_agents,
        use_llm=False
    )
    results['baseline'] = model_baseline.run(num_rounds=num_rounds)
    
    # 重置随机种子
    random.seed(42)
    
    # 2. LLM 驱动
    print("\n【优化实验 - LLM驱动】")
    llm = create_default_llm()
    model_llm = LLMInfoModel(
        num_agents=num_agents,
        llm=llm,
        use_llm=True
    )
    results['llm'] = model_llm.run(num_rounds=num_rounds)
    
    # 输出对比
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    
    print(f"\n{'指标':<25} {'基线(规则)':<15} {'LLM驱动':<15}")
    print("-"*55)
    print(f"{'传播轮数':<25} {results['baseline']['rounds']:<15} {results['llm']['rounds']:<15}")
    print(f"{'整体判断准确率':<25} {results['baseline']['accuracy']['overall']:<15.1%} {results['llm']['accuracy']['overall']:<15.1%}")
    print(f"{'虚假信息识别率':<25} {results['baseline']['accuracy']['false_detection']:<15.1%} {results['llm']['accuracy']['false_detection']:<15.1%}")
    
    # 保存结果
    output_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'info_spreading', 'llm_comparison.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM 信息传播实验')
    parser.add_argument('--agents', type=int, default=50, help='智能体数量')
    parser.add_argument('--rounds', type=int, default=8, help='传播轮数')
    parser.add_argument('--compare', action='store_true', help='运行对比实验')
    
    args = parser.parse_args()
    
    if args.compare:
        run_comparison_experiment(
            num_agents=args.agents,
            num_rounds=args.rounds
        )
    else:
        # 单独运行 LLM 版本
        llm = create_default_llm()
        model = LLMInfoModel(
            num_agents=args.agents,
            llm=llm,
            use_llm=True
        )
        results = model.run(num_rounds=args.rounds)
        
        output_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'info_spreading', 'llm.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n结果已保存到: {output_file}")

