"""
选举投票实验场景

模拟 2020 年美国总统大选辩论投票过程：
- 101 个选民智能体
- 小世界网络拓扑
- 6 轮辩论事件
- 投票演化追踪
"""

import mesa
import networkx as nx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from casevo import (
    AgentBase,
    ThoughtChain, BaseStep,
    TreeOfThought, ToTStep, EvaluatorStep, SearchStrategy,
    AdvancedMemory, AdvancedMemoryFactory,
    DecisionEvaluator, DecisionRecord, MetaCognitionModule
)


class PoliticalLeaning(Enum):
    """政治倾向（基于 Pew Research 分类）"""
    PROGRESSIVE_LEFT = "progressive_left"
    ESTABLISHMENT_LIBERAL = "establishment_liberal"
    DEMOCRATIC_MAINSTAYS = "democratic_mainstays"
    OUTSIDER_LEFT = "outsider_left"
    STRESSED_SIDELINERS = "stressed_sideliners"
    AMBIVALENT_RIGHT = "ambivalent_right"
    COMMITTED_CONSERVATIVES = "committed_conservatives"
    POPULIST_RIGHT = "populist_right"
    FAITH_AND_FLAG = "faith_and_flag"


class VoteChoice(Enum):
    """投票选择"""
    BIDEN = "Biden"
    TRUMP = "Trump"
    UNDECIDED = "Undecided"


@dataclass
class DebateEvent:
    """辩论事件"""
    round_num: int
    topic: str
    biden_statement: str
    trump_statement: str
    key_points: List[str]


@dataclass
class VoterProfile:
    """选民画像"""
    political_leaning: PoliticalLeaning
    age_group: str
    education_level: str
    key_issues: List[str]
    initial_preference: VoteChoice
    susceptibility: float  # 易受影响程度 (0-1)


class ElectionVoterAgent(AgentBase):
    """
    选举投票智能体
    
    模拟具有政治倾向的选民，能够观看辩论、与邻居讨论、
    进行反思并做出投票决策。
    """
    
    def __init__(self, unique_id: int, model: 'ElectionModel',
                 profile: VoterProfile, use_tot: bool = False):
        """
        初始化选民智能体
        
        Args:
            unique_id: 唯一标识
            model: 模型实例
            profile: 选民画像
            use_tot: 是否使用 Tree of Thought
        """
        description = self._generate_description(profile)
        context = {
            'political_leaning': profile.political_leaning.value,
            'key_issues': profile.key_issues,
            'current_preference': profile.initial_preference.value
        }
        
        super().__init__(unique_id, model, description, context)
        
        self.profile = profile
        self.current_vote = profile.initial_preference
        self.vote_history: List[Dict[str, Any]] = []
        self.confidence = 0.5
        self.use_tot = use_tot
        
        # 决策评估
        self.evaluator = DecisionEvaluator()
        self.metacog = MetaCognitionModule(self)
        
        # 邻居列表（由网络设置）
        self.neighbors: List[int] = []
    
    def _generate_description(self, profile: VoterProfile) -> str:
        """生成智能体描述"""
        return f"""你是一位美国选民，具有以下特征：
- 政治倾向：{profile.political_leaning.value}
- 年龄组：{profile.age_group}
- 教育水平：{profile.education_level}
- 关注议题：{', '.join(profile.key_issues)}
- 初始偏好：{profile.initial_preference.value}

你会根据辩论内容、与邻居的讨论以及自身的价值观来做出投票决策。"""
    
    def watch_debate(self, debate: DebateEvent):
        """
        观看辩论
        
        Args:
            debate: 辩论事件
        """
        # 记录辩论内容到记忆
        self.memory.add_short_memory(
            source="debate",
            target=self.component_id,
            action="watch",
            content=f"辩论第{debate.round_num}轮 - 主题：{debate.topic}\n"
                   f"Biden：{debate.biden_statement}\n"
                   f"Trump：{debate.trump_statement}"
        )
        
        # 评估辩论对自己的影响
        self._evaluate_debate_impact(debate)
    
    def _evaluate_debate_impact(self, debate: DebateEvent):
        """评估辩论对自己立场的影响"""
        # 检查议题是否是自己关注的
        relevant_issues = set(self.profile.key_issues) & set(debate.key_points)
        
        if relevant_issues:
            # 关注的议题，影响更大
            impact_factor = 0.3 * len(relevant_issues) / len(self.profile.key_issues)
        else:
            impact_factor = 0.1
        
        # 更新置信度（辩论可能增加或减少确定性）
        self.confidence = max(0.1, min(1.0, self.confidence + random.uniform(-0.1, 0.1)))
    
    def discuss_with_neighbors(self):
        """与邻居讨论"""
        if not self.neighbors:
            return
        
        for neighbor_id in self.neighbors:
            neighbor = self.model.schedule.agents[neighbor_id]
            if isinstance(neighbor, ElectionVoterAgent):
                # 交换观点
                self._exchange_views(neighbor)
    
    def _exchange_views(self, neighbor: 'ElectionVoterAgent'):
        """与邻居交换观点"""
        # 记录交流
        exchange_content = f"与{neighbor.component_id}讨论。" \
                          f"对方支持{neighbor.current_vote.value}，置信度{neighbor.confidence:.2f}"
        
        self.memory.add_short_memory(
            source=neighbor.component_id,
            target=self.component_id,
            action="discuss",
            content=exchange_content
        )
        
        # 社会影响：如果邻居与自己观点不同，可能被影响
        if neighbor.current_vote != self.current_vote:
            influence = self.profile.susceptibility * neighbor.confidence
            if random.random() < influence * 0.1:
                # 可能改变立场
                self._consider_position_change(neighbor.current_vote)
    
    def _consider_position_change(self, alternative: VoteChoice):
        """考虑改变立场"""
        if self.use_tot:
            # 使用 ToT 进行深度思考
            decision = self._tot_decision(alternative)
        else:
            # 使用简单决策
            decision = self._simple_decision(alternative)
        
        if decision['should_change']:
            self.current_vote = alternative
            self.confidence = decision['new_confidence']
    
    def _tot_decision(self, alternative: VoteChoice) -> Dict[str, Any]:
        """使用 Tree of Thought 进行决策"""
        # 这里简化实现，实际应使用完整的 ToT 流程
        current_strength = self.confidence
        alternative_appeal = random.uniform(0.3, 0.7)
        
        should_change = alternative_appeal > current_strength
        
        return {
            'should_change': should_change,
            'new_confidence': alternative_appeal if should_change else current_strength
        }
    
    def _simple_decision(self, alternative: VoteChoice) -> Dict[str, Any]:
        """简单决策"""
        should_change = random.random() < self.profile.susceptibility * 0.1
        
        return {
            'should_change': should_change,
            'new_confidence': random.uniform(0.4, 0.8) if should_change else self.confidence
        }
    
    def reflect(self):
        """进行反思"""
        # 检查是否需要反思
        decision_record = DecisionRecord(
            decision_id=f"vote_{self.unique_id}_{self.model.schedule.time}",
            timestamp=self.model.schedule.time,
            agent_id=self.component_id,
            decision_content=f"当前支持 {self.current_vote.value}",
            reasoning="基于辩论内容和邻居讨论",
            confidence=self.confidence
        )
        
        eval_result = self.metacog.evaluate_decision(decision_record)
        
        if eval_result['needs_reflection']:
            # 触发反思
            self.memory.reflect_memory()
            self.confidence = max(0.3, self.confidence - 0.1)  # 反思可能降低确定性
    
    def vote(self) -> VoteChoice:
        """进行投票"""
        # 记录投票历史
        self.vote_history.append({
            'round': self.model.schedule.time,
            'vote': self.current_vote.value,
            'confidence': self.confidence
        })
        
        return self.current_vote
    
    def step(self):
        """每轮执行"""
        # 1. 观看辩论（如果有）
        current_debate = self.model.get_current_debate()
        if current_debate:
            self.watch_debate(current_debate)
        
        # 2. 与邻居讨论
        self.discuss_with_neighbors()
        
        # 3. 反思
        self.reflect()
        
        # 4. 更新投票意向
        self.vote()


class ElectionModel(mesa.Model):
    """
    选举模拟模型
    
    管理选举过程，包括辩论事件、智能体调度和结果收集。
    """
    
    def __init__(self, num_voters: int = 101,
                 network_degree: int = 6,
                 network_rewire_prob: float = 0.3,
                 use_tot: bool = False,
                 llm_interface=None,
                 prompt_factory=None):
        """
        初始化选举模型
        
        Args:
            num_voters: 选民数量
            network_degree: 网络平均度数
            network_rewire_prob: 重连概率
            use_tot: 是否使用 ToT
            llm_interface: LLM 接口
            prompt_factory: Prompt 工厂
        """
        super().__init__()
        
        self.num_voters = num_voters
        self.use_tot = use_tot
        self.context = "2020年美国总统大选模拟"
        
        # 创建调度器
        self.schedule = mesa.time.RandomActivation(self)
        
        # 创建小世界网络
        self.network = nx.watts_strogatz_graph(
            num_voters, network_degree, network_rewire_prob
        )
        
        # 初始化记忆工厂（简化版，实际需要 LLM）
        self.memory_factory = self._create_mock_memory_factory()
        
        # 辩论事件
        self.debates = self._create_debates()
        self.current_debate_index = 0
        
        # 创建选民智能体
        self._create_voters()
        
        # 结果收集
        self.voting_results: List[Dict[str, Any]] = []
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Biden_Support": lambda m: m.count_votes(VoteChoice.BIDEN),
                "Trump_Support": lambda m: m.count_votes(VoteChoice.TRUMP),
                "Undecided": lambda m: m.count_votes(VoteChoice.UNDECIDED)
            },
            agent_reporters={
                "Vote": lambda a: a.current_vote.value if isinstance(a, ElectionVoterAgent) else None,
                "Confidence": lambda a: a.confidence if isinstance(a, ElectionVoterAgent) else None
            }
        )
    
    def _create_mock_memory_factory(self):
        """创建模拟记忆工厂（用于测试）"""
        class MockMemory:
            def __init__(self, agent):
                self.agent = agent
                self.short_memories = []
                self.long_memory = None
            
            def add_short_memory(self, source, target, action, content, ts=None):
                self.short_memories.append({
                    'source': source, 'target': target,
                    'action': action, 'content': content
                })
            
            def search_short_memory_by_doc(self, content_list):
                return {'metadatas': [self.short_memories[-5:]]}
            
            def reflect_memory(self):
                if self.short_memories:
                    self.long_memory = "基于最近经历的反思总结"
            
            def get_long_memory(self):
                return self.long_memory
        
        class MockFactory:
            def create_memory(self, agent):
                return MockMemory(agent)
        
        return MockFactory()
    
    def _create_debates(self) -> List[DebateEvent]:
        """创建辩论事件"""
        debates = [
            DebateEvent(
                round_num=1,
                topic="经济与就业",
                biden_statement="我将重建美国经济，创造高薪工作岗位...",
                trump_statement="我已经创造了史上最好的经济，将继续保持...",
                key_points=["经济", "就业", "税收"]
            ),
            DebateEvent(
                round_num=2,
                topic="新冠疫情应对",
                biden_statement="我们需要科学的防疫政策，保护人民生命...",
                trump_statement="我们已经做了很好的工作，疫苗即将到来...",
                key_points=["疫情", "医疗", "公共卫生"]
            ),
            DebateEvent(
                round_num=3,
                topic="种族正义",
                biden_statement="我们必须正视系统性种族主义问题...",
                trump_statement="我是为少数族裔做了最多事情的总统...",
                key_points=["种族", "平等", "司法"]
            ),
            DebateEvent(
                round_num=4,
                topic="气候变化",
                biden_statement="气候变化是生存威胁，我们要重返巴黎协定...",
                trump_statement="我们要平衡环境和经济发展...",
                key_points=["气候", "环境", "能源"]
            ),
            DebateEvent(
                round_num=5,
                topic="医疗保健",
                biden_statement="我将扩大奥巴马医改，降低医疗成本...",
                trump_statement="我要废除奥巴马医改，提供更好的方案...",
                key_points=["医疗", "保险", "药价"]
            ),
            DebateEvent(
                round_num=6,
                topic="外交政策",
                biden_statement="我将重建与盟友的关系，恢复美国领导力...",
                trump_statement="美国优先政策保护了美国利益...",
                key_points=["外交", "贸易", "军事"]
            )
        ]
        return debates
    
    def _create_voters(self):
        """创建选民智能体"""
        # 政治倾向分布（基于 Pew Research 数据）
        leaning_distribution = {
            PoliticalLeaning.PROGRESSIVE_LEFT: 0.06,
            PoliticalLeaning.ESTABLISHMENT_LIBERAL: 0.13,
            PoliticalLeaning.DEMOCRATIC_MAINSTAYS: 0.16,
            PoliticalLeaning.OUTSIDER_LEFT: 0.10,
            PoliticalLeaning.STRESSED_SIDELINERS: 0.15,
            PoliticalLeaning.AMBIVALENT_RIGHT: 0.12,
            PoliticalLeaning.COMMITTED_CONSERVATIVES: 0.07,
            PoliticalLeaning.POPULIST_RIGHT: 0.11,
            PoliticalLeaning.FAITH_AND_FLAG: 0.10
        }
        
        # 各倾向的初始偏好
        initial_preferences = {
            PoliticalLeaning.PROGRESSIVE_LEFT: VoteChoice.BIDEN,
            PoliticalLeaning.ESTABLISHMENT_LIBERAL: VoteChoice.BIDEN,
            PoliticalLeaning.DEMOCRATIC_MAINSTAYS: VoteChoice.BIDEN,
            PoliticalLeaning.OUTSIDER_LEFT: VoteChoice.UNDECIDED,
            PoliticalLeaning.STRESSED_SIDELINERS: VoteChoice.UNDECIDED,
            PoliticalLeaning.AMBIVALENT_RIGHT: VoteChoice.UNDECIDED,
            PoliticalLeaning.COMMITTED_CONSERVATIVES: VoteChoice.TRUMP,
            PoliticalLeaning.POPULIST_RIGHT: VoteChoice.TRUMP,
            PoliticalLeaning.FAITH_AND_FLAG: VoteChoice.TRUMP
        }
        
        # 关注议题
        issue_sets = {
            PoliticalLeaning.PROGRESSIVE_LEFT: ["气候", "种族", "医疗"],
            PoliticalLeaning.ESTABLISHMENT_LIBERAL: ["经济", "医疗", "外交"],
            PoliticalLeaning.DEMOCRATIC_MAINSTAYS: ["医疗", "就业", "种族"],
            PoliticalLeaning.OUTSIDER_LEFT: ["经济", "气候", "就业"],
            PoliticalLeaning.STRESSED_SIDELINERS: ["经济", "就业", "医疗"],
            PoliticalLeaning.AMBIVALENT_RIGHT: ["经济", "税收", "安全"],
            PoliticalLeaning.COMMITTED_CONSERVATIVES: ["税收", "外交", "安全"],
            PoliticalLeaning.POPULIST_RIGHT: ["移民", "经济", "安全"],
            PoliticalLeaning.FAITH_AND_FLAG: ["宗教", "传统", "安全"]
        }
        
        for i in range(self.num_voters):
            # 根据分布选择政治倾向
            leaning = random.choices(
                list(leaning_distribution.keys()),
                weights=list(leaning_distribution.values())
            )[0]
            
            profile = VoterProfile(
                political_leaning=leaning,
                age_group=random.choice(["18-29", "30-44", "45-64", "65+"]),
                education_level=random.choice(["高中", "本科", "研究生"]),
                key_issues=issue_sets[leaning],
                initial_preference=initial_preferences[leaning],
                susceptibility=random.uniform(0.2, 0.8)
            )
            
            agent = ElectionVoterAgent(i, self, profile, self.use_tot)
            self.schedule.add(agent)
        
        # 设置邻居关系
        for node in self.network.nodes():
            agent = self.schedule.agents[node]
            if isinstance(agent, ElectionVoterAgent):
                agent.neighbors = list(self.network.neighbors(node))
    
    def get_current_debate(self) -> Optional[DebateEvent]:
        """获取当前辩论事件"""
        if self.current_debate_index < len(self.debates):
            return self.debates[self.current_debate_index]
        return None
    
    def count_votes(self, choice: VoteChoice) -> int:
        """统计特定选择的票数"""
        count = 0
        for agent in self.schedule.agents:
            if isinstance(agent, ElectionVoterAgent) and agent.current_vote == choice:
                count += 1
        return count
    
    def step(self):
        """执行一轮模拟"""
        # 收集数据
        self.datacollector.collect(self)
        
        # 记录投票结果
        self.voting_results.append({
            'round': self.schedule.time,
            'biden': self.count_votes(VoteChoice.BIDEN),
            'trump': self.count_votes(VoteChoice.TRUMP),
            'undecided': self.count_votes(VoteChoice.UNDECIDED)
        })
        
        # 执行智能体步骤
        self.schedule.step()
        
        # 推进辩论
        if self.current_debate_index < len(self.debates):
            self.current_debate_index += 1
    
    def run_simulation(self, rounds: int = 6):
        """运行完整模拟"""
        for _ in range(rounds):
            self.step()
        
        return self.get_results()
    
    def _serialize_agent_data(self) -> Dict[str, Any]:
        """序列化智能体数据为 JSON 兼容格式"""
        try:
            df = self.datacollector.get_agent_vars_dataframe()
            if df is None or df.empty:
                return {}
            
            # 将 DataFrame 转换为简单的列表格式
            result = {}
            for col in df.columns:
                result[col] = df[col].tolist()
            
            # 添加索引信息
            result['_index'] = [str(idx) for idx in df.index.tolist()]
            return result
        except Exception as e:
            print(f"序列化 agent_data 失败: {e}")
            return {}
    
    def get_results(self) -> Dict[str, Any]:
        """获取模拟结果"""
        return {
            'voting_history': self.voting_results,
            'final_results': {
                'biden': self.count_votes(VoteChoice.BIDEN),
                'trump': self.count_votes(VoteChoice.TRUMP),
                'undecided': self.count_votes(VoteChoice.UNDECIDED)
            },
            'network_stats': {
                'nodes': self.network.number_of_nodes(),
                'edges': self.network.number_of_edges(),
                'avg_clustering': nx.average_clustering(self.network)
            },
            'agent_data': self._serialize_agent_data()
        }


def run_election_experiment(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    运行选举实验
    
    Args:
        config: 实验配置
        
    Returns:
        实验结果
    """
    if config is None:
        config = {
            'num_voters': 101,
            'network_degree': 6,
            'network_rewire_prob': 0.3,
            'use_tot': False,
            'num_rounds': 6
        }
    
    model = ElectionModel(
        num_voters=config.get('num_voters', 101),
        network_degree=config.get('network_degree', 6),
        network_rewire_prob=config.get('network_rewire_prob', 0.3),
        use_tot=config.get('use_tot', False)
    )
    
    results = model.run_simulation(rounds=config.get('num_rounds', 6))
    results['config'] = config
    
    return results


if __name__ == "__main__":
    # 运行基线实验
    print("运行基线实验（CoT）...")
    baseline_results = run_election_experiment({'use_tot': False})
    print(f"基线结果: {baseline_results['final_results']}")
    
    # 运行优化实验
    print("\n运行优化实验（ToT）...")
    optimized_results = run_election_experiment({'use_tot': True})
    print(f"优化结果: {optimized_results['final_results']}")
    
    # 保存结果
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'election', 'baseline.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': baseline_results,
            'optimized': optimized_results
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n结果已保存到 experiments/results/election/baseline.json")

