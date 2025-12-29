"""
CoT vs ToT 推理机制对比实验

对比线性思维链(CoT)和树状思维(ToT)在选举投票场景中的决策质量。
使用真正的 TreeOfThought 模块进行多路径推理。
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
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from casevo import (
    create_llm, create_default_llm,
    TreeOfThought, ToTStep, EvaluatorStep, SearchStrategy,
    PromptFactory
)

# Prompt 模板目录
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'casevo', 'prompts')


class VoteChoice(Enum):
    BIDEN = "Biden"
    TRUMP = "Trump"
    UNDECIDED = "Undecided"


@dataclass
class VoterProfile:
    """选民画像"""
    name: str
    age: int
    political_leaning: str  # liberal / conservative / moderate
    key_issues: List[str]
    initial_preference: VoteChoice


# 简化的选民配置
VOTER_CONFIGS = [
    {'leaning': 'liberal', 'initial': VoteChoice.BIDEN, 'dist': 0.35},
    {'leaning': 'conservative', 'initial': VoteChoice.TRUMP, 'dist': 0.35},
    {'leaning': 'moderate', 'initial': VoteChoice.UNDECIDED, 'dist': 0.30},
]

ISSUES_BY_LEANING = {
    'liberal': ['气候变化', '医疗改革', '种族正义', '移民权利'],
    'conservative': ['减税', '边境安全', '宗教自由', '法律秩序'],
    'moderate': ['经济发展', '医疗费用', '就业', '社会稳定'],
}

FIRST_NAMES = ['James', 'John', 'Robert', 'Michael', 'William', 'David',
               'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Susan']
LAST_NAMES = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Davis']


class CoTVoterAgent(mesa.Agent):
    """Chain of Thought 选民智能体"""
    
    def __init__(self, unique_id: int, model: 'ComparisonModel', profile: VoterProfile):
        super().__init__(unique_id, model)
        self.profile = profile
        self.current_vote = profile.initial_preference
        self.reasoning_history: List[str] = []
        self.neighbors: List[int] = []
    
    def make_decision_cot(self, debate_info: str, neighbor_opinions: List[str]) -> VoteChoice:
        """使用 Chain of Thought (线性推理) 进行决策"""
        
        prompt = f"""你是 {self.profile.name}，一位{self.profile.age}岁的美国选民。
政治倾向: {self.profile.political_leaning}
关注议题: {', '.join(self.profile.key_issues)}
当前倾向: {self.current_vote.value}

## 辩论内容
{debate_info}

## 邻居观点
{chr(10).join(neighbor_opinions) if neighbor_opinions else '(无)'}

## 任务
请使用线性推理方式（Chain of Thought），按步骤分析后做出投票决定。

按以下格式回答：
【步骤1-分析辩论】这场辩论中...（1句话）
【步骤2-评估候选人】Biden的优点...Trump的优点...（各1句话）
【步骤3-考虑个人议题】对于我关心的{self.profile.key_issues[0]}...（1句话）
【步骤4-最终决定】Biden / Trump / Undecided（只写一个）
"""
        
        try:
            response = self.model.llm.send_message(prompt)
            self.reasoning_history.append(response)
            return self._parse_vote(response)
        except Exception as e:
            print(f"  CoT Agent {self.unique_id} 错误: {e}")
            return self.current_vote
    
    def _parse_vote(self, response: str) -> VoteChoice:
        """解析投票结果"""
        response_lower = response.lower()
        if '【步骤4-最终决定】' in response or '【最终决定】' in response:
            final_part = response.split('决定】')[-1].lower()
            if 'biden' in final_part:
                return VoteChoice.BIDEN
            elif 'trump' in final_part:
                return VoteChoice.TRUMP
        return self.current_vote


class ToTVoterAgent(mesa.Agent):
    """Tree of Thought 选民智能体（使用真正的 ToT 模块）"""
    
    def __init__(self, unique_id: int, model: 'ComparisonModel', profile: VoterProfile):
        super().__init__(unique_id, model)
        self.profile = profile
        self.current_vote = profile.initial_preference
        self.reasoning_history: List[str] = []
        self.neighbors: List[int] = []
        self.component_id = f"voter_{unique_id}"
        self.description = f"{profile.name}, {profile.age}岁, {profile.political_leaning}"
        self.context = {'key_issues': profile.key_issues}
        
        # ToT 实例（延迟初始化）
        self._tot_instance: Optional[TreeOfThought] = None
        self._tot_initialized = False
    
    def _init_tot(self):
        """初始化 ToT 组件"""
        if self._tot_initialized:
            return
        self._tot_initialized = True
        
        # 创建 PromptFactory
        prompt_factory = PromptFactory(PROMPTS_DIR, self.model.llm)
        
        # 创建 ToT 步骤
        thought_step = ToTStep(
            step_id="vote_branch",
            tar_prompt=prompt_factory.get_template("tot_generate.j2"),
            num_branches=3
        )
        
        # 创建评估步骤
        evaluator_step = EvaluatorStep(
            step_id="vote_evaluate",
            tar_prompt=prompt_factory.get_template("tot_evaluate.j2"),
            score_range=(0.0, 1.0)
        )
        
        # 创建 ToT 实例
        self._tot_instance = TreeOfThought(
            agent=self,
            thought_step=thought_step,
            evaluator_step=evaluator_step,
            max_depth=4,
            beam_width=3,
            pruning_threshold=0.3,
            search_strategy=SearchStrategy.BEAM
        )
    
    def make_decision_tot(self, debate_info: str, neighbor_opinions: List[str]) -> VoteChoice:
        """使用真正的 Tree of Thought 模块进行决策"""
        
        # 初始化 ToT
        self._init_tot()
        
        # 准备输入状态
        initial_state = {
            'question': f"作为 {self.profile.name}，我应该投票给谁？",
            'voter_name': self.profile.name,
            'voter_age': self.profile.age,
            'political_leaning': self.profile.political_leaning,
            'key_issues': self.profile.key_issues,
            'current_preference': self.current_vote.value,
            'debate_info': debate_info,
            'neighbor_opinions': neighbor_opinions
        }
        
        try:
            # 运行 ToT
            self._tot_instance.set_input(initial_state)
            best_node = self._tot_instance.run()
            result = self._tot_instance.get_output()
            
            # 记录推理路径
            reasoning_path = result.get('reasoning_path', '')
            self.reasoning_history.append(f"[ToT] 探索 {result.get('total_nodes_explored', 0)} 节点\n{reasoning_path}")
            
            # 解析最佳节点的决策
            best_score = result.get('best_score', 0.5)
            best_state = result.get('best_state', {})
            
            # 从最佳状态中提取决策
            return self._extract_vote_from_state(best_state, best_score)
            
        except Exception as e:
            print(f"  ToT Agent {self.unique_id} 错误: {e}")
            # 回退到单次 LLM 调用
            return self._fallback_decision(debate_info, neighbor_opinions)
    
    def _extract_vote_from_state(self, state: Dict, score: float) -> VoteChoice:
        """从 ToT 状态中提取投票决策"""
        response = str(state.get('last_response', ''))
        response_lower = response.lower()
        
        # 尝试从响应中解析
        if 'biden' in response_lower:
            return VoteChoice.BIDEN
        elif 'trump' in response_lower:
            return VoteChoice.TRUMP
        
        # 根据分数和政治倾向决定
        if self.profile.political_leaning == 'liberal':
            return VoteChoice.BIDEN if score > 0.5 else VoteChoice.UNDECIDED
        elif self.profile.political_leaning == 'conservative':
            return VoteChoice.TRUMP if score > 0.5 else VoteChoice.UNDECIDED
        else:
            return VoteChoice.UNDECIDED
    
    def _fallback_decision(self, debate_info: str, neighbor_opinions: List[str]) -> VoteChoice:
        """ToT 失败时的回退决策"""
        prompt = f"""你是 {self.profile.name}，{self.profile.age}岁，政治倾向 {self.profile.political_leaning}。
关注议题: {', '.join(self.profile.key_issues)}

辩论内容: {debate_info}

请直接回答：你投票给 Biden 还是 Trump？（只回答一个名字）"""
        
        try:
            response = self.model.llm.send_message(prompt)
            if 'biden' in response.lower():
                return VoteChoice.BIDEN
            elif 'trump' in response.lower():
                return VoteChoice.TRUMP
        except:
            pass
        return self.current_vote


class ComparisonModel(mesa.Model):
    """CoT vs ToT 对比模型"""
    
    def __init__(self, num_voters: int = 30, llm=None, use_tot: bool = False):
        super().__init__()
        
        self.num_voters = num_voters
        self.llm = llm or create_default_llm()
        self.use_tot = use_tot
        self.schedule = mesa.time.RandomActivation(self)
        
        # 创建网络
        self.network = nx.watts_strogatz_graph(num_voters, 4, 0.3)
        
        self.current_round = 0
        self.results_history = []
        self.voters: Dict[int, mesa.Agent] = {}
        
        # 辩论内容
        self.debates = [
            {
                'topic': '经济政策',
                'biden': '我将投资基础设施，为中产阶级创造就业，提高最低工资。',
                'trump': '我的减税政策创造了史上最低失业率，股市屡创新高。'
            },
            {
                'topic': '医疗保健',
                'biden': '我将扩大医保覆盖，降低药价，保护既往病症患者。',
                'trump': '奥巴马医改是灾难，我会提供更好更便宜的方案。'
            },
            {
                'topic': '社会议题',
                'biden': '我支持种族正义改革，但反对打砸抢。我尊重所有人的权利。',
                'trump': '我是法律与秩序的总统，我保护社区安全，支持执法部门。'
            }
        ]
        
        # 创建选民
        self._create_voters()
    
    def _create_voters(self):
        """创建选民"""
        configs = VOTER_CONFIGS
        weights = [c['dist'] for c in configs]
        
        for i in range(self.num_voters):
            config = random.choices(configs, weights=weights)[0]
            leaning = config['leaning']
            
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            age = random.randint(25, 70)
            issues = random.sample(ISSUES_BY_LEANING[leaning], 2)
            
            profile = VoterProfile(
                name=name,
                age=age,
                political_leaning=leaning,
                key_issues=issues,
                initial_preference=config['initial']
            )
            
            if self.use_tot:
                agent = ToTVoterAgent(i, self, profile)
            else:
                agent = CoTVoterAgent(i, self, profile)
            
            agent.neighbors = list(self.network.neighbors(i))
            self.schedule.add(agent)
            self.voters[i] = agent
        
        # 打印分布
        dist = {'liberal': 0, 'conservative': 0, 'moderate': 0}
        for agent in self.schedule.agents:
            dist[agent.profile.political_leaning] += 1
        
        method = "ToT (树状推理)" if self.use_tot else "CoT (线性推理)"
        print(f"\n[{method}] 选民分布: liberal={dist['liberal']}, "
              f"conservative={dist['conservative']}, moderate={dist['moderate']}")
    
    def get_current_debate(self) -> Dict:
        idx = min(self.current_round, len(self.debates) - 1)
        return self.debates[idx]
    
    def count_votes(self) -> Dict[str, int]:
        counts = {'Biden': 0, 'Trump': 0, 'Undecided': 0}
        for agent in self.schedule.agents:
            counts[agent.current_vote.value] += 1
        return counts
    
    def step(self):
        """执行一轮"""
        method = "ToT" if self.use_tot else "CoT"
        print(f"\n--- [{method}] 第 {self.current_round + 1} 轮 ---")
        
        debate = self.get_current_debate()
        debate_info = f"""主题: {debate['topic']}
Biden: {debate['biden']}
Trump: {debate['trump']}"""
        
        for agent in self.schedule.agents:
            # 获取邻居意见
            neighbor_opinions = []
            for n_id in agent.neighbors[:2]:
                n_agent = self.voters.get(n_id)
                if n_agent:
                    neighbor_opinions.append(f"邻居支持: {n_agent.current_vote.value}")
            
            # 决策
            if self.use_tot:
                new_vote = agent.make_decision_tot(debate_info, neighbor_opinions)
            else:
                new_vote = agent.make_decision_cot(debate_info, neighbor_opinions)
            
            agent.current_vote = new_vote
            time.sleep(0.3)
        
        votes = self.count_votes()
        self.results_history.append({
            'round': self.current_round + 1,
            'votes': votes
        })
        
        print(f"结果: Biden={votes['Biden']}, Trump={votes['Trump']}, Undecided={votes['Undecided']}")
        
        self.current_round += 1
    
    def run(self, rounds: int = 3):
        """运行模拟"""
        method = "ToT (树状推理)" if self.use_tot else "CoT (线性推理)"
        print(f"\n{'='*50}")
        print(f"开始 {method} 实验 - {self.num_voters} 位选民, {rounds} 轮")
        print(f"{'='*50}")
        
        initial = self.count_votes()
        print(f"初始: Biden={initial['Biden']}, Trump={initial['Trump']}, Undecided={initial['Undecided']}")
        
        for _ in range(rounds):
            self.step()
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """获取结果"""
        # 计算决策一致性（相同倾向选民的投票一致性）
        consistency = self._calculate_consistency()
        
        return {
            'method': 'ToT' if self.use_tot else 'CoT',
            'num_voters': self.num_voters,
            'rounds': self.current_round,
            'history': self.results_history,
            'final_votes': self.count_votes(),
            'consistency': consistency,
            'agent_samples': [
                {
                    'id': a.unique_id,
                    'leaning': a.profile.political_leaning,
                    'initial': a.profile.initial_preference.value,
                    'final': a.current_vote.value,
                    'changed': a.profile.initial_preference != a.current_vote,
                    'reasoning_sample': a.reasoning_history[-1][:500] if a.reasoning_history else None
                }
                for a in list(self.schedule.agents)[:5]  # 采样5个
            ]
        }
    
    def _calculate_consistency(self) -> Dict:
        """计算决策一致性"""
        # 计算同一政治倾向的选民投票是否一致
        by_leaning = {'liberal': [], 'conservative': [], 'moderate': []}
        
        for agent in self.schedule.agents:
            leaning = agent.profile.political_leaning
            by_leaning[leaning].append(agent.current_vote.value)
        
        consistency = {}
        for leaning, votes in by_leaning.items():
            if not votes:
                consistency[leaning] = 0
                continue
            # 计算最常见投票的比例
            from collections import Counter
            counter = Counter(votes)
            most_common = counter.most_common(1)[0][1]
            consistency[leaning] = most_common / len(votes)
        
        return consistency


def run_cot_vs_tot_experiment(num_voters: int = 30, rounds: int = 3):
    """运行 CoT vs ToT 对比实验"""
    
    print("\n" + "="*60)
    print("CoT vs ToT 推理机制对比实验")
    print("="*60)
    
    results = {}
    
    llm = create_default_llm()
    
    # 固定随机种子
    random.seed(42)
    
    # 1. CoT 实验
    print("\n【实验1 - Chain of Thought (线性推理)】")
    model_cot = ComparisonModel(num_voters=num_voters, llm=llm, use_tot=False)
    results['cot'] = model_cot.run(rounds=rounds)
    
    # 重置随机种子
    random.seed(42)
    
    # 2. ToT 实验
    print("\n【实验2 - Tree of Thought (树状推理)】")
    model_tot = ComparisonModel(num_voters=num_voters, llm=llm, use_tot=True)
    results['tot'] = model_tot.run(rounds=rounds)
    
    # 输出对比
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    
    print(f"\n{'指标':<25} {'CoT (线性)':<15} {'ToT (树状)':<15}")
    print("-"*55)
    
    cot_final = results['cot']['final_votes']
    tot_final = results['tot']['final_votes']
    
    print(f"{'Biden 票数':<25} {cot_final['Biden']:<15} {tot_final['Biden']:<15}")
    print(f"{'Trump 票数':<25} {cot_final['Trump']:<15} {tot_final['Trump']:<15}")
    print(f"{'Undecided 票数':<25} {cot_final['Undecided']:<15} {tot_final['Undecided']:<15}")
    
    print(f"\n{'决策一致性（同倾向）':<25}")
    for leaning in ['liberal', 'conservative', 'moderate']:
        cot_cons = results['cot']['consistency'].get(leaning, 0)
        tot_cons = results['tot']['consistency'].get(leaning, 0)
        print(f"  {leaning:<23} {cot_cons:<15.1%} {tot_cons:<15.1%}")
    
    # 保存结果
    output_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'comparisons', 'cot_vs_tot.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CoT vs ToT 对比实验')
    parser.add_argument('--voters', type=int, default=30, help='选民数量')
    parser.add_argument('--rounds', type=int, default=3, help='辩论轮数')
    
    args = parser.parse_args()
    
    run_cot_vs_tot_experiment(
        num_voters=args.voters,
        rounds=args.rounds
    )

