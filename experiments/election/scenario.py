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
from datetime import datetime
import random
import json
import os
import sys

# 添加项目路径
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from casevo import (
    AgentBase,
    ThoughtChain, BaseStep,
    TreeOfThought, ToTStep, EvaluatorStep, SearchStrategy,
    AdvancedMemory, AdvancedMemoryFactory,
    DecisionEvaluator, DecisionRecord, MetaCognitionModule,
    PromptFactory
)

from experiments.utils.metrics import (
    ExperimentMetrics,
    SocialEffectMetrics,
    create_experiment_metrics
)
import time

# ToT Prompt 模板目录
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'casevo', 'prompts')


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
        
        # ToT 组件（仅在有 LLM 接口时初始化）
        self._tot_instance: Optional[TreeOfThought] = None
        self._tot_initialized = False
    
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
            neighbor = self.model.agents_dict.get(neighbor_id)
            if neighbor and isinstance(neighbor, ElectionVoterAgent):
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
    
    def _init_tot(self):
        """初始化 ToT 组件（需要 LLM 接口）"""
        if self._tot_initialized:
            return self._tot_instance is not None
        
        self._tot_initialized = True
        
        # 检查是否有 LLM 接口
        llm = getattr(self.model, 'llm_interface', None)
        if llm is None:
            raise RuntimeError(
                "ToT 决策需要 LLM 接口！请在创建模型时传入 llm_interface 参数。\n"
                "示例：ElectionModel(..., llm_interface=create_default_llm())"
            )
        
        # 创建 PromptFactory
        prompt_factory = PromptFactory(PROMPTS_DIR, llm)
        
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
            max_depth=self.model.tot_config.get('max_depth', 5),
            beam_width=self.model.tot_config.get('beam_width', 3),
            pruning_threshold=self.model.tot_config.get('pruning_threshold', 0.3),
            search_strategy=SearchStrategy.BEAM
        )
        return True
    
    def _tot_decision(self, alternative: VoteChoice) -> Dict[str, Any]:
        """
        使用 Tree of Thought 进行决策
        
        调用 ToT 模块进行多路径推理，探索不同的决策分支并选择最优路径。
        """
        # 初始化 ToT（如果未初始化会抛出异常）
        self._init_tot()
        
        # 准备输入状态
        recent_memories = self.memory.get_recent_memories(5)
        memory_context = "\n".join([
            f"- {m.get('content', '')}" for m in recent_memories
        ]) if recent_memories else "无近期记忆"
        
        initial_state = {
            'question': f"是否应该从支持 {self.current_vote.value} 改为支持 {alternative.value}?",
            'current_vote': self.current_vote.value,
            'alternative': alternative.value,
            'confidence': self.confidence,
            'political_leaning': self.profile.political_leaning.value,
            'key_issues': self.profile.key_issues,
            'susceptibility': self.profile.susceptibility,
            'recent_context': memory_context
        }
        
        # 运行 ToT（追踪性能）
        start_time = time.time()
        self._tot_instance.set_input(initial_state)
        best_node = self._tot_instance.run()
        result = self._tot_instance.get_output()
        duration_ms = (time.time() - start_time) * 1000
        
        # 解析结果
        best_score = result.get('best_score', 0.5)
        reasoning = result.get('reasoning_path', '')
        nodes_explored = result.get('total_nodes_explored', 0)
        
        # 记录 LLM 调用和推理指标
        if hasattr(self.model, 'metrics'):
            self.model.metrics.performance.record_llm_call(
                duration_ms=duration_ms,
                call_id=f"tot_decision_{self.unique_id}_{self.model.schedule.time}"
            )
            self.model.metrics.reasoning.record_tot_reasoning(
                agent_id=self.component_id,
                decision_id=f"vote_{self.unique_id}_{self.model.schedule.time}",
                depth=self.model.tot_config.get('max_depth', 5),
                branches_explored=nodes_explored,
                pruned_branches=max(0, nodes_explored - self.model.tot_config.get('beam_width', 3)),
                reasoning_path=reasoning.split('\n') if reasoning else [],
                final_score=best_score
            )
        
        # 根据分数决定是否改变
        # 分数 > 0.6 表示应该改变，< 0.4 表示不应该改变
        should_change = best_score > 0.6
        new_confidence = best_score if should_change else max(0.3, 1.0 - best_score)
        
        # 记录推理过程到记忆
        self.memory.add_short_memory(
            source=self.component_id,
            target=self.component_id,
            action="tot_decision",
            content=f"ToT 推理：探索了 {nodes_explored} 个节点，"
                   f"最佳分数 {best_score:.2f}，决定{'改变' if should_change else '保持'}立场"
        )
        
        # 记录思考过程
        if hasattr(self.model, 'thought_logger') and self.model.thought_logger:
            self.model.thought_logger.record_thought(
                agent_id=self.component_id,
                agent_name=self.component_id,  # 使用component_id作为名称
                round_num=self.model.schedule.time,
                input_context=json.dumps(initial_state, ensure_ascii=False),
                memories_retrieved=[m.get('content', '') for m in recent_memories] if recent_memories else [],
                reasoning_type="tot",
                reasoning_steps=reasoning.split('\n') if reasoning else [],
                tot_branches=result.get('branches', []),
                tot_evaluations=result.get('evaluations', []),
                decision=alternative.value if should_change else self.current_vote.value,
                confidence=new_confidence,
                reasoning_summary=f"探索{nodes_explored}节点，最佳分数{best_score:.2f}"
            )
        
        return {
            'should_change': should_change,
            'new_confidence': new_confidence,
            'reasoning': reasoning,
            'nodes_explored': result.get('total_nodes_explored', 0)
        }
    
    def _simple_decision(self, alternative: VoteChoice) -> Dict[str, Any]:
        """
        简单决策（Chain of Thought 风格）
        
        使用 LLM 进行单次推理，不进行多路径探索。
        支持增强记忆检索（上下文感知+时间衰减）。
        """
        # 检查是否有 LLM 接口
        llm = getattr(self.model, 'llm_interface', None)
        if llm is None:
            raise RuntimeError(
                "决策需要 LLM 接口！请在创建模型时传入 llm_interface 参数。\n"
                "示例：ElectionModel(..., llm_interface=create_default_llm())"
            )
        
        # 增强记忆检索
        if getattr(self.model, 'use_enhanced_memory', False):
            recent_memories = self._get_enhanced_memories(alternative)
        else:
            recent_memories = self.memory.get_recent_memories(5)
        
        # 构建 CoT 提示
        prompt = f"""你是一位美国选民，请根据以下信息做出投票决策。

## 你的背景
{self.description}

## 当前状态
- 当前支持: {self.current_vote.value}
- 置信度: {self.confidence:.2f}
- 有人建议你支持: {alternative.value}

## 请分析
1. 考虑你的政治倾向和关注议题
2. 评估改变立场的利弊
3. 做出决定

请按以下格式回答：
【分析】（1-2句话）
【决定】保持原立场 / 改变立场
【置信度】0.0-1.0 之间的数字
"""
        
        # 追踪 LLM 调用性能
        start_time = time.time()
        response = llm.send_message(prompt)
        duration_ms = (time.time() - start_time) * 1000
        
        # 记录 LLM 调用
        if hasattr(self.model, 'metrics'):
            self.model.metrics.performance.record_llm_call(
                duration_ms=duration_ms,
                call_id=f"cot_decision_{self.unique_id}_{self.model.schedule.time}"
            )
            # 记录推理指标
            self.model.metrics.reasoning.record_cot_reasoning(
                agent_id=self.component_id,
                decision_id=f"vote_{self.unique_id}_{self.model.schedule.time}",
                steps=["分析", "决定"],
                final_score=self.confidence
            )
        
        # 解析响应
        should_change = "改变立场" in response
        
        # 提取置信度
        import re
        confidence_match = re.search(r'【置信度】\s*([0-9.]+)', response)
        if confidence_match:
            new_confidence = float(confidence_match.group(1))
        else:
            new_confidence = 0.6 if should_change else self.confidence
        
        # 记录到记忆
        self.memory.add_short_memory(
            source=self.component_id,
            target=self.component_id,
            action="cot_decision",
            content=f"CoT 决策：{'改变' if should_change else '保持'}立场，置信度 {new_confidence:.2f}"
        )
        
        # 记录思考过程
        if hasattr(self.model, 'thought_logger') and self.model.thought_logger:
            memory_contents = []
            if recent_memories:
                for m in recent_memories:
                    if isinstance(m, dict):
                        memory_contents.append(m.get('content', str(m)))
                    else:
                        memory_contents.append(str(m))
            
            self.model.thought_logger.record_thought(
                agent_id=self.component_id,
                agent_name=self.component_id,  # 使用component_id作为名称
                round_num=self.model.schedule.time,
                input_context=prompt,
                memories_retrieved=memory_contents,
                reasoning_type="cot",
                reasoning_steps=[response],
                decision=alternative.value if should_change else self.current_vote.value,
                confidence=new_confidence,
                reasoning_summary=f"CoT决策：{'改变' if should_change else '保持'}立场"
            )
        
        return {
            'should_change': should_change,
            'new_confidence': new_confidence,
            'reasoning': response
        }
    
    def reflect(self):
        """
        进行反思
        
        支持两种模式：
        - 静态反思：基于元认知评估触发
        - 动态反思（use_dynamic_reflection）：基于置信度阈值触发，使用 LLM 重新评估
        """
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
        
        # 动态反思模式
        if getattr(self.model, 'use_dynamic_reflection', False):
            # 置信度低于阈值时触发 LLM 反思
            threshold = getattr(self.model, 'reflection_threshold', 0.6)
            if self.confidence < threshold:
                self._llm_reflection()
        elif eval_result['needs_reflection']:
            # 静态反思模式
            self.memory.reflect_memory()
            self.confidence = max(0.3, self.confidence - 0.1)
    
    def _llm_reflection(self):
        """使用 LLM 进行动态反思"""
        llm = getattr(self.model, 'llm_interface', None)
        if llm is None:
            return
        
        # 获取最近记忆
        recent_memories = self.memory.get_recent_memories(5)
        memory_context = "\n".join([
            f"- {m.get('content', '')}" for m in recent_memories
        ]) if recent_memories else "无近期记忆"
        
        prompt = f"""你是一位美国选民，正在反思自己的投票决定。

## 当前状态
- 当前支持: {self.current_vote.value}
- 置信度: {self.confidence:.2f}（较低，需要反思）

## 最近经历
{memory_context}

## 反思任务
请反思你的决策是否合理，考虑：
1. 你的政治倾向是否与当前选择一致
2. 辩论内容是否支持你的选择
3. 是否需要调整立场

请按格式回答：
【反思】（1-2句话）
【调整】保持原立场 / 调整置信度 / 改变立场
【新置信度】0.0-1.0
"""
        
        try:
            response = llm.send_message(prompt)
            
            # 解析置信度
            import re
            match = re.search(r'【新置信度】\s*([0-9.]+)', response)
            if match:
                new_confidence = float(match.group(1))
                self.confidence = max(0.3, min(1.0, new_confidence))
            
            # 记录反思
            self.memory.add_short_memory(
                source=self.component_id,
                target=self.component_id,
                action="dynamic_reflection",
                content=f"动态反思完成，置信度调整为 {self.confidence:.2f}"
            )
        except Exception as e:
            # 反思失败，轻微降低置信度
            self.confidence = max(0.3, self.confidence - 0.05)
    
    def _get_enhanced_memories(self, context=None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        增强记忆检索（上下文感知+时间衰减）
        
        Args:
            context: 当前决策上下文（用于相关性匹配）
            top_k: 返回记忆数量
        
        Returns:
            相关记忆列表
        """
        all_memories = self.memory.get_recent_memories(20)  # 获取更多候选
        
        if not all_memories:
            return []
        
        current_time = self.model.schedule.time
        scored_memories = []
        
        for mem in all_memories:
            score = 1.0
            
            # 时间衰减（越新的记忆权重越高）
            mem_time = mem.get('timestamp', 0)
            time_diff = current_time - mem_time
            time_decay = 1.0 / (1.0 + 0.1 * time_diff)
            score *= time_decay
            
            # 上下文相关性（简单关键词匹配）
            if context:
                content = mem.get('content', '')
                context_str = str(context)
                # 关键词重叠
                content_words = set(content.lower().split())
                context_words = set(context_str.lower().split())
                overlap = len(content_words & context_words)
                relevance = 1.0 + 0.2 * overlap
                score *= relevance
            
            scored_memories.append((score, mem))
        
        # 按分数排序，返回 top_k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [mem for score, mem in scored_memories[:top_k]]
    
    def _check_neighbor_consensus(self) -> Optional[VoteChoice]:
        """
        检查邻居共识（协同决策）
        
        Returns:
            如果邻居形成共识，返回共识选择；否则返回 None
        """
        if not self.neighbors:
            return None
        
        vote_counts = {VoteChoice.BIDEN: 0, VoteChoice.TRUMP: 0, VoteChoice.UNDECIDED: 0}
        total_confidence = {VoteChoice.BIDEN: 0.0, VoteChoice.TRUMP: 0.0, VoteChoice.UNDECIDED: 0.0}
        
        for neighbor_id in self.neighbors:
            neighbor = self.model.agents_dict.get(neighbor_id)
            if neighbor and isinstance(neighbor, ElectionVoterAgent):
                vote_counts[neighbor.current_vote] += 1
                total_confidence[neighbor.current_vote] += neighbor.confidence
        
        # 计算共识：如果某个选项超过 60% 且平均置信度 > 0.6
        total_neighbors = len(self.neighbors)
        if total_neighbors == 0:  # 额外的安全检查
            return None
        
        for vote, count in vote_counts.items():
            if count / total_neighbors > 0.6:
                avg_confidence = total_confidence[vote] / count if count > 0 else 0
                if avg_confidence > 0.6:
                    return vote
        
        return None
    
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
        """
        每轮执行
        
        支持协同决策模式（use_collaborative）：
        - 在个体决策后检查邻居共识
        - 如果邻居形成强共识，使用 LLM 评估是否跟随
        """
        # 记录本轮开始状态
        start_vote = self.current_vote
        start_confidence = self.confidence
        reasoning_steps = []
        
        # 1. 观看辩论（如果有）
        current_debate = self.model.get_current_debate()
        if current_debate:
            self.watch_debate(current_debate)
            # DebateEvent 是 dataclass，直接访问 topic 属性
            debate_topic = getattr(current_debate, 'topic', '未知主题')
            reasoning_steps.append(f"观看辩论: {debate_topic}")
        
        # 2. 与邻居讨论
        self.discuss_with_neighbors()
        reasoning_steps.append(f"与{len(self.neighbors)}位邻居讨论")
        
        # 3. 反思
        self.reflect()
        reasoning_steps.append(f"进行反思（置信度阈值: {getattr(self.model, 'reflection_threshold', 0.6)}）")
        
        # 4. 协同决策检查（如果启用）
        if getattr(self.model, 'use_collaborative', False):
            consensus = self._check_neighbor_consensus()
            if consensus and consensus != self.current_vote:
                # 邻居形成共识，考虑是否跟随
                self._consider_collaborative_change(consensus)
                reasoning_steps.append(f"协同决策: 邻居共识为{consensus.value}")
        
        # 5. 更新投票意向
        self.vote()
        
        # 6. 记录本轮思考过程
        self._log_round_thought(
            start_vote=start_vote,
            start_confidence=start_confidence,
            reasoning_steps=reasoning_steps
        )
    
    def _log_round_thought(self, start_vote: VoteChoice, start_confidence: float,
                           reasoning_steps: List[str]):
        """记录每轮的思考过程"""
        if not hasattr(self.model, 'thought_logger') or not self.model.thought_logger:
            return
        
        # 获取最近记忆
        recent_memories = []
        try:
            memories = self.memory.get_recent_memories(3)
            if memories:
                for m in memories:
                    if isinstance(m, dict):
                        recent_memories.append(m.get('content', str(m))[:100])
                    else:
                        recent_memories.append(str(m)[:100])
        except:
            pass
        
        # 判断是否改变了立场
        vote_changed = start_vote != self.current_vote
        
        self.model.thought_logger.record_thought(
            agent_id=self.component_id,
            agent_name=self.component_id,  # 使用component_id作为名称
            round_num=self.model.schedule.time,
            input_context=f"政治倾向: {self.profile.political_leaning.value}, "
                         f"初始投票: {start_vote.value}, 初始置信度: {start_confidence:.2f}",
            memories_retrieved=recent_memories,
            reasoning_type="tot" if self.use_tot else "cot",
            reasoning_steps=reasoning_steps,
            reflection_triggered=self.confidence < getattr(self.model, 'reflection_threshold', 0.6),
            decision=self.current_vote.value,
            confidence=self.confidence,
            reasoning_summary=f"{'改变' if vote_changed else '保持'}立场 ({start_vote.value}->{self.current_vote.value})"
        )
    
    def _consider_collaborative_change(self, consensus_vote: VoteChoice):
        """
        考虑协同改变立场
        
        Args:
            consensus_vote: 邻居共识选择
        """
        llm = getattr(self.model, 'llm_interface', None)
        if llm is None:
            # 没有 LLM，使用简单规则
            if self.profile.susceptibility > 0.6 and self.confidence < 0.5:
                self.current_vote = consensus_vote
                self.confidence = 0.5
            return
        
        prompt = f"""你是一位美国选民，周围的人形成了明显的共识。

## 你的状态
- 当前支持: {self.current_vote.value}
- 置信度: {self.confidence:.2f}
- 政治倾向: {self.profile.political_leaning.value}

## 邻居共识
- 共识选择: {consensus_vote.value}
- 超过60%的邻居高置信度支持此选择

## 请决策
考虑你是否应该跟随邻居的共识，还是坚持自己的立场。

【决策】跟随共识 / 坚持立场
【新置信度】0.0-1.0
"""
        
        try:
            response = llm.send_message(prompt)
            
            if '跟随共识' in response:
                self.current_vote = consensus_vote
                # 解析置信度
                import re
                match = re.search(r'【新置信度】\s*([0-9.]+)', response)
                if match:
                    self.confidence = float(match.group(1))
                else:
                    self.confidence = 0.6
                
                self.memory.add_short_memory(
                    source=self.component_id,
                    target=self.component_id,
                    action="collaborative_decision",
                    content=f"跟随邻居共识，改为支持 {consensus_vote.value}"
                )
        except Exception:
            pass  # 决策失败，保持原立场


class ElectionModel(mesa.Model):
    """
    选举模拟模型
    
    管理选举过程，包括辩论事件、智能体调度和结果收集。
    """
    
    def __init__(self, num_voters: int = 101,
                 network_degree: int = 6,
                 network_rewire_prob: float = 0.3,
                 use_tot: bool = False,
                 use_enhanced_memory: bool = False,
                 use_dynamic_reflection: bool = False,
                 use_collaborative: bool = False,
                 llm_interface=None,
                 prompt_factory=None,
                 tot_config: Optional[Dict[str, Any]] = None,
                 memory_path: Optional[str] = None,
                 thought_logger=None,
                 experiment_id: str = None):
        """
        初始化选举模型
        
        Args:
            num_voters: 选民数量
            network_degree: 网络平均度数
            network_rewire_prob: 重连概率
            use_tot: 是否使用 ToT 多层次推理
            use_enhanced_memory: 是否使用增强记忆检索（上下文感知+时间衰减）
            use_dynamic_reflection: 是否使用动态反思（置信度触发）
            use_collaborative: 是否使用协同决策（邻居共识）
            llm_interface: LLM 接口
            prompt_factory: Prompt 工厂
            tot_config: ToT 配置参数
            memory_path: 持久化记忆存储路径（None则使用内存模式）
            thought_logger: 思考过程日志记录器
            experiment_id: 实验唯一标识
        """
        super().__init__()
        
        self.num_voters = num_voters
        self.use_tot = use_tot
        self.use_enhanced_memory = use_enhanced_memory
        self.use_dynamic_reflection = use_dynamic_reflection
        self.use_collaborative = use_collaborative
        self.context = "2020年美国总统大选模拟"
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # LLM 接口（用于 ToT）
        self.llm_interface = llm_interface
        self.prompt_factory = prompt_factory
        
        # 思考过程日志
        self.thought_logger = thought_logger
        
        # ToT 配置
        self.tot_config = tot_config or {
            'max_depth': 5,
            'beam_width': 3,
            'pruning_threshold': 0.3
        }
        
        # 反思阈值（Proposal 要求 0.6）
        self.reflection_threshold = 0.6
        
        # 创建调度器
        self.schedule = mesa.time.RandomActivation(self)
        
        # 创建小世界网络
        self.network = nx.watts_strogatz_graph(
            num_voters, network_degree, network_rewire_prob
        )
        
        # 初始化记忆工厂（支持持久化）
        reflect_prompt = "请根据以下记忆进行反思，总结关键信息和决策依据。"
        self.memory_path = memory_path
        self.memory_factory = AdvancedMemoryFactory(
            tar_llm=self.llm_interface,
            memory_num=5,
            prompt=reflect_prompt,
            model=self,
            tar_path=memory_path  # 持久化路径
        )
        
        # 辩论事件
        self.debates = self._create_debates()
        self.current_debate_index = 0
        
        # 智能体字典（用于快速索引）- 必须在 _create_voters 之前初始化
        self.agents_dict: Dict[int, ElectionVoterAgent] = {}
        
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
        
        # 评估指标（符合 Proposal 四维度要求）
        self.metrics = create_experiment_metrics()
        self.metrics.start_experiment()
    
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
            self.agents_dict[i] = agent
        
        # 设置邻居关系
        for node in self.network.nodes():
            agent = self.agents_dict.get(node)
            if agent and isinstance(agent, ElectionVoterAgent):
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
        vote_distribution = {
            'Biden': self.count_votes(VoteChoice.BIDEN),
            'Trump': self.count_votes(VoteChoice.TRUMP),
            'Undecided': self.count_votes(VoteChoice.UNDECIDED)
        }
        
        self.voting_results.append({
            'round': self.schedule.time,
            'biden': vote_distribution['Biden'],
            'trump': vote_distribution['Trump'],
            'undecided': vote_distribution['Undecided']
        })
        
        # 记录投票分布用于社会效应指标
        self.metrics.record_vote_distribution(vote_distribution)
        
        # 执行智能体步骤
        self.schedule.step()
        
        # 推进辩论
        if self.current_debate_index < len(self.debates):
            self.current_debate_index += 1
    
    def run_simulation(self, rounds: int = 6, verbose: bool = True):
        """运行完整模拟"""
        for i in range(rounds):
            if verbose:
                votes = self.count_votes(VoteChoice.BIDEN), self.count_votes(VoteChoice.TRUMP)
                print(f"      辩论 {i+1}/{rounds}: Biden={votes[0]}, Trump={votes[1]}", end='\r')
            self.step()
        
        if verbose:
            final = self.count_votes(VoteChoice.BIDEN), self.count_votes(VoteChoice.TRUMP), self.count_votes(VoteChoice.UNDECIDED)
            print(f"      完成: Biden={final[0]}, Trump={final[1]}, Undecided={final[2]}    ")
        
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
        # 获取评估指标
        metrics_report = self.metrics.get_full_report()
        metrics_summary = self.metrics.get_summary()
        
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
            'agent_data': self._serialize_agent_data(),
            # Proposal 要求的四维度评估指标
            'evaluation_metrics': metrics_report,
            'metrics_summary': metrics_summary
        }


def run_election_experiment(config: Dict[str, Any] = None, 
                            llm_interface=None,
                            memory_path: str = None,
                            thought_logger=None,
                            experiment_id: str = None) -> Dict[str, Any]:
    """
    运行选举实验
    
    Args:
        config: 实验配置
        llm_interface: LLM 接口（智能体实验必须提供）
        memory_path: 持久化记忆存储路径（None则使用内存模式）
        thought_logger: 思考过程日志记录器
        experiment_id: 实验唯一标识
        
    Returns:
        实验结果
        
    Raises:
        RuntimeError: 如果未提供 llm_interface
    """
    if config is None:
        config = {
            'num_voters': 101,
            'network_degree': 6,
            'network_rewire_prob': 0.3,
            'use_tot': False,
            'num_rounds': 6
        }
    
    # 智能体实验必须有 LLM
    if llm_interface is None:
        raise RuntimeError(
            "选举实验需要 LLM 接口！\n"
            "这是一个智能体实验，必须使用 LLM 进行决策推理。\n\n"
            "使用方法：\n"
            "  from casevo import create_default_llm\n"
            "  llm = create_default_llm()\n"
            "  results = run_election_experiment(config, llm_interface=llm)\n\n"
            "如果只想运行测试，请使用 with_llm.py 中的函数。"
        )
    
    # 提取 ToT 配置
    tot_config = {
        'max_depth': config.get('tot_max_depth', 5),
        'beam_width': config.get('tot_beam_width', 3),
        'pruning_threshold': config.get('tot_pruning_threshold', 0.3)
    }
    
    # 如果配置中有memory_path，优先使用
    if memory_path is None:
        memory_path = config.get('memory_path', None)
    
    model = ElectionModel(
        num_voters=config.get('num_voters', 101),
        network_degree=config.get('network_degree', 6),
        network_rewire_prob=config.get('network_rewire_prob', 0.3),
        use_tot=config.get('use_tot', False),
        use_enhanced_memory=config.get('use_enhanced_memory', False),
        use_dynamic_reflection=config.get('use_dynamic_reflection', False),
        use_collaborative=config.get('use_collaborative', False),
        llm_interface=llm_interface,
        tot_config=tot_config,
        memory_path=memory_path,
        thought_logger=thought_logger,
        experiment_id=experiment_id
    )
    
    results = model.run_simulation(rounds=config.get('num_rounds', 6))
    results['config'] = config
    results['tot_enabled'] = config.get('use_tot', False)
    results['enhanced_memory_enabled'] = config.get('use_enhanced_memory', False)
    results['dynamic_reflection_enabled'] = config.get('use_dynamic_reflection', False)
    results['collaborative_enabled'] = config.get('use_collaborative', False)
    results['memory_path'] = memory_path
    results['experiment_id'] = model.experiment_id
    
    # 保存思考日志摘要
    if thought_logger:
        results['thought_summary'] = thought_logger.generate_summary()
    
    return results


if __name__ == "__main__":
    from casevo import create_default_llm
    
    print("=" * 60)
    print("选举投票智能体实验")
    print("=" * 60)
    
    # 初始化 LLM
    print("\n初始化 LLM...")
    try:
        llm = create_default_llm()
        test_response = llm.send_message("回复 OK")
        if not test_response:
            raise Exception("LLM 无响应")
        print(f"LLM 连接成功")
    except Exception as e:
        print(f"错误：无法连接 LLM - {e}")
        print("请检查 API 密钥配置后重试。")
        sys.exit(1)
    
    # 实验配置
    test_config = {
        'num_voters': 101,
        'num_rounds': 6,
        'use_tot': False
    }
    
    # 运行 CoT 实验
    print("\n运行 CoT 实验...")
    baseline_results = run_election_experiment(test_config, llm_interface=llm)
    print(f"CoT 结果: {baseline_results['final_results']}")
    
    # 运行 ToT 实验
    print("\n运行 ToT 实验...")
    test_config['use_tot'] = True
    optimized_results = run_election_experiment(test_config, llm_interface=llm)
    print(f"ToT 结果: {optimized_results['final_results']}")
    
    # 保存结果
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'election', 'baseline.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': baseline_results,
            'optimized': optimized_results
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n结果已保存到 {results_path}")

