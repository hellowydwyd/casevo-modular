"""
资源分配实验场景

模拟多智能体资源协商分配过程：
- 50 个智能体
- 固定总资源 1000 单位
- 多轮协商机制
- 公平性评估

注意：这是智能体实验，需要 LLM 接口才能运行。
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
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from casevo import (
    AgentBase,
    CollaborativeDecisionMaker, 
    Message, 
    StandardNegotiationProtocol,
    DecisionMode,
    DistributedConsensus,
    DecisionEvaluator, DecisionRecord,
    create_default_llm,
    AdvancedMemoryFactory,
    TreeOfThought, ToTStep, EvaluatorStep, SearchStrategy,
    PromptFactory
)

# Prompt 模板目录
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

from experiments.utils.metrics import (
    ExperimentMetrics,
    create_experiment_metrics
)
import time


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
        
        # ToT 组件（仅在有 LLM 接口时初始化）
        self._tot_instance: Optional[TreeOfThought] = None
        self._tot_initialized = False
    
    def _generate_description(self, need: ResourceNeed) -> str:
        """生成智能体描述"""
        return f"""你是一个资源申请方，具有以下需求特征：
- 最低需求：{need.minimum} 单位
- 期望需求：{need.desired} 单位
- 优先级：{need.priority.value}
- 需求理由：{need.justification}

你需要在协商中争取足够的资源，同时考虑整体公平性。"""
    
    def _init_tot(self):
        """
        初始化 ToT 组件（需要 LLM 接口）
        
        使用真正的 TreeOfThought 类进行多路径策略探索：
        - 生成多个协商策略分支（激进、保守、均衡）
        - 每个分支独立评估
        - Beam Search 选择最优策略
        """
        if self._tot_initialized:
            return self._tot_instance is not None
        
        self._tot_initialized = True
        
        # 检查是否有 LLM 接口
        llm = getattr(self.model, 'llm_interface', None)
        if llm is None:
            raise RuntimeError(
                "ToT 协商需要 LLM 接口！请在创建模型时传入 llm_interface 参数。"
            )
        
        # 创建 PromptFactory
        prompt_factory = PromptFactory(PROMPTS_DIR, llm)
        
        # 创建 ToT 步骤（生成3个策略分支）
        thought_step = ToTStep(
            step_id="negotiate_branch",
            tar_prompt=prompt_factory.get_template("resource_tot_generate.j2"),
            num_branches=3  # 激进、保守、均衡
        )
        
        # 创建评估步骤
        evaluator_step = EvaluatorStep(
            step_id="negotiate_evaluate",
            tar_prompt=prompt_factory.get_template("resource_tot_evaluate.j2"),
            score_range=(0.0, 1.0)
        )
        
        # 创建 ToT 实例
        self._tot_instance = TreeOfThought(
            agent=self,
            thought_step=thought_step,
            evaluator_step=evaluator_step,
            max_depth=2,            # 协商策略不需要太深
            beam_width=3,           # 保留3个最佳分支
            pruning_threshold=0.3,  # 低于0.3的分支剪枝
            search_strategy=SearchStrategy.BEAM
        )
        return True
    
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
        根据情况调整提案（使用 LLM 进行协商策略）
        
        支持三组对照设计（符合 Proposal 要求）：
        - 基线组（CoT）：使用 LLM 单次推理协商
        - 优化组 A（ToT）：使用 LLM 多路径策略探索
        - 优化组 B（全部优化）：ToT + 增强记忆 + 动态反思
        
        Args:
            total_requests: 总请求量
            available_resources: 可用资源
            others_proposals: 其他人的提案
            
        Returns:
            调整后的提案
        """
        # 获取 LLM 接口
        llm = getattr(self.model, 'llm_interface', None)
        if llm is None:
            raise RuntimeError(
                "资源协商需要 LLM 接口！请在创建模型时传入 llm_interface 参数。"
            )
        
        # 计算资源紧张程度
        scarcity = total_requests / available_resources if available_resources > 0 else float('inf')
        
        # 根据配置选择推理方式
        use_tot = getattr(self.model, 'use_tot', False)
        if use_tot:
            # ToT 多路径策略探索（优化组 A/B）
            new_request, justification = self._tot_negotiate(
                total_requests, available_resources, others_proposals, scarcity
            )
        else:
            # CoT 单次推理（基线组）
            new_request, justification = self._cot_negotiate(
                total_requests, available_resources, others_proposals, scarcity
            )
        
        # 更新妥协意愿
        new_compromise = min(1.0, self.current_proposal.willingness_to_compromise + 0.1)
        
        self.current_proposal = AllocationProposal(
            agent_id=self.component_id,
            requested_amount=new_request,
            justification=justification,
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
            'llm_used': True,
            'tot_used': use_tot
        })
        
        # 动态反思：如果变化太大，重新评估
        if getattr(self.model, 'use_dynamic_reflection', False):
            self._dynamic_reflect_on_proposal(new_request, scarcity)
        
        return self.current_proposal
    
    def _cot_negotiate(self, total_requests: int, available_resources: int,
                       others_proposals: List[AllocationProposal], 
                       scarcity: float) -> Tuple[int, str]:
        """
        CoT（Chain of Thought）单次推理协商 - 基线组
        
        使用 LLM 进行简单的单次推理决定协商策略。
        """
        llm = self.model.llm_interface
        
        # 增强记忆检索
        if getattr(self.model, 'use_enhanced_memory', False):
            memory_context = self._get_enhanced_memory_context(scarcity)
        else:
            memory_context = ""
        
        # 构建协商提示
        others_summary = "\n".join([
            f"- {p.agent_id}: 请求 {p.requested_amount}，妥协意愿 {p.willingness_to_compromise:.1f}"
            for p in others_proposals[:5]
        ]) if others_proposals else "暂无其他人的提案"
        
        memory_section = f"\n## 历史协商经验\n{memory_context}\n" if memory_context else ""
        
        prompt = f"""你是一个资源协商智能体，需要决定调整后的资源请求量。

## 你的情况
- 最低需求: {self.need.minimum} 单位
- 期望需求: {self.need.desired} 单位
- 优先级: {self.need.priority.value}
- 当前请求: {self.current_proposal.requested_amount} 单位
- 需求理由: {self.need.justification}

## 整体情况
- 总资源量: {available_resources} 单位
- 总请求量: {total_requests} 单位
- 资源紧张度: {scarcity:.2f} (>1 表示供不应求)

## 其他人的提案
{others_summary}
{memory_section}
## 协商策略
请根据以上信息，决定你的新请求量。考虑：
1. 如果资源紧张，适当让步以促成共识
2. 如果你是关键优先级，可以坚持更多
3. 平衡自身需求和整体公平性

请按以下格式回答：
【分析】（1句话分析当前形势）
【策略】（1句话说明你的协商策略）
【新请求量】（一个整数，介于 {self.need.minimum} 和 {self.current_proposal.requested_amount} 之间）
"""
        
        try:
            start_time = time.time()
            response = llm.send_message(prompt)
            duration_ms = (time.time() - start_time) * 1000
            
            if hasattr(self.model, 'metrics'):
                self.model.metrics.performance.record_llm_call(
                    duration_ms=duration_ms,
                    call_id=f"cot_negotiation_{self.component_id}_{self.model.current_round}"
                )
                self.model.metrics.reasoning.record_cot_reasoning(
                    agent_id=self.component_id,
                    decision_id=f"negotiate_{self.model.current_round}_{self.unique_id}",
                    steps=["分析", "策略", "决定"],
                    final_score=0.5
                )
            
            import re
            match = re.search(r'【新请求量】\s*(\d+)', response)
            if match:
                new_request = int(match.group(1))
                new_request = max(self.need.minimum, min(self.current_proposal.requested_amount, new_request))
            else:
                new_request = self._fallback_adjustment(scarcity)
            
            justification = response.split('【策略】')[-1].split('【')[0].strip() if '【策略】' in response else f"调整请求至 {new_request}"
            
            # 记录思考过程
            if hasattr(self.model, 'thought_logger') and self.model.thought_logger:
                self.model.thought_logger.record_thought(
                    agent_id=self.component_id,
                    agent_name=self.component_id,
                    round_num=self.model.current_round,
                    input_context=f"最低需求:{self.need.minimum}, 期望:{self.need.desired}, 稀缺度:{scarcity:.2f}",
                    memories_retrieved=[memory_context[:100]] if memory_context else [],
                    reasoning_type="cot",
                    reasoning_steps=[response[:500]],
                    decision=f"请求{new_request}单位",
                    confidence=0.5,
                    reasoning_summary=f"CoT协商: {self.current_proposal.requested_amount}->{new_request}"
                )
            
            return new_request, justification
            
        except Exception as e:
            print(f"  Agent {self.component_id} CoT 协商失败: {e}")
            return self._fallback_adjustment(scarcity), f"回退策略调整"
    
    def _tot_negotiate(self, total_requests: int, available_resources: int,
                       others_proposals: List[AllocationProposal],
                       scarcity: float) -> Tuple[int, str]:
        """
        ToT（Tree of Thought）多路径策略探索 - 优化组 A/B
        
        使用真正的 TreeOfThought 类进行多路径策略探索：
        - 生成多个协商策略分支（激进、保守、均衡）
        - 每个分支独立 LLM 评估
        - Beam Search 选择最优策略
        """
        # 初始化 ToT（如果未初始化会抛出异常）
        self._init_tot()
        
        # 增强记忆检索
        if getattr(self.model, 'use_enhanced_memory', False):
            memory_context = self._get_enhanced_memory_context(scarcity)
        else:
            memory_context = ""
        
        others_summary = "\n".join([
            f"- {p.agent_id}: 请求 {p.requested_amount}，妥协意愿 {p.willingness_to_compromise:.1f}"
            for p in others_proposals[:5]
        ]) if others_proposals else "暂无其他人的提案"
        
        # 准备 ToT 输入状态
        initial_state = {
            'question': f"如何调整资源请求量？",
            'minimum': self.need.minimum,
            'desired': self.need.desired,
            'priority': self.need.priority.value,
            'current_request': self.current_proposal.requested_amount,
            'justification': self.need.justification,
            'available_resources': available_resources,
            'total_requests': total_requests,
            'scarcity': scarcity,
            'others_summary': others_summary,
            'memory_context': memory_context
        }
        
        try:
            # 运行真正的 ToT（多次 LLM 调用）
            start_time = time.time()
            self._tot_instance.set_input(initial_state)
            best_node = self._tot_instance.run()
            result = self._tot_instance.get_output()
            duration_ms = (time.time() - start_time) * 1000
            
            # 解析 ToT 结果
            best_score = result.get('best_score', 0.5)
            reasoning_path = result.get('reasoning_path', '')
            nodes_explored = result.get('total_nodes_explored', 0)
            pruned_count = sum(1 for n in self._tot_instance.all_nodes if n.is_pruned)
            
            # 从最佳节点的状态中提取请求量
            best_state = result.get('best_state', {})
            new_request = self._extract_request_from_state(best_state, scarcity)
            
            # 确保请求量在合理范围内
            new_request = max(self.need.minimum, min(self.current_proposal.requested_amount, new_request))
            
            justification = f"ToT 多路径分析：探索 {nodes_explored} 个节点，最佳评分 {best_score:.2f}"
            
            # 记录 LLM 调用和推理指标
            if hasattr(self.model, 'metrics'):
                self.model.metrics.performance.record_llm_call(
                    duration_ms=duration_ms,
                    call_id=f"tot_negotiation_{self.component_id}_{self.model.current_round}"
                )
                self.model.metrics.reasoning.record_tot_reasoning(
                    agent_id=self.component_id,
                    decision_id=f"negotiate_{self.model.current_round}_{self.unique_id}",
                    depth=self._tot_instance.max_depth,
                    branches_explored=nodes_explored,
                    pruned_branches=pruned_count,
                    reasoning_path=reasoning_path.split('\n') if reasoning_path else [],
                    final_score=best_score
                )
            
            # 记录到记忆
            self.memory.add_short_memory(
                source=self.component_id,
                target=self.component_id,
                action="tot_negotiation",
                content=f"真ToT 协商：探索了 {nodes_explored} 个节点，最终请求 {new_request} 单位"
            )
            
            # 记录思考过程
            if hasattr(self.model, 'thought_logger') and self.model.thought_logger:
                self.model.thought_logger.record_thought(
                    agent_id=self.component_id,
                    agent_name=self.component_id,
                    round_num=self.model.current_round,
                    input_context=f"最低需求:{self.need.minimum}, 期望:{self.need.desired}, 稀缺度:{scarcity:.2f}",
                    memories_retrieved=[memory_context[:100]] if memory_context else [],
                    reasoning_type="tot_real",  # 标记为真正的 ToT
                    reasoning_steps=[
                        f"ToT 探索节点数: {nodes_explored}",
                        f"剪枝节点数: {pruned_count}",
                        f"最佳分支评分: {best_score:.2f}",
                        f"最终请求量: {new_request}"
                    ],
                    tot_branches=[
                        {"策略": "激进", "探索": True},
                        {"策略": "保守", "探索": True},
                        {"策略": "均衡", "探索": True}
                    ],
                    tot_evaluations=[{"best_score": best_score, "nodes": nodes_explored}],
                    decision=f"请求{new_request}单位",
                    confidence=best_score,
                    reasoning_summary=f"真ToT协商: {self.current_proposal.requested_amount}->{new_request}, 探索{nodes_explored}节点"
                )
            
            return new_request, justification
            
        except Exception as e:
            print(f"  Agent {self.component_id} ToT 协商失败: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_adjustment(scarcity), f"回退策略调整"
    
    def _extract_request_from_state(self, state: Dict[str, Any], scarcity: float) -> int:
        """从 ToT 最佳状态中提取请求量"""
        import re
        
        # 尝试从状态中直接获取
        if 'recommended_request' in state:
            return int(state['recommended_request'])
        
        # 尝试从 last_response 中提取
        response = state.get('last_response', '')
        
        # 尝试匹配 "建议请求量: X" 格式
        match = re.search(r'建议请求量[：:]\s*(\d+)', response)
        if match:
            return int(match.group(1))
        
        # 尝试匹配任何数字
        match = re.search(r'(\d+)\s*单位', response)
        if match:
            return int(match.group(1))
        
        # 回退：基于稀缺度计算
        return self._fallback_adjustment(scarcity)
    
    def _fallback_adjustment(self, scarcity: float) -> int:
        """回退调整策略"""
        if scarcity > 1.5:
            adjustment_factor = 0.7
        elif scarcity > 1.2:
            adjustment_factor = 0.85
        else:
            adjustment_factor = 1.0
        return max(self.need.minimum, int(self.current_proposal.requested_amount * adjustment_factor))
    
    def _get_enhanced_memory_context(self, current_scarcity: float) -> str:
        """
        获取增强记忆上下文
        
        检索相关历史协商信息，用于指导当前决策。
        """
        recent_memories = self.memory.get_recent_memories(10)
        
        if not recent_memories:
            return ""
        
        current_time = self.model.schedule.time
        relevant_memories = []
        
        for mem in recent_memories:
            content = mem.get('content', '')
            
            # 时间衰减
            mem_time = mem.get('timestamp', 0)
            time_diff = current_time - mem_time
            time_decay = 1.0 / (1.0 + 0.2 * time_diff)
            
            # 相关性：协商相关的记忆更重要
            relevance = 1.0
            if '协商' in content or '资源' in content or '分配' in content:
                relevance = 1.5
            if '紧张' in content and current_scarcity > 1.2:
                relevance = 2.0
            
            score = time_decay * relevance
            if score > 0.3:
                relevant_memories.append((score, content))
        
        # 按分数排序，取前 5 个
        relevant_memories.sort(key=lambda x: x[0], reverse=True)
        top_memories = [content for _, content in relevant_memories[:5]]
        
        return "\n".join([f"- {mem}" for mem in top_memories]) if top_memories else ""
    
    def _dynamic_reflect_on_proposal(self, new_request: int, scarcity: float):
        """
        动态反思提案
        
        在协商过程中反思策略是否合适。
        """
        # 检查是否需要反思：请求变化超过 30% 或资源非常紧张
        if self.current_proposal is None:
            return
        
        original_request = self.current_proposal.requested_amount
        change_ratio = abs(new_request - original_request) / original_request if original_request > 0 else 0
        
        needs_reflection = change_ratio > 0.3 or scarcity > 2.0
        
        if not needs_reflection:
            return
        
        llm = getattr(self.model, 'llm_interface', None)
        if llm is None:
            return
        
        prompt = f"""你刚才做出了一个协商决策，请反思这个决策是否合理。

## 决策情况
- 原请求量: {original_request}
- 新请求量: {new_request}
- 变化幅度: {change_ratio:.1%}
- 资源紧张度: {scarcity:.2f}

## 你的需求
- 最低需求: {self.need.minimum}
- 期望需求: {self.need.desired}
- 优先级: {self.need.priority.value}

## 请反思
这个让步幅度是否合适？是否应该调整？

【反思】（1句话）
【调整建议】保持/微调增加/微调减少
【建议请求量】（整数）
"""
        
        try:
            response = llm.send_message(prompt)
            
            import re
            match = re.search(r'【建议请求量】\s*(\d+)', response)
            if match:
                suggested = int(match.group(1))
                # 只允许小幅调整（±10%）
                min_allowed = int(new_request * 0.9)
                max_allowed = int(new_request * 1.1)
                adjusted = max(min_allowed, min(max_allowed, suggested))
                
                if adjusted != new_request:
                    self.current_proposal = AllocationProposal(
                        agent_id=self.component_id,
                        requested_amount=adjusted,
                        justification=f"反思后调整至 {adjusted}",
                        willingness_to_compromise=self.current_proposal.willingness_to_compromise,
                        alternative_amounts=[
                            int(adjusted * 0.9),
                            int(adjusted * 0.8),
                            self.need.minimum
                        ]
                    )
                    
                    self.memory.add_short_memory(
                        source=self.component_id,
                        target=self.component_id,
                        action="dynamic_reflection",
                        content=f"动态反思：将请求从 {new_request} 调整为 {adjusted}"
                    )
        except Exception:
            pass  # 反思失败，保持原决策
    
    def receive_allocation(self, amount: int):
        """
        接收分配结果
        
        Args:
            amount: 分配的资源量
        """
        self.allocated_amount = amount
        
        # 计算满意度（添加除零保护）
        if amount >= self.need.desired:
            self.satisfaction = 1.0
        elif amount >= self.need.minimum:
            # 防止 desired == minimum 导致的除零错误
            range_diff = self.need.desired - self.need.minimum
            if range_diff > 0:
                self.satisfaction = 0.5 + 0.5 * (amount - self.need.minimum) / range_diff
            else:
                self.satisfaction = 0.75  # desired == minimum 时，满足最低需求即给 0.75
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
        # 记录本轮思考过程
        self._log_round_thought()
    
    def _log_round_thought(self):
        """
        记录每轮的协商状态和思考过程
        
        在每轮结束时调用，记录智能体的完整状态
        """
        if not hasattr(self.model, 'thought_logger') or not self.model.thought_logger:
            return
        
        # 获取最近记忆
        recent_memories = []
        try:
            if hasattr(self.memory, 'get_recent_memories'):
                memories = self.memory.get_recent_memories(3)
                if memories:
                    for m in memories:
                        if isinstance(m, dict):
                            recent_memories.append(m.get('content', str(m))[:100])
                        else:
                            recent_memories.append(str(m)[:100])
        except:
            pass
        
        # 构建协商历史摘要
        history_summary = []
        for h in self.negotiation_history[-3:]:  # 最近3轮
            history_summary.append(
                f"Round {h.get('round', '?')}: 请求{h.get('request', '?')}单位, "
                f"稀缺度{h.get('scarcity', 0):.2f}"
            )
        
        # 当前状态
        current_request = self.current_proposal.requested_amount if self.current_proposal else 0
        compromise = self.current_proposal.willingness_to_compromise if self.current_proposal else 0
        
        # 记录思考过程
        self.model.thought_logger.record_thought(
            agent_id=self.component_id,
            agent_name=f"资源智能体_{self.component_id}",
            round_num=self.model.current_round,
            input_context=(
                f"优先级: {self.need.priority.value}, "
                f"最低需求: {self.need.minimum}, "
                f"期望需求: {self.need.desired}, "
                f"当前请求: {current_request}"
            ),
            memories_retrieved=recent_memories,
            reasoning_type="tot" if getattr(self.model, 'use_tot', False) else "cot",
            reasoning_steps=history_summary if history_summary else ["初始化阶段"],
            decision=f"请求 {current_request} 单位资源",
            confidence=1 - compromise,  # 妥协意愿越低，置信度越高
            reasoning_summary=(
                f"分配: {self.allocated_amount}单位, "
                f"满意度: {self.satisfaction:.2f}, "
                f"妥协意愿: {compromise:.2f}"
            )
        )
    
    def _log_final_allocation(self):
        """
        记录最终分配结果的思考日志
        
        在协商结束、资源分配完成后调用
        """
        if not hasattr(self.model, 'thought_logger') or not self.model.thought_logger:
            return
        
        # 计算需求满足率
        if self.need.desired > 0:
            fulfillment_rate = self.allocated_amount / self.need.desired
        else:
            fulfillment_rate = 1.0
        
        # 判断分配结果
        if self.allocated_amount >= self.need.desired:
            result_status = "完全满足"
        elif self.allocated_amount >= self.need.minimum:
            result_status = "基本满足"
        else:
            result_status = "未达最低需求"
        
        # 协商过程回顾
        negotiation_summary = []
        if self.negotiation_history:
            first_request = self.negotiation_history[0].get('request', 0)
            last_request = self.negotiation_history[-1].get('request', 0)
            negotiation_summary.append(f"初始请求: {first_request} 单位")
            negotiation_summary.append(f"最终请求: {last_request} 单位")
            negotiation_summary.append(f"让步幅度: {first_request - last_request} 单位")
            negotiation_summary.append(f"协商轮次: {len(self.negotiation_history)}")
        
        self.model.thought_logger.record_thought(
            agent_id=self.component_id,
            agent_name=f"资源智能体_{self.component_id}",
            round_num=self.model.current_round,
            input_context=(
                f"最终分配阶段 | "
                f"优先级: {self.need.priority.value}, "
                f"期望: {self.need.desired}, "
                f"最低: {self.need.minimum}"
            ),
            memories_retrieved=[],
            reasoning_type="final_allocation",
            reasoning_steps=negotiation_summary if negotiation_summary else ["直接分配"],
            decision=f"获得 {self.allocated_amount} 单位资源",
            confidence=self.satisfaction,
            reasoning_summary=(
                f"结果: {result_status} | "
                f"满足率: {fulfillment_rate:.1%} | "
                f"满意度: {self.satisfaction:.2f}"
            )
        )


class ResourceAllocationModel(mesa.Model):
    """
    资源分配模型
    
    管理资源协商和分配过程。
    """
    
    def __init__(self, num_agents: int = 50,
                 total_resources: int = 400,
                 max_negotiation_rounds: int = 10,
                 use_collaborative: bool = True,
                 use_tot: bool = False,
                 use_enhanced_memory: bool = False,
                 use_dynamic_reflection: bool = False,
                 llm_interface=None,
                 thought_logger=None,
                 experiment_id: str = None):
        """
        初始化资源分配模型
        
        Args:
            num_agents: 智能体数量
            total_resources: 总资源量
            max_negotiation_rounds: 最大协商轮次
            use_collaborative: 是否使用协同决策模块
            use_tot: 是否使用 ToT 多层次推理
            use_enhanced_memory: 是否使用增强记忆检索
            use_dynamic_reflection: 是否使用动态反思
            llm_interface: LLM 接口（智能体实验必须提供）
            thought_logger: 思考过程日志记录器
            experiment_id: 实验ID
        """
        super().__init__()
        
        if llm_interface is None:
            raise RuntimeError(
                "资源分配实验需要 LLM 接口！\n"
                "使用方法：\n"
                "  from casevo import create_default_llm\n"
                "  llm = create_default_llm()\n"
                "  model = ResourceAllocationModel(..., llm_interface=llm)"
            )
        
        self.num_agents = num_agents
        self.total_resources = total_resources
        self.max_rounds = max_negotiation_rounds
        self.use_collaborative = use_collaborative
        self.use_tot = use_tot
        self.use_enhanced_memory = use_enhanced_memory
        self.use_dynamic_reflection = use_dynamic_reflection
        self.llm_interface = llm_interface
        self.thought_logger = thought_logger
        self.experiment_id = experiment_id
        self.context = "资源分配协商"
        
        # 反思阈值（Proposal 要求 0.6）
        self.reflection_threshold = 0.6
        
        # 创建调度器
        self.schedule = mesa.time.BaseScheduler(self)
        
        # 初始化记忆工厂
        reflect_prompt = "请根据以下记忆进行反思，总结关键信息和协商策略。"
        self.memory_factory = AdvancedMemoryFactory(
            tar_llm=self.llm_interface,
            memory_num=5,
            prompt=reflect_prompt,
            model=self
        )
        
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
        
        # 评估指标（符合 Proposal 四维度要求）
        self.metrics = create_experiment_metrics()
        self.metrics.start_experiment()
    
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
            
            # 根据优先级设置需求范围（符合 Proposal：15-30 单位）
            if priority == AgentPriority.CRITICAL:
                min_need = random.randint(22, 28)
                desired = random.randint(28, 30)
            elif priority == AgentPriority.HIGH:
                min_need = random.randint(20, 25)
                desired = random.randint(25, 30)
            elif priority == AgentPriority.NORMAL:
                min_need = random.randint(18, 22)
                desired = random.randint(22, 28)
            else:  # LOW
                min_need = random.randint(15, 18)
                desired = random.randint(18, 25)
            
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
        
        # 触发每个智能体的 step() 方法，记录思考过程
        for agent in self.schedule.agents:
            if isinstance(agent, ResourceAgent):
                agent.step()
        
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
        
        # 分发分配结果并记录最终状态
        for agent in self.schedule.agents:
            if isinstance(agent, ResourceAgent):
                amount = allocations.get(agent.component_id, 0)
                agent.receive_allocation(amount)
                # 记录最终分配结果的思考日志
                agent._log_final_allocation()
        
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
        
        # 按优先级分析满意度
        priority_satisfaction = {}
        for agent in self.schedule.agents:
            if isinstance(agent, ResourceAgent):
                priority = agent.need.priority.value
                if priority not in priority_satisfaction:
                    priority_satisfaction[priority] = []
                priority_satisfaction[priority].append(agent.satisfaction)
        
        priority_avg_satisfaction = {
            p: statistics.mean(s) if s else 0 
            for p, s in priority_satisfaction.items()
        }
        
        return {
            'gini_coefficient': gini,
            'average_satisfaction': avg_satisfaction,
            'minimum_satisfaction': min_satisfaction,
            'maximum_satisfaction': max(satisfactions) if satisfactions else 0,
            'satisfaction_std': statistics.stdev(satisfactions) if len(satisfactions) > 1 else 0,
            'average_fulfillment_rate': avg_fulfillment,
            'allocation_variance': allocation_variance,
            'total_allocated': sum(allocations),
            'utilization_rate': sum(allocations) / self.total_resources,
            # 新增：按优先级分析
            'priority_satisfaction': priority_avg_satisfaction,
            # 新增：收敛信息
            'convergence_round': self.current_round if self.converged else -1,
            'convergence_speed': 1.0 - (self.current_round / self.max_rounds) if self.converged else 0
        }
    
    def _calculate_gini(self, values: List[float]) -> float:
        """计算基尼系数"""
        if not values or sum(values) == 0:
            return 0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = sum((i + 1) * v for i, v in enumerate(sorted_values))
        
        return (2 * cumsum) / (n * sum(sorted_values)) - (n + 1) / n
    
    def run_simulation(self, verbose: bool = True) -> Dict[str, Any]:
        """运行完整模拟"""
        # 初始提案
        for agent in self.schedule.agents:
            if isinstance(agent, ResourceAgent):
                agent.create_initial_proposal()
        
        # 协商过程
        while self.current_round < self.max_rounds and not self.converged:
            if verbose:
                print(f"      协商 {self.current_round+1}/{self.max_rounds}...", end='\r')
            self.negotiate_round()
        
        # 最终分配
        final_allocations = self.allocate_resources()
        
        # 计算公平性指标
        fairness = self.calculate_fairness_metrics()
        
        if verbose:
            print(f"      完成: {self.current_round}轮, 基尼={fairness.get('gini_coefficient', 0):.3f}, 满意度={fairness.get('average_satisfaction', 0):.2f}    ")
        
        # 获取评估指标
        metrics_report = self.metrics.get_full_report()
        metrics_summary = self.metrics.get_summary()
        
        return {
            'negotiation_rounds': self.current_round,
            'converged': self.converged,
            'allocation_history': self.allocation_history,
            'final_allocations': final_allocations,
            'fairness_metrics': fairness,
            # Proposal 要求的四维度评估指标
            'evaluation_metrics': metrics_report,
            'metrics_summary': metrics_summary
        }


def run_resource_experiment(config: Dict[str, Any] = None, llm_interface=None,
                            memory_path: str = None, thought_logger=None,
                            experiment_id: str = None, **kwargs) -> Dict[str, Any]:
    """
    运行资源分配实验
    
    Args:
        config: 实验配置
        llm_interface: LLM 接口实例（必须提供）
        memory_path: 持久化记忆路径（暂未使用）
        thought_logger: 思考日志记录器（暂未使用）
        experiment_id: 实验ID（暂未使用）
        
    Returns:
        实验结果
    """
    if llm_interface is None:
        raise RuntimeError(
            "资源分配实验需要 LLM 接口！\n"
            "这是一个智能体实验，必须使用 LLM 进行协商推理。\n\n"
            "使用方法：\n"
            "  from casevo import create_default_llm\n"
            "  llm = create_default_llm()\n"
            "  results = run_resource_experiment(config, llm_interface=llm)"
        )
    
    if config is None:
        config = {
            'num_agents': 50,
            'total_resources': 1000,
            'max_rounds': 10,
            'use_collaborative': True
        }
    
    model = ResourceAllocationModel(
        num_agents=config.get('num_agents', 50),
        total_resources=config.get('total_resources', 400),
        max_negotiation_rounds=config.get('max_rounds', 10),
        use_collaborative=config.get('use_collaborative', True),
        use_tot=config.get('use_tot', False),
        use_enhanced_memory=config.get('use_enhanced_memory', False),
        thought_logger=thought_logger,
        experiment_id=experiment_id,
        use_dynamic_reflection=config.get('use_dynamic_reflection', False),
        llm_interface=llm_interface
    )
    
    results = model.run_simulation()
    results['config'] = config
    results['tot_enabled'] = config.get('use_tot', False)
    results['enhanced_memory_enabled'] = config.get('use_enhanced_memory', False)
    results['dynamic_reflection_enabled'] = config.get('use_dynamic_reflection', False)
    results['collaborative_enabled'] = config.get('use_collaborative', True)
    
    return results


if __name__ == "__main__":
    # 创建 LLM 接口
    print("初始化 LLM 接口...")
    llm = create_default_llm()
    
    # 运行基线实验
    print("运行基线实验...")
    baseline_results = run_resource_experiment({
        'use_collaborative': False,
        'num_agents': 50,
        'total_resources': 1000
    }, llm_interface=llm)
    print(f"基线结果 - 协商轮次: {baseline_results['negotiation_rounds']}")
    print(f"基线公平性: {baseline_results['fairness_metrics']}")
    
    # 运行优化实验
    print("\n运行优化实验（协同决策）...")
    optimized_results = run_resource_experiment({
        'use_collaborative': True,
        'num_agents': 50,
        'total_resources': 1000
    }, llm_interface=llm)
    print(f"优化结果 - 协商轮次: {optimized_results['negotiation_rounds']}")
    print(f"优化公平性: {optimized_results['fairness_metrics']}")
    
    # 保存结果
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'resource', 'baseline.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': baseline_results,
            'optimized': optimized_results
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n结果已保存到 experiments/results/resource/baseline.json")

