"""
信息传播实验场景（LLM 智能体版本）

研究虚假信息在社交网络中的传播动力学：
- 200 个节点的无标度网络
- LLM 驱动的信息可信度评估
- 三组对照实验：CoT 基线 / ToT 优化 / 全部优化
- 传播抑制策略评估

注意：这是智能体实验，需要 LLM 接口才能运行。
"""

import mesa
import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json
import os
import sys
import statistics

# 添加项目路径
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from casevo import (
    AgentBase,
    DecisionEvaluator, 
    DecisionRecord,
    AdvancedMemoryFactory,
    TreeOfThought, ToTStep, EvaluatorStep, SearchStrategy,
    PromptFactory
)

# Prompt 模板目录
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

from experiments.utils.metrics import (
    ExperimentMetrics,
    SocialEffectMetrics,
    create_experiment_metrics
)
import time


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


# ============================================================
# 信息内容模板（关键：内容本身不暴露真假，需要推理判断）
# ============================================================

# 真实信息模板：有权威来源支持的事实
TRUE_INFO_TEMPLATES = [
    "世界卫生组织最新报告显示，全球疫苗接种覆盖率已达到65%，有效降低了重症率",
    "根据中国气象局数据，今年全国平均气温较往年同期上升0.8摄氏度",
    "NASA确认詹姆斯·韦伯望远镜成功拍摄到距离地球134亿光年的星系图像",
    "教育部统计：2024年全国高考报名人数达到1342万人，创历史新高",
    "中国科学院研究团队在《自然》期刊发表论文，证实量子计算机实现算力突破",
    "国家统计局公布：2024年第三季度GDP同比增长4.9%",
    "世界银行报告指出，东亚地区经济复苏速度领先全球",
    "联合国粮农组织数据显示，全球粮食产量连续第三年增长",
    "中国疾控中心监测数据表明，流感疫苗接种可降低60%感染风险",
    "国际能源署统计，可再生能源发电量首次超过煤电",
    "交通运输部数据：高铁网络总里程突破4.5万公里",
    "中国人民银行公告：数字人民币试点城市已扩展至26个",
]

# 虚假信息模板：常见谣言、伪科学、未证实的说法
FALSE_INFO_TEMPLATES = [
    "网传消息：下周全国将实施为期一个月的交通管制，建议囤积物资",
    "专家警告：5G信号塔辐射会干扰人体免疫系统，导致癌症发病率上升",
    "研究发现：每天饮用苏打水可以有效预防新冠病毒感染",
    "内部消息：某知名银行即将破产，建议立即转移存款",
    "科学家证实：地球磁极将在2025年发生翻转，届时通讯系统将全面瘫痪",
    "医学突破：某草药配方可在7天内彻底治愈糖尿病，已有数千人受益",
    "紧急通知：饮用矿泉水会导致肾结石，专家建议改喝纯净水",
    "震惊发现：手机充电时使用会导致电池爆炸，已有多起伤亡事故",
    "权威发布：某品牌食用油含有致癌物质，正在全国范围内召回",
    "独家爆料：某城市自来水检测出重金属超标，居民健康受到威胁",
    "最新研究：睡前玩手机可以改善睡眠质量，专家推荐每晚使用2小时",
    "内幕消息：某热门股票即将暴涨10倍，现在买入稳赚不赔",
]


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
        
        # 统计
        self.correct_judgments = 0
        self.incorrect_judgments = 0
        
        # ToT 组件（仅在有 LLM 接口时初始化）
        self._tot_instance: Optional[TreeOfThought] = None
        self._tot_initialized = False
    
    def _generate_description(self, agent_type: AgentType) -> str:
        """生成智能体描述"""
        type_descriptions = {
            AgentType.NORMAL: "普通社交媒体用户，具有一般的信息辨别能力",
            AgentType.SKEPTIC: "怀疑型用户，对信息持谨慎态度，倾向于验证",
            AgentType.GULLIBLE: "易信型用户，较容易相信收到的信息",
            AgentType.INFLUENCER: "影响力用户，有大量关注者，传播能力强"
        }
        return type_descriptions.get(agent_type, "社交媒体用户")
    
    def _init_tot(self):
        """
        初始化 ToT 组件（需要 LLM 接口）
        
        使用真正的 TreeOfThought 类进行多路径推理：
        - 生成多个评估分支（来源、逻辑、一致性）
        - 每个分支独立评估
        - Beam Search 选择最优路径
        """
        if self._tot_initialized:
            return self._tot_instance is not None
        
        self._tot_initialized = True
        
        # 检查是否有 LLM 接口
        llm = getattr(self.model, 'llm_interface', None)
        if llm is None:
            raise RuntimeError(
                "ToT 评估需要 LLM 接口！请在创建模型时传入 llm_interface 参数。"
            )
        
        # 创建 PromptFactory
        prompt_factory = PromptFactory(PROMPTS_DIR, llm)
        
        # 创建 ToT 步骤（生成3个分析角度）
        thought_step = ToTStep(
            step_id="info_evaluate_branch",
            tar_prompt=prompt_factory.get_template("info_tot_generate.j2"),
            num_branches=3  # 来源分析、逻辑分析、一致性分析
        )
        
        # 创建评估步骤
        evaluator_step = EvaluatorStep(
            step_id="info_evaluate_score",
            tar_prompt=prompt_factory.get_template("info_tot_evaluate.j2"),
            score_range=(0.0, 1.0)
        )
        
        # 创建 ToT 实例
        self._tot_instance = TreeOfThought(
            agent=self,
            thought_step=thought_step,
            evaluator_step=evaluator_step,
            max_depth=2,            # 信息评估不需要太深
            beam_width=3,           # 保留3个最佳分支
            pruning_threshold=0.3,  # 低于0.3的分支剪枝
            search_strategy=SearchStrategy.BEAM
        )
        return True
    
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
        评估信息可信度（LLM 驱动）
        
        支持三组对照设计（符合 Proposal 要求）：
        - 基线组（CoT）：使用 LLM 单次推理评估
        - 优化组 A（ToT）：使用 LLM 多路径推理评估
        - 优化组 B（全部优化）：LLM + 增强记忆 + 动态反思
        
        Args:
            info: 待评估的信息
            
        Returns:
            (是否相信, 置信度, 评估理由)
            
        Raises:
            RuntimeError: 如果未提供 LLM 接口
        """
        llm = getattr(self.model, 'llm_interface', None)
        if llm is None:
            raise RuntimeError(
                "信息传播实验需要 LLM 接口！\n"
                "这是一个智能体实验，必须使用 LLM 进行信息评估。\n\n"
                "使用方法：\n"
                "  from casevo import create_default_llm\n"
                "  llm = create_default_llm()\n"
                "  results = run_info_spreading_experiment(config, llm_interface=llm)"
            )
        
        use_tot = getattr(self.model, 'use_tot', False)
        if use_tot:
            # ToT 多路径推理（优化组 A/B）
            return self._tot_evaluate_information(info)
        else:
            # CoT 单次推理（基线组）
            return self._cot_evaluate_information(info)
    
    def _cot_evaluate_information(self, info: Information) -> Tuple[bool, float, str]:
        """
        CoT（Chain of Thought）单次推理评估 - 基线组
        
        使用 LLM 进行简单的单次推理判断信息可信度。
        """
        llm = self.model.llm_interface
        
        prompt = f"""你是一位社交网络用户，请快速判断以下信息是否可信。

## 你的特征
- 用户类型: {self.agent_type.value}
- 批判性思维能力: {self.critical_thinking:.2f}

## 待评估信息
- 内容: {info.content}
- 来源可信度: {info.source_credibility:.2f}

## 请直接判断
【判断】相信 / 不相信
【置信度】0.0-1.0
【理由】（一句话）
"""
        
        try:
            # 追踪 LLM 调用性能
            start_time = time.time()
            response = llm.send_message(prompt)
            duration_ms = (time.time() - start_time) * 1000
            
            # 记录 LLM 调用
            if hasattr(self.model, 'metrics'):
                self.model.metrics.performance.record_llm_call(
                    duration_ms=duration_ms,
                    call_id=f"cot_info_eval_{self.unique_id}_{info.info_id}"
                )
                # 记录推理指标
                self.model.metrics.reasoning.record_cot_reasoning(
                    agent_id=self.component_id,
                    decision_id=f"info_{info.info_id}_{self.unique_id}",
                    steps=["判断", "置信度"],
                    final_score=0.5
                )
            
            raw_result = self._parse_evaluation_response(response)
            # 应用批判性思维调整
            result = self._apply_critical_thinking(*raw_result)
            
            # 记录思考过程
            if hasattr(self.model, 'thought_logger') and self.model.thought_logger:
                self.model.thought_logger.record_thought(
                    agent_id=self.component_id,
                    agent_name=self.component_id,
                    round_num=self.model.schedule.time,
                    input_context=f"信息:{info.content[:80]}..., 来源可信度:{info.source_credibility:.2f}",
                    memories_retrieved=[],
                    reasoning_type="cot",
                    reasoning_steps=[
                        f"LLM原始判断: {'相信' if raw_result[0] else '不相信'}, 置信度{raw_result[1]:.2f}",
                        f"批判性思维: {self.critical_thinking:.2f}",
                        f"最终判断: {'相信' if result[0] else '不相信'}, 置信度{result[1]:.2f}",
                        response[:200]
                    ],
                    decision="相信" if result[0] else "不相信",
                    confidence=result[1],
                    reasoning_summary=f"CoT评估: {'相信' if result[0] else '不相信'}, 置信度{result[1]:.2f}, 批判性:{self.critical_thinking:.2f}"
                )
            
            return result
        except Exception as e:
            print(f"  CoT 评估失败: {e}")
            # 返回中性结果
            return False, 0.5, "评估失败，保持怀疑"
    
    def _tot_evaluate_information(self, info: Information) -> Tuple[bool, float, str]:
        """
        ToT（Tree of Thought）多路径推理评估 - 优化组 A/B
        
        使用真正的 TreeOfThought 类进行多路径推理：
        - 生成多个评估分支（来源、逻辑、一致性）
        - 每个分支独立 LLM 评估
        - Beam Search 选择最优路径
        - 支持增强记忆和动态反思（优化组 B）
        """
        # 初始化 ToT（如果未初始化会抛出异常）
        self._init_tot()
        
        # 获取相关记忆
        if getattr(self.model, 'use_enhanced_memory', False):
            recent_memories = self._get_enhanced_memories(info.content, 5)
        else:
            recent_memories = self.memory.get_recent_memories(5)
        
        memory_context = "\n".join([
            f"- {m.get('content', '')}" for m in recent_memories
        ]) if recent_memories else "无相关历史信息"
        
        # 准备 ToT 输入状态
        initial_state = {
            'question': f"该信息是否可信？",
            'info_content': info.content,
            'source_credibility': info.source_credibility,
            'spread_count': info.spread_count,
            'memory_context': memory_context,
            'agent_type': self.agent_type.value,
            'critical_thinking': self.critical_thinking
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
            # 计算剪枝数：从 all_nodes 中统计 is_pruned=True 的节点
            pruned_count = sum(1 for n in self._tot_instance.all_nodes if n.is_pruned)
            
            # 根据 best_score 决定是否相信
            # score > 0.5 倾向于相信，< 0.5 倾向于不相信
            raw_believed = best_score > 0.5
            raw_confidence = abs(best_score - 0.5) * 2  # 转换为 0-1 的置信度
            raw_reasoning = reasoning_path if reasoning_path else f"ToT 评分: {best_score:.2f}"
            
            # 应用批判性思维调整
            believed, confidence, reasoning = self._apply_critical_thinking(
                raw_believed, raw_confidence, raw_reasoning
            )
            
            # 动态反思：边界情况时重新评估
            reflection_triggered = False
            if getattr(self.model, 'use_dynamic_reflection', False):
                if 0.4 < confidence < 0.6:  # 边界情况
                    reflection_triggered = True
                    confidence, believed = self._dynamic_reflect(info, confidence)
            
            # 记录 LLM 调用和推理指标
            if hasattr(self.model, 'metrics'):
                self.model.metrics.performance.record_llm_call(
                    duration_ms=duration_ms,
                    call_id=f"tot_info_eval_{self.unique_id}_{info.info_id}"
                )
                self.model.metrics.reasoning.record_tot_reasoning(
                    agent_id=self.component_id,
                    decision_id=f"info_{info.info_id}_{self.unique_id}",
                    depth=self._tot_instance.max_depth,
                    branches_explored=nodes_explored,
                    pruned_branches=pruned_count,
                    reasoning_path=reasoning_path.split('\n') if reasoning_path else [],
                    final_score=best_score
                )
            
            # 记录思考过程
            if hasattr(self.model, 'thought_logger') and self.model.thought_logger:
                memory_contents = [m.get('content', str(m))[:50] for m in recent_memories] if recent_memories else []
                self.model.thought_logger.record_thought(
                    agent_id=self.component_id,
                    agent_name=self.component_id,
                    round_num=self.model.schedule.time,
                    input_context=f"信息:{info.content[:80]}..., 来源可信度:{info.source_credibility:.2f}",
                    memories_retrieved=memory_contents[:3],
                    reasoning_type="tot_real",  # 标记为真正的 ToT
                    reasoning_steps=[
                        f"ToT 探索节点数: {nodes_explored}",
                        f"剪枝节点数: {pruned_count}",
                        f"最佳分支评分: {best_score:.2f}",
                        f"批判性思维调整: {self.critical_thinking:.2f}",
                        f"最终判断: {'相信' if believed else '不相信'}, 置信度{confidence:.2f}"
                    ],
                    tot_branches=[
                        {"角度": "来源分析", "探索": True},
                        {"角度": "逻辑分析", "探索": True},
                        {"角度": "一致性分析", "探索": True}
                    ],
                    tot_evaluations=[{"best_score": best_score, "nodes": nodes_explored}],
                    reflection_triggered=reflection_triggered,
                    decision="相信" if believed else "不相信",
                    confidence=confidence,
                    reasoning_summary=f"真ToT评估: {'相信' if believed else '不相信'}, 置信度{confidence:.2f}, 探索{nodes_explored}节点"
                )
            
            return believed, confidence, reasoning
        except Exception as e:
            print(f"  ToT 评估失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回中性结果
            return False, 0.5, f"ToT评估失败: {str(e)}"
    
    def _parse_evaluation_response(self, response: str) -> Tuple[bool, float, str]:
        """解析 LLM 评估响应"""
        import re
        
        # 解析判断
        believed = False
        if '【判断】' in response:
            judgment_part = response.split('【判断】')[1].split('\n')[0]
            believed = '相信' in judgment_part and '不相信' not in judgment_part
        
        # 解析置信度
        match = re.search(r'【置信度】\s*([0-9.]+)', response)
        confidence = float(match.group(1)) if match else (0.7 if believed else 0.3)
        confidence = max(0.0, min(1.0, confidence))
        
        # 解析理由
        match = re.search(r'【理由】\s*(.+?)(?:\n|$)', response)
        reasoning = match.group(1) if match else "LLM 评估完成"
        
        return believed, confidence, reasoning
    
    def _apply_critical_thinking(self, believed: bool, confidence: float, 
                                  reasoning: str) -> Tuple[bool, float, str]:
        """
        应用批判性思维调整
        
        关键设计：critical_thinking 参数实际影响决策
        - 高批判性思维 → 更高的怀疑门槛
        - 低批判性思维 → 更容易相信
        
        Args:
            believed: LLM 初始判断
            confidence: LLM 初始置信度
            reasoning: LLM 推理理由
            
        Returns:
            调整后的 (believed, confidence, reasoning)
        """
        # 计算怀疑门槛：critical_thinking 越高，门槛越高
        # skeptic (0.7-0.95): 门槛 0.6-0.75
        # gullible (0.2-0.4): 门槛 0.25-0.35
        # normal (0.4-0.6): 门槛 0.4-0.5
        belief_threshold = 0.3 + self.critical_thinking * 0.5
        
        # 如果 LLM 说相信，但置信度低于门槛，则变为不相信
        if believed and confidence < belief_threshold:
            believed = False
            reasoning = f"[批判性思维调整] 置信度{confidence:.2f}低于门槛{belief_threshold:.2f}，保持怀疑。原因：{reasoning}"
        
        # 如果 LLM 说不相信，但置信度很高且批判性思维低，可能改变主意
        elif not believed and confidence > 0.8 and self.critical_thinking < 0.4:
            # 易信者可能被高置信度说服
            if random.random() < (0.4 - self.critical_thinking):
                believed = True
                reasoning = f"[易信倾向] 尽管有疑虑，但高置信度{confidence:.2f}使其相信。原因：{reasoning}"
        
        # 根据批判性思维调整置信度
        # 高批判性思维者对自己的判断更自信
        adjusted_confidence = confidence * (0.7 + self.critical_thinking * 0.3)
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        return believed, adjusted_confidence, reasoning
    
    def _get_memory_bonus(self, info: Information) -> float:
        """获取记忆加成（增强记忆检索）"""
        recent_memories = self.memory.get_recent_memories(10)
        
        if not recent_memories:
            return 0
        
        bonus = 0
        info_keywords = set(info.content.lower().split())
        
        for mem in recent_memories:
            content = mem.get('content', '').lower()
            mem_keywords = set(content.split())
            overlap = len(info_keywords & mem_keywords)
            
            if overlap > 2:
                # 找到相关历史记忆
                if '可信' in content or '真实' in content:
                    bonus += 0.05
                elif '虚假' in content or '不可信' in content:
                    bonus -= 0.05
        
        return max(-0.2, min(0.2, bonus))
    
    def _get_enhanced_memories(self, context: str, top_k: int = 5) -> list:
        """增强记忆检索（上下文感知+时间衰减）"""
        all_memories = self.memory.get_recent_memories(20)
        
        if not all_memories:
            return []
        
        current_time = self.model.schedule.time
        scored_memories = []
        
        for mem in all_memories:
            score = 1.0
            
            # 时间衰减
            mem_time = mem.get('timestamp', 0)
            time_diff = current_time - mem_time
            time_decay = 1.0 / (1.0 + 0.1 * time_diff)
            score *= time_decay
            
            # 上下文相关性
            content = mem.get('content', '').lower()
            context_words = set(context.lower().split())
            content_words = set(content.split())
            overlap = len(context_words & content_words)
            score *= (1.0 + 0.2 * overlap)
            
            scored_memories.append((score, mem))
        
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored_memories[:top_k]]
    
    def _dynamic_reflect(self, info: Information, current_score: float) -> Tuple[float, bool]:
        """动态反思：边界情况时重新评估"""
        llm = getattr(self.model, 'llm_interface', None)
        
        if llm is None:
            # 没有 LLM，使用随机微调
            noise = random.uniform(-0.1, 0.1)
            new_score = max(0, min(1, current_score + noise))
            threshold = 0.5 if self.agent_type == AgentType.NORMAL else (0.7 if self.agent_type == AgentType.SKEPTIC else 0.3)
            return new_score, new_score > threshold
        
        prompt = f"""你之前评估了一条信息，但结果不确定（置信度 {current_score:.2f}）。

## 信息内容
{info.content}

## 请再次思考
仔细考虑这条信息是否可信，给出最终判断。

【最终判断】相信 / 不相信
【调整后置信度】0.0-1.0
"""
        
        try:
            response = llm.send_message(prompt)
            
            import re
            believed = '相信' in response and '不相信' not in response.split('【最终判断】')[1].split('\n')[0]
            
            match = re.search(r'【调整后置信度】\s*([0-9.]+)', response)
            new_score = float(match.group(1)) if match else current_score
            
            return new_score, believed
        except Exception:
            return current_score, current_score > 0.5
    
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
            neighbor = self.model.agents_dict.get(neighbor_id)
            if neighbor and isinstance(neighbor, InfoSpreadingAgent):
                # 传播概率受边权重影响（安全获取边权重）
                try:
                    edge_weight = self.model.network[self.unique_id][neighbor_id].get('weight', 0.5)
                except KeyError:
                    edge_weight = 0.5
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
        # 记录本轮思考过程
        self._log_round_thought()
    
    def _log_round_thought(self):
        """
        记录每轮的信息处理状态和思考过程
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
        
        # 构建信念状态摘要
        belief_summary = []
        true_beliefs = sum(1 for b in self.believed_info.values() if b.believed and b.information.true_type.value == "true")
        false_beliefs = sum(1 for b in self.believed_info.values() if b.believed and b.information.true_type.value == "false")
        total_beliefs = len(self.believed_info)
        
        belief_summary.append(f"持有 {total_beliefs} 条信息")
        belief_summary.append(f"相信真信息: {true_beliefs} 条")
        belief_summary.append(f"相信假信息: {false_beliefs} 条")
        belief_summary.append(f"待处理: {len(self.received_info)} 条")
        
        # 判断准确率
        accuracy = self.get_judgment_accuracy()
        
        self.model.thought_logger.record_thought(
            agent_id=self.component_id,
            agent_name=f"信息智能体_{self.component_id}",
            round_num=self.model.schedule.time,
            input_context=(
                f"类型: {self.agent_type.value}, "
                f"批判性: {self.critical_thinking:.2f}, "
                f"传播倾向: {self.spread_tendency:.2f}"
            ),
            memories_retrieved=recent_memories,
            reasoning_type="tot" if getattr(self.model, 'use_tot', False) else "cot",
            reasoning_steps=belief_summary,
            decision=f"当前判断准确率: {accuracy:.1%}",
            confidence=self.critical_thinking,
            reasoning_summary=(
                f"真信息: {true_beliefs}, "
                f"假信息: {false_beliefs}, "
                f"准确率: {accuracy:.1%}"
            )
        )


class InfoSpreadingModel(mesa.Model):
    """
    信息传播模型
    
    管理信息在社交网络中的传播过程。
    """
    
    def __init__(self, num_agents: int = 200,
                 initial_infected: float = 0.1,
                 false_info_ratio: float = 0.3,
                 use_enhanced_evaluation: bool = False,  # 默认 False，与配置文件基线组一致
                 use_tot: bool = False,
                 use_enhanced_memory: bool = False,
                 use_dynamic_reflection: bool = False,
                 llm_interface=None,
                 thought_logger=None,
                 experiment_id: str = None):
        """
        初始化信息传播模型
        
        Args:
            num_agents: 智能体数量
            initial_infected: 初始感染比例
            false_info_ratio: 虚假信息比例
            use_enhanced_evaluation: 是否使用增强评估
            use_tot: 是否使用 ToT 多层次推理
            use_enhanced_memory: 是否使用增强记忆检索
            use_dynamic_reflection: 是否使用动态反思
            llm_interface: LLM 接口
            thought_logger: 思考过程日志记录器
            experiment_id: 实验ID
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.initial_infected = initial_infected
        self.false_info_ratio = false_info_ratio
        self.use_enhanced = use_enhanced_evaluation
        self.use_tot = use_tot
        self.use_enhanced_memory = use_enhanced_memory
        self.use_dynamic_reflection = use_dynamic_reflection
        self.llm_interface = llm_interface
        self.thought_logger = thought_logger
        self.experiment_id = experiment_id
        self.context = "社交网络信息传播"
        
        # 反思阈值（Proposal 要求 0.6）
        self.reflection_threshold = 0.6
        
        # 创建调度器
        self.schedule = mesa.time.RandomActivation(self)
        
        # 创建无标度网络
        self.network = nx.barabasi_albert_graph(num_agents, 3)
        
        # 为边添加权重
        for u, v in self.network.edges():
            self.network[u][v]['weight'] = random.uniform(0.3, 0.7)
        
        # 初始化记忆工厂
        reflect_prompt = "请根据以下记忆进行反思，总结信息评估的关键经验。"
        self.memory_factory = AdvancedMemoryFactory(
            tar_llm=self.llm_interface,
            memory_num=5,
            prompt=reflect_prompt,
            model=self
        )
        
        # 信息库
        self.information_pool: Dict[str, Information] = {}
        self.info_counter = 0
        
        # 智能体字典（用于快速索引）- 必须在 _create_agents 之前初始化
        self.agents_dict: Dict[int, InfoSpreadingAgent] = {}
        
        # 创建智能体
        self._create_agents()
        
        # 传播统计
        self.spread_history: List[Dict[str, Any]] = []
        
        # 评估指标（符合 Proposal 四维度要求）
        self.metrics = create_experiment_metrics()
        self.metrics.start_experiment()
        
        # 初始化信息并传播
        self._initialize_information()
    
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
            self.agents_dict[i] = agent
            
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
            agent = self.agents_dict.get(node_id)
            if agent and isinstance(agent, InfoSpreadingAgent):
                # 随机分配真假信息
                if random.random() < self.false_info_ratio:
                    agent.receive_information(false_info)
                else:
                    agent.receive_information(true_info)
    
    def _create_information(self, info_type: InformationType) -> Information:
        """
        创建信息
        
        关键设计原则：
        1. 信息内容本身不暴露真假（需要推理判断）
        2. 可信度范围重叠（无法单靠可信度判断）
        3. 使用真实场景的信息模板
        """
        self.info_counter += 1
        
        if info_type == InformationType.TRUE:
            # 从真实信息模板中随机选择
            template = random.choice(TRUE_INFO_TEMPLATES)
            # 可信度范围：0.4-0.85（与假信息重叠）
            credibility = random.uniform(0.4, 0.85)
        else:
            # 从虚假信息模板中随机选择
            template = random.choice(FALSE_INFO_TEMPLATES)
            # 可信度范围：0.35-0.8（与真信息重叠）
            # 注意：有些谣言看起来非常可信
            credibility = random.uniform(0.35, 0.8)
        
        # 添加时间戳和变体，使每条信息略有不同
        content = f"[消息 #{self.info_counter}] {template}"
        
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
        total_rejected_true = 0
        total_rejected_false = 0
        false_spread_count = 0
        true_spread_count = 0
        
        for agent in self.schedule.agents:
            if isinstance(agent, InfoSpreadingAgent):
                total_received += len(agent.received_info)
                
                for info_id, belief in agent.believed_info.items():
                    if belief.believed:
                        if belief.information.true_type == InformationType.TRUE:
                            total_believed_true += 1
                        else:
                            total_believed_false += 1
                    else:
                        # 不相信的信息
                        if belief.information.true_type == InformationType.TRUE:
                            total_rejected_true += 1
                        else:
                            total_rejected_false += 1
        
        # 统计信息传播次数
        for info in self.information_pool.values():
            if info.true_type == InformationType.FALSE:
                false_spread_count += info.spread_count
            else:
                true_spread_count += info.spread_count
        
        # 计算遏制率：被拒绝的虚假信息 / 总虚假信息评估次数
        total_false_evaluations = total_believed_false + total_rejected_false
        containment_rate = total_rejected_false / total_false_evaluations if total_false_evaluations > 0 else 0
        
        # 计算虚假信息接受率
        total_evaluations = total_believed_true + total_believed_false + total_rejected_true + total_rejected_false
        false_belief_ratio = total_believed_false / (total_believed_true + total_believed_false + 1)
        
        return {
            'total_agents': self.num_agents,
            'total_info_received': total_received,
            'total_believed_true': total_believed_true,
            'total_believed_false': total_believed_false,
            'total_rejected_true': total_rejected_true,
            'total_rejected_false': total_rejected_false,
            'false_info_spread_count': false_spread_count,
            'true_info_spread_count': true_spread_count,
            'false_belief_ratio': false_belief_ratio,
            # 新增：遏制率（越高越好，表示虚假信息被识别并拒绝）
            'containment_rate': containment_rate,
            # 新增：总传播量
            'total_spread_count': false_spread_count + true_spread_count,
            # 新增：虚假信息传播占比
            'false_spread_ratio': false_spread_count / (false_spread_count + true_spread_count + 1)
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
    
    def _calculate_spread_velocity(self) -> Dict[str, Any]:
        """
        计算传播速度分析
        
        基于 spread_history 计算每轮的传播增量
        """
        if len(self.spread_history) < 2:
            return {
                'avg_velocity': 0,
                'max_velocity': 0,
                'velocity_trend': 'stable',
                'per_round_velocity': []
            }
        
        # 计算每轮传播增量
        velocities = []
        false_velocities = []
        
        for i in range(1, len(self.spread_history)):
            prev = self.spread_history[i - 1]
            curr = self.spread_history[i]
            
            # 总传播速度
            prev_total = prev.get('total_believed_true', 0) + prev.get('total_believed_false', 0)
            curr_total = curr.get('total_believed_true', 0) + curr.get('total_believed_false', 0)
            velocity = curr_total - prev_total
            velocities.append(velocity)
            
            # 虚假信息传播速度
            false_velocity = curr.get('total_believed_false', 0) - prev.get('total_believed_false', 0)
            false_velocities.append(false_velocity)
        
        # 计算趋势
        if len(velocities) >= 3:
            first_half = statistics.mean(velocities[:len(velocities)//2])
            second_half = statistics.mean(velocities[len(velocities)//2:])
            if second_half > first_half * 1.2:
                trend = 'accelerating'
            elif second_half < first_half * 0.8:
                trend = 'decelerating'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'avg_velocity': statistics.mean(velocities) if velocities else 0,
            'max_velocity': max(velocities) if velocities else 0,
            'min_velocity': min(velocities) if velocities else 0,
            'velocity_std': statistics.stdev(velocities) if len(velocities) > 1 else 0,
            'velocity_trend': trend,
            'per_round_velocity': velocities,
            # 虚假信息传播速度
            'avg_false_velocity': statistics.mean(false_velocities) if false_velocities else 0,
            'max_false_velocity': max(false_velocities) if false_velocities else 0,
            'false_velocity_ratio': (
                statistics.mean(false_velocities) / statistics.mean(velocities) 
                if velocities and statistics.mean(velocities) > 0 else 0
            )
        }
    
    def run_simulation(self, rounds: int = 20, verbose: bool = True) -> Dict[str, Any]:
        """运行完整模拟"""
        for i in range(rounds):
            if verbose:
                stats = self.get_spread_statistics()
                print(f"      传播 {i+1}/{rounds}: 虚假接受率={stats.get('false_belief_ratio', 0):.2f}", end='\r')
            self.step()
        
        if verbose:
            final_stats = self.get_spread_statistics()
            print(f"      完成: 虚假接受率={final_stats.get('false_belief_ratio', 0):.3f}    ")
        
        # 获取评估指标
        metrics_report = self.metrics.get_full_report()
        metrics_summary = self.metrics.get_summary()
        
        # 计算虚假信息抵抗力
        final_stats = self.get_spread_statistics()
        misinformation_resistance = SocialEffectMetrics.calculate_misinformation_resistance(
            correct_rejections=sum(
                a.correct_judgments for a in self.schedule.agents 
                if isinstance(a, InfoSpreadingAgent)
            ),
            total_false_info_exposures=sum(
                a.correct_judgments + a.incorrect_judgments 
                for a in self.schedule.agents 
                if isinstance(a, InfoSpreadingAgent)
            )
        )
        
        # 计算传播速度分析
        spread_velocity_analysis = self._calculate_spread_velocity()
        
        return {
            'spread_history': self.spread_history,
            'final_statistics': final_stats,
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
            },
            # Proposal 要求的四维度评估指标
            'evaluation_metrics': metrics_report,
            'metrics_summary': metrics_summary,
            'misinformation_resistance': misinformation_resistance,
            # 新增：传播速度分析
            'spread_velocity': spread_velocity_analysis
        }


def run_info_spreading_experiment(config: Dict[str, Any] = None,
                                   llm_interface=None,
                                   memory_path: str = None,
                                   thought_logger=None,
                                   experiment_id: str = None,
                                   **kwargs) -> Dict[str, Any]:
    """
    运行信息传播实验
    
    Args:
        config: 实验配置
        llm_interface: LLM 接口（必须提供）
        memory_path: 持久化记忆路径（暂未使用）
        thought_logger: 思考日志记录器（暂未使用）
        experiment_id: 实验ID（暂未使用）
        
    Returns:
        实验结果
        
    Raises:
        RuntimeError: 如果未提供 llm_interface
    """
    # memory_path 暂未实现
    _ = memory_path
    
    if llm_interface is None:
        raise RuntimeError(
            "信息传播实验需要 LLM 接口！\n"
            "这是一个智能体实验，必须使用 LLM 进行信息评估。\n\n"
            "使用方法：\n"
            "  from casevo import create_default_llm\n"
            "  llm = create_default_llm()\n"
            "  results = run_info_spreading_experiment(config, llm_interface=llm)"
        )
    
    if config is None:
        config = {
            'num_agents': 200,
            'initial_infected': 0.1,
            'false_info_ratio': 0.3,
            'num_rounds': 20
        }
    
    model = InfoSpreadingModel(
        num_agents=config.get('num_agents', 200),
        initial_infected=config.get('initial_infected', 0.1),
        false_info_ratio=config.get('false_info_ratio', 0.3),
        use_enhanced_evaluation=config.get('use_enhanced_evaluation', True),
        use_tot=config.get('use_tot', False),
        use_enhanced_memory=config.get('use_enhanced_memory', False),
        use_dynamic_reflection=config.get('use_dynamic_reflection', False),
        llm_interface=llm_interface,
        thought_logger=thought_logger,
        experiment_id=experiment_id
    )
    
    results = model.run_simulation(rounds=config.get('num_rounds', 20))
    results['config'] = config
    results['tot_enabled'] = config.get('use_tot', False)
    results['enhanced_memory_enabled'] = config.get('use_enhanced_memory', False)
    results['dynamic_reflection_enabled'] = config.get('use_dynamic_reflection', False)
    
    return results


if __name__ == "__main__":
    from casevo import create_default_llm
    
    print("=" * 60)
    print("信息传播智能体实验")
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
    
    # 运行 CoT 基线实验
    print("\n运行 CoT 基线实验...")
    baseline_results = run_info_spreading_experiment({
        'use_tot': False,
        'use_enhanced_memory': False,
        'use_dynamic_reflection': False,
        'num_agents': 200,
        'num_rounds': 20
    }, llm_interface=llm)
    print(f"CoT 结果 - 虚假信息接受率: {baseline_results['final_statistics']['false_belief_ratio']:.3f}")
    print(f"CoT 判断准确率: {baseline_results['accuracy_statistics']['overall_accuracy']:.3f}")
    
    # 运行 ToT 优化实验
    print("\n运行 ToT 优化实验...")
    optimized_results = run_info_spreading_experiment({
        'use_tot': True,
        'use_enhanced_memory': True,
        'use_dynamic_reflection': True,
        'num_agents': 200,
        'num_rounds': 20
    }, llm_interface=llm)
    print(f"ToT 结果 - 虚假信息接受率: {optimized_results['final_statistics']['false_belief_ratio']:.3f}")
    print(f"ToT 判断准确率: {optimized_results['accuracy_statistics']['overall_accuracy']:.3f}")
    
    # 保存结果
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'info_spreading', 'baseline.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline_cot': baseline_results,
            'optimized_tot': optimized_results
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n结果已保存到 {results_path}")

