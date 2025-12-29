"""
智能体思考过程日志记录器

保存每个智能体在每轮决策中的完整思考过程，包括：
- 输入上下文
- 推理步骤（CoT/ToT）
- 记忆检索结果
- 反思内容
- 最终决策
"""

import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ThoughtRecord:
    """单次思考记录"""
    agent_id: str
    agent_name: str
    round_num: int
    timestamp: str
    
    # 输入上下文
    input_context: str
    memories_retrieved: List[str]
    
    # 推理过程
    reasoning_type: str  # "cot" or "tot"
    reasoning_steps: List[str]
    tot_branches: Optional[List[Dict[str, Any]]] = None
    tot_evaluations: Optional[List[Dict[str, float]]] = None
    
    # 反思
    reflection_triggered: bool = False
    reflection_content: Optional[str] = None
    
    # 协作
    neighbor_opinions: Optional[List[Dict[str, str]]] = None
    collaborative_influence: Optional[str] = None
    
    # 决策结果
    decision: str = ""
    confidence: float = 0.0
    reasoning_summary: str = ""


class ThoughtLogger:
    """思考过程日志记录器"""
    
    def __init__(self, output_dir: str = "experiments/results/thoughts",
                 experiment_name: str = None):
        """
        初始化日志记录器
        
        Args:
            output_dir: 输出目录
            experiment_name: 实验名称（用于文件命名）
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._records: Dict[str, List[ThoughtRecord]] = {}  # agent_id -> records
        self._lock = threading.Lock()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 实时日志文件
        self._log_file = os.path.join(
            output_dir, 
            f"thoughts_{self.experiment_name}.jsonl"
        )
    
    def record_thought(self, 
                       agent_id: str,
                       agent_name: str,
                       round_num: int,
                       input_context: str,
                       memories_retrieved: List[str],
                       reasoning_type: str,
                       reasoning_steps: List[str],
                       decision: str,
                       confidence: float,
                       reasoning_summary: str = "",
                       tot_branches: List[Dict] = None,
                       tot_evaluations: List[Dict] = None,
                       reflection_triggered: bool = False,
                       reflection_content: str = None,
                       neighbor_opinions: List[Dict] = None,
                       collaborative_influence: str = None):
        """
        记录一次完整的思考过程
        """
        record = ThoughtRecord(
            agent_id=agent_id,
            agent_name=agent_name,
            round_num=round_num,
            timestamp=datetime.now().isoformat(),
            input_context=input_context,
            memories_retrieved=memories_retrieved,
            reasoning_type=reasoning_type,
            reasoning_steps=reasoning_steps,
            tot_branches=tot_branches,
            tot_evaluations=tot_evaluations,
            reflection_triggered=reflection_triggered,
            reflection_content=reflection_content,
            neighbor_opinions=neighbor_opinions,
            collaborative_influence=collaborative_influence,
            decision=decision,
            confidence=confidence,
            reasoning_summary=reasoning_summary
        )
        
        with self._lock:
            if agent_id not in self._records:
                self._records[agent_id] = []
            self._records[agent_id].append(record)
            
            # 实时写入JSONL文件
            self._append_to_log(record)
    
    def _append_to_log(self, record: ThoughtRecord):
        """追加写入日志文件"""
        try:
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"写入思考日志失败: {e}")
    
    def get_agent_thoughts(self, agent_id: str) -> List[ThoughtRecord]:
        """获取指定智能体的所有思考记录"""
        return self._records.get(agent_id, [])
    
    def get_round_thoughts(self, round_num: int) -> List[ThoughtRecord]:
        """获取指定轮次的所有思考记录"""
        result = []
        for records in self._records.values():
            for record in records:
                if record.round_num == round_num:
                    result.append(record)
        return result
    
    def save_full_log(self, filename: str = None):
        """保存完整的结构化日志"""
        if filename is None:
            filename = f"thoughts_full_{self.experiment_name}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # 转换为可序列化格式
        data = {
            'experiment_name': self.experiment_name,
            'total_agents': len(self._records),
            'total_records': sum(len(r) for r in self._records.values()),
            'agents': {}
        }
        
        for agent_id, records in self._records.items():
            data['agents'][agent_id] = [asdict(r) for r in records]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"完整思考日志已保存到: {output_path}")
        return output_path
    
    def generate_summary(self) -> Dict[str, Any]:
        """生成思考过程摘要统计"""
        summary = {
            'total_thoughts': sum(len(r) for r in self._records.values()),
            'agents_count': len(self._records),
            'by_reasoning_type': {'cot': 0, 'tot': 0},
            'reflection_rate': 0,
            'avg_reasoning_steps': 0,
            'avg_confidence': 0,
            'decision_distribution': {}
        }
        
        total_steps = 0
        total_confidence = 0
        reflection_count = 0
        
        for records in self._records.values():
            for record in records:
                summary['by_reasoning_type'][record.reasoning_type] = \
                    summary['by_reasoning_type'].get(record.reasoning_type, 0) + 1
                
                total_steps += len(record.reasoning_steps)
                total_confidence += record.confidence
                
                if record.reflection_triggered:
                    reflection_count += 1
                
                summary['decision_distribution'][record.decision] = \
                    summary['decision_distribution'].get(record.decision, 0) + 1
        
        if summary['total_thoughts'] > 0:
            summary['avg_reasoning_steps'] = total_steps / summary['total_thoughts']
            summary['avg_confidence'] = total_confidence / summary['total_thoughts']
            summary['reflection_rate'] = reflection_count / summary['total_thoughts']
        
        return summary


# 全局日志记录器实例
_global_logger: Optional[ThoughtLogger] = None


def get_thought_logger(output_dir: str = None, 
                       experiment_name: str = None) -> ThoughtLogger:
    """获取或创建全局日志记录器"""
    global _global_logger
    
    if _global_logger is None or experiment_name is not None:
        _global_logger = ThoughtLogger(
            output_dir=output_dir or "experiments/results/thoughts",
            experiment_name=experiment_name
        )
    
    return _global_logger


def reset_thought_logger():
    """重置全局日志记录器"""
    global _global_logger
    _global_logger = None

