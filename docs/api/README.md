# API 参考文档

Casevo 框架的完整 API 参考。

## 核心模块

### AgentBase

智能体基类，所有智能体的父类。

**位置**: `casevo.agent_base.AgentBase`

**主要方法**:
- `setup_chain(chain_dict)`: 设置思维链
- `step()`: 智能体行为（需子类实现）

**示例**:
```python
from casevo import AgentBase

class MyAgent(AgentBase):
    def step(self):
        # 实现智能体行为
        pass
```

### ModelBase

模型基类，定义模拟场景。

**位置**: `casevo.model_base.ModelBase`

**主要方法**:
- `add_agent(agent, node_id)`: 添加智能体
- `step()`: 执行一轮模拟

**示例**:
```python
from casevo import ModelBase

class MyModel(ModelBase):
    def step(self):
        self.schedule.step()
        return 0
```

### Memory

记忆系统，管理智能体的短期和长期记忆。

**位置**: `casevo.memory.Memory`

**主要方法**:
- `add_short_memory(...)`: 添加短期记忆
- `reflect_memory()`: 反思并更新长期记忆
- `get_long_memory()`: 获取长期记忆

### ThoughtChain

思维链，定义推理步骤序列。

**位置**: `casevo.chain.ThoughtChain`

**主要步骤类型**:
- `BaseStep`: 基础步骤
- `ChoiceStep`: 选择步骤
- `ScoreStep`: 评分步骤
- `JsonStep`: JSON 格式步骤

## 增强模块

### TreeOfThought

树状思维推理实现。

**位置**: `casevo.enhanced_chain.TreeOfThought`

**特性**:
- 多路径探索
- 节点评估
- 搜索策略（BFS、DFS、最佳优先）

### AdvancedMemory

高级记忆系统，支持上下文感知检索。

**位置**: `casevo.advanced_memory.AdvancedMemory`

**特性**:
- 上下文感知检索
- 记忆压缩
- 重要性分级

### DecisionEvaluator

决策评估器，评估智能体决策质量。

**位置**: `casevo.decision_evaluator.DecisionEvaluator`

**功能**:
- 决策记录
- 置信度估计
- 元认知评估

### CollaborativeDecisionMaker

协同决策器，支持多智能体协商。

**位置**: `casevo.collaborative_decision.CollaborativeDecisionMaker`

**协议**:
- 标准协商协议
- 分布式共识
- 中心化聚合

## 工具模块

### PromptFactory

提示词工厂，管理模板和生成提示。

**位置**: `casevo.prompt.PromptFactory`

### TotLog

日志记录工具。

**位置**: `casevo.util.tot_log.TotLog`

### RequestCache

请求缓存，减少重复 API 调用。

**位置**: `casevo.util.cache.RequestCache`

## 详细文档

- [核心模块详细文档](core.md)
- [增强模块详细文档](enhanced.md)
- [工具模块文档](utils.md)

