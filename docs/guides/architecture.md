# 架构设计文档

本文档介绍 Casevo 框架的整体架构设计。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Casevo Framework                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   ModelBase  │  │  AgentBase  │  │    Memory    │   │
│  │              │  │              │  │              │   │
│  │ - Network    │  │ - Chains     │  │ - Short-term │   │
│  │ - Schedule   │  │ - Memory     │  │ - Long-term  │   │
│  │ - Context    │  │ - Reflection │  │ - Retrieval  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ ThoughtChain │  │ TreeOfThought│  │Collaborative │   │
│  │              │  │              │  │  Decision    │   │
│  │ - Steps      │  │ - Nodes      │  │ - Negotiate  │   │
│  │ - Execution  │  │ - Search     │  │ - Consensus   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │  LLM Interface│  │  Prompt      │                     │
│  │               │  │  Factory     │                     │
│  │ - OpenAI     │  │ - Templates  │                     │
│  │ - Custom     │  │ - Rendering  │                     │
│  └──────────────┘  └──────────────┘                     │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## 核心模块关系

### ModelBase 和 AgentBase

```
ModelBase
  ├── 管理多个 AgentBase 实例
  ├── 提供 NetworkGrid 网络空间
  ├── 管理 MemoryFactory（共享）
  └── 提供 PromptFactory（共享）

AgentBase
  ├── 属于一个 ModelBase
  ├── 拥有独立的 Memory 实例
  ├── 可以设置多个 ThoughtChain
  └── 通过 ModelBase 访问共享资源
```

### Memory 系统

```
MemoryFactory (Model 级别)
  ├── 创建 Memory 实例
  ├── 管理向量数据库
  └── 提供检索接口

Memory (Agent 级别)
  ├── 短期记忆（内存）
  ├── 长期记忆（向量数据库）
  └── 反思机制
```

### ThoughtChain 系统

```
ThoughtChain
  ├── 包含多个 Step
  ├── 顺序执行 Steps
  └── 传递中间结果

Step 类型:
  - BaseStep: 基础步骤
  - ChoiceStep: 选择步骤
  - ScoreStep: 评分步骤
  - JsonStep: JSON 格式步骤
```

## 数据流

### 模拟执行流程

```
1. Model.step() 被调用
   │
   ├─> 2. 执行全局事件
   │
   ├─> 3. Schedule.step() 调度智能体
   │     │
   │     └─> 4. Agent.step() 执行
   │           │
   │           ├─> 5. 调用 ThoughtChain
   │           │     │
   │           │     └─> 6. 执行 Steps
   │           │           │
   │           │           └─> 7. 调用 LLM
   │           │
   │           └─> 8. 更新 Memory
   │
   └─> 9. 收集结果
```

### 记忆检索流程

```
Agent 需要检索记忆
  │
  ├─> Memory.search_short_memory()
  │     │
  │     └─> MemoryFactory.search_short_memory_by_doc()
  │           │
  │           └─> ChromaDB 向量检索
  │
  └─> Memory.get_long_memory()
        │
        └─> MemoryFactory 检索长期记忆
```

## 设计模式

### 1. 工厂模式

- `PromptFactory`: 创建和管理 Prompt 实例
- `MemoryFactory`: 创建和管理 Memory 实例

### 2. 策略模式

- `SearchStrategy`: 不同的搜索策略（BFS、DFS、最佳优先）
- `DecisionMode`: 不同的决策模式（分布式、中心化、混合）

### 3. 模板方法模式

- `AgentBase.step()`: 定义算法骨架，子类实现具体步骤
- `ModelBase.step()`: 定义模拟流程，子类可以扩展

### 4. 观察者模式

- `TotLog`: 记录和观察系统事件
- `MesaLog`: Mesa 框架的日志系统

## 扩展点

### 添加新的 LLM 接口

实现 `LLM_INTERFACE` 接口：

```python
from casevo import LLM_INTERFACE

class MyLLM(LLM_INTERFACE):
    def send_message(self, prompt, json_flag=False):
        # 实现消息发送
        pass
    
    def send_embedding(self, text_list):
        # 实现嵌入生成
        pass
```

### 添加新的 Step 类型

继承 `BaseStep`：

```python
from casevo import BaseStep

class MyStep(BaseStep):
    def pre_process(self, input, agent=None, model=None):
        # 预处理逻辑
        return processed_input
    
    def after_process(self, input, response, agent=None, model=None):
        # 后处理逻辑
        return result
```

### 添加新的实验场景

继承 `ModelBase` 和 `AgentBase`：

```python
from casevo import ModelBase, AgentBase

class MyModel(ModelBase):
    def step(self):
        # 实现模拟逻辑
        pass

class MyAgent(AgentBase):
    def step(self):
        # 实现智能体行为
        pass
```

## 性能考虑

1. **LLM 调用优化**
   - 使用 `RequestCache` 缓存重复请求
   - 批量处理请求
   - 异步调用（`ThreadSend`）

2. **内存管理**
   - 定期压缩长期记忆
   - 限制短期记忆大小
   - 使用向量数据库索引

3. **并发处理**
   - 支持多线程执行
   - 异步 I/O 操作
   - 并行智能体处理

## 安全考虑

1. **API Key 管理**
   - 使用环境变量
   - 不在代码中硬编码
   - 使用配置文件（不提交到版本控制）

2. **数据隐私**
   - 敏感数据加密
   - 访问控制
   - 日志脱敏

## 未来扩展

- [ ] 支持更多 LLM 提供商
- [ ] 分布式模拟支持
- [ ] Web UI 界面
- [ ] 实时可视化
- [ ] 更多实验场景

