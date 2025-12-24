# 快速开始指南

本指南将帮助您快速上手 Casevo 框架。

## 安装

### 前置要求

- Python 3.11 或更高版本
- pip 或 conda 包管理器

### 安装步骤

1. **创建虚拟环境**（推荐）

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 使用 conda
conda create -n casevo python=3.11
conda activate casevo
```

2. **安装 Casevo**

```bash
# 从 wheel 文件安装
pip install casevo-0.3.*-py3-none-any.whl

# 或从源码安装
git clone https://github.com/rgCASS/casevo.git
cd casevo
pip install -e .
```

3. **安装可选依赖**（用于实验）

```bash
pip install -e ".[experiments]"
```

## 基本概念

### 核心组件

1. **Model（模型）**
   - 定义全局场景信息
   - 管理智能体调度
   - 处理全局事件

2. **Agent（智能体）**
   - 代表模拟中的个体
   - 具有记忆、推理能力
   - 可以与其他智能体交互

3. **Memory（记忆）**
   - 短期记忆：最近的事件
   - 长期记忆：重要经验
   - 支持向量检索

4. **ThoughtChain（思维链）**
   - 定义推理步骤
   - 支持 CoT 和 ToT
   - 可组合的步骤链

## 第一个示例

### 创建简单的智能体

```python
from casevo import AgentBase, ModelBase, create_default_llm
import networkx as nx

# 创建 LLM 接口
llm = create_default_llm()

# 创建网络
graph = nx.complete_graph(5)

# 创建模型
class SimpleModel(ModelBase):
    def step(self):
        self.schedule.step()
        return 0

# 创建智能体
class SimpleAgent(AgentBase):
    def step(self):
        # 智能体行为逻辑
        pass

# 初始化模型
model = SimpleModel(graph, llm)

# 添加智能体
for i in range(5):
    agent = SimpleAgent(i, model, f"Agent {i}", None)
    model.add_agent(agent, i)

# 运行模拟
for _ in range(10):
    model.step()
```

## 下一步

- 查看 [API 参考](api/README.md) 了解详细接口
- 阅读 [实验指南](experiments/README.md) 运行示例实验
- 参考 [架构设计](architecture.md) 深入理解系统

## 常见问题

### Q: 如何配置 LLM API？

A: 使用环境变量或创建 LLM 实例时指定：

```python
from casevo import OpenAILLM

llm = OpenAILLM(
    api_key="your-api-key",
    base_url="https://api.example.com",
    model="gpt-4o-mini"
)
```

### Q: 如何保存实验结果？

A: 实验结果默认保存在 `experiments/results/` 目录。可以使用 `ReportGenerator` 生成报告。

### Q: 支持哪些 LLM？

A: 目前支持 OpenAI 兼容的 API。可以通过实现 `LLM_INTERFACE` 接口添加其他 LLM。

## 获取帮助

- 查看 [GitHub Issues](https://github.com/rgCASS/casevo/issues)
- 阅读完整 [API 文档](api/README.md)
- 参考 [示例代码](../experiments/)

