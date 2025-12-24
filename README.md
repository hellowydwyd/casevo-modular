# Casevo Modular: 模块化多智能体社会模拟框架

> 基于 [Casevo](https://github.com/rgCASS/casevo) 项目的模块化重构版本，采用模块化架构设计

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目简介

**Casevo Modular** 是基于原 [Casevo](https://github.com/rgCASS/casevo) 项目的模块化重构版本，专注于清晰的模块架构和强大的推理能力。本项目重构了代码结构，提供了更清晰的模块划分和更强大的功能。

### 核心特性

- 🧠 **增强推理能力**
  - 思维链 (Chain of Thought, CoT)
  - 树状思维 (Tree of Thought, ToT)
  - 多智能体协同决策

- 💾 **高级记忆系统**
  - 短期/长期记忆管理
  - 上下文感知检索 (RAG)
  - 动态反思机制
  - 记忆压缩与重要性分级

- 🏗️ **模块化架构**
  - `casevo.core`: 核心模块 (AgentBase, ModelBase)
  - `casevo.llm`: LLM 接口抽象
  - `casevo.memory`: 记忆系统
  - `casevo.reasoning`: 推理模块
  - `casevo.utils`: 工具模块

- 🔬 **实验场景**
  - 选举投票模拟
  - 信息传播研究
  - 资源分配实验

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/hellowydwyd/casevo-modular.git
cd casevo-modular

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -e .
```

### 基本使用

```python
from casevo import AgentBase, ModelBase, create_default_llm
import networkx as nx

# 创建 LLM 实例
llm = create_default_llm()

# 创建网络图
graph = nx.complete_graph(10)

# 创建模型
class MyModel(ModelBase):
    def step(self):
        self.schedule.step()
        return 0

model = MyModel(graph, llm)

# 创建智能体
class MyAgent(AgentBase):
    def step(self):
        # 实现智能体行为
        pass

agent = MyAgent(0, model, "智能体描述", None)
model.add_agent(agent, 0)

# 运行模拟
for _ in range(10):
    model.step()
```

## 📚 文档

- **[文档中心](docs/README.md)** - 完整文档索引
- **[快速开始指南](docs/guides/getting_started.md)** - 详细使用教程
- **[API 参考](docs/api/README.md)** - 完整 API 文档
- **[架构设计](docs/guides/architecture.md)** - 系统架构说明
- **[实验指南](docs/experiments/README.md)** - 实验场景说明

## 🎯 主要改进

### 模块化重构

原项目采用扁平化结构，本项目重构为清晰的模块化架构：

```
src/casevo/
├── core/          # 核心模块
├── llm/           # LLM 接口
├── memory/        # 记忆系统
├── reasoning/     # 推理模块
└── utils/         # 工具模块
```

### 增强功能

- ✅ **Tree of Thought (ToT)** 完整实现
- ✅ **高级记忆系统** 支持上下文感知检索
- ✅ **协同决策** 多智能体协商机制
- ✅ **决策评估** 元认知和置信度估计
- ✅ **完整的测试覆盖**

## 🔬 实验场景

### 选举投票实验

模拟 2020 年美国总统大选辩论投票过程，研究选民决策演化。

```bash
cd experiments/election
python with_llm.py
```

### 信息传播实验

研究信息在网络中的传播机制和影响范围。

```bash
cd experiments/info_spreading
python with_llm.py
```

### 资源分配实验

模拟资源分配决策过程，研究协作与竞争机制。

```bash
cd experiments/resource
python with_llm.py
```

## 🛠️ 技术栈

- **Python 3.11+**
- **Mesa 2.4.0** - Agent-based Modeling 框架
- **ChromaDB** - 向量数据库
- **NetworkX** - 网络分析
- **Jinja2** - 模板引擎

## 📦 项目结构

```
casevo-modular/
├── src/casevo/          # 核心框架代码
│   ├── core/            # 核心模块
│   ├── llm/             # LLM 接口
│   ├── memory/          # 记忆系统
│   ├── reasoning/       # 推理模块
│   └── utils/           # 工具模块
├── experiments/         # 实验场景
│   ├── election/        # 选举投票
│   ├── info_spreading/  # 信息传播
│   └── resource/        # 资源分配
├── docs/                # 文档
├── tests/               # 测试代码
└── examples/            # 示例代码
```

## 🤝 贡献

欢迎贡献！请阅读 [贡献指南](docs/CONTRIBUTING.md) 了解详细信息。

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目基于 [Casevo](https://github.com/rgCASS/casevo) 项目开发，感谢原项目作者和贡献者：

- [Zexun Jiang](https://github.com/rgCASS)
- [Yafang Shi](https://github.com/Freya236)
- [Maoxu Li](https://github.com/limaoSure)
- [Hang Su](https://github.com/suhangha)

### 原项目论文

```bibtex
@misc{jiang2024casevocognitiveagentssocial,
      title={Casevo: A Cognitive Agents and Social Evolution Simulator}, 
      author={Zexun Jiang and Yafang Shi and Maoxu Li and Hongjiang Xiao and Yunxiao Qin and Qinglan Wei and Ye Wang and Yuan Zhang},
      year={2024},
      eprint={2412.19498},
      archivePrefix={arXiv},
      primaryClass={cs.SI},
      url={https://arxiv.org/abs/2412.19498}, 
}
```

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/hellowydwyd/casevo-modular/issues)
- 发送 Pull Request

---

**注意**: 本项目是原 Casevo 项目的模块化重构版本，专注于清晰的架构设计和功能扩展。如需使用原项目，请访问 [rgCASS/casevo](https://github.com/rgCASS/casevo)。
