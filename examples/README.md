# Casevo 示例

本目录包含 Casevo 框架的使用示例。

## 快速开始

### 1. 基础选举模拟

```python
from casevo import AgentBase, ModelBase, create_default_llm

# 创建 LLM 实例
llm = create_default_llm()

# 创建并运行模拟
# 详见 quickstart.py
```

### 2. 运行示例

```bash
# 安装依赖
pip install -e .

# 运行快速开始示例
python examples/quickstart.py
```

## 示例列表

| 示例 | 描述 |
|------|------|
| `quickstart.py` | 最简单的入门示例 |

## 更多信息

- [完整文档](../docs/README.md)
- [实验指南](../docs/experiments/README.md)
- [API 参考](../docs/api/README.md)

