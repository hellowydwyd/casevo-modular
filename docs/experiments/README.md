# 实验指南

Casevo 框架提供了三个主要实验场景，用于演示和验证框架功能。

## 实验概览

### 1. 选举投票模拟

模拟 2020 年美国总统大选投票过程，展示智能体如何：
- 接收辩论信息
- 进行讨论和意见交换
- 反思和更新观点
- 做出投票决策

**相关文件**:
- `experiments/election_scenario.py` - 基础实验
- `experiments/election_with_llm.py` - LLM 增强版本
- `experiments/configs/election_config.json` - 配置文件

**快速开始**:
```bash
cd experiments
python election_scenario.py
```

### 2. 资源分配协商

模拟多智能体资源协商分配过程，展示：
- 多智能体协商机制
- 公平性评估
- 协同决策能力

**相关文件**:
- `experiments/resource_allocation.py` - 基础实验
- `experiments/resource_allocation_llm.py` - LLM 增强版本
- `experiments/configs/resource_config.json` - 配置文件

**快速开始**:
```bash
cd experiments
python resource_allocation.py
```

### 3. 信息传播动力学

模拟信息在网络中的传播过程，展示：
- 信息扩散机制
- 真假信息识别
- 网络拓扑影响

**相关文件**:
- `experiments/info_spreading.py` - 基础实验
- `experiments/info_spreading_llm.py` - LLM 增强版本
- `experiments/configs/info_spreading_config.json` - 配置文件

**快速开始**:
```bash
cd experiments
python info_spreading.py
```

## 基线对比实验

运行完整的基线对比实验，比较不同配置的性能：

```bash
cd experiments
python baseline_comparison.py --runs 5 --experiment all
```

**输出**:
- 结果文件: `experiments/results/baseline_comparison_*.json`
- 摘要报告: `experiments/results/comparison_summary_*.txt`

## 实验配置

所有实验配置都在 `experiments/configs/` 目录中：

- `election_config.json` - 选举实验配置
- `resource_config.json` - 资源分配配置
- `info_spreading_config.json` - 信息传播配置

## 结果分析

实验结果保存在 `experiments/results/` 目录。可以使用 `ReportGenerator` 生成分析报告：

```python
from experiments.report_generator import ReportGenerator

generator = ReportGenerator()
report = generator.generate_full_report(
    election_results=...,
    resource_results=...,
    info_results=...
)
generator.save_report(report, "my_report.md")
```

## 详细文档

- [选举投票模拟详细指南](election.md)
- [资源分配协商详细指南](resource.md)
- [信息传播动力学详细指南](info_spreading.md)
- [基线对比实验指南](baseline.md)

## 实验最佳实践

1. **配置管理**: 使用 JSON 配置文件，便于版本控制和参数调整
2. **结果保存**: 所有结果统一保存到 `experiments/results/`
3. **可重现性**: 设置随机种子，确保实验可重现
4. **性能监控**: 记录运行时间和资源使用情况
5. **错误处理**: 添加适当的异常处理和日志记录

