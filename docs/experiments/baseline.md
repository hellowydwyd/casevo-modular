# 基线对比实验指南

## 实验简介

基线对比实验用于系统性地比较不同配置和优化方案在三个主要实验场景中的表现。

## 实验设计

### 对比维度

1. **推理方法**: CoT vs ToT
2. **记忆系统**: 基础记忆 vs 高级记忆
3. **反思机制**: 静态反思 vs 动态反思
4. **决策方式**: 独立决策 vs 协同决策

### 实验场景

- **选举投票**: 对比 CoT 和 ToT 的决策质量
- **资源分配**: 对比独立分配和协同分配
- **信息传播**: 对比基础记忆和高级记忆

## 运行实验

### 完整对比

运行所有实验的完整对比：

```bash
cd experiments
python baseline_comparison.py --runs 5 --experiment all
```

### 单个实验对比

只运行特定实验：

```bash
# 选举实验
python baseline_comparison.py --runs 5 --experiment election

# 资源分配实验
python baseline_comparison.py --runs 5 --experiment resource

# 信息传播实验
python baseline_comparison.py --runs 5 --experiment info
```

### 自定义输出目录

```bash
python baseline_comparison.py --output my_results --runs 3
```

## 结果分析

### 输出文件

- `baseline_comparison_*.json`: 完整对比结果
- `comparison_summary_*.txt`: 文本摘要报告

### 结果结构

```json
{
    "election": {
        "baseline_cot": {
            "avg_biden_support": 0.52,
            "avg_trump_support": 0.45,
            ...
        },
        "optimized_tot": {
            "avg_biden_support": 0.55,
            "avg_trump_support": 0.42,
            ...
        }
    },
    "resource": {...},
    "info_spreading": {...}
}
```

### 关键指标

#### 选举实验
- Biden/Trump 支持率
- 决策一致性
- 推理深度

#### 资源分配
- 基尼系数
- 协商轮次
- 满意度

#### 信息传播
- 传播准确性
- 虚假信息比例
- 验证率

## 使用报告生成器

生成详细的分析报告：

```python
from experiments.report_generator import ReportGenerator

generator = ReportGenerator()
generator.load_results("baseline_comparison_20251207_135410.json")
report = generator.generate_full_report()
generator.save_report(report, "comparison_report.md")
```

## 最佳实践

1. **多次运行**: 至少运行 5-10 次取平均值
2. **固定种子**: 使用相同的随机种子确保公平对比
3. **参数一致**: 除了对比维度外，其他参数保持一致
4. **结果保存**: 及时保存结果，避免丢失
5. **统计分析**: 进行统计显著性检验

## 实验配置

实验配置从 `experiments/configs/` 目录加载：

- `election_config.json`
- `resource_config.json`
- `info_spreading_config.json`

可以修改这些文件来调整实验参数。

## 常见问题

**Q: 实验运行时间很长？**

A: 减少运行次数或使用更小的配置（如减少智能体数量）。

**Q: 结果差异不明显？**

A: 增加运行次数或调整对比参数。

**Q: 如何解读结果？**

A: 查看生成的摘要报告，重点关注改进幅度和统计显著性。

