# 选举投票模拟实验

## 实验简介

本实验模拟 2020 年美国总统大选投票过程，展示智能体如何接收辩论信息、进行讨论、反思并做出投票决策。

## 实验设计

### 场景设置

- **选民数量**: 默认 101 人（可配置）
- **网络结构**: 小世界网络（Watts-Strogatz）
- **辩论轮数**: 6 轮
- **政治倾向分布**: 基于 Pew Research 政治类型学

### 实验流程

1. **公开辩论阶段**: 智能体接收候选人辩论内容
2. **讨论阶段**: 智能体之间进行意见交换
3. **反思阶段**: 智能体基于记忆进行反思
4. **投票阶段**: 智能体做出最终投票决定

## 配置说明

配置文件: `experiments/configs/election_config.json`

### 主要参数

```json
{
    "num_voters": 101,
    "network_degree": 6,
    "network_rewire_prob": 0.3,
    "use_tot": true,
    "num_rounds": 6,
    "llm_temperature": 0.7,
    "memory_top_k": 5,
    "reflection_threshold": 0.6
}
```

### 政治倾向分布

- Progressive Left: 6%
- Establishment Liberal: 13%
- Democratic Mainstays: 16%
- Outsider Left: 10%
- Stressed Sideliners: 15%
- Ambivalent Right: 12%
- Committed Conservatives: 7%
- Populist Right: 11%
- Faith and Flag: 10%

## 运行实验

### 基础版本

```bash
cd experiments
python election_scenario.py
```

### LLM 增强版本

```bash
python election_with_llm.py --voters 10 --rounds 3
```

### 多模型对比

```bash
python election_with_llm.py --compare --models gpt-4o-mini gpt-4o
```

## 结果分析

### 输出指标

- **投票准确率**: 与真实投票结果的匹配度
- **决策一致性**: 智能体决策的稳定性
- **观点稳定性**: 观点变化程度
- **推理深度**: ToT 搜索深度
- **响应时间**: 平均响应时间
- **记忆利用率**: 记忆检索频率

### 结果文件

- `election_results.json`: 基础实验结果
- `election_llm_results.json`: LLM 版本结果
- `multi_model_comparison.json`: 多模型对比结果

## 实验变体

### CoT vs ToT 对比

```bash
python cot_vs_tot_comparison.py --voters 30 --rounds 3
```

### 完整优化版本

启用所有优化功能：
- Tree of Thought
- Advanced Memory
- Dynamic Reflection

## 最佳实践

1. **小规模测试**: 先用少量选民（5-10人）测试
2. **API 成本**: LLM 版本会产生 API 调用费用
3. **随机种子**: 设置随机种子确保可重现性
4. **结果保存**: 定期保存中间结果

## 常见问题

**Q: 实验运行时间过长？**

A: 减少选民数量或轮数，或使用模拟版本而非 LLM 版本。

**Q: API 调用失败？**

A: 检查 API Key 配置，确保网络连接正常。

**Q: 结果不一致？**

A: 设置随机种子，确保实验可重现。

