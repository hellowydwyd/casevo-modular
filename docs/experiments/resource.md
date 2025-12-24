# 资源分配协商实验

## 实验简介

本实验模拟多智能体资源协商分配过程，展示协同决策机制如何提高资源分配的公平性和效率。

## 实验设计

### 场景设置

- **智能体数量**: 默认 50 个（可配置）
- **总资源**: 1000 单位（可配置）
- **优先级分布**: Critical/High/Normal/Low
- **最大协商轮数**: 10 轮

### 协商机制

1. **需求评估**: 智能体评估自身资源需求
2. **协商阶段**: 智能体之间进行资源协商
3. **分配决策**: 基于协商结果分配资源
4. **公平性评估**: 计算基尼系数等指标

## 配置说明

配置文件: `experiments/configs/resource_config.json`

### 主要参数

```json
{
    "num_agents": 50,
    "total_resources": 1000,
    "max_rounds": 10,
    "use_collaborative": true,
    "decision_mode": "hybrid",
    "convergence_threshold": 0.05
}
```

### 优先级分布

- Critical: 10%
- High: 20%
- Normal: 50%
- Low: 20%

## 运行实验

### 基础版本

```bash
cd experiments
python resource_allocation.py
```

### LLM 增强版本

```bash
python resource_allocation_llm.py --agents 30 --resources 600
```

### 对比实验

运行基线和优化版本对比：

```python
from experiments import run_resource_experiment

# 基线版本
baseline = run_resource_experiment({
    'use_collaborative': False
})

# 优化版本
optimized = run_resource_experiment({
    'use_collaborative': True,
    'decision_mode': 'hybrid'
})
```

## 结果分析

### 输出指标

- **协商轮次**: 达成共识所需轮数
- **基尼系数**: 资源分配公平性（0-1，越小越公平）
- **平均满意度**: 智能体平均满意度
- **最低满意度**: 最不满意智能体的满意度
- **资源利用率**: 资源使用率
- **收敛速度**: 达成共识的速度

### 结果文件

- `resource_results.json`: 基础实验结果
- `resource_llm_comparison.json`: LLM 版本对比结果

## 决策模式

### 分布式模式

智能体之间直接协商，无需中央协调。

### 中心化模式

通过中央聚合器统一分配。

### 混合模式

结合分布式和中心化的优势。

## 最佳实践

1. **资源约束**: 确保总资源小于总需求，模拟稀缺性
2. **优先级设置**: 合理设置优先级分布
3. **收敛条件**: 调整收敛阈值平衡效率和公平性
4. **大规模测试**: 测试不同规模的智能体群体

## 常见问题

**Q: 协商无法收敛？**

A: 增加最大轮数或调整收敛阈值。

**Q: 公平性指标异常？**

A: 检查优先级分布和需求范围设置。

**Q: 性能问题？**

A: 减少智能体数量或使用更简单的决策模式。

