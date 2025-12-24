# 信息传播动力学实验

## 实验简介

本实验模拟信息在网络中的传播过程，展示智能体如何识别、传播和验证信息，以及网络拓扑对信息传播的影响。

## 实验设计

### 场景设置

- **智能体数量**: 默认 50 个（可配置）
- **网络类型**: 小世界网络、随机网络等
- **信息类型**: 真实信息、虚假信息、未知信息
- **智能体类型**: 普通、验证者、传播者

### 传播机制

1. **信息生成**: 随机生成不同类型的信息
2. **初始传播**: 部分智能体接收初始信息
3. **传播阶段**: 智能体之间传播信息
4. **验证阶段**: 验证者智能体验证信息
5. **结果统计**: 统计信息传播范围和准确性

## 配置说明

配置文件: `experiments/configs/info_spreading_config.json`

### 主要参数

```json
{
    "num_agents": 50,
    "network_type": "small_world",
    "num_rounds": 8,
    "initial_infected_ratio": 0.1,
    "verifier_ratio": 0.2,
    "spreader_ratio": 0.3
}
```

### 信息类型

- **真实信息**: 已验证为真的信息
- **虚假信息**: 已验证为假的信息
- **未知信息**: 未经验证的信息

### 智能体类型

- **普通智能体**: 接收和传播信息
- **验证者**: 能够验证信息真伪
- **传播者**: 更积极地传播信息

## 运行实验

### 基础版本

```bash
cd experiments
python info_spreading.py
```

### LLM 增强版本

```bash
python info_spreading_llm.py --agents 50 --rounds 8
```

### 对比实验

运行基线和增强版本对比：

```python
from experiments import run_info_spreading_experiment

# 基线版本
baseline = run_info_spreading_experiment({
    'use_advanced_memory': False
})

# 增强版本
enhanced = run_info_spreading_experiment({
    'use_advanced_memory': True,
    'use_dynamic_reflection': True
})
```

## 结果分析

### 输出指标

- **传播范围**: 信息到达的智能体比例
- **准确性**: 正确识别信息类型的比例
- **虚假信息比例**: 虚假信息的传播比例
- **验证率**: 信息被验证的比例
- **传播速度**: 信息传播的速度
- **网络影响**: 网络拓扑对传播的影响

### 结果文件

- `info_spreading_results.json`: 基础实验结果
- `info_spreading_llm_comparison.json`: LLM 版本对比结果

## 网络拓扑影响

### 小世界网络

- 高聚类系数
- 短平均路径长度
- 信息传播速度快

### 随机网络

- 低聚类系数
- 随机连接
- 传播路径多样

### 无标度网络

- 幂律度分布
- 枢纽节点影响大
- 传播不均匀

## 最佳实践

1. **网络选择**: 根据研究问题选择合适的网络类型
2. **参数调优**: 调整传播概率和验证率
3. **初始条件**: 设置合理的初始感染比例
4. **多轮运行**: 多次运行取平均值

## 常见问题

**Q: 信息传播过快？**

A: 降低传播概率或增加验证者比例。

**Q: 虚假信息过多？**

A: 增加验证者比例或提高验证准确性。

**Q: 网络影响不明显？**

A: 尝试不同的网络拓扑或调整网络参数。

