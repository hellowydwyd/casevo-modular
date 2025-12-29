# Casevo 实验模块

本目录包含 Casevo 框架的社会模拟实验，**严格按照 Proposal 设计 + 最小化改进**。

## 📊 实验总览

### 改进后实验规模

| 场景 | 对照组 | 每组运行 | 小计 | 智能体规模 | LLM 调用/次 |
|------|--------|---------|------|-----------|------------|
| 选举投票 | 5 组 | 3 次 | 15 次 | 30 选民 × 6 轮 | ~300 次 |
| 资源分配 | 5 组 | 3 次 | 15 次 | 20 智能体 × 5 轮 | ~200 次 |
| 信息传播 | 5 组 | 3 次 | 15 次 | 50 节点 × 10 轮 | ~400 次 |
| **总计** | **15 组** | | **45 次** | | **~13500 次** |

### 预计运行时间

| 运行模式 | 串行时间 | 并行时间（5 Key） | API 成本 |
|---------|---------|------------------|---------|
| 快速验证（每组 1 次） | 2-3 小时 | 30-40 分钟 | $2-4 |
| **标准实验（每组 3 次）** | **6-10 小时** | **1.5-2 小时** | **$10-15** |

### 五组对照配置（含消融实验）

| 组别 | use_tot | use_enhanced_memory | use_dynamic_reflection | use_collaborative | 说明 |
|------|---------|---------------------|------------------------|-------------------|------|
| 基线组（CoT） | ❌ | ❌ | ❌ | ❌ | 原始链式推理 |
| 优化组 A（ToT） | ✅ | ❌ | ❌ | ❌ | 仅多路径推理 |
| 消融组 A1 | ✅ | ✅ | ❌ | ❌ | ToT + 增强记忆 |
| 消融组 A2 | ✅ | ❌ | ✅ | ❌ | ToT + 动态反思 |
| 优化组 B（全部） | ✅ | ✅ | ✅ | ✅ | 全部优化 |

## 🎯 最小化改进说明

本次更新针对实验设计的不足进行了最小化改进：

### 1. 随机种子控制 ✅
- 基础种子：`base_seed = 42`
- 每次运行种子：`42 + run_index`
- 同时设置 `random.seed()` 和 `np.random.seed()`
- 确保实验可复现

### 2. 运行次数增加到 3 次 ✅
- 每组实验运行 3 次
- 可计算均值和标准差
- 支持基本统计检验

### 3. 消融实验 ✅
- 新增 `ablation_tot_memory`：ToT + 增强记忆
- 新增 `ablation_tot_reflection`：ToT + 动态反思
- 可分离各优化组件的独立贡献

## 目录结构

```
experiments/
├── election/           # 选举投票实验（30选民、6轮辩论）
│   └── scenario.py     # LLM 智能体场景
├── resource/           # 资源分配实验（20智能体、500资源）
│   └── scenario.py     # LLM 协商场景
├── info_spreading/     # 信息传播实验（50节点无标度网络）
│   └── scenario.py     # 规则型基线 + 增强评估
├── comparisons/        # 对比实验（含消融实验）
│   ├── cot_vs_tot.py           # CoT vs ToT 推理对比
│   ├── memory_optimization.py  # 记忆系统优化对比
│   └── full_optimization.py    # 完整优化对比（五组对照）
├── utils/              # 工具模块
├── configs/            # 配置文件
└── results/            # 实验结果输出
```

## Proposal 对应关系

| Proposal 实验设计 | 实验文件 | 说明 |
|------------------|----------|------|
| 选举投票场景 | `election/scenario.py` | ✅ 已简化为30选民 |
| 资源分配场景 | `resource/scenario.py` | ✅ 已简化为20智能体 |
| 信息传播场景 | `info_spreading/scenario.py` | ✅ 已简化为50节点 |
| 基线组（原始 CoT） | `comparisons/full_optimization.py` | ✅ |
| 优化组 A（仅 ToT） | `comparisons/full_optimization.py` | ✅ |
| **消融组 A1（ToT + 记忆）** | `comparisons/full_optimization.py` | ✅ 新增 |
| **消融组 A2（ToT + 反思）** | `comparisons/full_optimization.py` | ✅ 新增 |
| 优化组 B（全部优化） | `comparisons/full_optimization.py` | ✅ |
| 每组运行 3 次 | 默认 `--runs 3` | ✅ 从10次简化 |

## 环境准备

```powershell
# 1. 激活 conda 环境
conda activate d2l

# 2. 设置 PYTHONPATH（Windows PowerShell）
$env:PYTHONPATH = "D:\DEVELOPE\Casevo"

# 或者在 Linux/Mac 上
export PYTHONPATH=/path/to/Casevo
```

## 运行实验

### 🎯 核心对比实验（含消融实验）

#### 完整优化对比（五组对照，每组 3 次）

```powershell
# 运行所有场景的完整对比（推荐）
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment all

# 仅运行选举场景对比
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment election

# 仅运行资源分配场景对比
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment resource

# 仅运行信息传播场景对比
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment info
```

#### 快速验证（每组 1 次）

```powershell
python -u experiments/comparisons/full_optimization.py --runs 1 --experiment election
```

#### CoT vs ToT 推理机制对比

```powershell
python -u experiments/comparisons/cot_vs_tot.py --voters 30 --rounds 3
```

### 📊 单场景实验

#### 选举投票实验

```powershell
python -u experiments/election/scenario.py
```

#### 资源分配实验

```powershell
python -u experiments/resource/scenario.py
```

#### 信息传播实验

```powershell
python -u experiments/info_spreading/scenario.py
```

## 实验配置

### 选举投票场景

| 参数 | 当前值 | Proposal值 | 说明 |
|------|--------|-----------|------|
| 选民数量 | 30 | 101 | 简化版 |
| 网络类型 | 小世界网络 | 小世界网络 | ✅ |
| 辩论轮次 | 6 | 6 | ✅ 完整 |
| LLM 温度 | 0.7 | 0.7 | ✅ |
| 反思阈值 | 0.6 | 0.6 | ✅ |

### 资源分配场景

| 参数 | 当前值 | Proposal值 | 说明 |
|------|--------|-----------|------|
| 智能体数量 | 20 | 50 | 简化版 |
| 总资源量 | 500 | 1000 | 简化版 |
| 需求范围 | 15-30 单位 | 15-30 单位 | ✅ |
| 最大协商轮次 | 5 | 10 | 简化版 |

### 信息传播场景

| 参数 | 当前值 | Proposal值 | 说明 |
|------|--------|-----------|------|
| 节点数量 | 50 | 200 | 简化版 |
| 初始感染率 | 10% | 10% | ✅ |
| 虚假信息比例 | 30% | 30% | ✅ |
| 传播轮次 | 10 | 20 | 简化版 |

## 五组对照实验设计

按照 Proposal 要求 + 消融实验，设置五组对照：

| 组别 | 配置 | 分析目的 |
|------|------|---------|
| **基线组** | 原始 CoT | 基准性能 |
| **优化组 A** | 仅 ToT | ToT 的独立效果 |
| **消融组 A1** | ToT + 记忆 | 记忆的增量贡献 |
| **消融组 A2** | ToT + 反思 | 反思的增量贡献 |
| **优化组 B** | 全部优化 | 整体效果 |

每组独立运行 **3 次**，使用固定随机种子（42, 43, 44），确保可复现。

## 评估指标（四个维度）

### 1. 决策质量 (DecisionQualityMetrics)

| 指标 | 说明 | 计算方法 |
|------|------|---------|
| 准确率 | 决策与政治倾向一致性 | 政治对齐评估 |
| 一致性 | 决策与历史的一致程度 | 最近 3 次决策相似度 |
| 置信度校准 | 置信度与准确率匹配度 | `1 - |confidence - accuracy|` |

### 2. 推理能力 (ReasoningMetrics)

| 指标 | 说明 | 计算方法 |
|------|------|---------|
| 平均深度 | 推理链平均深度 | CoT=步骤数, ToT=树深度 |
| 分支数 | 探索的推理路径数 | ToT 特有指标 |
| 连贯性分数 | 推理连贯程度 | 基于最终得分 |
| 剪枝率 | ToT 剪枝效率 | `剪枝分支 / 总分支` |

### 3. 计算效率 (PerformanceTracker)

| 指标 | 说明 | 计算方法 |
|------|------|---------|
| 总调用次数 | LLM API 调用次数 | 直接计数 |
| 平均响应时间 | 单次调用平均耗时 | 毫秒 |
| 效率分数 | 响应速度评分 | 2000ms理想，10000ms零分 |

### 4. 社会效应 (SocialEffectMetrics)

| 指标 | 说明 | 计算方法 |
|------|------|---------|
| 共识度指数 | 群体意见一致程度 | **基于信息熵**（已修复） |
| 极化指数 | 意见两极分化程度 | 标准差/最大值 |
| 意见稳定性 | 轮间意见变化程度 | 相邻轮差异 |

## 结果输出

结果自动保存到 `experiments/results/` 目录：

```
results/
├── comparisons/
│   ├── full_optimization_{timestamp}.json   # 完整优化对比结果
│   ├── analysis_{timestamp}.md              # 分析报告
│   └── ...
├── memories/                                 # 持久化记忆（ChromaDB）
│   └── {timestamp}/
│       ├── election_baseline_cot_run1/      # 每个实验的向量存储
│       ├── election_optimized_tot_only_run1/
│       └── ...
├── thoughts/                                 # 智能体思考过程日志
│   ├── thoughts_{timestamp}.jsonl           # 实时日志（每行一条记录）
│   └── thoughts_full_{timestamp}.json       # 完整结构化日志
├── election/
├── resource/
└── info_spreading/
```

### 思考过程日志格式

每条思考记录包含：

```json
{
  "agent_id": "voter_1",
  "agent_name": "Alice",
  "round_num": 2,
  "timestamp": "2024-12-27T15:30:00",
  "input_context": "当前支持Biden，邻居建议Trump...",
  "memories_retrieved": ["昨天的辩论内容...", "与邻居的讨论..."],
  "reasoning_type": "tot",
  "reasoning_steps": ["考虑经济政策...", "评估社会议题..."],
  "tot_branches": [...],
  "reflection_triggered": false,
  "decision": "Biden",
  "confidence": 0.75,
  "reasoning_summary": "探索12节点，最佳分数0.72"
}
```

### 结果格式

```json
{
  "election": {
    "baseline_cot": { "avg_biden_support": 12.0, "std_biden_support": 0.0, ... },
    "optimized_tot_only": { ... },
    "ablation_tot_memory": { ... },
    "ablation_tot_reflection": { ... },
    "optimized_full": { ... }
  },
  "resource": { ... },
  "info_spreading": { ... },
  "experiment_config": {
    "base_seed": 42,
    "num_runs": 3,
    "timestamp": "..."
  }
}
```

## 📋 完整实验列表

### 选举投票场景（15 次运行）

| 序号 | 实验组 | 配置 | 预计时间 |
|------|--------|------|---------|
| 1-3 | 基线组 | CoT, 无优化 | 10-15 分钟 |
| 4-6 | 优化组 A | ToT 多路径推理 | 15-25 分钟 |
| 7-9 | 消融组 A1 | ToT + 增强记忆 | 20-30 分钟 |
| 10-12 | 消融组 A2 | ToT + 动态反思 | 20-30 分钟 |
| 13-15 | 优化组 B | 全部优化 | 25-40 分钟 |

### 资源分配场景（15 次运行）

| 序号 | 实验组 | 配置 | 预计时间 |
|------|--------|------|---------|
| 1-3 | 基线组 | 独立决策 | 15-25 分钟 |
| 4-6 | 优化组 A | ToT 推理 | 25-40 分钟 |
| 7-9 | 消融组 A1 | ToT + 增强记忆 | 30-45 分钟 |
| 10-12 | 消融组 A2 | ToT + 动态反思 | 30-45 分钟 |
| 13-15 | 优化组 B | 全部优化 | 40-60 分钟 |

### 信息传播场景（15 次运行）

| 序号 | 实验组 | 配置 | 预计时间 |
|------|--------|------|---------|
| 1-3 | 基线组 | 规则评估 | 5-10 分钟 |
| 4-6 | 优化组 A | ToT 评估 | 20-40 分钟 |
| 7-9 | 消融组 A1 | ToT + 增强记忆 | 25-45 分钟 |
| 10-12 | 消融组 A2 | ToT + 动态反思 | 25-45 分钟 |
| 13-15 | 优化组 B | 全部优化 | 30-50 分钟 |

## 🚀 推荐运行顺序

```powershell
# Step 1: 快速验证（确保代码正确）
python -u experiments/comparisons/full_optimization.py --runs 1 --experiment election

# Step 2: 单场景完整测试
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment election

# Step 3: 其他场景（可分开运行）
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment resource
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment info

# Step 4: 或者一次运行全部
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment all
```

## 🚀 多API Key并行加速（5倍速）

如果你有多个API Key，可以同时开5个终端并行运行不同实验组：

### 准备工作

准备5个API Key，例如：
- `KEY1`: sk-xxxx1111
- `KEY2`: sk-xxxx2222
- `KEY3`: sk-xxxx3333
- `KEY4`: sk-xxxx4444
- `KEY5`: sk-xxxx5555

### 并行运行命令

同时开5个PowerShell终端，分别运行：

```powershell
# 终端1: 基线组
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment election --group baseline_cot --api-key sk-xxxx1111

# 终端2: ToT组
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment election --group optimized_tot_only --api-key sk-xxxx2222

# 终端3: 消融组A1
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment election --group ablation_tot_memory --api-key sk-xxxx3333

# 终端4: 消融组A2
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment election --group ablation_tot_reflection --api-key sk-xxxx4444

# 终端5: 全优化组
python -u experiments/comparisons/full_optimization.py --runs 3 --experiment election --group optimized_full --api-key sk-xxxx5555
```

### 结果文件

每个组会独立保存结果：
```
results/comparisons/
├── election_baseline_cot_{timestamp}.json
├── election_optimized_tot_only_{timestamp}.json
├── election_ablation_tot_memory_{timestamp}.json
├── election_ablation_tot_reflection_{timestamp}.json
└── election_optimized_full_{timestamp}.json
```

### 预计时间对比

| 运行方式 | 选举场景 | 全部场景 |
|---------|---------|---------|
| 串行（1个key） | 2-3小时 | 6-10小时 |
| **并行（5个key）** | **30-40分钟** | **1.5-2小时** |

## 注意事项

1. **API 费用**：LLM 实验会消耗 API 额度，完整实验（3次 × 5组 × 3场景）约 $10-15（gpt-4o-mini）
2. **运行时间**：完整对比实验约 6-10 小时
3. **日志输出**：使用 `python -u` 参数强制实时输出日志
4. **随机种子**：使用固定种子（42, 43, 44）确保可重复性
5. **断点续传**：每个实验组完成后立即保存，可中断后继续

## API 配置

API 密钥配置在 `src/casevo/llm/openai.py` 中：

```bash
# 在 .env 文件中配置
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

## 📈 预期结果摘要

完成全部 45 次实验后，预期可得到以下对比数据：

| 评估维度 | 基线组 | ToT | ToT+记忆 | ToT+反思 | 全优化 |
|---------|--------|-----|----------|----------|--------|
| 决策一致性 | ~60% | ~70% | ~72% | ~73% | ~80% |
| 推理深度 | 1 层 | 3-5 层 | 3-5 层 | 3-5 层 | 3-5 层 |
| 群体共识度 | ~40% | ~50% | ~52% | ~55% | ~65% |
| 计算时间 | 1x | 2-3x | 2.5-3.5x | 2.5-3.5x | 3-5x |

消融分析可揭示：
- **记忆贡献**：ToT+记忆 vs ToT → 记忆系统的增量效果
- **反思贡献**：ToT+反思 vs ToT → 反思机制的增量效果
- **协同效应**：全优化 vs (ToT+记忆+反思) → 协同决策的额外贡献

---

**总计：45 次实验运行，预计 6-10 小时，API 成本 ~$10-15（gpt-4o-mini）**
