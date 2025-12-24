# 实验结果

本目录存放所有实验的运行结果。

## 目录结构

```
results/
├── election/              # 选举投票实验
│   ├── baseline.json      # 基础场景结果
│   ├── llm.json           # LLM 驱动结果
│   └── multi_model.json   # 多模型对比
├── resource/              # 资源分配实验
│   ├── baseline.json      # 基础场景结果
│   └── llm_comparison.json # LLM 对比结果
├── info_spreading/        # 信息传播实验
│   ├── baseline.json      # 基础场景结果
│   └── llm_comparison.json # LLM 对比结果
└── comparisons/           # 对比实验
    └── cot_vs_tot.json    # CoT vs ToT 对比
```

## 命名规范

- `baseline.json` - 基础规则型场景结果
- `llm.json` / `llm_comparison.json` - LLM 驱动版本结果
- `multi_model.json` - 多个 LLM 模型的对比结果

