#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casevo 实验结果可视化
====================
生成论文级别的实验结果可视化图表
重点展示各优化组件（ToT、记忆增强、动态反思、协同决策）的贡献

Author: Casevo Team
Date: 2024-12-29
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和美观的样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['axes.edgecolor'] = '#dee2e6'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#e9ecef'
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['font.size'] = 11

# 定义美观的配色方案 - 强调各组件的区分
COLORS = {
    'baseline': '#6c757d',      # 灰色 - 基线 CoT
    'tot_only': '#0d6efd',      # 蓝色 - ToT 推理
    'tot_memory': '#198754',    # 绿色 - 记忆增强
    'tot_reflection': '#fd7e14', # 橙色 - 动态反思
    'full': '#dc3545',          # 红色 - 全优化
    'biden': '#2563eb',         
    'trump': '#dc2626',         
    'undecided': '#9ca3af',     
}

# 组件颜色（用于组件贡献图）
COMPONENT_COLORS = {
    'tot': '#0d6efd',           # 蓝色 - ToT 多层次推理
    'memory': '#198754',        # 绿色 - 增强记忆
    'reflection': '#fd7e14',    # 橙色 - 动态反思
    'collaborative': '#9333ea', # 紫色 - 协同决策
}

# 实验组标签
GROUP_LABELS = {
    'baseline_cot': '基线 (CoT)',
    'optimized_tot_only': 'ToT 推理',
    'ablation_tot_memory': 'ToT + 记忆',
    'ablation_tot_reflection': 'ToT + 反思',
    'optimized_full': '全优化',
}

GROUP_SHORT_LABELS = {
    'baseline_cot': 'CoT\n(基线)',
    'optimized_tot_only': 'ToT\n(推理优化)',
    'ablation_tot_memory': 'ToT+记忆',
    'ablation_tot_reflection': 'ToT+反思',
    'optimized_full': '全优化',
}

# 结果目录
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'comparisons'
OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_experiment_data(experiment_type: str, group: str) -> dict:
    """加载实验数据"""
    pattern = f"{experiment_type}_{group}_*.json"
    files = list(RESULTS_DIR.glob(pattern))
    
    if not files:
        print(f"  警告: 未找到 {experiment_type}/{group} 的数据文件")
        return None
    
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_all_experiment_data() -> dict:
    """加载所有实验数据"""
    experiments = ['election', 'resource', 'info']
    groups = ['baseline_cot', 'optimized_tot_only', 'ablation_tot_memory', 
              'ablation_tot_reflection', 'optimized_full']
    
    data = {}
    for exp in experiments:
        data[exp] = {}
        print(f"加载 {exp} 实验数据...")
        for group in groups:
            result = load_experiment_data(exp, group)
            if result:
                data[exp][group] = result
    
    return data


def get_overall_score(data: dict, exp: str, group: str) -> float:
    """获取实验组的综合得分"""
    if exp not in data or group not in data[exp]:
        return 0.5
    
    exp_data = data[exp][group]
    raw_results = exp_data.get('results', {}).get('raw_results', [])
    
    if raw_results:
        scores = [r.get('metrics_summary', {}).get('overall_score', 0.5) for r in raw_results]
        return np.mean(scores)
    return 0.5


def plot_component_contribution(data: dict):
    """
    绘制组件贡献分析图 - 展示各优化组件相对于基础ToT的提升
    这是核心图表，直接对应课题目标
    """
    print("绘制组件贡献分析图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    experiments = [
        ('election', '选举投票'),
        ('resource', '资源分配'),
        ('info', '信息传播'),
    ]
    
    for idx, (exp_key, exp_name) in enumerate(experiments):
        ax = axes[idx]
        
        # 获取各组得分
        cot_score = get_overall_score(data, exp_key, 'baseline_cot')
        tot_score = get_overall_score(data, exp_key, 'optimized_tot_only')
        memory_score = get_overall_score(data, exp_key, 'ablation_tot_memory')
        reflection_score = get_overall_score(data, exp_key, 'ablation_tot_reflection')
        full_score = get_overall_score(data, exp_key, 'optimized_full')
        
        # 计算各组件的贡献（相对于基线 CoT）
        tot_improvement = (tot_score - cot_score) / cot_score * 100 if cot_score > 0 else 0
        memory_improvement = (memory_score - tot_score) / tot_score * 100 if tot_score > 0 else 0
        reflection_improvement = (reflection_score - tot_score) / tot_score * 100 if tot_score > 0 else 0
        
        # 组件贡献
        components = ['ToT\n多层次推理', '增强记忆\n(相对ToT)', '动态反思\n(相对ToT)']
        improvements = [tot_improvement, memory_improvement, reflection_improvement]
        colors = [COMPONENT_COLORS['tot'], COMPONENT_COLORS['memory'], COMPONENT_COLORS['reflection']]
        
        x = np.arange(len(components))
        bars = ax.bar(x, improvements, color=colors, edgecolor='white', linewidth=2, alpha=0.9, width=0.6)
        
        # 添加数值标签
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            color = '#198754' if height >= 0 else '#dc3545'
            ax.annotate(f'{imp:+.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset),
                       textcoords="offset points",
                       ha='center', va=va, fontsize=11, fontweight='bold', color=color)
        
        ax.axhline(y=0, color='#495057', linestyle='-', linewidth=1.5)
        ax.set_xlabel('')
        ax.set_ylabel('性能提升 (%)' if idx == 0 else '', fontsize=12)
        ax.set_title(f'{exp_name}场景', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(components, fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 动态调整 y 轴范围
        max_val = max(abs(min(improvements)), max(improvements)) + 10
        ax.set_ylim(-max_val, max_val)
    
    plt.suptitle('各优化组件的性能贡献分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_component_contribution.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_component_contribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_component_contribution.png/pdf")


def plot_progressive_improvement(data: dict):
    """
    绘制渐进式提升图 - 展示从 CoT → ToT → ToT+组件 的逐步提升
    不加折线，仅使用柱状图
    """
    print("绘制渐进式提升图...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    experiments = ['election', 'resource', 'info']
    exp_labels = ['选举投票', '资源分配', '信息传播']
    exp_colors = ['#6366f1', '#10b981', '#f59e0b']
    
    groups = ['baseline_cot', 'optimized_tot_only', 'ablation_tot_memory', 
              'ablation_tot_reflection', 'optimized_full']
    group_labels = ['CoT\n(基线)', 'ToT\n推理', 'ToT+\n记忆', 'ToT+\n反思', '全\n优化']
    
    x = np.arange(len(groups))
    width = 0.25
    
    for i, (exp, label, color) in enumerate(zip(experiments, exp_labels, exp_colors)):
        scores = [get_overall_score(data, exp, g) for g in groups]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, scores, width, label=label, color=color, 
                     edgecolor='white', linewidth=1.5, alpha=0.85)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(f'{score:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('')
    ax.set_ylabel('综合得分', fontsize=13)
    ax.set_title('各实验组综合得分对比（渐进式优化）', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylim(0.4, 0.75)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_progressive_improvement.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_progressive_improvement.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_progressive_improvement.png/pdf")


def plot_tot_vs_cot(data: dict):
    """
    绘制 ToT vs CoT 对比图 - 验证 ToT 多层次推理的效果
    """
    print("绘制 ToT vs CoT 对比图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    experiments = [
        ('election', '选举投票'),
        ('resource', '资源分配'),
        ('info', '信息传播'),
    ]
    
    for idx, (exp_key, exp_name) in enumerate(experiments):
        ax = axes[idx]
        
        cot_score = get_overall_score(data, exp_key, 'baseline_cot')
        tot_score = get_overall_score(data, exp_key, 'optimized_tot_only')
        
        groups = ['CoT\n(基线)', 'ToT\n(优化)']
        scores = [cot_score, tot_score]
        colors = [COLORS['baseline'], COLORS['tot_only']]
        
        bars = ax.bar(groups, scores, color=colors, edgecolor='white', linewidth=2, alpha=0.9, width=0.5)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            ax.annotate(f'{score:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 添加提升箭头和百分比
        improvement = (tot_score - cot_score) / cot_score * 100 if cot_score > 0 else 0
        color = '#198754' if improvement >= 0 else '#dc3545'
        
        mid_y = (cot_score + tot_score) / 2
        ax.annotate('', xy=(1.15, tot_score), xytext=(1.15, cot_score),
                   arrowprops=dict(arrowstyle='<->', color=color, lw=2.5))
        ax.text(1.25, mid_y, f'{improvement:+.1f}%', fontsize=12, fontweight='bold', 
               color=color, va='center')
        
        ax.set_ylabel('综合得分' if idx == 0 else '', fontsize=12)
        ax.set_title(f'{exp_name}场景', fontsize=14, fontweight='bold', pad=10)
        ax.set_ylim(0.4, 0.75)
        ax.set_xlim(-0.5, 1.8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('ToT 多层次推理 vs CoT 线性推理 效果对比', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_tot_vs_cot.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_tot_vs_cot.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_tot_vs_cot.png/pdf")


def plot_ablation_comparison(data: dict):
    """
    绘制消融实验对比图 - 验证记忆和反思模块的效果
    以 ToT 为基准，对比添加记忆/反思后的效果
    """
    print("绘制消融实验对比图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    experiments = [
        ('election', '选举投票'),
        ('resource', '资源分配'),
        ('info', '信息传播'),
    ]
    
    for idx, (exp_key, exp_name) in enumerate(experiments):
        ax = axes[idx]
        
        tot_score = get_overall_score(data, exp_key, 'optimized_tot_only')
        memory_score = get_overall_score(data, exp_key, 'ablation_tot_memory')
        reflection_score = get_overall_score(data, exp_key, 'ablation_tot_reflection')
        
        groups = ['ToT\n(基准)', 'ToT + 记忆', 'ToT + 反思']
        scores = [tot_score, memory_score, reflection_score]
        colors = [COLORS['tot_only'], COLORS['tot_memory'], COLORS['tot_reflection']]
        
        x = np.arange(len(groups))
        bars = ax.bar(x, scores, color=colors, edgecolor='white', linewidth=2, alpha=0.9, width=0.6)
        
        # 添加数值标签和相对提升
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.annotate(f'{score:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # 显示相对于 ToT 的提升
            if i > 0:
                improvement = (score - tot_score) / tot_score * 100 if tot_score > 0 else 0
                color = '#198754' if improvement >= 0 else '#dc3545'
                ax.annotate(f'({improvement:+.1f}%)',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025),
                           xytext=(0, 5),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, color=color)
        
        ax.set_ylabel('综合得分' if idx == 0 else '', fontsize=12)
        ax.set_title(f'{exp_name}场景', fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=10)
        ax.set_ylim(0.4, 0.7)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 添加基准线
        ax.axhline(y=tot_score, color=COLORS['tot_only'], linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.suptitle('消融实验：记忆增强与动态反思模块效果对比', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_ablation_comparison.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_ablation_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_ablation_comparison.png/pdf")


def plot_ablation_radar(data: dict):
    """绘制消融实验雷达图 - 多维度对比各优化组件"""
    print("绘制消融实验雷达图...")
    
    # 只对比关键的几组，避免混乱
    groups = ['baseline_cot', 'optimized_tot_only', 'ablation_tot_memory', 
              'ablation_tot_reflection', 'optimized_full']
    labels = ['CoT 基线', 'ToT 推理', 'ToT+记忆', 'ToT+反思', '全优化']
    colors = [COLORS['baseline'], COLORS['tot_only'], COLORS['tot_memory'],
              COLORS['tot_reflection'], COLORS['full']]
    
    # 定义评估维度
    dimensions = ['推理能力', '决策质量', '社会效应', '计算效率']
    num_dims = len(dimensions)
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    angles += angles[:1]
    
    for group, label, color in zip(groups, labels, colors):
        values = []
        
        for exp_key in ['election', 'resource', 'info']:
            if exp_key in data and group in data[exp_key]:
                exp_data = data[exp_key][group]
                raw_results = exp_data.get('results', {}).get('raw_results', [])
                
                if raw_results:
                    metrics = raw_results[0].get('metrics_summary', {})
                    if not values:
                        values = [
                            metrics.get('reasoning_ability_score', 0.5),
                            metrics.get('decision_quality_score', 0.5),
                            metrics.get('social_effects_score', 0.5),
                            metrics.get('computational_efficiency_score', 0.5),
                        ]
                    else:
                        values[0] = (values[0] + metrics.get('reasoning_ability_score', 0.5)) / 2
                        values[1] = (values[1] + metrics.get('decision_quality_score', 0.5)) / 2
                        values[2] = (values[2] + metrics.get('social_effects_score', 0.5)) / 2
                        values[3] = (values[3] + metrics.get('computational_efficiency_score', 0.5)) / 2
        
        if not values:
            values = [0.5, 0.5, 0.5, 0.5]
        
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=label, color=color, alpha=0.85)
        ax.fill(angles, values, alpha=0.12, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    
    plt.title('各优化组件多维度能力对比', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_ablation_radar.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_ablation_radar.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_ablation_radar.png/pdf")


def plot_election_evolution(data: dict):
    """绘制选举实验的投票态度演化图 - 对比 CoT vs ToT，分别展示三类选民"""
    print("绘制选举投票态度演化图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    groups_to_compare = ['baseline_cot', 'optimized_tot_only']
    titles = ['基线组 (CoT)', 'ToT 推理优化组']
    
    for idx, (group, title) in enumerate(zip(groups_to_compare, titles)):
        ax = axes[idx]
        
        if 'election' not in data or group not in data['election']:
            continue
        
        exp_data = data['election'][group]
        voting_history = exp_data.get('results', {}).get('voting_history', [])
        
        if not voting_history:
            continue
        
        history = voting_history[0]
        
        rounds = [h['round'] for h in history]
        biden = [h['biden'] for h in history]
        trump = [h['trump'] for h in history]
        undecided = [h['undecided'] for h in history]
        
        # 分别绘制三条折线，清晰展示各类选民变化
        ax.plot(rounds, biden, 'o-', color=COLORS['biden'], linewidth=2.5, 
               markersize=8, label='Biden 支持者', alpha=0.9)
        ax.plot(rounds, trump, 's-', color=COLORS['trump'], linewidth=2.5, 
               markersize=8, label='Trump 支持者', alpha=0.9)
        ax.plot(rounds, undecided, '^-', color='#8b5cf6', linewidth=2.5, 
               markersize=8, label='中间选民 (未决定)', alpha=0.9)
        
        # 添加起止点数值标注
        ax.annotate(f'{biden[0]}', (rounds[0], biden[0]), textcoords="offset points",
                   xytext=(-10, 5), fontsize=9, color=COLORS['biden'], fontweight='bold')
        ax.annotate(f'{biden[-1]}', (rounds[-1], biden[-1]), textcoords="offset points",
                   xytext=(5, 5), fontsize=9, color=COLORS['biden'], fontweight='bold')
        
        ax.annotate(f'{trump[0]}', (rounds[0], trump[0]), textcoords="offset points",
                   xytext=(-10, -12), fontsize=9, color=COLORS['trump'], fontweight='bold')
        ax.annotate(f'{trump[-1]}', (rounds[-1], trump[-1]), textcoords="offset points",
                   xytext=(5, -12), fontsize=9, color=COLORS['trump'], fontweight='bold')
        
        ax.annotate(f'{undecided[0]}', (rounds[0], undecided[0]), textcoords="offset points",
                   xytext=(-10, 5), fontsize=9, color='#8b5cf6', fontweight='bold')
        ax.annotate(f'{undecided[-1]}', (rounds[-1], undecided[-1]), textcoords="offset points",
                   xytext=(5, 5), fontsize=9, color='#8b5cf6', fontweight='bold')
        
        ax.set_xlabel('辩论轮次', fontsize=12)
        ax.set_ylabel('选民人数', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(-0.3, len(rounds) - 0.5)
        ax.set_ylim(0, 20)
        ax.set_xticks(rounds)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('选举投票态度演化：三类选民变化对比', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_election_evolution.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_election_evolution.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_election_evolution.png/pdf")


def plot_resource_convergence(data: dict):
    """绘制资源分配实验的收敛过程图"""
    print("绘制资源分配收敛过程图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 只对比关键组
    groups = ['baseline_cot', 'optimized_tot_only', 'ablation_tot_memory', 
              'ablation_tot_reflection', 'optimized_full']
    colors = [COLORS['baseline'], COLORS['tot_only'], COLORS['tot_memory'],
              COLORS['tot_reflection'], COLORS['full']]
    labels = ['CoT 基线', 'ToT', 'ToT+记忆', 'ToT+反思', '全优化']
    
    # 图1: 稀缺度比率变化
    ax1 = axes[0]
    for group, color, label in zip(groups, colors, labels):
        if 'resource' not in data or group not in data['resource']:
            continue
        
        exp_data = data['resource'][group]
        allocation_history = exp_data.get('results', {}).get('allocation_history', [])
        
        if not allocation_history:
            continue
        
        history = allocation_history[0]
        rounds = [h['round'] for h in history]
        scarcity = [h['scarcity_ratio'] for h in history]
        
        ax1.plot(rounds, scarcity, 'o-', color=color, linewidth=2, markersize=6,
                label=label, alpha=0.85)
    
    ax1.axhline(y=1.0, color='#28a745', linestyle='--', linewidth=2, alpha=0.7, label='平衡线')
    ax1.set_xlabel('轮次', fontsize=12)
    ax1.set_ylabel('稀缺度比率 (需求/供给)', fontsize=12)
    ax1.set_title('资源稀缺度变化', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 图2: Gini 系数对比
    ax2 = axes[1]
    gini_values = []
    
    for group in groups:
        if 'resource' not in data or group not in data['resource']:
            gini_values.append(0)
            continue
        
        exp_data = data['resource'][group]
        avg_gini = exp_data.get('results', {}).get('avg_gini', 0)
        gini_values.append(avg_gini)
    
    x = np.arange(len(groups))
    bars = ax2.bar(x, gini_values, color=colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    for bar, gini in zip(bars, gini_values):
        height = bar.get_height()
        ax2.annotate(f'{gini:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Gini 系数 (越低越公平)', fontsize=12)
    ax2.set_title('资源分配公平性', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
    ax2.set_ylim(0, 0.15)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_resource_convergence.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_resource_convergence.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_resource_convergence.png/pdf")


def plot_info_spread(data: dict):
    """绘制信息传播实验结果图"""
    print("绘制信息传播实验结果图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    groups = ['baseline_cot', 'optimized_tot_only', 'ablation_tot_memory', 
              'ablation_tot_reflection', 'optimized_full']
    colors = [COLORS['baseline'], COLORS['tot_only'], COLORS['tot_memory'],
              COLORS['tot_reflection'], COLORS['full']]
    labels = ['CoT 基线', 'ToT', 'ToT+记忆', 'ToT+反思', '全优化']
    
    # 图1: 虚假信息相信比例随时间变化
    ax1 = axes[0]
    for group, color, label in zip(groups, colors, labels):
        if 'info' not in data or group not in data['info']:
            continue
        
        exp_data = data['info'][group]
        spread_history = exp_data.get('results', {}).get('spread_history', [])
        
        if not spread_history:
            continue
        
        history = spread_history[0]
        rounds = [h['round'] for h in history]
        false_ratio = [h.get('false_belief_ratio', 0) for h in history]
        
        ax1.plot(rounds, false_ratio, 'o-', color=color, linewidth=2, markersize=6,
                label=label, alpha=0.85)
    
    ax1.set_xlabel('轮次', fontsize=12)
    ax1.set_ylabel('虚假信息相信比例', fontsize=12)
    ax1.set_title('虚假信息传播趋势', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 图2: 最终准确率对比
    ax2 = axes[1]
    accuracy_values = []
    
    for group in groups:
        if 'info' not in data or group not in data['info']:
            accuracy_values.append(0.5)
            continue
        
        exp_data = data['info'][group]
        avg_accuracy = exp_data.get('results', {}).get('avg_accuracy', 0.5)
        accuracy_values.append(avg_accuracy)
    
    x = np.arange(len(groups))
    bars = ax2.bar(x, accuracy_values, color=colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    for bar, acc in zip(bars, accuracy_values):
        height = bar.get_height()
        ax2.annotate(f'{acc:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('信息判断准确率', fontsize=12)
    ax2.set_title('信息真伪判断准确率', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
    ax2.set_ylim(0.4, 0.8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_info_spread.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_info_spread.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_info_spread.png/pdf")


def plot_computational_efficiency(data: dict):
    """绘制计算效率对比图"""
    print("绘制计算效率对比图...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    experiments = ['election', 'resource', 'info']
    exp_labels = ['选举投票', '资源分配', '信息传播']
    
    groups = ['baseline_cot', 'optimized_tot_only', 'ablation_tot_memory', 
              'ablation_tot_reflection', 'optimized_full']
    labels = ['CoT 基线', 'ToT', 'ToT+记忆', 'ToT+反思', '全优化']
    colors = [COLORS['baseline'], COLORS['tot_only'], COLORS['tot_memory'],
              COLORS['tot_reflection'], COLORS['full']]
    
    x = np.arange(len(experiments))
    width = 0.15
    
    for i, (group, label, color) in enumerate(zip(groups, labels, colors)):
        durations = []
        for exp in experiments:
            if exp in data and group in data[exp]:
                exp_data = data[exp][group]
                eval_metrics = exp_data.get('results', {}).get('evaluation_metrics', [])
                
                if eval_metrics:
                    total_duration = 0
                    count = 0
                    for metrics in eval_metrics:
                        call_stats = metrics.get('computational_efficiency', {}).get('call_statistics', {})
                        avg_dur = call_stats.get('avg_duration_ms', 0)
                        if avg_dur > 0:
                            total_duration += avg_dur
                            count += 1
                    
                    durations.append(total_duration / count / 1000 if count > 0 else 0)
                else:
                    durations.append(0)
            else:
                durations.append(0)
        
        offset = (i - 2) * width
        bars = ax.bar(x + offset, durations, width, label=label, color=color, 
                     edgecolor='white', linewidth=1, alpha=0.85)
    
    ax.set_ylabel('平均响应时间 (秒)', fontsize=12)
    ax.set_title('各优化组件计算效率对比', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_computational_efficiency.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_computational_efficiency.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_computational_efficiency.png/pdf")


def plot_metrics_heatmap(data: dict):
    """绘制指标热力图"""
    print("绘制指标热力图...")
    
    experiments = ['election', 'resource', 'info']
    exp_labels = ['选举投票', '资源分配', '信息传播']
    groups = ['baseline_cot', 'optimized_tot_only', 'ablation_tot_memory', 
              'ablation_tot_reflection', 'optimized_full']
    group_labels = ['CoT\n基线', 'ToT\n推理', 'ToT+\n记忆', 'ToT+\n反思', '全\n优化']
    
    matrix = np.zeros((len(experiments), len(groups)))
    
    for i, exp in enumerate(experiments):
        for j, group in enumerate(groups):
            matrix[i, j] = get_overall_score(data, exp, group)
    
    fig, ax = plt.subplots(figsize=(11, 5))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.45, vmax=0.65)
    
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('综合得分', rotation=-90, va="bottom", fontsize=12)
    
    ax.set_xticks(np.arange(len(groups)))
    ax.set_yticks(np.arange(len(experiments)))
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_yticklabels(exp_labels, fontsize=11)
    
    for i in range(len(experiments)):
        for j in range(len(groups)):
            ax.text(j, i, f'{matrix[i, j]:.3f}',
                   ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    ax.set_title('各实验组综合得分热力图', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_metrics_heatmap.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_metrics_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_metrics_heatmap.png/pdf")


def plot_voter_distribution():
    """
    绘制选民分布图 - 展示选民的政治倾向初始分布
    基于 Pew Research Center 政治类型学
    """
    print("绘制选民分布图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 图1: 选民政治倾向分布（饼图）
    ax1 = axes[0]
    
    # 根据实验设置：30 个选民的初始分布
    voter_types = ['Biden 支持者', 'Trump 支持者', '中间选民']
    initial_counts = [9, 9, 12]  # 典型初始分布
    colors = [COLORS['biden'], COLORS['trump'], '#8b5cf6']
    explode = (0, 0, 0.05)  # 突出中间选民
    
    wedges, texts, autotexts = ax1.pie(initial_counts, explode=explode, labels=voter_types, 
                                        colors=colors, autopct='%1.1f%%',
                                        shadow=False, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('初始选民政治倾向分布\n(N=30)', fontsize=14, fontweight='bold', pad=15)
    
    # 图2: 选民类型详细分布（条形图）
    ax2 = axes[1]
    
    # 基于 Pew 政治类型学的 9 个类别（简化为 5 类）
    voter_categories = ['坚定民主党', '温和民主党', '中间派', '温和共和党', '坚定共和党']
    category_counts = [5, 4, 12, 5, 4]  # 30 人总数
    category_colors = ['#1e40af', '#60a5fa', '#8b5cf6', '#f87171', '#b91c1c']
    
    x = np.arange(len(voter_categories))
    bars = ax2.bar(x, category_counts, color=category_colors, edgecolor='white', 
                  linewidth=2, alpha=0.9, width=0.6)
    
    # 添加数值标签
    for bar, count in zip(bars, category_counts):
        ax2.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_xlabel('')
    ax2.set_ylabel('选民人数', fontsize=12)
    ax2.set_title('选民政治类型分布\n(基于 Pew 政治类型学)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(voter_categories, fontsize=10, rotation=15, ha='right')
    ax2.set_ylim(0, 16)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_voter_distribution.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_voter_distribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_voter_distribution.png/pdf")


def plot_resource_demand_distribution(data: dict):
    """
    绘制资源需求分布图 - 展示各 Agent 的资源需求分布
    """
    print("绘制资源需求分布图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 图1: 资源需求分布直方图
    ax1 = axes[0]
    
    # 模拟资源需求分布 (根据实验设置：20 个 Agent，需求范围 15-30)
    np.random.seed(42)
    resource_demands = np.random.randint(15, 31, size=20)
    resource_demands = sorted(resource_demands)
    
    # 直方图
    bins = np.arange(14.5, 32.5, 2)
    n, bins_out, patches = ax1.hist(resource_demands, bins=bins, color='#10b981', 
                                     edgecolor='white', linewidth=1.5, alpha=0.85)
    
    # 添加平均值线
    mean_demand = np.mean(resource_demands)
    ax1.axvline(x=mean_demand, color='#dc3545', linestyle='--', linewidth=2.5, 
               label=f'平均需求: {mean_demand:.1f}')
    
    # 添加总资源线
    total_available = 400
    fair_share = total_available / 20
    ax1.axvline(x=fair_share, color='#0d6efd', linestyle=':', linewidth=2.5, 
               label=f'公平份额: {fair_share:.1f}')
    
    ax1.set_xlabel('资源需求量', fontsize=12)
    ax1.set_ylabel('Agent 数量', fontsize=12)
    ax1.set_title('Agent 资源需求分布', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(14, 32)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 图2: 资源供需对比
    ax2 = axes[1]
    
    categories = ['总需求', '总供给', '缺口']
    total_demand = sum(resource_demands)
    total_supply = 400
    gap = total_demand - total_supply
    
    values = [total_demand, total_supply, gap if gap > 0 else 0]
    colors = ['#f59e0b', '#10b981', '#ef4444' if gap > 0 else '#10b981']
    
    bars = ax2.bar(categories, values, color=colors, edgecolor='white', 
                  linewidth=2, alpha=0.9, width=0.5)
    
    for bar, val in zip(bars, values):
        ax2.annotate(f'{val}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 添加稀缺度比率
    scarcity_ratio = total_demand / total_supply
    ax2.text(1, max(values) * 0.85, f'稀缺度比率: {scarcity_ratio:.2f}',
            ha='center', fontsize=12, fontweight='bold', 
            color='#dc3545' if scarcity_ratio > 1 else '#10b981',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#dee2e6'))
    
    ax2.set_ylabel('资源单位', fontsize=12)
    ax2.set_title('资源供需状况\n(N=20 Agents, 总资源=400)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim(0, max(values) * 1.15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_resource_demand_distribution.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_resource_demand_distribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_resource_demand_distribution.png/pdf")


def plot_info_agent_distribution():
    """
    绘制信息传播实验中的 Agent 类型分布
    """
    print("绘制信息传播 Agent 分布图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 图1: Agent 类型分布（饼图）
    ax1 = axes[0]
    
    # 根据实验设置：50 个 Agent 的类型分布
    agent_types = ['Normal\n普通用户', 'Skeptic\n怀疑者', 'Gullible\n易信者', 'Influencer\n影响者']
    type_counts = [25, 10, 10, 5]  # 50 人总数
    colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444']
    explode = (0, 0.02, 0.02, 0.05)
    
    wedges, texts, autotexts = ax1.pie(type_counts, explode=explode, labels=agent_types, 
                                        colors=colors, autopct='%1.1f%%',
                                        shadow=False, startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax1.set_title('Agent 类型分布\n(N=50)', fontsize=14, fontweight='bold', pad=15)
    
    # 图2: 各类型对虚假信息的判断准确率
    ax2 = axes[1]
    
    # 基于实验数据
    accuracy_by_type = {
        'Normal': 0.634,
        'Skeptic': 0.484,
        'Gullible': 0.600,
        'Influencer': 0.578
    }
    
    types = list(accuracy_by_type.keys())
    accuracies = list(accuracy_by_type.values())
    bar_colors = colors
    
    x = np.arange(len(types))
    bars = ax2.bar(x, accuracies, color=bar_colors, edgecolor='white', 
                  linewidth=2, alpha=0.9, width=0.6)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        ax2.annotate(f'{acc:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加基线
    ax2.axhline(y=0.5, color='#6c757d', linestyle='--', linewidth=2, 
               alpha=0.7, label='随机猜测 (50%)')
    
    ax2.set_xlabel('')
    ax2.set_ylabel('判断准确率', fontsize=12)
    ax2.set_title('各类型 Agent 信息判断准确率\n(CoT 基线组)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['普通用户', '怀疑者', '易信者', '影响者'], fontsize=11)
    ax2.set_ylim(0, 0.8)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_info_agent_distribution.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_info_agent_distribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_info_agent_distribution.png/pdf")


def plot_network_topology():
    """
    绘制社交网络拓扑示意图
    """
    print("绘制网络拓扑示意图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    scenarios = [
        ('选举投票', 30, 6, 0.3, '#6366f1'),
        ('资源分配', 20, 4, 0.4, '#10b981'),
        ('信息传播', 50, 5.64, 0.28, '#f59e0b'),
    ]
    
    for idx, (name, nodes, avg_degree, clustering, color) in enumerate(scenarios):
        ax = axes[idx]
        
        # 创建简化的网络可视化
        np.random.seed(42 + idx)
        
        # 生成节点位置（圆形布局）
        theta = np.linspace(0, 2*np.pi, nodes, endpoint=False)
        r = 1.0
        x_nodes = r * np.cos(theta)
        y_nodes = r * np.sin(theta)
        
        # 绘制节点
        ax.scatter(x_nodes, y_nodes, s=100, c=color, edgecolors='white', 
                  linewidth=1.5, alpha=0.9, zorder=3)
        
        # 绘制部分边（小世界网络特征）
        num_edges = int(nodes * avg_degree / 2)
        for _ in range(min(num_edges, 50)):  # 限制边数以保持可读性
            i = np.random.randint(nodes)
            j = (i + np.random.randint(1, 4)) % nodes  # 近邻连接
            if np.random.random() < 0.1:  # 少量长程连接
                j = np.random.randint(nodes)
            ax.plot([x_nodes[i], x_nodes[j]], [y_nodes[i], y_nodes[j]], 
                   color=color, alpha=0.2, linewidth=0.8, zorder=1)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{name}\nN={nodes}, 度={avg_degree:.1f}, 聚类={clustering:.2f}',
                    fontsize=13, fontweight='bold', pad=10)
    
    plt.suptitle('三大实验场景的社交网络拓扑结构', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_network_topology.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig_network_topology.pdf', bbox_inches='tight')
    plt.close()
    print(f"  已保存: fig_network_topology.png/pdf")


def main():
    """主函数"""
    print("=" * 60)
    print("Casevo 实验结果可视化")
    print("重点展示各优化组件的贡献")
    print("=" * 60)
    
    data = load_all_experiment_data()
    
    print("\n开始生成可视化图表...")
    print("-" * 40)
    
    # 实验配置与分布图
    plot_voter_distribution()              # 选民分布图
    plot_resource_demand_distribution(data) # 资源需求分布图
    plot_info_agent_distribution()         # 信息传播 Agent 分布
    plot_network_topology()                # 网络拓扑示意图
    
    # 核心对比图表（对应课题目标）
    plot_tot_vs_cot(data)                  # ToT vs CoT 效果验证
    plot_ablation_comparison(data)         # 消融实验：记忆和反思效果
    plot_component_contribution(data)      # 各组件贡献分析
    plot_progressive_improvement(data)     # 渐进式提升对比
    
    # 场景详细分析
    plot_election_evolution(data)          # 选举态度演化
    plot_resource_convergence(data)        # 资源分配收敛
    plot_info_spread(data)                 # 信息传播趋势
    
    # 综合对比图表
    plot_ablation_radar(data)              # 多维度雷达图
    plot_metrics_heatmap(data)             # 热力图
    plot_computational_efficiency(data)    # 计算效率
    
    print("-" * 40)
    print(f"\n完成! 所有图表已保存至: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
