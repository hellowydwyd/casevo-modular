import matplotlib.pyplot as plt
import numpy as np

# Set font and style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme (academic-friendly)
COLORS = {
    'biden': '#3B82F6',     # Blue
    'trump': '#EF4444',     # Red
    'undecided': '#9CA3AF', # Gray
    'baseline': '#6B7280',  # Dark gray
    'llm': '#10B981',       # Green
}

# ========== Figure 1: Vote Evolution ==========
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

rounds = ['Initial', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
biden = [35, 43, 46, 49, 52, 53, 52]
trump = [33, 32, 41, 44, 46, 46, 46]
undecided = [33, 26, 14, 8, 3, 2, 3]

ax.plot(rounds, biden, 'o-', color=COLORS['biden'], linewidth=2.5, markersize=8, label='Biden')
ax.plot(rounds, trump, 's-', color=COLORS['trump'], linewidth=2.5, markersize=8, label='Trump')
ax.plot(rounds, undecided, '^-', color=COLORS['undecided'], linewidth=2.5, markersize=8, label='Undecided')

ax.set_xlabel('Debate Round', fontsize=12)
ax.set_ylabel('Number of Votes', fontsize=12)
ax.set_ylim(0, 60)
ax.legend(loc='upper right', fontsize=10)
ax.set_title('Vote Evolution Process (GPT-4o Model)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_vote_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_vote_evolution.pdf', bbox_inches='tight')
print("✓ Figure 1 saved")

# ========== Figure 2: Model Comparison ==========
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

models = ['GPT-4o-mini', 'GPT-4o', '2020 Actual']
biden_pct = [58.4, 51.5, 51.3]
trump_pct = [41.6, 48.5, 48.7]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, biden_pct, width, label='Biden', color=COLORS['biden'], edgecolor='white')
bars2 = ax.bar(x + width/2, trump_pct, width, label='Trump', color=COLORS['trump'], edgecolor='white')

ax.set_ylabel('Vote Share (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 70)
ax.legend(fontsize=10)
ax.bar_label(bars1, fmt='%.1f%%', fontsize=9)
ax.bar_label(bars2, fmt='%.1f%%', fontsize=9)
ax.set_title('Election Simulation Results by Model', fontsize=14, fontweight='bold')

# Add reference line for actual result
ax.axhline(y=51.3, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('fig_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_model_comparison.pdf', bbox_inches='tight')
print("✓ Figure 2 saved")

# ========== Figure 3: Resource Allocation Comparison ==========
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

metrics = ['Avg. Fulfillment', 'Min. Fulfillment', 'Utilization']
baseline = [84.9, 43.8, 98.2]
llm = [87.2, 52.1, 99.5]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, baseline, width, label='Rule-based Baseline', color=COLORS['baseline'])
bars2 = ax.bar(x + width/2, llm, width, label='LLM-driven', color=COLORS['llm'])

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 110)
ax.legend(fontsize=10)
ax.bar_label(bars1, fmt='%.1f', fontsize=9)
ax.bar_label(bars2, fmt='%.1f', fontsize=9)
ax.set_title('Resource Allocation Fairness Comparison', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_resource_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_resource_comparison.pdf', bbox_inches='tight')
print("✓ Figure 3 saved")

print("\nAll figures generated successfully!")