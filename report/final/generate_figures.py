"""
Generate all figures for the final report
Modern, professional visualization with English labels
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import os

# ============ Style Configuration ============
# Use clean, professional fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'medium'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#E0E0E0'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.5

# Create figures directory
os.makedirs('figures', exist_ok=True)

# Modern color palette - Vibrant but professional
COLORS = {
    'primary': '#6366F1',      # Indigo
    'secondary': '#EC4899',    # Pink
    'success': '#10B981',      # Emerald
    'warning': '#F59E0B',      # Amber
    'danger': '#EF4444',       # Red
    'info': '#06B6D4',         # Cyan
    'purple': '#8B5CF6',       # Purple
    'dark': '#1F2937',         # Dark gray
    'gray': '#6B7280',         # Gray
    'light': '#F3F4F6',        # Light gray
    'bg_blue': '#EEF2FF',      # Light indigo bg
    'bg_green': '#ECFDF5',     # Light green bg
    'bg_amber': '#FFFBEB',     # Light amber bg
    'bg_rose': '#FFF1F2',      # Light rose bg
}

# Gradient backgrounds helper
def create_gradient_bg(ax, color1='#667EEA', color2='#764BA2', alpha=0.1):
    """Create a subtle gradient background"""
    from matplotlib.colors import LinearSegmentedColormap
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    cmap = LinearSegmentedColormap.from_list('custom', [color1, color2])
    ax.imshow(gradient, aspect='auto', cmap=cmap, alpha=alpha,
              extent=[ax.get_xlim()[0], ax.get_xlim()[1], 
                      ax.get_ylim()[0], ax.get_ylim()[1]], zorder=0)

# ========== Figure 1: Experiment Scale Statistics ==========
print("Generating Figure 1: Experiment Scale Statistics...")

fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Background gradient effect
gradient = np.linspace(0, 1, 100).reshape(1, -1)
ax.imshow(gradient, extent=[0, 1, 0, 1], aspect='auto', 
          cmap='Blues', alpha=0.08, zorder=0)

# Main container with shadow effect
shadow = FancyBboxPatch((0.052, 0.08), 0.9, 0.82,
                        boxstyle="round,pad=0.02,rounding_size=0.03",
                        facecolor='#00000010', edgecolor='none',
                        transform=ax.transAxes, zorder=1)
ax.add_patch(shadow)

main_box = FancyBboxPatch((0.05, 0.1), 0.9, 0.82,
                          boxstyle="round,pad=0.02,rounding_size=0.03",
                          facecolor='white', edgecolor=COLORS['primary'],
                          linewidth=2.5, transform=ax.transAxes, zorder=2)
ax.add_patch(main_box)

# Header bar
header = FancyBboxPatch((0.05, 0.78), 0.9, 0.14,
                        boxstyle="round,pad=0.01,rounding_size=0.03",
                        facecolor=COLORS['primary'], edgecolor='none',
                        transform=ax.transAxes, zorder=3)
ax.add_patch(header)

ax.text(0.5, 0.85, 'EXPERIMENT SCALE OVERVIEW',
        ha='center', va='center', fontsize=18, fontweight='bold',
        color='white', transform=ax.transAxes, zorder=4)

# Statistics grid
stats = [
    {'icon': '🎯', 'value': '3', 'label': 'Scenarios', 'sublabel': 'Election, Resource, Information'},
    {'icon': '⚡', 'value': '5', 'label': 'Configurations', 'sublabel': 'baseline_cot → full'},
    {'icon': '🔄', 'value': '45', 'label': 'Total Runs', 'sublabel': '3 runs per config'},
    {'icon': '🤖', 'value': '100', 'label': 'Agents/Group', 'sublabel': '30 + 20 + 50'},
]

positions = [0.175, 0.375, 0.575, 0.775]
for i, (stat, x) in enumerate(zip(stats, positions)):
    # Stat card
    card = FancyBboxPatch((x - 0.08, 0.2), 0.16, 0.5,
                          boxstyle="round,pad=0.01,rounding_size=0.02",
                          facecolor=COLORS['bg_blue'] if i % 2 == 0 else COLORS['bg_green'],
                          edgecolor=COLORS['primary'] if i % 2 == 0 else COLORS['success'],
                          linewidth=1.5, alpha=0.8,
                          transform=ax.transAxes, zorder=3)
    ax.add_patch(card)
    
    # Value (large number)
    ax.text(x, 0.52, stat['value'],
            ha='center', va='center', fontsize=32, fontweight='bold',
            color=COLORS['primary'] if i % 2 == 0 else COLORS['success'],
            transform=ax.transAxes, zorder=4)
    
    # Label
    ax.text(x, 0.36, stat['label'],
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=COLORS['dark'], transform=ax.transAxes, zorder=4)
    
    # Sublabel
    ax.text(x, 0.27, stat['sublabel'],
            ha='center', va='center', fontsize=9,
            color=COLORS['gray'], transform=ax.transAxes, zorder=4)

plt.tight_layout()
plt.savefig('figures/fig_experiment_scale.pdf', bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig('figures/fig_experiment_scale.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print("✓ Figure 1 saved")

# ========== Figure 2: ToT Effect Comparison ==========
print("Generating Figure 2: ToT Effect Comparison...")

fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

scenarios = ['Election\nVoting', 'Resource\nAllocation', 'Information\nPropagation']
positive_effects = [121, 33, 23]
negative_effects = [0, 0, -10]

x = np.arange(len(scenarios))
width = 0.38

# Create bars with gradients (using multiple overlapping bars)
for i, (pos, neg) in enumerate(zip(positive_effects, negative_effects)):
    # Positive bar with gradient effect
    if pos > 0:
        base_color = COLORS['primary'] if i == 0 else (COLORS['info'] if i == 1 else COLORS['success'])
        bar = ax.bar(x[i] - width/2, pos, width, 
                    color=base_color, edgecolor='white', linewidth=2,
                    zorder=3, alpha=0.9)
        # Highlight effect
        ax.bar(x[i] - width/2, pos * 0.15, width * 0.6, bottom=pos * 0.7,
               color='white', alpha=0.3, zorder=4)
    
    # Negative bar
    if neg < 0:
        ax.bar(x[i] + width/2, neg, width,
               color=COLORS['danger'], edgecolor='white', linewidth=2,
               zorder=3, alpha=0.85)

# Value labels with enhanced styling
for i, (pos, neg) in enumerate(zip(positive_effects, negative_effects)):
    if pos > 0:
        # Background box for label
        ax.text(x[i] - width/2, pos + 5, f'+{pos}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold',
                color=COLORS['dark'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='none', alpha=0.9))
    if neg < 0:
        ax.text(x[i] + width/2, neg - 5, f'{neg}%',
                ha='center', va='top', fontsize=12, fontweight='bold',
                color=COLORS['danger'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='none', alpha=0.9))

# Styling
ax.set_ylabel('Performance Change (%)', fontsize=13, fontweight='bold', 
              color=COLORS['dark'], labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=12, fontweight='medium')
ax.set_ylim(-25, 145)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS['primary'], edgecolor='white', label='Positive Effect', alpha=0.9),
    Patch(facecolor=COLORS['danger'], edgecolor='white', label='Negative Effect', alpha=0.85),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
          framealpha=0.95, edgecolor='none', fancybox=True)

# Grid and axis styling
ax.axhline(y=0, color=COLORS['dark'], linewidth=1.2, zorder=2)
ax.grid(axis='y', alpha=0.3, linestyle='-', color='#E5E7EB', zorder=1)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#E5E7EB')
ax.spines['bottom'].set_color('#E5E7EB')
ax.tick_params(colors=COLORS['dark'])

# Title
ax.set_title('ToT Enhancement Effect Across Scenarios', 
             fontsize=16, fontweight='bold', color=COLORS['dark'], pad=20)

plt.tight_layout()
plt.savefig('figures/fig_tot_effect.pdf', bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig('figures/fig_tot_effect.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print("✓ Figure 2 saved")

# ========== Figure 3: ToT Behavior Patterns ==========
print("Generating Figure 3: ToT Behavior Patterns...")

fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Subtle background
gradient = np.linspace(0, 1, 100).reshape(-1, 1)
ax.imshow(gradient, extent=[0, 1, 0, 1], aspect='auto',
          cmap='Purples', alpha=0.05, zorder=0)

# ToT central node - glowing effect
for r, alpha in [(0.12, 0.1), (0.10, 0.2), (0.08, 0.4)]:
    glow = Circle((0.5, 0.82), r, color=COLORS['secondary'], 
                  alpha=alpha, transform=ax.transAxes, zorder=1)
    ax.add_patch(glow)

tot_circle = Circle((0.5, 0.82), 0.065, color=COLORS['secondary'],
                    transform=ax.transAxes, zorder=2)
ax.add_patch(tot_circle)
ax.text(0.5, 0.82, 'ToT', ha='center', va='center',
        fontsize=20, fontweight='bold', color='white',
        transform=ax.transAxes, zorder=3)

# Scenario cards data
scenarios_data = [
    {
        'x': 0.17, 'y': 0.35, 
        'title': 'NEGOTIATION',
        'subtitle': 'Resource Allocation',
        'effects': ['✓ Smarter bargaining', '✓ Faster convergence'],
        'color': COLORS['info'],
        'bg': COLORS['bg_blue'],
        'effect': '+33%',
        'effect_type': 'positive'
    },
    {
        'x': 0.5, 'y': 0.35,
        'title': 'JUDGMENT',
        'subtitle': 'Information Propagation',
        'effects': ['⚠ Over-cautious', '⚠ Rejects all info'],
        'color': COLORS['warning'],
        'bg': COLORS['bg_amber'],
        'effect': '±0%',
        'effect_type': 'neutral'
    },
    {
        'x': 0.83, 'y': 0.35,
        'title': 'ATTITUDE',
        'subtitle': 'Election Voting',
        'effects': ['✓ Deeper reasoning', '✓ More uncertainty'],
        'color': COLORS['success'],
        'bg': COLORS['bg_green'],
        'effect': '+121%',
        'effect_type': 'positive'
    },
]

for s in scenarios_data:
    # Card shadow
    shadow = FancyBboxPatch((s['x'] - 0.132, s['y'] - 0.202), 0.264, 0.44,
                            boxstyle="round,pad=0.01,rounding_size=0.02",
                            facecolor='#00000015', edgecolor='none',
                            transform=ax.transAxes, zorder=1)
    ax.add_patch(shadow)
    
    # Main card
    card = FancyBboxPatch((s['x'] - 0.13, s['y'] - 0.2), 0.26, 0.44,
                          boxstyle="round,pad=0.01,rounding_size=0.02",
                          facecolor='white', edgecolor=s['color'],
                          linewidth=2.5, transform=ax.transAxes, zorder=2)
    ax.add_patch(card)
    
    # Header stripe
    header = FancyBboxPatch((s['x'] - 0.13, s['y'] + 0.17), 0.26, 0.07,
                            boxstyle="round,pad=0.005,rounding_size=0.02",
                            facecolor=s['color'], edgecolor='none',
                            transform=ax.transAxes, zorder=3)
    ax.add_patch(header)
    
    # Title
    ax.text(s['x'], s['y'] + 0.2, s['title'],
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='white', transform=ax.transAxes, zorder=4)
    
    # Subtitle
    ax.text(s['x'], s['y'] + 0.1, s['subtitle'],
            ha='center', va='center', fontsize=11, fontweight='medium',
            color=s['color'], transform=ax.transAxes, zorder=4)
    
    # Effects list
    for i, effect in enumerate(s['effects']):
        ax.text(s['x'], s['y'] - 0.02 - i * 0.06, effect,
                ha='center', va='center', fontsize=10,
                color=COLORS['dark'], transform=ax.transAxes, zorder=4)
    
    # Effect badge
    badge_color = COLORS['success'] if s['effect_type'] == 'positive' else COLORS['warning']
    badge = FancyBboxPatch((s['x'] - 0.045, s['y'] - 0.18), 0.09, 0.05,
                           boxstyle="round,pad=0.005,rounding_size=0.01",
                           facecolor=badge_color, edgecolor='none',
                           transform=ax.transAxes, zorder=3)
    ax.add_patch(badge)
    ax.text(s['x'], s['y'] - 0.155, s['effect'],
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='white', transform=ax.transAxes, zorder=4)
    
    # Connection line from ToT
    line_color = s['color']
    # Curved arrow effect
    start_y = 0.75
    end_y = s['y'] + 0.24
    mid_y = (start_y + end_y) / 2
    
    ax.annotate('', xy=(s['x'], end_y), xytext=(0.5, start_y),
                arrowprops=dict(arrowstyle='->', color=line_color, lw=2.5,
                               connectionstyle='arc3,rad=0'),
                transform=ax.transAxes, zorder=1)

# Title at top
ax.text(0.5, 0.96, 'ToT Behavior Patterns Across Task Types',
        ha='center', va='top', fontsize=18, fontweight='bold',
        color=COLORS['dark'], transform=ax.transAxes)

plt.tight_layout()
plt.savefig('figures/fig_tot_behavior.pdf', bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig('figures/fig_tot_behavior.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print("✓ Figure 3 saved")

# ========== Figure 4: Cost-Benefit Analysis ==========
print("Generating Figure 4: Cost-Benefit Analysis...")

fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

# Configuration data with actual computed costs
configs = {
    'baseline_cot': {'x': 1, 'y': 0, 'color': COLORS['gray'], 'size': 180},
    'tot_only': {'x': 55, 'y': 35, 'color': COLORS['info'], 'size': 220},
    'tot_memory': {'x': 62, 'y': 42, 'color': COLORS['success'], 'size': 280},
    'tot_reflection': {'x': 68, 'y': 28, 'color': COLORS['warning'], 'size': 220},
    'full': {'x': 78, 'y': 33, 'color': COLORS['purple'], 'size': 250},
}

# Draw recommended zone first (background)
recommend_zone = FancyBboxPatch((48, 36), 20, 12,
                                boxstyle="round,pad=0.5,rounding_size=2",
                                facecolor=COLORS['bg_green'], 
                                edgecolor=COLORS['success'],
                                linewidth=2.5, linestyle='--',
                                alpha=0.6, zorder=1)
ax.add_patch(recommend_zone)
ax.text(58, 50, '★ RECOMMENDED ZONE', ha='center', va='bottom',
        fontsize=11, color=COLORS['success'], fontweight='bold')

# Plot points with glow effect
for name, data in configs.items():
    # Glow effect
    for size_mult, alpha in [(3, 0.1), (2, 0.15), (1.5, 0.2)]:
        ax.scatter(data['x'], data['y'], s=data['size'] * size_mult,
                  color=data['color'], alpha=alpha, zorder=2)
    
    # Main point
    ax.scatter(data['x'], data['y'], s=data['size'], color=data['color'],
              edgecolors='white', linewidths=2.5, zorder=4)
    
    # Labels with smart positioning
    label = name.replace('_', ' ').title()
    offset_y = 4 if name != 'tot_reflection' else -5
    va = 'bottom' if name != 'tot_reflection' else 'top'
    
    fontweight = 'bold' if name == 'tot_memory' else 'medium'
    fontsize = 11 if name == 'tot_memory' else 10
    
    ax.text(data['x'], data['y'] + offset_y, label,
            ha='center', va=va, fontsize=fontsize, fontweight=fontweight,
            color=data['color'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='none', alpha=0.85))

# Axis labels and styling
ax.set_xlabel('Computational Cost (Relative)', fontsize=13, fontweight='bold',
              color=COLORS['dark'], labelpad=12)
ax.set_ylabel('Performance Improvement (%)', fontsize=13, fontweight='bold',
              color=COLORS['dark'], labelpad=12)
ax.set_xlim(-5, 95)
ax.set_ylim(-5, 55)

# Grid styling
ax.grid(True, alpha=0.3, linestyle='-', color='#E5E7EB', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#E5E7EB')
ax.spines['bottom'].set_color('#E5E7EB')
ax.tick_params(colors=COLORS['dark'], labelsize=10)

# Title
ax.set_title('Cost-Benefit Analysis of ToT Configurations',
             fontsize=16, fontweight='bold', color=COLORS['dark'], pad=20)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['gray'],
           markersize=10, label='Baseline CoT'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['info'],
           markersize=10, label='ToT Only'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['success'],
           markersize=12, label='ToT + Memory (Best)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['warning'],
           markersize=10, label='ToT + Reflection'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['purple'],
           markersize=10, label='Full Config'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
          framealpha=0.95, edgecolor='none', fancybox=True)

plt.tight_layout()
plt.savefig('figures/fig_cost_benefit.pdf', bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig('figures/fig_cost_benefit.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print("✓ Figure 4 saved")

# ========== Figure 5: Decision Flowchart ==========
print("Generating Figure 5: Decision Flowchart...")

fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Background
gradient = np.linspace(0, 1, 100).reshape(-1, 1)
ax.imshow(gradient, extent=[0, 1, 0, 1], aspect='auto',
          cmap='Greys', alpha=0.03, zorder=0)

# Node definitions
nodes = {
    'start': {
        'pos': (0.5, 0.92), 
        'text': 'START', 
        'style': 'start',
        'color': COLORS['primary']
    },
    'q1': {
        'pos': (0.5, 0.75),
        'text': 'Need Deep\nReasoning?',
        'style': 'decision',
        'color': COLORS['warning']
    },
    'q2': {
        'pos': (0.75, 0.55),
        'text': 'Negotiation\nTask?',
        'style': 'decision',
        'color': COLORS['warning']
    },
    'q3': {
        'pos': (0.25, 0.55),
        'text': 'Info Judgment\nTask?',
        'style': 'decision',
        'color': COLORS['warning']
    },
    'r1': {
        'pos': (0.75, 0.25),
        'text': 'tot_memory',
        'subtext': 'Best Balance',
        'style': 'result',
        'color': COLORS['success']
    },
    'r2': {
        'pos': (0.92, 0.55),
        'text': 'tot_only',
        'subtext': 'General Use',
        'style': 'result',
        'color': COLORS['info']
    },
    'r3': {
        'pos': (0.25, 0.25),
        'text': 'baseline_cot',
        'subtext': 'Safe Choice',
        'style': 'result',
        'color': COLORS['gray']
    },
    'r4': {
        'pos': (0.08, 0.55),
        'text': 'tot_only',
        'subtext': 'Zero Tolerance',
        'style': 'result',
        'color': COLORS['danger']
    },
}

# Draw nodes
for name, node in nodes.items():
    x, y = node['pos']
    style = node['style']
    color = node['color']
    
    if style == 'start':
        # Pill shape for start
        box = FancyBboxPatch((x - 0.06, y - 0.025), 0.12, 0.05,
                            boxstyle="round,pad=0.005,rounding_size=0.02",
                            facecolor=color, edgecolor='white',
                            linewidth=2, transform=ax.transAxes, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, node['text'], ha='center', va='center',
                fontsize=12, fontweight='bold', color='white',
                transform=ax.transAxes, zorder=4)
                
    elif style == 'decision':
        # Diamond shape
        size = 0.07
        diamond_verts = [
            (x, y + size), (x + size, y), (x, y - size), (x - size, y)
        ]
        diamond = Polygon(diamond_verts, closed=True,
                         facecolor=COLORS['bg_amber'], edgecolor=color,
                         linewidth=2.5, transform=ax.transAxes, zorder=3)
        ax.add_patch(diamond)
        ax.text(x, y, node['text'], ha='center', va='center',
                fontsize=9, fontweight='medium', color=COLORS['dark'],
                transform=ax.transAxes, zorder=4, linespacing=1.2)
                
    elif style == 'result':
        # Rounded rectangle with icon
        box = FancyBboxPatch((x - 0.065, y - 0.045), 0.13, 0.09,
                            boxstyle="round,pad=0.008,rounding_size=0.015",
                            facecolor=color, edgecolor='white',
                            linewidth=2, transform=ax.transAxes, zorder=3)
        ax.add_patch(box)
        ax.text(x, y + 0.01, node['text'], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                transform=ax.transAxes, zorder=4)
        if 'subtext' in node:
            ax.text(x, y - 0.02, node['subtext'], ha='center', va='center',
                    fontsize=8, color='white', alpha=0.9,
                    transform=ax.transAxes, zorder=4)

# Arrow connections with labels
connections = [
    ('start', 'q1', None),
    ('q1', 'q2', 'YES'),
    ('q1', 'q3', 'NO'),
    ('q2', 'r1', 'YES'),
    ('q2', 'r2', 'NO'),
    ('q3', 'r3', 'YES\n(allow spread)'),
    ('q3', 'r4', 'YES\n(strict)'),
]

for start_name, end_name, label in connections:
    start = nodes[start_name]['pos']
    end = nodes[end_name]['pos']
    
    # Calculate arrow start and end points (offset from node centers)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Offset for node sizes
    if nodes[start_name]['style'] == 'decision':
        start_offset = 0.07
    else:
        start_offset = 0.03
    
    if nodes[end_name]['style'] == 'decision':
        end_offset = 0.07
    else:
        end_offset = 0.05
    
    start_adj = (start[0] + dx/dist * start_offset, start[1] + dy/dist * start_offset)
    end_adj = (end[0] - dx/dist * end_offset, end[1] - dy/dist * end_offset)
    
    ax.annotate('', xy=end_adj, xytext=start_adj,
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'],
                               lw=2, shrinkA=0, shrinkB=0),
                transform=ax.transAxes, zorder=2)
    
    # Add label
    if label:
        mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
        offset_x = 0.03 if start[0] < end[0] else -0.03
        ax.text(mid[0] + offset_x, mid[1], label,
                ha='center', va='center', fontsize=8, fontweight='medium',
                color=COLORS['dark'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#E5E7EB', alpha=0.95),
                transform=ax.transAxes, zorder=5)

# Title
ax.text(0.5, 0.98, 'Configuration Selection Decision Flow',
        ha='center', va='top', fontsize=18, fontweight='bold',
        color=COLORS['dark'], transform=ax.transAxes)

plt.tight_layout()
plt.savefig('figures/fig_decision_flow.pdf', bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig('figures/fig_decision_flow.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print("✓ Figure 5 saved")

print("\n" + "="*50)
print("All figures generated successfully!")
print(f"Saved to: {os.path.abspath('figures')}")
print("="*50)
