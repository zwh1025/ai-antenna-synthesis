"""生成答辩PPT所需的所有图表。"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'charts')
os.makedirs(OUTPUT, exist_ok=True)

# Try Chinese font
for name in ['WenQuanYi Micro Hei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']:
    try:
        fp = fm.FontProperties(family=name)
        if fp.get_name() == name or name == 'DejaVu Sans':
            plt.rcParams['font.family'] = name
            break
    except:
        continue
plt.rcParams['axes.unicode_minus'] = False

# Color scheme
C_NAVY = '#0F2D4E'
C_TEAL = '#00A8A8'
C_GREEN = '#2E8B57'
C_RED = '#CC3333'
C_ORANGE = '#E67E22'
C_GRAY = '#888888'
C_LIGHT = '#E8EDF2'

def save_chart(fig, name):
    path = os.path.join(OUTPUT, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {name}')
    return path

# ============================================================
# Chart 1: NPU vs CPU training speed
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
models = ['128-dim\n(116K params)', '256-dim\n(463K params)']
npu_times = [2.5, 2.5]
cpu_times = [38.03, 214.97]
x = np.arange(len(models))
w = 0.35
bars1 = ax.bar(x - w/2, npu_times, w, label='NPU (Ascend 910)', color=C_TEAL, edgecolor='white')
bars2 = ax.bar(x + w/2, cpu_times, w, label='CPU', color=C_GRAY, edgecolor='white')
ax.set_ylabel('ms / epoch', fontsize=13)
ax.set_title('NPU vs CPU Training Speed', fontsize=15, fontweight='bold', color=C_NAVY)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars1, npu_times):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}', ha='center', fontsize=10, fontweight='bold', color=C_TEAL)
for bar, val in zip(bars2, cpu_times):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}', ha='center', fontsize=10, color=C_GRAY)
ax.annotate('15.7x', xy=(0, 20), fontsize=14, fontweight='bold', color=C_GREEN, ha='center')
ax.annotate('68.6x', xy=(1, 120), fontsize=14, fontweight='bold', color=C_GREEN, ha='center')
chart_npu_speed = save_chart(fig, 'npu_speed.png')

# ============================================================
# Chart 2: Curved array SOCP verification
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
alphas = ['0.00\n(flat)', '0.02', '0.05', '0.10', '0.15']
taylor = [-35.6, -35.6, -26.3, -20.8, -17.1]
socp = [-35.6, -35.6, -26.3, -24.9, -21.7]
x = np.arange(len(alphas))
w = 0.35
ax.bar(x - w/2, taylor, w, label='Taylor', color=C_NAVY, edgecolor='white')
ax.bar(x + w/2, socp, w, label='SOCP', color=C_TEAL, edgecolor='white')
ax.set_ylabel('SLL (dB)', fontsize=13)
ax.set_xlabel('Curvature alpha', fontsize=13)
ax.set_title('Curved Array: Taylor vs SOCP', fontsize=15, fontweight='bold', color=C_NAVY)
ax.set_xticks(x)
ax.set_xticklabels(alphas, fontsize=10)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=-35, color=C_RED, linestyle='--', linewidth=1, alpha=0.7)
ax.text(4, -34, 'Target -35dB', fontsize=9, color=C_RED, ha='right')
for i in [3, 4]:
    ax.annotate(f'{socp[i]-taylor[i]:+.1f}', xy=(i, socp[i]+0.5), fontsize=11, fontweight='bold', color=C_GREEN, ha='center')
chart_curved_socp = save_chart(fig, 'curved_socp.png')

# ============================================================
# Chart 3: AI results v1 (Taylor vs SOCP vs AI)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
labels = ['Val\n(30 samples)', 'Test\n(50 samples)']
taylor_v1 = [-19.70, -20.18]
socp_v1 = [-23.11, -23.27]
ai_v1 = [-23.51, -23.72]
x = np.arange(len(labels))
w = 0.25
ax.bar(x - w, taylor_v1, w, label='Taylor', color=C_NAVY)
ax.bar(x, socp_v1, w, label='SOCP', color=C_TEAL)
ax.bar(x + w, ai_v1, w, label='AI (DeepSets)', color=C_GREEN)
ax.set_ylabel('SLL (dB)', fontsize=13)
ax.set_title('v1: Single Scan Direction Results', fontsize=15, fontweight='bold', color=C_NAVY)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=10, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i in range(2):
    ax.text(x[i]+w, ai_v1[i]+0.2, f'{ai_v1[i]:.1f}', ha='center', fontsize=9, fontweight='bold', color=C_GREEN)
    ax.text(x[i]+w, ai_v1[i]+0.8, f'114.6%', ha='center', fontsize=8, color=C_GREEN) if i == 1 else \
    ax.text(x[i]+w, ai_v1[i]+0.8, f'111.5%', ha='center', fontsize=8, color=C_GREEN)
chart_v1 = save_chart(fig, 'ai_v1_results.png')

# ============================================================
# Chart 4: AI results v2 per-direction
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))
thetas = ['0 deg', '15 deg', '30 deg', '45 deg', '60 deg']
taylor_v2 = [-25.8, -25.7, -19.4, -19.0, -20.3]
socp_v2 = [-25.8, -25.7, -22.8, -22.5, -23.4]
ai_v2 = [-25.8, -25.7, -22.4, -22.4, -22.2]
recovery = [0, 0, 86, 97, 60]
x = np.arange(len(thetas))
w = 0.25
ax.bar(x - w, taylor_v2, w, label='Taylor', color=C_NAVY)
ax.bar(x, socp_v2, w, label='SOCP', color=C_TEAL)
ax.bar(x + w, ai_v2, w, label='AI (DeepSets)', color=C_GREEN)
ax.set_ylabel('SLL (dB)', fontsize=13)
ax.set_xlabel('Scan angle theta', fontsize=13)
ax.set_title('v2: Multi-Scan Direction Results (Test Set)', fontsize=14, fontweight='bold', color=C_NAVY)
ax.set_xticks(x)
ax.set_xticklabels(thetas, fontsize=10)
ax.legend(fontsize=10, loc='lower left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i in [2, 3, 4]:
    color = C_GREEN if recovery[i] >= 80 else C_ORANGE
    ax.text(x[i]+w, ai_v2[i]+0.3, f'{recovery[i]}%', ha='center', fontsize=10, fontweight='bold', color=color)
chart_v2 = save_chart(fig, 'ai_v2_per_direction.png')

# ============================================================
# Chart 5: Cylindrical array results
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
Rs = ['R=5', 'R=8', 'R=10', 'R=15', 'R=20']
taylor_cyl = [-13.2, -15.0, -19.0, -32.0, -34.9]
socp_cyl = [-21.6, -21.2, -21.7, -32.0, -34.9]
x = np.arange(len(Rs))
w = 0.35
ax.bar(x - w/2, taylor_cyl, w, label='Taylor', color=C_NAVY)
ax.bar(x + w/2, socp_cyl, w, label='SOCP', color=C_TEAL)
ax.set_ylabel('SLL (dB)', fontsize=13)
ax.set_xlabel('Cylinder radius R (alpha_equiv = 1/2R)', fontsize=11)
ax.set_title('Cylindrical Array: Taylor vs SOCP', fontsize=15, fontweight='bold', color=C_NAVY)
ax.set_xticks(x)
ax.set_xticklabels(Rs, fontsize=11)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i in [0, 1, 2]:
    ax.annotate(f'{socp_cyl[i]-taylor_cyl[i]:+.1f}', xy=(i, socp_cyl[i]+0.3), fontsize=11, fontweight='bold', color=C_GREEN, ha='center')
chart_cyl = save_chart(fig, 'cylindrical_results.png')

# ============================================================
# Chart 6: Non-ideal conditions (flat array)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))
conditions = ['Ideal', 'Quant\n0.5dB\n+6bit', 'Position\n+/-lambda/20', '5%\nFailure', '10%\nFailure', '20%\nFailure', '-10%\nFreq\nOffset']
sll_nonideal = [-35.5, -34.5, -33.4, -31.5, -28.7, -25.5, -20.3]
colors = [C_GREEN, C_ORANGE, C_ORANGE, C_RED, C_RED, C_RED, C_RED]
bars = ax.bar(range(len(conditions)), sll_nonideal, color=colors, edgecolor='white', width=0.6)
ax.set_ylabel('SLL (dB)', fontsize=13)
ax.set_title('Non-Ideal Conditions (Flat Array, Fixed Excitation)', fontsize=14, fontweight='bold', color=C_NAVY)
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(conditions, fontsize=9)
ax.axhline(y=-35, color=C_RED, linestyle='--', linewidth=1, alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, sll_nonideal):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
chart_nonideal = save_chart(fig, 'nonideal_flat.png')

# ============================================================
# Chart 7: Curved non-ideal (Taylor vs SOCP vs AI)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))
conds = ['Ideal', '5%\nFailure', '10%\nFailure', '20%\nFailure', 'Quant\n+5% Fail']
taylor_ni = [-20.2, -19.8, -19.7, -19.0, -20.1]
socp_ni = [-22.8, -22.2, -21.9, -21.2, -22.1]
ai_ni = [-23.6, -23.1, -22.2, -21.4, -23.1]
x = np.arange(len(conds))
w = 0.25
ax.bar(x - w, taylor_ni, w, label='Taylor', color=C_NAVY)
ax.bar(x, socp_ni, w, label='SOCP', color=C_TEAL)
ax.bar(x + w, ai_ni, w, label='AI (DeepSets)', color=C_GREEN)
ax.set_ylabel('SLL (dB)', fontsize=13)
ax.set_title('Curved Array: Non-Ideal Conditions', fontsize=14, fontweight='bold', color=C_NAVY)
ax.set_xticks(x)
ax.set_xticklabels(conds, fontsize=9)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
chart_curved_ni = save_chart(fig, 'curved_nonideal.png')

# ============================================================
# Chart 8: AI negative results (SOCP improvement comparison)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
scenarios = ['Failure\nComp\n(5%)', 'Non-uniform\nCoord\n(+/-0.05lambda)', 'Curved\nalpha=0.10', 'Curved\nalpha=0.15']
improvement = [0.0, 0.0, -4.2, -4.6]
colors_bar = [C_RED, C_RED, C_GREEN, C_GREEN]
bars = ax.bar(range(len(scenarios)), improvement, color=colors_bar, edgecolor='white', width=0.5)
ax.set_ylabel('SOCP Improvement (dB)', fontsize=13)
ax.set_title('SOCP Improvement: Negative vs Positive Results', fontsize=14, fontweight='bold', color=C_NAVY)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(scenarios, fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, improvement):
    y_pos = val - 0.3 if val < 0 else 0.2
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.1f}', ha='center', fontsize=11, fontweight='bold')
ax.text(0.5, -3.5, 'Negative\n(no AI value)', fontsize=9, color=C_RED, ha='center', style='italic')
ax.text(2.5, -3.5, 'Positive\n(AI valuable)', fontsize=9, color=C_GREEN, ha='center', style='italic')
chart_negative = save_chart(fig, 'ai_negative_positive.png')

# ============================================================
# Chart 9: Acceptance results (73 directions)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
metrics = ['Sum\nSLL', 'After\nLCMV', 'Diff\nSLL', 'Null\nDepth', 'Pointing\nError']
values = [-35.6, -35.6, -21.5, -45.5, 0.24]  # pointing error (absolute)
targets = [-35, -35, -20, -30, 0.195]
# Normalize for display (all as ratio to target)
ratios = [v/t for v, t in zip(values, targets)]
colors_r = [C_GREEN if r <= 1.0 else C_RED for r in ratios]
bars = ax.barh(range(len(metrics)), [v for v in values], color=colors_r, edgecolor='white')
ax.set_xlabel('Value (dB or degrees)', fontsize=13)
ax.set_title('73-Direction Acceptance Results', fontsize=15, fontweight='bold', color=C_NAVY)
ax.set_yticks(range(len(metrics)))
ax.set_yticklabels(metrics, fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i, (bar, v, t) in enumerate(zip(bars, values, targets)):
    status = 'PASS' if v <= t else 'FAIL'
    ax.text(v, i, f' {v:.1f} ({status})', va='center', fontsize=10, fontweight='bold')
chart_accept = save_chart(fig, 'acceptance_73.png')

print('\nAll charts generated.')
