#!/usr/bin/env python3
"""Generate 3-panel overview figure optimized for ICML double-column layout.

Layout: 2x2 grid where Panel B spans both columns in the bottom row.
Panel A and C sit on the top row; Panel B (method convergence) gets full width.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
})

fig = plt.figure(figsize=(7.0, 4.6))

gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], wspace=0.32, hspace=0.42)
ax_a = fig.add_subplot(gs[0, 0])
ax_c = fig.add_subplot(gs[0, 1])
ax_b = fig.add_subplot(gs[1, :])

# ============================================================
# Panel A: Protocol Sensitivity
# ============================================================
ax = ax_a

protocols = ['LOO', 'In-sample', 'Global\npooled']
qwen_fe = [0.720, 0.588, 0.536]
llama_fe = [0.747, 0.579, 0.571]
qwen_sig = [0.626, 0.685, 0.538]
llama_sig = [0.678, 0.742, 0.636]

x = np.arange(3)
w = 0.18

ax.bar(x - 1.5*w, qwen_fe, w, color='#4C72B0', label='Final ent. (Qwen)', edgecolor='white', linewidth=0.5)
ax.bar(x - 0.5*w, llama_fe, w, color='#4C72B0', alpha=0.5, label='Final ent. (Llama)', edgecolor='white', linewidth=0.5)
ax.bar(x + 0.5*w, qwen_sig, w, color='#DD8452', label='Sig 2D (Qwen)', edgecolor='white', linewidth=0.5)
ax.bar(x + 1.5*w, llama_sig, w, color='#DD8452', alpha=0.5, label='Sig 2D (Llama)', edgecolor='white', linewidth=0.5)

ax.set_ylabel('Median AUROC')
ax.set_xticks(x)
ax.set_xticklabels(protocols)
ax.set_ylim(0.30, 0.95)
ax.set_yticks([0.4, 0.6, 0.8])
ax.set_title('(a) Protocol Sensitivity', fontweight='bold', pad=6)
ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5)
ax.legend(loc='lower left', framealpha=0.9, ncol=2, fontsize=7, columnspacing=0.5)

# Delta annotation lowered
ax.annotate('', xy=(2, 0.785), xytext=(0, 0.785),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.2))
ax.text(1, 0.788, r'$\Delta$ up to 0.18', ha='center', va='bottom', fontsize=8, color='red')

# ============================================================
# Panel C: Signal Localization (Prefix Truncation)
# ============================================================
ax = ax_c

pct = [10, 30, 50, 70, 90, 100]
qwen_fe_trunc = [0.585, 0.571, 0.590, 0.603, 0.573, 0.608]
qwen_me_trunc = [0.611, 0.580, 0.602, 0.597, 0.619, 0.650]
llama_fe_trunc = [0.587, 0.626, 0.606, 0.639, 0.617, 0.608]
llama_me_trunc = [0.612, 0.615, 0.609, 0.626, 0.615, 0.626]

ax.plot(pct, qwen_fe_trunc, 'o-', color='#4C72B0', markersize=3.5, linewidth=1.3, label='Final ent. (Qwen)')
ax.plot(pct, qwen_me_trunc, 's--', color='#4C72B0', markersize=3.5, linewidth=1.1, alpha=0.6, label='Mean ent. (Qwen)')
ax.plot(pct, llama_fe_trunc, 'o-', color='#C44E52', markersize=3.5, linewidth=1.3, label='Final ent. (Llama)')
ax.plot(pct, llama_me_trunc, 's--', color='#C44E52', markersize=3.5, linewidth=1.1, alpha=0.6, label='Mean ent. (Llama)')

ax.axhline(0.60, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.text(8, 0.555, 'Null', fontsize=7, color='gray', va='bottom')

ax.set_xlabel('Sequence prefix used (%)')
ax.set_ylabel('Median AUROC (Hard)')
ax.set_xlim(0, 110)
ax.set_ylim(0.54, 0.70)
ax.set_title('(c) Signal Localization', fontweight='bold', pad=6)
ax.legend(loc='upper left', framealpha=0.85, fontsize=7,
          fancybox=True, edgecolor='lightgray')

# ============================================================
# Panel B: Performance Band (spans full width)
# ============================================================
ax = ax_b

methods = ['Sig 2D', 'OT Bary.', 'Fin. ent.', 'First ent.', 'Ent. mean',
           'Ent. std', 'Ent. slope', 'Seq. len.', 'Last 10%', 'Top-5 lin.']
qwen_med = [0.602, 0.650, 0.720, 0.726, 0.655, 0.642, 0.645, 0.660, 0.663, 0.665]
qwen_lo  = [0.581, 0.605, 0.644, 0.685, 0.621, 0.597, 0.598, 0.609, 0.620, 0.620]
qwen_hi  = [0.667, 0.701, 0.771, 0.768, 0.706, 0.665, 0.667, 0.704, 0.688, 0.710]
llama_med = [0.673, 0.605, 0.747, 0.671, 0.654, 0.650, 0.616, 0.622, 0.663, 0.640]
llama_lo  = [0.618, 0.575, 0.695, 0.640, 0.617, 0.622, 0.595, 0.580, 0.612, 0.614]
llama_hi  = [0.717, 0.634, 0.799, 0.697, 0.691, 0.676, 0.652, 0.671, 0.711, 0.680]

y = np.arange(len(methods))
h = 0.35

ax.barh(y - h/2, qwen_med, height=h, color='#4C72B0', alpha=0.8, label='Qwen', edgecolor='white', linewidth=0.3)
ax.barh(y + h/2, llama_med, height=h, color='#C44E52', alpha=0.8, label='Llama', edgecolor='white', linewidth=0.3)

for i in range(len(methods)):
    ax.plot([qwen_lo[i], qwen_hi[i]], [y[i] - h/2, y[i] - h/2], color='#4C72B0', linewidth=0.9, alpha=0.5)
    ax.plot([llama_lo[i], llama_hi[i]], [y[i] + h/2, y[i] + h/2], color='#C44E52', linewidth=0.9, alpha=0.5)

# Null baselines — distinct hues from bars so they stand out as reference lines
ax.axvline(0.60, color='#2980B9', linestyle='--', linewidth=1.5, alpha=0.95)
ax.axvline(0.58, color='#C0392B', linestyle='--', linewidth=1.5, alpha=0.95)

# Performance band shading
ax.axvspan(0.60, 0.75, alpha=0.10, color='orange')

# Callouts placed in the upper-right blank space, right-aligned to the same x
# so their right edges line up vertically above the Qwen/Llama legend.
right_align_x = 0.845
ax.text(right_align_x, len(methods) - 0.4, 'Performance band\n(0.60–0.75)', fontsize=7.5,
        color='#B07000', style='italic', va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#E0C070', alpha=0.9, linewidth=0.5))

ax.text(right_align_x, len(methods) - 2.6, 'Null baseline\nQwen: 0.60  |  Llama: 0.58', fontsize=7,
        color='#555555', va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='lightgray', alpha=0.9, linewidth=0.5))

ax.set_xlabel('Median AUROC (Hard)')
ax.set_yticks(y)
ax.set_yticklabels(methods)
ax.set_xlim(0.52, 0.85)
ax.set_ylim(-1.2, len(methods) - 0.3)
ax.set_title('(b) Method Convergence', fontweight='bold', pad=6)
ax.legend(loc='lower right', framealpha=0.9, fontsize=8.5)

fig.savefig('paper/fig_overview_icml.pdf', bbox_inches='tight')
fig.savefig('paper/fig_overview_icml.png', bbox_inches='tight', dpi=300)
print("Saved fig_overview_icml.pdf and fig_overview_icml.png")
