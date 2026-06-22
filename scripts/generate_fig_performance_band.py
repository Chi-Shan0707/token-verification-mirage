#!/usr/bin/env python3
"""Performance band figure — clean horizontal bar chart."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 300,
})

methods = [
    'Markov 4×4', 'Top-5 lin.', 'Seq. len.', 'Ent. std',
    'Ent. mean', 'Sig 4D', 'OT Bary.', 'Sig 2D',
    'Ent. slope', 'Fourier 4D', 'Final ent.', 'MLP x-prob',
]
qm = [0.612,0.615,0.615,0.627,0.633,0.642,0.687,0.653,0.670,0.696,0.687,0.728]
lm = [0.654,0.645,0.645,0.650,0.727,0.679,0.654,0.667,0.699,0.633,0.714,0.749]

C_BLUE = '#4472C4'; C_RED = '#C44E52'; C_GRAY = '#7F7F7F'

fig, ax = plt.subplots(figsize=(6.8, 4.2))
y = np.arange(len(methods)); h = 0.33
ax.barh(y-h/2, qm, h, color=C_BLUE, alpha=0.85, label='Qwen2.5-Coder-7B', edgecolor='none')
ax.barh(y+h/2, lm, h, color=C_RED, alpha=0.85, label='Llama-3.1-8B', edgecolor='none')

ax.axvspan(0.61, 0.73, alpha=0.08, color='gray')
ax.axvline(0.50, color=C_GRAY, ls=':', lw=0.8)
ax.axvline(0.61, color='#777', ls='--', lw=0.8)

# Labels placed above bars (y > max method y)
ax.text(0.755, len(methods)-0.3, 'Hard-MLP  0.73 / 0.75', fontsize=8, color='#444', va='top', fontweight='bold')
ax.text(0.64, len(methods)+0.6, 'Observed band  0.61 – 0.73', fontsize=8.5, color='#888', style='italic', va='bottom')
ax.text(0.515, len(methods)+0.6, 'Permutation null ~0.61', fontsize=7.5, color='#777', va='bottom')

ax.set_xlabel('AUROC (within-problem LOO)')
ax.set_yticks(y); ax.set_yticklabels(methods)
ax.set_xlim(0.48, 0.82); ax.set_ylim(-0.6, len(methods)+0.6)
ax.set_title('Shallow Token-Level Features on Hard Math (BigMath)', fontweight='bold')
ax.legend(loc='lower right', framealpha=0.9, fontsize=9)

plt.tight_layout()
fig.savefig('paper/fig_performance_band.pdf', bbox_inches='tight')
fig.savefig('paper/fig_performance_band.png', bbox_inches='tight', dpi=300)
print("Saved fig_performance_band.pdf")
