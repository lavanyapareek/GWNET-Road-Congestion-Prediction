"""Additional charts for the LaTeX end-to-end report (same visual language as make_charts.py),
covering the full ablation table (R0-R6, A1-A3) and the Gate E bootstrap deltas — neither of
which existed yet when make_charts.py was written for the 2026-09-15 progress review."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'charts'
OUT.mkdir(exist_ok=True)

INK, INK2, MUTED, GRID, AXIS, REF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#6f6d68'
BEST, SIG, NOTSIG = '#1baf7a', '#2a78d6', '#c3c2b7'
plt.rcParams.update({'font.family': ['Helvetica Neue', 'Arial', 'DejaVu Sans'], 'font.size': 12, 'axes.edgecolor': AXIS,
                     'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'savefig.dpi': 220, 'savefig.bbox': 'tight', 'figure.facecolor': 'white'})

# 1. Full ablation bar chart (R0-R6, A1-A3, plus baselines), R4 highlighted as the shipped model.
rows = [
    ('Time-of-day profile', 29.8, REF),
    ('Your LSTM baseline', 26.35, REF),
    ('A3 · no graph', 27.921, MUTED),
    ('A1 · road graph only', 22.901, MUTED),
    ('A2 · adaptive graph only', 21.841, MUTED),
    ('R0 · full graph, flow-only', 20.347, MUTED),
    ('R1 · + congestion head', 20.406, MUTED),
    ('R2 · + missingness fix', 20.388, MUTED),
    ('R3 · + utilization/time/static', 18.903, SIG),
    ('R4 · + holiday/school calendar', 18.557, BEST),
    ('R5 · + weather', 18.637, MUTED),
    ('R6 · + incidents', 18.701, MUTED),
    ('Paper GWNet (~80 ep, A6000)', 17.74, REF),
]
fig, ax = plt.subplots(figsize=(8.0, 5.6))
y = np.arange(len(rows))[::-1]
ax.set_axisbelow(True)
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.64)
for yi, (n, v, _) in zip(y, rows):
    ax.text(v + 0.35, yi, f'{v:.2f}', va='center', color=INK, fontsize=10.5)
ax.set_yticks(y, [r[0] for r in rows], color=INK, fontsize=10.5)
ax.set_xlim(0, 34)
ax.set_xlabel('Average test MAE over 12 horizons (vehicles / 5 min) — lower is better')
ax.grid(axis='y', visible=False)
ax.tick_params(axis='y', length=0)
fig.savefig(OUT / 'ablation_all.png')
plt.close(fig)

# 2. Gate E bootstrap deltas — a "football field" forest plot.
deltas = [
    ('R1 → R2  (missingness fix)', -0.018, -0.140, 0.135, False),
    ('R2 → R3  (utilization / time / static)', -1.486, -1.811, -1.233, True),
    ('R3 → R4  (holiday / school calendar)', -0.346, -0.604, -0.131, True),
    ('R4 → R5  (weather)', 0.080, -0.082, 0.264, False),
    ('R4 → R6  (incidents)', 0.144, -0.029, 0.343, False),
    ('A3 → A1  (road graph)', -5.020, -5.732, -4.345, True),
    ('A3 → A2  (adaptive graph)', -6.079, -6.939, -5.274, True),
    ('A3 → R0  (both graphs)', -7.574, -8.488, -6.680, True),
]
fig, ax = plt.subplots(figsize=(8.0, 4.6))
y = np.arange(len(deltas))[::-1]
for yi, (name, point, lo, hi, sig) in zip(y, deltas):
    color = SIG if sig else NOTSIG
    ax.plot([lo, hi], [yi, yi], color=color, lw=2.4, solid_capstyle='round')
    ax.scatter([point], [yi], color=color, s=60, zorder=5, edgecolor='white', linewidth=1.2)
ax.axvline(0, color=INK, lw=1.2, ls='--')
ax.set_yticks(y, [d[0] for d in deltas], color=INK, fontsize=10.5)
ax.set_xlabel(r'$\Delta$ test MAE, 95% CI (1000-resample day-block bootstrap) — negative = improvement')
ax.grid(axis='y', visible=False)
ax.tick_params(axis='y', length=0)
handles = [plt.Line2D([0], [0], color=SIG, lw=3, label='95% CI excludes zero (significant)'),
           plt.Line2D([0], [0], color=NOTSIG, lw=3, label='95% CI crosses zero (not significant)')]
ax.legend(handles=handles, frameon=False, fontsize=10.5, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)
fig.savefig(OUT / 'bootstrap_forest.png')
plt.close(fig)

print('report charts written to', OUT)
