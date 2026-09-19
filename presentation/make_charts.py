"""Charts for the progress deck, drawn from the saved result files (light theme, sized for 16:9 slides)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'data' / 'traffic' / 'congestion' / 'results'
OUT = Path(__file__).resolve().parent / 'charts'
OUT.mkdir(exist_ok=True)

INK, INK2, MUTED, GRID, AXIS, REF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#6f6d68'
C = {'R0': '#2a78d6', 'A1': '#eb6834', 'A3': '#1baf7a', 'LSTM': '#eda100', 'occ': '#4a3aa7', 'flow': '#e34948'}
plt.rcParams.update({'font.family': ['Helvetica Neue', 'Arial', 'DejaVu Sans'], 'font.size': 12, 'axes.edgecolor': AXIS,
                     'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'savefig.dpi': 220, 'savefig.bbox': 'tight', 'figure.facecolor': 'white'})

final = {r: json.load(open(RES / f'{r}_final.json'))['final'] for r in ('R0', 'A1', 'A3')}
LSTM = {1: 13.72, 2: 16.65, 4: 21.16, 6: 25.79, 8: 29.68, 12: 37.45}
PAPER = {3: 15.24, 6: 17.74, 12: 21.56}
HA = json.load(open(RES / 'gate0.json'))['mae_table']['avg']['HA profile (ours)']
hlab = lambda h: f'{h * 15} min' if h * 15 < 60 else f'{h * 15 / 60:g} h'

# 1. test MAE by horizon
fig, ax = plt.subplots(figsize=(7.6, 4.3))
hs = np.arange(1, 13)
ends = []
for k, name in (('R0', 'GWNet R0 (full graph)'), ('A1', 'A1 road graph only'), ('A3', 'A3 no graph')):
    v = final[k]['test']['MAE']
    ax.plot(hs, v, color=C[k], lw=2.4, label=name, solid_capstyle='round')
    ends.append((v[-1], f'{k} {v[-1]:.1f}', C[k]))
ax.plot(list(LSTM), list(LSTM.values()), color=C['LSTM'], lw=2.4, marker='o', ms=6, label='Your LSTM baseline',
        markeredgecolor='white', markeredgewidth=1.5)
ends.append((LSTM[12], f'LSTM {LSTM[12]:.1f}', C['LSTM']))
ax.scatter(list(PAPER), list(PAPER.values()), color=REF, s=46, zorder=5, label='Paper GWNet (~80 epochs)',
           edgecolor='white', linewidth=1.5)
ax.axhline(HA, color=REF, lw=1.5, ls='--')
ax.text(1.05, HA + 0.6, f'time-of-day profile {HA:.1f}', color=INK2, fontsize=10.5)
ends.sort()
for i in range(1, len(ends)):
    if ends[i][0] - ends[i - 1][0] < 1.6:
        ends[i] = (ends[i - 1][0] + 1.6, *ends[i][1:])
for y, t, c in ends:
    ax.text(12.25, y, t, va='center', color=INK, fontsize=10.5, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))
ax.set_xticks([1, 4, 6, 8, 12], [hlab(h) for h in (1, 4, 6, 8, 12)])
ax.set_xlim(0.7, 13.9)
ax.set_ylim(10, 42)
ax.set_xlabel('Forecast horizon')
ax.set_ylabel('Test MAE (vehicles / 5 min)')
ax.legend(frameon=False, fontsize=10.5, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3)
fig.savefig(OUT / 'horizon.png')
plt.close(fig)

# 2. average test MAE per model
rows = [('Paper GWNet (~80 ep, A6000)', 17.74, REF), ('R0 · GWNet full graph', final['R0']['test']['avg'][0], C['R0']),
        ('A1 · road graph only', final['A1']['test']['avg'][0], C['A1']), ('Your LSTM', 26.35, C['LSTM']),
        ('A3 · no graph', final['A3']['test']['avg'][0], C['A3']), ('Time-of-day profile', HA, REF)]
fig, ax = plt.subplots(figsize=(7.6, 4.0))
y = np.arange(len(rows))[::-1]
ax.set_axisbelow(True)
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
for yi, (n, v, _) in zip(y, rows):
    ax.text(v + 0.35, yi, f'{v:.2f}', va='center', color=INK, fontsize=11.5)
ax.set_yticks(y, [r[0] for r in rows], color=INK, fontsize=11.5)
ax.set_xlim(0, 34)
ax.set_xlabel('Average test MAE over 12 horizons (vehicles / 5 min) — lower is better')
ax.grid(axis='y', visible=False)
ax.tick_params(axis='y', length=0)
fig.savefig(OUT / 'models_bar.png')
plt.close(fig)

# 3. occupancy label vs flow-only label on regular vs holiday days (Stage L, cell L6 output)
groups = ['Regular test weekdays', 'Holiday-period weekdays', 'Christmas Day']
occ = [8.52, 2.93, 0.03]
flw = [2.65, 17.63, 46.80]
fig, ax = plt.subplots(figsize=(7.6, 4.0))
ax.set_axisbelow(True)
x = np.arange(3)
b1 = ax.bar(x - 0.19, occ, 0.36, color=C['occ'], label='Occupancy label (PeMS): congested')
b2 = ax.bar(x + 0.19, flw, 0.36, color=C['flow'], label='Flow-only label: "congested"')
for bars in (b1, b2):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f'{v:.2f}%' if v < 0.1 else f'{v:.1f}%', ha='center', color=INK, fontsize=11)
ax.set_xticks(x, groups, color=INK)
ax.set_ylabel('Share of sensor time-steps (%)')
ax.set_ylim(0, 52)
ax.grid(axis='x', visible=False)
ax.legend(frameon=False, loc='upper left', fontsize=11)
fig.savefig(OUT / 'labels.png')
plt.close(fig)

# 4. timeline to Sept 21
tasks = [('Training queue R2 → R6 (M2 Air, overnight)', '2026-09-15', '2026-09-17', 'M1'),
         ('Final evaluation · ablation table · strata', '2026-09-17', '2026-09-18', 'M1'),
         ('Prediction export for the dashboard', '2026-09-16', '2026-09-18', 'M2'),
         ('Streamlit dashboard: map, drill-down, results', '2026-09-16', '2026-09-19', 'M3'),
         ('Gemini chatbot: tools, grounding, UI', '2026-09-16', '2026-09-19', 'M4'),
         ('Integration & end-to-end QA', '2026-09-19', '2026-09-20', 'M5 + all'),
         ('Deck, demo script, backup video', '2026-09-19', '2026-09-20', 'M5'),
         ('Rehearsal', '2026-09-20', '2026-09-20', 'all')]
fig, ax = plt.subplots(figsize=(11.2, 4.2))
for i, (name, a, b, who) in enumerate(tasks):
    s, e = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(hours=20)
    yi = len(tasks) - 1 - i
    ax.barh(yi, (e - s).total_seconds() / 86400, left=mdates.date2num(s), height=0.58, color='#86b6ef')
    ax.text(mdates.date2num(s) + 0.06, yi, f'{name}  ·  {who}', va='center', color=INK, fontsize=10.5)
for d, lab in (('2026-09-15', 'Today: progress review'), ('2026-09-21', 'Final demo')):
    xd = mdates.date2num(pd.Timestamp(d) + pd.Timedelta(hours=10))
    ax.axvline(xd, color=INK, lw=1.4)
    left = d.endswith('15')
    ax.text(xd + (0.05 if left else -0.05), len(tasks) - 0.25, lab, ha='left' if left else 'right', color=INK,
            fontsize=11, fontweight='bold')
ax.set_ylim(-0.7, len(tasks) + 0.2)
ax.set_xlim(mdates.date2num(pd.Timestamp('2026-09-15')), mdates.date2num(pd.Timestamp('2026-09-22')))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.grid(axis='y', visible=False)
fig.savefig(OUT / 'gantt.png')
plt.close(fig)
print('charts written to', OUT)
