# ── 12 · train.py — resumable background trainer (checkpoints every quarter-epoch) ──
"""Resumable GWNet training for the R/A runs — launched in the background (nohup / caffeinate) by cell 14.

Every quarter-epoch: evaluate on a validation subsample, write <proj>/ckpt/<run>/last.pt (and best.pt when
the validation MAE improves) atomically, and append to <proj>/results/<run>_history.json. Re-running the
same command resumes from last.pt; each epoch's shuffle is derived from (seed, epoch), so a resumed run
visits exactly the windows an uninterrupted run would.

LR: linear warmup -> constant (LargeST's 1e-3) -> cosine decay over the last `decay_frac` of the run.
The total length can therefore still be changed on resume as long as the decay has not started.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conglib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--data', required=True, help='LargeST data/sd folder')
ap.add_argument('--proj', required=True, help='project results folder (ckpt/, results/ live here)')
ap.add_argument('--repo', required=True, help='LargeST repo (for src.models.gwnet)')
ap.add_argument('--epochs', type=float, default=12)
ap.add_argument('--stop-after-quarters', type=int, default=0, help='pause after this many quarters (0 = run to the end)')
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--wd', type=float, default=1e-4)
ap.add_argument('--dropout', type=float, default=0.3)
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--adj-type', default='doubletransition')
ap.add_argument('--adp', type=int, default=1)
ap.add_argument('--warmup-quarters', type=float, default=1)
ap.add_argument('--decay-frac', type=float, default=0.3)
ap.add_argument('--val-stride', type=int, default=4)
ap.add_argument('--seed', type=int, default=2023)
ap.add_argument('--task', choices=['flow', 'mt'], default='flow', help='mt = flow + congestion-class head')
ap.add_argument('--lam', type=float, default=15.0, help='weight of the class-CE term (task mt)')
ap.add_argument('--inputs', choices=['R0', 'R2', 'R3', 'R4', 'R5', 'R6'], default='R0', help='input feature level (features.py)')
ap.add_argument('--debug-steps', type=int, default=0, help='tests only: cap steps per quarter')
ap.add_argument('--skip-final', action='store_true', help='tests only: no full val/test evaluation at the end')
a = ap.parse_args()

T_IN, HORIZON = conglib.T_IN, conglib.HORIZON
ckpt_dir = Path(a.proj) / 'ckpt' / a.run
res_dir = Path(a.proj) / 'results'
ckpt_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)
(ckpt_dir / 'pid').write_text(str(os.getpid()))

def log(msg):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)

# ── data: R0 = LargeST's 3 input channels; targets = raw flow, kept separate from the inputs ──
DEVICE = conglib.pick_device()
AMP = DEVICE.type == 'cuda'
MICRO_BS = 8 if DEVICE.type == 'mps' else None
EVAL_BS = 16 if DEVICE.type == 'mps' else 64
with np.load(Path(a.data) / '2019' / 'his.npz') as z:
    X = z['data'].astype(np.float32)
    MEAN, STD = float(z['mean']), float(z['std'])
F = pd.read_hdf(Path(a.data) / 'sd_his_2019.h5', key='t').to_numpy(np.float32)
assert np.abs(X[..., 0] * STD + MEAN - F).max() < 1e-2, 'input channel 0 is not the normalised target'
idx = {k: np.load(Path(a.data) / '2019' / f'idx_{k}.npy') for k in ('train', 'val', 'test')}
TRAIN_END = int(idx['val'][0] - T_IN)
MT = a.task == 'mt'
if MT:                                                # hourly occupancy classes -> 15-min grid (see Stage L)
    LAB = np.repeat(np.load(Path(a.proj) / 'factors' / 'cong_label.npz')['label_hour'], 4, axis=0)
    share = np.array([(LAB[:TRAIN_END] == k).sum() for k in range(3)], float)
    share /= share.sum()
    CLASS_W = torch.tensor(1 / np.sqrt(share) / (1 / np.sqrt(share)).mean(), dtype=torch.float32, device=DEVICE)
if a.inputs == 'R0':
    data = conglib.Windows(X, F, DEVICE, Yc=LAB if MT else None)
    IN_NAMES = ['flow', 'tod', 'dow/7']
else:
    import features
    fin = features.assemble(a.inputs, Path(a.data), Path(a.proj) / 'factors')
    ok = ~fin['miss']                                 # channel 0 = normalised flow; untouched where observed
    assert np.abs(fin['dyn'][..., 0][ok].astype(np.float32) * STD + MEAN - F[ok]).max() < 0.6, 'flow channel != target'
    data = conglib.Windows(fin['dyn'], F, DEVICE, x_dtype=torch.float16, Yc=LAB if MT else None, time=fin['time'],
                           static=fin['static'], wx=fin['wx'], wx_w=fin['wx_w'])
    IN_NAMES = fin['names']
    del fin
del X
adj = np.load(Path(a.data) / 'sd_rn_adj.npy')
val_sub = idx['val'][::a.val_stride]

# ── schedule ──────────────────────────────────────────────────────────────
total_q = int(round(4 * a.epochs))
spq = (len(idx['train']) // 4) // a.bs                          # optimiser steps per quarter
if a.debug_steps:
    spq = min(spq, a.debug_steps)
total_steps = total_q * spq
warm = int(a.warmup_quarters * spq)
decay_start = int((1 - a.decay_frac) * total_steps)

def lr_at(step):
    if step < warm:
        return a.lr * (0.1 + 0.9 * step / max(warm, 1))
    if step < decay_start:
        return a.lr
    return a.lr * 0.5 * (1 + math.cos(math.pi * (step - decay_start) / max(total_steps - decay_start, 1)))

# ── model / state (fresh or resumed) ──────────────────────────────────────
conglib.set_seed(a.seed)
IN_DIM = data.n_channels if a.inputs != 'R0' else 3
model = conglib.build_gwnet(adj, a.adj_type, a.adp, input_dim=IN_DIM, device=DEVICE, repo=a.repo, dropout=a.dropout)
if MT:
    model = conglib.GWNetMT(model).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
scaler = torch.amp.GradScaler(DEVICE.type, enabled=AMP)
state = {'q': 0, 'step': 0, 'best_val': float('inf'), 'history': [], 'args': vars(a)}
last = ckpt_dir / 'last.pt'
if last.exists():
    ck = torch.load(last, map_location='cpu', weights_only=False)
    old = ck['state']['args']
    for k in ('lr', 'wd', 'dropout', 'bs', 'adj_type', 'adp', 'seed', 'warmup_quarters', 'decay_frac', 'val_stride', 'task', 'lam', 'inputs'):
        assert old.get(k, vars(a)[k]) == vars(a)[k], f'--{k.replace("_", "-")} differs from the checkpoint ({old[k]} vs {vars(a)[k]})'
    old_decay = int((1 - old['decay_frac']) * int(round(4 * old['epochs'])) * spq)
    assert old['epochs'] == a.epochs or ck['state']['step'] < min(old_decay, decay_start), \
        'cannot change --epochs after the LR decay has started'
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    scaler.load_state_dict(ck['scaler'])
    torch.set_rng_state(ck['rng_cpu'])
    if DEVICE.type == 'mps' and ck.get('rng_mps') is not None:
        torch.mps.set_rng_state(ck['rng_mps'])
    if DEVICE.type == 'cuda' and ck.get('rng_cuda') is not None:
        torch.cuda.set_rng_state(ck['rng_cuda'])
    state = ck['state']
    state['args'] = vars(a)
    log(f'resumed {a.run} at quarter {state["q"]}/{total_q}, step {state["step"]}, best val {state["best_val"]:.3f}')
else:
    log(f'new run {a.run} ({a.task}, inputs {a.inputs}: {IN_DIM} channels): {total_q} quarters x {spq} steps (bs {a.bs}, micro {MICRO_BS}), device {DEVICE}, '
        f'params {sum(p.numel() for p in model.parameters()):,}, warmup {warm} steps, decay from step {decay_start}'
        + (f', lam {a.lam}, class weights {CLASS_W.cpu().numpy().round(3).tolist()}' if MT else ''))

def save(path, extra=None):
    tmp = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                'rng_cpu': torch.get_rng_state(),
                'rng_mps': torch.mps.get_rng_state() if DEVICE.type == 'mps' else None,
                'rng_cuda': torch.cuda.get_rng_state() if DEVICE.type == 'cuda' else None,
                'state': state, **(extra or {})}, tmp)
    os.replace(tmp, path)

# ── train ─────────────────────────────────────────────────────────────────
while state['q'] < total_q:
    if a.stop_after_quarters and state['q'] >= a.stop_after_quarters:
        log(f'paused after {state["q"]} quarters (--stop-after-quarters); re-run without it to continue')
        sys.exit(0)
    q = state['q']
    epoch, part = divmod(q, 4)
    perm = np.random.default_rng(a.seed + epoch).permutation(idx['train'])
    chunk = np.array_split(perm, 4)[part][:spq * a.bs].reshape(spq, a.bs)
    t0 = time.time()
    losses, ces = [], []
    for b in chunk:
        for g in opt.param_groups:
            g['lr'] = lr_at(state['step'])
        if MT:
            l, c = conglib.train_step_mt(model, opt, scaler, *data.batch(b), MEAN, STD, AMP, a.lam, CLASS_W, micro_bs=MICRO_BS)
            losses.append(l)
            ces.append(c)
        else:
            losses.append(conglib.train_step(model, opt, scaler, *data.batch(b), MEAN, STD, AMP, micro_bs=MICRO_BS))
        state['step'] += 1
    conglib.sync(DEVICE)
    t_train = time.time() - t0
    if MT:
        val, vconf = conglib.evaluate_mt(model, data, val_sub, MEAN, STD, AMP, bs=EVAL_BS)
        _, _, vf1, vmacro = conglib.class_scores(vconf)
    else:
        val = conglib.evaluate(model, data, val_sub, MEAN, STD, AMP, bs=EVAL_BS)
    state['q'] = q + 1
    rec = {'q': q + 1, 'epoch': (q + 1) / 4, 'step': state['step'], 'lr': lr_at(state['step'] - 1),
           'train_mae': float(np.mean(losses)), 'val_mae': float(val[0].mean()),
           'val_mae_h': [round(float(v), 3) for v in val[0]], 's_train': t_train,
           'ms_per_step': 1000 * t_train / len(chunk), 's_total': time.time() - t0}
    if MT:
        rec.update({'train_ce': float(np.mean(ces)), 'val_macro_f1': float(vmacro.mean()),
                    'val_f1_congested': float(vf1[:, 2].mean()), 'val_macro_f1_h': [round(float(v), 4) for v in vmacro]})
    state['history'].append(rec)
    improved = rec['val_mae'] < state['best_val']
    if improved:
        state['best_val'] = rec['val_mae']
        save(ckpt_dir / 'best.pt')
    save(last)
    (res_dir / f'{a.run}_history.json').write_text(
        json.dumps({'total_q': total_q, 'args': vars(a), 'history': state['history']}, indent=0))
    log(f"q {q + 1:3d}/{total_q} (ep {rec['epoch']:5.2f})  train {rec['train_mae']:7.3f}  val-sub {rec['val_mae']:7.3f}"
        f"{' *' if improved else '  '}  lr {rec['lr']:.2e}  {rec['ms_per_step']:.0f} ms/step  {rec['s_total']:.0f} s"
        + (f"  | ce {rec['train_ce']:.3f}  val macro-F1 {rec['val_macro_f1']:.3f}  F1(congested) {rec['val_f1_congested']:.3f}" if MT else ''))

# ── final: best checkpoint on the full val and test sets (LargeST protocol) ──
if a.skip_final:
    log('done (final evaluation skipped)')
    sys.exit(0)
best = torch.load(ckpt_dir / 'best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(best['model'])
final = {}
for split in ('val', 'test'):
    ix = idx[split][::50] if a.debug_steps else idx[split]
    if MT:
        m, conf = conglib.evaluate_mt(model, data, ix, MEAN, STD, AMP, bs=EVAL_BS)
    else:
        m = conglib.evaluate(model, data, ix, MEAN, STD, AMP, bs=EVAL_BS)
    final[split] = {'MAE': m[0].tolist(), 'RMSE': m[1].tolist(), 'MAPE': m[2].tolist(),
                    'avg': [float(m[0].mean()), float(m[1].mean()), float(m[2].mean())]}
    if MT:
        prec, recl, f1, macro = conglib.class_scores(conf)
        final[split]['cls'] = {'macro_f1_h': macro.tolist(), 'macro_f1': float(macro.mean()),
                               'precision': prec.mean(0).tolist(), 'recall': recl.mean(0).tolist(), 'f1': f1.mean(0).tolist(),
                               'confusion_h': conf.tolist(), 'classes': ['free', 'heavy', 'congested']}
        log(f"{split}: macro-F1 {macro.mean():.3f} (h1 {macro[0]:.3f}, h12 {macro[11]:.3f})  recall free/heavy/congested "
            f"{' / '.join(f'{v:.2f}' for v in recl.mean(0))}  precision {' / '.join(f'{v:.2f}' for v in prec.mean(0))}")
    log(f"{split}: avg MAE {final[split]['avg'][0]:.3f}  RMSE {final[split]['avg'][1]:.3f}  MAPE {final[split]['avg'][2]:.4f}"
        f"  | MAE h3 {m[0][2]:.2f} h6 {m[0][5]:.2f} h12 {m[0][11]:.2f}")
(res_dir / f'{a.run}_final.json').write_text(json.dumps({'final': final, 'best_val_sub': state['best_val'],
                                                          'args': vars(a)}, indent=1))
log('done')
