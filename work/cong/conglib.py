# ── 07 · conglib.py — shared model/training code (notebook + train.py) ─────
"""Shared training code for congestion_sd — imported by the notebook and by train.py (nohup runs).

Runs on CUDA (Colab T4), Apple MPS (M2) or CPU."""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

T_IN, HORIZON = 12, 12


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if os.environ.get('CONG_DEVICE'):                      # e.g. CONG_DEVICE=cpu for bitwise-reproducible tests
        return torch.device(os.environ['CONG_DEVICE'])
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def reset_mem(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def mem_gib(device):
    """CUDA: peak allocated since reset_mem. MPS: memory the Metal driver currently holds (incl. cache). CPU: nan."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated() / 2**30
    if device.type == 'mps':
        return torch.mps.driver_allocated_memory() / 2**30
    return float('nan')


def is_oom(err):
    return isinstance(err, torch.cuda.OutOfMemoryError) or 'out of memory' in str(err).lower()


class Windows:
    """Whole (T, N, C) input tensor and raw (T, N) flow targets on one device; batches are index gathers.

    Window semantics match LargeST's DataLoader: x = inputs[t-11 .. t], y = flow[t+1 .. t+12].
    Targets live in their own array, so no covariate channel can ever reach the loss or the scaler.
    """

    def __init__(self, X, Y, device, x_dtype=torch.float32, Yc=None, time=None, static=None, wx=None, wx_w=None):
        # Optional factored inputs, joined per batch so the full (T, N, C) tensor never exists:
        # time (T, Ct) shared by all sensors, static (N, Cs) shared by all steps, wx (T, S, k) weather stations
        # mixed onto sensors by wx_w (N, S).
        self.X = torch.as_tensor(np.asarray(X), dtype=x_dtype).to(device)
        self.Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32).to(device)
        self.Yc = None if Yc is None else torch.as_tensor(np.asarray(Yc), dtype=torch.long).to(device)
        f32 = lambda a: None if a is None else torch.as_tensor(np.asarray(a), dtype=torch.float32).to(device)
        self.Xt, self.Xs, self.Ww = f32(time), f32(static), f32(wx_w)
        self.Xw = None if wx is None else torch.as_tensor(np.asarray(wx), dtype=torch.float16).to(device)
        self.x_off = torch.arange(-(T_IN - 1), 1, device=device)
        self.y_off = torch.arange(1, HORIZON + 1, device=device)
        self.n_channels = (self.X.shape[-1] + (0 if self.Xt is None else self.Xt.shape[-1]) +
                           (0 if self.Xs is None else self.Xs.shape[-1]) + (0 if self.Xw is None else self.Xw.shape[-1]))

    def batch(self, starts):
        starts = torch.as_tensor(np.asarray(starts), device=self.X.device)
        ti = starts[:, None] + self.x_off
        parts = [self.X[ti].float()]                          # (B, T_IN, N, Cd)
        b, L, n = ti.shape[0], ti.shape[1], self.X.shape[1]
        if self.Xt is not None:
            parts.append(self.Xt[ti][:, :, None, :].expand(b, L, n, -1))
        if self.Xs is not None:
            parts.append(self.Xs[None, None].expand(b, L, -1, -1))
        if self.Xw is not None:
            parts.append(torch.einsum('blsk,ns->blnk', self.Xw[ti].float(), self.Ww))
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)      # (B, T_IN, N, C)
        y = self.Y[starts[:, None] + self.y_off]             # (B, HORIZON, N) raw veh/5min
        if self.Yc is None:
            return x, y
        return x, y, self.Yc[starts[:, None] + self.y_off]   # + (B, HORIZON, N) class labels, -1 = ignore


def masked_mae(pred, y):
    """LargeST's masked MAE: cells whose raw label is 0 (missing) are excluded."""
    m = (y != 0).float()
    return ((pred - y).abs() * m).sum() / m.sum().clamp_min(1)


def build_gwnet(adj, adj_type, adp_adj, input_dim, device, repo, dropout=0.3,
                init_dim=32, skip_dim=256, end_dim=512):
    """LargeST's GWNET with its default SD hyper-parameters; adj_type='none' drops the road graph."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.models.gwnet import GWNET
    from src.utils.graph_algo import normalize_adj_mx
    mats = [] if adj_type == 'none' else normalize_adj_mx(adj, adj_type)
    supports = [torch.tensor(np.asarray(a), dtype=torch.float32, device=device) for a in mats]
    return GWNET(node_num=adj.shape[0], input_dim=input_dim, output_dim=1, supports=supports,
                 adp_adj=adp_adj, dropout=dropout, residual_channels=init_dim, dilation_channels=init_dim,
                 skip_channels=skip_dim, end_channels=end_dim).to(device)


def forward_raw(model, x, mean, std, amp):
    """Model output (B, HORIZON, N, 1) -> raw-scale flow (B, HORIZON, N) in fp32."""
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
        out = model(x)
    return out.float().squeeze(-1) * std + mean


def train_step(model, opt, scaler, x, y, mean, std, amp, clip=5.0, micro_bs=None):
    """One optimiser step on the batch (x, y); returns its masked MAE.

    With micro_bs, the batch is processed in chunks whose losses are each divided by the valid-cell count
    of the *whole* batch, so the accumulated gradient equals the full-batch masked-MAE gradient exactly.
    Only BatchNorm differs: it normalises over each micro-batch instead of the full batch.
    """
    model.train()
    opt.zero_grad(set_to_none=True)
    n_valid = (y != 0).sum().clamp_min(1)
    chunk = micro_bs or len(x)
    total = 0.0
    for i in range(0, len(x), chunk):
        yb = y[i:i + chunk]
        pred = forward_raw(model, x[i:i + chunk], mean, std, amp)
        loss = ((pred - yb).abs() * (yb != 0)).sum() / n_valid
        scaler.scale(loss).backward()
        total += loss.item()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(opt)
    scaler.update()
    return total


@torch.no_grad()
def predict(model, data, starts, mean, std, amp, bs=64):
    """Eval-mode raw predictions and labels for the given window starts: (S, HORIZON, N) each."""
    model.eval()
    preds, labels = [], []
    for i in range(0, len(starts), bs):
        x, y = data.batch(starts[i:i + bs])
        preds.append(forward_raw(model, x, mean, std, amp))
        labels.append(y)
    return torch.cat(preds), torch.cat(labels)


@torch.no_grad()
def evaluate(model, data, starts, mean, std, amp, bs=64):
    """Per-horizon masked MAE / RMSE / MAPE over the given windows (LargeST protocol) -> (3, HORIZON) array.

    Sums are accumulated batch by batch in float64 on the CPU (MPS has no float64), so full val/test
    sets never have to sit in device memory."""
    model.eval()
    s_abs = s_sq = s_pct = cnt = 0
    for i in range(0, len(starts), bs):
        x, y = data.batch(starts[i:i + bs])
        m = y != 0
        e = torch.where(m, forward_raw(model, x, mean, std, amp) - y, torch.zeros_like(y))
        s_abs = s_abs + e.abs().sum(dim=(0, 2)).cpu().double()
        s_sq = s_sq + (e ** 2).sum(dim=(0, 2)).cpu().double()
        s_pct = s_pct + (e.abs() / torch.where(m, y, torch.ones_like(y))).sum(dim=(0, 2)).cpu().double()
        cnt = cnt + m.sum(dim=(0, 2)).cpu().double()
    cnt = cnt.clamp_min(1)
    return torch.stack([s_abs / cnt, (s_sq / cnt).sqrt(), s_pct / cnt]).numpy()


# ── multi-task (flow + congestion class) ─────────────────────────────────────
class GWNetMT(torch.nn.Module):
    """GWNet trunk with two 1x1 heads on the shared end_conv_1 features: flow (the unchanged LargeST head)
    and congestion-class logits for every sensor and horizon."""

    def __init__(self, base, n_classes=3):
        super().__init__()
        self.base, self.K = base, n_classes
        self.cls = torch.nn.Conv2d(base.end_conv_2.in_channels, base.horizon * n_classes, kernel_size=(1, 1)).to(
            next(base.parameters()).device)                   # same device as the trunk
        self._feat = None
        base.end_conv_2.register_forward_pre_hook(self._grab)

    def _grab(self, module, inputs):
        self._feat = inputs[0]

    def forward(self, x):
        flow = self.base(x)                                   # (B, HORIZON, N, 1)
        logits = self.cls(self._feat)                         # (B, HORIZON*K, N, 1)
        b, _, n, _ = logits.shape
        return flow, logits.view(b, self.base.horizon, self.K, n).permute(0, 1, 3, 2)   # (B, HORIZON, N, K)


def forward_mt(model, x, mean, std, amp):
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
        flow, logits = model(x)
    return flow.float().squeeze(-1) * std + mean, logits.float()


def train_step_mt(model, opt, scaler, x, y, yc, mean, std, amp, lam, class_w, clip=5.0, micro_bs=None):
    """Masked MAE + lam * class-weighted CE (ignore -1), accumulated exactly over micro-batches:
    each term is normalised by its full-batch denominator (valid flow cells / summed class weights)."""
    model.train()
    opt.zero_grad(set_to_none=True)
    n_valid = (y != 0).sum().clamp_min(1)
    lab = yc >= 0
    w_total = class_w[yc.clamp_min(0)][lab].sum().clamp_min(1e-6)
    chunk = micro_bs or len(x)
    tot_mae = tot_ce = 0.0
    for i in range(0, len(x), chunk):
        yb, cb = y[i:i + chunk], yc[i:i + chunk]
        pred, logits = forward_mt(model, x[i:i + chunk], mean, std, amp)
        mae = ((pred - yb).abs() * (yb != 0)).sum() / n_valid
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), cb.reshape(-1),
                                                weight=class_w, ignore_index=-1, reduction='sum') / w_total
        scaler.scale(mae + lam * nll).backward()
        tot_mae += mae.item()
        tot_ce += nll.item()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(opt)
    scaler.update()
    return tot_mae, tot_ce


@torch.no_grad()
def evaluate_mt(model, data, starts, mean, std, amp, bs=64, n_classes=3):
    """Flow metrics (3, HORIZON) as in evaluate(), plus a per-horizon confusion matrix (HORIZON, K, K)
    with rows = true class, columns = predicted class (argmax), over labelled cells only."""
    model.eval()
    s_abs = s_sq = s_pct = cnt = 0
    conf = torch.zeros(HORIZON, n_classes, n_classes, dtype=torch.float64)
    for i in range(0, len(starts), bs):
        x, y, yc = data.batch(starts[i:i + bs])
        pred, logits = forward_mt(model, x, mean, std, amp)
        m = y != 0
        e = torch.where(m, pred - y, torch.zeros_like(y))
        s_abs = s_abs + e.abs().sum(dim=(0, 2)).cpu().double()
        s_sq = s_sq + (e ** 2).sum(dim=(0, 2)).cpu().double()
        s_pct = s_pct + (e.abs() / torch.where(m, y, torch.ones_like(y))).sum(dim=(0, 2)).cpu().double()
        cnt = cnt + m.sum(dim=(0, 2)).cpu().double()
        yhat = logits.argmax(-1)
        for h in range(HORIZON):
            t, p = yc[:, h].reshape(-1), yhat[:, h].reshape(-1)
            keep = t >= 0
            conf[h] += torch.bincount((t[keep] * n_classes + p[keep]).cpu(), minlength=n_classes ** 2).view(
                n_classes, n_classes).double()
    cnt = cnt.clamp_min(1)
    return torch.stack([s_abs / cnt, (s_sq / cnt).sqrt(), s_pct / cnt]).numpy(), conf.numpy()


def class_scores(conf):
    """conf (..., K, K) rows = truth -> per-class precision, recall, F1 and macro-F1 (last axis = class)."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    prec = tp / np.maximum(conf.sum(-2), 1)
    rec = tp / np.maximum(conf.sum(-1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return prec, rec, f1, f1.mean(-1)


# ── Stage E · evaluation primitives (load a finished run; single-pass caching for stratified eval) ──
def load_run(run, data_dir, proj_dir, repo, device, adj, F, LAB=None):
    """Rebuild a finished run's exact model + Windows object from its own ckpt/<run>/best.pt — inference
    only, the checkpoint is only read. `adj` and `F` (raw flow targets) are passed in since they're the
    same for every run; `LAB` (congestion class labels, or None) is only used if the run is multi-task.
    Returns (model, data, mean, std, args) where args is the training-time argparse dict from the checkpoint."""
    data_dir, proj_dir = Path(data_dir), Path(proj_dir)
    ck = torch.load(proj_dir / 'ckpt' / run / 'best.pt', map_location='cpu', weights_only=False)
    args = ck['state']['args']
    args.setdefault('task', 'flow')                     # R0/A1/A3 predate --task/--inputs/--lam (train.py's own
    args.setdefault('inputs', 'R0')                      # argparse defaults — those runs are genuinely flow-only,
    args.setdefault('lam', 15.0)                         # R0-input runs, so the defaults are the correct fill-in)
    mt = args['task'] == 'mt'
    with np.load(data_dir / '2019' / 'his.npz') as z:
        mean, std = float(z['mean']), float(z['std'])
    if args['inputs'] == 'R0':
        with np.load(data_dir / '2019' / 'his.npz') as z:
            X = z['data'].astype(np.float32)
        data = Windows(X, F, device, Yc=LAB if mt else None)
        in_dim = 3
    else:
        import features
        fin = features.assemble(args['inputs'], data_dir, proj_dir / 'factors')
        data = Windows(fin['dyn'], F, device, x_dtype=torch.float16, Yc=LAB if mt else None,
                       time=fin['time'], static=fin['static'], wx=fin['wx'], wx_w=fin['wx_w'])
        in_dim = data.n_channels
    set_seed(args['seed'])
    model = build_gwnet(adj, args['adj_type'], args['adp'], input_dim=in_dim, device=device, repo=repo,
                        dropout=args['dropout'])
    if mt:
        model = GWNetMT(model).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, data, mean, std, args


@torch.no_grad()
def predict_mt(model, data, starts, mean, std, amp, bs=64):
    """Like predict(), but for a multi-task model: also returns the predicted class (argmax) and the
    true class labels, so a single forward pass over the test set can drive every stratified/bootstrap
    number downstream in plain numpy. Returns (pred_flow, true_flow, pred_class, true_class), each
    (S, HORIZON, N); the class arrays are int8 (values are only -1/0/1/2)."""
    model.eval()
    pf, tf, pc, tc = [], [], [], []
    for i in range(0, len(starts), bs):
        x, y, yc = data.batch(starts[i:i + bs])
        flow, logits = forward_mt(model, x, mean, std, amp)
        pf.append(flow)
        tf.append(y)
        pc.append(logits.argmax(-1).to(torch.int8))
        tc.append(yc.to(torch.int8))
    return torch.cat(pf), torch.cat(tf), torch.cat(pc), torch.cat(tc)
