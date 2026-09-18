# ── F5 · features.py — assemble factored inputs for each ablation level ─────
"""Factored model inputs per ablation level, built from the Stage F factor files. Levels are cumulative:
R2 data fixes, R3 +4a, R4 +4b, R5 +4c weather, R6 +4d incidents. Channel 0 of `dyn` is always the
normalised flow (LargeST's mean/std); the flow *target* is never taken from these inputs."""
import numpy as np

LEVELS = ['R0', 'R2', 'R3', 'R4', 'R5', 'R6']


def assemble(level, data_dir, fact_dir):
    """-> dict(dyn (T,N,Cd) float16, time (T,Ct) | None, static (N,Cs) | None, wx (T,S,k) | None,
    wx_w (N,S) | None, miss (T,N) bool, names [dyn..., time..., static..., wx...])"""
    lv = LEVELS.index(level)
    assert lv >= 1, 'R0 uses his.npz directly'
    with np.load(data_dir / '2019' / 'his.npz') as z:
        mean, std = float(z['mean']), float(z['std'])
        largest_time = z['data'][:, 0, 1:3].astype(np.float32)          # LargeST's (tod, dow/7), same for every sensor
    r2 = np.load(fact_dir / 'inputs_r2.npz')
    dyn, names = [(r2['flow_in'] - mean) / std, r2['miss'].astype(np.float32)], ['flow', 'missing']
    time, t_names, static, s_names, wx, wx_w, w_names = largest_time, ['tod', 'dow/7'], None, [], None, None, []
    if lv >= 2:                                                          # R3: 4a
        a = np.load(fact_dir / 'inputs_4a.npz')
        dyn += [a['util'].astype(np.float32), a['fpl'].astype(np.float32)]
        names += ['util', 'flow_per_lane']
        time, t_names = a['time'], ['tod_sin', 'tod_cos'] + [f'dow_{i}' for i in range(7)]
        static, s_names = a['static'], [str(x) for x in a['static_names']]
    if lv >= 3:                                                          # R4: 4b (major holidays -> Sunday in the one-hot)
        b = np.load(fact_dir / 'inputs_4b.npz')
        time = np.concatenate([time[:, :2], b['dow_remap'], b['time']], axis=1)
        t_names = t_names[:2] + [f'dow_{i}' for i in range(7)] + [str(x) for x in b['names']]
    if lv >= 4:                                                          # R5: 4c weather
        c = np.load(fact_dir / 'inputs_4c.npz')
        wx, wx_w, w_names = c['wx'], c['w_idw'], [f'wx_{x}' for x in c['names']]
    if lv >= 5:                                                          # R6: 4d incidents (known at the forecast origin)
        i = np.load(fact_dir / 'incidents.npz')
        dyn += [np.minimum(i['active'], 3) / 3, i['collision'].astype(np.float32),
                np.log1p(i['since'].astype(np.float32)) / np.log1p(600), i['up_miles'].astype(np.float32) / 3]
        names += ['inc_active', 'inc_collision', 'inc_since', 'inc_up_miles']
    return {'dyn': np.stack(dyn, axis=-1).astype(np.float16), 'time': time, 'static': static, 'wx': wx, 'wx_w': wx_w,
            'miss': r2['miss'], 'names': names + t_names + s_names + w_names}
