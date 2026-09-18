"""Export a compact prediction cache for the dashboard + chatbot (local demo only).

Runs the chosen finished run (default R4, the best model per Gate E) once over the
full test period and saves pred/true flow + class arrays, so the Streamlit app and
the chatbot's tools never need torch/conglib at runtime — just this .npz plus the
existing factor/meta/csv files already on disk.

Usage: python scripts/export_predictions.py [RUN ...]   (default: R4)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'work'
REPO = WORK / 'LargeST'
DATA = REPO / 'data' / 'sd'
DRIVE_PROJ = ROOT / 'data' / 'traffic' / 'congestion'
OUT = DRIVE_PROJ / 'predictions'

sys.path.insert(0, str(WORK / 'cong'))
import conglib  # noqa: E402

RUNS = sys.argv[1:] or ['R4']


def main():
    DEVICE = conglib.pick_device()
    print(f'device={DEVICE}')

    flow_df = pd.read_hdf(DATA / 'sd_his_2019.h5', key='t')
    meta = pd.read_csv(DATA / 'sd_meta.csv')
    adj = np.load(DATA / 'sd_rn_adj.npy')
    idx = {k: np.load(DATA / '2019' / f'idx_{k}.npy') for k in ('train', 'val', 'test')}
    F = flow_df.to_numpy(dtype=np.float32)
    test_starts = idx['test']

    LAB = np.repeat(np.load(DRIVE_PROJ / 'factors' / 'cong_label.npz')['label_hour'], 4, axis=0)

    OUT.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        t_paths = flow_df.index[test_starts + 1]  # target time of horizon h=1 for each window
        model, data, mean, std, args = conglib.load_run(run, DATA, DRIVE_PROJ, REPO, DEVICE, adj, F, LAB=LAB)
        mt = args['task'] == 'mt'
        if not mt:
            print(f'{run}: flow-only run, skipping (dashboard needs a multitask run for class predictions)')
            continue
        pf, tf, pc, tc = conglib.predict_mt(model, data, test_starts, mean, std, False,
                                            bs=16 if DEVICE.type == 'mps' else 64)
        pf, tf, pc, tc = (a.cpu().numpy() for a in (pf, tf, pc, tc))
        del model, data
        conglib.reset_mem(DEVICE)

        p = OUT / f'{run}_test.npz'
        np.savez_compressed(
            p,
            pred_flow=pf.astype(np.float16), true_flow=tf.astype(np.float16),
            pred_class=pc.astype(np.int8), true_class=tc.astype(np.int8),
            test_starts=test_starts.astype(np.int32),
            target_times_h1=t_paths.values.astype('datetime64[m]'),
            sensor_ids=meta['ID'].to_numpy(),
        )
        print(f'{run}: saved {p} ({p.stat().st_size / 2**20:.1f} MiB)')


if __name__ == '__main__':
    main()
