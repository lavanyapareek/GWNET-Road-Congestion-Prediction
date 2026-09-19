"""Cached data access for the congestion dashboard + chatbot.

Everything here reads plain files (csv/npz/json) — no torch, no model code — so the
Streamlit app and the chatbot's tools stay lightweight. Predictions come from the cache
built by scripts/export_predictions.py (data/traffic/congestion/predictions/<RUN>_test.npz).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_SD = ROOT / 'work' / 'LargeST' / 'data' / 'sd'
CONG = ROOT / 'data' / 'traffic' / 'congestion'
PEMS = CONG / 'pems'
PRED = CONG / 'predictions'
RESULTS = CONG / 'results'
CHARTS = ROOT / 'presentation' / 'charts'

CLASSES = ['free', 'heavy', 'congested']
CLASS_COLORS = {'free': '#2ecc71', 'heavy': '#f39c12', 'congested': '#e74c3c', 'unknown': '#7f8c8d'}
DEFAULT_RUN = 'R4'
HORIZON = 12


@st.cache_data(show_spinner=False)
def load_meta() -> pd.DataFrame:
    """Sensor metadata: ID, Lat, Lng, Fwy, Direction, Lanes, plus a human-readable Name from PeMS D11 where it matches."""
    meta = pd.read_csv(DATA_SD / 'sd_meta.csv')
    pems_meta = pd.read_csv(PEMS / 'd11_station_meta_sd.csv', usecols=['ID', 'Name', 'City'])
    pems_meta = pems_meta.drop_duplicates('ID')
    meta = meta.merge(pems_meta, on='ID', how='left')
    meta['Name'] = meta['Name'].fillna(meta['Fwy'] + ' ' + meta['Direction'])
    meta['label'] = meta['Fwy'] + ' ' + meta['Direction'] + ' — ' + meta['Name'].astype(str) + ' (' + meta['ID'].astype(str) + ')'
    return meta.set_index('ID', drop=False)


@st.cache_resource(show_spinner=False)
def load_predictions(run: str = DEFAULT_RUN) -> dict:
    """pred_flow/true_flow/pred_class/true_class: (S, HORIZON, N) arrays, plus test_starts and target_times_h1."""
    p = PRED / f'{run}_test.npz'
    if not p.exists():
        raise FileNotFoundError(
            f'{p} not found — run `python scripts/export_predictions.py {run}` first.')
    z = np.load(p, allow_pickle=False)
    times_h1 = pd.to_datetime(z['target_times_h1'])
    return {
        'pred_flow': z['pred_flow'], 'true_flow': z['true_flow'],
        'pred_class': z['pred_class'], 'true_class': z['true_class'],
        'test_starts': z['test_starts'], 'times_h1': times_h1,
        'sensor_ids': z['sensor_ids'],
    }


@st.cache_data(show_spinner=False)
def load_incidents() -> dict:
    z = np.load(CONG / 'factors' / 'incidents.npz')
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def load_incident_details() -> pd.DataFrame:
    df = pd.read_csv(PEMS / 'chp_incident_details_d11_2019.csv', low_memory=False)
    return df


@st.cache_data(show_spinner=False)
def load_result(name: str) -> dict:
    p = RESULTS / f'{name}.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@st.cache_data(show_spinner=False)
def load_stratified_metrics() -> dict:
    """{stratum_name: {run_name: MAE}} — data/traffic/congestion/results/eval_stratified.json.
    Confirmed shape: d['rain']['R4'] == 26.1531. Strata: overall, rain, dry, incident,
    non-incident, peak, off-peak, holiday-period, non-holiday, true=free, true=heavy, true=congested."""
    p = RESULTS / 'eval_stratified.json'
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def ablation_table() -> pd.DataFrame:
    """avg test MAE/RMSE/MAPE + macro-F1 for every finished run, for the chatbot + a summary panel."""
    rows = []
    for run in ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'A1', 'A2', 'A3']:
        d = load_result(f'{run}_final')
        if not d:
            continue
        test = d['final'].get('test', d['final'].get('val'))
        row = {'run': run, 'MAE': float(np.mean(test['MAE'])), 'RMSE': float(np.mean(test['RMSE'])),
               'MAPE': float(np.mean(test['MAPE'])) * 100}
        if 'cls' in test:
            row['macro_F1'] = test['cls']['macro_f1']
        rows.append(row)
    return pd.DataFrame(rows).set_index('run')


def window_index(pred: dict, target_time, horizon: int):
    """Map a target time + horizon (1-12) to the window row s = t - h in pred_flow/true_flow/*_class.
    Returns None if that (time, horizon) combination falls outside the test window range."""
    h1_times = pred['times_h1']
    # target time for a window at horizon h is times_h1[s] + (h-1)*15min
    origin_time = pd.Timestamp(target_time) - pd.Timedelta(minutes=15 * (horizon - 1))
    matches = np.nonzero(h1_times == origin_time)[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def slice_at(pred: dict, s: int, horizon: int):
    """(pred_flow, true_flow, pred_class, true_class) at window s, horizon h (1-12), each shape (N,)."""
    h = horizon - 1
    return (pred['pred_flow'][s, h].astype(np.float32), pred['true_flow'][s, h].astype(np.float32),
            pred['pred_class'][s, h], pred['true_class'][s, h])


def network_kpis(pred_flow: np.ndarray, true_flow: np.ndarray, pred_class: np.ndarray, active: np.ndarray) -> dict:
    """Network-wide summary stats for the currently selected time/horizon slice, for the KPI header row."""
    labelled = pred_class >= 0
    valid_actual = true_flow > 0
    err = np.abs(pred_flow - true_flow)
    return {
        'avg_flow': float(pred_flow.mean()),
        'congested_share': float((pred_class == 2).sum() / max(labelled.sum(), 1)),
        'heavy_share': float((pred_class == 1).sum() / max(labelled.sum(), 1)),
        'active_incidents': int(active.sum()),
        'window_mae': float(err[valid_actual].mean()) if valid_actual.any() else float('nan'),
    }


def _sensor_frame(meta: pd.DataFrame, sensor_ids, pred_flow: np.ndarray, true_flow: np.ndarray,
                   pred_class: np.ndarray) -> pd.DataFrame:
    """Unsorted, untruncated per-sensor rows — shared by top_bottlenecks and rank_sensors."""
    m = meta.loc[sensor_ids]
    has_actual = true_flow > 0  # LargeST convention: 0 = missing-data sentinel, not real zero flow
    return pd.DataFrame({
        'Sensor': m['label'].to_numpy(), 'Freeway': m['Fwy'].to_numpy(), 'Direction': m['Direction'].to_numpy(),
        'Predicted class': [CLASSES[c] if c >= 0 else 'unknown' for c in pred_class],
        'Predicted flow': pred_flow.round(1),
        'Actual flow': np.where(has_actual, true_flow.round(1), np.nan),
        'Error': np.where(has_actual, (pred_flow - true_flow).round(1), np.nan),
    })


def top_bottlenecks(meta: pd.DataFrame, sensor_ids, pred_flow: np.ndarray, true_flow: np.ndarray,
                     pred_class: np.ndarray, top_n: int = 12) -> pd.DataFrame:
    """Ranked watchlist: congested sensors first, then heavy, sorted by predicted flow within each class."""
    df = _sensor_frame(meta, sensor_ids, pred_flow, true_flow, pred_class)
    class_rank = {'congested': 0, 'heavy': 1, 'free': 2, 'unknown': 3}
    df['_rank'] = df['Predicted class'].map(class_rank)
    return df.sort_values(['_rank', 'Predicted flow'], ascending=[True, False]).drop(columns='_rank').head(top_n).reset_index(drop=True)


def rank_sensors(meta: pd.DataFrame, sensor_ids, pred_flow: np.ndarray, true_flow: np.ndarray,
                  pred_class: np.ndarray, criterion: str = 'congestion', top_n: int = 10) -> pd.DataFrame:
    """criterion: 'congestion' (top_bottlenecks' own ordering), 'error' (descending abs error,
    only cells with a real actual reading), or 'flow' (descending predicted flow)."""
    if criterion == 'congestion':
        return top_bottlenecks(meta, sensor_ids, pred_flow, true_flow, pred_class, top_n=top_n)
    df = _sensor_frame(meta, sensor_ids, pred_flow, true_flow, pred_class)
    if criterion == 'error':
        df = df[df['Actual flow'].notna()].copy()
        df['_abserr'] = df['Error'].abs()
        df = df.sort_values('_abserr', ascending=False).drop(columns='_abserr')
    elif criterion == 'flow':
        df = df.sort_values('Predicted flow', ascending=False)
    else:
        raise ValueError(f"unknown criterion {criterion!r}; use 'congestion', 'error', or 'flow'")
    return df.head(top_n).reset_index(drop=True)
