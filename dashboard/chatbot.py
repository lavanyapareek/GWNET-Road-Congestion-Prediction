"""Gemini-powered chatbot over the congestion predictions, run locally (free tier).

Manual tool-use loop (mirrors the Gemini SDK's own automatic-function-calling loop,
see google.genai.models._generate_content) so Streamlit's per-turn rerun model stays
in full control of what's shown and when the loop stops. Needs a Gemini API key
(free, from https://aistudio.google.com/apikey) in .streamlit/secrets.toml as
GEMINI_API_KEY, or the env var of the same name.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import data as D

MODEL = 'gemini-3.1-flash-lite'  # free tier: the newest generation (gemini-3.8-flash, and the
# 'gemini-flash-latest' alias which resolves to it) has only a 20-requests/day free quota — far
# too tight for iterative testing or a live demo. Older/lite tiers carry a much higher free quota.
MAX_TOOL_TURNS = 6
MAX_API_RETRIES = 5  # Gemini free tier returns occasional transient 503s under load; observed recovery in 1-2 retries

SYSTEM_PROMPT = """You are the analyst assistant embedded in a San Diego freeway congestion \
forecasting dashboard (LargeST-SD 2019 dataset, 716 sensors, 15-minute steps, GWNet model \
with a congestion classification head). You answer questions about specific predictions, \
model performance, and traffic incidents using the tools provided — always call a tool \
rather than guessing a number. Congestion classes are free / heavy / congested, derived from \
PeMS occupancy thresholds. The best model is R4 (adds utilization/time/static features plus \
holiday & school-calendar corrections on top of the base GWNet); weather (R5) and incidents \
(R6) features roughly wash out on top of R4 in aggregate MAE. Test period is 2019-10-19 22:15 \
through 2019-12-31 23:45. Keep answers concise and quantitative; cite the sensor ID/name and \
timestamp you looked up."""

TOOLS = [
    {
        'name': 'get_prediction',
        'description': ('Get the predicted vs. actual flow and congestion class for one sensor '
                         'at one real-world (target) time and forecast horizon.'),
        'parameters': {
            'type': 'object',
            'properties': {
                'sensor_id': {'type': 'integer', 'description': 'LargeST sensor ID, e.g. 1108680'},
                'target_time': {'type': 'string', 'description': 'ISO datetime, e.g. 2019-12-03T17:00:00'},
                'horizon': {'type': 'integer', 'description': 'Forecast horizon in 15-min steps, 1-12 (default 4 = 1h ahead)'},
            },
            'required': ['sensor_id', 'target_time'],
        },
    },
    {
        'name': 'find_sensor',
        'description': 'Search sensors by freeway, direction, or name text (e.g. "I-5 N", "Nordahl").',
        'parameters': {
            'type': 'object',
            'properties': {'query': {'type': 'string'}},
            'required': ['query'],
        },
    },
    {
        'name': 'get_model_metrics',
        'description': 'Get test-set MAE/RMSE/MAPE by horizon and classification macro-F1 for one run (R0-R6, A1-A3).',
        'parameters': {
            'type': 'object',
            'properties': {'run': {'type': 'string', 'description': 'Run name, e.g. R4'}},
            'required': ['run'],
        },
    },
    {
        'name': 'compare_runs',
        'description': ('Get the full feature-ablation comparison table (avg test MAE/RMSE/MAPE, '
                         'macro-F1) across every finished run R0-R6, A1-A3.'),
        'parameters': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'get_incidents_near_sensor',
        'description': 'List CHP incidents recorded near one sensor within a date range (checks the matched incident factor and detail table).',
        'parameters': {
            'type': 'object',
            'properties': {
                'sensor_id': {'type': 'integer'},
                'start_date': {'type': 'string', 'description': 'YYYY-MM-DD'},
                'end_date': {'type': 'string', 'description': 'YYYY-MM-DD'},
            },
            'required': ['sensor_id', 'start_date', 'end_date'],
        },
    },
]

GENAI_TOOLS = [types.Tool(function_declarations=[
    {'name': t['name'], 'description': t['description'], 'parameters': t['parameters']} for t in TOOLS
])]


def _client() -> genai.Client:
    try:
        key = st.secrets.get('GEMINI_API_KEY', None)
    except Exception:  # noqa: BLE001 — no secrets.toml on disk at all
        key = None
    key = key or os.environ.get('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('No GEMINI_API_KEY — add it to .streamlit/secrets.toml or export it in the shell. '
                            'Free key from https://aistudio.google.com/apikey')
    return genai.Client(api_key=key)


def _tool_get_prediction(sensor_id, target_time, horizon=4):
    pred = D.load_predictions()
    meta = D.load_meta()
    s = D.window_index(pred, pd.Timestamp(target_time), int(horizon))
    if s is None:
        return {'error': 'target_time/horizon combination falls outside the test period (2019-10-19 22:15 to 2019-12-31 23:45), or is not on a 15-min boundary.'}
    sid_pos = np.nonzero(pred['sensor_ids'] == sensor_id)[0]
    if len(sid_pos) == 0:
        return {'error': f'sensor_id {sensor_id} not found'}
    n = int(sid_pos[0])
    pf, tf, pc, tc = D.slice_at(pred, s, int(horizon))
    label = meta.loc[sensor_id, 'label'] if sensor_id in meta.index else str(sensor_id)
    return {
        'sensor': label, 'target_time': str(target_time), 'horizon_steps': int(horizon),
        'predicted_flow_veh_5min': round(float(pf[n]), 1), 'actual_flow_veh_5min': round(float(tf[n]), 1),
        'predicted_class': D.CLASSES[int(pc[n])] if pc[n] >= 0 else 'unknown',
        'actual_class': D.CLASSES[int(tc[n])] if tc[n] >= 0 else 'unknown',
    }


def _tool_find_sensor(query):
    meta = D.load_meta()
    q = query.lower()
    hit = meta[meta['label'].str.lower().str.contains(q) | meta['Fwy'].str.lower().str.contains(q)]
    return hit[['ID', 'Fwy', 'Direction', 'Name', 'Lat', 'Lng']].head(10).to_dict('records')


def _tool_get_model_metrics(run):
    d = D.load_result(f'{run}_final')
    if not d:
        return {'error': f'no results for run {run}'}
    test = d['final'].get('test', d['final'].get('val'))
    out = {'run': run, 'avg_MAE': round(float(np.mean(test['MAE'])), 2),
           'avg_RMSE': round(float(np.mean(test['RMSE'])), 2),
           'avg_MAPE_pct': round(float(np.mean(test['MAPE'])) * 100, 2),
           'MAE_by_horizon': [round(v, 2) for v in test['MAE']]}
    if 'cls' in test:
        out['macro_F1'] = round(test['cls']['macro_f1'], 3)
        out['per_class_f1'] = dict(zip(D.CLASSES, [round(v, 3) for v in test['cls']['f1']]))
    return out


def _tool_compare_runs():
    return D.ablation_table().round(3).reset_index().to_dict('records')


def _tool_get_incidents_near_sensor(sensor_id, start_date, end_date):
    inc = D.load_incidents()
    pred = D.load_predictions()
    sid_pos = np.nonzero(pred['sensor_ids'] == sensor_id)[0]
    if len(sid_pos) == 0:
        return {'error': f'sensor_id {sensor_id} not found'}
    n = int(sid_pos[0])
    times = pred['times_h1']
    mask = (times >= pd.Timestamp(start_date)) & (times <= pd.Timestamp(end_date))
    starts = pred['test_starts'][mask]
    active_flags = inc['active'][starts, n]
    active_times = times[mask][active_flags > 0]
    return {'sensor_id': sensor_id, 'active_incident_steps': len(active_times),
            'first_active': str(active_times.min()) if len(active_times) else None,
            'last_active': str(active_times.max()) if len(active_times) else None}


DISPATCH = {
    'get_prediction': _tool_get_prediction,
    'find_sensor': _tool_find_sensor,
    'get_model_metrics': _tool_get_model_metrics,
    'compare_runs': _tool_compare_runs,
    'get_incidents_near_sensor': _tool_get_incidents_near_sensor,
}


def new_user_content(text: str) -> types.Content:
    return types.Content(role='user', parts=[types.Part(text=text)])


def _generate_with_retry(client, contents, config):
    """Gemini's free tier occasionally returns a transient 503 (model overloaded) or 429
    (rate limited) — retry those with backoff; anything else (bad request, auth) raises immediately."""
    for attempt in range(MAX_API_RETRIES):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except genai_errors.ServerError:
            if attempt == MAX_API_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
        except genai_errors.ClientError as e:
            if getattr(e, 'code', None) == 429 and attempt < MAX_API_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def run_turn(contents: list) -> list:
    """Runs the manual tool-use loop for one user turn; returns the updated Content list."""
    client = _client()
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=GENAI_TOOLS)
    for _ in range(MAX_TOOL_TURNS):
        response = _generate_with_retry(client, contents, config)
        model_content = response.candidates[0].content
        contents.append(model_content)
        calls = response.function_calls
        if not calls:
            break
        response_parts = []
        for call in calls:
            fn = DISPATCH.get(call.name)
            try:
                result = fn(**call.args) if fn else {'error': f'unknown tool {call.name}'}
            except Exception as e:  # noqa: BLE001 — surfaced to the model as a tool error, not a crash
                result = {'error': str(e)}
            response_parts.append(types.Part.from_function_response(
                name=call.name, response={'result': json.loads(json.dumps(result, default=str))}))
        contents.append(types.Content(role='user', parts=response_parts))
    return contents


def content_text(content: types.Content) -> str:
    """Plain text of a Content's parts, skipping function-call/response parts."""
    if content is None or content.parts is None:
        return ''
    return ''.join(p.text for p in content.parts if p.text)
