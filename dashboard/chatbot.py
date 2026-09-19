"""Gemini-powered chatbot over the congestion predictions, run locally (free tier).

Manual tool-use loop (mirrors the Gemini SDK's own automatic-function-calling loop,
see google.genai.models._generate_content) so Streamlit's per-turn rerun model stays
in full control of what's shown and when the loop stops. Needs a Gemini API key
(free, from https://aistudio.google.com/apikey) in .streamlit/secrets.toml as
GEMINI_API_KEY, or the env var of the same name.
"""
import json
import os
import re
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
MAX_TOOL_TURNS = 8
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
timestamp you looked up. For comparative or investigative questions ("does X help when Y", \
"which sensors are worst right now", "why is Z happening"), chain multiple tool calls — e.g. \
get_stratified_metrics for a specific condition, rank_sensors to find the worst offenders, then \
get_incidents_near_sensor to check a cause — rather than answering from one lookup. Call \
navigate_dashboard whenever the user asks to see/show/jump to/go to a specific time, so the \
dashboard itself moves there; it does not return numbers, so call get_prediction too if you \
need to discuss values."""

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
    {
        'name': 'get_stratified_metrics',
        'description': 'Get test-set MAE for one run, restricted to a stratum (rain/dry/incident/peak/holiday/true-class buckets) — use this to answer "does X help in condition Y" questions instead of aggregate compare_runs.',
        'parameters': {
            'type': 'object',
            'properties': {
                'stratum': {'type': 'string', 'enum': ['overall', 'rain', 'dry', 'incident', 'non-incident',
                                                        'peak', 'off-peak', 'holiday-period', 'non-holiday',
                                                        'true=free', 'true=heavy', 'true=congested']},
                'run': {'type': 'string', 'description': 'Run name, e.g. R4 (default R4)'},
            },
            'required': ['stratum'],
        },
    },
    {
        'name': 'rank_sensors',
        'description': 'Rank all 716 sensors at one target time/horizon by congestion, prediction error, or flow — for investigative/comparative questions like "which sensors were most congested" or "where was the model most wrong".',
        'parameters': {
            'type': 'object',
            'properties': {
                'target_time': {'type': 'string', 'description': 'ISO datetime, e.g. 2019-12-03T17:00:00'},
                'horizon': {'type': 'integer', 'description': 'Forecast horizon in 15-min steps, 1-12 (default 4)'},
                'criterion': {'type': 'string', 'enum': ['congestion', 'error', 'flow']},
                'top_n': {'type': 'integer', 'description': 'How many sensors to return (default 10)'},
            },
            'required': ['target_time'],
        },
    },
    {
        'name': 'navigate_dashboard',
        'description': 'Move the dashboard\'s own date/time/horizon controls to a specific window so the user can see it — use when asked to "show", "jump to", or "go to" a time. Does not itself return prediction numbers; call get_prediction if you need to discuss values.',
        'parameters': {
            'type': 'object',
            'properties': {
                'target_time': {'type': 'string', 'description': 'ISO datetime, e.g. 2019-12-03T17:00:00'},
                'horizon': {'type': 'integer', 'description': 'Forecast horizon in 15-min steps, 1-12 (default 4)'},
            },
            'required': ['target_time'],
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


def _tool_get_stratified_metrics(stratum, run='R4'):
    d = D.load_stratified_metrics()
    if stratum not in d:
        return {'error': f'unknown stratum {stratum!r}; valid: {sorted(d.keys())}'}
    if run not in d[stratum]:
        return {'error': f'no data for run {run!r} in stratum {stratum!r}'}
    return {'stratum': stratum, 'run': run, 'MAE': round(float(d[stratum][run]), 2)}


def _tool_rank_sensors(target_time, horizon=4, criterion='congestion', top_n=10):
    pred = D.load_predictions()
    meta = D.load_meta()
    s = D.window_index(pred, pd.Timestamp(target_time), int(horizon))
    if s is None:
        return {'error': 'target_time/horizon combination falls outside the test period (2019-10-19 22:15 to 2019-12-31 23:45), or is not on a 15-min boundary.'}
    pred_flow, true_flow, pred_class, true_class = D.slice_at(pred, s, int(horizon))
    try:
        df = D.rank_sensors(meta, pred['sensor_ids'], pred_flow, true_flow, pred_class,
                             criterion=criterion, top_n=int(top_n))
    except ValueError as e:
        return {'error': str(e)}
    return df.to_dict('records')  # any NaN (masked missing-actual) is sanitized centrally in run_turn


def _tool_navigate_dashboard(target_time, horizon=4):
    pred = D.load_predictions()
    try:
        tt = pd.Timestamp(target_time)
    except Exception as e:  # noqa: BLE001
        return {'error': f'could not parse target_time: {e}'}
    if not (1 <= int(horizon) <= D.HORIZON):
        return {'error': f'horizon must be between 1 and {D.HORIZON} (15-min steps)'}
    if D.window_index(pred, tt, int(horizon)) is None:
        return {'error': 'target_time/horizon combination falls outside the test period (2019-10-19 22:15 to 2019-12-31 23:45), or is not on a 15-min boundary.'}
    st.session_state['pending_nav'] = {'date': tt.strftime('%Y-%m-%d'), 'time': tt.strftime('%H:%M'), 'horizon': int(horizon)}
    return {'navigated_to': str(tt), 'horizon_steps': int(horizon),
            'note': 'the dashboard sidebar will jump to this date/time/horizon'}


DISPATCH = {
    'get_prediction': _tool_get_prediction,
    'find_sensor': _tool_find_sensor,
    'get_model_metrics': _tool_get_model_metrics,
    'compare_runs': _tool_compare_runs,
    'get_incidents_near_sensor': _tool_get_incidents_near_sensor,
    'get_stratified_metrics': _tool_get_stratified_metrics,
    'rank_sensors': _tool_rank_sensors,
    'navigate_dashboard': _tool_navigate_dashboard,
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


def _json_safe(obj):
    """Recursively replace NaN/Infinity floats with None. Python's own json.dumps silently allows
    the non-standard NaN/Infinity tokens (allow_nan=True by default) and numpy.float64 is a float
    subclass so it hits this path too — but the Gemini API's own JSON parser is strict (RFC 8259)
    and rejects a payload containing a literal NaN outright ('Invalid JSON payload ... Unexpected
    token'). rank_sensors is the one tool that can return NaN today (LargeST's missing-actual-
    reading sentinel), but this is applied to every tool's result so a future one can't reintroduce
    the same crash — a pandas .where(df.notna(), None) does NOT reliably work for this: assigning
    None into a float64 column silently reverts to NaN, since numpy floats can't hold Python None."""
    if isinstance(obj, float):
        return obj if obj == obj and abs(obj) != float('inf') else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


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
            safe_result = _json_safe(json.loads(json.dumps(result, default=str)))
            response_parts.append(types.Part.from_function_response(
                name=call.name, response={'result': safe_result}))
        contents.append(types.Content(role='user', parts=response_parts))
    if contents[-1].role != 'model' or not content_text(contents[-1]):
        # Hit MAX_TOOL_TURNS with tool calls still pending — synthesize a closing message so the
        # UI (and iter_turns below) always has real text to show for a turn, never a silent blank.
        contents.append(types.Content(role='model', parts=[types.Part(
            text="I gathered some data but ran out of investigation steps before finishing — "
                 "here's what I found so far; try a more specific follow-up question.")]))
    return contents


def content_text(content: types.Content) -> str:
    """Plain text of a Content's parts, skipping function-call/response parts."""
    if content is None or content.parts is None:
        return ''
    return ''.join(p.text for p in content.parts if p.text)


def iter_turns(contents: list):
    """Groups the flat Content list into logical turns for rendering. Matches tool-calls to
    tool-responses positionally (a model call-batch is always followed by a same-order user
    response-batch, per run_turn's own construction)."""
    steps, pending_calls = [], []
    for content in contents:
        if content is None or not content.parts:
            continue
        if content.role == 'model':
            calls = [p.function_call for p in content.parts if p.function_call is not None]
            if calls:
                pending_calls = [{'tool': c.name, 'args': dict(c.args or {})} for c in calls]
                steps.extend(pending_calls)
                continue
            yield {'role': 'model', 'steps': steps, 'text': content_text(content)}
            steps = []
        elif content.role == 'user':
            responses = [p.function_response for p in content.parts if p.function_response is not None]
            if responses:
                for step, resp in zip(pending_calls, responses):
                    r = resp.response or {}
                    step['result'] = r.get('result', r)
                pending_calls = []
                continue
            text = content_text(content)
            if text:
                yield {'role': 'user', 'text': text}
    if steps:
        yield {'role': 'model', 'steps': steps, 'text': ''}


_ISO_DATETIME_RE = re.compile(r'\b\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?\b')
_MONTH_NAMES = ('January|February|March|April|May|June|July|August|September|October|November|December')
_NATURAL_DATE_RE = re.compile(
    rf'\b(?:{_MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?\b', re.IGNORECASE)
_CLOCK_TIME_RE = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm])?\b')
_SENSOR_ID_RE = re.compile(r'\b\d{7}\b')  # every LargeST-SD sensor ID is exactly 7 digits (1108285-1127052)
_YEAR_RE = re.compile(r'\b(?:19|20)\d{2}\b')  # bare years (e.g. a lingering "2019" from a natural-
# language date the other date regexes didn't fully consume) — safe to always strip: no legitimate
# metric in this project is a bare 4-digit integer in the 1900-2099 range.
_ROUTE_RE = re.compile(r'\b(?:I|SR|US)-?\d{1,3}\b', re.IGNORECASE)  # I-15, I15, SR163, US101
_ROUTE_DIR_RE = re.compile(  # "15 SB", "163 N", "NB15", "SB 805" — PeMS station names spell route
    r'\b(?:\d{1,3}\s?(?:NB|SB|EB|WB|N|S|E|W)|(?:NB|SB|EB|WB)\s?\d{1,3})\b', re.IGNORECASE)
_NUMBER_RE = re.compile(r'-?\d+\.\d+|-?\d+')


def _flatten_numbers(obj) -> set:
    out = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if obj == obj and abs(obj) != float('inf'):  # skip NaN/inf
            out.add(round(float(obj), 4))
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= _flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _flatten_numbers(v)
    return out


def _flatten_strings(obj) -> list:
    """Recursively collect string leaf values (e.g. sensor/freeway labels) from a tool result, so
    their embedded route numbers ("15 SB N/O Carrol Cyn", "NB15 @ 163 Merge" — real PeMS station
    names, not metrics) can be stripped verbatim from the answer text before number-checking,
    instead of trying to regex-guess every route-naming convention PeMS data uses."""
    out = []
    if isinstance(obj, str) and len(obj) > 3:
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_flatten_strings(v))
    return out


def _extract_checkable_numbers(text: str) -> list:
    """Numbers worth fact-checking: strip ISO dates, natural-language dates ("December 3, 2019"),
    clock times, 7-digit sensor IDs, bare years, and freeway/route mentions ("I-15", "163 Merge")
    first (so their digits are never mistaken for metrics), then drop bare integers <= HORIZON (12)
    as noise — horizons/step-counts/top_n/run-name digits (R4, A2, ...) are near-universally <= 12,
    while every real metric this project reports is either a decimal or > 12."""
    stripped = text
    for pattern in (_ISO_DATETIME_RE, _NATURAL_DATE_RE, _CLOCK_TIME_RE, _SENSOR_ID_RE, _YEAR_RE,
                    _ROUTE_RE, _ROUTE_DIR_RE):
        stripped = pattern.sub(' ', stripped)
    out = []
    for m in _NUMBER_RE.finditer(stripped):
        tok = m.group()
        val = float(tok)
        if '.' not in tok and abs(val) <= D.HORIZON:
            continue
        out.append(val)
    return out


def verify_answer(text: str, steps: list) -> dict:
    """Zero-extra-API-call self-check: flags numbers in `text` that don't approximately match any
    value returned by this turn's tool calls (0.5 absolute or 1% relative tolerance). Known
    limitation, accepted given the no-extra-API-call constraint: correct arithmetic the model
    derives from two tool values (a stated delta, say) will still be flagged, since only literal
    tool-returned values count as ground truth."""
    if not steps:
        return {'flagged': [], 'verified': True}
    known = set()
    known_strings = []
    for step in steps:
        result = step.get('result', {})
        known |= _flatten_numbers(result)
        known_strings.extend(_flatten_strings(result))
    stripped_text = text
    for s in sorted(set(known_strings), key=len, reverse=True):  # longest first: avoid a short
        stripped_text = stripped_text.replace(s, ' ')            # string shadowing a longer one
    flagged = [v for v in _extract_checkable_numbers(stripped_text)
               if not any(abs(v - k) <= max(0.5, 0.01 * abs(k)) for k in known)]
    return {'flagged': flagged, 'verified': len(flagged) == 0}
