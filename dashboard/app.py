"""San Diego congestion forecasting — local demo dashboard.

Run with:  streamlit run dashboard/app.py
Needs data/traffic/congestion/predictions/R4_test.npz — build it first with:
    python scripts/export_predictions.py R4
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chatbot as C
import data as D
import map_view as MV
import theme as T

st.set_page_config(page_title='SD Congestion Forecast', layout='wide', page_icon='🚦')
T.inject_css()

RUN = D.DEFAULT_RUN

try:
    pred = D.load_predictions(RUN)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

meta = D.load_meta()
incidents = D.load_incidents()
sensor_ids = pred['sensor_ids']
times_h1 = pred['times_h1']
test_starts = pred['test_starts']

def _day_times_for(date_val):
    """(day_times, time_labels) for one calendar date — defined once, used by both the sidebar
    controls and pending_nav resolution below."""
    mask = times_h1.normalize() == pd.Timestamp(date_val)
    dt = times_h1[mask]
    return dt, [t.strftime('%H:%M') for t in dt]


if 'pending_nav' in st.session_state:
    nav = st.session_state['pending_nav']
    del st.session_state['pending_nav']
    nav_date = pd.Timestamp(nav['date']).date()
    _, nav_labels = _day_times_for(nav_date)
    st.session_state.sel_date = nav_date
    st.session_state.horizon = nav['horizon']
    st.session_state.time_idx = nav_labels.index(nav['time']) if nav['time'] in nav_labels else len(nav_labels) // 2

T.hero(
    'San Diego Freeway Congestion Forecast',
    f'Model {RUN} (GWNet + congestion head) &nbsp;·&nbsp; 716 sensors &nbsp;·&nbsp; test replay '
    f'{times_h1.min():%b %d} – {(times_h1.max() + pd.Timedelta(minutes=11 * 15)):%b %d, %Y}',
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
# None of these widgets are given a `key=` — each is driven by a plain session_state variable
# (value= computed fresh each run, return value written back right after), which is what lets
# the chatbot's navigate_dashboard tool (via pending_nav above) safely drive them. A widget
# cannot have its own session_state[key] written to once it's been instantiated in the same run
# (StreamlitWidgetAlreadyInstantiatedError) — this pattern sidesteps that entirely.
st.sidebar.header('Replay controls')
min_day, max_day = times_h1.min().normalize(), times_h1.max().normalize()
sel_date = st.sidebar.date_input('Date', value=st.session_state.get('sel_date', min_day.date()),
                                  min_value=min_day.date(), max_value=max_day.date())
st.session_state.sel_date = sel_date
day_times, time_labels = _day_times_for(sel_date)
day_mask = times_h1.normalize() == pd.Timestamp(sel_date)
if len(day_times) == 0:
    st.sidebar.warning('No test data on this day.')
    st.stop()

if 'time_idx' not in st.session_state or st.session_state.time_idx >= len(time_labels):
    st.session_state.time_idx = len(time_labels) // 2
if 'playing' not in st.session_state:
    st.session_state.playing = False

pcol1, pcol2 = st.sidebar.columns([1, 1.4])
if st.session_state.playing:
    if pcol1.button('⏸ Pause', width='stretch'):
        st.session_state.playing = False
else:
    if pcol1.button('▶ Play', width='stretch'):
        st.session_state.playing = True
play_speed = pcol2.select_slider('Speed', options=['0.5x', '1x', '2x', '4x'], value='1x', label_visibility='collapsed')

sel_label = st.sidebar.select_slider('Time of day (target)', options=time_labels,
                                     value=time_labels[st.session_state.time_idx])
st.session_state.time_idx = time_labels.index(sel_label)
target_time = day_times[st.session_state.time_idx]

horizon = st.sidebar.slider('Forecast horizon (15-min steps ahead)', 1, D.HORIZON,
                            value=st.session_state.get('horizon', 4),
                            help='How far ahead this prediction was made — 4 = 1 hour ahead')
st.session_state.horizon = horizon
st.sidebar.caption(f'Prediction made at {target_time - pd.Timedelta(minutes=15 * horizon):%Y-%m-%d %H:%M} '
                   f'for {target_time:%Y-%m-%d %H:%M}')

s = D.window_index(pred, target_time, horizon)
if s is None:
    st.warning('This time/horizon combination falls outside the test window range — pick an earlier time or smaller horizon.')
    st.stop()

pred_flow, true_flow, pred_class, true_class = D.slice_at(pred, s, horizon)
abs_t = int(test_starts[s] + horizon)
active = incidents['active'][abs_t]

# ── KPI header row ─────────────────────────────────────────────────────────────
kpis = D.network_kpis(pred_flow, true_flow, pred_class, active)
model_avg_mae = D.ablation_table().loc[RUN, 'MAE'] if RUN in D.ablation_table().index else float('nan')
T.kpi_row([
    {'label': 'Network avg. predicted flow', 'value': f"{kpis['avg_flow']:.0f} veh/5min"},
    {'label': 'Share congested / heavy', 'value': f"{kpis['congested_share']*100:.1f}%",
     'delta': f"+{kpis['heavy_share']*100:.1f}% heavy", 'delta_kind': 'neutral'},
    {'label': 'Active CHP incidents', 'value': f"{kpis['active_incidents']}",
     'delta': 'this time slot' , 'delta_kind': 'neutral'},
    {'label': 'This window’s MAE', 'value': f"{kpis['window_mae']:.1f}" if kpis['window_mae'] == kpis['window_mae'] else '—',
     'delta': f"model avg {model_avg_mae:.1f}" if model_avg_mae == model_avg_mae else None, 'delta_kind': 'neutral'},
])

# A manual segmented-control view switch, not st.tabs() — deliberately. st.tabs() runs every
# tab's Python code (and mounts every tab's widgets in the DOM, just CSS-hidden when inactive) on
# every rerun, regardless of which tab is visible. That's what breaks st.chat_input two different
# ways: (1) nested inside a tab, it has documented cross-browser click/focus bugs (confirmed
# Safari-specific in this app — streamlit/streamlit issues #7814, #8564); (2) even hoisted to the
# top level, this Streamlit version (1.64) auto-scrolls the *entire app* to the bottom on every
# load whenever a top-level chat_input exists anywhere in the still-mounted tab tree — turning a
# dashboard that should open on the hero/KPI row into one that opens scrolled down to "Ops
# Watchlist" or worse. A plain `if/elif` is not a Streamlit container at all: only the selected
# view's code — and therefore its widgets — actually run each rerun, so chat_input (and the chat
# history that goes with it) are simply absent from the DOM until the user picks that view.
if 'view' not in st.session_state:
    st.session_state.view = 'Map views'
view = st.segmented_control('View', ['Map views', 'Sensor drill-down', 'Model performance', 'Ask the model (chatbot)'],
                            default=st.session_state.view, label_visibility='collapsed')
st.session_state.view = view or st.session_state.view
view = st.session_state.view

if view == 'Map views':
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(MV.congestion_state_map(meta, sensor_ids, pred_class), width='stretch')
    with c2:
        st.plotly_chart(MV.flow_compare_map(meta, sensor_ids, pred_flow, true_flow), width='stretch')
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(MV.incident_map(meta, sensor_ids, active, pred_class), width='stretch')
    with c4:
        st.plotly_chart(MV.error_map(meta, sensor_ids, pred_flow, true_flow), width='stretch')

    st.divider()
    st.markdown('<div class="section-eyebrow">OPS WATCHLIST</div>', unsafe_allow_html=True)
    st.subheader('Top bottlenecks, this time slot')
    watchlist = D.top_bottlenecks(meta, sensor_ids, pred_flow, true_flow, pred_class, top_n=12)
    st.dataframe(
        watchlist, width='stretch', hide_index=True,
        column_config={
            'Predicted class': st.column_config.TextColumn(),
            'Predicted flow': st.column_config.ProgressColumn(min_value=0, max_value=float(max(watchlist['Predicted flow'].max(), 1)), format='%.0f'),
        },
    )

elif view == 'Sensor drill-down':
    labels = meta.loc[sensor_ids, 'label'].to_numpy()
    pick = st.selectbox('Sensor', options=range(len(sensor_ids)), format_func=lambda i: labels[i])
    sid = int(sensor_ids[pick])
    actual_series = pred['true_flow'][day_mask, 0, pick].astype(np.float32)
    pred_by_h = {h: pred['pred_flow'][day_mask, h - 1, pick].astype(np.float32) for h in (1, 4, 12)}
    st.plotly_chart(MV.sensor_history_chart(day_times, actual_series, pred_by_h), width='stretch')
    m1, m2, m3 = st.columns(3)
    m1.metric('Predicted flow (this slot)', f'{pred_flow[pick]:.0f} veh/5min',
              delta=f'{pred_flow[pick] - true_flow[pick]:+.0f} vs. actual')
    m2.metric('Predicted class', D.CLASSES[pred_class[pick]] if pred_class[pick] >= 0 else 'unknown')
    m3.metric('Actual class', D.CLASSES[true_class[pick]] if true_class[pick] >= 0 else 'unknown')

elif view == 'Model performance':
    st.markdown('<div class="section-eyebrow">GATE E — STATISTICAL VERIFICATION</div>', unsafe_allow_html=True)
    st.subheader('Which engineering choices actually moved the number')
    T.callout('finding', 'FINDING',
              'R4 is the statistically best model, not merely the lowest-MAE one. A paired day-block '
              'bootstrap (1,000 resamples, 74 test days) confirms the R2→R3 and R3→R4 feature '
              'additions are real (95% CI excludes zero); R4→R5 (weather) and R4→R6 (incidents) '
              'are not distinguishable from noise in aggregate.')
    ic1, ic2 = st.columns(2)
    with ic1:
        st.image(str(D.CHARTS / 'ablation_all.png'), width='stretch',
                  caption='Average test MAE, every run and baseline (R4 highlighted).')
    with ic2:
        st.image(str(D.CHARTS / 'bootstrap_forest.png'), width='stretch',
                  caption="95% bootstrap CIs on every row-to-row delta.")
    T.callout('limitation', 'LIMITATION',
              'Weather (R5) genuinely helps inside the rain stratum (26.15→25.79 MAE) despite washing '
              'out in aggregate — rain is only 8.3% of test cells. Incidents (R6) do not help even inside '
              'the incident stratum (28.40→28.53, worse) — a genuine negative result, not just dilution.')
    st.divider()
    st.subheader('Full feature-ablation table')
    st.dataframe(D.ablation_table().round(3), width='stretch')

elif view == 'Ask the model (chatbot)':
    st.caption('Ask about specific predictions, model metrics, or incidents. Needs a free GEMINI_API_KEY in '
               '.streamlit/secrets.toml or the environment (get one at https://aistudio.google.com/apikey).')
    if 'chat_contents' not in st.session_state:
        st.session_state.chat_contents = []

    for turn in C.iter_turns(st.session_state.chat_contents):
        if turn['role'] == 'user':
            st.chat_message('user').write(turn['text'])
            continue
        with st.chat_message('assistant'):
            if turn['steps']:
                with st.expander(f"Reasoning ({len(turn['steps'])} step(s))"):
                    for step in turn['steps']:
                        args = ', '.join(f'{k}={v!r}' for k, v in step['args'].items())
                        st.markdown(f"**{step['tool']}**({args})")
                        st.json(step.get('result', {'status': 'no result recorded'}))
            if turn['text']:
                st.write(turn['text'])
                v = C.verify_answer(turn['text'], turn['steps'])
                if not v['verified']:
                    st.caption(f"⚠️ unverified figures: {', '.join(str(x) for x in v['flagged'])}")
            else:
                st.caption('No final answer was returned for this turn — see reasoning above.')

    if q := st.chat_input('e.g. "What is predicted for I-5 N at Poinsettia Ln on Dec 3 at 5pm?"'):
        st.session_state.chat_contents.append(C.new_user_content(q))
        with st.spinner('thinking...'):
            try:
                st.session_state.chat_contents = C.run_turn(st.session_state.chat_contents)
            except Exception as e:  # noqa: BLE001
                st.error(f'Chatbot error: {e}')
            else:
                if st.session_state.get('pending_nav'):
                    st.toast('Jumped to the requested date/time on the dashboard.', icon='🧭')
                st.rerun()

# ── Auto-play tick (placed last so the whole page renders at the current slice first) ──
if st.session_state.playing:
    delay = {'0.5x': 1.8, '1x': 0.9, '2x': 0.45, '4x': 0.22}[play_speed]
    time.sleep(delay)
    st.session_state.time_idx = (st.session_state.time_idx + 1) % len(time_labels)
    st.rerun()
