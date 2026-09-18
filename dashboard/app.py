"""San Diego congestion forecasting — local demo dashboard.

Run with:  streamlit run dashboard/app.py
Needs data/traffic/congestion/predictions/R4_test.npz — build it first with:
    python scripts/export_predictions.py R4
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chatbot as C
import data as D
import map_view as MV

st.set_page_config(page_title='SD Congestion Forecast', layout='wide')

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

st.title('San Diego Freeway Congestion Forecast')
st.caption(f'Model: {RUN} (GWNet + congestion head) — test period '
           f'{times_h1.min():%Y-%m-%d %H:%M} to {(times_h1.max() + pd.Timedelta(minutes=11 * 15)):%Y-%m-%d %H:%M}')

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header('Replay controls')
min_day, max_day = times_h1.min().normalize(), times_h1.max().normalize()
sel_date = st.sidebar.date_input('Date', value=min_day.date(), min_value=min_day.date(), max_value=max_day.date())
day_mask = times_h1.normalize() == pd.Timestamp(sel_date)
day_times = times_h1[day_mask]
if len(day_times) == 0:
    st.sidebar.warning('No test data on this day.')
    st.stop()

time_labels = [t.strftime('%H:%M') for t in day_times]
sel_label = st.sidebar.select_slider('Time of day (target)', options=time_labels, value=time_labels[len(time_labels) // 2])
target_time = day_times[time_labels.index(sel_label)]

horizon = st.sidebar.slider('Forecast horizon (15-min steps ahead)', 1, D.HORIZON, 4,
                            help='How far ahead this prediction was made — 4 = 1 hour ahead')
st.sidebar.caption(f'Prediction made at {target_time - pd.Timedelta(minutes=15 * horizon):%Y-%m-%d %H:%M} '
                   f'for {target_time:%Y-%m-%d %H:%M}')

s = D.window_index(pred, target_time, horizon)
if s is None:
    st.warning('This time/horizon combination falls outside the test window range — pick an earlier time or smaller horizon.')
    st.stop()

pred_flow, true_flow, pred_class, true_class = D.slice_at(pred, s, horizon)
abs_t = int(test_starts[s] + horizon)
active = incidents['active'][abs_t]

tab_map, tab_sensor, tab_chat = st.tabs(['Map views', 'Sensor drill-down', 'Ask the model (chatbot)'])

with tab_map:
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
    st.subheader('Model comparison (feature ablation, test set)')
    st.dataframe(D.ablation_table().round(3), width='stretch')

with tab_sensor:
    labels = meta.loc[sensor_ids, 'label'].to_numpy()
    pick = st.selectbox('Sensor', options=range(len(sensor_ids)), format_func=lambda i: labels[i])
    sid = int(sensor_ids[pick])
    day_starts = test_starts[day_mask]
    actual_series = pred['true_flow'][day_mask, 0, pick].astype(np.float32)
    pred_by_h = {h: pred['pred_flow'][day_mask, h - 1, pick].astype(np.float32) for h in (1, 4, 12)}
    st.plotly_chart(MV.sensor_history_chart(day_times, actual_series, pred_by_h), width='stretch')
    st.metric('Predicted flow (this slot)', f'{pred_flow[pick]:.0f} veh/5min',
              delta=f'{pred_flow[pick] - true_flow[pick]:+.0f} vs. actual')
    st.metric('Predicted class', D.CLASSES[pred_class[pick]] if pred_class[pick] >= 0 else 'unknown')

with tab_chat:
    st.caption('Ask about specific predictions, model metrics, or incidents. Needs a free GEMINI_API_KEY in '
               '.streamlit/secrets.toml or the environment (get one at https://aistudio.google.com/apikey).')
    if 'chat_contents' not in st.session_state:
        st.session_state.chat_contents = []
    for content in st.session_state.chat_contents:
        if content.role not in ('user', 'model'):
            continue
        text = C.content_text(content)
        if text:
            st.chat_message('assistant' if content.role == 'model' else 'user').write(text)

    if q := st.chat_input('e.g. "What is predicted for I-5 N at Poinsettia Ln on Dec 3 at 5pm?"'):
        st.chat_message('user').write(q)
        st.session_state.chat_contents.append(C.new_user_content(q))
        with st.spinner('thinking...'):
            try:
                st.session_state.chat_contents = C.run_turn(st.session_state.chat_contents)
            except Exception as e:  # noqa: BLE001
                st.error(f'Chatbot error: {e}')
            else:
                text = C.content_text(st.session_state.chat_contents[-1])
                st.chat_message('assistant').write(text)
