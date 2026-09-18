"""Plotly scattermapbox builders for the four map blocks. Uses the tokenless
'open-street-map' style, so no Mapbox account is needed for a local demo."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data import CLASS_COLORS, CLASSES

MAP_STYLE = 'open-street-map'
CENTER = dict(lat=32.85, lon=-117.05)
ZOOM = 9.2


def _base_layout(title: str) -> dict:
    return dict(
        title=title, map_style=MAP_STYLE, map_center=CENTER, map_zoom=ZOOM,
        margin=dict(l=0, r=0, t=36, b=0), height=430,
        legend=dict(orientation='h', yanchor='bottom', y=0.01, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.7)'),
    )


def congestion_state_map(meta: pd.DataFrame, sensor_ids, pred_class: np.ndarray) -> go.Figure:
    """Block 1 — predicted free/heavy/congested class, one trace per class for a clean legend."""
    fig = go.Figure()
    m = meta.loc[sensor_ids]
    for i, cls in enumerate(CLASSES):
        sel = pred_class == i
        fig.add_trace(go.Scattermap(
            lat=m['Lat'][sel], lon=m['Lng'][sel], mode='markers', name=cls,
            marker=dict(size=8, color=CLASS_COLORS[cls]),
            text=m['label'][sel], hovertemplate='%{text}<br>predicted: ' + cls + '<extra></extra>',
        ))
    sel = pred_class < 0
    if sel.any():
        fig.add_trace(go.Scattermap(
            lat=m['Lat'][sel], lon=m['Lng'][sel], mode='markers', name='no label',
            marker=dict(size=6, color=CLASS_COLORS['unknown']), text=m['label'][sel],
            hovertemplate='%{text}<br>no label this step<extra></extra>',
        ))
    fig.update_layout(**_base_layout('Predicted congestion state'))
    return fig


def flow_compare_map(meta: pd.DataFrame, sensor_ids, pred_flow: np.ndarray, true_flow: np.ndarray) -> go.Figure:
    """Block 2 — predicted vs actual flow. Color = signed % error, size = actual flow volume."""
    m = meta.loc[sensor_ids]
    valid = true_flow > 0
    pct_err = np.where(valid, (pred_flow - true_flow) / np.maximum(true_flow, 1), 0.0) * 100
    size = 6 + 10 * np.clip(true_flow, 0, 400) / 400
    fig = go.Figure(go.Scattermap(
        lat=m['Lat'], lon=m['Lng'], mode='markers',
        marker=dict(size=size, color=pct_err, colorscale='RdBu_r', cmin=-50, cmax=50,
                    colorbar=dict(title='pred − actual\n(% of actual)', thickness=14), reversescale=False),
        text=[f'{lbl}<br>pred {p:.0f}  actual {t:.0f} veh/5min'
              for lbl, p, t in zip(m['label'], pred_flow, true_flow)],
        hovertemplate='%{text}<extra></extra>',
    ))
    fig.update_layout(**_base_layout('Predicted vs. actual flow'))
    return fig


def incident_map(meta: pd.DataFrame, sensor_ids, active: np.ndarray, pred_class: np.ndarray) -> go.Figure:
    """Block 3 — congestion state, with sensors under an active incident highlighted."""
    m = meta.loc[sensor_ids]
    fig = go.Figure()
    base = ~active.astype(bool)
    for i, cls in enumerate(CLASSES):
        sel = base & (pred_class == i)
        fig.add_trace(go.Scattermap(
            lat=m['Lat'][sel], lon=m['Lng'][sel], mode='markers', name=cls,
            marker=dict(size=7, color=CLASS_COLORS[cls], opacity=0.55),
            text=m['label'][sel], hovertemplate='%{text}<extra></extra>', showlegend=True,
        ))
    sel = active.astype(bool)
    if sel.any():
        fig.add_trace(go.Scattermap(
            lat=m['Lat'][sel], lon=m['Lng'][sel], mode='markers', name='active incident',
            marker=dict(size=13, color='#8e44ad', symbol='circle'),
            text=m['label'][sel], hovertemplate='%{text}<br>incident active<extra></extra>',
        ))
    fig.update_layout(**_base_layout('Incidents (active CHP incidents highlighted)'))
    return fig


def error_map(meta: pd.DataFrame, sensor_ids, pred_flow: np.ndarray, true_flow: np.ndarray) -> go.Figure:
    """Block 4 — model error magnitude only, for a model-quality read separate from the raw comparison."""
    m = meta.loc[sensor_ids]
    valid = true_flow > 0
    err = np.where(valid, np.abs(pred_flow - true_flow), np.nan)
    fig = go.Figure(go.Scattermap(
        lat=m['Lat'], lon=m['Lng'], mode='markers',
        marker=dict(size=9, color=err, colorscale='YlOrRd', cmin=0, cmax=60,
                    colorbar=dict(title='|error| veh/5min', thickness=14)),
        text=[f'{lbl}<br>|error| {e:.1f} veh/5min' if np.isfinite(e) else f'{lbl}<br>no actual (masked)'
              for lbl, e in zip(m['label'], err)],
        hovertemplate='%{text}<extra></extra>',
    ))
    fig.update_layout(**_base_layout('Error magnitude (model-quality view)'))
    return fig


def sensor_history_chart(times, actual, pred_by_horizon: dict) -> go.Figure:
    """Line chart for the sensor drill-down: actual flow plus a couple of predicted horizons."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=actual, name='actual', line=dict(color='#2c3e50', width=2)))
    palette = ['#3498db', '#e67e22', '#9b59b6']
    for (h, series), color in zip(pred_by_horizon.items(), palette):
        fig.add_trace(go.Scatter(x=times, y=series, name=f'pred h={h}', line=dict(color=color, dash='dot')))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                       title='Sensor flow: actual vs. predicted', yaxis_title='veh/5min')
    return fig
