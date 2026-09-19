"""Visual identity for the dashboard — Accenture-style palette (purple/black/white) for chrome,
kept separate from the semantic traffic colors (free=green/heavy=orange/congested=red) in
map_view.py, which stay as-is since they're a domain convention, not brand decoration."""
import streamlit as st

PURPLE = '#A100FF'          # Accenture Purple
PURPLE_DARK = '#3B0069'     # deep purple for banners / headers
PURPLE_DARKER = '#1E0038'   # near-black purple, gradient end
PURPLE_LIGHT = '#F4EAFC'    # tint for card backgrounds
PURPLE_LINE = '#DCC2F2'     # tint for borders
INK = '#161616'
MUTED = '#6E6E6E'
WHITE = '#FFFFFF'
GOOD = '#1BAF7A'
BAD = '#E4572E'

_FONT_STACK = "'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif"


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: {_FONT_STACK}; }}

    .block-container {{ padding-top: 1.6rem; max-width: 1200px; }}

    /* ---- Hero banner ---- */
    .hero {{
        background: linear-gradient(120deg, {PURPLE_DARKER} 0%, {PURPLE_DARK} 55%, {PURPLE} 130%);
        border-radius: 14px; padding: 1.6rem 2rem; margin-bottom: 1.3rem;
        box-shadow: 0 6px 20px rgba(59,0,105,0.18);
    }}
    .hero h1 {{ color: white; font-weight: 800; font-size: 1.9rem; margin: 0 0 0.3rem 0; letter-spacing: -0.01em; }}
    .hero p {{ color: #E9D5FA; font-size: 0.98rem; margin: 0; font-weight: 400; }}
    .hero .tag {{
        display: inline-block; background: rgba(255,255,255,0.14); color: white; font-weight: 600;
        font-size: 0.72rem; letter-spacing: 0.06em; padding: 0.18rem 0.6rem; border-radius: 20px;
        margin-bottom: 0.6rem; text-transform: uppercase;
    }}

    /* ---- KPI tiles ---- */
    .kpi-row {{ display: flex; gap: 0.9rem; margin-bottom: 1.4rem; flex-wrap: wrap; }}
    .kpi-card {{
        flex: 1 1 200px; background: {WHITE}; border: 1px solid {PURPLE_LINE}; border-radius: 12px;
        padding: 0.95rem 1.1rem; box-shadow: 0 2px 8px rgba(59,0,105,0.06);
        border-top: 3px solid {PURPLE};
    }}
    .kpi-card .kpi-label {{
        color: {MUTED}; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.05em; margin-bottom: 0.3rem;
    }}
    .kpi-card .kpi-value {{ color: {INK}; font-size: 1.65rem; font-weight: 800; line-height: 1.1; }}
    .kpi-card .kpi-delta {{ font-size: 0.8rem; font-weight: 600; margin-top: 0.2rem; }}
    .kpi-delta.good {{ color: {GOOD}; }}
    .kpi-delta.bad {{ color: {BAD}; }}
    .kpi-delta.neutral {{ color: {MUTED}; }}

    /* ---- Callout boxes (mirrors the LaTeX report's FINDING/IMPLICATION boxes) ---- */
    .callout {{
        border-radius: 8px; padding: 0.85rem 1.05rem; margin: 0.8rem 0; border-left: 4px solid;
        font-size: 0.93rem; line-height: 1.5;
    }}
    .callout .callout-tag {{
        display: inline-block; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; padding: 0.12rem 0.5rem; border-radius: 4px; color: white;
        margin-bottom: 0.35rem;
    }}
    .callout.finding {{ background: {PURPLE_LIGHT}; border-color: {PURPLE}; }}
    .callout.finding .callout-tag {{ background: {PURPLE}; }}
    .callout.limitation {{ background: #FDEDE7; border-color: {BAD}; }}
    .callout.limitation .callout-tag {{ background: {BAD}; }}

    /* ---- Section labels ---- */
    .section-eyebrow {{
        color: {PURPLE}; font-weight: 700; font-size: 0.78rem; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: -0.4rem;
    }}

    /* Tabs accent */
    button[data-baseweb="tab"] {{ font-weight: 600; }}
    </style>
    """, unsafe_allow_html=True)


def hero(title: str, subtitle: str, tag: str = 'LIVE MODEL REPLAY'):
    st.markdown(f'<div class="hero"><span class="tag">{tag}</span><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def kpi_row(cards: list[dict]):
    """cards: list of {'label', 'value', 'delta' (optional str), 'delta_kind' ('good'|'bad'|'neutral')}

    Built as ONE unindented line — st.markdown's HTML-block detection (CommonMark) only recognizes
    a block-level tag when it starts at column 0; any line with 4+ leading spaces (easy to get by
    accident from multi-line f-strings glued together) gets read back as an indented code block
    instead of HTML, which silently renders as literal text."""
    parts = ['<div class="kpi-row">']
    for c in cards:
        delta_html = ''
        if c.get('delta'):
            kind = c.get('delta_kind', 'neutral')
            delta_html = f'<div class="kpi-delta {kind}">{c["delta"]}</div>'
        parts.append(f'<div class="kpi-card"><div class="kpi-label">{c["label"]}</div>'
                     f'<div class="kpi-value">{c["value"]}</div>{delta_html}</div>')
    parts.append('</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


def callout(kind: str, tag: str, body: str):
    """kind: 'finding' or 'limitation'. Single-line HTML — see kpi_row's docstring for why."""
    st.markdown(f'<div class="callout {kind}"><span class="callout-tag">{tag}</span><br/>{body}</div>',
                unsafe_allow_html=True)
