"""Page setup (config + CSS, every run), chart colors, small HTML helpers (show_html, esc, stat tiles) and the
color rules (tone, toned), plus the text helpers the Details panels share (caption / warning / info text,
stat cards, usd, MONEY).

Markdown ends an HTML block at the first blank line and turns 4-space-indented lines into code, so every custom
HTML block goes through show_html(): one line, no indentation, dynamic text escaped with esc()."""
import html
import re
from urllib.parse import quote

import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st


CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
    .stApp, .main { background: #f1f5f9; }
    .main .block-container, [data-testid="stMainBlockContainer"] { padding: 1.25rem 2rem 2.5rem 2rem !important; max-width: 1400px; }
    .stApp > header, header[data-testid="stHeader"], [data-testid="stDecoration"] { display: none !important; height: 0 !important; }
    #MainMenu, footer, header { visibility: hidden; }
    html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }
    h1, h2, h3 { color: #0f172a; font-weight: 600; letter-spacing: -0.02em; }
    [data-testid="stMetricValue"] { color: #0f172a; font-weight: 700; font-size: 1.05rem; font-family: 'JetBrains Mono', monospace; }
    [data-testid="stMetricLabel"] { color: #64748b; font-weight: 500; font-size: 0.8rem; }
    .stButton > button { background: #0f766e; color: #fff; border: none; border-radius: 10px; font-weight: 600; }
    .stButton > button:hover { background: #0d9488; color: #fff; }
    [data-testid="stExpander"] { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; }
    [data-baseweb="tab-list"] { background: #e2e8f0; border-radius: 12px; padding: 4px; gap: 4px; }
    [data-baseweb="tab"] { border-radius: 10px; font-weight: 600; color: #64748b; }
    [data-testid="stExpander"] summary p { font-weight: 600; color: #0f172a; }
    [data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
    [data-testid="stCaptionContainer"] { color: #64748b; line-height: 1.45; }
    .js-plotly-plot { border-radius: 12px; background: #fff; border: 1px solid #e2e8f0; padding: 4px; }
    .symbol-link { color: #0f766e; text-decoration: none; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }
    .symbol-link:hover { color: #0d9488; text-decoration: underline; }
    .sa-topbar { display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; margin-bottom: 0.4rem; flex-wrap: wrap; }
    .sa-topbar h1 { margin: 0; font-size: 1.45rem; font-weight: 700; }
    .sa-topbar p { margin: 0.2rem 0 0; color: #64748b; font-size: 0.9rem; }
    .sa-chip { font-size: 0.75rem; font-weight: 600; color: #0f766e; background: #ccfbf1; border: 1px solid #99f6e4;
               padding: 0.35rem 0.7rem; border-radius: 999px; white-space: nowrap; }
    .sa-section { font-weight: 700; color: #0f172a; margin: 0.6rem 0 0.2rem; font-size: 1.02rem; }
    .sa-chip-warn { color: #b45309; background: #fffbeb; border-color: #fde68a; }
    .sa-hero, .sa-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1rem 1.25rem; margin: 0.5rem 0 0.9rem;
                         box-shadow: 0 1px 3px rgba(15,23,42,.06); }
    .sa-card-title { font-size: 1.02rem; font-weight: 700; color: #0f172a; margin-bottom: 0.35rem; }
    .sa-card-title span { font-size: 0.8rem; font-weight: 500; color: #64748b; margin-left: 0.35rem; }
    .sa-tier { display: flex; gap: 0.75rem; align-items: baseline; padding: 0.45rem 0; border-top: 1px solid #f1f5f9; }
    .sa-tier > b { flex: 0 0 8.5rem; font-size: 0.7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #64748b; }
    .sa-tier-syms { display: flex; flex-wrap: wrap; gap: 0.3rem; font-size: 0.8rem; color: #64748b; align-items: center; }
    .sa-tier-syms .symbol-link { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 7px; padding: 1px 7px; font-size: 0.8rem; color: #334155; }
    .sa-tier-syms .symbol-link.t-good { color: #15803d; background: #f0fdf4; border-color: #bbf7d0; }
    .sa-tier-syms .symbol-link.t-warn { color: #b45309; background: #fffbeb; border-color: #fde68a; }
    .sa-tier-syms .symbol-link.t-bad { color: #b91c1c; background: #fef2f2; border-color: #fecaca; }
    .sa-tier-syms .symbol-link.sel { outline: 2px solid #0f766e; outline-offset: 1px; }
    .sa-note { font-size: 0.78rem; color: #475569; background: #f0fdfa; border: 1px solid #ccfbf1; border-radius: 10px;
               padding: 0.55rem 0.8rem; margin-top: 0.6rem; line-height: 1.5; }
    .sa-ident { display: flex; align-items: center; gap: 0.85rem; flex-wrap: wrap; }
    .sa-sym { font-size: 1.75rem; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; }
    .sa-price { font-size: 1.25rem; font-weight: 700; color: #334155; font-family: 'JetBrains Mono', monospace; }
    .sa-live { font-size: 0.85rem; font-weight: 600; color: #334155; font-family: 'JetBrains Mono', monospace; margin-top: 0.15rem; }
    .sa-live-dot { display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 9999px; background: #16a34a; margin-right: 0.3rem; }
    .sa-live-sub { font-weight: 400; color: #64748b; }
    .sa-badge { display: inline-block; padding: 0.35rem 0.8rem; border-radius: 999px; font-weight: 700; font-size: 0.8rem; }
    .sa-badge-bull { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .sa-badge-bear { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .sa-badge-hold { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .sa-badge-grey { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
    .sa-why { font-size: 0.8rem; color: #64748b; margin-top: 0.35rem; }
    .sa-stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(9.5rem, 1fr)); gap: 0.5rem; margin-top: 0.8rem; }
    .sa-stat { background: #f8fafc; border: 1px solid #eef2f7; border-radius: 10px; padding: 0.45rem 0.7rem; }
    .sa-stats .sa-stat-label { white-space: normal; }
    .sa-group { padding: 0.75rem 0 0.25rem; border-top: 1px solid #f1f5f9; }
    .sa-group-title { font-size: 0.7rem; font-weight: 700; letter-spacing: .09em; color: #64748b; text-transform: uppercase; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 0.6rem 0.85rem; }
    [data-testid="stAlert"] { border-radius: 12px; }
    .sa-stat-label { font-size: 0.66rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }
    .sa-stat-val { font-size: 0.95rem; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; white-space: nowrap; }
</style>
"""

CHART_FONT = "DM Sans, system-ui, sans-serif"
PCT_COL, SCORE_COL = st.column_config.NumberColumn(format="%.2f%%"), st.column_config.NumberColumn(format="%.1f")


MA_COLS = ['ma_10', 'ma_30', 'ma_50', 'ma_100', 'ma_200']
MA_COLORS = ['#0ea5e9', '#8b5cf6', '#f59e0b', '#a16207', '#94a3b8']   # chart lines: sky, violet, amber, brown, slate
TEAL, HOLD_SHADE = "#0f766e", "rgba(245,158,11,0.10)"               # price line; held periods (yellow = hold)
EARN_LINE = dict(color="rgba(234,88,12,0.45)", width=1, dash="dot")  # dotted vertical line at each earnings date


esc = html.escape


def show_html(markup):
    st.markdown("".join(line.strip() for line in str(markup).splitlines()), unsafe_allow_html=True)


def symbol_link(sym, cls=""):
    return f'<a href="?symbol={quote(sym)}" class="symbol-link {cls}" target="_self">{esc(sym)}</a>'


def section(title):
    show_html(f'<div class="sa-section">{esc(title)}</div>')


def num(v):
    return float(v) if pd.notna(v) else None


def fmt(v, spec, prefix="", suffix=""):
    return f"{prefix}{v:{spec}}{suffix}" if v is not None else "—"


# One color rule for both tabs: green = good / Bullish / buy / positive, yellow = hold / neutral / caution / warning,
# red = bad / Bearish / sell / negative / error. Plain labels and headings stay slate (INK).
GOOD, CAUTION, BAD, INK, MUTED = "#15803d", "#b45309", "#b91c1c", "#0f172a", "#94a3b8"   # MUTED = no data / cash
DD_RED = -10.0           # max drawdown: down to -10% yellow, worse than -10% red (0 = none yet, plain)
SCORE_GOOD = 40          # scores (strategy, technical, strength): above 40 green (about top-20 level), 0-40 yellow, below 0 red
TONE_WORDS = ((GOOD, ("buy", "bull", "fresh", "ok", "success", "add")),
              (BAD, ("sell", "sold", "bear", "score below", "missing", "error", "fail", "drop", "exit")),
              (CAUTION, ("hold", "keep", "watch", "neutral", "stale", "warn", "wait", "pending", "swap", "skip", "earn", "within")))
TONE_CLASS = {GOOD: "t-good", CAUTION: "t-warn", BAD: "t-bad"}


def tone(v, good=0.0, bad=0.0):
    """Color of a value. Text: by its first word (Buy / Hold / Sold, fresh / stale / missing...; other text stays plain).
    Number: above `good` green, below `bad` red, in between yellow (good == bad: exactly that value stays plain)."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return INK
    if isinstance(v, str):
        t = re.sub(r"^[^a-z]+", "", v.strip().lower())
        return next((c for c, words in TONE_WORDS if t.startswith(words)), INK)
    v = float(v)
    return GOOD if v > good else BAD if v < bad else (INK if good == bad else CAUTION)


def score_tone(v):
    return tone(v, SCORE_GOOD, 0.0)


def toned(df, cols, fn=None, **kw):
    """DataFrame (or Styler) -> Styler with `cols` colored by fn (default tone(v, **kw)); st.dataframe keeps the
    column_config formats."""
    sty = df if isinstance(df, Styler) else df.style
    fn = fn or (lambda v: tone(v, **kw))
    css = lambda v: "" if fn(v) == INK else f"color: {fn(v)}; font-weight: 600"
    return sty.map(css, subset=[c for c in cols if c in sty.data.columns])


def drawdown_tone(v):
    return INK if v is None or pd.isna(v) or v == 0 else tone(v, 0.0, DD_RED)


def live_row(sty, col, name):
    """Highlight the row whose `col` starts with `name` (the live rules) in the teal theme."""
    return sty.apply(lambda r: ["background-color: #f0fdfa; font-weight: 700" if str(r[col]).startswith(name) else ""] * len(r),
                     axis=1)


def sign_color(v):
    return tone(v)


def md_tone(text, color):
    """Markdown text in the tone color (st.caption / st.markdown color syntax)."""
    return {GOOD: f":green[{text}]", CAUTION: f":orange[{text}]", BAD: f":red[{text}]"}.get(color, text)


def stat_html(label, value, color="#0f172a", tip=None):
    """One label/value pair in the stock header."""
    title = f' title="{esc(tip)}"' if tip else ""
    return (f'<div class="sa-stat"{title}><div class="sa-stat-label">{esc(label)}{" ⓘ" if tip else ""}</div>'
            f'<div class="sa-stat-val" style="color:{color};">{esc(str(value))}</div></div>')


def caption_text(text):
    """st.caption with literal dollar signs (Streamlit reads $...$ as math)."""
    st.caption(str(text).replace("$", "\\$"))


def warning_text(text):
    st.warning(str(text).replace("$", "\\$"))


def info_text(text):
    st.info(str(text).replace("$", "\\$"))


def stat_cards(items):
    """A row of card-style numbers: [(label, value text, color)]."""
    show_html('<div class="sa-stats">' + "".join(
        f'<div class="sa-stat"><div class="sa-stat-label">{esc(lbl)}</div><div class="sa-stat-val" style="color:{c}">{esc(v)}</div></div>'
        for lbl, v, c in items) + '</div>')


def usd(v, sign=True):
    if v is None or v != v:
        return "—"
    return f"{'+' if sign and v > 0.005 else '-' if v < -0.005 else ''}${abs(v):,.2f}"


MONEY = st.column_config.NumberColumn(format="dollar")


def setup_page():
    """The page config (the first Streamlit call) and the CSS, on every run."""
    st.set_page_config(page_title="Stock Analysis Report", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
