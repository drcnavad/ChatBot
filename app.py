"""
Stock Analysis dashboard (Streamlit).

Page layout, top to bottom:
  1. Title bar.
  2. "Dashboard" tab: the single-stock view (clickable rank tiers, stock picker, chart). Open any stock directly with
     http://localhost:8502/?symbol=NVDA
  3. "Details" tab, each in its own expander: live Alpaca holdings, latest signals, the last decision (one view),
     data freshness, and the strategy rules at the bottom.

The app only READS the Reports/*.csv files written by `python run_all.py` for strategy data, plus the live holdings from the
Alpaca account (read-only GETs via alpaca_paper.py, at most once a minute). It never places orders and never calls
a paid data API (the optional "AI analysis" button uses the Hugging Face token from .env). The stock header additionally
shows a display-only live quote from yfinance (free), and yfinance also draws the price chart of stocks too new to trade
(short history); neither ever feeds back into signals, picks, backtests, or orders.
"""
import html
import json
import os
import re
import sys
from datetime import datetime
from types import SimpleNamespace
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.io.formats.style import Styler
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)  # project modules (backtest_engine, sector_mapping) importable from any cwd


# =====================================================================================================================
# 1. Live-rule settings (read from backtest_engine.WINNER, used for labels and captions only)
# =====================================================================================================================
try:
    from sector_mapping import sector_etf_for, symbol_sector
except Exception:  # deployed without the pipeline modules: relative strength vs SPY only
    symbol_sector = {}

    def sector_etf_for(symbol):
        return None

try:
    from backtest_engine import LIVE_INVESTED, MIN_BARS, WINNER, winner_max_per_sector
    SECTOR_MAX = winner_max_per_sector()
except Exception:
    WINNER, SECTOR_MAX, LIVE_INVESTED, MIN_BARS = {}, 4, 0.99, 200

STRATEGY_TAG = WINNER.get("tag", "C6")
RS_LABEL = {"etf": "vs sector ETF and SPY",
            "sector_median": "vs the median of its sector peers, and sector ETF vs SPY",
            "median_all": "vs the median of its sector peers, and sector median vs the universe median",
            }.get(WINNER.get("rs_benchmark", "etf"), "vs sector ETF and SPY")
MIDWEEK = WINNER.get("midweek_swap")                                 # Mon/Wed swap check (None = weekly only)
EXIT_BELOW = WINNER.get("midweek_exit_below") if MIDWEEK else None   # mid-week exit below this rank
MAX_PICK = WINNER.get("max_pick_rank")                               # picks only from ranks 1..MAX_PICK
CAP_SOFT = bool(WINNER.get("cap_soft"))                              # sector limit relaxed to fill 10 slots
EARNINGS = WINNER.get("earnings_block_days")                         # no new buys with earnings within N days (None = off)

N_PICKS = WINNER.get("n", 10)                                        # portfolio size (top 10)
W_TECH = WINNER.get("w_tech", 0.5)                                   # score = W_TECH x technical + (1 - W_TECH) x RS
REGIME_OFF = f"{WINNER.get('regime_symbol', 'QQQ')} at or below its 200-day average"   # market filter off
SCALE = WINNER.get("regime_scale", 0.5) if WINNER.get("use_regime", True) else None
HALVED = "halved" if SCALE == 0.5 else f"multiplied by {SCALE:g}" if SCALE else "unchanged"
MAX_WEIGHT = WINNER.get("max_weight")                                # no stock above this weight; the extra stays cash


def rules_text():
    """The live trading rules in plain words, in one place (numbers from backtest_engine.WINNER)."""
    band = WINNER.get("rebalance_band")
    days = " and ".join(MIDWEEK["days"]) if MIDWEEK else ""
    return (f"**Strategy rules ({STRATEGY_TAG})**\n"
            "- **When:** decisions use the prices at about 2:30 PM CT, 30 min before the close (the pipeline starts at 2:30 PM CT "
            "and trades in market hours: sells first, then buys sized from the cash free after the sells; any rest at 9 AM CT): "
            "the Friday rebalance (the week's last trading day)"
            + (f" and the {days} checks (the next trading day after a holiday)" if MIDWEEK else "") + ".\n"
            f"- **Score:** {W_TECH:g} × Technical + {1 - W_TECH:g} × Relative Strength {RS_LABEL}. Only stocks with a score "
            f"above {WINNER.get('min_score', 0):g} and at least {MIN_BARS} trading days of prices are ranked "
            "(newer stocks are listed as not traded yet).\n"
            f"- **Friday picks:** the {N_PICKS} best-ranked stocks" + (f" from ranks 1–{MAX_PICK}" if MAX_PICK else "")
            + f", max {SECTOR_MAX} per sector"
            + (f"; slots the sector limit leaves empty are filled from ranks 1–{MAX_PICK or 20} anyway" if CAP_SOFT else "")
            + "; fewer qualifying stocks = the rest in cash. Stocks that are not picked are sold.\n"
            f"- **Size:** weights ∝ 1 / 63-day volatility (less volatile = larger), scaled to {LIVE_INVESTED:.0%} invested, "
            "each weight rounded down to 0.01%.\n"
            + (f"- **Market filter:** with {REGIME_OFF} at a rebalance, every weight is {HALVED}.\n" if SCALE else "")
            + (f"- **Max per stock:** No single stock gets more than {MAX_WEIGHT:.0%}; any extra stays in cash.\n" if MAX_WEIGHT else "")
            + (f"- **Rebalance:** every pick is brought back to its weight unless it is within {band * 100:g} percentage point "
               "of it; overweight holdings are trimmed so new buys get their full weight.\n" if band else "")
            + (f"- **{days} swap:** if a stock that is not held ranks in the top {MIDWEEK['enter_top']} and a held stock has "
               f"fallen below rank {MIDWEEK['exit_below']}, the worst-ranked held stock is sold and the new one bought for "
               "the same dollar amount (repeated while both are true; "
               + ("the sector limit does not apply" if CAP_SOFT else f"max {SECTOR_MAX} per sector still applies") + ").\n"
               if MIDWEEK else "")
            + (f"- **{days} exit:** after the swaps, any holding ranked worse than {EXIT_BELOW} (or no longer ranked) is "
               "sold; the cash waits for the Friday rebalance.\n" if EXIT_BELOW else "")
            + (f"- **Earnings:** a stock that is not held is not bought when its next earnings date is within {EARNINGS} "
               "calendar days; on Friday its slot goes to the next eligible stock (else cash), mid-week it is just not bought. "
               "A held stock is not topped up before them.\n" if EARNINGS else "")
            + "- **Pre-earnings stop:** a held stock with earnings within 7 calendar days is sold if its price falls to its "
            "highest close since bought - 3 × ATR(14), from 7 days before the report through the reaction day (checked "
            "every 10 minutes in the pre-market, regular and after-hours sessions; in regular hours every share at once, "
            "outside them the whole shares and the fraction at the 9 AM CT check). One sale per report; the cash waits for "
            "the next scheduled run, and the stock is not bought back until after its reaction day.\n"
            + "- **Orders:** sells go first. Every order is a limit at the live quote: buy at the ask + 0.05%, sell at the "
            "bid - 0.05% (no market orders). An order whose quote is stale or wider than 0.5% waits for the next 9 AM CT "
            "check. The 2:30 PM CT run trades in market hours (2-decimal shares); from 5 minutes before the close, "
            "whole-share after-hours limit orders (until 7 PM CT). The 9 AM CT check the next trading day sends any rest. A missed decision is caught up at "
            "the next market session, unless the next decision is already due; a decision never runs twice.\n"
            "- **Signals** (the strategy's decision, dated): Buy = enters the portfolio · Hold = stays · Sold = leaves · "
            f"Watch = ranked but not picked · Score below {WINNER.get('min_score', 0):g} = not eligible. The plan chip = what "
            "the next Friday rebalance would do at the latest close.\n"
            "- **Changes:** Don't change the strategy until 12+ weeks of forward results (from Oct 2, 2026) compare against QQQ.\n")


# =====================================================================================================================
# 2. Page setup, styling and file locations
# =====================================================================================================================
load_dotenv()
st.set_page_config(page_title="Stock Analysis Report", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
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
""", unsafe_allow_html=True)

CHART_FONT = "DM Sans, system-ui, sans-serif"
PCT_COL, SCORE_COL = st.column_config.NumberColumn(format="%.2f%%"), st.column_config.NumberColumn(format="%.1f")

# Report files (all written by run_all.py)
REPORTS = os.path.join(ROOT, "Reports")
SIGNAL_CSV = os.path.join(REPORTS, "signal_analysis.csv")
EARNINGS_CSV = os.path.join(REPORTS, "earnings_date.csv")
PICKS_CSV = os.path.join(REPORTS, "strategy_picks.csv")
CHANGES_CSV = os.path.join(REPORTS, "strategy_changes.csv")
HOLDINGS_CSV = os.path.join(REPORTS, "strategy_holdings.csv")
DECISIONS_CSV = os.path.join(REPORTS, "strategy_decisions.csv")
MIDWEEK_CSV = os.path.join(REPORTS, "strategy_midweek_check.csv")
BENCH_CSV = os.path.join(REPORTS, "benchmark_prices.csv")
NEWS_CSV = os.path.join(REPORTS, "news_cleaned_df.csv")
SHORT_HISTORY_CSV = os.path.join(REPORTS, "short_history_reference.csv")   # main_signal_analysis.ipynb: too new to trade
COMPANY_XLSX = os.path.join(REPORTS, "complete_company_analysis.xlsx")
CT = ZoneInfo("America/Chicago")
MA_COLS = ['ma_10', 'ma_30', 'ma_50', 'ma_100', 'ma_200']
MA_COLORS = ['#0ea5e9', '#8b5cf6', '#f59e0b', '#a16207', '#94a3b8']   # chart lines: sky, violet, amber, brown, slate
TEAL, HOLD_SHADE = "#0f766e", "rgba(245,158,11,0.10)"               # price line; held periods (yellow = hold)
EARN_LINE = dict(color="rgba(234,88,12,0.45)", width=1, dash="dot")  # dotted vertical line at each earnings date

# file -> (what it is, max age in days before it is flagged stale)
FRESHNESS = {
    "signal_analysis.csv": ("prices, signals, strategy weights", 3),
    "strategy_picks.csv": ("current / provisional portfolio", 3),
    "strategy_decisions.csv": ("decision history: weekly rebalances + mid-week swaps (chart markers)", 3),
    "strategy_midweek_check.csv": ("this week's decisions: Friday rebalance + Mon/Wed swap checks", 3),
    "benchmark_prices.csv": ("SPY / QQQ / sector ETF closes (RS lines)", 3),
    "weighted_sentiment.csv": ("news sentiment scores", 7),
    "news_cleaned_df.csv": ("news articles for AI summaries", 7),
    "earnings_date.csv": ("earnings calendar", 14),
    "complete_company_analysis.xlsx": ("fundamentals / fair value", 30),
    "balance_sheet_weights.csv": ("balance-sheet scores", 30),
    "balance_sheet.csv": ("raw quarterly fundamentals", 100),
    "forward_test_daily.csv": ("forward test, one row per trading day (forward_test.py)", 4),
}
APP_FILES = {"signal_analysis.csv", "strategy_picks.csv", "news_cleaned_df.csv", "earnings_date.csv",
             "complete_company_analysis.xlsx", "strategy_decisions.csv", "benchmark_prices.csv"}


# =====================================================================================================================
# 3. Small HTML helpers
#    Markdown ends an HTML block at the first blank line and turns 4-space-indented lines into code, so every custom
#    HTML block goes through show_html(): one line, no indentation, dynamic text escaped with esc().
# =====================================================================================================================
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


# =====================================================================================================================
# 4. Data loading (cached; every cache key includes the file's modification time so new data shows up at once)
# =====================================================================================================================
@st.cache_data(ttl=3600)
def _read_csv_cached(path, mtime):
    return pd.read_csv(path)


def read_report_csv(path):
    """Cached CSV read; None when the file is missing."""
    return _read_csv_cached(path, os.path.getmtime(path)) if os.path.exists(path) else None


@st.cache_data(ttl=3600)
def load_signals(mtime: float):
    """Reports/signal_analysis.csv: one row per symbol and day (prices, indicators, strategy columns)."""
    return pd.read_csv(SIGNAL_CSV, parse_dates=['Date'])


@st.cache_data(ttl=3600)
def latest_rows(_df, mtime: float):
    """Most recent row per symbol (the leading underscore stops Streamlit hashing the whole frame)."""
    return _df.sort_values('Date', ascending=False).drop_duplicates(subset='Symbol', keep='first')


def load_decisions():
    """Reports/strategy_decisions.csv: the engine's log of every weekly / mid-week decision (with reasons)."""
    d = read_report_csv(DECISIONS_CSV)
    if d is None:
        return None
    d = d.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    return d


def load_benchmarks():
    b = read_report_csv(BENCH_CSV)
    if b is None:
        return None
    b = b.copy()
    b["Date"] = pd.to_datetime(b["Date"])
    return b.set_index("Date")


def load_midweek_rows():
    """Reports/strategy_midweek_check.csv (this week's Friday rebalance + Mon/Wed checks) or None."""
    m = read_report_csv(MIDWEEK_CSV)
    return None if m is None or m.empty or "Message" not in m.columns else m


def row_for(path, symbol):
    """First row of a Reports CSV for one symbol, or None."""
    t = read_report_csv(path)
    if t is None:
        return None
    row = t[t["Symbol"] == symbol]
    return None if row.empty else row.iloc[0]


def data_freshness(latest_bar):
    """One row per Reports file: last update (CT), age and a stale flag."""
    now = datetime.now(tz=CT)
    rows = []
    for name, (what, max_days) in FRESHNESS.items():
        path = os.path.join(REPORTS, name)
        if not os.path.exists(path):
            status = "⚠️ missing" if name in APP_FILES else "— not present (local pipeline file)"
            rows.append({"File": name, "Contents": what, "Last update (CT)": "—", "Latest data": "—", "Age": "—", "Status": status})
            continue
        ts = datetime.fromtimestamp(os.path.getmtime(path), tz=CT)
        age_d = (now - ts).total_seconds() / 86400
        latest = "—"
        if name == "signal_analysis.csv":  # judge by the newest price bar, not the file time (a git checkout resets mtimes)
            bar = pd.Timestamp(latest_bar)
            latest = f"{bar:%Y-%m-%d}"
            age_d = max(age_d, (pd.Timestamp(now.date()) - bar.normalize()).days - 1)
        rows.append({"File": name, "Contents": what, "Last update (CT)": ts.strftime("%Y-%m-%d %I:%M %p"), "Latest data": latest,
                     "Age": f"{age_d * 24:.0f} h" if age_d < 2 else f"{age_d:.0f} d",
                     "Status": "✅ fresh" if age_d <= max_days else f"⚠️ stale (> {max_days} d)"})
    return pd.DataFrame(rows)


# --- Decision-day ranks (Fri rebalance + Mon/Wed checks, holiday-shifted): rank change compares the last two decision days, not calendar days ---
def _decision_days(df):
    """Sorted decision dates (Rebalance_Day or Midweek_Check) present in the signal data."""
    mask = (df["Rebalance_Day"] == 1) | (df["Midweek_Check"] == 1)
    return sorted(df.loc[mask, "Date"].unique())


def _day_ranks(df, day):
    """Symbol -> rank (1 = best) for one date: the pipeline's Strategy_Rank when present, else by combined_signal.
    Only ranked symbols (non-null rank key) get a rank; unranked symbols are excluded, not parked at the bottom."""
    key = "Strategy_Rank" if "Strategy_Rank" in df.columns else "combined_signal"
    sub = (df.loc[df["Date"] == day, ["Symbol", key]].dropna(subset=[key])
           .sort_values([key], ascending=(key == "Strategy_Rank"))
           .drop_duplicates(subset=["Symbol"]))
    return pd.Series(range(1, len(sub) + 1), index=sub["Symbol"].values)


@st.cache_data(ttl=3600)
def day_rank_change(_df, mtime: float):
    """Symbol -> (prev_decision_rank - latest_decision_rank, latest_rank); positive = moved up.
    Compares the last two decision days, not consecutive calendar days."""
    df = _df
    days = _decision_days(df)
    if len(days) < 2:
        return {}
    t = _day_ranks(df, days[-1])
    y = _day_ranks(df, days[-2])
    common = t.index.intersection(y.index)
    return {s: (int(y[s]) - int(t[s]), int(t[s])) for s in common}


@st.cache_data(ttl=3600)
def _load_earnings(mtime: float):
    ed = pd.read_csv(EARNINGS_CSV)
    ed['Symbol'] = ed['Symbol'].astype(str).str.strip().str.upper()
    ed['Earnings Date'] = pd.to_datetime(ed['Earnings Date'], errors='coerce')
    return ed.dropna(subset=['Earnings Date'])


def load_earnings():
    if not os.path.exists(EARNINGS_CSV):
        return pd.DataFrame(columns=["Symbol", "Earnings Date", "Time"])
    return _load_earnings(os.path.getmtime(EARNINGS_CSV))


def last_next_earnings(symbols):
    """Per symbol: most recent past and nearest upcoming earnings date (YYYY-MM-DD or '')."""
    ed = load_earnings()
    today = pd.Timestamp.now().normalize()
    rows = []
    for sym in symbols:
        dates = ed.loc[ed['Symbol'] == sym, 'Earnings Date']
        last, nxt = dates[dates <= today].max(), dates[dates >= today].min()
        rows.append({'Symbol': sym, 'Last ED': last.strftime('%Y-%m-%d') if pd.notna(last) else '',
                     'Next ED': nxt.strftime('%Y-%m-%d') if pd.notna(nxt) else ''})
    return pd.DataFrame(rows, columns=['Symbol', 'Last ED', 'Next ED'])


@st.cache_data(ttl=3600)
def _load_company(mtime: float):
    return pd.read_excel(COMPANY_XLSX, sheet_name="2_Latest_Quarter_Complete", engine="openpyxl")


def load_company():
    """Latest-quarter fundamentals workbook; empty frame when missing."""
    if not os.path.exists(COMPANY_XLSX):
        return pd.DataFrame(columns=["Symbol"])
    return _load_company(os.path.getmtime(COMPANY_XLSX))


def company_metrics(ticker, company_df):
    """Fair value and key ratios for one ticker ({} if none on file)."""
    row = company_df[company_df['Symbol'].astype(str).str.strip().str.upper() == ticker]
    if row.empty:
        return {}
    r = row.iloc[0]
    cols = [('FairValue_Composite', 'fair_value'), ('PE_Ratio', 'pe_ratio'), ('PB_Ratio', 'pb_ratio'),
            ('RevenueGrowth_YoY', 'revenue_growth_yoy'), ('TTM_ROE', 'roe'), ('TTM_NetProfitMargin', 'net_margin'),
            ('Debt_to_Equity', 'debt_to_equity')]
    return {key: float(r[col]) for col, key in cols if pd.notna(r[col])}


def load_news():
    news = read_report_csv(NEWS_CSV)
    return news if news is not None else pd.DataFrame(columns=["symbol", "date", "headline", "summary", "source", "sentiment_label"])


# =====================================================================================================================
# 4b. Live quotes (display only; never touches strategy, signals, picks, backtests, or orders)
# =====================================================================================================================
try:
    import yfinance as _yf
    _YF_OK = True
except ImportError:  # graceful degradation: dashboard works without live quotes
    _yf, _YF_OK = None, False


@st.cache_data(ttl=120)
def live_quote(ticker: str):
    """Latest yfinance quote for one ticker -> (price, bar_time) or None.

    Display-only helper for the stock header. The strategy always uses the completed
    daily-bar Close from Reports/*.csv; this result never feeds back into signals,
    picks, backtests, or orders. Returns None when yfinance is missing, the fetch
    fails/rate-limits, or there is no usable bar.
    """
    if not _YF_OK or not ticker:
        return None
    try:
        df = _yf.download(ticker, period="1d", interval="1m", prepost=True,
                          progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        closes = df["Close"]
        if isinstance(closes, pd.DataFrame):  # yfinance>=0.12 returns MultiIndex columns
            closes = closes.iloc[:, 0]
        closes = closes.dropna()
        if closes.empty:
            return None
        return float(closes.iloc[-1]), pd.Timestamp(closes.index[-1])
    except Exception:
        return None


# =====================================================================================================================
# 5. Optional AI analysis (Hugging Face; only runs when the button is pressed)
# =====================================================================================================================
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@st.cache_resource(show_spinner=False)
def hf_token():
    """HF token from Streamlit secrets, falling back to .env (cached: st.secrets lookups are slow)."""
    try:
        return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN", "")
    except Exception:
        return os.getenv("HF_TOKEN", "")


def llm_chat(messages, max_tokens):
    """Run a Llama chat completion and strip trailing prompt artifacts."""
    try:
        from huggingface_hub import InferenceClient   # imported on first use
        response = InferenceClient(token=hf_token()).chat_completion(
            model=LLM_MODEL, messages=messages, max_tokens=max_tokens, temperature=0.2)
        return re.split(r'\[/?USER\]|Can you|Could you', response.choices[0].message.content.strip())[0].strip()
    except Exception as e:
        return f"Error generating summary: {e}"


def trend_deltas_text(ticker_df, windows=(14, 50, 200)):
    """Compact multi-window trend text for the LLM prompt."""
    recent = ticker_df.sort_values("Date")
    latest = recent.iloc[-1]
    text = "Trend Deltas:\n"
    for w in windows:
        if len(recent) < w:
            continue
        past, tail = recent.iloc[-w], recent.tail(w)
        text += (f"Last {w} days: Price {(latest['Close'] / past['Close'] - 1) * 100:.2f}%, "
                 f"RSI change {latest['RSI'] - past['RSI']:.2f}, MACD change {latest['macd'] - past['macd']:.2f}, "
                 f"Price vs MA30 {(latest['Close'] / latest['ma_30'] - 1) * 100:.2f}%, "
                 f"Price vs MA200 {(latest['Close'] / latest['ma_200'] - 1) * 100:.2f}%, "
                 f"Above MA200 {(tail['Close'] > tail['ma_200']).mean() * 100:.2f}% of days\n")
    return text


def ai_stock_summary(ticker, ticker_df, signal, why):
    """AI summary + recommendation for one ticker."""
    latest = ticker_df.nlargest(1, 'Date').iloc[0]
    price = latest['Close']
    ma_lines = "\n".join(f"Price - {ma.upper().replace('_', '')}: ${price - latest[ma]:.2f} ({(price / latest[ma] - 1) * 100:.2f}%)"
                         for ma in MA_COLS)
    context = (f"Stock: {ticker}\nDate: {latest['Date']:%Y-%m-%d}\nCurrent Price: ${price:.2f}\n"
               f"Weekly strategy signal: {signal} ({why})\nStrategy Rank: {latest.get('Strategy_Rank')}\n"
               f"Technical Score: {latest['Technical_Score']:.2f}\n"
               f"Relative Strength Score: {latest.get('RS_Score', float('nan')):.2f}\n"
               f"Strategy Score (0.5 technical + 0.5 relative strength): {latest['combined_signal']:.2f}\n"
               f"RSI: {latest['RSI']:.2f}\nMACD: {latest['macd']:.2f}\n\n"
               f"Price vs Moving Averages (Difference):\n{ma_lines}\n\n"
               f"Balance Sheet Score: {latest['Fundamental_Weight']:.2f}\nSentiment Score: {latest['SentimentScore']:.2f}\n\n"
               f"{trend_deltas_text(ticker_df)}")
    system = ("You are a financial advisor. Provide ONLY a concise summary (4-5 sentences) followed by a clear AI recommendation. "
              "DO NOT list individual metrics, scores, or numbers in your response. "
              "DO NOT mention specific values like 'Balance Sheet Score: X', 'News Sentiment Score: Y', or 'RSI: Z'. "
              "Instead, synthesize all the data into a brief, readable summary that considers all factors holistically. "
              "Keep numbers and units intact when absolutely necessary. Ensure text is clean and readable (no LaTeX/special fonts). "
              "Your output should be brief, precise, and easy to read - focus on the overall picture, not individual data points.")
    user = ("Analyze the following stock data comprehensively. Consider ALL factors: "
            "- Price trends and moving average positions (positive % = above MA/bullish, negative % = below MA/bearish) "
            "- Balance Sheet Score (above 11=excellent, above 5=good, above 2=average, below 2=bad, below -5=very bad) "
            "- News Sentiment Score (above 7=excellent, above 4=good, above 0=neutral, below -1=bad, below -4=very bad) "
            "- Technical indicators (MA, RSI, MACD) and trend deltas \n\n"
            "Provide ONLY: 1. A concise 4-5 sentence summary synthesizing the key factors (DO NOT list individual metrics or scores) "
            "2. A clear AI recommendation: BULLISH, BEARISH, or HOLD with brief 1-2 sentences reasoning \n\n"
            "Remember: Do NOT mention specific score values or metrics in your response. Synthesize everything into a holistic view. "
            f"\n\n{context}")
    return llm_chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=400)


def ai_news_summary(news, sentiment_type, symbol, max_articles=20):
    """AI bullet summary of the positive or negative news for a symbol."""
    news = news.sort_values('date', ascending=False).head(max_articles)
    articles = "".join(f"Article {i}:\nDate: {r['date']}\nSource: {r['source']}\nHeadline: {r['headline']}\nSummary: {r['summary']}\n\n"
                       for i, (_, r) in enumerate(news.iterrows(), 1))
    system = ("You are a financial news analyst. Provide a concise summary of the news articles provided. "
              "Focus on key themes, trends, and important information that would be relevant for stock analysis. "
              "Respond in 2-4 bullet points, each on a new line. Keep the summary factual and objective. Do not repeat the same information.")
    user = (f"Analyze the following {sentiment_type} news articles for {symbol} and provide a summary:\n\n"
            f"Total articles: {len(news)}\n\n{articles}\n\n"
            f"Provide a concise summary highlighting the main themes and key information from these {sentiment_type} news articles. "
            "Ensure that the text is clean and readable. Do not use LaTeX formatting or special fonts for numbers (e.g. use '100' not '$100$'). "
            "Make sure words are not broken up and sentences are complete.")
    return llm_chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=500)


@st.dialog("AI Analysis", width="large")
def ai_analysis_dialog(ticker, ticker_df, signal, why):
    """Pop-up: AI technical summary plus positive / negative news summaries."""
    with st.spinner(f"Generating AI summary for {ticker}..."):
        summary = ai_stock_summary(ticker, ticker_df, signal, why)
    st.markdown(f"### {ticker}")
    st.markdown(summary)
    st.divider()
    news = load_news()
    symbol_news = news[news['symbol'] == ticker]
    if symbol_news.empty:
        st.info(f"No news articles found for {ticker}")
        return
    for col, label in zip(st.columns(2), ("positive", "negative")):
        subset = symbol_news[symbol_news['sentiment_label'] == label]
        with col:
            st.markdown(f"**{label.capitalize()} news**")
            if subset.empty:
                st.caption("None found")
                continue
            with st.spinner(f"Summarizing {label} headlines..."):
                st.markdown(ai_news_summary(subset, label, ticker))


# =====================================================================================================================
# 6. Plain-language signals (display only; the CSV values stay unchanged)
# =====================================================================================================================
SIGNALS = ["Buy", "Hold", "Sold", "Score below 0", "Watch", "Watch (sector limit)", "Not ranked"]
SIGNAL_BADGE = {"Buy": "sa-badge-bull", "Hold": "sa-badge-hold", "Sold": "sa-badge-bear",
                "Score below 0": "sa-badge-bear", "Watch": "sa-badge-hold", "Watch (sector limit)": "sa-badge-hold",
                "Not ranked": "sa-badge-grey"}
PLAN_BADGE = {"Buy": "sa-badge-bull", "Keep": "sa-badge-hold", "Sell": "sa-badge-bear"}


def plain_reason(signal, reason, rank=None, score=None):
    """Engine reason string -> short plain-English explanation for the given display signal."""
    reason = "" if reason is None or (isinstance(reason, float) and pd.isna(reason)) else str(reason)
    r = f"{rank:.0f}" if rank is not None and pd.notna(rank) else "?"
    sc = f"{score:.1f}" if score is not None and pd.notna(score) else "?"
    m = re.search(r"rank (\d+)", reason)
    if reason.startswith("earnings in"):
        return f"rank {r}, not bought: {reason.split(': not bought')[0]} (no new buys within {EARNINGS or 5} days of earnings)"
    if reason.startswith("mid-week swap in"):
        rep_m = re.search(r"replaces (\S+)", reason)
        return (f"mid-week swap: jumped into the top {MIDWEEK['enter_top'] if MIDWEEK else 3} at rank "
                f"{m.group(1) if m else r}" + (f", replaces {rep_m.group(1)}" if rep_m else ""))
    if reason.startswith("mid-week exit"):
        return (f"mid-week exit: {'rank ' + m.group(1) if m else 'no longer ranked'} is worse than {EXIT_BELOW or 30}; "
                "sold, cash until the Friday rebalance")
    if reason.startswith("mid-week swap out"):
        by = re.search(r"replaced by (\S+)", reason)
        return (f"mid-week swap: fell to {'rank ' + m.group(1) if m else 'no longer qualifying'} "
                f"(below {MIDWEEK['exit_below'] if MIDWEEK else 15})" + (f", replaced by {by.group(1)}" if by else ""))
    if signal == "Buy":
        rk = int(m.group(1)) if m else (int(rank) if rank is not None and pd.notna(rank) else None)
        if "sector cap relaxed" in reason:
            return f"made the portfolio at rank {rk} (free slot filled from the top {MAX_PICK or 20}, sector limit relaxed)"
        if rk is not None and rk > N_PICKS:
            return (f"made the portfolio at rank {rk} (higher-ranked stocks were skipped by the {SECTOR_MAX}-per-sector limit"
                    + (" or the earnings rule)" if EARNINGS else ")"))
        return f"made the top {N_PICKS} at rank {rk if rk is not None else r}"
    if signal == "Hold":
        return f"in top {N_PICKS}, rank {r}" if rank is not None and pd.notna(rank) and rank <= N_PICKS else \
            f"still selected at rank {r} (higher-ranked stocks skipped by the sector limit)"
    if signal == "Sold":
        if reason.startswith("score"):
            return f"score fell below 0 ({sc})"
        if "picks only from ranks" in reason:
            return f"fell to rank {m.group(1) if m else r}, worse than {MAX_PICK} (picks only from ranks 1–{MAX_PICK})"
        if reason.startswith("skipped"):
            return f"skipped: already {SECTOR_MAX} stocks from this sector"
        if "outside top" in reason:
            return f"fell to rank {m.group(1) if m else r}, outside top {N_PICKS}"
        if reason.startswith("not eligible"):
            return "not enough data / not eligible"
        return reason or f"left the top {N_PICKS}"
    if signal == "Watch (sector limit)":
        return f"rank {r} but skipped: already {SECTOR_MAX} stocks from this sector"
    if signal == "Watch":
        return f"rank {r}, positive score but outside top {N_PICKS} — watch"
    if signal == "Score below 0":
        return f"score below 0 ({sc})"
    return "benchmark / not enough history to rank"


def decision_tag(signal, reason):
    """Short reason shown in brackets on the dated decision badge, e.g. 'Sold (sector limit)'."""
    reason = "" if reason is None or (isinstance(reason, float) and pd.isna(reason)) else str(reason)
    if signal == "Sold":
        for key, tag in (("mid-week exit", "mid-week exit"), ("mid-week swap out", "mid-week swap"), ("skipped", "sector limit"),
                         ("picks only from ranks", f"rank worse than {MAX_PICK}"), ("outside top", f"outside top {N_PICKS}"),
                         ("score", "score below 0"), ("not eligible", "not eligible")):
            if key in reason:
                return tag
    if signal == "Buy" and reason.startswith("mid-week swap in"):
        return "mid-week swap"
    return ""


def signal_board(df):
    """Every symbol's signal for the decisions in force (the last rebalance / mid-week check).

    Uses Reports/strategy_changes.csv (the engine's decisions) plus signal_analysis.csv for stocks not in it.
    Returns (board DataFrame, decision date)."""
    ch = read_report_csv(CHANGES_CSV)
    sub = ch[ch["Symbol"].notna()] if ch is not None else pd.DataFrame()
    if not sub.empty:
        # The decisions in force are the LATEST ones: sort newest-first so iloc[0]
        # and drop_duplicates(keep="first") both pick the latest decision per symbol.
        sub = sub.sort_values("Date", ascending=False)
        date = pd.Timestamp(sub["Date"].iloc[0])
    else:
        reb = df.loc[df["Rebalance_Day"] == 1, "Date"]
        date = reb.max() if len(reb) else df["Date"].max()
    day = df[df["Date"] == date].drop_duplicates("Symbol").set_index("Symbol")
    dec = sub.drop_duplicates("Symbol").set_index("Symbol") if not sub.empty else pd.DataFrame()
    next_ed = last_next_earnings(list(day.index)).set_index("Symbol")["Next ED"]
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    rows = []
    for sym, r in day.iterrows():
        d = dec.loc[sym] if sym in dec.index else None
        score = d["Score"] if d is not None and pd.notna(d["Score"]) else r.get("Strategy_Score")
        rank = d["Rank"] if d is not None and pd.notna(d["Rank"]) else r.get("Strategy_Rank")
        status = d["Status"] if d is not None else None
        if status == "add":
            sig, weight = "Buy", d["New_Weight"]
        elif status == "hold":
            sig, weight = "Hold", d["New_Weight"]
        elif status == "drop":
            sig, weight = "Sold", d["Old_Weight"]
        elif d is not None and str(d["Reason"]).startswith("skipped"):
            sig, weight = "Watch (sector limit)", 0.0
        elif pd.isna(score):
            sig, weight = "Not ranked", np.nan
        else:
            sig, weight = ("Score below 0" if score <= 0 else "Watch"), 0.0
        ned = next_ed.get(sym, "")
        soon = ""
        if ned:
            n_days = int(np.busday_count(today.date(), pd.Timestamp(ned).date()))
            soon = "⚠️ within 2 sessions" if 0 <= n_days <= 2 else ""
        sector = d["Sector"] if d is not None and "Sector" in d and pd.notna(d["Sector"]) else symbol_sector.get(sym)
        rows.append({"Symbol": sym, "Signal": sig, "Rank": rank, "Score": score,
                     "Portfolio weight %": weight * 100 if pd.notna(weight) else np.nan, "Sector": sector or "—",
                     "Why": plain_reason(sig, d["Reason"] if d is not None else None, rank, score),
                     "Tag": decision_tag(sig, d["Reason"] if d is not None else None),
                     "Next earnings": ned, "Earnings soon": soon})
    board = pd.DataFrame(rows, columns=["Symbol", "Signal", "Rank", "Score", "Portfolio weight %", "Sector", "Why",
                                        "Next earnings", "Earnings soon", "Tag"])
    board = board.sort_values(["Rank", "Symbol"], na_position="last").reset_index(drop=True)
    picked = board["Signal"].isin(["Buy", "Hold"])
    board.insert(2, "Portfolio slot", "—")
    board.loc[picked, "Portfolio slot"] = [str(i) for i in range(1, int(picked.sum()) + 1)]
    return board, date


def next_full_rebalance(day):
    """Date of the first full (Friday) rebalance after `day` (backtest_engine's decision calendar), or None."""
    try:
        from backtest_engine import next_decision
        for _ in range(6):
            day, kind, _fill = next_decision(pd.Timestamp(day))
            if kind == "full rebalance":
                return day
    except Exception:
        pass
    return None


def rebalance_plan(df):
    """(rebalance date, as-of date, {SYMBOL: (action, weight %)}) for the next full rebalance.

    Provisional_Weight in strategy_picks.csv = the full rebalance computed at the latest close - the numbers the trade
    step (paper_trade.py) trades on the rebalance day. Action vs the holdings going into that rebalance: Buy (new),
    Keep (stays, brought to the weight) or Sell (leaves); a stock not listed is not picked."""
    picks = read_report_csv(PICKS_CSV)
    if picks is None or picks.empty or "Provisional_Weight" not in picks.columns:
        return None, None, {}
    as_of, last_reb = pd.Timestamp(picks["As_Of"].iloc[0]), pd.Timestamp(picks["Last_Rebalance"].iloc[0])
    day = as_of if as_of == last_reb else next_full_rebalance(as_of)
    before = df[df["Date"] < day] if day is not None else df
    prev = before[before["Date"] == before["Date"].max()]
    held = set(prev.loc[prev["Strategy_Weight"].fillna(0) > 0, "Symbol"])
    plan = {s: ("Keep" if s in held else "Buy", w * 100)
            for s, w in zip(picks["Symbol"], picks["Provisional_Weight"].fillna(0.0)) if w > 0}
    plan.update({s: ("Sell", 0.0) for s in held if s not in plan})
    return day, as_of, plan


def plan_text(plan, sym):
    """'Buy 14.23%' / 'Keep 9.14%' / 'Sell' / 'not picked'."""
    action, w = plan.get(sym, (None, 0.0))
    return f"{action} {w:.2f}%" if action in ("Buy", "Keep") else (action or "not picked")


def next_decision_date(latest_day):
    """(date, kind) of the next decision close: 'full rebalance' or 'mid-week check' (from backtest_engine)."""
    try:
        from backtest_engine import next_decision
        d, kind, _fill = next_decision(pd.Timestamp(latest_day))
        return d, kind
    except Exception:
        return None, None


# =====================================================================================================================
# 7. Live-strategy history for one ticker (entries / exits / held periods, portfolio slots, relative strength)
# =====================================================================================================================
EVENT_COLS = ["Kind", "Decision", "Fill", "Price", "Reason", "Rank", "Score", "Weight", "Regime_On"]


def _fallback_reason(kind, row, n=10):
    """Reason when the decision log is unavailable (derived from the saved rank/score columns)."""
    if kind == "entry":
        return f"selected (rank {row.Strategy_Rank:.0f})" if pd.notna(row.Strategy_Rank) else "selected"
    if pd.isna(row.Strategy_Score):
        return "not eligible / no data"
    if row.Strategy_Score <= 0:
        return f"score {row.Strategy_Score:.1f} <= 0"
    if pd.notna(row.Strategy_Rank) and row.Strategy_Rank > n:
        return f"rank {row.Strategy_Rank:.0f} outside top {n}"
    return "skipped: sector cap"


def strategy_events(ticker_df, decisions, symbol):
    """Entries/exits and held periods of the live strategy for one ticker.

    Strategy_Weight on day d is the target decided at d's close (orders go out that day at 2:30 PM CT), so an entry/exit is the
    first day the weight turns >0 / back to 0; its marker sits on the following session ("Fill"). Reasons, rank and score come from
    Reports/strategy_decisions.csv. Returns (events, periods) with periods = [(first held session, last held session)]."""
    t = ticker_df.sort_values("Date")[["Date", "Close", "Strategy_Weight", "Strategy_Rank", "Strategy_Score", "Regime_On"]]
    t = t.reset_index(drop=True)
    empty = pd.DataFrame(columns=EVENT_COLS)
    if t["Strategy_Weight"].notna().sum() == 0:
        return empty, []
    held = (t["Strategy_Weight"].fillna(0) > 0).to_numpy()
    prev = np.r_[False, held[:-1]]
    dec = pd.DataFrame()
    if decisions is not None:
        dec = decisions[decisions["Symbol"] == symbol].drop_duplicates("Date", keep="last").set_index("Date")
    rows = []
    for i in np.flatnonzero(held != prev):
        if i == 0:  # already held when the data window starts
            continue
        kind = "entry" if held[i] else "exit"
        r = t.iloc[i]
        has_next = i + 1 < len(t)
        d = dec.loc[r.Date] if r.Date in dec.index else None
        rows.append({
            "Kind": kind, "Decision": r.Date, "Fill": t["Date"].iloc[i + 1] if has_next else pd.NaT,
            "Price": t["Close"].iloc[i + 1] if has_next else r.Close,
            "Reason": d["Reason"] if d is not None else _fallback_reason(kind, r),
            "Rank": d["Rank"] if d is not None and pd.notna(d["Rank"]) else r.Strategy_Rank,
            "Score": d["Score"] if d is not None and pd.notna(d["Score"]) else r.Strategy_Score,
            "Weight": r.Strategy_Weight if kind == "entry" else t["Strategy_Weight"].iloc[i - 1],
            "Regime_On": r.Regime_On,
        })
    events = pd.DataFrame(rows, columns=EVENT_COLS) if rows else empty
    periods, start = [], (t["Date"].iloc[0] if held[0] else None)
    for e in events.itertuples():
        if e.Kind == "entry":
            start = e.Fill if pd.notna(e.Fill) else None
        elif start is not None:
            periods.append((start, e.Fill if pd.notna(e.Fill) else t["Date"].iloc[-1]))
            start = None
    if start is not None:
        periods.append((start, t["Date"].iloc[-1]))
    return events, periods


def event_hover(e):
    """Hover text for an entry/exit marker on the price chart."""
    sig = "Buy" if e.Kind == "entry" else "Sold"
    text = f"<b>{sig}</b>: {html.escape(plain_reason(sig, e.Reason, e.Rank, e.Score))}"
    text += f"<br>Decided at the close {e.Decision:%a %b %-d} · orders go out that day" + ("" if pd.notna(e.Fill) else " (pending)")
    bits = ([f"Rank #{e.Rank:.0f}"] if pd.notna(e.Rank) else []) + ([f"Score {e.Score:.1f}"] if pd.notna(e.Score) else [])
    if pd.notna(e.Weight):
        bits.append(f"{'Portfolio weight' if e.Kind == 'entry' else 'Weight sold'} {e.Weight * 100:.2f}%")
    if bits:
        text += "<br>" + " · ".join(bits)
    if e.Regime_On == 0:
        text += f"<br>Market filter OFF that week ({REGIME_OFF}): positions {HALVED}"
    return text


RS_COLORS = {"stock": "#1d4ed8", "market": "#64748b", "sector": "#d97706"}   # stock blue, SPY grey, sector ETF orange


def performance_names(symbol):
    """Display names of the comparison lines: {'stock': 'FTNT', 'market': 'SPY (market)', 'sector': 'XLK (Technology sector)'}."""
    etf, sector = sector_etf_for(symbol), symbol_sector.get(symbol)
    names = {"stock": symbol, "market": "SPY (market)"}
    if etf and etf != symbol:
        names["sector"] = f"{etf} ({sector} sector)" if sector else f"{etf} (sector ETF)"
    return names


def relative_strength_lines(chart, symbol):
    """% price change since the start of the chart window: the stock, SPY and its sector ETF -> {key: (name, series)}."""
    bench = load_benchmarks()
    if bench is None:
        return {}
    b = bench.reindex(chart.index).ffill()
    names, refs = performance_names(symbol), {"stock": None, "market": "SPY", "sector": sector_etf_for(symbol)}
    lines = {}
    for key, name in names.items():
        if key != "stock" and (refs[key] not in b.columns):
            continue
        px = (chart["Close"] if key == "stock" else b[refs[key]]).replace([np.inf, -np.inf], np.nan)
        first = px.first_valid_index()
        if first is not None and px.loc[first]:
            lines[key] = (name, (px / px.loc[first] - 1) * 100)
    return lines if len(lines) > 1 else {}


def daily_status(weight, score):
    """Status on an ordinary day (between decisions) for the chart hover text."""
    if score is None or pd.isna(score):
        return "Not ranked"
    if weight is not None and pd.notna(weight) and weight > 0:
        return "Hold"
    return "Score below 0" if score <= 0 else "Watch"


def legacy_periods(mask):
    """(start, end) pairs for each run of True in a date-indexed boolean Series (end = next trading day)."""
    runs = (mask != mask.shift()).cumsum()
    next_day = pd.Series(mask.index, index=mask.index).shift(-1).fillna(mask.index[-1] + pd.Timedelta(days=1))
    return [(g.index[0], next_day[g.index[-1]]) for _, g in mask[mask].groupby(runs[mask])]


def legacy_flips(frame):
    """Rows where the old-rule final_trade flips between BUY and SELL (HOLD/EARNING days ignored)."""
    direction = frame[frame['final_trade'].isin(['BUY', 'SELL'])].sort_values(['Symbol', 'Date'])
    return direction[direction['final_trade'].ne(direction.groupby('Symbol')['final_trade'].shift())]


# =====================================================================================================================
# 8. Page state: everything the render functions need, computed once per run
# =====================================================================================================================
def build_page():
    mtime = os.path.getmtime(SIGNAL_CSV)
    df = load_signals(mtime)
    latest = latest_rows(df, mtime)
    board_off, off_date = signal_board(df)
    next_dec, next_kind = next_decision_date(df["Date"].max())
    plan_day, plan_asof, plan = rebalance_plan(df)
    ref = load_short_history()
    short = (ref if ref is not None else pd.DataFrame(columns=["Symbol"])).set_index("Symbol")
    return SimpleNamespace(
        df=df, mtime=mtime, latest=latest, by_symbol=latest.set_index("Symbol"),
        options=latest.sort_values('combined_signal', ascending=False)['Symbol'].tolist()   # dropdown: best score first,
        + [s for s in short.index if s not in set(latest['Symbol'])],                       # then short-history stocks
        short=short,
        rank_change=day_rank_change(df, mtime),
        board_off=board_off.drop(columns="Tag"), off_date=off_date,
        tag_off=dict(zip(board_off["Symbol"], board_off["Tag"])),
        rank_off=dict(zip(board_off["Symbol"], board_off["Rank"])),
        plan_day=plan_day, plan_asof=plan_asof, plan=plan,
        sig_off=dict(zip(board_off["Symbol"], board_off["Signal"])),
        why_off=dict(zip(board_off["Symbol"], board_off["Why"])),
        slot_off=dict(zip(board_off["Symbol"], board_off["Portfolio slot"])),
        next_dec=next_dec, next_kind=next_kind, midweek=load_midweek_rows(),
        freshness=data_freshness(df["Date"].max()),
    )


def open_symbol(table, event, key):
    """Row click in a table -> open that stock in the stock view (applied on the rerun, before the picker is drawn)."""
    rows = event.selection.rows if event is not None and hasattr(event, "selection") else []
    if not rows:
        return
    pick = table.iloc[rows[0]]["Symbol"]
    if st.session_state.get(f"_last_pick_{key}") != pick:
        st.session_state[f"_last_pick_{key}"] = pick
        st.session_state["_pending_ticker"] = pick
        st.rerun()


# =====================================================================================================================
# 9. Top of the page: title bar
# =====================================================================================================================
def render_top_bar(p):
    stale = p.freshness.loc[p.freshness["Status"].str.startswith("⚠️"), "File"].tolist()
    updated = datetime.fromtimestamp(p.mtime, tz=CT).strftime("%m/%d/%Y %I:%M %p CT")
    note = f" · ⚠️ {len(stale)} stale file(s), see Details" if stale else ""
    show_html(f"""
        <div class="sa-topbar">
          <div>
            <h1>Stock Analysis</h1>
            <p>Weekly top-{N_PICKS} ranking{" + Mon/Wed swap check" if MIDWEEK else ""} · technical + strength vs sector/SPY</p>
          </div>
          <div class="sa-chip{" sa-chip-warn" if stale else ""}">Updated {esc(updated + note)}</div>
        </div>""")


# =====================================================================================================================
# 10. Dashboard tab: single-stock view
# =====================================================================================================================
def ticker_label(p, s):
    if s in p.short.index:
        return f"{s}  ·  not traded yet (short history)  ·  rough {p.short.loc[s, 'Rough_Signal']} (less reliable)  ·  not ranked"
    r = p.by_symbol.loc[s]
    rank, score = r.get('Strategy_Rank'), r['combined_signal']
    parts = [s, f"{p.sig_off.get(s, 'Not ranked')} ({p.off_date:%b %-d})"]
    if pd.notna(rank):
        parts.append(f"rank #{rank:.0f}")
    parts.append(f"score {score:.0f}" if pd.notna(score) else "score —")
    return "  ·  ".join(parts)


def render_rank_tiers(p):
    """Horizontal clickable rank tiers: Rank 1 to 20, Rank 21 to 50, Rank 51+.

    Each symbol is a ?symbol= link, handled in main() exactly like a table click
    (opens the stock in the stock view below). The dropdown underneath stays for manual typing."""
    la = p.by_symbol
    ranked = la[la["Strategy_Rank"].notna()].sort_values("Strategy_Rank")
    tiers = [("Rank 1 to 20", ranked[ranked["Strategy_Rank"] <= 20]),
             ("Rank 21 to 50", ranked[(ranked["Strategy_Rank"] > 20) & (ranked["Strategy_Rank"] <= 50)]),
             ("Rank 51+", ranked[ranked["Strategy_Rank"] > 50])]
    current = st.session_state.get("ticker_dropdown")
    held = lambda s: TONE_CLASS.get(tone(p.sig_off.get(s)), "") if p.sig_off.get(s) in ("Buy", "Hold", "Sold") else ""
    rows = []
    for title, df in tiers:
        links = " ".join(symbol_link(s, held(s) + (" sel" if s == current else "")) for s in df.index)
        rows.append(f'<div class="sa-tier"><b>{title}:</b><div class="sa-tier-syms">{links or "\u2014"}</div></div>')
    if len(p.short):
        links = " ".join(f'{symbol_link(s)} <span style="color:{tone(r["Rough_Signal"])};">(rough {esc(str(r["Rough_Signal"]))}, '
                         'less reliable)</span>' for s, r in p.short.iterrows())
        rows.append(f'<div class="sa-tier"><b>Not traded yet (short history, not ranked):</b><div class="sa-tier-syms">{links}</div></div>')
    day = p.df["Date"].max()
    show_html(f'<div class="sa-card"><div class="sa-card-title">Stock list<span>ranks at the {day:%a %b %-d} close · '
              'click a stock to open it</span></div>' + "".join(rows)
              + f'<div class="sa-note">Color = the decision in force ({p.off_date:%b %-d}): <b style="color:{GOOD};">green Buy</b> · '
              f'<b style="color:{CAUTION};">yellow Hold</b> · <b style="color:{BAD};">red Sold</b> · grey = not in the portfolio. '
              'Teal outline = the stock shown below.</div></div>')


def render_stock_picker(p, jumped):
    """Rank tiers (click a symbol to open it) + dropdown for manual typing + AI button.

    ?symbol=X, a table click, a tier click or the dropdown choose the stock."""
    pending = st.session_state.pop("_pending_ticker", None)
    if pending in p.options:
        st.session_state.ticker_dropdown = pending
    elif st.session_state.get("ticker_dropdown") not in p.options:
        st.session_state.ticker_dropdown = p.options[0]
    st.markdown('<div id="ticker-focus"></div>', unsafe_allow_html=True)
    if jumped or pending:  # opened from a link or a table: scroll the stock view into sight
        components.html("<script>const el = window.parent.document.getElementById('ticker-focus');"
                        "if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});</script>", height=0)
    render_rank_tiers(p)
    pick_col, ai_col = st.columns([3, 1])
    ticker = pick_col.selectbox("Ticker", options=p.options, format_func=lambda s: ticker_label(p, s),
                                key="ticker_dropdown", label_visibility="collapsed")
    tdata = p.df[p.df['Symbol'] == ticker]
    if ticker not in p.short.index and ai_col.button("Generate AI Analysis", type="primary", width="stretch", key="generate_ai_btn"):
        ai_analysis_dialog(ticker, tdata, p.sig_off.get(ticker, 'Not ranked'), p.why_off.get(ticker, ''))
    return ticker, tdata


def render_stock_header(p, ticker, tdata):
    """Name, price, official signal and the numbers that decide it (score, rank, slot, weight)."""
    latest = tdata.nlargest(1, 'Date').iloc[0]
    # Display-only live quote: shown under the official bar close, never used by the strategy.
    live_html = ""
    live = live_quote(ticker)
    if live is not None:
        lp, lts = live
        lts = pd.Timestamp(lts)
        # Only label it LIVE when the quote is from today's session; otherwise the bar close above is already the latest.
        if lts.tzinfo is not None and lts.tz_convert(CT).date() == datetime.now(tz=CT).date():
            bar_close = num(latest['Close'])
            chg = (lp / bar_close - 1) * 100 if bar_close else None
            live_html = (
                f'<div class="sa-live" title="Live quote from yfinance (display only — the strategy uses the bar close above)">'
                f'<span class="sa-live-dot"></span>LIVE {fmt(lp, ",.2f", "$")} '
                f'<span style="color:{sign_color(chg)};">{fmt(chg, "+.2f", suffix="%")}</span>'
                f' <span class="sa-live-sub">vs {pd.Timestamp(latest["Date"]):%a} close · '
                f'{lts.tz_convert(CT).strftime("%I:%M %p")} CT</span></div>'
            )
    score, rank = num(latest.get('Strategy_Score')), num(latest.get('Strategy_Rank'))
    weight = num(latest.get('Strategy_Weight')) or 0.0
    n_ranked = int(p.latest['Strategy_Score'].notna().sum())
    slot = p.slot_off.get(ticker, "—")
    status, why = p.sig_off.get(ticker, "Not ranked"), p.why_off.get(ticker, "")
    chips = []
    if latest['final_trade'] == 'EARNING':
        chips.append(("Earnings within 2 sessions", "sa-badge-hold"))
    chips_html = "".join(f'<span class="sa-badge {c}" style="font-weight:600;font-size:0.72rem;">{esc(t)}</span>' for t, c in chips)
    next_ed = last_next_earnings([ticker])['Next ED'].iloc[0]
    day = pd.Timestamp(latest["Date"])
    rank_then = p.rank_off.get(ticker)
    badge = f"Strategy decision · {p.off_date:%a %b %-d}: {status}" + (f" ({p.tag_off[ticker]})" if p.tag_off.get(ticker) else "")
    plan_html = ""
    if ticker in symbol_sector or ticker in p.plan:                      # tradable stocks only (not the QQQ benchmark)
        action = p.plan.get(ticker, ("not picked",))[0]
        when = f"{p.plan_day:%a %b %-d}" if p.plan_day is not None else "Next rebalance"
        tip = (f"Full rebalance computed at the {p.plan_asof:%a %b %-d} close (strategy_picks.csv, the numbers the trade "
               f"step uses). Final at the {when} close; orders go out that day at 2:30 PM CT." if p.plan_asof is not None else "")
        plan_html = (f'<span class="sa-badge {PLAN_BADGE.get(action, "sa-badge-grey")}" title="{esc(tip)}">'
                     f'{esc(when + " plan: " + plan_text(p.plan, ticker))}</span>')
    stats = [
        stat_html("Strategy score", fmt(score, ".1f"), score_tone(score)),
        stat_html(f"Strategy rank · {day:%a %b %-d} close", f"#{rank:.0f} / {n_ranked}" if rank is not None else "—",
                  tip=f"Position by strategy score among the {n_ranked} ranked stocks at the {day:%a %b %-d} close (1 = best)"
                      + (f". At the {p.off_date:%a %b %-d} decision it was #{rank_then:.0f}." if rank_then is not None
                         and pd.notna(rank_then) else "")),
        stat_html(f"Portfolio slot · {p.off_date:%b %-d}", f"{slot} of 10" if slot != "—" else "— (not picked)",
                  tip="Position among the 10 stocks picked at the last decision, in rank order. The picks skip stocks "
                      f"whose sector already has {SECTOR_MAX}, so the slot can be smaller than the rank."),
        stat_html("Portfolio weight", fmt(weight * 100 if weight > 0 else None, ".1f", suffix="%")),
        stat_html("Technical", fmt(num(latest.get('Technical_Score')), ".1f"), score_tone(num(latest.get('Technical_Score')))),
        stat_html("Strength vs sector/SPY", fmt(num(latest.get('RS_Score')), ".1f"), score_tone(num(latest.get('RS_Score')))),
        stat_html("Next earnings", pd.Timestamp(next_ed).strftime("%b %d") if next_ed else "—"),
    ]
    hold = row_for(HOLDINGS_CSV, ticker) if weight > 0 else None
    if hold is not None:
        stats += [
            stat_html("Held since", f"{pd.Timestamp(hold['Entry_Date']):%b %-d} · {int(hold['Days_Held'])} sessions"
                      if pd.notna(hold['Entry_Date']) else "—"),
            stat_html("P&L since entry", fmt(num(hold['PnL_%']), "+.1f", suffix="%"), sign_color(num(hold['PnL_%']))),
        ]
    show_html(f"""
        <div class="sa-hero">
          <div class="sa-ident">
            <div class="sa-sym">{esc(ticker)}</div>
            <div class="sa-price" title="Latest completed daily bar — the strategy's official price">{fmt(num(latest['Close']), ",.2f", "$")}</div>
            {live_html}
            <span class="sa-badge {SIGNAL_BADGE.get(status, 'sa-badge-grey')}" title="The strategy's decision in force (last decision)">{esc(badge)}</span>
            {plan_html}
            {chips_html}
          </div>
          <div class="sa-why">{esc(status + ": " + why) if why else ""}</div>
          <div class="sa-stats">{"".join(stats)}</div>
          <div class="sa-note">Scores: <b style="color:{GOOD};">green above {SCORE_GOOD}</b> · <b style="color:{CAUTION};">yellow 0 to {SCORE_GOOD}</b> ·
            <b style="color:{BAD};">red below 0</b> (only stocks above 0 can be picked). Rank 1 = best. P&amp;L and returns: green up, red down.</div>
        </div>""")


def build_price_chart(ticker, tdata, show_strategy, show_rs, show_classic, show_legacy):
    """Last 12 months: price + moving averages + buy/sell markers, optional score/rank, relative strength, RSI/MACD panels."""
    chart = tdata[tdata['Date'] >= tdata['Date'].max() - pd.Timedelta(days=365)].sort_values('Date').set_index('Date')
    x_start, x_end = chart.index[0], chart.index[-1]
    events, periods = strategy_events(tdata, load_decisions(), ticker)
    rs_lines = relative_strength_lines(chart, ticker) if show_rs else {}

    panels = ["price"] + (["score"] if show_strategy else []) + (["rs"] if rs_lines else []) \
        + (["rsi", "macd"] if show_classic else [])
    height_of = {"price": 0.44, "score": 0.20, "rs": 0.18, "rsi": 0.11, "macd": 0.11}
    titles = {
        "price": "<b>Price</b> · ▲ Buy / ▼ Sold decisions · yellow shading = held",
        "score": f"<b>Strategy score</b> (teal) = {W_TECH:g} × Technical (grey) + {1 - W_TECH:g} × Strength vs sector/SPY (violet)",
        "rs": ("<b>Performance since " + f"{x_start:%b %-d, %Y}" + "</b> (% price change): " + " · ".join(
            f'<span style="color:{RS_COLORS[k]};">━ {name}</span>' for k, (name, _s) in rs_lines.items())),
        "rsi": "<b>RSI</b> (above 70 = stretched up, below 30 = stretched down)", "macd": "<b>MACD</b> (indigo) vs signal (dashed)",
    }
    heights = [height_of[x] for x in panels]
    row_of = {x: i + 1 for i, x in enumerate(panels)}
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[h / sum(heights) for h in heights], subplot_titles=[titles[x] for x in panels])
    fig.update_annotations(font=dict(size=12, color='#334155', family=CHART_FONT), yshift=4)

    # Price line (hover shows the day's status), earnings dates, moving averages
    status_txt = []
    for w, r, s in zip(chart['Strategy_Weight'], chart['Strategy_Rank'], chart['Strategy_Score']):
        sig = daily_status(w, s)
        bits = [sig] + ([f"weight {w * 100:.1f}%"] if sig == "Hold" else []) \
            + ([f"rank #{r:.0f}"] if pd.notna(r) else []) + ([f"score {s:.1f}"] if pd.notna(s) else [])
        status_txt.append(" · ".join(bits))
    fig.add_trace(go.Scatter(x=chart.index, y=chart['Close'], name='Close', line=dict(color=TEAL, width=2.5), mode='lines',
                             customdata=status_txt, hovertemplate='<b>Close</b> $%{y:.2f}<br>%{customdata}<extra></extra>'), row=1, col=1)
    earnings = chart[chart['is_earnings_date'] == 1]
    fig.add_trace(go.Scatter(x=earnings.index, y=earnings['Close'], name='Earnings date', mode='markers',
                             marker=dict(symbol='circle', size=9, color='#f97316'),
                             hovertemplate='<b>Earnings</b> %{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>'), row=1, col=1)
    for d in earnings.index:
        fig.add_vline(x=d, line=EARN_LINE, row=1, col=1)
    fig.add_trace(go.Scatter(x=[x_start], y=[None], mode="lines", name="Earnings (dotted line; next one ahead)",
                             line=EARN_LINE, hoverinfo="skip"), row=1, col=1)
    for ma, color in zip(MA_COLS, MA_COLORS):
        name = ma.upper().replace('_', ' ')
        fig.add_trace(go.Scatter(x=chart.index, y=chart[ma], name=name, line=dict(color=color, width=1), mode='lines',
                                 hovertemplate=f'<b>{name}</b> $%{{y:.2f}}<extra></extra>'), row=1, col=1)
    if periods:
        fig.add_trace(go.Scatter(x=[x_start], y=[None], mode="markers", name="Hold (shaded period)", hoverinfo="skip",
                                 marker=dict(symbol="square", size=12, color="rgba(245,158,11,0.35)")), row=1, col=1)

    # Entry / exit markers on the session after the decision (hollow = decided at the latest close, orders pending)
    shown = events[events['Fill'].fillna(x_end) >= x_start] if len(events) else events
    for kind, marker, color, name in (("entry", "triangle-up", GOOD, "Buy (strategy decision)"),
                                      ("exit", "triangle-down", BAD, "Sold (strategy decision)")):
        e = shown[shown['Kind'] == kind] if len(shown) else shown
        if e.empty:
            continue
        pending = e['Fill'].isna()
        fig.add_trace(go.Scatter(
            x=e['Fill'].fillna(x_end), y=e['Price'], mode='markers', name=name,
            marker=dict(symbol=[marker + ("-open" if x else "") for x in pending], size=13, color=color,
                        line=dict(width=1.5, color=color if pending.any() else "#ffffff")),
            hovertext=[event_hover(x) for x in e.itertuples()], hovertemplate="%{hovertext}<extra></extra>"), row=1, col=1)

    # Market filter OFF weeks (amber diamonds above the price)
    span = chart['Close'].max() - chart['Close'].min()
    top_y = chart['Close'].max() + span * 0.06
    regime_off = chart[(chart['Rebalance_Day'] == 1) & (chart['Regime_On'] == 0)]
    if len(regime_off):
        fig.add_trace(go.Scatter(
            x=regime_off.index, y=[top_y] * len(regime_off), mode='markers', name=f'Market filter OFF (positions {HALVED})',
            marker=dict(symbol='diamond', size=8, color=CAUTION),
            hovertemplate=f'<b>Market filter OFF</b> %{{x|%b %d}}: {REGIME_OFF},<br>all positions {HALVED} that week<extra></extra>'),
            row=1, col=1)

    # While held: entry price dotted line
    hold = row_for(HOLDINGS_CSV, ticker) if (num(chart['Strategy_Weight'].iloc[-1]) or 0) > 0 else None
    if hold is not None and pd.notna(hold['Entry_Price']):
        fig.add_hline(y=float(hold['Entry_Price']), line=dict(color='#64748b', width=1, dash='dot'), row=1, col=1,
                      annotation_text=f"entry ${hold['Entry_Price']:,.2f}", annotation_position="top left",
                      annotation_font=dict(size=10, color='#64748b'))

    # Next earnings date (extends the x-axis when it is within ~2 months)
    x_right = x_end
    ned = last_next_earnings([ticker])['Next ED'].iloc[0]
    if ned and pd.Timestamp(ned) - x_end <= pd.Timedelta(days=62):
        ned = pd.Timestamp(ned)
        x_right = max(x_end, ned) + pd.Timedelta(days=4)
        fig.add_vline(x=ned, line=EARN_LINE | dict(width=1.5), row="all", col=1)
        fig.add_annotation(x=ned, y=1, xref="x", yref="y domain", text=f"next earnings {ned:%b %-d}", showarrow=False,
                           yanchor="bottom", xanchor="right", font=dict(size=10, color='#c2410c'))

    # Live price marker (display only): a single dot at today's date so the price panel reaches the current
    # session. Bars, moving averages, and every strategy panel still end at the last completed bar.
    live = live_quote(ticker)
    if live is not None:
        lp, lts = live
        lts = pd.Timestamp(lts)
        if lts.tzinfo is not None and lts.tz_convert(CT).date() == datetime.now(tz=CT).date():
            live_day = lts.tz_convert(CT).normalize().tz_localize(None)
            fig.add_trace(go.Scatter(
                x=[live_day], y=[lp], name="Live", mode="markers",
                marker=dict(symbol="circle", size=8, color=TEAL, line=dict(width=2, color="#ffffff")),
                hovertemplate="<b>Live</b> $%{y:.2f}<br>%{x|%b %d, %Y} · intraday, display only<extra></extra>"),
                row=1, col=1)
            x_right = max(x_right, live_day) + pd.Timedelta(days=2)

    # Old rules (off by default): streak bars + BUY/SELL flip lines from final_trade
    if show_legacy:
        legacy_top = top_y + span * 0.04
        buy_on, sell_on = chart['Buy Streak'] > 0, chart['Sell Streak'] > 0
        for mask, color in ((buy_on, "#2ca02c"), (sell_on, "#d62728"), (~buy_on & ~sell_on, "#FFD700")):
            for start, end in legacy_periods(mask):
                fig.add_shape(type="line", x0=start, x1=end, y0=legacy_top, y1=legacy_top, line=dict(color=color, width=3),
                              opacity=0.6, row=1, col=1)
        flips = legacy_flips(tdata)
        for day, trade in flips.loc[flips['Date'] >= x_start, ['Date', 'final_trade']].itertuples(index=False):
            fig.add_vline(x=day, line=dict(color={'BUY': "#2ca02c", 'SELL': "#d62728"}[trade], width=1, dash="dot"),
                          opacity=0.45, row=1, col=1)

    if "score" in row_of:  # score panel
        r = row_of["score"]
        for col, name, color, width, dash in (("Technical_Score", "Technical", "#9ca3af", 1.25, "dot"),
                                              ("RS_Score", "Strength vs sector/SPY", "#8b5cf6", 1.5, "solid"),
                                              ("Strategy_Score", "Strategy score", TEAL, 2, "solid")):
            fig.add_trace(go.Scatter(x=chart.index, y=chart[col], name=name, mode='lines', line=dict(color=color, width=width, dash=dash),
                                     showlegend=False, hovertemplate=f'<b>{name}</b> %{{y:.1f}}<extra></extra>'), row=r, col=1)
        fig.add_hline(y=0, line=dict(color='rgba(100,116,139,0.5)', width=1, dash='dot'), row=r, col=1)
    if "rs" in row_of:
        r = row_of["rs"]
        for key, (label, series) in rs_lines.items():
            fig.add_trace(go.Scatter(x=series.index, y=series, name=label, mode='lines',
                                     line=dict(color=RS_COLORS[key], width=2.25 if key == "stock" else 1.5),
                                     showlegend=False, hovertemplate=f'<b>{label}</b> %{{y:+.1f}}%<extra></extra>'), row=r, col=1)
        fig.add_hline(y=0, line=dict(color='rgba(100,116,139,0.5)', width=1, dash='dot'), row=r, col=1)
    if "rsi" in row_of:
        r = row_of["rsi"]
        fig.add_trace(go.Scatter(x=chart.index, y=chart['RSI'], name='RSI', line=dict(color='#0ea5e9', width=1.5), mode='lines',
                                 showlegend=False, hovertemplate='<b>RSI</b> %{y:.1f}<extra></extra>'), row=r, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(185,28,28,0.35)", row=r, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(21,128,61,0.35)", row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)
        r = row_of["macd"]
        fig.add_trace(go.Scatter(x=chart.index, y=chart['macd'], name='MACD', line=dict(color='#4f46e5', width=1.5), mode='lines',
                                 showlegend=False, hovertemplate='<b>MACD</b> %{y:.3f}<extra></extra>'), row=r, col=1)
        fig.add_trace(go.Scatter(x=chart.index, y=chart['MACD Signal'], name='MACD signal', line=dict(color='#94a3b8', width=1.5, dash='dash'),
                                 mode='lines', showlegend=False, hovertemplate='<b>Signal</b> %{y:.3f}<extra></extra>'), row=r, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(128, 128, 128, 0.4)", row=r, col=1)

    # Held periods shaded on all panels (added last, with exclude_empty_subplots=False, or plotly drops them)
    for start, end in periods:
        if end >= x_start:
            fig.add_vrect(x0=max(start, x_start), x1=end, fillcolor=HOLD_SHADE, line_width=0, layer="below",
                          row="all", col=1, exclude_empty_subplots=False)

    grid = dict(showgrid=True, gridcolor='#eef2f7', showline=True, linecolor='#e2e8f0', tickfont=dict(size=10, color='#64748b'),
                zeroline=False)
    fig.update_layout(
        height=int(560 + 170 * (len(panels) - 1)), hovermode='x unified', margin=dict(l=50, r=30, t=90, b=64),
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', dragmode=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, font=dict(size=10, color='#334155'),
                    title=dict(text="<b>Legend</b>  ", font=dict(size=10, color='#64748b')),
                    bgcolor='#f8fafc', bordercolor='#e2e8f0', borderwidth=1),
        font=dict(family=CHART_FONT, size=11, color='#334155'),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", font_size=11, font_family=CHART_FONT))
    fig.update_xaxes(type="date", range=[x_start, x_right], showspikes=True, spikemode="across", spikethickness=1, spikecolor="#94a3b8",
                     dtick=7 * 24 * 60 * 60 * 1000, tickformat='%b %d', tickangle=-45,
                     **(grid | dict(showgrid=False)))
    fig.update_yaxes(**grid)
    fig.update_yaxes(title_text="Price ($)", tickformat='$,.0f', row=1, col=1)
    return fig


def stock_chart_inputs(ticker, tdata):
    """Chart-option checkboxes + the built figure (displayed later, below the detail block)."""
    has_strategy = bool(tdata['Strategy_Score'].notna().any())
    show_html('<div class="sa-group-title" style="margin:0.4rem 0 0.1rem;">Chart panels</div>')
    o = st.columns(4)
    show_strategy = o[0].checkbox("Strategy score", value=True, key="show_strategy", disabled=not has_strategy)
    show_rs = o[1].checkbox("Relative strength", value=True, key="show_rs")
    show_classic = o[2].checkbox("RSI & MACD", value=True, key="show_classic")
    show_legacy = o[3].checkbox("Legacy signals (old rules)", value=False, key="show_legacy",
                                help="The old BUY/SELL streak bars and flip lines from final_trade (pre-v3 rules). Not the live strategy.")
    fig = build_price_chart(ticker, tdata, show_strategy and has_strategy, show_rs, show_classic, show_legacy)
    return fig, has_strategy


def render_stock_figure(fig, ticker):
    """The price chart itself, under the always-open detail block."""
    names = performance_names(ticker)
    st.plotly_chart(fig, width="stretch", config={
        'displaylogo': False, 'scrollZoom': False, 'doubleClick': 'reset',
        'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'zoomIn2d', 'zoomOut2d']})
    vs = " and ".join(f"<b>{esc(name)}</b> ({color})" for name, color in ((names.get("market"), "grey"),
                                                                          (names.get("sector"), "orange")) if name)
    show_html(f'<div class="sa-note"><b>How to read the chart</b> · <b>Price</b>: <b style="color:{GOOD};">▲ Buy</b> / '
              f'<b style="color:{BAD};">▼ Sold</b> = the strategy\'s decisions (hollow = orders pending), yellow shading = held, '
              'orange dots and dotted orange lines = earnings dates (the line past the last bar = the next one). · <b>Strategy score</b>: above 0 and rising is good (a positive trend and stronger than '
              'its peers); below 0 and falling is bad. · <b>Performance</b>: % price change since the chart start: '
              f'<b>{esc(ticker)}</b> (blue) vs {vs}. {esc(ticker)} above the others = it has beaten them; a widening gap = it is '
              'getting stronger than the market / its sector.</div>')


def _detail_group(title, stats, note=""):
    """One titled row of stat tiles inside the stock detail card; stats = [(label, value, color)]."""
    return (f'<div class="sa-group"><div class="sa-group-title">{esc(title)}</div><div class="sa-stats" style="margin-top:0.5rem;">'
            + "".join(stat_html(l, "\u2014" if v is None else v, c) for l, v, c in stats) + "</div>"
            + (f'<div class="sa-why">{esc(note)}</div>' if note else "") + "</div>")


def render_stock_more(ticker, tdata, has_strategy):
    """Always-open detail card above the chart: moving averages, fundamentals/news, per-stock forward test."""
    latest = tdata.nlargest(1, 'Date').iloc[0]
    close = num(latest['Close'])
    ma_stats = [(ma.upper().replace('_', ' '), fmt(num(latest[ma]), ",.2f", "$"),     # green = the close is above it
                 tone(close - num(latest[ma])) if close and num(latest[ma]) else INK) for ma in MA_COLS]

    # Context: latest-day values only, NOT part of the backtested rules
    company_df = load_company()
    comp = company_metrics(ticker, company_df)
    fv = comp.get('fair_value')
    upside = (fv / close - 1) * 100 if fv and close else None
    sentiment = num(latest['SentimentScore'])
    comp_row = company_df[company_df['Symbol'].astype(str).str.upper() == ticker] if 'Symbol' in company_df else pd.DataFrame()
    fiscal = pd.to_datetime(comp_row['FiscalDateEnding'].iloc[0], errors='coerce') \
        if len(comp_row) and 'FiscalDateEnding' in comp_row else pd.NaT
    news = load_news()
    news_dates = pd.to_datetime(news.loc[news['symbol'] == ticker, 'date'], utc=True, format='mixed', errors='coerce')
    last_news = news_dates.max() if len(news_dates) else pd.NaT
    fund_stats = [
        ("Balance sheet", fmt(num(latest['Fundamental_Weight']), ".2f"), tone(num(latest['Fundamental_Weight']))),
        ("Sentiment", fmt(sentiment, ".2f"), tone(sentiment)),
        ("Fair value", fmt(fv, ",.2f", "$"), INK),
        ("Upside", fmt(upside, "+.1f", suffix="%"), tone(upside)),
        ("P/E", fmt(comp.get('pe_ratio'), ".1f"), INK),
        ("P/B", fmt(comp.get('pb_ratio'), ".2f"), INK),
        ("Rev YoY", fmt(comp.get('revenue_growth_yoy'), ".1f", suffix="%"), tone(comp.get('revenue_growth_yoy'))),
        ("ROE", fmt(comp.get('roe'), ".1f", suffix="%"), tone(comp.get('roe'))),
        ("Net margin", fmt(comp.get('net_margin'), ".1f", suffix="%"), tone(comp.get('net_margin'))),
        ("Debt/Eq", fmt(comp.get('debt_to_equity'), ".2f"), INK),
    ]
    fund_note = ("Context only (latest day, not part of the backtested rules) \u00b7 fundamentals: "
                 + (f"quarter ending {fiscal:%Y-%m-%d}" if pd.notna(fiscal) else "none on file")
                 + (f" \u00b7 newest relevant news {last_news:%b %-d}" if pd.notna(last_news)
                    else " \u00b7 no relevant news in the last 10 days"))

    fw_stats, fw_note = [], ""
    if has_strategy:            # per-stock forward test: only sessions from FORWARD_START on count
        from backtest_engine import FORWARD_START, forward_test
        s_ = tdata.sort_values('Date').set_index('Date')
        fw = forward_test(s_['Close'], s_['Strategy_Weight'], FORWARD_START)
        start_txt, since = f"{fw['Start']:%b %-d, %Y}", f"since {fw['Start']:%b %-d}"
        bench = load_benchmarks()
        q = bench["QQQ"].loc[fw["Start"]:].dropna() if bench is not None and "QQQ" in bench else pd.Series(dtype=float)
        qqq = (q.iloc[-1] / q.iloc[0] - 1) * 100 if len(q) else None
        n, ot, med, bnh = fw["Closed trades"], fw["Open trade"], fw["Median trade %"], fw["Buy & hold %"]
        fw_stats = [("Start", start_txt, INK), ("Closed trades", f"{n}", INK),
                    ("Open trade", f"{ot['Entry']:%b %-d} @ ${ot['Price']:,.2f} · {ot['Change %']:+.2f}%" if ot else "none",
                     sign_color(ot["Change %"]) if ot else INK)]
        if n:
            fw_stats += [("Win rate", fmt(fw["Win rate %"], ".0f", suffix="%"), tone(fw["Win rate %"], 50, 50)),
                         ("Median trade", fmt(med, "+.2f", suffix="%"), sign_color(med)),
                         ("Median hold", f"{fw['Median hold (sessions)']:.0f} sessions", INK)]
        fw_stats += [("Held % of sessions", fmt(fw["Held % of sessions"], ".0f", suffix="%"), INK),
                     (f"Buy & hold {since}", fmt(bnh, "+.2f", suffix="%"), sign_color(bnh)),
                     (f"QQQ {since}", fmt(qqq, "+.2f", suffix="%"), sign_color(qqq))]
        fw_note = (("" if n else f"Forward test started {start_txt}; no closed trades yet. ")
                   + f"Live rules from the {start_txt} close on: only sessions from then count (earlier history only "
                   "warms up the indicators); trades at the decision-day close, 0.1%/side. QQQ is not traded, "
                   "comparison only.")

    groups = _detail_group("Moving averages (green = the close is above it)", ma_stats)
    groups += _detail_group("Fundamentals & news", fund_stats, fund_note)
    if fw_stats:
        groups += _detail_group("Per-stock forward test", fw_stats, fw_note)
    show_html(f'<div class="sa-card"><div class="sa-card-title">More about {esc(ticker)}</div>' + groups + '</div>')


# =====================================================================================================================
# 11. Details tab
# =====================================================================================================================
def latest_signals_title(p):
    return f"Latest signals · {p.df['Date'].max():%a %b %-d} close"


def render_latest_signals(p):
    """Every stock at the LATEST close (signal_analysis.csv) + the next-rebalance plan (strategy_picks.csv)."""
    day = p.df["Date"].max()
    la = p.latest[p.latest["Date"] == day].set_index("Symbol")
    plan_when = f"{p.plan_day:%a %b %-d}" if p.plan_day is not None else "next"
    decided_today = pd.Timestamp(p.off_date) == pd.Timestamp(day)
    rows = []
    for sym, r in la.iterrows():
        w = num(r.get("Strategy_Weight")) or 0.0
        rows.append({"Symbol": sym,
                     "Signal today": p.sig_off.get(sym, "Not ranked") if decided_today else daily_status(w, r.get("Strategy_Score")),
                     "Rank today": r.get("Strategy_Rank"), "Score today": r.get("Strategy_Score"),
                     f"{plan_when} plan": plan_text(p.plan, sym) if sym in symbol_sector or sym in p.plan else "—",
                     "Held now %": w * 100 if w > 0 else np.nan, "Sector": symbol_sector.get(sym, "—")})
    table = pd.DataFrame(rows).sort_values(["Rank today", "Symbol"], na_position="last").reset_index(drop=True)
    table["Rank today"] = table["Rank today"].round(0).astype("Int64")
    st.caption(f"Every stock at the {day:%a %b %-d} close (Reports/signal_analysis.csv) · Signal today = "
               + ("the decision made at this close" if decided_today else
                  "no decision today: Hold = in the portfolio, Watch = ranked but not held, Score below 0 = not eligible")
               + f" · {plan_when} plan = what the {plan_when} full rebalance would do at this close (Reports/strategy_picks.csv, "
               "the numbers the trade step uses; final at that close). Click a row to open the stock.")
    table = table.round({"Score today": 1, "Held now %": 2})
    event = st.dataframe(toned(toned(table, ["Signal today", f"{plan_when} plan"]), ["Score today"], good=SCORE_GOOD),
                         hide_index=True, width="stretch",
                         on_select="rerun", selection_mode="single-row", key="latest_signals_tbl",
                         column_config={"Score today": SCORE_COL, "Held now %": PCT_COL})
    open_symbol(table, event, "latest_signals")


def load_short_history():
    """Reports/short_history_reference.csv (stocks in the list with < 200 days of prices); None when missing or empty."""
    d = read_report_csv(SHORT_HISTORY_CSV)
    return None if d is None or d.empty else d


@st.cache_data(ttl=3600)
def short_history_closes(symbol, start):
    """Display-only daily closes from yfinance for a stock too new to trade; None when unavailable."""
    if not _YF_OK:
        return None
    try:
        df = _yf.download(symbol, start=start, interval="1d", progress=False, auto_adjust=True)
        closes = df["Close"]
        closes = (closes.iloc[:, 0] if isinstance(closes, pd.DataFrame) else closes).dropna()
        return closes if len(closes) else None
    except Exception:
        return None


def render_short_stock(sym, r):
    """Stock view for a stock in the list with fewer than 200 trading days: never scored, ranked or traded.
    Numbers come from Reports/short_history_reference.csv; the chart is display-only (yfinance)."""
    day = lambda d: f"{pd.Timestamp(d):%a %b %-d, %Y}"
    section(f"{sym} · {r['Name']}")
    st.warning(f"Not traded yet (short history): {int(r['Days_Of_History'])} of {int(r['Days_Needed'])} trading days, "
               f"first traded {day(r['First_Trade'])}. The strategy does not score, rank or buy it until its 200th trading day "
               f"(about {day(r['Est_Eligible_Date'])}, est.).")
    c = st.columns(6)
    c[0].metric(f"Close · {pd.Timestamp(r['Last_Date']):%a %b %-d}", f"${r['Last_Close']:,.2f}")
    c[1].metric("Rough signal (less reliable)", r["Rough_Signal"])
    c[2].metric("RSI 14", f"{r['RSI_14']:.1f}")
    for i, n in enumerate((10, 30, 50)):
        c[3 + i].metric(f"MA {n}", f"${r[f'MA_{n}']:,.2f}")
    ret = lambda v: md_tone(f"{v:+.1f}%", tone(v))
    st.caption(f"Since first close {ret(r['Return_Since_First_Close_%'])} · 21-day {ret(r['Return_21d_%'])} · "
               f"63-day {ret(r['Return_63d_%'])} · {abs(r['Off_High_%']):.1f}% below its high of ${r['High_Since_First']:,.2f}. "
               "Rough signal: Buy if the close is above the 10-day, the 10-day above the 30-day and the 30-day above the 50-day "
               "average with RSI 50-70; Sell if the close is below both the 30- and 50-day averages or RSI is under 40; otherwise "
               "Hold. Less reliable than the real signal (no 200-day history) and never used for trading.")
    closes = short_history_closes(sym, str(r["First_Trade"]))
    if closes is None:
        st.caption("Price chart unavailable right now (yfinance).")
        return
    fig = go.Figure(go.Scatter(x=closes.index, y=closes, name="Close", line=dict(color=TEAL, width=2.5)))
    for n, color in zip((10, 30, 50), MA_COLORS):
        fig.add_trace(go.Scatter(x=closes.index, y=closes.rolling(n).mean(), name=f"MA {n}", line=dict(color=color, width=1)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(type="date"),
                      title=f"{sym} daily close (display only, yfinance)", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                      font=dict(family=CHART_FONT, size=11, color="#334155"), yaxis=dict(tickformat="$,.0f", gridcolor="#eef2f7"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="#f8fafc",
                                  bordercolor="#e2e8f0", borderwidth=1, title=dict(text="<b>Legend</b>  ")))
    st.plotly_chart(fig, width="stretch", key=f"short_chart_{sym}")


def render_last_decision(p):
    """The decisions in force, in ONE view: decision dates, then every stock with its signal, reason, ranks, weight before
    and after, and the next rebalance plan (filters: portfolio & changes / watch list / all). Click a row to open the stock."""
    reb_days = p.df.loc[p.df["Rebalance_Day"] == 1, "Date"]
    mw = p.midweek[p.midweek["Event"] == "mid-week check"] if p.midweek is not None else None
    if mw is not None and len(mw):
        last_day = mw["Event_Date"].iloc[-1]
        acts = set(mw.loc[mw["Event_Date"] == last_day, "Action"])
        mw_val = (f"{pd.Timestamp(last_day):%a %b %-d} · "
                  + (" + ".join(x for x, k in (("swap", "SWAP"), ("exit", "SELL")) if k in acts) or "no trade"))
    else:
        mw_val = "none yet this week" if MIDWEEK else "off"
    c = st.columns(3)
    c[0].metric("Last weekly rebalance", f"{reb_days.max():%a %b %-d}" if not reb_days.empty else "—")
    c[1].metric("Last mid-week check", mw_val)
    c[2].metric("Next decision", f"{p.next_kind} · {p.next_dec:%a %b %-d}" if p.next_dec is not None else "—")
    today = p.df["Date"].max()
    plan_when = f"{p.plan_day:%a %b %-d}" if p.plan_day is not None else "next"
    rank_then = f"Rank at {p.off_date:%b %-d} decision"
    board = p.board_off.rename(columns={"Rank": rank_then})
    board[rank_then] = board[rank_then].round(0).astype("Int64")
    board.insert(4, "Rank today", board["Symbol"].map(p.by_symbol["Strategy_Rank"]).round(0).astype("Int64"))
    board.insert(5, "Next rebalance plan", [plan_text(p.plan, s) if s in symbol_sector or s in p.plan else "—"
                                            for s in board["Symbol"]])
    board.insert(6, "Rank change", board["Symbol"].map(lambda s: p.rank_change.get(s, (np.nan,))[0]))
    ch = read_report_csv(CHANGES_CSV)
    ok = ch is not None and {"Symbol", "Old_Weight"} <= set(ch.columns)
    # Newest decision first so drop_duplicates keeps the latest Old_Weight per symbol.
    _ch = ch.sort_values("Date", ascending=False) if ok else None
    before = _ch[_ch["Symbol"].notna()].drop_duplicates("Symbol").set_index("Symbol")["Old_Weight"] if ok else {}
    board.insert(board.columns.get_loc("Portfolio weight %"), "Weight before %",
                 board["Symbol"].map(lambda s: before.get(s, np.nan) * 100))
    board.loc[board["Signal"] == "Sold", "Portfolio weight %"] = 0.0      # after the decision a sold stock weighs 0
    counts = board["Signal"].value_counts()
    filters = {"Portfolio & changes": ["Buy", "Hold", "Sold"],
               "Watch list": ["Watch", "Watch (sector limit)"],
               "All stocks": SIGNALS}
    show = st.radio("Show", list(filters), horizontal=True, key="signals_filter", label_visibility="collapsed")
    part = board[board["Signal"].isin(filters[show])].reset_index(drop=True)
    st.caption(f"Buy {counts.get('Buy', 0)} · Hold {counts.get('Hold', 0)} · "
               f"Sold {counts.get('Sold', 0)} · Watch {counts.get('Watch', 0) + counts.get('Watch (sector limit)', 0)} · "
               f"Score below 0 {counts.get('Score below 0', 0)}. {rank_then} = the rank the decision used; Rank today = at "
               f"the {today:%a %b %-d} close (Rank change: + = moved up since the decision day); Weight before % → Portfolio "
               f"weight % = the portfolio before and after the decision; Portfolio slot = position among the {N_PICKS} picks; "
               f"Next rebalance plan = what the {plan_when} rebalance would do at the latest close (the numbers the trade "
               "step uses). Orders for a decision go out that day at 2:30 PM CT. Click a row to open the stock on the "
               "Dashboard tab.")
    shown = toned(part.round({"Score": 1, "Weight before %": 2, "Portfolio weight %": 2}),
                  ["Signal", "Next rebalance plan", "Rank change", "Earnings soon"])
    event = st.dataframe(toned(shown, ["Score"], good=SCORE_GOOD), hide_index=True,
                         width="stretch", on_select="rerun", selection_mode="single-row", key=f"sig_tbl_{show}",
                         column_config={"Why": st.column_config.TextColumn("Why", width="large"), "Score": SCORE_COL,
                                        "Rank change": st.column_config.NumberColumn(format="%+.0f"),
                                        "Weight before %": PCT_COL, "Portfolio weight %": PCT_COL})
    open_symbol(part, event, "signals")


def render_data_and_settings(p):
    """Data freshness table."""
    st.dataframe(toned(p.freshness, ["Status"]), width="stretch", hide_index=True)
    stale = p.freshness[p.freshness["Status"].str.startswith("⚠️")]
    if not stale.empty:
        st.warning("Stale or missing: " + ", ".join(stale["File"]) + " — run `python run_all.py`.")
    else:
        st.success("All report files are within their expected refresh window.")


@st.cache_resource(show_spinner=False)
def _fill_history():
    """The fill history kept for the whole dashboard process (alpaca_paper.FillHistory: only new fills are read)."""
    import alpaca_paper as ap
    return ap.FillHistory()


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def _read_holdings(refresh_key):
    """One read of the Alpaca LIVE account (GET only: positions, account, fills) and the QQQ quote per refresh_key
    (alpaca_paper.holdings_refresh_key: a new key each minute in market hours, each hour otherwise). A failure raises,
    and Streamlit never caches a raise, so the next minute tries again."""
    import alpaca_paper as ap
    acct = ap.PaperAccount()
    q = live_quote("QQQ")
    return {"positions": acct.position_dicts(), "fills": _fill_history().update(acct), "equity": acct.account_summary()["Equity"],
            "qqq_now": q[0] if q else None, "as_of": datetime.now(CT)}


def live_holdings():
    """(data, None) or (None, plain message). Keys come from .env via alpaca_paper; error messages never hold them."""
    if os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off":                # tests: never call Alpaca
        return None, "Live holdings are turned off here (STOCK_ANALYSIS_LIVE_HOLDINGS=off)."
    try:
        import alpaca_paper as ap
        return _read_holdings(ap.holdings_refresh_key()), None
    except Exception as e:
        return None, f"Live holdings unavailable right now ({type(e).__name__}: {str(e)[:200]}). It tries again in a minute."


@st.fragment(run_every=60)        # reruns only this table each minute; it reads Alpaca only when the refresh key changes
def render_live_holdings():
    """Details tab: the real Alpaca positions with cost, value, P/L and the first purchase date, plus QQQ for comparison."""
    data, err = live_holdings()
    if err:
        st.info(err)
        return
    import alpaca_paper as ap
    last = load_benchmarks()                                     # no live QQQ quote: the last daily close
    last = last["QQQ"].dropna() if last is not None and "QQQ" in last else ()
    table = ap.holdings_table(data["positions"], data["fills"], data["equity"],
                              data["qqq_now"] or (float(last.iloc[-1]) if len(last) else None))
    if table.empty:
        st.info(f"No open positions in the Alpaca account (as of {data['as_of']:%a %b %-d %I:%M %p} CT).")
        return
    table["First bought"] = [f"{d:%a %b %-d, %Y}" if d is not None and d == d else "" for d in table["First bought"]]
    money, pct = st.column_config.NumberColumn(format="dollar"), st.column_config.NumberColumn(format="%+.2f%%")
    st.dataframe(toned(table, ["P/L $", "P/L %", "Today %"]), hide_index=True, width="stretch",
                 height=35 * (len(table) + 1) + 3,                  # every row visible, no inner scroll
                 column_config={"Shares": st.column_config.NumberColumn(format="%.2f"), "Avg price": money,
                                "Cost basis": money, "Market value": money, "P/L $": money, "Price": money,
                                "P/L %": pct, "Today %": pct, "Weight %": st.column_config.NumberColumn(format="%.2f%%")})
    st.caption(f"As of {data['as_of']:%a %b %-d %I:%M:%S %p} CT, read from Alpaca (read-only). Updates by itself: every "
               "minute in market hours (8:30 AM-3:00 PM CT on trading days), every hour otherwise. "
               "Cost basis = what you paid; Market value = shares x the latest price; P/L \\$ and P/L % = market value vs "
               "cost basis; Today % = price change since the last close; Weight % = share of the account's equity (the rest "
               "is cash). First bought = the earliest buy still in the position (sells use up the oldest shares first). "
               f"QQQ is not held: its row invests the same total cost basis in QQQ at its Fri Oct 2, 2026 close "
               f"(\\${ap.QQQ_BASE_CLOSE:,.2f}, fixed) and values it at QQQ's latest price, to compare with the Total row.")


def render_forward_test():
    """Details tab: the forward-test leaderboard from FORWARD_START (forward_test.py): every paper strategy
    (forward_test.STRATEGIES, Reports/forward_strategies.csv), the real account (Reports/forward_test_daily.csv), QQQ, SPY."""
    import forward_test as ft
    from backtest_engine import FORWARD_START
    start = f"{pd.Timestamp(FORWARD_START):%b %-d, %Y}"
    daily, values = read_report_csv(ft.DAILY_CSV), read_report_csv(ft.STRATEGIES_CSV)
    if (daily is None or daily.empty) and (values is None or values.empty):
        st.info(f"Forward test started {start}; the first daily row is saved after the close (4:15 PM CT on trading days).")
        return
    bench = load_benchmarks()
    board = ft.leaderboard(daily, bench.reset_index() if bench is not None else None, values)
    board = board.rename(columns={"Total return %": f"Total return since {start[:-6]} %"}).assign(
        Rank=lambda b: b["Rank"].map(lambda r: "–" if pd.isna(r) else str(r)))   # – = not ranked (comparison / no full week yet)
    sty = toned(toned(board, [c for c in board.columns if "return" in c]), ["Max drawdown %"], fn=drawdown_tone)
    st.dataframe(live_row(sty, "Strategy", ft.LIVE), hide_index=True, width="stretch",
                 height=35 * (len(board) + 1) + 3,                                                # every row, no scrolling
                 column_config={c: PCT_COL for c in board.columns if c.endswith("%")})
    acct = ""
    if daily is not None and len(daily):
        last = daily.iloc[-1]
        n, w, traded, cost = int(last["Closed_Picks"]), int(last["Winning_Picks"]), float(last["Traded_USD"]), float(last["Cost_USD"])
        acct = (f" Your account as of {last['Date']} {last['Time_CT']} CT: "
                + (f"closed picks {n}, win rate {w / n:.0%}" if n else "no closed picks yet")
                + f", traded \\${traded:,.0f}, cost vs the decision price {'-' if cost < 0 else ''}\\${abs(cost):,.2f}"
                + (f" ({cost / traded * 1e4:+.1f} bps; + = it cost money)" if traded else "") + ".")
    st.caption(f"**{ft.verdict(board)}** {ft.RANK_RULE} Every strategy is paper only (never traded): it decides at the "
               "day's close, pays 0.1% per trade side, holds no stock above 20%, invests at most 99% and earns nothing on cash; each "
               f"starts at 1.0 on the {start} close. Your account = equity net of new deposits; QQQ / SPY = closes, "
               "comparison only (not ranked). Weekly = Friday to Friday; None / – = no full week yet." + acct
               + " Saved by the 4:15 PM CT job (no orders); each strategy's rule and holdings are in the next section.")


def render_forward_rules():
    """Details tab: each forward-test strategy's one-line rule and its holdings on the latest saved day."""
    import forward_test as ft
    held, h = ft.holdings(), read_report_csv(ft.HOLDINGS_CSV)
    rules = pd.DataFrame([{"Strategy": c["name"], "Rule": c["rule"], "Holdings now (target weight)": held.get(c["name"], "cash")}
                          for c in ft.STRATEGIES])
    sty = toned(rules, ["Holdings now (target weight)"], fn=lambda v: MUTED if v == "cash" else INK)
    st.dataframe(live_row(sty, "Strategy", ft.LIVE), hide_index=True, width="stretch")
    if h is not None and len(h):
        st.caption(f"Holdings as of the {pd.Timestamp(h['Date'].max()):%a %b %-d} close (Reports/forward_strategies_holdings.csv "
                   "has every day). Paper only: none of these is traded.")


def render_earnings_stops():
    """The live pre-earnings stop (earnings_stop.py, launchd every 10 min): its latest check and the stop sales."""
    path = os.path.join(REPORTS, "earnings_stops.csv")
    try:
        t = pd.read_csv(path)
    except (OSError, ValueError):
        t = None
    if t is None:
        st.info("No check yet: the stop job checks every 10 minutes on trading days, 3:00 AM to 7:00 PM CT.")
    elif t.empty:
        st.caption(f"No held stock has earnings within 7 days (last check "
                   f"{datetime.fromtimestamp(os.path.getmtime(path)):%a %b %-d %-I:%M %p} CT).")
    else:
        st.dataframe(t.drop(columns="Checked_At_CT"), hide_index=True, width="stretch")
        st.caption(f"Last check {t['Checked_At_CT'].iloc[0]} CT. Stop = highest close since entry - 3 × ATR(14); "
                   "Price = the latest quote's mid price.")
    try:
        with open(os.path.join(REPORTS, "earnings_stop_state.json")) as f:
            sold = json.load(f).get("sold", {})
    except (OSError, ValueError):
        sold = {}
    if sold:
        st.caption("Stop sales: " + "; ".join(f"{k.split('|')[0]} before its {k.split('|')[1]} report, "
                                               f"{str(v.get('at'))[:16].replace('T', ' ')} CT" for k, v in sorted(sold.items())))


def render_details(p):
    with st.expander("Live holdings (Alpaca account)", expanded=True):
        render_live_holdings()
    import forward_test as ft
    with st.expander(f"Forward test · {len(ft.STRATEGIES)} strategies vs your account, QQQ and SPY since Oct 2, 2026",
                     expanded=True):
        render_forward_test()
    with st.expander("Strategy rules and holdings (paper strategies)", expanded=False):
        render_forward_rules()
    with st.expander(latest_signals_title(p), expanded=True):
        render_latest_signals(p)
    with st.expander(f"Last decision · {p.off_date:%a %b %-d} (decisions in force, every stock)", expanded=False):
        render_last_decision(p)
    with st.expander("Pre-earnings stops (3× ATR, live account)", expanded=False):
        render_earnings_stops()
    with st.expander("Data freshness and settings", expanded=False):
        render_data_and_settings(p)
    with st.expander("Strategy rules", expanded=False):
        st.markdown(rules_text())


# =====================================================================================================================
# 12. Main
# =====================================================================================================================
def main():
    with st.spinner("Loading data..."):
        p = build_page()
    # ?symbol=X opens that stock (table clicks and the Telegram messages use this)
    q_symbol = str(st.query_params.get("symbol", "")).strip().upper()
    jumped = q_symbol in p.options
    if jumped:
        st.session_state["_pending_ticker"] = q_symbol
        del st.query_params["symbol"]

    render_top_bar(p)
    # One-click freshness: every data cache is already keyed on the Reports/*.csv modification times,
    # so clearing the caches and rerunning always shows the newest pipeline output + live quotes.
    if st.button("Refresh data",
                   help="Clear all cached data and reload the latest Reports/*.csv files and live quotes."):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    tab_main, tab_details = st.tabs(["Dashboard", "Details"])
    with tab_main:
        ticker, tdata = render_stock_picker(p, jumped)
        if ticker in p.short.index:
            render_short_stock(ticker, p.short.loc[ticker])
        else:
            render_stock_header(p, ticker, tdata)
            fig, has_strategy = stock_chart_inputs(ticker, tdata)
            render_stock_more(ticker, tdata, has_strategy)
            render_stock_figure(fig, ticker)
    with tab_details:
        render_details(p)


main()
