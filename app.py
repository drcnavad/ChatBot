"""
Stock Analysis dashboard (Streamlit).

Page layout, top to bottom:
  1. Title bar + ONE-line holdings alert (what to do today).
  2. "Dashboard" tab: a compact summary (key numbers, current picks, earnings this week)
     and the single-stock view. Open any stock directly with  http://localhost:8501/?symbol=NVDA
  3. "Details" tab: everything else (all signals, rank history, rules and decisions, holdings risk,
     order preview, backtest results, data freshness), each in its own expander.

The app only READS the Reports/*.csv files written by `python run_all.py`. It never places orders and never calls
a paid data API (the optional "AI analysis" button uses the Hugging Face token from .env).
"""
import html
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
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)  # project modules (backtest_engine, holdings_alert, paper_trade) importable from any cwd


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
    from backtest_engine import WINNER, winner_max_per_sector
    SECTOR_MAX = winner_max_per_sector()
except Exception:
    WINNER, SECTOR_MAX = {}, 4

try:
    import sector_mapping as _sm
    N_TRADABLE, EXPANSION = len(_sm.tradable_symbols), getattr(_sm, "EXPANDED_UNIVERSE", None)
except Exception:
    N_TRADABLE, EXPANSION = None, None

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

MIDWEEK_NOTE = (f"Mid-week swap: at the {' and '.join(MIDWEEK['days'])} closes (next session if a holiday), if a stock that is not "
                f"held ranks in the top {MIDWEEK['enter_top']} and a held stock has fallen below rank {MIDWEEK['exit_below']}, "
                "the worst-ranked held stock is sold and the new one bought with the same dollar amount at the next open "
                + ("(any sector: the sector limit does not block it). " if CAP_SOFT else f"(max {SECTOR_MAX} per sector still applies). ")
                if MIDWEEK else "")
EXIT_NOTE = (f"Mid-week exit: after the swap step, any holding ranked worse than {EXIT_BELOW} (or no longer ranked) is sold at the "
             "next open and the cash stays idle until the Friday rebalance. " if EXIT_BELOW else "")
EARNINGS_NOTE = (f"Earnings rule: at the Friday rebalance and the mid-week checks a "
                 f"stock that is not held is not bought if its next earnings date is within the next {EARNINGS} calendar days; "
                 f"its slot goes to the next eligible stock, else cash. Held stocks are never sold because of earnings. " if EARNINGS else "")


def rules_text():
    """The live trading rules in plain words."""
    return (f"**Strategy rules ({STRATEGY_TAG})**\n"
            f"- **When:** every Mon / Wed / Fri at the close (decision days).\n"
            f"- **What:** hold the top 10 stocks by Strategy Score (0.5 × Technical + 0.5 × Relative Strength {RS_LABEL}), score > 0, max {SECTOR_MAX} per sector.\n"
            f"- **Size:** weights ∝ 1 / 63-day volatility (less volatile = larger position).\n"
            f"- **Market filter:** if QQQ closes at or below its 200-day average, every position is halved (50% cash).\n"
            f"- **Signals:** Bullish (Buy) = enters the top 10 · Hold = stays · Bearish (Sell) = leaves.\n"
            + ("- **Mon/Wed check:** " + MIDWEEK_NOTE.strip() + "\n" if MIDWEEK else "")
            + ("- **Exit rule:** " + EXIT_NOTE.strip() + "\n" if EXIT_BELOW else "")
            + ("- **Earnings:** " + EARNINGS_NOTE.strip() + "\n" if EARNINGS else ""))


# =====================================================================================================================
# 2. Page setup, styling and file locations
# =====================================================================================================================
load_dotenv()
st.set_page_config(page_title="Stock Analysis Report", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
    .stApp, .main { background: #f1f5f9; }
    .main .block-container { padding: 1rem 2rem 2.5rem 2rem !important; max-width: 1400px; }
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
    .js-plotly-plot { border-radius: 12px; background: #fff; border: 1px solid #e2e8f0; padding: 4px; }
    .symbol-link { color: #0f766e; text-decoration: none; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }
    .symbol-link:hover { color: #0d9488; text-decoration: underline; }
    .sa-topbar { display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; margin-bottom: 0.4rem; flex-wrap: wrap; }
    .sa-topbar h1 { margin: 0; font-size: 1.45rem; font-weight: 700; }
    .sa-topbar p { margin: 0.2rem 0 0; color: #64748b; font-size: 0.9rem; }
    .sa-chip { font-size: 0.75rem; font-weight: 600; color: #0f766e; background: #ccfbf1; border: 1px solid #99f6e4;
               padding: 0.35rem 0.7rem; border-radius: 999px; white-space: nowrap; }
    .sa-section { font-weight: 700; color: #0f172a; margin: 0.6rem 0 0.2rem; font-size: 1.02rem; }
    .sa-hero { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1rem 1.25rem; margin: 0.5rem 0 1rem; }
    .sa-hero-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
    .sa-ident { display: flex; align-items: center; gap: 0.85rem; }
    .sa-sym { font-size: 1.75rem; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; }
    .sa-price { font-size: 1.25rem; font-weight: 700; color: #334155; font-family: 'JetBrains Mono', monospace; }
    .sa-badge { display: inline-block; padding: 0.35rem 0.8rem; border-radius: 999px; font-weight: 700; font-size: 0.8rem; }
    .sa-badge-bull { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .sa-badge-bear { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .sa-badge-hold { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .sa-badge-holdpos { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
    .sa-badge-grey { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
    .sa-why { font-size: 0.8rem; color: #64748b; margin-top: 0.35rem; }
    .sa-stats { display: flex; flex-wrap: wrap; gap: 0.5rem 1.35rem; justify-content: flex-end; }
    .sa-stat { min-width: 3.5rem; }
    .sa-stat-label { font-size: 0.66rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }
    .sa-stat-val { font-size: 0.95rem; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

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
PER_STOCK_CSV = os.path.join(REPORTS, "backtest_per_stock.csv")        # written by backtest.ipynb
NEWS_CSV = os.path.join(REPORTS, "news_cleaned_df.csv")
COMPANY_XLSX = os.path.join(REPORTS, "complete_company_analysis.xlsx")
POSITIONS_CSV = os.path.join(ROOT, "my_positions.csv")
CT = ZoneInfo("America/Chicago")
MA_COLS = ['ma_10', 'ma_30', 'ma_50', 'ma_100', 'ma_200']

# file -> (what it is, max age in days before it is flagged stale)
FRESHNESS = {
    "signal_analysis.csv": ("prices, signals, strategy weights", 3),
    "strategy_picks.csv": ("current / provisional portfolio", 3),
    "strategy_tracking.csv": ("forward tracking log", 3),
    "strategy_decisions.csv": ("decision history: weekly rebalances + mid-week swaps (chart markers)", 3),
    "strategy_midweek_check.csv": ("this week's decisions: Friday rebalance + Mon/Wed swap checks", 3),
    "benchmark_prices.csv": ("SPY / QQQ / sector ETF closes (RS lines)", 3),
    "weighted_sentiment.csv": ("news sentiment scores", 7),
    "news_cleaned_df.csv": ("news articles for AI summaries", 7),
    "earnings_date.csv": ("earnings calendar", 14),
    "complete_company_analysis.xlsx": ("fundamentals / fair value", 30),
    "balance_sheet_weights.csv": ("balance-sheet scores", 30),
    "balance_sheet.csv": ("raw quarterly fundamentals", 100),
    "backtest_summary.csv": ("backtest of the live rules (backtest.ipynb)", 120),
}
APP_FILES = {"signal_analysis.csv", "strategy_picks.csv", "strategy_tracking.csv", "news_cleaned_df.csv", "earnings_date.csv",
             "complete_company_analysis.xlsx", "backtest_summary.csv", "strategy_decisions.csv", "benchmark_prices.csv"}


# =====================================================================================================================
# 3. Small HTML helpers
#    Markdown ends an HTML block at the first blank line and turns 4-space-indented lines into code, so every custom
#    HTML block goes through show_html(): one line, no indentation, dynamic text escaped with esc().
# =====================================================================================================================
esc = html.escape


def show_html(markup):
    st.markdown("".join(line.strip() for line in str(markup).splitlines()), unsafe_allow_html=True)


def symbol_link(sym):
    return f'<a href="?symbol={quote(sym)}" class="symbol-link" target="_self">{esc(sym)}</a>'


def section(title):
    show_html(f'<div class="sa-section">{esc(title)}</div>')


def num(v):
    return float(v) if pd.notna(v) else None


def fmt(v, spec, prefix="", suffix=""):
    return f"{prefix}{v:{spec}}{suffix}" if v is not None else "—"


def sign_color(v):
    return "#0f172a" if v is None else ("#15803d" if v >= 0 else "#b91c1c")


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


# --- Decision-day ranks (Mon/Wed/Fri): rank change compares the last two decision days, not calendar days ---
def _decision_days(df):
    """Sorted Mon/Wed/Fri decision dates present in the signal data."""
    mask = (df["Rebalance_Day"] == 1) | (df["Midweek_Check"] == 1)
    return sorted(df.loc[mask, "Date"].unique())


def _day_ranks(df, day):
    """Symbol -> rank (1 = best) for one date: the pipeline's Strategy_Rank when present, else by combined_signal."""
    has_rank = "Strategy_Rank" in df.columns
    sub = (df.loc[df["Date"] == day, ["Symbol", "combined_signal"] + (["Strategy_Rank"] if has_rank else [])]
           .sort_values(["Strategy_Rank"] if has_rank else ["combined_signal"], ascending=has_rank)
           .drop_duplicates(subset=["Symbol"]))
    return pd.Series(range(1, len(sub) + 1), index=sub["Symbol"].values)


@st.cache_data(ttl=3600)
def day_rank_change(_df, mtime: float):
    """Symbol -> (prev_decision_rank - latest_decision_rank, latest_rank); positive = moved up.
    Compares the last two Mon/Wed/Fri decision days, not consecutive calendar days."""
    df = _df
    days = _decision_days(df)
    if len(days) < 2:
        return {}
    t = _day_ranks(df, days[-1])
    y = _day_ranks(df, days[-2])
    common = t.index.intersection(y.index)
    return {s: (int(y[s]) - int(t[s]), int(t[s])) for s in common}


def rank_trend(ranks_old_to_new):
    """Rank momentum (information only, not a trade rule): Bullish = 4+ rank improvements in a row ending today,
    Bearish = 3+ declines in a row, otherwise Hold. Runs over decision days (Mon/Wed/Fri)."""
    vals = list(ranks_old_to_new)
    if len(vals) < 4:
        return "Hold"
    improve = worsen = 0
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] < vals[i - 1] and not worsen:
            improve += 1
        elif vals[i] > vals[i - 1] and not improve:
            worsen += 1
        else:
            break
    return "Bullish" if improve >= 4 else ("Bearish" if worsen >= 3 else "Hold")


@st.cache_data(ttl=3600)
def rank_pivot(_df, mtime: float, n_days=20):
    """Symbol x last-n Mon/Wed/Fri decision-day rank table (newest first) + Trend."""
    df = _df
    dec = df[(df["Rebalance_Day"] == 1) | (df["Midweek_Check"] == 1)]
    work = dec[['Symbol', 'Date', 'combined_signal']].dropna(subset=['combined_signal'])
    dates = sorted(work['Date'].unique())[-n_days:]
    work = work[work['Date'].isin(dates)].sort_values('combined_signal', ascending=False).drop_duplicates(subset=['Symbol', 'Date'])
    if 'Strategy_Rank' in df.columns:  # the pipeline's rank (ties: higher RS, then A-Z), identical to the header/chart
        work = work.merge(df[['Symbol', 'Date', 'Strategy_Rank']].drop_duplicates(['Symbol', 'Date']), on=['Symbol', 'Date'], how='left')
    by_score = work.groupby('Date')['combined_signal'].rank(ascending=False, method='first')
    work['rank'] = (work['Strategy_Rank'].fillna(by_score) if 'Strategy_Rank' in work else by_score).astype(int)
    pivot = work.pivot(index='Symbol', columns='Date', values='rank')
    pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)
    trend = pivot.apply(lambda row: rank_trend(int(v) for v in row.iloc[::-1] if pd.notna(v)), axis=1)
    pivot.columns = [pd.Timestamp(c).strftime("%m/%d") for c in pivot.columns]
    pivot["Trend"] = trend
    return pivot.sort_index().rename_axis("Symbol").reset_index()


# --- Earnings / fundamentals / news ---
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


def upcoming_earnings(symbols, days=7):
    """Earnings within the next `days` days, soonest first (one row per symbol)."""
    ed = load_earnings()
    today = pd.Timestamp.now().normalize()
    soon = ed[ed['Symbol'].isin(symbols) & ed['Earnings Date'].between(today, today + pd.Timedelta(days=days))]
    return soon.sort_values(['Earnings Date', 'Symbol']).drop_duplicates('Symbol').fillna({'Time': '—'})


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
SIGNAL_COLOR = {"Bullish (Buy)": "#15803d", "Hold": "#2563eb", "Bearish (Sell)": "#b91c1c", "Bearish": "#b91c1c",
                "Neutral": "#b45309", "Neutral (sector cap)": "#b45309", "Not ranked": "#64748b"}
SIGNAL_BADGE = {"Bullish (Buy)": "sa-badge-bull", "Hold": "sa-badge-holdpos", "Bearish (Sell)": "sa-badge-bear",
                "Bearish": "sa-badge-bear", "Neutral": "sa-badge-hold", "Neutral (sector cap)": "sa-badge-hold",
                "Not ranked": "sa-badge-grey"}
MARKET_FILTER_TIP = "ON = QQQ above its 200-day average; OFF halves all positions"


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
    if signal == "Bullish (Buy)":
        rk = int(m.group(1)) if m else (int(rank) if rank is not None and pd.notna(rank) else None)
        if "sector cap relaxed" in reason:
            return f"made the portfolio at rank {rk} (free slot filled from the top {MAX_PICK or 20}, sector limit relaxed)"
        if rk is not None and rk > 10:
            return f"made the portfolio at rank {rk} (higher-ranked stocks were skipped by the {SECTOR_MAX}-per-sector limit)"
        return f"made the top 10 at rank {rk if rk is not None else r}"
    if signal == "Hold":
        return f"in top 10, rank {r}" if rank is not None and pd.notna(rank) and rank <= 10 else \
            f"still selected at rank {r} (higher-ranked stocks skipped by the sector limit)"
    if signal == "Bearish (Sell)":
        if reason.startswith("score"):
            return f"score fell below 0 ({sc})"
        if "picks only from ranks" in reason:
            return f"fell to rank {m.group(1) if m else r}, worse than {MAX_PICK} (picks only from ranks 1–{MAX_PICK})"
        if reason.startswith("skipped"):
            return f"skipped: already {SECTOR_MAX} stocks from this sector"
        if "outside top" in reason:
            return f"fell to rank {m.group(1) if m else r}, outside top 10"
        if reason.startswith("not eligible"):
            return "not enough data / not eligible"
        return reason or "left the top 10"
    if signal == "Neutral (sector cap)":
        return f"rank {r} but skipped: already {SECTOR_MAX} stocks from this sector"
    if signal == "Neutral":
        return f"rank {r}, positive score but outside top 10 — watch"
    if signal == "Bearish":
        return f"score below 0 ({sc})"
    return "benchmark / not enough history to rank"


def signal_board(df):
    """Every symbol's signal for the decisions in force (the last rebalance / mid-week check).

    Uses Reports/strategy_changes.csv (the engine's decisions) plus signal_analysis.csv for stocks not in it.
    Returns (board DataFrame, decision date)."""
    ch = read_report_csv(CHANGES_CSV)
    sub = ch[ch["View"] == "last rebalance"] if ch is not None else pd.DataFrame()
    if not sub.empty:
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
            sig, weight = "Bullish (Buy)", d["New_Weight"]
        elif status == "hold":
            sig, weight = "Hold", d["New_Weight"]
        elif status == "drop":
            sig, weight = "Bearish (Sell)", d["Old_Weight"]
        elif d is not None and str(d["Reason"]).startswith("skipped"):
            sig, weight = "Neutral (sector cap)", 0.0
        elif pd.isna(score):
            sig, weight = "Not ranked", np.nan
        else:
            sig, weight = ("Bearish" if score <= 0 else "Neutral"), 0.0
        ned = next_ed.get(sym, "")
        soon = ""
        if ned:
            n_days = int(np.busday_count(today.date(), pd.Timestamp(ned).date()))
            soon = "⚠️ within 2 sessions" if 0 <= n_days <= 2 else ""
        sector = d["Sector"] if d is not None and "Sector" in d and pd.notna(d["Sector"]) else symbol_sector.get(sym)
        rows.append({"Symbol": sym, "Signal": sig, "Rank": rank, "Score": score,
                     "Portfolio weight %": weight * 100 if pd.notna(weight) else np.nan, "Sector": sector or "—",
                     "Why": plain_reason(sig, d["Reason"] if d is not None else None, rank, score),
                     "Next earnings": ned, "Earnings soon": soon})
    board = pd.DataFrame(rows, columns=["Symbol", "Signal", "Rank", "Score", "Portfolio weight %", "Sector", "Why",
                                        "Next earnings", "Earnings soon"])
    board = board.sort_values(["Rank", "Symbol"], na_position="last").reset_index(drop=True)
    picked = board["Signal"].isin(["Bullish (Buy)", "Hold"])
    board.insert(2, "Portfolio slot", "—")
    board.loc[picked, "Portfolio slot"] = [str(i) for i in range(1, int(picked.sum()) + 1)]
    return board, date


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

    Strategy_Weight on day d is the target decided at d's close and filled at the next open, so an entry/exit is the
    first day the weight turns >0 / back to 0 and the fill is the following session. Reasons, rank and score come from
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


def slot_series(symbol, dates):
    """Daily portfolio slot (1..10 = position among the picks of the decision in force; NaN when not held)."""
    d = load_decisions()
    out = pd.Series(np.nan, index=pd.DatetimeIndex(dates))
    if d is None:
        return out
    sel = d[d["Status"].isin(["add", "hold"])].sort_values(["Date", "Rank", "Symbol"]).copy()
    sel["Slot"] = sel.groupby("Date").cumcount() + 1
    dec_days = sorted(sel["Date"].unique())
    if not dec_days:
        return out
    mine = sel[sel["Symbol"] == symbol].set_index("Date")["Slot"]
    in_force = pd.Series(dec_days, index=dec_days).reindex(out.index, method="ffill")
    return pd.Series([mine.get(x, np.nan) if pd.notna(x) else np.nan for x in in_force], index=out.index)


def event_hover(e):
    """Hover text for an entry/exit marker on the price chart."""
    sig = "Bullish (Buy)" if e.Kind == "entry" else "Bearish (Sell)"
    text = f"<b>{sig}</b>: {html.escape(plain_reason(sig, e.Reason, e.Rank, e.Score))}"
    when = f"filled at the open {e.Fill:%a %b %d}" if pd.notna(e.Fill) else "fills at the next open (pending)"
    text += f"<br>Decided at the close {e.Decision:%a %b %d} · {when}"
    bits = ([f"Rank #{e.Rank:.0f}"] if pd.notna(e.Rank) else []) + ([f"Score {e.Score:.1f}"] if pd.notna(e.Score) else [])
    if pd.notna(e.Weight):
        bits.append(f"{'Portfolio weight' if e.Kind == 'entry' else 'Weight sold'} {e.Weight * 100:.1f}%")
    if bits:
        text += "<br>" + " · ".join(bits)
    if e.Regime_On == 0:
        text += "<br>Market filter OFF that week (QQQ below its 200-day average): positions halved"
    return text


def relative_strength_lines(chart, symbol):
    """Price ratio of the stock vs its sector ETF and vs SPY, rebased to 100 at the start of the chart window."""
    bench = load_benchmarks()
    if bench is None:
        return {}
    b = bench.reindex(chart.index).ffill()
    etf = sector_etf_for(symbol)
    lines = {}
    for label, ref in ((f"vs {etf} (sector ETF)", etf), ("vs SPY", "SPY")):
        if not ref or ref == symbol or ref not in b.columns:
            continue
        ratio = (chart["Close"] / b[ref]).replace([np.inf, -np.inf], np.nan)
        first = ratio.first_valid_index()
        if first is not None:
            lines[label] = ratio / ratio.loc[first] * 100
    return lines


def daily_status(weight, score):
    """Status on an ordinary day (between decisions) for the chart hover text."""
    if score is None or pd.isna(score):
        return "Not ranked"
    if weight is not None and pd.notna(weight) and weight > 0:
        return "Hold"
    return "Bearish" if score <= 0 else "Neutral"


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
    return SimpleNamespace(
        df=df, mtime=mtime, latest=latest, by_symbol=latest.set_index("Symbol"),
        options=latest.sort_values('combined_signal', ascending=False)['Symbol'].tolist(),  # dropdown: best score first
        rank_change=day_rank_change(df, mtime),
        board_off=board_off, off_date=off_date,
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
# 9. Top of the page: title bar and the one-line holdings alert
# =====================================================================================================================
def render_top_bar(p):
    stale = p.freshness.loc[p.freshness["Status"].str.startswith("⚠️"), "File"].tolist()
    updated = datetime.fromtimestamp(p.mtime, tz=CT).strftime("%m/%d/%Y %I:%M %p CT")
    note = f" · ⚠️ {len(stale)} stale file(s), see Details" if stale else ""
    show_html(f"""
        <div class="sa-topbar">
          <div>
            <h1>Stock Analysis</h1>
            <p>Weekly top-10 ranking{" + Mon/Wed swap check" if MIDWEEK else ""} · technical + strength vs sector/SPY</p>
          </div>
          <div class="sa-chip">Updated {esc(updated + note)}</div>
        </div>""")


@st.cache_data(ttl=600)
def _cached_alert(mtimes, use_positions):
    import holdings_alert
    return holdings_alert.build_alert(use_positions=use_positions)


def render_alert():
    """ONE line saying what to do today (from holdings_alert.py; tickers link to the stock view).
    Uses my_positions.csv when it exists, unless 'Strategy holdings' is chosen under Details → Data freshness and settings."""
    try:
        import holdings_alert
    except Exception as e:  # deployed without the pipeline modules
        st.caption(f"Holdings alert unavailable: {e}")
        return
    use_pos = st.session_state.get("alert_source", "Your positions file") == "Your positions file"
    mt = tuple(os.path.getmtime(f) if os.path.exists(f) else 0 for f in (SIGNAL_CSV, MIDWEEK_CSV, POSITIONS_CSV))
    try:
        a = _cached_alert(mt, use_pos)
    except Exception as e:
        st.warning(f"Holdings alert unavailable: {e}")
        return
    parts = " · ".join("".join(esc(x) if isinstance(x, str) else
                               f'<a href="{holdings_alert.APP_URL}{quote(x[1])}" class="symbol-link" target="_self">{esc(x[1])}</a>'
                               for x in seg) for seg in a["lines"])
    color = {"red": "#b91c1c", "green": "#166534", "blue": "#1e40af"}.get(a["level"], "#0f172a")
    d = pd.Timestamp(a["data_date"])
    tag = f'{a["source"]}, data {d:%a %b} {d.day}'
    show_html(f'<div class="sa-alert sa-alert-{a["level"]}" style="font-size:0.95rem;font-weight:600;color:{color};background:#fff;'
              f'border:1px solid #e2e8f0;border-left:4px solid {color};border-radius:10px;padding:8px 12px;margin:2px 0 8px 0;">'
              f'{holdings_alert.LEVEL_ICON[a["level"]]} {parts} '
              f'<span style="font-weight:400;color:#64748b;font-size:0.8rem;">({esc(tag)})</span></div>')
    if a["stale"]:
        st.caption("⚠️ " + a["stale"])


# =====================================================================================================================
# 10. Dashboard tab: summary
# =====================================================================================================================
def render_key_metrics(p):
    """Five headline numbers: market filter, invested, stocks held, last and next decision."""
    la = p.by_symbol
    regime = bool(la["Regime_On"].dropna().iloc[0]) if la["Regime_On"].notna().any() else None
    c = st.columns(5)
    c[0].metric("Market filter", "ON ✅" if regime else ("OFF ⚠️ · positions halved" if regime is not None else "—"),
                help=MARKET_FILTER_TIP)
    c[1].metric("Invested", f"{la['Strategy_Weight'].fillna(0).sum():.0%}")
    c[2].metric("Stocks held", int((la["Strategy_Weight"] > 0).sum()))
    c[3].metric("Last decision", f"{p.off_date:%a %b %d}")
    c[4].metric("Next decision", f"{p.next_dec:%a %b %d}" if p.next_dec is not None else "—", help=p.next_kind or None)


def render_week_decisions(p):
    """This week's decisions in plain English (Friday rebalance + Mon/Wed checks) and the next decision."""
    m = p.midweek
    if m is None:
        if p.next_dec is not None:
            st.caption(f"Next decision: {p.next_kind} at the close of {p.next_dec:%a %b %d}.")
        return
    items = "".join(f'<li style="margin:2px 0;{"font-weight:700;" if int(r.Is_Latest) else ""}">{esc(str(r.Message))} '
                    f'<span style="color:#64748b;font-weight:400;">({esc(str(r.Status))})</span></li>' for r in m.itertuples())
    show_html(f'<div class="sa-midweek" style="border:1px solid #cbd5e1;border-left:4px solid #0f766e;border-radius:10px;'
              f'background:#f8fafc;padding:8px 12px;margin:4px 0 10px 0;font-size:0.88rem;">'
              f'<div style="font-weight:700;color:#0f766e;">This week · {esc(str(m["Rules"].iloc[0]))} · data through '
              f'{pd.Timestamp(m["As_Of"].iloc[0]):%a %b %d}</div>'
              f'<ul style="margin:4px 0 2px 18px;padding:0;">{items}</ul>'
              f'<div style="color:#334155;">{esc(str(m["Next_Message"].iloc[0]))}</div></div>')


def render_picks_table(p):
    """Current portfolio (the decisions in force). Click a row to open it."""
    la = p.by_symbol
    held = la[la["Strategy_Weight"] > 0]
    w = held["Strategy_Weight"].fillna(0)
    table = pd.DataFrame({
        "Symbol": held.index,
        "Signal": [p.sig_off.get(s, "—") for s in held.index],
        "Rank": held["Strategy_Rank"].round(0).to_numpy(),
        "Portfolio weight %": (w * 100).round(1).to_numpy(),
        "Sector": [symbol_sector.get(s, "—") for s in held.index],
        "Next earnings": last_next_earnings(list(held.index))["Next ED"].to_numpy(),
        "Why": [p.why_off.get(s, "") for s in held.index],
    }).sort_values(["Portfolio weight %", "Rank"], ascending=[False, True]).reset_index(drop=True)
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                         key="summary_tbl", column_config={"Why": st.column_config.TextColumn("Why", width="large")})
    st.caption(f"Decisions in force (last decision {p.off_date:%a %b %d}). Click a row to open the stock.")
    open_symbol(table, event, "summary")


def render_earnings_line(p):
    """Earnings in the next 7 days, one line with links."""
    up = upcoming_earnings(sorted(p.by_symbol.index))
    if up.empty:
        st.caption("Earnings · next 7 days: none.")
        return
    items = " · ".join(f"{symbol_link(s)} {d:%a %b %d} {esc(str(t))}" for s, d, t in zip(up["Symbol"], up["Earnings Date"], up["Time"]))
    show_html(f'<div style="font-size:0.88rem;margin:2px 0 4px 0;"><b>Earnings · next 7 days:</b> {items} '
              f'<span style="color:#64748b;font-size:0.78rem;">(AM = before the open, PM = after the close; unannounced '
              f'times are predicted)</span></div>')


def render_summary(p):
    section("Summary")
    render_key_metrics(p)
    render_week_decisions(p)
    render_picks_table(p)
    render_earnings_line(p)


# =====================================================================================================================
# 11. Dashboard tab: single-stock view
# =====================================================================================================================
def ticker_label(p, s):
    r = p.by_symbol.loc[s]
    rank, score = r.get('Strategy_Rank'), r['combined_signal']
    parts = [s, p.sig_off.get(s, "Not ranked")]
    if pd.notna(rank):
        parts.append(f"rank #{rank:.0f}")
    parts.append(f"score {score:.0f}" if pd.notna(score) else "score —")
    return "  ·  ".join(parts)


def render_stock_picker(p, jumped):
    """Stock dropdown (best score first) + AI button. ?symbol=X, a table click or the dropdown choose the stock."""
    pending = st.session_state.pop("_pending_ticker", None)
    if pending in p.options:
        st.session_state.ticker_dropdown = pending
    elif st.session_state.get("ticker_dropdown") not in p.options:
        st.session_state.ticker_dropdown = p.options[0]
    st.markdown('<div id="ticker-focus"></div>', unsafe_allow_html=True)
    if jumped or pending:  # opened from a link or a table: scroll the stock view into sight
        components.html("<script>const el = window.parent.document.getElementById('ticker-focus');"
                        "if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});</script>", height=0)
    section("Stock view")
    pick_col, ai_col = st.columns([3, 1])
    ticker = pick_col.selectbox("Ticker", options=p.options, format_func=lambda s: ticker_label(p, s),
                                key="ticker_dropdown", label_visibility="collapsed")
    tdata = p.df[p.df['Symbol'] == ticker]
    if ai_col.button("Generate AI Analysis", type="primary", width="stretch", key="generate_ai_btn"):
        ai_analysis_dialog(ticker, tdata, p.sig_off.get(ticker, 'Not ranked'), p.why_off.get(ticker, ''))
    return ticker, tdata


def render_stock_header(p, ticker, tdata):
    """Name, price, official signal and the numbers that decide it (score, rank, slot, weight)."""
    latest = tdata.nlargest(1, 'Date').iloc[0]
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
    stats = [
        stat_html("Strategy score", fmt(score, ".1f")),
        stat_html("Rank today", f"#{rank:.0f} / {n_ranked}" if rank is not None else "—",
                  tip=f"Position by strategy score among the {n_ranked} ranked stocks at the latest close (1 = best)"),
        stat_html(f"Portfolio slot · {p.off_date:%b %d}", f"{slot} of 10" if slot != "—" else "— (not picked)",
                  tip="Position among the 10 stocks picked at the last decision, in rank order. The picks skip stocks "
                      f"whose sector already has {SECTOR_MAX}, so the slot can be smaller than the rank."),
        stat_html("Portfolio weight", fmt(weight * 100 if weight > 0 else None, ".1f", suffix="%")),
        stat_html("Technical", fmt(num(latest.get('Technical_Score')), ".1f")),
        stat_html("Strength vs sector/SPY", fmt(num(latest.get('RS_Score')), ".1f"), sign_color(num(latest.get('RS_Score')))),
        stat_html("Next earnings", pd.Timestamp(next_ed).strftime("%b %d") if next_ed else "—"),
    ]
    hold = row_for(HOLDINGS_CSV, ticker) if weight > 0 else None
    if hold is not None:
        stats += [
            stat_html("Held since", f"{pd.Timestamp(hold['Entry_Date']):%b %d} · {int(hold['Days_Held'])} sessions"
                      if pd.notna(hold['Entry_Date']) else "—"),
            stat_html("P&L since entry", fmt(num(hold['PnL_%']), "+.1f", suffix="%"), sign_color(num(hold['PnL_%']))),
            stat_html("To 3×ATR stop", fmt(num(hold['Dist_to_Stop_%']), ".1f", suffix="%")),
        ]
    show_html(f"""
        <div class="sa-hero">
          <div class="sa-hero-row">
            <div class="sa-ident">
              <div class="sa-sym">{esc(ticker)}</div>
              <div class="sa-price">{fmt(num(latest['Close']), ",.2f", "$")}</div>
              <span class="sa-badge {SIGNAL_BADGE.get(status, 'sa-badge-grey')}" title="Official signal from the decisions in force">{esc(status)}</span>
              {chips_html}
            </div>
            <div class="sa-stats">{"".join(stats)}</div>
          </div>
          <div class="sa-why">{esc(status + ": " + why) if why else ""}</div>
        </div>""")


def build_price_chart(p, ticker, tdata, show_strategy, show_rs, show_classic, show_legacy):
    """Last 12 months: price + moving averages + buy/sell markers, optional score/rank, relative strength, RSI/MACD panels."""
    chart = tdata[tdata['Date'] >= tdata['Date'].max() - pd.Timedelta(days=365)].sort_values('Date').set_index('Date')
    x_start, x_end = chart.index[0], chart.index[-1]
    events, periods = strategy_events(tdata, load_decisions(), ticker)
    rs_lines = relative_strength_lines(chart, ticker) if show_rs else {}

    panels = ["price"] + (["score", "rank"] if show_strategy else []) + (["rs"] if rs_lines else []) \
        + (["rsi", "macd"] if show_classic else [])
    height_of = {"price": 0.5, "score": 0.16, "rank": 0.11, "rs": 0.14, "rsi": 0.11, "macd": 0.11}
    titles = {
        "price": "<b>Price · Bullish (Buy) ▲ / Bearish (Sell) ▼ signals · shaded = Hold</b>",
        "score": "Strategy score (purple) = 0.5 × Technical (grey) + 0.5 × Strength vs sector/SPY (teal)",
        "rank": f"Rank (1 = best · dashed = rank 10 · blue = held) · picks skip stocks whose sector already has {SECTOR_MAX}, so a held stock can rank below 10",
        "rs": "Strength vs sector ETF / SPY · price ratio rebased to 100 (rising = beating it)",
        "rsi": "RSI", "macd": "MACD",
    }
    heights = [height_of[x] for x in panels]
    row_of = {x: i + 1 for i, x in enumerate(panels)}
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[h / sum(heights) for h in heights], subplot_titles=[titles[x] for x in panels])
    fig.update_annotations(font=dict(size=12, color='#374151', family='Arial, sans-serif'), yshift=4)

    # Price line (hover shows the day's status), earnings dates, moving averages
    status_txt = []
    for w, r, s in zip(chart['Strategy_Weight'], chart['Strategy_Rank'], chart['Strategy_Score']):
        sig = daily_status(w, s)
        bits = [sig] + ([f"weight {w * 100:.1f}%"] if sig == "Hold" else []) \
            + ([f"rank #{r:.0f}"] if pd.notna(r) else []) + ([f"score {s:.1f}"] if pd.notna(s) else [])
        status_txt.append(" · ".join(bits))
    fig.add_trace(go.Scatter(x=chart.index, y=chart['Close'], name='Close', line=dict(color='#27ae60', width=2), mode='lines',
                             customdata=status_txt, hovertemplate='<b>Close</b> $%{y:.2f}<br>%{customdata}<extra></extra>'), row=1, col=1)
    earnings = chart[chart['is_earnings_date'] == 1]
    fig.add_trace(go.Scatter(x=earnings.index, y=earnings['Close'], name='Earnings date', mode='markers',
                             marker=dict(symbol='circle', size=9, color='#f97316'),
                             hovertemplate='<b>Earnings</b> %{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>'), row=1, col=1)
    for ma, color in zip(MA_COLS, ['#ffd700', '#e74c3c', '#3498db', '#8b4513', '#808080']):
        name = ma.upper().replace('_', ' ')
        fig.add_trace(go.Scatter(x=chart.index, y=chart[ma], name=name, line=dict(color=color, width=1), mode='lines',
                                 hovertemplate=f'<b>{name}</b> $%{{y:.2f}}<extra></extra>'), row=1, col=1)
    if periods:
        fig.add_trace(go.Scatter(x=[x_start], y=[None], mode="markers", name="Hold (shaded period)", hoverinfo="skip",
                                 marker=dict(symbol="square", size=12, color="rgba(37,99,235,0.25)")), row=1, col=1)

    # Entry / exit markers (filled = executed at that open, hollow = pending next open)
    shown = events[events['Fill'].fillna(x_end) >= x_start] if len(events) else events
    for kind, marker, color, name in (("entry", "triangle-up", "#15803d", "Bullish (Buy) · bought at next open"),
                                      ("exit", "triangle-down", "#b91c1c", "Bearish (Sell) · sold at next open")):
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
            x=regime_off.index, y=[top_y] * len(regime_off), mode='markers', name='Market filter OFF (positions halved)',
            marker=dict(symbol='diamond', size=8, color='#d97706'),
            hovertemplate='<b>Market filter OFF</b> %{x|%b %d}: QQQ below its 200-day average,<br>all positions halved that week<extra></extra>'),
            row=1, col=1)

    # While held: entry price and the 3×ATR stop (reference only; stops were tested as C8 and not adopted)
    hold = row_for(HOLDINGS_CSV, ticker) if (num(chart['Strategy_Weight'].iloc[-1]) or 0) > 0 else None
    if hold is not None and pd.notna(hold['ATR_Stop']):
        fig.add_hline(y=float(hold['ATR_Stop']), line=dict(color='#b91c1c', width=1, dash='dash'), row=1, col=1,
                      annotation_text=f"3×ATR stop ${hold['ATR_Stop']:,.2f} ({hold['Dist_to_Stop_%']:.1f}% below close) · reference only",
                      annotation_position="bottom left", annotation_font=dict(size=10, color='#b91c1c'))
        if pd.notna(hold['Entry_Price']):
            fig.add_hline(y=float(hold['Entry_Price']), line=dict(color='#64748b', width=1, dash='dot'), row=1, col=1,
                          annotation_text=f"entry ${hold['Entry_Price']:,.2f}", annotation_position="top left",
                          annotation_font=dict(size=10, color='#64748b'))

    # Next earnings date (extends the x-axis when it is within ~2 months)
    x_right = x_end
    ned = last_next_earnings([ticker])['Next ED'].iloc[0]
    if ned and pd.Timestamp(ned) - x_end <= pd.Timedelta(days=62):
        ned = pd.Timestamp(ned)
        x_right = max(x_end, ned) + pd.Timedelta(days=4)
        fig.add_vline(x=ned, line=dict(color='#f97316', width=1.2, dash='dash'), row="all", col=1)
        fig.add_annotation(x=ned, y=1, xref="x", yref="y domain", text=f"next earnings {ned:%b %d}", showarrow=False,
                           yanchor="bottom", xanchor="right", font=dict(size=10, color='#c2410c'))

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

    if "score" in row_of:  # score panel + rank panel
        r = row_of["score"]
        for col, name, color, width, dash in (("Technical_Score", "Technical", "#9ca3af", 1, "dot"),
                                              ("RS_Score", "Strength vs sector/SPY", "#0d9488", 1.2, "solid"),
                                              ("Strategy_Score", "Strategy score", "#7c3aed", 2, "solid")):
            fig.add_trace(go.Scatter(x=chart.index, y=chart[col], name=name, mode='lines', line=dict(color=color, width=width, dash=dash),
                                     showlegend=False, hovertemplate=f'<b>{name}</b> %{{y:.1f}}<extra></extra>'), row=r, col=1)
        fig.add_hline(y=0, line=dict(color='rgba(100,116,139,0.5)', width=1, dash='dot'), row=r, col=1)
        r = row_of["rank"]
        n_day = p.df[p.df['Symbol'] != 'QQQ'].groupby('Date')['Strategy_Score'].count().reindex(chart.index)
        slots = slot_series(ticker, chart.index)
        rank_txt = [f"#{rk:.0f} of {n:.0f}" + (f" · portfolio slot {sl:.0f} of 10" if pd.notna(sl) else " · not in portfolio")
                    if pd.notna(rk) else "not ranked" for rk, n, sl in zip(chart['Strategy_Rank'], n_day, slots)]
        fig.add_trace(go.Scatter(x=chart.index, y=chart['Strategy_Rank'], name='Rank', mode='lines', line=dict(color='#334155', width=1.5),
                                 customdata=rank_txt, showlegend=False, hovertemplate='<b>Rank</b> %{customdata}<extra></extra>'), row=r, col=1)
        held_days = chart.index[slots.notna().to_numpy()]
        if len(held_days):
            fig.add_trace(go.Scatter(x=held_days, y=chart.loc[held_days, 'Strategy_Rank'], name='Held (portfolio slot)', mode='markers',
                                     marker=dict(size=4, color='#2563eb'), showlegend=False, hoverinfo='skip'), row=r, col=1)
        fig.add_hline(y=10.5, line=dict(color='#15803d', width=1, dash='dash'), row=r, col=1)
        fig.update_yaxes(autorange="reversed", row=r, col=1)
    if "rs" in row_of:
        r = row_of["rs"]
        for (label, series), color in zip(rs_lines.items(), ("#0d9488", "#6366f1")):
            fig.add_trace(go.Scatter(x=series.index, y=series, name=label, mode='lines', line=dict(color=color, width=1.5),
                                     showlegend=False, hovertemplate=f'<b>{label}</b> %{{y:.1f}}<extra></extra>'), row=r, col=1)
        fig.add_hline(y=100, line=dict(color='rgba(100,116,139,0.5)', width=1, dash='dot'), row=r, col=1)
    if "rsi" in row_of:
        r = row_of["rsi"]
        fig.add_trace(go.Scatter(x=chart.index, y=chart['RSI'], name='RSI', line=dict(color='#ff7f0e', width=1.5), mode='lines',
                                 showlegend=False, hovertemplate='<b>RSI</b> %{y:.1f}<extra></extra>'), row=r, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(200, 0, 0, 0.3)", row=r, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0, 200, 0, 0.3)", row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)
        r = row_of["macd"]
        fig.add_trace(go.Scatter(x=chart.index, y=chart['macd'], name='MACD', line=dict(color='#d62728', width=1.5), mode='lines',
                                 showlegend=False, hovertemplate='<b>MACD</b> %{y:.3f}<extra></extra>'), row=r, col=1)
        fig.add_trace(go.Scatter(x=chart.index, y=chart['MACD Signal'], name='MACD signal', line=dict(color='#1f77b4', width=1.5, dash='dash'),
                                 mode='lines', showlegend=False, hovertemplate='<b>Signal</b> %{y:.3f}<extra></extra>'), row=r, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(128, 128, 128, 0.4)", row=r, col=1)

    # Held periods shaded on all panels (added last, with exclude_empty_subplots=False, or plotly drops them)
    for start, end in periods:
        if end >= x_start:
            fig.add_vrect(x0=max(start, x_start), x1=end, fillcolor="rgba(37,99,235,0.08)", line_width=0, layer="below",
                          row="all", col=1, exclude_empty_subplots=False)

    grid = dict(showgrid=True, gridcolor='rgba(200, 198, 195, 0.35)', showline=True, linecolor='rgba(200, 198, 195, 0.4)',
                tickfont=dict(size=10, color='#6b7280'), zeroline=False)
    fig.update_layout(
        height=int(470 + 140 * (len(panels) - 1)), hovermode='x unified', margin=dict(l=50, r=30, t=90, b=40),
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', dragmode=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, font=dict(size=10, color='#374151'),
                    bgcolor='rgba(255, 255, 255, 0.95)', bordercolor='#e2e8f0', borderwidth=1),
        font=dict(family="Arial, sans-serif", size=11, color='#374151'),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", font_size=11, font_family="Arial, sans-serif"))
    fig.update_xaxes(type="date", range=[x_start, x_right], showspikes=True, spikemode="across", spikethickness=1, spikecolor="#6b7280",
                     tickformat='%b %Y', **(grid | dict(showgrid=False)))
    fig.update_yaxes(**grid)
    fig.update_yaxes(title_text="Price ($)", tickformat='$,.0f', row=1, col=1)
    return fig, chart, events, x_start, x_end


def render_stock_chart(p, ticker, tdata):
    """Chart options + chart + a one-line legend. Returns what the 'More about' expander needs."""
    has_strategy = bool(tdata['Strategy_Score'].notna().any())
    o = st.columns(4)
    show_strategy = o[0].checkbox("Strategy score & rank", value=True, key="show_strategy", disabled=not has_strategy)
    show_rs = o[1].checkbox("Relative strength", value=True, key="show_rs")
    show_classic = o[2].checkbox("RSI & MACD", value=False, key="show_classic")
    show_legacy = o[3].checkbox("Legacy signals (old rules)", value=False, key="show_legacy",
                                help="The old BUY/SELL streak bars and flip lines from final_trade (pre-v3 rules). Not the live strategy.")
    fig, chart, events, x_start, x_end = build_price_chart(p, ticker, tdata, show_strategy and has_strategy, show_rs,
                                                           show_classic, show_legacy)
    st.plotly_chart(fig, width="stretch", config={
        'displaylogo': False, 'scrollZoom': False, 'doubleClick': 'reset',
        'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'zoomIn2d', 'zoomOut2d']})
    st.caption("▲ Bullish (Buy) · ▼ Bearish (Sell) · blue shading = Hold · hollow marker = pending · ◆ market filter OFF. "
               "Hover a marker for the reason.")
    return has_strategy, chart, events, x_start, x_end


def render_stock_more(p, ticker, tdata, has_strategy, chart, events, x_start, x_end):
    """Expander with the secondary stock details: moving averages, fundamentals/news context, chart guide, per-stock backtest."""
    latest = tdata.nlargest(1, 'Date').iloc[0]
    with st.expander(f"More about {ticker} (moving averages, fundamentals, news, per-stock backtest)", expanded=False):
        for col, ma in zip(st.columns(len(MA_COLS)), MA_COLS):
            col.metric(ma.upper().replace('_', ' '), fmt(num(latest[ma]), ",.2f", "$"))

        # Context: latest-day values only, NOT part of the backtested rules
        company_df = load_company()
        comp = company_metrics(ticker, company_df)
        close, fv = num(latest['Close']), comp.get('fair_value')
        upside = (fv / close - 1) * 100 if fv and close else None
        sentiment = num(latest['SentimentScore'])
        comp_row = company_df[company_df['Symbol'].astype(str).str.upper() == ticker] if 'Symbol' in company_df else pd.DataFrame()
        fiscal = pd.to_datetime(comp_row['FiscalDateEnding'].iloc[0], errors='coerce') \
            if len(comp_row) and 'FiscalDateEnding' in comp_row else pd.NaT
        news = load_news()
        news_dates = pd.to_datetime(news.loc[news['symbol'] == ticker, 'date'], utc=True, format='mixed', errors='coerce')
        last_news = news_dates.max() if len(news_dates) else pd.NaT
        context = "".join([
            stat_html("Balance sheet", fmt(num(latest['Fundamental_Weight']), ".2f")),
            stat_html("Sentiment", fmt(sentiment, ".2f"), sign_color(sentiment)),
            stat_html("Fair value", fmt(fv, ",.2f", "$")),
            stat_html("Upside", fmt(upside, "+.1f", suffix="%"), sign_color(upside)),
            stat_html("P/E", fmt(comp.get('pe_ratio'), ".1f")),
            stat_html("P/B", fmt(comp.get('pb_ratio'), ".2f")),
            stat_html("Rev YoY", fmt(comp.get('revenue_growth_yoy'), ".1f", suffix="%")),
            stat_html("ROE", fmt(comp.get('roe'), ".1f", suffix="%")),
            stat_html("Net margin", fmt(comp.get('net_margin'), ".1f", suffix="%")),
            stat_html("Debt/Eq", fmt(comp.get('debt_to_equity'), ".2f")),
        ])
        show_html(f'<div class="sa-stats" style="justify-content:flex-start;margin:6px 0;">{context}</div>')
        st.caption("Context only (latest day, not part of the backtested rules) · fundamentals: "
                   + (f"quarter ending {fiscal:%Y-%m-%d}" if pd.notna(fiscal) else "none on file")
                   + (f" · newest relevant news {last_news:%b %d}" if pd.notna(last_news) else " · no relevant news in the last 10 days"))

        notes = ["Chart guide: green ▲ = the rules put the stock in the top-10 portfolio (bought at the next open); red ▼ = it "
                 "dropped out (sold at the next open); blue shading = held. Rank panel: blue dots = held days; hover shows the "
                 "rank among all ranked stocks and the portfolio slot (1–10 = position among the picks). The picks skip score ≤ 0 "
                 f"and stocks whose sector already has {SECTOR_MAX}, which is why a held stock can sit below the rank-10 line."]
        if has_strategy:
            n_entries = int(((events['Kind'] == 'entry') & (events['Fill'].fillna(x_end) >= x_start)).sum()) if len(events) else 0
            held_share = (chart['Strategy_Weight'].fillna(0) > 0).mean() * 100
            notes.append(f"Last 12 months: held on {held_share:.0f}% of sessions, "
                         f"{n_entries} Bullish (Buy) signal{'' if n_entries == 1 else 's'}.")
        ps = row_for(PER_STOCK_CSV, ticker)
        if ps is not None and pd.notna(ps.get("First bar")):
            trades = int(ps["Closed trades"])
            notes.append(
                f"Backtest of the live rules from {ps['First bar']} (next-open fills, 0.1%/side): {trades} closed trade"
                f"{'' if trades == 1 else 's'}"
                + (f", win rate {ps['Win rate %']:.0f}%, median trade {ps['Median trade %']:+.1f}%, median hold "
                   f"{ps['Median hold (sessions)']:.0f} sessions" if trades else "")
                + f"; held on {ps['Held % of sessions']:.0f}% of sessions; buy & hold of the stock {ps['Buy & hold %']:+.0f}%.")
        st.caption(" ".join(notes))


# =====================================================================================================================
# 12. Details tab
# =====================================================================================================================
def render_all_signals(p):
    """Every stock with its signal (the decisions in force) in one table; click a row to open the stock."""
    st.caption(f"Decisions in force (last decision {p.off_date:%a %b %d}).")
    board = p.board_off.copy()
    board.insert(3, "Rank change", board["Symbol"].map(lambda s: p.rank_change.get(s, (np.nan,))[0]))
    counts = board["Signal"].value_counts()
    filters = {"Portfolio & changes": ["Bullish (Buy)", "Hold", "Bearish (Sell)"],
               "Watch list (Neutral)": ["Neutral", "Neutral (sector cap)"],
               "All stocks": list(SIGNAL_COLOR)}
    show = st.radio("Show", list(filters), horizontal=True, key="signals_filter", label_visibility="collapsed")
    part = board[board["Signal"].isin(filters[show])].reset_index(drop=True)
    st.caption(f"Bullish (Buy) {counts.get('Bullish (Buy)', 0)} · Hold {counts.get('Hold', 0)} · "
               f"Bearish (Sell) {counts.get('Bearish (Sell)', 0)} · Neutral {counts.get('Neutral', 0) + counts.get('Neutral (sector cap)', 0)} · "
               f"Bearish (score below 0) {counts.get('Bearish', 0)}. Rank = position among all ranked stocks (Rank change: + = moved up "
               "since the last decision day); Portfolio slot = position among the 10 picks. Click a row to open the stock on the Dashboard tab.")
    event = st.dataframe(part.round({"Rank": 0, "Score": 1, "Portfolio weight %": 1}), hide_index=True, width="stretch",
                         on_select="rerun", selection_mode="single-row", key=f"sig_tbl_{show}",
                         column_config={"Why": st.column_config.TextColumn("Why", width="large")})
    open_symbol(part, event, "signals")


def render_rank_history(p, ticker):
    """Symbol x last-20 decision-day rank table (HTML, the selected stock highlighted)."""
    pivot = rank_pivot(p.df, p.mtime, 20)
    date_cols = [c for c in pivot.columns if c not in ("Symbol", "Trend")]
    pivot = pivot.merge(last_next_earnings(pivot["Symbol"].tolist()), on="Symbol", how="left")
    short = {"Bullish (Buy)": "Buy", "Hold": "Hold", "Bearish (Sell)": "Sell", "Neutral": "Neutral",
             "Neutral (sector cap)": "Neutral (cap)", "Bearish": "Bearish", "Not ranked": ""}

    def signal_text(sym):
        return short.get(p.sig_off.get(sym, "Not ranked"), "")

    pivot["Signal"] = pivot["Symbol"].map(signal_text)
    pivot["Slot"] = pivot["Symbol"].map(p.slot_off).fillna("—")
    table = pivot[["Symbol", "Signal", "Slot", "Last ED", "Next ED", "Trend"] + date_cols]
    st.caption("Rank 1 = highest Strategy Score, recomputed on Mon/Wed/Fri decision days among all ranked stocks "
               f"({int(p.latest['Strategy_Rank'].notna().sum())} today; QQQ is a benchmark). Slot = position among the 10 picks. "
               "Signal = the decision in force. Trend = rank momentum "
               "(Bullish = 4+ better decision days in a row, Bearish = 3+ worse decision days), information only. "
               "Earnings dates: last ≤10 days red, next ≤7 days green.")
    today = pd.Timestamp.now().normalize()
    GREEN, RED, AMBER, TEXT = "#1a7f37", "#c41e3a", "#9e6a03", "#374151"

    def streak_flags(row):
        """Mark rank cells inside a 4+ step improving run (Bullish) or a 3+ step worsening run (Bearish)."""
        cols = [c for c in reversed(date_cols) if pd.notna(row[c])]
        ranks = [int(row[c]) for c in cols]
        flags, n, i = {}, len(ranks), 0
        while i < n - 1:
            j = i
            while j + 1 < n and ranks[j + 1] < ranks[j]:
                j += 1
            if j - i >= 4:
                flags.update({cols[k]: "Bullish" for k in range(i, j + 1)})
                i = j + 1
                continue
            j = i
            while j + 1 < n and ranks[j + 1] > ranks[j]:
                j += 1
            if j - i >= 3:
                for k in range(i, j + 1):
                    flags.setdefault(cols[k], "Bearish")
                i = j + 1
                continue
            i += 1
        return flags

    def td(text, color=TEXT, weight="400", extra=""):
        return f'<td style="{extra}color:{color};font-weight:{weight};text-align:center;padding:6px 8px;white-space:nowrap;">{text}</td>'

    def row_html(row):
        sel_bg = "background-color:#e8f0fe;" if row['Symbol'] == ticker else ""
        cells = [f'<td style="font-weight:600;text-align:left;padding:6px 8px;position:sticky;left:0;z-index:1;'
                 f'background:{"#e8f0fe" if sel_bg else "#ffffff"};">{row["Symbol"]}</td>',
                 td(esc(row['Signal']), {"Buy": GREEN, "Hold": "#2563eb", "Sell": RED, "Bearish": RED}.get(
                     row['Signal'].split(" →")[0], AMBER if row['Signal'] else TEXT), "700", sel_bg),
                 td(esc(str(row['Slot'])), "#1d4ed8" if row['Slot'] != "—" else TEXT, "700" if row['Slot'] != "—" else "400", sel_bg)]
        for col, lo, hi in (("Last ED", -10, 0), ("Next ED", 0, 7)):
            d = pd.to_datetime(row[col], errors="coerce")
            hit = pd.notna(d) and today + pd.Timedelta(days=lo) <= d <= today + pd.Timedelta(days=hi)
            cells.append(td(row[col] or "", (RED if col == "Last ED" else GREEN) if hit else TEXT, "700" if hit else "400", sel_bg))
        cells.append(td(row['Trend'], {"Bullish": GREEN, "Bearish": RED}.get(row['Trend'], AMBER),
                        "700" if row['Trend'] != "Hold" else "600", sel_bg))
        flags = streak_flags(row)
        for c in date_cols:
            flag = flags.get(c)
            cells.append(td("" if pd.isna(row[c]) else int(row[c]), {"Bullish": GREEN, "Bearish": RED}.get(flag, TEXT),
                            "700" if flag else "400", sel_bg))
        return f"<tr>{''.join(cells)}</tr>"

    thead = "".join(f'<th style="position:sticky;top:0;background:#f1f5f9;padding:8px;text-align:center;font-size:0.8rem;'
                    f'color:#374151;border-bottom:1px solid #e2e8f0;white-space:nowrap;">{c}</th>' for c in table.columns)
    body = "".join(row_html(row) for _, row in table.iterrows())
    show_html(f"""
        <div style="max-height:520px;overflow:auto;border:1px solid #e2e8f0;border-radius:12px;background:#ffffff;">
          <table style="border-collapse:collapse;width:100%;font-size:0.85rem;font-family:Arial,sans-serif;">
            <thead><tr>{thead}</tr></thead>
            <tbody>{body}</tbody>
          </table>
        </div>""")


def render_rules_and_changes(p):
    """Decision dates, the full rules text and each change at the latest decision with the rule that decided it."""
    reb_days = p.df.loc[p.df["Rebalance_Day"] == 1, "Date"]
    mw = p.midweek[p.midweek["Event"] == "mid-week check"] if p.midweek is not None else None
    if mw is not None and len(mw):
        last_day = mw["Event_Date"].iloc[-1]
        acts = set(mw.loc[mw["Event_Date"] == last_day, "Action"])
        mw_val = (f"{pd.Timestamp(last_day):%a %b %d} · "
                  + (" + ".join(x for x, k in (("swap", "SWAP"), ("exit", "SELL")) if k in acts) or "no trade"))
    else:
        mw_val = "none yet this week" if MIDWEEK else "off"
    c = st.columns(3)
    c[0].metric("Last weekly rebalance", f"{reb_days.max():%a %b %d}" if not reb_days.empty else "—")
    c[1].metric("Last mid-week check", mw_val)
    c[2].metric("Next decision", f"{p.next_kind} · {p.next_dec:%a %b %d}" if p.next_dec is not None else "—")
    st.markdown(rules_text())

    changes = read_report_csv(CHANGES_CSV)
    if changes is None or changes.empty:
        st.caption("Run `python run_all.py` to list the portfolio changes.")
        return
    st.markdown("**What changed at the latest decision**")
    sub = changes[changes["View"] == "last rebalance"].copy()
    earn = sub["Reason"].astype(str).str.startswith("earnings in")
    if not st.checkbox("Show Neutral (sector cap) names too", value=False, key="changes_all"):
        sub, earn = sub[(sub["Status"] != "not selected") | earn], earn[(sub["Status"] != "not selected") | earn]
    sub[["Old_Weight", "New_Weight"]] = (sub[["Old_Weight", "New_Weight"]] * 100).round(1)
    order = {"add": 0, "drop": 1, "hold": 2, "not selected": 3}
    sub = sub.sort_values(["Status", "Rank"], key=lambda s: s.map(order) if s.name == "Status" else s)
    sub["Signal"] = sub["Status"].map({"add": "Bullish (Buy)", "hold": "Hold", "drop": "Bearish (Sell)",
                                       "not selected": "Neutral (sector cap)"}).where(~earn, "Not bought (earnings)")
    sub["Why"] = [plain_reason(g, rs, rk, sc) for g, rs, rk, sc in zip(sub["Signal"], sub["Reason"], sub["Rank"], sub["Score"])]
    st.dataframe(sub[["Symbol", "Signal", "Why", "Rank", "Score", "Sector", "Old_Weight", "New_Weight"]]
                 .rename(columns={"Old_Weight": "Old portfolio weight %", "New_Weight": "New portfolio weight %"}).round(2),
                 width="stretch", hide_index=True)
    st.caption(f"Decision date {sub['Date'].iloc[0] if len(sub) else '—'} · filled at the next open.")


def render_holdings_and_tracking():
    """Per-holding risk / P&L since entry."""
    holdings = read_report_csv(HOLDINGS_CSV)
    if holdings is not None and not holdings.empty:
        h = holdings.copy()
        h["Weight"] = (h["Weight"] * 100).round(1)
        st.markdown("**Holdings · risk & P&L since entry**")
        st.dataframe(h.rename(columns={"Weight": "Portfolio weight %", "PnL_%": "P&L %", "Days_Held": "Days held",
                                       "Vol_63d_%": "Vol 63d %", "ATR_Stop": "ATR stop (3×)", "Dist_to_Stop_%": "To stop %"}).round(2),
                     width="stretch", hide_index=True)
        st.caption("Entry = next open after the decision that added the stock (before costs). The 3×ATR stop is a risk "
                   "reference only; stops are not part of the trading rules.")


def parse_positions(text):
    """'SYMBOL,SHARES' lines -> ({symbol: shares}, ignored_lines)."""
    positions, bad = {}, []
    for line in text.splitlines():
        parts = [x.strip() for x in line.replace(";", ",").split(",")]
        if len(parts) == 2 and parts[0]:
            try:
                positions[parts[0].upper()] = float(parts[1])
            except ValueError:
                bad.append(line)
    return positions, bad


def render_order_preview():
    """Share counts to reach the target weights (nothing is sent anywhere)."""
    try:
        from paper_trade import plan_orders
        c = st.columns(2)
        acct = c[0].number_input("Account size ($)", min_value=100.0, value=25_000.0, step=1_000.0, key="acct_size")
        src = c[1].selectbox("Target weights", ["auto", "current", "provisional", "midweek"], key="order_target",
                             help="current = decisions in force; provisional = if fully rebalanced at the latest close; "
                                  "midweek = only the swap(s) and rank exits (sell only) of today's Mon/Wed check; auto = "
                                  "provisional on a Friday rebalance day, midweek on a Mon/Wed check with a swap or exit, otherwise current")
        text = st.text_area("Current positions (optional, one per line: SYMBOL,SHARES)", key="order_positions", height=80)
        positions, bad = parse_positions(text)
        for line in bad:
            st.warning(f"Ignored line: {line}")
        orders, meta, _targets = plan_orders(src, acct, positions, PICKS_CSV, SIGNAL_CSV, MIDWEEK_CSV)
        if meta["source"] == "midweek":
            for msg in meta["swaps"]["Message"]:
                st.info(msg)
        st.caption(f"Targets: {meta['source']} weights as of {meta['as_of']} (last weekly rebalance {meta['last_rebalance']}, "
                   f"last decision {meta['last_decision']}), {meta['invested']:.0%} invested · priced at the latest close · whole shares")
        st.dataframe(orders.round(2), width="stretch", hide_index=True)
        st.caption("Preview only. To send these to Alpaca PAPER run `python paper_trade.py --submit --paper` yourself.")
    except Exception as e:
        st.warning(f"Order preview unavailable: {e}")


BACKTEST_COLS = ["Strategy", "Start", "End", "Total Return %", "CAGR %", "Sharpe", "Max DD %", "Trades", "Win rate %",
                 "Median trade %", "Median hold (sessions)"]


def render_data_and_settings(p):
    """Data freshness table + the holdings-alert source setting."""
    st.dataframe(p.freshness, width="stretch", hide_index=True)
    stale = p.freshness[p.freshness["Status"].str.startswith("⚠️")]
    if not stale.empty:
        st.warning("Stale or missing: " + ", ".join(stale["File"]) + " — run `python run_all.py` (see docs/README.md).")
    else:
        st.success("All report files are within their expected refresh window.")
    if os.path.exists(POSITIONS_CSV):
        st.radio("Top alert based on", ["Your positions file", "Strategy holdings"], horizontal=True, key="alert_source",
                 help="my_positions.csv (Symbol,Shares) vs the strategy's own holdings")


def render_details(p, ticker):
    with st.expander("All signals (every stock)", expanded=True):
        render_all_signals(p)
    with st.expander("Rank history · last 20 decision days", expanded=False):
        render_rank_history(p, ticker)
    with st.expander("Rules and latest decisions", expanded=False):
        render_rules_and_changes(p)
    with st.expander("Holdings risk", expanded=False):
        render_holdings_and_tracking()
    with st.expander("Order preview for a paper account (nothing is sent)", expanded=False):
        render_order_preview()
    with st.expander("Data freshness and settings", expanded=False):
        render_data_and_settings(p)


# =====================================================================================================================
# 13. Main
# =====================================================================================================================
def main():
    with st.spinner("Loading data..."):
        p = build_page()
    # ?symbol=X opens that stock (the links in the alert, the tables and the Telegram messages use this)
    q_symbol = str(st.query_params.get("symbol", "")).strip().upper()
    jumped = q_symbol in p.options
    if jumped:
        st.session_state["_pending_ticker"] = q_symbol
        del st.query_params["symbol"]

    render_top_bar(p)
    render_alert()
    tab_main, tab_details = st.tabs(["📈 Dashboard", "🔎 Details"])
    with tab_main:
        render_summary(p)
        ticker, tdata = render_stock_picker(p, jumped)
        render_stock_header(p, ticker, tdata)
        chart_info = render_stock_chart(p, ticker, tdata)
        render_stock_more(p, ticker, tdata, *chart_info)
    with tab_details:
        render_details(p, ticker)


main()
