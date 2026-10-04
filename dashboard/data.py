"""Data loading (cached; every cache key includes the file's modification time so new data shows up at once) and the
display-only live quote (yfinance; never touches strategy, signals, picks, backtests, or orders)."""
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.settings import (APP_FILES, BENCH_CSV, COMPANY_XLSX, CT, DECISIONS_CSV, EARNINGS_CSV, FRESHNESS,
    MIDWEEK_CSV, NEWS_CSV, REPORTS, SIGNAL_CSV)


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
