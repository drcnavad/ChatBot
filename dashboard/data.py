"""Data loading and display-only live prices.

Files: every cache key includes the file's modification time, so data a scheduled job writes shows up on the next
read (no page reload). Live: Alpaca snapshots / bars (SIP, IEX on any SIP error) through dashboard.alpaca_data;
yfinance only for what Alpaca does not carry. Nothing here feeds signals, picks, backtests, or orders."""
import os
from datetime import datetime

import pandas as pd
import db  # noqa: E402  tables live in SQLite (db.py), not CSV files
import streamlit as st

from dashboard.alpaca_data import refresh_every, refresh_slot  # noqa: F401  (re-exported)
from dashboard.settings import (APP_FILES, BAR_CACHES, BARS_TABLE, BENCH_CSV, COMPANY_XLSX, CT, DECISIONS_CSV, EARNINGS_CSV, FRESHNESS,
    NEWS_CSV, REPORTS, SIGNAL_CSV)

try:
    import yfinance as _yf
    _YF_OK = True
except ImportError:  # graceful degradation: dashboard works without yfinance quotes
    _yf, _YF_OK = None, False

NY = "America/New_York"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# ---- files (mtime-keyed caches) ----

def mtime(path):
    return db.mtime(path) if db.exists(path) else 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def _read(path, mtime, dates=None, excel_sheet=None):
    if excel_sheet:
        return pd.read_excel(path, sheet_name=excel_sheet, engine="openpyxl")
    return db.read_csv(path, parse_dates=list(dates) if dates else None)

def read_report_csv(path, dates=None):
    """Cached CSV read (re-read when the file changes); None when the file is missing. Treat the result as read-only."""
    return _read(path, mtime(path), dates) if db.exists(path) else None

@st.cache_data(ttl=3600)
def load_signals(mtime: float):
    """Reports/signal_analysis.csv: one row per symbol and day (prices, indicators, strategy columns)."""
    return db.read_csv(SIGNAL_CSV, parse_dates=['Date'])

def signals():
    """The current signals frame (cached per file version)."""
    return load_signals(mtime(SIGNAL_CSV))

@st.cache_data(ttl=3600)
def latest_rows(_df, mtime: float):
    """Most recent row per symbol (the leading underscore stops Streamlit hashing the whole frame)."""
    return _df.sort_values('Date', ascending=False).drop_duplicates(subset='Symbol', keep='first')

def load_decisions():
    """Reports/strategy_decisions.csv: the engine's log of every weekly / mid-week decision (with reasons)."""
    return read_report_csv(DECISIONS_CSV, dates=("Date",))

def load_benchmarks():
    b = read_report_csv(BENCH_CSV, dates=("Date",))
    return None if b is None else b.set_index("Date")

def row_for(path, symbol):
    """First row of a Reports CSV for one symbol, or None."""
    t = read_report_csv(path)
    row = None if t is None else t[t["Symbol"] == symbol]
    return None if row is None or row.empty else row.iloc[0]

def data_freshness(latest_bar):
    """One row per Reports file: last update (CT), age and a stale flag."""
    now = datetime.now(tz=CT)
    rows = []
    for name, (what, max_days) in FRESHNESS.items():
        path = os.path.join(REPORTS, name)
        if not db.exists(path):
            status = "⚠️ missing" if name in APP_FILES else "— not present (local pipeline file)"
            rows.append({"File": name, "Contents": what, "Last update (CT)": "—", "Latest data": "—", "Age": "—", "Status": status})
            continue
        ts = datetime.fromtimestamp(db.mtime(path), tz=CT)
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

def load_earnings():
    ed = read_report_csv(EARNINGS_CSV)
    if ed is None:
        return pd.DataFrame(columns=["Symbol", "Earnings Date", "Time"])
    ed = ed.assign(Symbol=ed['Symbol'].astype(str).str.strip().str.upper(),
                   **{'Earnings Date': pd.to_datetime(ed['Earnings Date'], errors='coerce')})
    return ed.dropna(subset=['Earnings Date'])

def last_next_earnings(symbols):
    """Per symbol: most recent past and nearest upcoming earnings date (YYYY-MM-DD or '')."""
    ed = load_earnings()
    today = pd.Timestamp.now().normalize()
    day = lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) else ''
    rows = []
    for sym in symbols:
        dates = ed.loc[ed['Symbol'] == sym, 'Earnings Date']
        rows.append({'Symbol': sym, 'Last ED': day(dates[dates <= today].max()), 'Next ED': day(dates[dates >= today].min())})
    return pd.DataFrame(rows, columns=['Symbol', 'Last ED', 'Next ED'])

def load_company():
    """Latest-quarter fundamentals workbook; empty frame when missing."""
    if not db.exists(COMPANY_XLSX):
        return pd.DataFrame(columns=["Symbol"])
    return _read(COMPANY_XLSX, mtime(COMPANY_XLSX), excel_sheet="2_Latest_Quarter_Complete")

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

@st.cache_data(ttl=3600, show_spinner=False)
def _bars(mtime_key: float):
    """Daily OHLCV of every stock from SQLite bars_daily (single source of truth)."""
    b = db.load_bars()
    if not len(b):
        return pd.DataFrame(columns=["Symbol", "Date"] + OHLCV)
    b["Date"] = pd.to_datetime(b["Date"])
    return b

def ohlcv(symbol):
    """Display only (the stock chart): one stock's daily Open/High/Low/Close/Volume by Date."""
    b = _bars(db.bars_mtime())
    return b[b["Symbol"] == symbol].set_index("Date").sort_index()[OHLCV]

# ---- live (Alpaca) ----

def _live_off():
    return os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off"          # tests: never call Alpaca

def _snapshots(symbols):
    """{symbol: Alpaca snapshot} in one request (SIP -> IEX); {} when live data is off or neither feed answers."""
    if _live_off():
        return {}
    from alpaca.data.requests import StockSnapshotRequest
    from dashboard.alpaca_data import client, sip_then_iex
    try:
        snaps, _ = sip_then_iex(lambda f: client().get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=list(symbols), feed=f)))
        return snaps or {}
    except Exception:
        return {}

@st.cache_data(ttl=600, max_entries=200, show_spinner=False)
def _alpaca_last_trade(ticker: str, slot: int):
    """(price, trade time UTC) of the latest trade; None if Alpaca does not answer."""
    t = getattr(_snapshots((ticker,)).get(ticker), "latest_trade", None)
    if t is None:
        return None
    ts = pd.Timestamp(t.timestamp)
    return float(t.price), ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")

@st.cache_data(ttl=60, max_entries=64, show_spinner=False)
def today_bar(ticker: str, slot: int):
    """Today's daily bar so far from the Alpaca snapshot: (NY date, open, high, low, close, volume), close = the latest
    trade; None before today's first trade or when Alpaca does not answer. New every `slot` (5 s)."""
    try:
        snap = _snapshots((ticker,))[ticker]
        d = snap.daily_bar
        day = pd.Timestamp(d.timestamp).tz_convert(NY).tz_localize(None).normalize()
        px = float(snap.latest_trade.price) if snap.latest_trade else float(d.close)
        return day, float(d.open), max(float(d.high), px), min(float(d.low), px), px, float(d.volume)
    except Exception:
        return None

@st.cache_data(ttl=20, show_spinner=False)     # < the 30 s fragment timers, so every tick gets new prices
def index_quotes(symbols: tuple):
    """Display only: {symbol: (last price, day change %)} from one Alpaca snapshot request; change = last trade vs the
    previous session's close. {} when neither feed answers."""
    out = {}
    for s, x in _snapshots(symbols).items():
        try:
            px = float(x.latest_trade.price) if x.latest_trade else float(x.daily_bar.close)
            prev = float(x.previous_daily_bar.close) if x.previous_daily_bar else float("nan")
            out[s] = (px, (px / prev - 1) * 100 if prev == prev and prev else float("nan"))
        except Exception:
            continue
    return out

def live_quote(ticker: str):
    """Display-only latest price -> (price, time) or None: Alpaca latest trade, new every refresh slot (30 s in market
    hours, 5 min outside); yfinance only when Alpaca does not answer (e.g. non-US symbols)."""
    if not ticker:
        return None
    return _alpaca_last_trade(ticker, refresh_slot()) or _yf_live_quote(ticker)

def _yf_last(df):
    """Close column(s) of a yfinance download, one column per ticker."""
    c = df["Close"]
    return c if isinstance(c, pd.DataFrame) else c.to_frame()

@st.cache_data(ttl=120, show_spinner=False)
def _yf_live_quote(ticker: str):
    """Latest yfinance 1-minute close -> (price, bar time) or None (missing yfinance, a failed fetch or no bar)."""
    if not _YF_OK:
        return None
    try:
        c = _yf_last(_yf.download(ticker, period="1d", interval="1m", prepost=True, progress=False, auto_adjust=False))
        c = c.iloc[:, 0].dropna()
        return (float(c.iloc[-1]), pd.Timestamp(c.index[-1])) if len(c) else None
    except Exception:
        return None

@st.cache_data(ttl=120, show_spinner=False)
def etf_prices(symbols: tuple, start: str):
    """Display only (the Trading Account comparison): (daily closes by date from `start`, adjusted for dividends,
    one column per ETF; {ETF: latest price, pre/after hours included}). yfinance, two requests per call (cached 2 min);
    Reports/benchmark_prices.csv fills in QQQ / SPY closes and the latest daily close stands in for a missing quote."""
    closes, now = pd.DataFrame(), {}
    if _YF_OK:
        try:
            df = _yf.download(list(symbols), start=str(pd.Timestamp(start) - pd.Timedelta(days=7))[:10], interval="1d",
                              progress=False, auto_adjust=True)
            closes = _yf_last(df).dropna(how="all")
            closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
        except Exception:
            closes = pd.DataFrame()
        try:
            q = _yf_last(_yf.download(list(symbols), period="1d", interval="1m", prepost=True, progress=False, auto_adjust=False))
            now = {s: float(q[s].dropna().iloc[-1]) for s in q.columns if s in symbols and len(q[s].dropna())}
        except Exception:
            now = {}
    bench = load_benchmarks()
    for s in symbols:
        if (s not in closes or closes[s].dropna().empty) and bench is not None and s in bench:
            closes[s] = bench[s]
        if s not in now and s in closes and len(closes[s].dropna()):
            now[s] = float(closes[s].dropna().iloc[-1])
    return closes, now

def alpaca_bars(symbols, minutes, start):
    """Display only: {symbol: OHLCV DataFrame indexed by New York time without zone} from Alpaca bars since `start`
    (minutes per bar; None = daily), one request for all symbols; {} when neither feed answers."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from dashboard.alpaca_data import client, sip_then_iex
    frame = TimeFrame.Day if minutes is None else TimeFrame(minutes, TimeFrameUnit.Minute)
    try:
        data, _ = sip_then_iex(lambda f: client().get_stock_bars(StockBarsRequest(
            symbol_or_symbols=list(symbols), timeframe=frame, start=pd.Timestamp(start).to_pydatetime(), feed=f)).data)
    except Exception:
        data = None
    out = {}
    for sym, rows in (data or {}).items():
        df = pd.DataFrame([(r.timestamp, r.open, r.high, r.low, r.close, r.volume) for r in rows], columns=["t"] + OHLCV)
        if len(df):
            df.index = pd.to_datetime(df.pop("t"), utc=True).dt.tz_convert(NY).dt.tz_localize(None).rename(None)
            out[sym] = df.astype(float)
    return out

def last_sessions(df, n):
    """Regular-hours (9:30-16:00 New York) bars of the last n sessions that have any."""
    hm = df.index.hour * 60 + df.index.minute
    df = df[(hm >= 570) & (hm < 960)]
    days = sorted(set(df.index.normalize()))[-n:]
    return df[df.index.normalize().isin(days)]

def _days_back(n):
    return pd.Timestamp.now(tz=NY).normalize() - pd.Timedelta(days=n)

@st.cache_data(ttl=600, show_spinner=False)
def prior_week_close(symbols: tuple, week: str):
    """Display only: {symbol: last daily close before the week starting `week` (Monday, New York)} from one Alpaca
    daily-bar request, cached 10 min per week; the base of a live week-so-far return."""
    out, monday = {}, pd.Timestamp(week)
    for s, df in alpaca_bars(symbols, None, monday - pd.Timedelta(days=14)).items():
        c = df.loc[df.index.normalize() < monday, "Close"]
        if len(c):
            out[s] = float(c.iloc[-1])
    return out

@st.cache_data(ttl=60, show_spinner=False)
def intraday(symbols: tuple):
    """Display only (card / tile popups): {symbol: (epoch times, closes, previous close)} of the latest session's
    5-minute bars (today's, or the last session's while the market is closed). Cached 60 s; unreadable series left out."""
    if _live_off():
        return {}
    out = {}
    try:
        from concurrent.futures import ThreadPoolExecutor
        from dashboard.alpaca_data import market_open
        quotes = index_quotes(symbols)
        start = _days_back(0) if market_open() else _days_back(5)   # in the session: today's bars only (one page per chunk)
        chunks = [symbols[k:k + 25] for k in range(0, len(symbols), 25)]
        with ThreadPoolExecutor(4) as ex:                            # 25 symbols per request, 4 at a time
            bars = {s: df for part in ex.map(lambda c: alpaca_bars(c, 5, start), chunks) for s, df in part.items()}
        for s, df in bars.items():
            df = last_sessions(df, 1)
            last, chg = quotes.get(s, (None, float("nan")))
            if len(df) and chg == chg:
                out[s] = ([t.timestamp() for t in df.index.tz_localize(NY)], df["Close"].tolist(), last / (1 + chg / 100))
    except Exception:
        pass
    return out

@st.cache_data(ttl=60, show_spinner=False)
def yf_intraday(symbols: tuple):
    """Display only (header cards Alpaca does not carry, e.g. CL=F, BTC-USD): {symbol: ((price, day %), (epoch times,
    closes, previous close))} from yfinance 5-minute bars of the latest trading date. Cached 60 s; unreadable left out."""
    if _live_off() or not _YF_OK:
        return {}
    out = {}
    for s in symbols:
        try:
            bars = _yf.Ticker(s).history(period="5d", interval="5m")["Close"].dropna()
            daily = _yf.Ticker(s).history(period="7d", interval="1d")["Close"].dropna()
            if bars.empty or len(daily) < 2:
                continue
            today = bars[bars.index.date == bars.index[-1].date()]
            prev = float(daily[daily.index.date < bars.index[-1].date()].iloc[-1])
            px = float(today.iloc[-1])
            out[s] = ((px, (px / prev - 1) * 100), ([t.timestamp() for t in today.index], today.tolist(), prev))
        except Exception:
            continue
    return out

INTRADAY_VIEWS = {"1D": (5, 1), "5D": (15, 5)}      # range button -> (minutes per candle, sessions shown)

@st.cache_data(ttl=60, show_spinner=False)
def intraday_ohlcv(symbol, view):
    """Display only (stock chart 1D / 5D): OHLCV candles of the last session (5-minute) or the last 5 sessions
    (15-minute), regular hours, index = New York time without zone. Cached 60 s; empty when it cannot be read."""
    if _live_off() or view not in INTRADAY_VIEWS:
        return pd.DataFrame(columns=OHLCV)
    minutes, sessions = INTRADAY_VIEWS[view]
    try:
        df = alpaca_bars((symbol,), minutes, _days_back(4 + sessions * 2))
        return last_sessions(df[symbol], sessions) if symbol in df else pd.DataFrame(columns=OHLCV)
    except Exception:
        return pd.DataFrame(columns=OHLCV)
