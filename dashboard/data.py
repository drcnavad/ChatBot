"""Data loading (cached; every cache key includes the file's modification time so new data shows up at once) and the
display-only live quote (yfinance; never touches strategy, signals, picks, backtests, or orders)."""
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.settings import (APP_FILES, BAR_CACHES, BENCH_CSV, COMPANY_XLSX, CT, DECISIONS_CSV, EARNINGS_CSV, FRESHNESS,
    NEWS_CSV, REPORTS, SIGNAL_CSV)

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

def refresh_every():
    """Seconds between live-data refreshes: 30 in the regular session (Alpaca clock), 300 outside it."""
    from dashboard.page import market_open
    return 30 if market_open() else 300

def refresh_slot():
    """A number that changes every refresh_every() seconds: the cache key of every live (Alpaca) read."""
    import time
    return int(time.time() // refresh_every())

@st.cache_data(ttl=600, max_entries=200, show_spinner=False)
def _alpaca_last_trade(ticker: str, slot: int):
    """(price, trade time UTC) of the latest trade from Alpaca, SIP first and IEX on any SIP error; None if neither."""
    if _live_off():
        return None
    import backtest_engine as be
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockLatestTradeRequest
    client = be.market_data_client()
    for feed in ("sip", "iex"):
        try:
            t = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=ticker, feed=DataFeed(feed)))[ticker]
            return float(t.price), pd.Timestamp(t.timestamp).tz_convert("UTC") if pd.Timestamp(t.timestamp).tzinfo else pd.Timestamp(t.timestamp, tz="UTC")
        except Exception:
            continue
    return None

def live_quote(ticker: str):
    """Display-only latest price -> (price, time) or None: Alpaca latest trade (SIP -> IEX), new every refresh slot
    (30 s in market hours, 5 min outside); yfinance only when Alpaca does not answer (e.g. non-US symbols)."""
    if not ticker:
        return None
    try:
        q = _alpaca_last_trade(ticker, refresh_slot())
        if q is not None:
            return q
    except Exception:
        pass
    return _yf_live_quote(ticker)

@st.cache_data(ttl=120)
def _yf_live_quote(ticker: str):
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

def _closes(df):
    """The Close column(s) of a yfinance download as a DataFrame with one column per ticker."""
    c = df["Close"]
    return c if isinstance(c, pd.DataFrame) else c.to_frame()

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
            closes = _closes(df).dropna(how="all")
            closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
        except Exception:
            closes = pd.DataFrame()
        try:
            q = _closes(_yf.download(list(symbols), period="1d", interval="1m", prepost=True, progress=False,
                                     auto_adjust=False))
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

def _live_off():
    return os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off"          # tests: never call Alpaca

@st.cache_data(ttl=30, show_spinner=False)
def index_quotes(symbols: tuple):
    """Display only (header cards): {symbol: (last price, day change %)} from Alpaca snapshots, SIP first and IEX on any
    SIP error (the same feeds as paper_trade's order quotes). Change = last trade vs the previous session's close.
    Cached 30 s; {} when neither feed answers."""
    if _live_off():
        return {}
    import backtest_engine as be
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockSnapshotRequest
    try:
        client = be.market_data_client()
    except Exception:
        return {}
    for feed in ("sip", "iex"):
        try:
            snaps = client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=list(symbols), feed=DataFeed(feed)))
            out = {}
            for s, x in snaps.items():
                px = float(x.latest_trade.price) if x.latest_trade else float(x.daily_bar.close)
                prev = float(x.previous_daily_bar.close) if x.previous_daily_bar else float("nan")
                out[s] = (px, (px / prev - 1) * 100 if prev == prev and prev else float("nan"))
            return out
        except Exception:
            continue
    return {}

@st.cache_data(ttl=3600, show_spinner=False)
def _bars(mtimes: tuple):
    """Daily OHLCV of every stock: the long bar cache with the forward-test bars (fresher, from May 2026) on top."""
    parts = [pd.read_pickle(p) for p in BAR_CACHES if os.path.exists(p)]
    b = pd.concat(parts) if parts else pd.DataFrame(columns=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])
    b["Date"] = pd.to_datetime(b["Date"])
    return b.drop_duplicates(["Symbol", "Date"], keep="last")

def ohlcv(symbol):
    """Display only (the stock chart): one stock's daily Open/High/Low/Close/Volume by Date."""
    b = _bars(tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in BAR_CACHES))
    return b[b["Symbol"] == symbol].set_index("Date").sort_index()[["Open", "High", "Low", "Close", "Volume"]]

NY = "America/New_York"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

def alpaca_bars(symbols, minutes, start):
    """Display only: {symbol: OHLCV DataFrame indexed by New York time without zone} from Alpaca bars since `start`
    (minutes per bar; None = daily), SIP first and IEX on any SIP error; {} when neither answers."""
    import backtest_engine as be
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    frame = TimeFrame.Day if minutes is None else TimeFrame(minutes, TimeFrameUnit.Minute)
    client = be.market_data_client()
    for feed in ("sip", "iex"):
        try:
            data = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=frame,
                                                          start=pd.Timestamp(start).to_pydatetime(), feed=DataFeed(feed))).data
            break
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

@st.cache_data(ttl=60, show_spinner=False)
def intraday(symbols: tuple):
    """Display only (header card popups): {symbol: (epoch times, closes, previous close)} of the latest session's
    5-minute bars (today's, or the last session's while the market is closed). Cached 60 s; a series that cannot be read is left out."""
    if _live_off():
        return {}
    out = {}
    try:
        quotes = index_quotes(symbols)
        bars = alpaca_bars(symbols, 5, pd.Timestamp.now(tz=NY).normalize() - pd.Timedelta(days=5))
        for s, df in bars.items():
            df = last_sessions(df, 1)
            last, chg = quotes.get(s, (None, float("nan")))
            if len(df) and chg == chg:
                times = [t.timestamp() for t in df.index.tz_localize(NY)]
                out[s] = (times, df["Close"].tolist(), last / (1 + chg / 100))
    except Exception:
        pass
    return out

@st.cache_data(ttl=60, show_spinner=False)
def yf_intraday(symbols: tuple):
    """Display only (header cards Alpaca does not carry, e.g. CL=F, BTC-USD): {symbol: ((price, day %), (epoch times,
    closes, previous close))} from yfinance (free, keyless): 5-minute bars of the latest trading date, previous close =
    the prior daily close. Cached 60 s; a symbol that cannot be read is left out."""
    if _live_off():
        return {}
    out = {}
    try:
        import yfinance as yf
        for s in symbols:
            try:
                bars = yf.Ticker(s).history(period="5d", interval="5m")["Close"].dropna()
                daily = yf.Ticker(s).history(period="7d", interval="1d")["Close"].dropna()
                if bars.empty or len(daily) < 2:
                    continue
                today = bars[bars.index.date == bars.index[-1].date()]
                prev = float(daily[daily.index.date < bars.index[-1].date()].iloc[-1])
                px = float(today.iloc[-1])
                out[s] = ((px, (px / prev - 1) * 100), ([t.timestamp() for t in today.index], today.tolist(), prev))
            except Exception:
                continue
    except Exception:
        pass
    return out

INTRADAY_VIEWS = {"1D": (5, 1), "1W": (15, 5)}      # range button -> (minutes per candle, sessions shown)

@st.cache_data(ttl=60, show_spinner=False)
def intraday_ohlcv(symbol, view):
    """Display only (stock chart 1D / 1W buttons): OHLCV candles of the last session (5-minute) or the last 5 sessions
    (15-minute), regular hours, index = New York time without zone. Cached 60 s; empty when it cannot be read."""
    if _live_off() or view not in INTRADAY_VIEWS:
        return pd.DataFrame(columns=OHLCV)
    minutes, sessions = INTRADAY_VIEWS[view]
    try:
        df = alpaca_bars((symbol,), minutes, pd.Timestamp.now(tz=NY).normalize() - pd.Timedelta(days=4 + sessions * 2))
        return last_sessions(df[symbol], sessions) if symbol in df else pd.DataFrame(columns=OHLCV)
    except Exception:
        return pd.DataFrame(columns=OHLCV)
