"""Home tab, bottom: sector performance over 14 days, 1, 2, 3 and 6 months and 1 year (Chirag, Oct 16, 2026).

Each sector (sector_mapping.symbol_sector) = the equal-weighted average of its tradable stocks' price returns, from the
closes in Reports/signal_analysis.csv: the latest close vs the last close on or before that many calendar days / months
earlier (a stock without a close that far back is left out of that column). Display only. (The sector-rotation half
ranks sectors by their ETF's 63-day return instead - forward_test.strategy_inputs; this table is per stock.)"""
import pandas as pd
import streamlit as st

from dashboard.settings import symbol_sector
from dashboard.style import PCT_COL, section, table

try:
    from sector_mapping import tradable_symbols
except Exception:
    tradable_symbols = []

WINDOWS = {"14 days %": pd.DateOffset(days=14), "1 month %": pd.DateOffset(months=1),
           "2 months %": pd.DateOffset(months=2), "3 months %": pd.DateOffset(months=3),
           "6 months %": pd.DateOffset(months=6), "1 year %": pd.DateOffset(years=1)}

def sector_returns(df, symbols=None):
    """DataFrame Sector, Stocks, one column per WINDOWS entry (%), sorted by the 1-month return (best first)."""
    symbols = set(symbols or tradable_symbols)
    c = (df[df["Symbol"].isin(symbols)].pivot_table(index="Date", columns="Symbol", values="Close", aggfunc="last")
         .sort_index().ffill())
    last = c.index.max()
    rets = {}
    for col, off in WINDOWS.items():
        past = c[c.index <= last - off]
        rets[col] = (c.loc[last] / past.iloc[-1] - 1) * 100 if len(past) else pd.Series(dtype=float)
    r = pd.DataFrame(rets)
    r["Sector"] = [symbol_sector.get(s, "—") for s in r.index]
    out = r.groupby("Sector").agg(Stocks=("1 month %", "size"), **{k: (k, "mean") for k in WINDOWS})
    return out.reset_index().sort_values("1 month %", ascending=False, na_position="last").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def _cached(_df, mtime):
    return sector_returns(_df)

# ---- live prices: major indexes, commodities and the SPDR sector ETFs (Alpaca, SIP -> IEX). Read-only market data.
try:
    from sector_mapping import sector_etfs
except Exception:
    sector_etfs = {}
MARKETS = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWM": "Russell 2000",
           "DBC": "Commodities"}   # gold, silver, crude: the futures rows (GLD, SLV, USO removed Oct 21)
LIVE_WINDOWS = {"3 days %": pd.DateOffset(days=3), "1 week %": pd.DateOffset(weeks=1), **WINDOWS,
                "2 years %": pd.DateOffset(years=2)}

from dashboard.alpaca_data import client as _client, sip_then_iex as _feeds  # noqa: E402  (shared SIP -> IEX)

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _daily_closes(syms):
    """Daily closes (columns = symbols), ~25 months, from Alpaca."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import Adjustment
    from alpaca.data.timeframe import TimeFrame
    start = pd.Timestamp.now(tz="America/New_York") - pd.DateOffset(days=760)
    bars, _ = _feeds(lambda f: _client().get_stock_bars(
        StockBarsRequest(symbol_or_symbols=list(syms), timeframe=TimeFrame.Day, start=start, adjustment=Adjustment.SPLIT, feed=f)).df)
    c = bars.reset_index().pivot(index="timestamp", columns="symbol", values="close")
    c.index = c.index.tz_convert("America/New_York").normalize().tz_localize(None)
    return c

FUTURES = {"CL=F": "Crude oil WTI (US)", "BZ=F": "Crude oil Brent (global)", "GC=F": "Gold futures",
           "SI=F": "Silver futures"}

def _one_future(sym_name, today):
    """One futures row from yfinance, or None."""
    import yfinance as yf
    sym, name = sym_name
    try:
        t = yf.Ticker(sym)
        c = t.history(period="3y", interval="1d", auto_adjust=False)["Close"].dropna()
        c.index = c.index.tz_localize(None).normalize()
        # Futures stay 24-hour (Chirag Oct 29: same exception as CL=F / BTC-USD header cards).
        live = float(t.fast_info["last_price"])
        prev = c[c.index < today]
        row = {"Group": "Futures", "Name": name, "Symbol": sym, "Live price": live,
               "Today %": (live / prev.iloc[-1] - 1) * 100 if len(prev) else None}
        for col, off in LIVE_WINDOWS.items():
            past = c[c.index <= today - off]
            row[col] = (live / past.iloc[-1] - 1) * 100 if len(past) else None
        return row
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def futures_rows(today):
    """Front-month futures from yfinance ($ per barrel / ounce): live = latest price, % vs the daily close that far back.
    A failed symbol is left out. Fetched in parallel (4 symbols)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        return [r for r in ex.map(lambda sn: _one_future(sn, today), list(FUTURES.items())) if r]

@st.cache_data(ttl=30, show_spinner=False)
def live_table():
    """Group, Name, Symbol, Live price, Today %, then WINDOWS % (live price vs that past daily close). -> (table, feed)."""
    from alpaca.data.requests import StockSnapshotRequest
    names = {**MARKETS, **sector_etfs}
    snaps, feed = _feeds(lambda f: _client().get_stock_snapshot(
        StockSnapshotRequest(symbol_or_symbols=list(names), feed=f)))
    closes, today, rows = _daily_closes(tuple(names)), pd.Timestamp.now().normalize(), []
    from dashboard.data import _rth_quote
    for s, name in names.items():
        sn = snaps.get(s)
        q = _rth_quote(sn) if sn else None
        live, day_pct = (q[0], q[1]) if q else (None, None)
        row = {"Group": "Market" if s in MARKETS else "Sector", "Name": name, "Symbol": s, "Live price": live,
               "Today %": day_pct}
        c = closes[s].dropna() if s in closes else pd.Series(dtype=float)
        for col, off in LIVE_WINDOWS.items():
            past = c[c.index <= today - off]
            row[col] = (live / past.iloc[-1] - 1) * 100 if live and len(past) else None
        rows.append(row)
    try:
        rows[len(rows) - len(sector_etfs):len(rows) - len(sector_etfs)] = futures_rows(today)   # before the sectors
    except Exception:
        pass
    return pd.DataFrame(rows), feed

@st.fragment(run_every=30)
def render_live_table():
    """Refreshes itself every 30 s (prices cached 30 s)."""
    try:
        t, feed = live_table()
    except Exception as e:
        st.warning(f"Live prices unavailable: {str(e)[:150]}")
        return
    pcts = ["Today %"] + list(LIVE_WINDOWS)
    table(t.round(2), pcts, cols={"Live price": st.column_config.NumberColumn(format="$%.2f"), **{k: PCT_COL for k in pcts}})

def render_sector_table(p):
    section("Sector performance")
    render_live_table()
    t = _cached(p.df, p.mtime)
    table(t.round(2), WINDOWS, cols={k: PCT_COL for k in WINDOWS})
