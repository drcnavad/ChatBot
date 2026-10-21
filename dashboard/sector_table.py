"""Home tab, bottom: sector performance over 14 days, 1, 2, 3 and 6 months and 1 year (Chirag, Oct 16, 2026).

Each sector (sector_mapping.symbol_sector) = the equal-weighted average of its tradable stocks' price returns, from the
closes in Reports/signal_analysis.csv: the latest close vs the last close on or before that many calendar days / months
earlier (a stock without a close that far back is left out of that column). Display only. (The sector-rotation half
ranks sectors by their ETF's 63-day return instead - forward_test.strategy_inputs; this table is per stock.)"""
import pandas as pd
import streamlit as st

from dashboard.settings import symbol_sector
from dashboard.style import PCT_COL, section, toned

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
           "DBC": "Commodities", "GLD": "Gold", "SLV": "Silver", "USO": "Crude Oil"}

def _feeds(fetch):
    """fetch(feed) on SIP, then IEX if SIP fails (like paper_trade's quotes) -> (result, feed)."""
    from alpaca.data.enums import DataFeed
    err = None
    for feed in ("sip", "iex"):
        try:
            return fetch(DataFeed(feed)), feed
        except Exception as e:
            err = e
    raise err

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _daily_closes(syms):
    """Daily closes (columns = symbols), ~13 months, from Alpaca."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import Adjustment
    from alpaca.data.timeframe import TimeFrame
    import backtest_engine as be
    start = pd.Timestamp.now(tz="America/New_York") - pd.DateOffset(days=400)
    bars, _ = _feeds(lambda f: be.market_data_client().get_stock_bars(
        StockBarsRequest(symbol_or_symbols=list(syms), timeframe=TimeFrame.Day, start=start, adjustment=Adjustment.SPLIT, feed=f)).df)
    c = bars.reset_index().pivot(index="timestamp", columns="symbol", values="close")
    c.index = c.index.tz_convert("America/New_York").normalize().tz_localize(None)
    return c

@st.cache_data(ttl=30, show_spinner=False)
def live_table():
    """Group, Name, Symbol, Live price, Today %, then WINDOWS % (live price vs that past daily close). -> (table, feed)."""
    from alpaca.data.requests import StockSnapshotRequest
    import backtest_engine as be
    names = {**MARKETS, **sector_etfs}
    snaps, feed = _feeds(lambda f: be.market_data_client().get_stock_snapshot(
        StockSnapshotRequest(symbol_or_symbols=list(names), feed=f)))
    closes, today, rows = _daily_closes(tuple(names)), pd.Timestamp.now().normalize(), []
    for s, name in names.items():
        sn = snaps.get(s)
        live = sn.latest_trade.price if sn and sn.latest_trade else None
        prev = sn.previous_daily_bar.close if sn and sn.previous_daily_bar else None
        row = {"Group": "Market" if s in MARKETS else "Sector", "Name": name, "Symbol": s, "Live price": live,
               "Today %": (live / prev - 1) * 100 if live and prev else None}
        c = closes[s].dropna() if s in closes else pd.Series(dtype=float)
        for col, off in WINDOWS.items():
            past = c[c.index <= today - off]
            row[col] = (live / past.iloc[-1] - 1) * 100 if live and len(past) else None
        rows.append(row)
    return pd.DataFrame(rows), feed

@st.fragment(run_every=60)
def render_live_table():
    """Refreshes itself every minute (prices cached 30 s)."""
    try:
        t, feed = live_table()
    except Exception as e:
        st.warning(f"Live prices unavailable: {str(e)[:150]}")
        return
    pcts = ["Today %"] + list(WINDOWS)
    st.dataframe(toned(t.round(2), pcts), hide_index=True, width="stretch",
                 column_config={"Live price": st.column_config.NumberColumn(format="$%.2f"), **{k: PCT_COL for k in pcts}})
    st.caption(f"Live price = latest Alpaca trade ({feed.upper()}), refreshed every minute, "
               f"{pd.Timestamp.now(tz='America/Chicago'):%-I:%M:%S %p} CT. Indexes and commodities via ETFs. "
               "Today % vs yesterday's close; other columns = live price vs the daily close that far back.")

def render_sector_table(p):
    section("Sector performance")
    render_live_table()
    t = _cached(p.df, p.mtime)
    st.dataframe(toned(t.round(2), list(WINDOWS)), hide_index=True, width="stretch",
                 column_config={k: PCT_COL for k in WINDOWS})
    st.caption(f"Each sector = the average price return of its stocks (equal weight), to the {p.df['Date'].max():%a %b %-d} "
               "close. Sorted by the 1-month return.")
