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
           "DBC": "Commodities", "GLD": "Gold", "SLV": "Silver", "USO": "USO (crude ETF)"}
LIVE_WINDOWS = {"3 days %": pd.DateOffset(days=3), "1 week %": pd.DateOffset(weeks=1), **WINDOWS,
                "2 years %": pd.DateOffset(years=2)}

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
    """Daily closes (columns = symbols), ~25 months, from Alpaca."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import Adjustment
    from alpaca.data.timeframe import TimeFrame
    import backtest_engine as be
    start = pd.Timestamp.now(tz="America/New_York") - pd.DateOffset(days=760)
    bars, _ = _feeds(lambda f: be.market_data_client().get_stock_bars(
        StockBarsRequest(symbol_or_symbols=list(syms), timeframe=TimeFrame.Day, start=start, adjustment=Adjustment.SPLIT, feed=f)).df)
    c = bars.reset_index().pivot(index="timestamp", columns="symbol", values="close")
    c.index = c.index.tz_convert("America/New_York").normalize().tz_localize(None)
    return c

def portfolio_row(today):
    """'Our portfolio' row from the Alpaca LIVE account (read-only GETs): Live = equity now; % columns = time-weighted
    growth (alpaca_paper.account_twr_index, so deposits/withdrawals are not returns) vs the daily equity that far back."""
    import alpaca_paper as ap
    acct = ap.PaperAccount()
    h = acct._get("/account/portfolio/history", {"timeframe": "1D", "pnl_reset": "no_reset", "cashflow_types": "CSD,CSW,JNLC",
                  "start": f"{today - pd.DateOffset(years=2, days=7):%Y-%m-%d}", "end": f"{today:%Y-%m-%d}"})
    # (daily_history sends only a start, and Alpaca then returns about one month; an end gives the whole range)
    days = pd.to_datetime(h["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    flows = pd.DataFrame({k: pd.to_numeric(pd.Series(v), errors="coerce") for k, v in (h.get("cashflow") or {}).items()})
    d = pd.DataFrame({"Date": days, "Equity": pd.to_numeric(pd.Series(h["equity"]), errors="coerce").to_numpy(),
                      "Cashflow": flows.fillna(0).sum(axis=1).to_numpy() if len(flows.columns) else 0.0})
    d = d[d["Date"] < today].assign(Net_Deposits=lambda x: x["Cashflow"].cumsum())
    d = d[d["Equity"].fillna(0) > 0]                      # days before the account was funded
    live = acct.account_summary()["Equity"]
    d = pd.concat([d, pd.DataFrame([{"Date": today, "Equity": live, "Net_Deposits": d["Net_Deposits"].iloc[-1]}])])
    idx = ap.account_twr_index(d[["Date", "Equity", "Net_Deposits"]])
    row = {"Group": "Portfolio", "Name": "Our portfolio", "Symbol": "Alpaca", "Live price": live,
           "Today %": (idx.iloc[-1] / idx.iloc[-2] - 1) * 100 if len(idx) > 1 else None}
    for col, off in LIVE_WINDOWS.items():
        past = idx[idx.index <= today - off]
        row[col] = (idx.iloc[-1] / past.iloc[-1] - 1) * 100 if len(past) else None
    return row

FUTURES = {"CL=F": "Crude oil WTI (US)", "BZ=F": "Crude oil Brent (global)", "GC=F": "Gold futures",
           "SI=F": "Silver futures"}

@st.cache_data(ttl=60, show_spinner=False)
def futures_rows(today):
    """Front-month futures from yfinance ($ per barrel / ounce): live = latest price, % vs the daily close that far back.
    A failed symbol is left out."""
    import yfinance as yf
    rows = []
    for sym, name in FUTURES.items():
        try:
            t = yf.Ticker(sym)
            c = t.history(period="3y", interval="1d", auto_adjust=False)["Close"].dropna()
            c.index = c.index.tz_localize(None).normalize()
            live = float(t.fast_info["last_price"])
            prev = c[c.index < today]
            row = {"Group": "Futures", "Name": name, "Symbol": sym, "Live price": live,
                   "Today %": (live / prev.iloc[-1] - 1) * 100 if len(prev) else None}
            for col, off in LIVE_WINDOWS.items():
                past = c[c.index <= today - off]
                row[col] = (live / past.iloc[-1] - 1) * 100 if len(past) else None
            rows.append(row)
        except Exception:
            pass
    return rows

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
        for col, off in LIVE_WINDOWS.items():
            past = c[c.index <= today - off]
            row[col] = (live / past.iloc[-1] - 1) * 100 if live and len(past) else None
        rows.append(row)
    try:
        rows.insert(0, portfolio_row(today))
    except Exception:
        pass   # account unavailable: the ETF rows still show
    try:
        rows[len(rows) - len(sector_etfs):len(rows) - len(sector_etfs)] = futures_rows(today)   # before the sectors
    except Exception:
        pass
    return pd.DataFrame(rows), feed

@st.fragment(run_every=60)
def render_live_table():
    """Refreshes itself every minute (prices cached 30 s)."""
    try:
        t, feed = live_table()
    except Exception as e:
        st.warning(f"Live prices unavailable: {str(e)[:150]}")
        return
    pcts = ["Today %"] + list(LIVE_WINDOWS)
    st.dataframe(toned(t.round(2), pcts), hide_index=True, width="stretch",
                 column_config={"Live price": st.column_config.NumberColumn(format="$%.2f"), **{k: PCT_COL for k in pcts}})
    st.caption(f"Live price = latest Alpaca trade ({feed.upper()}), refreshed every minute, "
               f"{pd.Timestamp.now(tz='America/Chicago'):%-I:%M:%S %p} CT. Indexes and commodities via ETFs; futures rows (crude $/barrel, gold/silver $/oz) from Yahoo Finance. "
               "Our portfolio = Alpaca account equity, returns time-weighted (deposits excluded), blank = before the account's "
               "history. Today % vs yesterday's close; other columns = live price vs the daily close that far back.")

def render_sector_table(p):
    section("Sector performance")
    render_live_table()
    t = _cached(p.df, p.mtime)
    st.dataframe(toned(t.round(2), list(WINDOWS)), hide_index=True, width="stretch",
                 column_config={k: PCT_COL for k in WINDOWS})
    st.caption(f"Each sector = the average price return of its stocks (equal weight), to the {p.df['Date'].max():%a %b %-d} "
               "close. Sorted by the 1-month return.")
