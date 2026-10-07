"""Trading Account tab: live Alpaca holdings (read-only GETs, at most once a minute in market hours) and the
per-run holdings read the tax view and trade audit reuse."""
import os
from datetime import datetime

import streamlit as st

from dashboard.data import etf_prices
from dashboard.settings import CT
from dashboard.style import live_row, section, toned

@st.cache_resource(show_spinner=False)
def _fill_history():
    """The fill history kept for the whole dashboard process (alpaca_paper.FillHistory: only new fills are read)."""
    import alpaca_paper as ap
    return ap.FillHistory()

@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def _read_holdings(refresh_key):
    """One read of the Alpaca LIVE account (GET only: positions, fills, account + deposits/withdrawals as one snapshot,
    daily account history) per refresh_key (alpaca_paper.holdings_refresh_key: a new key each minute in market hours,
    each hour otherwise). A failure raises, and Streamlit never caches a raise, so the next minute tries again."""
    import alpaca_paper as ap
    import backtest_engine as be
    acct = ap.PaperAccount()
    positions, fills, snap = acct.position_dicts(), _fill_history().update(acct), acct.snapshot()
    return {"positions": positions, "fills": fills, "equity": snap["equity"], "snapshot": snap,
            "history": acct.daily_history(be.FORWARD_START), "as_of": datetime.now(CT)}

def live_holdings():
    """(data, None) or (None, plain message). Keys come from .env via alpaca_paper; error messages never hold them."""
    if os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off":                # tests: never call Alpaca
        return None, "Live holdings are turned off here (STOCK_ANALYSIS_LIVE_HOLDINGS=off)."
    try:
        import alpaca_paper as ap
        return _read_holdings(ap.holdings_refresh_key()), None
    except Exception as e:
        return None, f"Live holdings unavailable right now ({type(e).__name__}: {str(e)[:200]}). It tries again in a minute."

def new_run():
    """Start of each page run (app.py): forget the last run's holdings read. When the panels lived in app.py this was a
    module dict rebuilt on every run; it is kept per session here so two browser tabs never share one."""
    st.session_state["_run_holdings"] = {}

def _run_holdings():
    """This run's live holdings read: set by render_live_holdings, reused by the tax view and trade audit."""
    return st.session_state.setdefault("_run_holdings", {})

def holdings_this_run():
    """(data, err) of this page run's live holdings read (one Alpaca read per run, also when it fails)."""
    run = _run_holdings()
    if "v" not in run:
        run["v"] = live_holdings()
    return run["v"]

@st.fragment(run_every=60)        # reruns only this table each minute; it reads Alpaca only when the refresh key changes
def render_live_holdings():
    """Trading Account tab: the real Alpaca positions with cost, value, P/L and the first purchase date, then the account
    vs the index ETFs (render_benchmarks)."""
    data, err = _run_holdings()["v"] = live_holdings()
    if err:
        st.info(err)
        return
    import alpaca_paper as ap
    table = ap.holdings_table(data["positions"], data["fills"], data["equity"])
    if table.empty:
        st.info(f"No open positions in the Alpaca account (as of {data['as_of']:%a %b %-d %I:%M %p} CT).")
    else:
        render_positions(table, data)
    render_benchmarks(data)

def render_positions(table, data):
    """The holdings table (alpaca_paper.holdings_table) and its caption."""
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
               "is cash). First bought = the earliest buy still in the position (sells use up the oldest shares first).")

def render_benchmarks(data):
    """The account vs QQQ / SPY / IWM / DIA since the forward-test start close, the same deposits on the same days, all
    from Alpaca (alpaca_paper.account_vs_etfs: daily account history + this read's balance and deposits)."""
    import alpaca_paper as ap
    import backtest_engine as be
    closes, now = etf_prices(tuple(ap.BENCHMARK_ETFS), str(be.FORWARD_START))
    out = ap.account_vs_etfs(data["history"], data["snapshot"], closes, now, be.FORWARD_START)
    if out is None:
        return
    table, start, put_in = out
    section(f"Your account vs index ETFs since {start:%a %b %-d, %Y}")
    pct, money = st.column_config.NumberColumn(format="%+.2f%%"), st.column_config.NumberColumn(format="dollar")
    sty = live_row(toned(table, ["Return %", "Gain $", "Account ahead by (pts)"]), "Compared with", "Your account")
    st.dataframe(sty, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3,
                 column_config={"Return %": pct, "Value now": money, "Gain $": money,
                                "Account ahead by (pts)": st.column_config.NumberColumn(format="%+.2f")})
    st.caption(f"The same money in each ETF instead: your {start:%b %-d} closing balance plus every later deposit "
               f"(\\${put_in:,.2f} in total), each deposit or withdrawal bought or sold at the close of its date, valued at "
               "the latest price (ETF closes adjusted for dividends, from yfinance; display only). Return % is "
               "time-weighted, so deposits never count as gains: for an ETF it is its price change; for your account each "
               "day's growth from Alpaca's daily account history, with a deposit counted from the session after it arrives "
               "(Alpaca books deposits around 4:15 PM CT, after the close). Gain \\$ = value now - money put in. Account "
               "ahead by: green = your account beats that ETF, red = it trails.")
