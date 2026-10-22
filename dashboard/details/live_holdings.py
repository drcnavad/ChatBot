"""Trading Account tab: live Alpaca holdings (read-only GETs, at most once a minute in market hours) and the
per-run holdings read the tax view and trade audit reuse."""
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.data import etf_prices
from dashboard.settings import CT
from dashboard.style import MONEY, SPCT, live_row, section, toned
from dashboard.style import table as show

@st.cache_resource(show_spinner=False)
def _fill_history():
    """The fill history kept for the whole dashboard process (alpaca_paper.FillHistory: only new fills are read)."""
    import alpaca_paper as ap
    return ap.FillHistory()

@st.cache_data(ttl=600, max_entries=4, show_spinner=False)
def _read_holdings(refresh_key):
    """One read of the Alpaca LIVE account (GET only: positions, fills, account + deposits/withdrawals as one snapshot,
    daily account history) per refresh_key (alpaca_paper.holdings_refresh_key: a new key each minute in market hours,
    each hour otherwise). A failure raises, and Streamlit never caches a raise, so the next minute tries again."""
    import alpaca_paper as ap
    import backtest_engine as be
    acct = ap.PaperAccount()
    import holdings
    positions, fills, snap = holdings.position_rows(fresh=True), _fill_history().update(acct), acct.snapshot()
    return {"positions": positions, "fills": fills, "equity": snap["equity"], "snapshot": snap,
            "history": acct.daily_history(be.FORWARD_START), "as_of": datetime.now(CT)}

def live_holdings():
    """(data, None) or (None, plain message). Keys come from .env via alpaca_paper; error messages never hold them."""
    if os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off":                # tests: never call Alpaca
        return None, "Live holdings are turned off here (STOCK_ANALYSIS_LIVE_HOLDINGS=off)."
    try:
        import alpaca_paper as ap
        from dashboard.data import refresh_slot
        return _read_holdings(refresh_slot()), None
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
    from dashboard.data import refresh_slot
    run, key = _run_holdings(), refresh_slot()
    if "v" not in run or run.get("key", key) != key:      # a 5-minute Home fragment rerun reads again once the key moves
        run["v"], run["key"] = live_holdings(), key
    return run["v"]

@st.fragment(run_every=30)        # reruns every 30 s; reads Alpaca when the refresh slot moves (30 s open, 5 min closed)
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

HALF_NAMES = {"lowvol": "Low Vol", "sector": "Sector"}

MANUAL_BUYS = {"MNST"}   # bought on Chirag's request but run by the 9 AM fill check, so not in manual_orders.json

def strategy_of(symbols, reports=None):
    """{symbol: strategy}: the half that owns it in Reports/sleeve_ledger.json (the bot's own tags); not tagged = Manual.
    " (M)" = a manual buy: a BUY in Reports/manual_orders.json (manual_trade.py's log) or MANUAL_BUYS."""
    import json
    from dashboard.settings import ROOT
    reports = reports or os.path.join(ROOT, "Reports")
    try:
        halves = json.load(open(os.path.join(reports, "sleeve_ledger.json")))["halves"]
        tag = {s: HALF_NAMES.get(h, h) for h, v in halves.items() for s in v.get("tags", [])}
    except Exception:
        return {s: "Unknown" for s in symbols}
    manual = set(MANUAL_BUYS)
    try:
        mo = json.load(open(os.path.join(reports, "manual_orders.json")))
        manual |= {o["symbol"] for o in mo if str(o.get("action", "")).startswith("buy")}
    except Exception:
        pass
    return {s: (tag[s] + (" (M)" if s in manual else "")) if s in tag else "Manual" for s in symbols}

def rank_at_buy(sym, strategy, bought):
    """The stock's rank in its half at the latest saved decision on or before its first buy (Reports/decision_ranks.csv);
    None for manual buys or when no rank was saved then (ranks are saved from Fri Oct 16, 2026)."""
    col = {"Low Vol": "Rank_Vol", "Sector": "Rank_Sector"}.get(strategy)
    if col is None or bought is None or bought != bought:
        return None
    try:
        import decision_ranks
        r = pd.read_csv(decision_ranks.RANKS_CSV, parse_dates=["Date"])
    except Exception:
        return None
    r = r[(r["Symbol"] == sym) & (r["Date"].dt.date <= pd.Timestamp(bought).date())].sort_values("Date")
    return int(r[col].iloc[-1]) if len(r) and pd.notna(r[col].iloc[-1]) else None

def render_positions(table, data):
    """The holdings table (alpaca_paper.holdings_table) and its caption."""
    strat = strategy_of(list(table["Stock"]))
    table.insert(1, "Strategy", [strat.get(s, "") if not str(s).startswith("Total") else "" for s in table["Stock"]])
    table.insert(2, "Rank", pd.array([rank_at_buy(s, g, d) for s, g, d in zip(table["Stock"], table["Strategy"], table["First bought"])],
                                     dtype="Int64"))
    table["First bought"] = [f"{d:%a %b %-d, %Y}" if d is not None and d == d else "" for d in table["First bought"]]
    t = table.drop(columns=["Weight %"], errors="ignore")   # equal weights: not shown (Oct 21, 2026)
    show(t, ["P/L $", "P/L %", "Today %"], fit=True,
         cols={"Rank": st.column_config.NumberColumn(format="%d", help="Rank in its half on the day it was first bought"),
               "Shares": st.column_config.NumberColumn(format="%.2f"), "P/L %": SPCT, "Today %": SPCT,
               **dict.fromkeys(("Avg price", "Cost basis", "Market value", "P/L $", "Price"), MONEY)})

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
    show(live_row(toned(table, ["Return %", "Gain $", "Account ahead by (pts)"]), "Compared with", "Your account"), fit=True,
         cols={"Return %": SPCT, "Value now": MONEY, "Gain $": MONEY,
               "Account ahead by (pts)": st.column_config.NumberColumn(format="%+.2f")})
