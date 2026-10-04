"""Details tab: live Alpaca holdings (read-only GETs, at most once a minute in market hours) and the
per-run holdings read the earnings planner, tax view and trade audit reuse."""
import os
from datetime import datetime

import streamlit as st

from dashboard.data import live_quote, load_benchmarks
from dashboard.settings import CT
from dashboard.style import toned


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


def new_run():
    """Start of each page run (app.py): forget the last run's holdings read. When the panels lived in app.py this was a
    module dict rebuilt on every run; it is kept per session here so two browser tabs never share one."""
    st.session_state["_run_holdings"] = {}


def _run_holdings():
    """This run's live holdings read: set by render_live_holdings, reused by the earnings planner, tax view and audit."""
    return st.session_state.setdefault("_run_holdings", {})


def holdings_this_run():
    """(data, err) of this page run's live holdings read (one Alpaca read per run, also when it fails)."""
    run = _run_holdings()
    if "v" not in run:
        run["v"] = live_holdings()
    return run["v"]


@st.fragment(run_every=60)        # reruns only this table each minute; it reads Alpaca only when the refresh key changes
def render_live_holdings():
    """Details tab: the real Alpaca positions with cost, value, P/L and the first purchase date, plus QQQ for comparison."""
    data, err = _run_holdings()["v"] = live_holdings()
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
