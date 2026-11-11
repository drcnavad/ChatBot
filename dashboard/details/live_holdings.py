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

@st.cache_data(ttl=60, max_entries=8, show_spinner=False)
def _read_core(refresh_key):
    """Positions + account snapshot only (no fills, no portfolio history). Used by next_plan / plan chips so every tab's
    build_page does not wait on ~2000 fill pages. Parallel GETs."""
    import alpaca_paper as ap
    import holdings
    from concurrent.futures import ThreadPoolExecutor
    acct = ap.PaperAccount()
    with ThreadPoolExecutor(2) as ex:
        f_pos, f_snap = ex.submit(holdings.position_rows, True), ex.submit(acct.snapshot)
        positions, snap = f_pos.result(), f_snap.result()
    return {"positions": positions, "equity": snap["equity"], "snapshot": snap, "as_of": datetime.now(CT)}


def account_core():
    """(data, None) or (None, message): light Alpaca read for plan chips / last-decision preview."""
    if os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off":
        return None, "Live holdings are turned off here (STOCK_ANALYSIS_LIVE_HOLDINGS=off)."
    try:
        from dashboard.data import refresh_slot
        return _read_core(refresh_slot()), None
    except Exception as e:
        return None, f"Live holdings unavailable right now ({type(e).__name__}: {str(e)[:200]}). It tries again in a minute."


@st.cache_data(ttl=600, max_entries=4, show_spinner=False)
def _read_holdings(refresh_key):
    """Full Alpaca LIVE read: reuses _read_core (same refresh_key → one /positions + snapshot), then fills + daily
    history in parallel. Fills use FillHistory (disk-backed) so a process restart is incremental. A failure raises,
    and Streamlit never caches a raise, so the next minute tries again."""
    import alpaca_paper as ap
    import backtest_engine as be
    from concurrent.futures import ThreadPoolExecutor
    core = _read_core(refresh_key)  # cache hit when plan chips already warmed this slot
    acct = ap.PaperAccount()
    with ThreadPoolExecutor(2) as ex:
        f_fills = ex.submit(_fill_history().update, acct)
        f_hist = ex.submit(acct.daily_history, be.FORWARD_START)
        fills, history = f_fills.result(), f_hist.result()
    return {"positions": core["positions"], "fills": fills, "equity": core["equity"], "snapshot": core["snapshot"],
            "history": history, "as_of": core["as_of"]}

def live_holdings():
    """(data, None) or (None, plain message). Keys come from .env via alpaca_paper; error messages never hold them."""
    if os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off":                # tests: never call Alpaca
        return None, "Live holdings are turned off here (STOCK_ANALYSIS_LIVE_HOLDINGS=off)."
    try:
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
    try:
        _render_contribution(data)
    except Exception:
        pass

@st.cache_data(ttl=300, show_spinner=False)
def _decision_days():
    """{symbol: As_Of decision date of its latest completed buy of 1+ shares (skips tiny top-ups)} from live_orders_log; {} if unavailable."""
    try:
        import sqlite3
        db = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "stock_analysis.db")
        with sqlite3.connect(db) as c:
            o = pd.read_sql("select Symbol, As_Of, Submitted_At_CT from live_orders_log where upper(Side)='BUY' "
                            "and Status like 'COMPLETED%' and Shares >= 1 and As_Of is not null and As_Of != '' order by Submitted_At_CT", c)
        return dict(zip(o["Symbol"], pd.to_datetime(o["As_Of"])))
    except Exception:
        return {}

def _decision_day(sym):
    return _decision_days().get(str(sym))

def rank_at_buy(sym, bought):
    """The stock's RS short-term rank at the latest saved decision on or before its first buy (Reports/decision_ranks.csv);
    None when no rank was saved then (ranks are saved from Fri Oct 16, 2026)."""
    col = "Rank_RS_Short"
    if bought is None or bought != bought:
        return None
    try:
        import decision_ranks
        from dashboard.data import read_report_csv
        r = read_report_csv(decision_ranks.RANKS_CSV).assign(Date=lambda d: pd.to_datetime(d["Date"]))
    except Exception:
        return None
    bought = _decision_day(sym) or bought   # Nov 4, 2026: rank on the latest buy's decision day, not the fill day
    r = r[(r["Symbol"] == sym) & (r["Date"].dt.date <= pd.Timestamp(bought).date())].sort_values("Date")
    if col not in r.columns:
        return None
    if len(r) and pd.notna(r[col].iloc[-1]):
        return int(r[col].iloc[-1])
    return _computed_rank(sym, pd.Timestamp(bought).date().isoformat())   # no saved row then: recompute (display only)

@st.cache_data(ttl=3600, show_spinner=False)
def _computed_ranks(day):
    """{symbol: rank} recomputed for session `day` from the signal table (decision_ranks.ranks_on); {} if unavailable."""
    try:
        import decision_ranks
        import sleeves
        idx = sleeves.inputs()["close"].index
        idx = idx[idx <= pd.Timestamp(day)]                 # a weekend / holiday buy date: the session before it
        if not len(idx):
            return {}
        _, r = decision_ranks.ranks_on(idx[-1])
        return {s: int(v) for s, v in zip(r["Symbol"], r["Rank_RS_Short"]) if pd.notna(v)}
    except Exception:
        return {}

def _computed_rank(sym, day):
    return _computed_ranks(day).get(sym)

@st.cache_data(ttl=900, show_spinner=False)
def _rank_now_full():
    """(as-of text, {symbol: rank}): the live score re-ranked on the latest FINAL close (rank_close, fresh Alpaca daily
    bars); falls back to sleeves.live_ranks (the signal table's last bar = the 2:30 PM decision price)."""
    try:
        import rank_close
        d, r = rank_close.close_ranks(all_stocks=True)
        return f"{d:%a %b %-d} close", {s: int(v[0]) for s, v in r.items()}
    except Exception:
        try:
            import sleeves
            d = sleeves.inputs()["close"].index[-1]
            return f"{d:%a %b %-d} 2:30 PM", {s: int(v[0]) for s, v in sleeves.live_ranks(all_stocks=True).items()}
        except Exception:
            return "", {}

def _rank_now():
    return _rank_now_full()[1]

def rank_now(sym):
    """Live RS short-term rank at the latest close (None = not ranked: score <= 0, not eligible or do_not_buy)."""
    return _rank_now().get(str(sym))

def render_positions(table, data):
    """The holdings table (alpaca_paper.holdings_table) and its caption."""
    table.insert(1, "Rank", pd.array([rank_at_buy(s, d) if not str(s).startswith("Total") else None
                                      for s, d in zip(table["Stock"], table["First bought"])], dtype="Int64"))
    table.insert(1, "Rank now", pd.array([rank_now(s) if not str(s).startswith("Total") else None
                                          for s in table["Stock"]], dtype="Int64"))
    table["First bought"] = [f"{d:%a %b %-d, %Y}" if d is not None and d == d else "" for d in table["First bought"]]
    asof = _rank_now_full()[0]
    t = table.drop(columns=["Weight %"], errors="ignore")   # equal weights: not shown (Oct 21, 2026)
    try:
        from dashboard.data import last_next_earnings
        stocks = [s for s in t["Stock"] if not str(s).startswith("Total")]
        ed = last_next_earnings(stocks)
        nxt = dict(zip(ed["Symbol"].astype(str), ed["Next ED"])) if ed is not None and len(ed) else {}
        t.insert(len(t.columns), "Next earnings", ["" if str(s).startswith("Total") else (nxt.get(s) or "—") for s in t["Stock"]])
    except Exception:
        pass
    show(t, ["P/L $", "P/L %", "Today %"], fit=True,
         cols={"Rank now": st.column_config.NumberColumn(f"Rank now ({asof})" if asof else "Rank now", format="%d",
                                                     help=f"Live RS short-term rank (10/21/63-day, 63 doubled) at the "
                                                     f"latest completed close: {asof or 'n/a'}. Refreshed from Alpaca "
                                                     "daily bars; Friday-only rebalance; blank = not ranked (score <= 0)"),
               "Rank": st.column_config.NumberColumn("Rank at buy", format="%d", help="RS short-term rank on the decision day of its latest buy"),
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
    table, start, _put_in = out
    section(f"Your account vs index ETFs since {start:%a %b %-d, %Y}")
    show(live_row(toned(table, ["Return %", "Gain $", "Account ahead by (pts)"]), "Compared with", "Your account"), fit=True,
         cols={"Return %": SPCT, "Value now": MONEY, "Gain $": MONEY,
               "Account ahead by (pts)": st.column_config.NumberColumn(format="%+.2f")})


def _render_contribution(data):
    """This week and since Oct 29 contribution of current holdings vs QQQ (display only)."""
    import trade_records as tr
    from dashboard.style import MONEY, SPCT, caption_text
    positions = data.get("positions") or []
    first = {}
    try:
        fills = data.get("fills") or []
        import alpaca_paper as ap
        first = {str(k).upper(): v for k, v in (ap.first_buy_dates(fills) or {}).items()}
    except Exception:
        first = {}
    as_of = data.get("as_of")
    hold, sec = tr.contribution_from_store(positions, as_of=as_of, first_bought=first)
    if hold is None or not len(hold):
        return
    section("This week and since Oct 29 vs QQQ (current book)")
    cfg = {"This week $": MONEY, "Since Oct 29 $": MONEY, "This week %": SPCT, "Since Oct 29 %": SPCT,
           "vs QQQ this week pp": SPCT, "vs QQQ since Oct 29 pp": SPCT, "Price": MONEY, "Shares": st.column_config.NumberColumn(format="%.2f")}
    show(hold, ["This week $", "Since Oct 29 $", "vs QQQ this week pp", "vs QQQ since Oct 29 pp"], fit=True,
         key="contrib_hold", cols=cfg)
    if sec is not None and len(sec):
        show(sec, ["This week $", "Since Oct 29 $"], fit=True, key="contrib_sector",
             cols={"This week $": MONEY, "Since Oct 29 $": MONEY})
    caption_text("P/L of the current holdings from the last Friday rebalance (this week) and from Oct 29, 2026, "
                 "vs QQQ over the same window for each name. Names bought mid-period start from their first close "
                 "after the purchase. Display only; this does not change trades.")
