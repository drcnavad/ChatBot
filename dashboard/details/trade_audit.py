"""Trading Account tab: the trade audit (trade_audit.py; read-only, hourly)."""
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.details.live_holdings import holdings_this_run
from dashboard.settings import CT
from dashboard.style import MONEY, SHARES, SPCT, INT, caption_text, info_text, table

# ---------------------------------------------------------------------------- trade audit (read-only)
@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def _read_audit(hour_key, _positions, _fills=()):
    """trade_audit.report() once per hour_key (the hour + the number of fills; GET only: account, orders, fills, clock;
    positions from the holdings read, not part of the cache key since their prices change every minute)."""
    import alpaca_paper as ap
    import trade_audit as ta
    acct = ap.PaperAccount()
    fills = _fills

    class _Acct:                          # the account, but its fills come from the live holdings' fill history
        def __getattr__(self, name):
            return getattr(acct, name)

        def fills(self, after=None, max_pages=50):
            return list(fills) if after is None else acct.fills(after=after, max_pages=max_pages)
    return ta.report(ta.fetch(_Acct(), positions=list(_positions)))

def _by_stock_day(trips):
    """Merge FIFO lots sold on the same day: total shares, share-weighted avg buy/sell price, summed P/L $,
    P/L % on cost, earliest buy date, share-weighted days held."""
    t = trips.copy()
    t["_day"] = pd.to_datetime(t["Sold_CT"]).dt.date
    t["_cost"] = t["Shares"] * t["Buy_Price"]
    t["_proc"] = t["Shares"] * t["Sell_Price"]
    t["_dh"] = t["Shares"] * t["Days_Held"]
    g = t.groupby(["Symbol", "_day"], sort=False).agg(
        Shares=("Shares", "sum"), Bought_CT=("Bought_CT", "min"), _cost=("_cost", "sum"),
        Sold_CT=("Sold_CT", "max"), _proc=("_proc", "sum"), **{"P/L $": ("P/L $", "sum")}, _dh=("_dh", "sum")).reset_index()
    g["Buy_Price"] = g["_cost"] / g["Shares"]
    g["Sell_Price"] = g["_proc"] / g["Shares"]
    g["P/L %"] = (g["P/L $"] / g["_cost"] * 100).where(g["_cost"] > 0)
    g["Days_Held"] = (g["_dh"] / g["Shares"]).round()
    return g[["Symbol", "Shares", "Bought_CT", "Buy_Price", "Sold_CT", "Sell_Price", "P/L $", "P/L %", "Days_Held"]]

def render_trade_audit():
    """Trading Account tab: each sale with its purchase (FIFO) and its gain / loss, plus a fixed Total row below
    (trade_audit.py; read-only, hourly). Oct 21, 2026 (Chirag): the alerts, cards and order ledger were removed."""
    data, err = holdings_this_run()
    if err:
        caption_text("The trade audit needs the live account, which is not available right now (see Live holdings above).")
        return
    try:
        rep = _read_audit(f"{datetime.now(CT):%Y-%m-%d %H}-{len(data['fills'])}", tuple(data["positions"]), tuple(data["fills"]))
    except Exception as e:
        info_text(f"Trade audit unavailable right now ({type(e).__name__}: {str(e)[:200]}). It tries again next hour.")
        return
    from backtest_engine import FORWARD_START
    trips = rep["round_trips"]
    if len(trips):                          # only round trips bought and sold on / after Oct 2 (Chirag, Oct 21)
        trips = trips[pd.to_datetime(trips["Bought_CT"]).dt.date >= pd.Timestamp(FORWARD_START).date()]
    if not len(trips):
        caption_text("No sales yet.")
        return
    trips = _by_stock_day(trips)            # one row per stock per sell day (Chirag, Nov 4)
    shown = trips.assign(**{c: [f"{x:%b %-d, %Y}" if x == x else "" for x in trips[c]] for c in ("Bought_CT", "Sold_CT")})
    cfg = {"Buy_Price": MONEY, "Sell_Price": MONEY, "P/L $": MONEY, "P/L %": SPCT, "Days_Held": INT, "Shares": SHARES}
    table(shown.round(2), ["P/L $", "P/L %"], max_h=300, cols=cfg)
    total = pd.DataFrame([{c: None for c in shown.columns}]).astype(object)     # its own table: never sorted with the rows
    cost = float((trips["Shares"] * trips["Buy_Price"]).sum())
    proc = float((trips["Shares"] * trips["Sell_Price"]).sum())
    pl = float(trips["P/L $"].sum())
    total.loc[0, shown.columns[0]] = f"Total ({len(trips)} sales)"
    total.loc[0, "Shares"] = round(float(trips["Shares"].sum()), 2)
    total.loc[0, "Buy_Price"] = round(cost, 2)          # total cost of all sold shares
    total.loc[0, "Sell_Price"] = round(proc, 2)         # total proceeds
    total.loc[0, "P/L $"] = round(pl, 2)
    total.loc[0, "P/L %"] = round(pl / cost * 100, 2) if cost else None
    total.loc[0, "Days_Held"] = round(float(trips["Days_Held"].mean()), 1)
    for c in ("Shares", "Buy_Price", "Sell_Price", "P/L $", "P/L %", "Days_Held"):   # object dtype renders blank in number columns
        total[c] = pd.to_numeric(total[c], errors="coerce")
    total = total.drop(columns=[c for c in ("Bought_CT", "Sold_CT") if c in total.columns])
    total = total.rename(columns={"Buy_Price": "Total cost", "Sell_Price": "Total proceeds", "Days_Held": "Avg days held"})
    win = (trips["P/L $"] > 0).mean() * 100
    caption_text(f"Average P/L per sale: ${trips['P/L $'].mean():,.2f} · average P/L %: {trips['P/L %'].mean():.2f}% · win rate: {win:.0f}%")
    table(total, ["P/L $", "P/L %"], cols={**cfg, "Total cost": MONEY, "Total proceeds": MONEY, "Avg days held": INT}, key="trade_audit_total")
