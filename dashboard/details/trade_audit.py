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
    shown = trips.assign(**{c: [f"{x:%b %-d, %Y}" if x == x else "" for x in trips[c]] for c in ("Bought_CT", "Sold_CT")})
    cfg = {"Buy_Price": MONEY, "Sell_Price": MONEY, "P/L $": MONEY, "P/L %": SPCT, "Days_Held": INT, "Shares": SHARES}
    table(shown.round(2), ["P/L $", "P/L %"], max_h=300, cols=cfg)
    total = pd.DataFrame([{c: None for c in shown.columns}]).astype(object)     # its own table: never sorted with the rows
    total.loc[0, shown.columns[0]] = "Total"
    total.loc[0, "P/L $"] = round(float(trips["P/L $"].sum()), 2)
    table(total, ["P/L $"], cols=cfg, key="trade_audit_total")
