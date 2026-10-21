"""Trading Account tab: the trade audit (trade_audit.py; read-only, hourly)."""
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.details.live_holdings import holdings_this_run
from dashboard.settings import CT
from dashboard.style import MONEY, caption_text, info_text, toned

# ---------------------------------------------------------------------------- trade audit (read-only)
@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def _read_audit(hour_key, _positions):
    """trade_audit.report() once per hour_key (the hour + the number of fills; GET only: account, orders, fills, clock;
    positions from the holdings read, not part of the cache key since their prices change every minute)."""
    import alpaca_paper as ap
    import trade_audit as ta
    return ta.report(ta.fetch(ap.PaperAccount(), positions=list(_positions)))

def render_trade_audit():
    """Trading Account tab: each sale with its purchase (FIFO) and its gain / loss, plus a fixed Total row below
    (trade_audit.py; read-only, hourly). Oct 21, 2026 (Chirag): the alerts, cards and order ledger were removed."""
    data, err = holdings_this_run()
    if err:
        caption_text("The trade audit needs the live account, which is not available right now (see Live holdings above).")
        return
    try:
        rep = _read_audit(f"{datetime.now(CT):%Y-%m-%d %H}-{len(data['fills'])}", tuple(data["positions"]))
    except Exception as e:
        info_text(f"Trade audit unavailable right now ({type(e).__name__}: {str(e)[:200]}). It tries again next hour.")
        return
    trips = rep["round_trips"]
    if not len(trips):
        caption_text("No sales yet.")
        return
    shown = trips.assign(**{c: [f"{x:%b %-d, %Y}" if x == x else "" for x in trips[c]] for c in ("Bought_CT", "Sold_CT")})
    cfg = {"Buy_Price": MONEY, "Sell_Price": MONEY, "P/L $": MONEY, "P/L %": st.column_config.NumberColumn(format="%+.2f%%"),
           "Days_Held": st.column_config.NumberColumn(format="%d"), "Shares": st.column_config.NumberColumn(format="%.4g")}
    st.dataframe(toned(shown.round(2), ["P/L $", "P/L %"]), hide_index=True, width="stretch", height=300, column_config=cfg)
    total = pd.DataFrame([{c: None for c in shown.columns}]).astype(object)     # its own table: never sorted with the rows
    total.loc[0, shown.columns[0]] = "Total"
    total.loc[0, "Shares"], total.loc[0, "P/L $"] = round(float(trips["Shares"].sum()), 4), round(float(trips["P/L $"].sum()), 2)
    st.dataframe(toned(total, ["P/L $"]), hide_index=True, width="stretch", column_config=cfg, key="trade_audit_total")
