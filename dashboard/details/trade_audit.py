"""Trading Account tab: the trade audit (trade_audit.py; read-only, hourly)."""
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.details.live_holdings import holdings_this_run
from dashboard.settings import CT
from dashboard.style import (BAD, CAUTION, GOOD, INK, MONEY, caption_text, info_text, stat_cards, tone, toned, usd,
    warning_text)

# ---------------------------------------------------------------------------- trade audit (read-only)
@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def _read_audit(hour_key, positions):
    """trade_audit.report() once per hour (GET only: account, orders, fills, clock; positions from the holdings read)."""
    import alpaca_paper as ap
    import trade_audit as ta
    return ta.report(ta.fetch(ap.PaperAccount(), positions=list(positions)))

AUDIT_LEDGER_COLS = ["Submitted", "Symbol", "Side", "Source", "Status", "Qty", "Filled_Qty", "Plan_Price", "Limit", "Fill_Price",
                     "Slippage_vs_Plan_%", "Slippage_vs_Plan_$"]

def render_trade_audit():
    """Trading Account tab: every order vs its plan price (slippage), each sale with its buy and sell price, reconciliation with
    Alpaca and alerts (trade_audit.py; read-only, hourly)."""
    import trade_audit as ta
    data, err = holdings_this_run()
    if err:
        caption_text("The trade audit needs the live account, which is not available right now (see Live holdings above).")
        return
    try:
        rep = _read_audit(f"{datetime.now(CT):%Y-%m-%d %H}", tuple(data["positions"]))
    except Exception as e:
        info_text(f"Trade audit unavailable right now ({type(e).__name__}: {str(e)[:200]}). It tries again next hour.")
        return
    alerts = [a for a in rep["findings"] + rep["alerts"] if a["Level"] in ("warning", "failed")]
    for a in alerts:
        warning_text(("Alert: " if a["Level"] == "failed" else "Check: ") + a["Check"])
    if not alerts:
        st.success("No alerts: account active, no stray or untracked orders, daily loss, Mac clock, NYSE calendar, fill "
                   "slippage, rejected orders and Alpaca reads all fine.")
    s, led, trips = rep["summary"], rep["ledger"], rep["round_trips"]
    open_n = int(led["Status"].astype(str).str.lower().isin(ta.OPEN).sum()) if len(led) else 0
    stat_cards([("Bot orders filled (with plan price)", f"{s['filled_with_plan']}", INK),
                ("Traded by them", usd(s["traded"], sign=False), INK),
                ("Cost vs plan price", usd(s["cost"]), BAD if s["cost"] > 0.005 else GOOD if s["cost"] < -0.005 else INK),
                ("Cost vs plan %", f"{s['cost_pct']:+.3f}%", BAD if s["cost_pct"] > 0.0005 else GOOD if s["cost_pct"] < -0.0005 else INK),
                ("Open orders", f"{open_n}", CAUTION if open_n else INK),
                ("Sales matched (FIFO)", f"{len(trips):,}", INK)])
    caption_text(f"As of {rep['as_of']:%a %b %-d %I:%M %p} CT, read from Alpaca hourly (read-only). Plan price = the price the "
                 "bot planned the order at (stamped in its client order id); cost vs plan = (fill - plan) x shares for buys, "
                 "(plan - fill) x shares for sells, so + = it cost money. Manual orders have no plan price. Alerts only: "
                 "nothing here sends, changes or stops an order.")
    checks = pd.DataFrame([{"Check": f["Check"], "Result": f["Level"]} for f in rep["findings"]])
    if len(checks):
        caption_text(f"Reconciliation with Alpaca, as of {rep['as_of']:%a %b %-d %I:%M %p} CT:")
        st.dataframe(checks, hide_index=True, width="stretch", column_config={"Check": st.column_config.TextColumn(width="large")})
    if len(led):
        shown = led.head(100).assign(Submitted=[f"{t:%a %b %-d %I:%M %p}" if t == t else "" for t in led["Submitted_CT"].head(100)])
        st.dataframe(toned(shown[AUDIT_LEDGER_COLS].round(3), ["Slippage_vs_Plan_$"], fn=lambda v: tone(-v)), hide_index=True,
                     width="stretch", height=300,
                     column_config={"Plan_Price": MONEY, "Limit": MONEY, "Fill_Price": MONEY, "Slippage_vs_Plan_$": MONEY,
                                    "Slippage_vs_Plan_%": st.column_config.NumberColumn(format="%+.3f%%"),
                                    "Qty": st.column_config.NumberColumn(format="%.4g"),
                                    "Filled_Qty": st.column_config.NumberColumn(format="%.4g")})
        caption_text(f"Order ledger, newest first ({min(len(led), 100)} of {len(led):,} orders shown; all in "
                     "Reports/live_trade_ledger.csv when trade_audit.py --write runs). Times in CT.")
    if len(trips):
        t = trips.head(100)
        shown = t.assign(**{c: [f"{x:%b %-d, %Y}" if x == x else "" for x in t[c]] for c in ("Bought_CT", "Sold_CT")})
        st.dataframe(toned(shown.round(2), ["P/L $", "P/L %"]), hide_index=True, width="stretch", height=300,
                     column_config={"Buy_Price": MONEY, "Sell_Price": MONEY, "P/L $": MONEY,
                                    "P/L %": st.column_config.NumberColumn(format="%+.2f%%"),
                                    "Days_Held": st.column_config.NumberColumn(format="%d"),
                                    "Shares": st.column_config.NumberColumn(format="%.4g")})
        caption_text(f"Every sale with its purchase (FIFO, oldest shares first, as Alpaca), newest first ({len(t)} of "
                     f"{len(trips):,} shown; all in Reports/live_round_trips.csv). Prices are fill prices; fees are not included.")
