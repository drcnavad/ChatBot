"""Details tab: pre-earnings stops (live account)."""
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.settings import REPORTS


def render_earnings_stops():
    """The live pre-earnings stop (earnings_stop.py, launchd every 10 min): its latest check and the stop sales."""
    path = os.path.join(REPORTS, "earnings_stops.csv")
    try:
        t = pd.read_csv(path)
    except (OSError, ValueError):
        t = None
    if t is None:
        st.info("No check yet: the stop job checks every 10 minutes on trading days, 3:00 AM to 7:00 PM CT.")
    elif t.empty:
        st.caption(f"No held stock has earnings within 7 days (last check "
                   f"{datetime.fromtimestamp(os.path.getmtime(path)):%a %b %-d %-I:%M %p} CT).")
    else:
        st.dataframe(t.drop(columns="Checked_At_CT"), hide_index=True, width="stretch")
        st.caption(f"Last check {t['Checked_At_CT'].iloc[0]} CT. Stop = highest close since entry - 3 × ATR(14); "
                   "Price = the latest quote's mid price.")
    try:
        with open(os.path.join(REPORTS, "earnings_stop_state.json")) as f:
            sold = json.load(f).get("sold", {})
    except (OSError, ValueError):
        sold = {}
    if sold:
        st.caption("Stop sales: " + "; ".join(f"{k.split('|')[0]} before its {k.split('|')[1]} report, "
                                               f"{str(v.get('at'))[:16].replace('T', ' ')} CT" for k, v in sorted(sold.items())))
