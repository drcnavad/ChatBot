"""Strategy tab: data freshness and settings."""
import streamlit as st

from dashboard.style import table

def render_data_and_settings(p):
    """Data freshness table."""
    table(p.freshness, ["Status"])
    stale = p.freshness[p.freshness["Status"].str.startswith("⚠️")]
    if not stale.empty:
        st.warning("Stale or missing: " + ", ".join(stale["File"]) + " — run `python run_all.py`.")
    else:
        st.success("All report files are within their expected refresh window.")
