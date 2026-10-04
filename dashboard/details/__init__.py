"""Details tab: every panel in its own expander, top to bottom (rules last)."""
import streamlit as st

from dashboard.details.data_settings import render_data_and_settings
from dashboard.details.earnings_planner import render_earnings_planner
from dashboard.details.earnings_stops import render_earnings_stops
from dashboard.details.forward_test import render_forward_rules, render_forward_test
from dashboard.details.last_decision import render_last_decision
from dashboard.details.latest_signals import latest_signals_title, render_latest_signals
from dashboard.details.live_holdings import render_live_holdings
from dashboard.details.tax_view import render_tax_view
from dashboard.details.trade_audit import render_trade_audit
from dashboard.settings import rules_text


def render_details(p):
    with st.expander("Live holdings (Alpaca account)", expanded=True):
        render_live_holdings()
    import forward_test as ft
    with st.expander(f"Forward test · {len(ft.STRATEGIES)} strategies vs your account, QQQ and SPY since Oct 2, 2026",
                     expanded=True):
        render_forward_test()
    with st.expander("Strategy rules and holdings (paper strategies)", expanded=False):
        render_forward_rules()
    with st.expander(latest_signals_title(p), expanded=True):
        render_latest_signals(p)
    with st.expander(f"Last decision · {p.off_date:%a %b %-d} (decisions in force, every stock)", expanded=False):
        render_last_decision(p)
    with st.expander("Earnings planner · held stocks reporting in the next 14 days", expanded=False):
        render_earnings_planner()
    with st.expander("Pre-earnings stops (3× ATR, live account)", expanded=False):
        render_earnings_stops()
    import tax_lots as tl
    with st.expander(f"Tax view · live account since {tl.TAX_START:%b %-d, %Y} (estimate, not tax advice)", expanded=False):
        render_tax_view(p)
    with st.expander("Trade audit · every order vs its plan price, buy / sell prices, checks (read-only)", expanded=False):
        render_trade_audit()
    with st.expander("Data freshness and settings", expanded=False):
        render_data_and_settings(p)
    with st.expander("Strategy rules", expanded=False):
        st.markdown(rules_text())
