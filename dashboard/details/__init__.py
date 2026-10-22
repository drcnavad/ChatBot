"""Details tabs: Strategy and Trading Account, every panel in its own expander."""
import os

import streamlit as st

from dashboard.details.data_settings import render_data_and_settings
from dashboard.details.equity_chart import render_equity_chart
from dashboard.details.forward_test import render_forward_test
from dashboard.details.last_decision import render_last_decision
from dashboard.details.live_holdings import render_live_holdings
from dashboard.details.manual_orders import render_manual_orders
from dashboard.details.tax_view import render_tax_view
from dashboard.details.trade_audit import render_trade_audit
from dashboard.settings import rules_text

def render_strategy_tab(p):
    """Strategy tab: last decision, forward test (Top 5 consensus, chart, leaderboard with rule/holdings on hover), data freshness and settings."""
    with st.expander("Last decision · portfolio and both strategies' ranks", expanded=True):
        render_last_decision(p)
    import forward_test as ft
    with st.expander(f"Forward test · {len(ft.STRATEGIES)} strategies vs your account, QQQ and SPY since Oct 2, 2026",
                     expanded=True):
        render_forward_test(p)
    with st.expander("Manual orders and buy blocks (manual_trade.py)", expanded=False):
        render_manual_orders()
    with st.expander("Data freshness and settings", expanded=False):
        render_data_and_settings(p)

def _prefetch():
    """Read the tab's slow live-account data in parallel (portfolio history, then trade audit and tax inputs once the
    fills are known) so each panel below finds it cached: the cold load costs the slowest read, not their sum."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    from dashboard.data import refresh_slot
    from dashboard.details import equity_chart as ec, tax_view as tv, trade_audit as ta
    from dashboard.details.live_holdings import holdings_this_run
    from dashboard.settings import CT
    ctx = get_script_run_ctx()

    def quiet(fn, *a):
        add_script_run_ctx(threading.current_thread(), ctx)
        try:
            fn(*a)
        except Exception:                   # the panel itself shows the error
            pass
    with ThreadPoolExecutor(3) as ex:
        if os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") != "off":
            ex.submit(quiet, ec.equity_series, ec.DATA.get(st.session_state.get("eq_view", "1D"), ec.SINCE), refresh_slot())
        data, err = holdings_this_run()
        if not err:
            key = f"{datetime.now(CT):%Y-%m-%d %H}-{len(data['fills'])}"
            ex.submit(quiet, ta._read_audit, key, tuple(data["positions"]), tuple(data["fills"]))
            ex.submit(quiet, tv._read_tax_inputs, key)

def render_trading_tab(p):
    """Trading Account tab: live holdings, trade audit, tax view, strategy rules (the portfolio by half and both strategies'
    ranks are in the Strategy tab's Last decision)."""
    _prefetch()
    with st.expander("Portfolio value", expanded=True):
        render_equity_chart()
    with st.expander("Live holdings (Alpaca account)", expanded=True):
        render_live_holdings()
    with st.expander("Trade audit · gain / loss per sale (read-only)", expanded=False):
        render_trade_audit()
    import tax_lots as tl
    with st.expander(f"Tax view · live account since {tl.TAX_START:%b %-d, %Y} (estimate, not tax advice)", expanded=False):
        render_tax_view(p)
    with st.expander("Strategy rules", expanded=False):
        st.markdown(rules_text())
