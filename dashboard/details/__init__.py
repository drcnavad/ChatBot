"""Details tabs: Strategy and Trading Account, every panel in its own expander."""
import os

import streamlit as st

from dashboard.details.data_settings import render_data_and_settings
from dashboard.details.equity_chart import render_equity_chart
from dashboard.details.forward_test import render_forward_test, render_live_push, render_top10
from dashboard.details.last_decision import render_last_decision
from dashboard.details.live_holdings import render_live_holdings
from dashboard.details.tax_view import render_tax_view
from dashboard.details.trade_audit import render_trade_audit
from dashboard.details.drop_guard import render_drop_guard
from dashboard.settings import rules_text


def _lazy(title, key, render_fn, expanded=False, always=False):
    """Expander that skips heavy work until opened (collapsed panels cost a button click, not a cold Alpaca/tax load).
    STOCK_ANALYSIS_LAZY_EXPANDERS=off (tests) renders every panel immediately so AppTest can read markdown/tables."""
    if os.getenv("STOCK_ANALYSIS_LAZY_EXPANDERS", "on") == "off":
        always = True
    with st.expander(title, expanded=expanded):
        if always or expanded or st.session_state.get(f"_lazy_{key}"):
            render_fn()
        elif st.button(f"Load · {title.split('·')[0].strip()}", key=f"_lazy_btn_{key}"):
            st.session_state[f"_lazy_{key}"] = True
            st.rerun()


def render_strategy_tab(p):
    """Strategy tab: last decision, forward test (Top 5 consensus, chart, leaderboard with rule/holdings on hover), data freshness and settings."""
    with st.expander("Last decision · portfolio and short-term relative strength ranks", expanded=True):
        render_last_decision(p)
    import forward_test as ft
    with st.expander(f"Forward test · {len(ft.STRATEGIES)} strategies vs your account, QQQ and SPY since Oct 29, 2026",
                     expanded=True):
        render_top10(p)
        from dashboard.details.forward_test import render_current_holdings
        render_current_holdings()
        render_forward_test(p)
        render_live_push()
    _lazy("Data freshness and settings", "data_settings", lambda: render_data_and_settings(p), expanded=False)


def _prefetch():
    """Read the tab's slow live-account data in parallel (portfolio history, then trade audit and tax inputs once the
    fills are known) so each panel below finds it cached: the cold load costs the slowest read, not their sum.
    Skips audit/tax prefetch until those expanders are opened (lazy)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    from dashboard.data import refresh_slot
    from dashboard.details import equity_chart as ec
    ctx = get_script_run_ctx()

    def quiet(fn, *a):
        add_script_run_ctx(threading.current_thread(), ctx)
        try:
            fn(*a)
        except Exception:                   # the panel itself shows the error
            pass
    with ThreadPoolExecutor(2) as ex:
        if os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") != "off":
            ex.submit(quiet, ec.equity_series, ec.DATA.get(st.session_state.get("eq_view", "1D"), ec.SINCE), refresh_slot())
        # Warm the holdings cache used by Live holdings + Strategy live_returns (same refresh slot).
        if os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") != "off":
            from dashboard.details.live_holdings import holdings_this_run
            ex.submit(quiet, holdings_this_run)


def render_trading_tab(p):
    """Trading Account tab: live holdings, trade audit, tax view, strategy rules (the portfolio and the RS short-term
    ranks are in the Strategy tab's Last decision)."""
    _prefetch()
    with st.expander("Portfolio value", expanded=True):
        render_equity_chart()
    with st.expander("Live holdings (Alpaca account)", expanded=True):
        render_live_holdings()

    _lazy("Drop guard · intraday AI sell / replace", "drop_guard", render_drop_guard)

    def _audit():
        from datetime import datetime
        from dashboard.details.live_holdings import holdings_this_run
        from dashboard.details import trade_audit as ta
        from dashboard.settings import CT
        data, err = holdings_this_run()
        if not err:
            key = f"{datetime.now(CT):%Y-%m-%d %H}-{len(data['fills'])}"
            ta._read_audit(key, tuple(data["positions"]), tuple(data["fills"]))
        render_trade_audit()

    def _tax():
        from datetime import datetime
        from dashboard.details.live_holdings import holdings_this_run
        from dashboard.details import tax_view as tv
        from dashboard.settings import CT
        data, err = holdings_this_run()
        if not err:
            key = f"{datetime.now(CT):%Y-%m-%d %H}-{len(data['fills'])}"
            tv._read_tax_inputs(key)
        render_tax_view(p)

    _lazy("Trade audit · gain / loss per sale (read-only)", "trade_audit", _audit)
    import tax_lots as tl
    _lazy(f"Tax view · live account since {tl.TAX_START:%b %-d, %Y} (estimate, not tax advice)", "tax_view", _tax)
    _lazy("Strategy rules", "rules", lambda: st.markdown(rules_text()))
