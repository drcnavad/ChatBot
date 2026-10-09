"""
Stock Analysis dashboard (Streamlit).

Page layout, top to bottom:
  1. Title bar.
  2. "Home" tab: your notes table first (editable, saved to Reports/my_notes.csv), then the single-stock view
     (clickable rank tiers, stock picker, chart). Open any stock directly with http://localhost:8502/?symbol=NVDA
  3. "Strategy" tab: last decision, forward test, strategy rules and holdings, data freshness and settings.
  4. "Trading Account" tab: live holdings, latest signals, trade audit, tax view, strategy rules.

The app only READS the Reports/*.csv files written by `python run_all.py` for strategy data, plus the live holdings from the
Alpaca account (read-only GETs via alpaca_paper.py, at most once a minute). It never places orders and never calls
a paid data API (the optional "AI analysis" button uses the Hugging Face token from .env). The stock header additionally
shows a display-only live quote from yfinance (free), and yfinance also draws the price chart of stocks too new to trade
(short history); neither ever feeds back into signals, picks, backtests, or orders.

Code layout: this file is only the entry point (page setup + the three tabs); every tab and panel lives in the dashboard/
package (dashboard/__init__.py lists the modules; dashboard/details/ has one module per Details expander).
"""
import os
import sys

import streamlit as st
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)  # project modules (backtest_engine, sector_mapping, dashboard) importable from any cwd

from dashboard.details import render_strategy_tab, render_trading_tab  # noqa: E402
from dashboard.details.live_holdings import new_run  # noqa: E402
from dashboard.notes import render_notes  # noqa: E402
from dashboard.page import build_page, render_top_bar  # noqa: E402
from dashboard.short_stock import render_short_stock  # noqa: E402
from dashboard.stock_chart import render_stock_figure, stock_chart_inputs  # noqa: E402
from dashboard.stock_view import render_stock_picker, render_stock_section  # noqa: E402
from dashboard.style import setup_page  # noqa: E402

load_dotenv()
setup_page()      # the first Streamlit call, on every run
new_run()

def main():
    with st.spinner("Loading data..."):
        p = build_page()
    # ?symbol=X opens that stock (table clicks and the Telegram messages use this)
    q_symbol = str(st.query_params.get("symbol", "")).strip().upper()
    jumped = q_symbol in p.options
    if jumped:
        st.session_state["_pending_ticker"] = q_symbol
        del st.query_params["symbol"]

    render_top_bar(p)
    tab_home, tab_strategy, tab_trading = st.tabs(["Home", "Strategy", "Trading Account"])
    with tab_home:
        with st.expander("My notes", expanded=True):
            render_notes()
        ticker, tdata = render_stock_picker(p, jumped)
        if ticker in p.short.index:
            render_short_stock(ticker, p.short.loc[ticker])
        else:
            render_stock_section(p, ticker, tdata)
            fig = stock_chart_inputs(ticker, tdata)
            render_stock_figure(fig, ticker)
    with tab_strategy:
        render_strategy_tab(p)
    with tab_trading:
        render_trading_tab(p)

main()
