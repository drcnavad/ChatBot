"""
Stock Analysis dashboard (Streamlit).

Page layout, top to bottom:
  1. Title bar.
  2. "Home" tab: sector performance (header cards, held-stock ticker strip and news banner sit above every tab).
  3. "Stock Analysis" tab: your notes table (editable, saved to Reports/my_notes.csv), then the single-stock view
     (clickable stock list, stock picker, AI analysis, card, news, chart with drawings). Open any stock directly with
     http://localhost:8502/?symbol=NVDA (or ?stock=NVDA); a table row click opens it there too.
  4. "Strategy" tab: last decision, forward test, strategy rules and holdings, data freshness and settings.
  5. "Trading Account" tab: portfolio value, live holdings, trade audit, tax view, strategy rules.

The app only READS the Reports/*.csv files written by `python run_all.py` for strategy data, plus the live holdings from the
Alpaca account (read-only GETs via alpaca_paper.py, at most once a minute). It never places orders and never calls
a paid data API (the optional "AI analysis" button uses the Hugging Face token from .env). The stock header additionally
shows a display-only live quote from yfinance (free), and yfinance also draws the price chart of stocks too new to trade
(short history); neither ever feeds back into signals, picks, backtests, or orders.

Code layout: this file is only the entry point (page setup + the four tabs); every tab and panel lives in the dashboard/
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
from dashboard.news import render_news_banner, render_stock_news  # noqa: E402
from dashboard.page import build_page, render_kpis, render_ticker, render_top_bar  # noqa: E402
from dashboard.sector_table import render_sector_table  # noqa: E402
from dashboard.short_stock import render_short_stock  # noqa: E402
from dashboard.stock_chart import render_stock_chart  # noqa: E402
from dashboard.stock_view import render_stock_picker, render_stock_section  # noqa: E402
from dashboard.style import setup_page  # noqa: E402

load_dotenv()
setup_page()      # the first Streamlit call, on every run
new_run()

def render_stock(p, ticker, tdata):
    """One stock: its card and chart (short-history stocks: their own short view)."""
    if ticker in p.short.index:
        render_short_stock(ticker, p.short.loc[ticker])
    else:
        render_stock_section(p, ticker, tdata)
        render_stock_news(ticker)
        render_stock_chart(ticker, tdata)

@st.fragment(run_every=30)
def auto_refresh():
    """Whole-page live refresh: checks every 30 s and reruns the full page when the refresh slot moves (every 30 s in
    market hours, every 5 min outside), so every panel, quote and Reports file (mtime-keyed caches) is re-read."""
    from datetime import datetime
    from dashboard.data import refresh_every, refresh_slot
    from dashboard.settings import CT
    slot = refresh_slot()
    if st.session_state.get("_app_slot") not in (None, slot):
        st.session_state["_app_slot"] = slot
        st.rerun(scope="app")
    every = refresh_every()
    st.caption(f"🟢 Live · last updated {st.session_state.get('_app_updated', datetime.now(CT)):%I:%M:%S %p} CT · "
               f"auto-refresh every {'30 s (market open)' if every == 30 else '5 min (market closed)'} · prices and account from Alpaca")

def main():
    from datetime import datetime
    from dashboard.data import refresh_slot
    from dashboard.settings import CT
    st.session_state["_app_slot"], st.session_state["_app_updated"] = refresh_slot(), datetime.now(CT)
    with st.spinner("Loading data..."):
        p = build_page()
    # ?symbol=X (table clicks, Telegram messages) or ?stock=X (ticker strip, old links) opens X in the Stock Analysis tab
    q_symbol = str(st.query_params.get("symbol") or st.query_params.get("stock") or "").strip().upper()
    jumped = q_symbol in p.options
    if jumped:
        st.session_state["_pending_ticker"] = q_symbol
    for k in ("symbol", "stock"):
        if k in st.query_params:
            del st.query_params[k]

    ticker_box = render_top_bar(p)
    auto_refresh()
    opening = "_pending_ticker" in st.session_state          # a link or a row click: show the stock
    tab_home, tab_stock, tab_strategy, tab_trading = st.tabs(
        ["Home", "Stock Analysis", "Strategy", "Trading Account"], default="Stock Analysis" if opening else None)
    with tab_home:
        render_kpis()                     # the market cards (Home only; refresh themselves every 30 s)
        render_sector_table(p)
    with tab_stock:
        with st.expander("My notes", expanded=True):
            render_notes()
        ticker, tdata = render_stock_picker(p, jumped)
        render_stock(p, ticker, tdata)
    with tab_strategy:
        render_strategy_tab(p)
    with tab_trading:
        render_trading_tab(p)
    render_ticker(ticker_box)
    render_news_banner(ticker_box)

main()
