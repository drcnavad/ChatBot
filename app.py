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
Alpaca account (read-only GETs via alpaca_paper.py, at most once a minute). The dashboard itself never places orders except from the bottom-right chat button (dashboard/chat.py), which
asks for your PIN (chat_pin.py) before every order / cancel / close. It never calls
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
from dashboard.chat_ui import render_chat  # noqa: E402
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

@st.cache_resource(show_spinner=False)
def prewarm():
    """Once per server process: fill the slow caches (news feeds, live account, portfolio history) in a background
    thread, so the first visit of each page does not wait for them in turn."""
    import threading
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    ctx = get_script_run_ctx()

    from dashboard.data import refresh_slot
    from dashboard.details import equity_chart as ec
    from dashboard.details.live_holdings import live_holdings
    from dashboard.news import headlines

    def quiet(fn):
        try:
            fn()
        except Exception:               # the page shows its own errors when it reads again
            pass
    for fn in (headlines, live_holdings, lambda: ec.equity_series("1D", refresh_slot())):    # each in its own thread
        t = threading.Thread(target=quiet, args=(fn,), name="sa-prewarm", daemon=True)
        add_script_run_ctx(t, ctx)
        t.start()
    return True

if os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") != "off":
    prewarm()

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
    """The Live / last-updated line only (a fragment). No whole-page rerun: each live panel (cards, sector table, stock
    price, portfolio chart, holdings) is its own 30 s fragment; news, ticker strip, notes and searches never rerun on a timer."""
    from datetime import datetime
    from dashboard.data import refresh_every
    from dashboard.settings import CT
    every = refresh_every()
    st.caption(f"🟢 Live · last updated {datetime.now(CT):%I:%M:%S %p} CT · "
               f"auto-refresh every {'30 s (market open)' if every == 30 else '5 min (market closed)'} · prices and account from Alpaca")

def main():
    try:
        p = build_page()
    except (OSError, ValueError, KeyError) as e:              # no / unreadable signals file: say so, nothing to draw
        st.error(f"Reports/signal_analysis.csv is missing or unreadable ({type(e).__name__}). Run `python run_all.py`, "
                 "then reload this page.")
        st.stop()
    # ?symbol=X (table clicks, Telegram messages) or ?stock=X (ticker strip, old links) opens X in the Stock Analysis tab
    q_symbol = str(st.query_params.get("symbol") or st.query_params.get("stock") or "").strip().upper()
    jumped = q_symbol in p.options
    if jumped:
        st.session_state["_pending_ticker"] = q_symbol
    for k in ("symbol", "stock"):
        if k in st.query_params:
            del st.query_params[k]

    if "_pending_ticker" in st.session_state:                # a link or a row click: show the stock
        st.session_state["nav"] = "Stock Analysis"
    ticker_box = render_top_bar(p)
    auto_refresh()
    page = st.session_state.get("nav") or "Home"             # only the selected page is drawn (no tab reset on reruns)
    if page == "Home":
        render_kpis()                     # the market cards (Home only; refresh themselves every 30 s)
        render_sector_table(p)
    elif page == "Stock Analysis":
        with st.expander("My notes", expanded=True):
            render_notes()
        ticker, tdata = render_stock_picker(p, jumped)
        render_stock(p, ticker, tdata)
    elif page == "Strategy":
        render_strategy_tab(p)
    else:
        render_trading_tab(p)
    render_ticker(ticker_box, p)
    render_news_banner(ticker_box)
    render_chat()                     # floating bottom-right chat button (every tab)

main()
