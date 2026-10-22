"""The dashboard's live-data layer (with live_stream.py for the websocket): the market clock and refresh slot, one Alpaca market-data client (read-only; backtest_engine.market_data_client, built once per server
process) and the SIP -> IEX fallback every display fetch uses (as paper_trade's quotes). Display only."""
import streamlit as st

@st.cache_resource(show_spinner=False)
def client():
    import backtest_engine as be
    return be.market_data_client()

@st.cache_data(ttl=60, show_spinner=False)
def market_open():
    """True during the regular session: Alpaca's clock (read-only GET /clock; knows holidays), else Mon-Fri
    9:30-16:00 New York."""
    try:
        import alpaca_paper as ap
        return bool(ap.PaperAccount()._get("/clock").get("is_open"))
    except Exception:
        import pandas as pd
        now = pd.Timestamp.now(tz="America/New_York")
        return now.weekday() < 5 and 570 <= now.hour * 60 + now.minute < 960

def refresh_every():
    """Seconds between live-data refreshes: 30 in the regular session, 300 outside it."""
    return 30 if market_open() else 300

def refresh_slot():
    """A number that changes every refresh_every() seconds: the cache key of the live (Alpaca) account reads."""
    import time
    return int(time.time() // refresh_every())

def sip_then_iex(fetch):
    """fetch(DataFeed) on SIP, then IEX if SIP fails -> (result, feed name); raises the last error if both fail."""
    from alpaca.data.enums import DataFeed
    err = None
    for feed in ("sip", "iex"):
        try:
            return fetch(DataFeed(feed)), feed
        except Exception as e:
            err = e
    raise err
