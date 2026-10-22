"""The dashboard's one Alpaca market-data client (read-only; backtest_engine.market_data_client, built once per server
process) and the SIP -> IEX fallback every display fetch uses (as paper_trade's quotes). Display only."""
import streamlit as st

@st.cache_resource(show_spinner=False)
def client():
    import backtest_engine as be
    return be.market_data_client()

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
