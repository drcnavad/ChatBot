"""Home tab: stocks too new to trade (short history; yfinance chart, display only)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import _YF_OK, _yf, read_report_csv
from dashboard.settings import SHORT_HISTORY_CSV
from dashboard.style import CHART_FONT, MA_COLORS, TEAL, md_tone, section, tone


def load_short_history():
    """Reports/short_history_reference.csv (stocks in the list with < 200 days of prices); None when missing or empty."""
    d = read_report_csv(SHORT_HISTORY_CSV)
    return None if d is None or d.empty else d


@st.cache_data(ttl=3600)
def short_history_closes(symbol, start):
    """Display-only daily closes from yfinance for a stock too new to trade; None when unavailable."""
    if not _YF_OK:
        return None
    try:
        df = _yf.download(symbol, start=start, interval="1d", progress=False, auto_adjust=True)
        closes = df["Close"]
        closes = (closes.iloc[:, 0] if isinstance(closes, pd.DataFrame) else closes).dropna()
        return closes if len(closes) else None
    except Exception:
        return None


def render_short_stock(sym, r):
    """Stock view for a stock in the list with fewer than 200 trading days: never scored, ranked or traded.
    Numbers come from Reports/short_history_reference.csv; the chart is display-only (yfinance)."""
    day = lambda d: f"{pd.Timestamp(d):%a %b %-d, %Y}"
    section(f"{sym} · {r['Name']}")
    st.warning(f"Not traded yet (short history): {int(r['Days_Of_History'])} of {int(r['Days_Needed'])} trading days, "
               f"first traded {day(r['First_Trade'])}. The strategy does not score, rank or buy it until its 200th trading day "
               f"(about {day(r['Est_Eligible_Date'])}, est.).")
    c = st.columns(6)
    c[0].metric(f"Close · {pd.Timestamp(r['Last_Date']):%a %b %-d}", f"${r['Last_Close']:,.2f}")
    c[1].metric("Rough signal (less reliable)", r["Rough_Signal"])
    c[2].metric("RSI 14", f"{r['RSI_14']:.1f}")
    for i, n in enumerate((10, 30, 50)):
        c[3 + i].metric(f"MA {n}", f"${r[f'MA_{n}']:,.2f}")
    ret = lambda v: md_tone(f"{v:+.1f}%", tone(v))
    st.caption(f"Since first close {ret(r['Return_Since_First_Close_%'])} · 21-day {ret(r['Return_21d_%'])} · "
               f"63-day {ret(r['Return_63d_%'])} · {abs(r['Off_High_%']):.1f}% below its high of ${r['High_Since_First']:,.2f}. "
               "Rough signal: Buy if the close is above the 10-day, the 10-day above the 30-day and the 30-day above the 50-day "
               "average with RSI 50-70; Sell if the close is below both the 30- and 50-day averages or RSI is under 40; otherwise "
               "Hold. Less reliable than the real signal (no 200-day history) and never used for trading.")
    closes = short_history_closes(sym, str(r["First_Trade"]))
    if closes is None:
        st.caption("Price chart unavailable right now (yfinance).")
        return
    fig = go.Figure(go.Scatter(x=closes.index, y=closes, name="Close", line=dict(color=TEAL, width=2.5)))
    for n, color in zip((10, 30, 50), MA_COLORS):
        fig.add_trace(go.Scatter(x=closes.index, y=closes.rolling(n).mean(), name=f"MA {n}", line=dict(color=color, width=1)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(type="date"),
                      title=f"{sym} daily close (display only, yfinance)", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                      font=dict(family=CHART_FONT, size=11, color="#334155"), yaxis=dict(tickformat="$,.0f", gridcolor="#eef2f7"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="#f8fafc",
                                  bordercolor="#e2e8f0", borderwidth=1, title=dict(text="<b>Legend</b>  ")))
    st.plotly_chart(fig, width="stretch", key=f"short_chart_{sym}")
