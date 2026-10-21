"""Trading Account tab: the account's equity over a chosen period (Alpaca portfolio history, read-only GET), with the
period's gain, QQQ on the same starting money and the drawdown. Deposits / withdrawals are taken out of the line
(daily periods), so the line and the gain show what the market did, not money moved in."""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.style import BAD, CHART_FONT, GOOD, show_html, usd

PERIODS = {"1D": ("1D", "5Min"), "5D": ("5D", "15Min"), "14D": ("2W", "1D"), "1M": ("1M", "1D"), "3M": ("3M", "1D"),
           "1Y": ("1A", "1D"), "2Y": ("2A", "1D"), "All": ("10A", "1D")}

@st.cache_data(ttl=60, show_spinner=False)
def equity_series(key):
    """(deposit-adjusted equity = the period's first equity grown by the time-weighted return, the actual equity,
    QQQ closes on the same clock) for PERIODS[key]."""
    import alpaca_paper as ap
    period, tf = PERIODS[key]
    q = {"period": period, "timeframe": tf, "pnl_reset": "no_reset", "cashflow_types": "CSD,CSW,JNLC"}
    if tf != "1D":
        q["intraday_reporting"] = "market_hours"
    h = ap.PaperAccount()._get("/account/portfolio/history", q)
    ts = pd.to_datetime(h.get("timestamp") or [], unit="s", utc=True).tz_convert("America/Chicago")
    raw = pd.Series(pd.to_numeric(pd.Series(h.get("equity") or [], dtype=object), errors="coerce").to_numpy(), index=ts)
    flow = pd.Series([sum(float(v[i] or 0) for v in (h.get("cashflow") or {}).values() if i < len(v or []))
                      for i in range(len(ts))], index=ts)
    prev = raw.shift(1)
    ratio = ((raw - flow) / prev).where((prev > 0) & (raw - flow > 0), 1.0)   # time-weighted: deposits aren't gains
    keep = (raw > 0) & raw.notna()                                            # before funding / after closing: left out
    raw, ratio, flow = raw[keep], ratio[keep], flow[keep]
    if len(raw) < 2:
        return raw, raw, pd.Series(dtype=float)
    ratio.iloc[0] = 1.0
    eq = raw.iloc[0] * ratio.cumprod()                 # the period's first equity grown by the time-weighted return
    eq.attrs["pnl"] = float(raw.iloc[-1] - raw.iloc[0] - flow.iloc[1:].sum())   # market gain in $, net of cash flows
    if tf == "1D":
        eq.index = raw.index = eq.index.tz_localize(None).normalize()
    return eq, raw, _qqq(eq.index, tf)

def _qqq(index, tf):
    """QQQ closes at the same times (Alpaca bars, SIP then IEX); empty when unavailable."""
    if not len(index):
        return pd.Series(dtype=float)
    try:
        import backtest_engine as be
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        frame = TimeFrame.Day if tf == "1D" else TimeFrame(int(tf[:-3]), TimeFrameUnit.Minute)
        start = pd.Timestamp(index[0]).tz_localize("America/Chicago") if index.tz is None else index[0]
        client = be.market_data_client()
        for feed in ("sip", "iex"):
            try:
                bars = client.get_stock_bars(StockBarsRequest(symbol_or_symbols="QQQ", timeframe=frame,
                                                              start=(start - pd.Timedelta(days=3)).to_pydatetime(),
                                                              feed=DataFeed(feed))).data.get("QQQ", [])
                break
            except Exception:
                bars = []
        s = pd.Series({pd.Timestamp(b.timestamp).tz_convert("America/Chicago"): float(b.close) for b in bars}).sort_index()
        if tf == "1D":
            s.index = s.index.tz_localize(None).normalize()
            return s.reindex(index, method="ffill")
        return s.reindex(index, method="ffill", tolerance=pd.Timedelta(minutes=30))
    except Exception:
        return pd.Series(dtype=float)

def equity_figure(eq, qqq, show_qqq=True, show_dd=True, intraday=False):
    """The equity line (green / red fill vs the period start), dashed start baseline, QQQ on the same money, drawdown."""
    base, last = float(eq.iloc[0]), float(eq.iloc[-1])
    color = GOOD if last >= base else BAD
    fill = "rgba(21,128,61,0.10)" if last >= base else "rgba(185,28,28,0.10)"
    fig = make_subplots(rows=2 if show_dd else 1, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.75, 0.25] if show_dd else [1])
    fig.add_trace(go.Scatter(x=eq.index, y=[base] * len(eq), mode="lines", line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Portfolio", fill="tonexty", fillcolor=fill,
                             line=dict(color=color, width=2.2), hovertemplate="Portfolio $%{y:,.2f}<extra></extra>"), row=1, col=1)
    if show_qqq and qqq is not None and qqq.notna().any():
        qn = qqq / qqq.dropna().iloc[0] * base
        fig.add_trace(go.Scatter(x=qn.index, y=qn.values, mode="lines", name="QQQ (same start)",
                                 line=dict(color="#64748b", width=1.4, dash="dot"),
                                 hovertemplate="QQQ $%{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=base, line=dict(color="#94a3b8", width=1, dash="dash"), row=1, col=1)
    if show_dd:
        dd = (eq / eq.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown", fill="tozeroy",
                                 line=dict(color=BAD, width=1), fillcolor="rgba(185,28,28,0.12)", showlegend=False,
                                 hovertemplate="Drawdown %{y:.2f}%<extra></extra>"), row=2, col=1)
        fig.update_yaxes(title_text="Drawdown", ticksuffix="%", tickformat=".1f", row=2, col=1)
    fig.update_layout(height=430 if show_dd else 340, margin=dict(l=10, r=70, t=10, b=10), hovermode="x unified",
                      plot_bgcolor="#fff", paper_bgcolor="#fff", font=dict(family=CHART_FONT, size=11, color="#334155"),
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=11)),
                      hoverlabel=dict(bgcolor="#fff", bordercolor="#e2e8f0", font_family=CHART_FONT))
    fig.update_xaxes(showgrid=False, showspikes=True, spikemode="across", spikethickness=1, spikecolor="#94a3b8",
                     spikedash="dot", tickformat="%-I:%M %p" if intraday else "%b %-d",
                     rangebreaks=[dict(bounds=["sat", "mon"])] + ([dict(bounds=[15, 8.5], pattern="hour")] if intraday else []))
    fig.update_yaxes(side="right", gridcolor="#f1f5f9", zeroline=False, tickfont=dict(size=10, color="#64748b"))
    fig.update_yaxes(tickformat="$,.0f", hoverformat="$,.2f", row=1, col=1)
    return fig

@st.fragment
def render_equity_chart():
    if os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off":
        st.info("Portfolio chart is turned off here (STOCK_ANALYSIS_LIVE_HOLDINGS=off).")
        return
    c1, c2, c3 = st.columns([5, 1.2, 1.2], vertical_alignment="center")
    key = c1.radio("Period", list(PERIODS), index=list(PERIODS).index("1M"), key="eq_period", horizontal=True,
                   label_visibility="collapsed")   # radio, not segmented_control: AppTest cannot rerun button groups
    show_qqq = c2.toggle("QQQ", value=True, key="eq_qqq")
    show_dd = c3.toggle("Drawdown", value=True, key="eq_dd")
    try:
        eq, raw, qqq = equity_series(key)
    except Exception as e:
        st.info(f"Portfolio history unavailable right now ({type(e).__name__}). It tries again in a minute.")
        return
    if len(eq) < 2:
        st.info("Not enough portfolio history for this period yet.")
        return
    base, last = float(eq.iloc[0]), float(eq.iloc[-1])
    pct = (last / base - 1) * 100
    chg = eq.attrs.get("pnl", last - base)
    qtxt = ""
    if qqq is not None and qqq.notna().sum() > 1:
        qq = qqq.dropna()
        qtxt = f' · QQQ {(qq.iloc[-1] / qq.iloc[0] - 1) * 100:+.2f}%'
    show_html(f'<div class="sa-eq-head"><span class="sa-eq-val">${float(raw.iloc[-1]):,.2f}</span>'
              f'<span class="sa-eq-chg" style="color:{GOOD if chg >= 0 else BAD}">{usd(chg)} ({pct:+.2f}%)</span>'
              f'<span class="sa-eq-sub">{key} · deposit-adjusted{qtxt}</span></div>')
    st.plotly_chart(equity_figure(eq, qqq, show_qqq, show_dd, intraday=PERIODS[key][1] != "1D"), width="stretch",
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}, key="eq_chart")
