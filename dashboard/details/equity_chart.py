"""Trading Account tab: the account's equity over a chosen period (Alpaca portfolio history, read-only GET), with the
period's gain, QQQ / SPY / DIA bought with the same cash flows and the drawdown. The line is the real equity in $ (from Oct 2); the
$ gain and the % return leave deposits / withdrawals out, so money moved in is never a gain (nor out a loss)."""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.style import BAD, CHART_FONT, GOOD, show_html, usd

START = pd.Timestamp("2026-10-02")          # the live account's history starts here; nothing earlier is shown
BENCH = {"QQQ": "#64748b", "SPY": "#d97706", "DIA": "#7c3aed"}   # comparisons: Nasdaq-100, S&P 500, Dow
SINCE = f"Since {START:%b %-d}"
PERIODS = {"1D": ("1D", "5Min"), "5D": ("5D", "15Min"), "14D": ("2W", "1D"), SINCE: ("3M", "1D")}

@st.cache_data(ttl=60, show_spinner=False)
def equity_series(key):
    """(DataFrame equity (actual $), flow (deposits - withdrawals on that bar), twr (time-weighted growth), invested,
    {QQQ, SPY, DIA: closes on the same clock}) for PERIODS[key]."""
    import alpaca_paper as ap
    period, tf = PERIODS[key]
    q = {"period": period, "timeframe": tf, "pnl_reset": "no_reset", "cashflow_types": "CSD,CSW,JNLC"}
    if tf != "1D":
        q["intraday_reporting"] = "market_hours"
    h = ap.PaperAccount()._get("/account/portfolio/history", q)
    ts = pd.to_datetime(h.get("timestamp") or [], unit="s", utc=True).tz_convert("America/Chicago")
    raw = pd.Series(pd.to_numeric(pd.Series(h.get("equity") or [], dtype=object), errors="coerce").to_numpy(), index=ts)
    # deposits (CSD), withdrawals (CSW) and cash journals (JNLC) as portfolio history books them: on the bar whose
    # equity first includes them (a transfer shows in the equity a trading day after its activity date)
    cf = h.get("cashflow") or {}
    flow = pd.Series([sum(float(v[i] or 0) for v in cf.values() if i < len(v or [])) for i in range(len(ts))], index=ts)
    if tf == "1D":                                      # daily bars are stamped at midnight New York time of their day
        raw.index = flow.index = ts.tz_convert("America/New_York").tz_localize(None).normalize()
        acct = ap.PaperAccount()._get("/account")                 # today's live equity as the last point
        today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
        if float(acct.get("equity") or 0) > 0 and (not len(raw) or raw.index[-1] < today):
            raw.loc[today], flow.loc[today] = float(acct["equity"]), 0.0
    keep_from = raw.index.tz_localize(None) >= START if raw.index.tz is not None else raw.index >= START
    raw, flow = raw[keep_from], flow[keep_from]
    prev = raw.shift(1)
    ratio = ((raw - flow) / prev).where((prev > 0) & (raw - flow > 0), 1.0)   # time-weighted: deposits aren't gains
    keep = (raw > 0) & raw.notna()                                            # before funding / after closing: left out
    raw, ratio, flow = raw[keep], ratio[keep], flow[keep]
    if len(raw) < 2:
        return pd.DataFrame(columns=["equity", "flow", "twr", "invested"]), pd.Series(dtype=float)
    ratio.iloc[0] = 1.0
    flow.iloc[0] = 0.0                                 # money already in at the start is the starting balance
    out = pd.DataFrame({"equity": raw, "flow": flow, "twr": ratio.cumprod()})
    out["invested"] = raw.iloc[0] + flow.cumsum()      # starting balance + deposits - withdrawals since
    return out, {sym: _bench(sym, out.index, tf) for sym in BENCH}

def gains(d):
    """The period's numbers, cash flows taken out: $ gain = end - start - net deposits; % = time-weighted return."""
    start, end, net = float(d["equity"].iloc[0]), float(d["equity"].iloc[-1]), float(d["flow"].sum())
    return {"start": start, "end": end, "net_flows": net, "gain": end - start - net,
            "twr_pct": (float(d["twr"].iloc[-1]) - 1) * 100}

def _bench(sym, index, tf):
    """sym's closes at the same times (Alpaca bars, SIP then IEX); empty when unavailable."""
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
                bars = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=frame,
                                                              start=(start - pd.Timedelta(days=3)).to_pydatetime(),
                                                              feed=DataFeed(feed))).data.get(sym, [])
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

def equity_figure(d, bench, shown=("QQQ",), show_dd=True, intraday=False):
    """The actual equity in $ (green / red fill vs the money put in), the dashed money-put-in line (start balance +
    deposits - withdrawals), QQQ / SPY / DIA bought with the same cash flows, and the drawdown of the time-weighted return."""
    eq, inv = d["equity"], d["invested"]
    up = eq.iloc[-1] >= inv.iloc[-1]
    color, fill = (GOOD, "rgba(21,128,61,0.10)") if up else (BAD, "rgba(185,28,28,0.10)")
    fig = make_subplots(rows=2 if show_dd else 1, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.75, 0.25] if show_dd else [1])
    fig.add_trace(go.Scatter(x=inv.index, y=inv.values, mode="lines", name="Start + net deposits",
                             line=dict(color="#94a3b8", width=1.2, dash="dash", shape="hv"),
                             hovertemplate="Start + net deposits $%{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Portfolio", fill="tonexty", fillcolor=fill,
                             line=dict(color=color, width=2.2), hovertemplate="Portfolio $%{y:,.2f}<extra></extra>"), row=1, col=1)
    for sym in shown:                                  # same starting money and the same deposits, in the index fund
        px = (bench or {}).get(sym)
        if px is None or not px.notna().any():
            continue
        px = px.ffill().bfill()
        qv = ((eq.iloc[0] / px.iloc[0]) + (d["flow"] / px).cumsum()) * px
        fig.add_trace(go.Scatter(x=qv.index, y=qv.values, mode="lines", name=f"{sym} (same start + deposits)",
                                 line=dict(color=BENCH[sym], width=1.4, dash="dot"),
                                 hovertemplate=f"{sym} $%{{y:,.2f}}<extra></extra>"), row=1, col=1)
    if show_dd:
        dd = (d["twr"] / d["twr"].cummax() - 1) * 100
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
    c1, *tg = st.columns([5, 0.9, 0.9, 0.9, 1.2], vertical_alignment="center")
    key = c1.radio("Period", list(PERIODS), index=list(PERIODS).index(SINCE), key="eq_period", horizontal=True,
                   label_visibility="collapsed")   # radio, not segmented_control: AppTest cannot rerun button groups
    shown = [sym for c, sym in zip(tg, BENCH) if c.toggle(sym, value=sym == "QQQ", key=f"eq_{sym.lower()}")]
    show_dd = tg[3].toggle("Drawdown", value=True, key="eq_dd")
    try:
        d, bench = equity_series(key)
    except Exception as e:
        st.info(f"Portfolio history unavailable right now ({type(e).__name__}). It tries again in a minute.")
        return
    if len(d) < 2:
        st.info("Not enough portfolio history for this period yet.")
        return
    m = gains(d)
    qtxt = "".join(f' · {sym} {(px.dropna().iloc[-1] / px.dropna().iloc[0] - 1) * 100:+.2f}%'
                   for sym, px in bench.items() if px is not None and px.notna().sum() > 1)
    show_html(f'<div class="sa-eq-head"><span class="sa-eq-val">${m["end"]:,.2f}</span>'
              f'<span class="sa-eq-chg" style="color:{GOOD if m["gain"] >= 0 else BAD}">{usd(m["gain"])} ({m["twr_pct"]:+.2f}%)</span>'
              f'<span class="sa-eq-sub">{key} · net of {usd(m["net_flows"])} deposits / withdrawals · time-weighted{qtxt}</span></div>')
    st.plotly_chart(equity_figure(d, bench, shown, show_dd, intraday=PERIODS[key][1] != "1D"), width="stretch",
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}, key="eq_chart")
