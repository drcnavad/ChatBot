"""Trading Account tab: the account's equity over a chosen period (Alpaca portfolio history, read-only GET), with the
period's gain, QQQ / SPY / DIA bought with the same cash flows and the drawdown. The line is the real equity in $ (from Oct 9); the
$ gain and the % return leave deposits / withdrawals out, so money moved in is never a gain (nor out a loss)."""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.style import BAD, CHART_FONT, GOOD, show_html, usd

START = pd.Timestamp("2026-10-09")          # the live account's history starts here; nothing earlier is shown
BENCH = {"QQQ": "#64748b", "SPY": "#d97706", "DIA": "#7c3aed"}   # comparisons: Nasdaq-100, S&P 500, Dow
SINCE = f"Since {START:%b %-d}"
PERIODS = {"1D": ("1D", "5Min"), "5D": ("5D", "15Min"), SINCE: ("1A", "1D")}
# the chart's range buttons (the stock chart's component, read-only): 1D / 5D load intraday data, the rest zoom the
# daily line (which starts Oct 9, so a longer button never shows empty space before it)
VIEWS = ["1D", "5D", "1M", "3M", "1Y", "2Y"]
DATA = {"1D": "1D", "5D": "5D"}             # view -> PERIODS key; any other view: SINCE

@st.cache_data(ttl=600, max_entries=12, show_spinner=False)
def equity_series(key, slot=0):
    """(DataFrame equity (actual $), flow (deposits - withdrawals on that bar), twr (time-weighted growth),
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
        now = pd.Timestamp.now(tz="America/New_York")
        today = now.tz_localize(None).normalize()
        trading = now.weekday() < 5 and now.hour * 60 + now.minute >= 570     # not on weekends or before the open
        if trading and float(acct.get("equity") or 0) > 0 and (not len(raw) or raw.index[-1] < today):
            raw.loc[today], flow.loc[today] = float(acct["equity"]), 0.0
    elif len(raw):                                      # intraday: today's live equity (GET /account) as the latest point
        now = pd.Timestamp.now(tz="America/Chicago")
        if raw.index[-1].date() == now.date() and now > raw.index[-1]:
            eq_now = float(ap.PaperAccount()._get("/account").get("equity") or 0)
            if eq_now > 0:
                raw.loc[now], flow.loc[now] = eq_now, 0.0
        if key == "1D" and len(raw):                    # start the day at yesterday's close, not at today's open
            prev_close = float(ap.PaperAccount()._get("/account").get("last_equity") or 0)
            if prev_close > 0:
                t0 = raw.index[0] - pd.Timedelta(minutes=5)
                raw = pd.concat([pd.Series([prev_close], index=[t0]), raw])
                flow = pd.concat([pd.Series([0.0], index=[t0]), flow])
    keep_from = raw.index.tz_localize(None) >= START if raw.index.tz is not None else raw.index >= START
    raw, flow = raw[keep_from], flow[keep_from]
    prev = raw.shift(1)
    ratio = ((raw - flow) / prev).where((prev > 0) & (raw - flow > 0), 1.0)   # time-weighted: deposits aren't gains
    keep = (raw > 0) & raw.notna()                                            # before funding / after closing: left out
    raw, ratio, flow = raw[keep], ratio[keep], flow[keep]
    if len(raw) < 2:
        return pd.DataFrame(columns=["equity", "flow", "twr"]), {}
    ratio.iloc[0] = 1.0
    flow.iloc[0] = 0.0                                 # money already in at the start is the starting balance
    out = pd.DataFrame({"equity": raw, "flow": flow, "twr": ratio.cumprod()})
    return out, {sym: _bench(sym, out.index, tf) for sym in BENCH}

def gains(d):
    """The period's numbers, cash flows taken out: $ gain = end - start - net deposits; % = time-weighted return."""
    start, end, net = float(d["equity"].iloc[0]), float(d["equity"].iloc[-1]), float(d["flow"].sum())
    return {"start": start, "end": end, "net_flows": net, "gain": end - start - net,
            "twr_pct": (float(d["twr"].iloc[-1]) - 1) * 100}

def _bench(sym, index, tf):
    """sym's closes at the same times (data.alpaca_bars); empty when unavailable."""
    if not len(index):
        return pd.Series(dtype=float)
    try:
        from dashboard.data import NY, alpaca_bars
        start = pd.Timestamp(index[0])
        start = (start.tz_localize(NY) if start.tz is None else start) - pd.Timedelta(days=3)
        bars = alpaca_bars((sym,), None if tf == "1D" else int(tf[:-3]), start)
        if sym not in bars:
            return pd.Series(dtype=float)
        s = bars[sym]["Close"]
        if tf == "1D":                                  # daily bars: their New York date, as the equity index
            s.index = s.index.normalize()
            return s.reindex(index, method="ffill")
        s.index = s.index.tz_localize(NY).tz_convert(index.tz)
        return s.reindex(index, method="ffill", tolerance=pd.Timedelta(minutes=30))
    except Exception:
        return pd.Series(dtype=float)

def equity_figure(d, bench, shown=("QQQ",), show_dd=False, intraday=False):
    """The actual equity in $ (green / red fill vs a dashed line at the period's start value), QQQ / SPY / DIA from the
    same start (plus any later deposits, on the same days), and the drawdown of the time-weighted return."""
    d = d.copy()
    if intraday:                                       # the component's clock: New York time, no zone (as the stock chart)
        d.index = d.index.tz_convert("America/New_York").tz_localize(None)
        bench = {k: (v.set_axis(d.index) if v is not None and len(v) == len(d) else v) for k, v in (bench or {}).items()}
    eq = d["equity"]
    base = float(eq.iloc[0])
    up = gains(d)["gain"] >= 0
    color, fill = (GOOD, "rgba(21,128,61,0.10)") if up else (BAD, "rgba(185,28,28,0.10)")
    fig = make_subplots(rows=2 if show_dd else 1, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.75, 0.25] if show_dd else [1])
    fig.add_trace(go.Scatter(x=eq.index, y=[base] * len(eq), mode="lines", line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Portfolio", fill="tonexty", fillcolor=fill,
                             line=dict(color=color, width=2.2), hoverinfo="none"), row=1, col=1)
    fig.add_hline(y=base, line=dict(color="#94a3b8", width=1, dash="dash"), row=1, col=1)   # the period's start value
    for sym in shown:                                  # same starting money and the same deposits, in the index fund
        px = (bench or {}).get(sym)
        if px is None or not px.notna().any():
            continue
        px = px.ffill().bfill()
        qv = ((eq.iloc[0] / px.iloc[0]) + (d["flow"] / px).cumsum()) * px
        fig.add_trace(go.Scatter(x=qv.index, y=qv.values, mode="lines", name=f"{sym} (same start)",
                                 line=dict(color=BENCH[sym], width=1.4, dash="dot"), hoverinfo="none"), row=1, col=1)
    if show_dd:
        dd = (d["twr"] / d["twr"].cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown", fill="tozeroy",
                                 line=dict(color=BAD, width=1), fillcolor="rgba(185,28,28,0.12)", showlegend=False,
                                 hoverinfo="none"), row=2, col=1)
        fig.update_yaxes(title_text="Drawdown", ticksuffix="%", tickformat=".1f", row=2, col=1)
    fig.update_layout(height=470 if show_dd else 380, margin=dict(l=10, r=70, t=44, b=24), hovermode="closest",
                      dragmode=False, plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(family=CHART_FONT, size=11, color="#334155"),
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.0, yanchor="bottom", font=dict(size=10),
                                  bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showgrid=False, fixedrange=True, tickformat="%-I:%M %p" if intraday else "%b %-d",
                     rangebreaks=[dict(bounds=["sat", "mon"])] + ([dict(bounds=[16, 9.4], pattern="hour")] if intraday else []))
    fig.update_yaxes(side="right", gridcolor="#f1f5f9", zeroline=False, fixedrange=True, tickfont=dict(size=10, color="#64748b"))
    fig.update_yaxes(tickformat="$,.0f", row=1, col=1)
    return fig

@st.fragment(run_every=30)      # new Alpaca portfolio history every refresh slot (30 s open, 5 min closed)
def render_equity_chart():
    if os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off":
        st.info("Portfolio chart is turned off here (STOCK_ANALYSIS_LIVE_HOLDINGS=off).")
        return
    view = st.session_state.get("eq_view", "1D")      # a fresh load shows today; a picked range is kept for the session
    key = DATA.get(view, SINCE)
    tg = st.columns(3)
    shown = [sym for c, sym in zip(tg, BENCH) if c.toggle(sym, value=sym == "QQQ", key=f"eq_{sym.lower()}")]
    show_dd = False                                   # drawdown panel removed from this chart
    try:
        from dashboard.data import refresh_slot
        d, bench = equity_series(key, refresh_slot())
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
              f'<span class="sa-eq-sub">{view if view in DATA else SINCE} · net of {usd(m["net_flows"])} deposits / withdrawals · time-weighted{qtxt}</span></div>')
    intraday = PERIODS[key][1] != "1D"
    fig = equity_figure(d, bench, shown, show_dd, intraday=intraday)
    x = d.index.tz_convert("America/New_York").tz_localize(None) if intraday else d.index
    fmt = "%Y-%m-%dT%H:%M" if intraday else "%Y-%m-%d"           # as the stock chart
    eq = [float(v) for v in d["equity"]]
    from dashboard.drawings import _chart                 # the stock chart's component, as a read-only line chart
    out = _chart(spec=fig.to_json(), ohlc={"x": [t.strftime(fmt) for t in x], "open": eq, "high": eq, "low": eq, "close": eq},
                 view=view, mode="Line", intraday=intraday, plain=True, ranges=VIEWS, key="eq_chart", default=None)
    if out and out.get("view") in VIEWS and out.get("t") != st.session_state.get("_eq_t"):
        st.session_state["_eq_t"] = out["t"]
        if out["view"] != view:
            st.session_state["eq_view"] = out["view"]
            if DATA.get(out["view"], SINCE) != key:            # other data (intraday <-> daily): redraw
                st.rerun()                      # app scope: scope="fragment" fails when this runs in a full-page run
