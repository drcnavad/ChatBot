"""Strategy tab: the real live account vs QQQ and SPY, straight from Alpaca (read-only) and yfinance closes.

Account = time-weighted from Alpaca's daily history (GET /account/portfolio/history) plus the equity now: each day's growth
= (equity - that day's deposits) / the previous day's equity, so deposits are never returns. QQQ / SPY = price change
(dashboard.data.etf_prices: daily closes + the latest quote). Display only: nothing switches on its own."""
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.settings import CT
from dashboard.style import PCT_COL, esc, show_html, toned

ACCOUNT = "Your account (time-weighted)"

def account_index(history, equity_now=None, flows_today=0.0, today=None):
    """Series by date: 1.0 on the first history row. The live equity (when given) replaces / extends `today`'s row."""
    h = history.assign(Date=pd.to_datetime(history["Date"]).dt.normalize()).sort_values("Date")
    if equity_now is not None and today is not None:
        h = pd.concat([h[h["Date"] < today], pd.DataFrame({"Date": [today], "Equity": [equity_now],
                                                            "Cashflow": [flows_today]})], ignore_index=True)
    eq, cf = h["Equity"].astype(float).to_numpy(), h["Cashflow"].astype(float).to_numpy()
    growth = np.ones(len(h))
    growth[1:] = (eq[1:] - cf[1:]) / eq[:-1]
    return pd.Series(np.cumprod(growth), index=h["Date"].to_numpy())

def price_index(closes, now=None, today=None):
    s = closes.dropna().copy()
    s.index = pd.to_datetime(s.index).normalize()
    if now is not None and today is not None:
        s = pd.concat([s[s.index < today], pd.Series([float(now)], index=[today])])
    return s

def returns_table(series, switch, start):
    """One row per (name, Series): Today % (last vs previous point), Since <switch> close %, Since <start> close %.
    'Since the switch' is NaN until a session after the switch has a value (no fake zeros)."""
    switch, start = pd.Timestamp(switch), pd.Timestamp(start)
    col_sw, col_st = f"Since {switch:%b %-d} switch %", f"Since {start:%b %-d} %"
    rows = []
    for name, s in series:
        s = s.dropna()
        last = s.iloc[-1] if len(s) else np.nan
        base_sw = s[s.index <= switch]
        base_st = s[s.index >= start]
        rows.append({"Line": name,
                     "Today %": 100 * (last / s.iloc[-2] - 1) if len(s) > 1 else np.nan,
                     col_sw: 100 * (last / base_sw.iloc[-1] - 1) if len(base_sw) and s.index[-1] > switch else np.nan,
                     col_st: 100 * (last / base_st.iloc[0] - 1) if len(base_st) else np.nan,
                     "As of": f"{s.index[-1]:%a %b %-d}" if len(s) else ""})
    return pd.DataFrame(rows, columns=["Line", "Today %", col_sw, col_st, "As of"]), col_sw

def render_live_vs_benchmark():
    import alpaca_paper as ap
    import backtest_engine as be
    import forward_test as ft
    from dashboard.data import etf_prices
    from dashboard.details.live_holdings import holdings_this_run
    switch, start = pd.Timestamp(ft.LIVE_SWITCH), pd.Timestamp(be.FORWARD_START)
    data, err = holdings_this_run()
    if data is None:
        st.info(f"Live vs benchmark: {err}")
        return
    now = datetime.now(CT)
    today = pd.Timestamp(now.date()) if be.is_session(now.date()) else None
    snap = data["snapshot"]
    flows_today = sum(float(f["amount"]) for f in snap.get("flows", []) if today is not None
                      and ap._flow_time(f).date() == today.date())
    series = [(ACCOUNT, account_index(data["history"], data["equity"], flows_today, today))]
    closes, px = etf_prices(("QQQ", "SPY"), str(start.date()))
    series += [(s, price_index(closes[s], px.get(s), today)) for s in ("QQQ", "SPY") if s in closes]
    t, col_sw = returns_table(series, switch, start)
    acc = t.set_index("Line")
    started = pd.notna(acc.at[ACCOUNT, col_sw])
    if started and "SPY" in acc.index and pd.notna(acc.at["SPY", col_sw]):
        gap = acc.at[ACCOUNT, col_sw] - acc.at["SPY", col_sw]
        head, color = (f"{'Winning' if gap > 0 else 'Losing'} vs SPY by {abs(gap):.2f} pts since the switch",
                       "#15803d" if gap > 0 else "#b91c1c")
    else:
        head, color = (f"The live 50/50 started at the {switch:%a %b %-d} 2:30 PM CT run; its first full day is the next "
                       "session, so 'since the switch' fills in from that close", "#b45309")
    show_html(f'<div class="sa-card"><div class="sa-card-title">Live vs benchmark<span>real account (Alpaca) vs QQQ and SPY'
              f' · decision date {pd.Timestamp(ft.DECISION_DATE):%a %b %-d, %Y}</span></div>'
              f'<div class="sa-tier"><b>Status</b><div class="sa-tier-syms"><span style="color:{color};font-weight:600">'
              f'{esc(head)}</span><span>{esc(ft.SPY_RULE)}</span></div></div></div>')
    pct = [c for c in t.columns if c.endswith("%")]
    st.dataframe(toned(t, pct), hide_index=True, width="stretch", column_config={c: PCT_COL for c in pct})
    st.caption(f"Today = the equity now (Alpaca, read {data['as_of']:%I:%M %p} CT) vs the previous close; QQQ / SPY = latest "
               f"quote vs the previous close. Since {start:%b %-d} = from that close (the forward test's start). The account "
               "is time-weighted: deposits are not returns. Blank = not available yet.")
