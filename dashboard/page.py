"""Page state (everything the render functions need, computed once per run) and the title bar."""
import os
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from dashboard.data import data_freshness, latest_rows, load_signals
from dashboard.settings import CT, SIGNAL_CSV
from dashboard.short_stock import load_short_history
from dashboard.signals import rebalance_plan, signal_board
from dashboard.style import esc, show_html

def build_page():
    mtime = os.path.getmtime(SIGNAL_CSV)
    df = load_signals(mtime)
    latest = latest_rows(df, mtime)
    import sector_mapping as sm                   # a stock commented out of the list leaves the page at once (Oct 19)
    latest = latest[latest['Symbol'].isin(set(sm.stock_symbols))]
    board_off, off_date = signal_board(df)
    plan_day, plan_asof, plan = rebalance_plan(df)
    ref = load_short_history()
    short = (ref if ref is not None else pd.DataFrame(columns=["Symbol"])).set_index("Symbol")
    return SimpleNamespace(
        df=df, mtime=mtime, latest=latest, by_symbol=latest.set_index("Symbol"),
        options=latest.sort_values('combined_signal', ascending=False)['Symbol'].tolist()   # dropdown: best score first,
        + [s for s in short.index if s not in set(latest['Symbol'])],                       # then short-history stocks
        short=short,
        off_date=off_date,
        tag_off=dict(zip(board_off["Symbol"], board_off["Tag"])),
        plan_day=plan_day, plan_asof=plan_asof, plan=plan,
        sig_off=dict(zip(board_off["Symbol"], board_off["Signal"])),
        why_off=dict(zip(board_off["Symbol"], board_off["Why"])),
        freshness=data_freshness(df["Date"].max()),
    )

def open_symbol(table, event, key):
    """Row click in a table -> open that stock in the stock view (applied on the rerun, before the picker is drawn)."""
    rows = event.selection.rows if event is not None and hasattr(event, "selection") else []
    if not rows:
        return
    pick = table.iloc[rows[0]]["Symbol"]
    if st.session_state.get(f"_last_pick_{key}") != pick:
        st.session_state[f"_last_pick_{key}"] = pick
        st.session_state["_pending_ticker"] = pick
        st.rerun()

def kpis():
    """Header KPI numbers from the saved forward test (the same leaderboard the Strategy tab shows); {} when missing."""
    try:
        import forward_test as ft
        from dashboard.data import read_report_csv
        from dashboard.details.forward_test import _board
        daily, values = read_report_csv(ft.DAILY_CSV), read_report_csv(ft.STRATEGIES_CSV)
        if values is None or values.empty:
            return {}
        b = _board(daily, values)[0]
        b = b.assign(name=b["Strategy"].str.replace(ft.LIVE_MARK, "", regex=False)).set_index("name")
        row = lambda n: b.loc[n] if n in b.index else None
        out = {"live": row(ft.LIVE), "account": row(ft.ACCOUNT), "qqq": row("QQQ (comparison)"),
               "spy": row("SPY (comparison)"), "ranked": int(b["Rank"].notna().sum())}
        if daily is not None and len(daily):
            out["equity"] = float(daily["Equity"].iloc[-1])
        return out
    except Exception:          # the header never breaks the page
        return {}

def _pct(r, col="Total return %"):
    return f"{r[col]:+.2f}%" if r is not None and pd.notna(r[col]) else "—"

def render_top_bar(p):
    stale = p.freshness.loc[p.freshness["Status"].str.startswith("⚠️"), "File"].tolist()
    updated = datetime.fromtimestamp(p.mtime, tz=CT).strftime("%b %-d, %I:%M %p CT")
    c1, c2, c3 = st.columns([5, 3, 1.2], vertical_alignment="center")
    with c1:
        show_html('<div class="sa-brand"><div class="sa-logo">SA</div><div><div class="sa-brand-name">Stock Analysis</div>'
                  '<div class="sa-brand-sub">Signals · forward test · live account</div></div></div>')
    with c2:
        note = f" · ⚠️ {len(stale)} stale" if stale else ""
        show_html(f'<div style="text-align:right"><span class="sa-chip{" sa-chip-warn" if stale else ""}" '
                  f'title="{esc(", ".join(stale) or "All data files are fresh")}">Updated {esc(updated + note)}</span></div>')
    with c3:
        # every data cache is keyed on the Reports/*.csv modification times: clearing them reloads the newest output
        if st.button("Refresh data", width="stretch",
                     help="Clear all cached data and reload the latest Reports/*.csv files and live quotes."):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    k = kpis()
    if not k:
        return
    live, acc, qqq, spy = k.get("live"), k.get("account"), k.get("qqq"), k.get("spy")
    m = st.columns(5)
    m[0].metric("Account equity", f"${k['equity']:,.0f}" if "equity" in k else "—",
                _pct(acc, "Daily return %") if acc is not None else None, help="Alpaca account, last saved close")
    m[1].metric("Account since Oct 2", _pct(acc), help="Time-weighted: deposits and withdrawals are not returns")
    m[2].metric("Live strategy since Oct 2", _pct(live),
                f"Rank #{int(live['Rank'])} of {k['ranked']}" if live is not None and pd.notna(live["Rank"]) else None,
                delta_color="off", help="Forward-test leaderboard (Strategy tab)")
    m[3].metric("QQQ since Oct 2", _pct(qqq), _pct(qqq, "Daily return %") if qqq is not None else None)
    m[4].metric("SPY since Oct 2", _pct(spy), _pct(spy, "Daily return %") if spy is not None else None)
