"""Page state (everything the render functions need, computed once per run) and the title bar."""
import os
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from dashboard.data import data_freshness, day_rank_change, latest_rows, load_midweek_rows, load_signals
from dashboard.settings import CT, SIGNAL_CSV
from dashboard.short_stock import load_short_history
from dashboard.signals import next_decision_date, rebalance_plan, signal_board
from dashboard.style import esc, show_html


def build_page():
    mtime = os.path.getmtime(SIGNAL_CSV)
    df = load_signals(mtime)
    latest = latest_rows(df, mtime)
    board_off, off_date = signal_board(df)
    next_dec, next_kind = next_decision_date(df["Date"].max())
    plan_day, plan_asof, plan = rebalance_plan(df)
    ref = load_short_history()
    short = (ref if ref is not None else pd.DataFrame(columns=["Symbol"])).set_index("Symbol")
    return SimpleNamespace(
        df=df, mtime=mtime, latest=latest, by_symbol=latest.set_index("Symbol"),
        options=latest.sort_values('combined_signal', ascending=False)['Symbol'].tolist()   # dropdown: best score first,
        + [s for s in short.index if s not in set(latest['Symbol'])],                       # then short-history stocks
        short=short,
        rank_change=day_rank_change(df, mtime),
        board_off=board_off.drop(columns="Tag"), off_date=off_date,
        tag_off=dict(zip(board_off["Symbol"], board_off["Tag"])),
        rank_off=dict(zip(board_off["Symbol"], board_off["Rank"])),
        plan_day=plan_day, plan_asof=plan_asof, plan=plan,
        sig_off=dict(zip(board_off["Symbol"], board_off["Signal"])),
        why_off=dict(zip(board_off["Symbol"], board_off["Why"])),
        slot_off=dict(zip(board_off["Symbol"], board_off["Portfolio slot"])),
        next_dec=next_dec, next_kind=next_kind, midweek=load_midweek_rows(),
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


def render_top_bar(p):
    stale = p.freshness.loc[p.freshness["Status"].str.startswith("⚠️"), "File"].tolist()
    updated = datetime.fromtimestamp(p.mtime, tz=CT).strftime("%m/%d/%Y %I:%M %p CT")
    note = f" · ⚠️ {len(stale)} stale file(s), see Details" if stale else ""
    c1, c2, c3 = st.columns([4, 3, 1], vertical_alignment="center")
    with c1:
        show_html("<h1 style='margin:0;font-size:1.45rem;font-weight:700;'>Stock Analysis</h1>")
    with c2:
        show_html(f'<div class="sa-chip{" sa-chip-warn" if stale else ""}">Updated {esc(updated + note)}</div>')
    with c3:
        # One-click freshness: every data cache is already keyed on the Reports/*.csv modification times,
        # so clearing the caches and rerunning always shows the newest pipeline output + live quotes.
        if st.button("Refresh data",
                       help="Clear all cached data and reload the latest Reports/*.csv files and live quotes."):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
