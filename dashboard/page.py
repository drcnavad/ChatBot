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
    """The live strategy's leaderboard row and the ranked count (the same board as the Strategy tab) + the last saved
    account row (fallback when Alpaca cannot be read); {} when missing."""
    try:
        import forward_test as ft
        from dashboard.data import read_report_csv
        from dashboard.details.forward_test import _board
        daily, values = read_report_csv(ft.DAILY_CSV), read_report_csv(ft.STRATEGIES_CSV)
        out = {"daily": daily.iloc[-1] if daily is not None and len(daily) else None}
        if values is not None and not values.empty:
            b = _board(daily, values)[0]
            live = b[b["Strategy"].str.replace(ft.LIVE_MARK, "", regex=False) == ft.LIVE]
            out.update(live=live.iloc[0] if len(live) else None, ranked=int(b["Rank"].notna().sum()))
        return out
    except Exception:          # the header never breaks the page
        return {}

def account_now():
    """Header cards: equity, cash, today's change and open positions from this run's live holdings read
    (holdings_this_run: no extra Alpaca call). Today's change = equity vs the last finished day in the account
    history, net of today's deposits. None when the account cannot be read."""
    from dashboard.details.live_holdings import holdings_this_run
    data, err = holdings_this_run()
    if err or not data:
        return None
    eq, pos, h = float(data["equity"]), data["positions"], data.get("history")
    invested = sum(float(x.get("market_value") or 0) for x in pos)
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    prev = h[h["Date"] < today] if h is not None and len(h) else None
    flow = float(h.loc[h["Date"] == today, "Cashflow"].sum()) if h is not None and len(h) else 0.0
    chg = eq - flow - float(prev["Equity"].iloc[-1]) if prev is not None and len(prev) else None
    return {"equity": eq, "cash": eq - invested, "positions": len(pos), "chg": chg,
            "chg_pct": chg / float(prev["Equity"].iloc[-1]) * 100 if chg is not None else None}

def _sp(v, d=2, unit="%"):
    return f"{v:+,.{d}f}{unit}" if v is not None and v == v else "—"

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
    return st.container()          # the KPI cards, filled by render_kpis at the end of the run

def render_kpis(box):
    """The header cards, drawn last (into render_top_bar's container) so they reuse this run's holdings read."""
    k = kpis()
    from dashboard.data import index_quotes
    acct, q, live = account_now(), index_quotes(("QQQ", "SPY", "DIA")), k.get("live")
    if acct is None and k.get("daily") is not None:                   # Alpaca not readable: the last saved close
        d = k["daily"]
        acct = {"equity": float(d["Equity"]), "cash": float(d["Cash"]), "chg": None, "chg_pct": None,
                "positions": int(d["Positions"]), "saved": True}
    cards = []
    if acct:
        cards += [("Portfolio", f"${acct['equity']:,.0f}",
                   "Last saved close" if acct.get("saved") or acct["chg"] is None else f"{'-' if acct['chg'] < 0 else '+'}${abs(acct['chg']):,.0f} ({_sp(acct['chg_pct'])}) today",
                   "off" if acct.get("saved") else "normal", "Alpaca account equity and today's change"),
                  ("Cash", f"{acct['cash'] / acct['equity']:.1%}" if acct["equity"] else "—",
                   f"${acct['cash']:,.0f} · {acct['positions']} positions", "off", "Cash share of equity; open positions")]
    for sym, name in (("QQQ", "Nasdaq 100 · QQQ"), ("SPY", "S&P 500 · SPY"), ("DIA", "Dow · DIA")):
        px, chg = q.get(sym, (None, None))
        cards.append((name, f"${px:,.2f}" if px else "—", f"{_sp(chg)} today" if px else "Quote unavailable",
                      "normal" if px else "off", "Live price (Alpaca SIP, IEX fallback), change vs the previous close"))
    if live is not None:
        cards.append(("Live strategy", _sp(live["Total return %"]),
                      f"Rank #{int(live['Rank'])} of {k['ranked']}" if pd.notna(live["Rank"]) else "Not ranked", "off",
                      "Forward-test total return and rank (Strategy tab)"))
    for col, (label, value, delta, dc, tip) in zip(box.columns(len(cards), gap="small"), cards):
        col.metric(label, value, delta, delta_color=dc, help=tip)
