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

def render_ticker(box):
    """The scrolling strip of held stocks (this run's Alpaca positions): live price and day change from
    data.index_quotes (Alpaca SIP / IEX, 30 s cache), else the position's own current price and change_today."""
    from dashboard.data import index_quotes
    from dashboard.details.live_holdings import holdings_this_run
    data, err = holdings_this_run()
    pos = sorted((data or {}).get("positions") or [], key=lambda x: x["symbol"]) if not err else []
    if not pos:
        return
    q = index_quotes(tuple(x["symbol"] for x in pos))
    items = []
    for x in pos:
        px, chg = q.get(x["symbol"], (None, None))
        if px is None:
            px = float(x.get("current_price") or "nan")
            chg = float(x.get("change_today") or "nan") * 100
        cls = "up" if chg == chg and chg > 0 else "down" if chg == chg and chg < 0 else ""
        items.append(f'<a class="sa-tick" href="?stock={esc(x["symbol"])}" target="_self"><b>{esc(x["symbol"])}</b>'
                     f'<span>${px:,.2f}</span><span class="{cls}">{"▲" if cls == "up" else "▼" if cls == "down" else ""}'
                     f'{abs(chg):.2f}%</span></a>' if px == px else "")
    row = "".join(items)
    secs = max(20, 3 * len(items))                       # the same speed for any number of stocks
    box.markdown(f'<div class="sa-ticker"><div class="sa-ticker-track" style="animation-duration:{secs}s">'
                 f'{row}{row}</div></div>', unsafe_allow_html=True)

def render_top_bar(p):
    ticker = st.container()          # the held-stock strip, filled at the end of the run (render_ticker)
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
    return ticker, st.container()          # the KPI cards, filled by render_kpis at the end of the run

def render_kpis(box):
    """The header cards, drawn last (into render_top_bar's container) so they reuse this run's holdings read."""
    k = kpis()
    from dashboard.data import index_quotes
    acct, q, live = account_now(), index_quotes(("QQQ", "SPY", "DIA")), k.get("live")
    if acct is None and k.get("daily") is not None:                   # Alpaca not readable: the last saved close
        d = k["daily"]
        acct = {"equity": float(d["Equity"]), "cash": float(d["Cash"]), "chg": None, "chg_pct": None,
                "positions": int(d["Positions"]), "saved": True}
    from dashboard.data import intraday
    spark = intraday(("QQQ", "SPY", "DIA"))
    cards = []
    if acct:
        cards += [("Portfolio", f"${acct['equity']:,.0f}",
                   "Last saved close" if acct.get("saved") or acct["chg"] is None
                   else f"{'-' if acct['chg'] < 0 else '+'}${abs(acct['chg']):,.0f} ({_sp(acct['chg_pct'])}) today",
                   None if acct.get("saved") else acct["chg"], "Alpaca account equity and today's change", spark.get("PORTFOLIO"), ",.0f"),
                  ("Cash", f"{acct['cash'] / acct['equity']:.1%}" if acct["equity"] else "—",
                   f"${acct['cash']:,.0f} · {acct['positions']} positions", None, "Cash share of equity; open positions", None, "")]
    for sym, name in (("QQQ", "Nasdaq 100 · QQQ"), ("SPY", "S&P 500 · SPY"), ("DIA", "Dow · DIA")):
        px, chg = q.get(sym, (None, None))
        cards.append((name, f"${px:,.2f}" if px else "—", f"{_sp(chg)} today" if px else "Quote unavailable", chg,
                      "Live price (Alpaca SIP, IEX fallback), change vs the previous close", spark.get(sym), ",.2f"))
    if live is not None:
        cards.append(("Live strategy", _sp(live["Total return %"]),
                      f"Rank #{int(live['Rank'])} of {k['ranked']}" if pd.notna(live["Rank"]) else "Not ranked", None,
                      "Forward-test total return and rank (Strategy tab)", None, ""))
    with box:
        show_html('<div class="sa-kpis">' + "".join(kpi_card(*c) for c in cards) + "</div>")

def sparkline(times, values, base, fmt_, w=260, h=90):
    """Inline SVG of today's line: green above / red below the previous close (dashed), with the latest value."""
    if len(values) < 2:
        return '<div class="sa-pop-empty">No intraday data yet today</div>'
    lo, hi = min(min(values), base), max(max(values), base)
    span = (hi - lo) or 1
    t0, t1 = times[0], times[-1] if times[-1] > times[0] else times[0] + 1
    X = lambda t: 4 + (t - t0) / (t1 - t0) * (w - 8)
    Y = lambda v: 6 + (hi - v) / span * (h - 12)
    color = "#15803d" if values[-1] >= base else "#b91c1c"
    pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(times, values))
    t_last = datetime.fromtimestamp(times[-1], tz=CT).strftime("%I:%M %p CT").lstrip("0")
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><line x1="0" x2="{w}" y1="{Y(base):.1f}" y2="{Y(base):.1f}" '
            f'stroke="#94a3b8" stroke-dasharray="4 3" stroke-width="1"/><polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linejoin="round"/><circle cx="{X(times[-1]):.1f}" cy="{Y(values[-1]):.1f}" r="3" fill="{color}"/></svg>'
            f'<div class="sa-pop-foot"><span>Prev close ${format(base, fmt_)}</span>'
            f'<b style="color:{color}">${format(values[-1], fmt_)} · {t_last}</b></div>')

def kpi_card(label, value, delta, sign, tip, series=None, fmt_=""):
    """One header card (all the same size); with `series` a pure-CSS hover popup holds today's intraday line."""
    cls = "up" if sign is not None and sign == sign and sign > 0 else "down" if sign is not None and sign == sign and sign < 0 else ""
    pop = (f'<div class="sa-pop"><div class="sa-pop-title">{esc(label)} · today (5-min)</div>{sparkline(*series, fmt_)}</div>'
           if series else "")
    return (f'<div class="sa-kpi{" has-pop" if series else ""}" title="{esc(tip) if not series else ""}">'
            f'<div class="sa-kpi-label">{esc(label)}</div><div class="sa-kpi-value">{esc(value)}</div>'
            f'<div class="sa-kpi-delta {cls}">{esc(delta)}</div>{pop}</div>')
