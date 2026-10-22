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

def _sp(v, d=2, unit="%"):
    return f"{v:+,.{d}f}{unit}" if v is not None and v == v else "—"

def render_ticker(box):
    """The scrolling strip of held stocks (this run's Alpaca positions): live price and day change from
    data.index_quotes (Alpaca SIP / IEX, 30 s cache), else the position's own current price and change_today."""
    from dashboard.data import index_quotes
    from dashboard.details.live_holdings import holdings_this_run
    data, err = holdings_this_run()
    pos = sorted((data or {}).get("positions") or [], key=lambda x: x["symbol"]) if not err else []
    if any(x["symbol"] == "GOOGL" for x in pos):                  # both share classes held: show GOOGL once
        pos = [x for x in pos if x["symbol"] != "GOOG"]
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
        items.append(f'<a class="sa-tick" href="?symbol={esc(x["symbol"])}" target="_self"><b>{esc(x["symbol"])}</b>'
                     f'<span>${px:,.2f}</span><span class="{cls}">{"▲" if cls == "up" else "▼" if cls == "down" else ""}'
                     f'{f"{abs(chg):.2f}%" if chg == chg else ""}</span></a>' if px == px else "")
    row = "".join(items)
    secs = max(20, 3 * len(items))                       # the same speed for any number of stocks
    box.markdown(f'<div class="sa-ticker"><div class="sa-ticker-track" style="animation-duration:{secs}s">'
                 f'{row}{row}</div></div>', unsafe_allow_html=True)

PAGES = ["Home", "Stock Analysis", "Strategy", "Trading Account"]

def render_top_bar(p):
    ticker = st.container()          # the held-stock strip, filled at the end of the run (render_ticker)
    stale = p.freshness.loc[p.freshness["Status"].str.startswith("⚠️"), "File"].tolist()
    updated = datetime.fromtimestamp(p.mtime, tz=CT).strftime("%b %-d, %I:%M %p CT")
    # the tab names sit in the middle of this row: st.tabs comes right after it and CSS lifts its tab bar up here
    c1, cnav, c2, c3 = st.columns([2.2, 4.6, 1.9, 1.1], vertical_alignment="center")
    with cnav:                       # keyed page selector: kept in session_state across every rerun (st.tabs is not)
        st.session_state.setdefault("nav", PAGES[0])
        st.radio("Page", PAGES, key="nav", horizontal=True, label_visibility="collapsed")
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
    return ticker

INDEXES = {"QQQ": "Nasdaq 100", "SPY": "S&P 500", "DIA": "Dow", "IWM": "Russell 2000", "CL=F": "US Crude", "BTC-USD": "Bitcoin"}
SECTORS = {"XLK": "Technology", "XLF": "Financials", "XLV": "Health Care", "XLY": "Cons. Discretionary", "XLP": "Cons. Staples",
           "XLE": "Energy", "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate",
           "XLC": "Communication", "DBC": "Commodities"}
# Top 5 holdings by weight, shown small at the card bottom: yfinance Ticker(etf).funds_data.top_holdings, checked
# 2026-10-21; GOOG (class C) is shown once, as GOOGL, and the next holding fills in. DBC holds futures + T-bills:
# its largest commodity exposures instead. Static: refresh by hand.
HOLDINGS = {"XLK": "NVDA AAPL MSFT AMD AVGO", "XLF": "BRK-B JPM V MA BAC", "XLV": "LLY JNJ ABBV MRK UNH",
            "XLY": "AMZN TSLA HD MCD TJX", "XLP": "WMT COST PG KO PM", "XLE": "XOM CVX COP VLO MPC",
            "XLI": "CAT GE GEV RTX DE", "XLB": "LIN NEM FCX ECL SHW", "XLU": "NEE SO DUK CEG AEP",
            "XLRE": "WELL PLD EQIX AMT SPG", "XLC": "META GOOGL WBD DIS TTWO", "DBC": "Brent WTI Diesel Gasoline Gold"}
YF = ("CL=F", "BTC-USD")                     # not on Alpaca stock data: yfinance (free, keyless)

@st.cache_data(ttl=60, show_spinner=False)
def market_open():
    """True during the regular session: Alpaca's clock (read-only GET /clock; knows holidays), else Mon-Fri
    9:30-16:00 New York."""
    try:
        import alpaca_paper as ap
        return bool(ap.PaperAccount()._get("/clock").get("is_open"))
    except Exception:
        now = pd.Timestamp.now(tz="America/New_York")
        return now.weekday() < 5 and 570 <= now.hour * 60 + now.minute < 960

@st.cache_data(ttl=1800, show_spinner=False)
def _card_data(syms, slot):
    """Quotes and intraday lines of the cards, fetched once per `slot` (a 30 s step in market hours, 30 min outside)."""
    from dashboard.data import index_quotes, intraday, yf_intraday
    alp = tuple(s for s in syms if s not in YF)
    q, spark = dict(index_quotes(alp)), dict(intraday(alp))
    for s, (quote, series) in yf_intraday(YF).items():
        q[s], spark[s] = quote, series
    return q, spark

@st.fragment(run_every=30)
def render_kpis():
    """The header cards: indexes, crude and bitcoin, then the 11 SPDR sector ETFs and commodities (DBC); live price,
    day %, an inline line and a hover intraday popup. (Portfolio figures live on the Trading Account tab.)
    A fragment: only the cards rerun, every 30 s; new quotes every 30 s in market hours, every 30 min outside."""
    step = 30 if market_open() else 300                 # reruns every 30 s; new data every 30 s open, 5 min closed
    q, spark = _card_data((*INDEXES, *SECTORS), int(datetime.now().timestamp() // step))
    def card(sym, name):
        px, chg = q.get(sym, (None, None))
        return kpi_card(f"{name}", f"${px:,.2f}" if px else "—", f"{_sp(chg)} today" if px else "Quote unavailable",
                        chg, spark.get(sym), ",.2f", sym.replace("=F", "").replace("-USD", ""), HOLDINGS.get(sym, ""))
    show_html("".join('<div class="sa-kpis">' + "".join(card(*x) for x in group.items()) + "</div>" for group in (INDEXES, SECTORS)))

def _line(times, values, base, w, h, dot=True):
    """SVG body of an intraday line: dashed previous close, green above / red below."""
    lo, hi = min(min(values), base), max(max(values), base)
    span = (hi - lo) or 1
    t0, t1 = times[0], times[-1] if times[-1] > times[0] else times[0] + 1
    X = lambda t: 4 + (t - t0) / (t1 - t0) * (w - 8)
    Y = lambda v: 3 + (hi - v) / span * (h - 6)
    color = "#15803d" if values[-1] >= base else "#b91c1c"
    pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(times, values))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><line x1="0" x2="{w}" y1="{Y(base):.1f}" y2="{Y(base):.1f}" '
            f'stroke="#94a3b8" stroke-dasharray="4 3" stroke-width="1"/><polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1.6" stroke-linejoin="round"/>'
            + (f'<circle cx="{X(times[-1]):.1f}" cy="{Y(values[-1]):.1f}" r="3" fill="{color}"/>' if dot else "") + "</svg>"), color

def sparkline(times, values, base, fmt_, w=260, h=90):
    """Inline SVG of today's line: green above / red below the previous close (dashed), with the latest value."""
    if len(values) < 2:
        return '<div class="sa-pop-empty">No intraday data yet today</div>'
    svg, color = _line(times, values, base, w, h)
    t_last = datetime.fromtimestamp(times[-1], tz=CT).strftime("%I:%M %p CT").lstrip("0")
    return (svg +
            f'<div class="sa-pop-foot"><span>Prev close ${format(base, fmt_)}</span>'
            f'<b style="color:{color}">${format(values[-1], fmt_)} · {t_last}</b></div>')

def kpi_card(label, value, delta, sign, series=None, fmt_="", badge="", foot=""):
    """One header card (all the same size): symbol badge, name, price, day change and an inline line, accent colored by
    the day's direction; hovering opens a popup with the full intraday line."""
    cls = "up" if sign is not None and sign == sign and sign > 0 else "down" if sign is not None and sign == sign and sign < 0 else ""
    ok = series and len(series[1]) > 1
    pop = (f'<div class="sa-pop"><div class="sa-pop-title">{esc(label)} · latest session (5-min)</div>'
           + (sparkline(*series, fmt_) if series else '<div class="sa-pop-empty">No intraday data right now</div>') + "</div>")
    mini = f'<div class="sa-kpi-spark">{_line(*series, 150, 16 if foot else 22, dot=False)[0].replace("<svg ", "<svg preserveAspectRatio=\"none\" ", 1)}</div>' if ok else ""
    return (f'<div class="sa-kpi {cls} has-pop">'
            + (f'<span class="sa-kpi-badge">{esc(badge)}</span>' if badge else "")
            + f'<div class="sa-kpi-label">{esc(label)}</div><div class="sa-kpi-value">{esc(value)}</div>'
            f'<div class="sa-kpi-delta {cls}">{esc(delta)}</div>{mini}'
            + (f'<div class="sa-kpi-foot" title="Top 5 holdings">{esc(" · ".join(foot.split()))}</div>' if foot else "") + f'{pop}</div>')
