"""Live-strategy history for one ticker (entries / exits / held periods, portfolio slots, relative strength)."""
import html

import numpy as np
import pandas as pd

from dashboard.data import load_benchmarks
from dashboard.settings import HALVED, REGIME_OFF, sector_etf_for, symbol_sector
from dashboard.signals import plain_reason


EVENT_COLS = ["Kind", "Decision", "Fill", "Price", "Reason", "Rank", "Score", "Weight", "Regime_On"]


def _fallback_reason(kind, row, n=10):
    """Reason when the decision log is unavailable (derived from the saved rank/score columns)."""
    if kind == "entry":
        return f"selected (rank {row.Strategy_Rank:.0f})" if pd.notna(row.Strategy_Rank) else "selected"
    if pd.isna(row.Strategy_Score):
        return "not eligible / no data"
    if row.Strategy_Score <= 0:
        return f"score {row.Strategy_Score:.1f} <= 0"
    if pd.notna(row.Strategy_Rank) and row.Strategy_Rank > n:
        return f"rank {row.Strategy_Rank:.0f} outside top {n}"
    return "not picked"


def strategy_events(ticker_df, decisions, symbol):
    """Entries/exits and held periods of the live strategy for one ticker.

    Strategy_Weight on day d is the target decided at d's close (orders go out that day at 2:30 PM CT), so an entry/exit is the
    first day the weight turns >0 / back to 0; its marker sits on the following session ("Fill"). Reasons, rank and score come from
    Reports/strategy_decisions.csv. Returns (events, periods) with periods = [(first held session, last held session)]."""
    t = ticker_df.sort_values("Date")[["Date", "Close", "Strategy_Weight", "Strategy_Rank", "Strategy_Score", "Regime_On"]]
    t = t.reset_index(drop=True)
    empty = pd.DataFrame(columns=EVENT_COLS)
    if t["Strategy_Weight"].notna().sum() == 0:
        return empty, []
    held = (t["Strategy_Weight"].fillna(0) > 0).to_numpy()
    prev = np.r_[False, held[:-1]]
    dec = pd.DataFrame()
    if decisions is not None:
        dec = decisions[decisions["Symbol"] == symbol].drop_duplicates("Date", keep="last").set_index("Date")
    rows = []
    for i in np.flatnonzero(held != prev):
        if i == 0:  # already held when the data window starts
            continue
        kind = "entry" if held[i] else "exit"
        r = t.iloc[i]
        has_next = i + 1 < len(t)
        d = dec.loc[r.Date] if r.Date in dec.index else None
        rows.append({
            "Kind": kind, "Decision": r.Date, "Fill": t["Date"].iloc[i + 1] if has_next else pd.NaT,
            "Price": t["Close"].iloc[i + 1] if has_next else r.Close,
            "Reason": d["Reason"] if d is not None else _fallback_reason(kind, r),
            "Rank": d["Rank"] if d is not None and pd.notna(d["Rank"]) else r.Strategy_Rank,
            "Score": d["Score"] if d is not None and pd.notna(d["Score"]) else r.Strategy_Score,
            "Weight": r.Strategy_Weight if kind == "entry" else t["Strategy_Weight"].iloc[i - 1],
            "Regime_On": r.Regime_On,
        })
    events = pd.DataFrame(rows, columns=EVENT_COLS) if rows else empty
    periods, start = [], (t["Date"].iloc[0] if held[0] else None)
    for e in events.itertuples():
        if e.Kind == "entry":
            start = e.Fill if pd.notna(e.Fill) else None
        elif start is not None:
            periods.append((start, e.Fill if pd.notna(e.Fill) else t["Date"].iloc[-1]))
            start = None
    if start is not None:
        periods.append((start, t["Date"].iloc[-1]))
    return events, periods


def event_hover(e):
    """Hover text for an entry/exit marker on the price chart."""
    sig = "Buy" if e.Kind == "entry" else "Sold"
    text = f"<b>{sig}</b>: {html.escape(plain_reason(sig, e.Reason, e.Rank, e.Score))}"
    text += f"<br>Decided at the close {e.Decision:%a %b %-d} · orders go out that day" + ("" if pd.notna(e.Fill) else " (pending)")
    bits = ([f"Rank #{e.Rank:.0f}"] if pd.notna(e.Rank) else []) + ([f"Score {e.Score:.1f}"] if pd.notna(e.Score) else [])
    if pd.notna(e.Weight):
        bits.append(f"{'Portfolio weight' if e.Kind == 'entry' else 'Weight sold'} {e.Weight * 100:.2f}%")
    if bits:
        text += "<br>" + " · ".join(bits)
    if e.Regime_On == 0:
        text += f"<br>Market filter OFF that week ({REGIME_OFF}): positions {HALVED}"
    return text


RS_COLORS = {"stock": "#1d4ed8", "market": "#64748b", "sector": "#d97706"}   # stock blue, SPY grey, sector ETF orange


def performance_names(symbol):
    """Display names of the comparison lines: {'stock': 'FTNT', 'market': 'SPY (market)', 'sector': 'XLK (Technology sector)'}."""
    etf, sector = sector_etf_for(symbol), symbol_sector.get(symbol)
    names = {"stock": symbol, "market": "SPY (market)"}
    if etf and etf != symbol:
        names["sector"] = f"{etf} ({sector} sector)" if sector else f"{etf} (sector ETF)"
    return names


def relative_strength_lines(chart, symbol):
    """% price change since the start of the chart window: the stock, SPY and its sector ETF -> {key: (name, series)}."""
    bench = load_benchmarks()
    if bench is None:
        return {}
    b = bench.reindex(chart.index).ffill()
    names, refs = performance_names(symbol), {"stock": None, "market": "SPY", "sector": sector_etf_for(symbol)}
    lines = {}
    for key, name in names.items():
        if key != "stock" and (refs[key] not in b.columns):
            continue
        px = (chart["Close"] if key == "stock" else b[refs[key]]).replace([np.inf, -np.inf], np.nan)
        first = px.first_valid_index()
        if first is not None and px.loc[first]:
            lines[key] = (name, (px / px.loc[first] - 1) * 100)
    return lines if len(lines) > 1 else {}


def daily_status(weight, score):
    """Status on an ordinary day (between decisions) for the chart hover text."""
    if score is None or pd.isna(score):
        return "Not ranked"
    if weight is not None and pd.notna(weight) and weight > 0:
        return "Hold"
    return "Score below 0" if score <= 0 else "Watch"
