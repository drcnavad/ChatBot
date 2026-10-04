"""Earnings planner for the dashboard's Details tab (approved by Chirag, Sun Oct 4, 2026). Read-only, display only.

For each stock the LIVE account holds with an earnings date in the next 14 days: the date and time (before the open /
after the close), days until it, whether the pre-earnings stop window is open, the current 3x ATR stop (the same
earnings_stop.stop_level / active_event code the live stop job uses) and its % distance below the last price, and the
stock's past earnings reactions from the cached daily bars (median gap and 5-day move, and how many reports that is).
None in the next 14 days: the next 3 upcoming reports instead. Nothing here sends orders or calls a paid API.
"""
import os

import numpy as np
import pandas as pd

import backtest_engine as be
import earnings_stop as es

HORIZON_DAYS, SHOW_NEXT = 14, 3
CACHE_FILES = ("bars_daily_long.pkl", "forward_bars.pkl")      # Reports/cache, written by the pipeline / forward test
COLS = ["Stock", "Earnings", "Time", "Days", "Stop window", "Last price", "Stop", "To stop %", "Median gap %",
        "Median 5-day %", "Reports"]


def cached_bars(cache_dir=None):
    """Daily bars from the local caches (no download): the long backtest cache + the forward-test cache (newer wins)."""
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reports", "cache")
    parts = []
    for f in CACHE_FILES:
        try:
            parts.append(pd.read_pickle(os.path.join(cache_dir, f)))
        except (OSError, ValueError, EOFError):
            continue
    if not parts:
        return pd.DataFrame(columns=["Symbol", "Date", "Open", "High", "Low", "Close"])
    bars = pd.concat(parts, ignore_index=True)
    bars["Date"] = pd.to_datetime(bars["Date"]).dt.normalize()
    return bars.drop_duplicates(["Symbol", "Date"], keep="last").sort_values(["Symbol", "Date"]).reset_index(drop=True)


def time_label(t):
    t = str(t).strip().upper() if t is not None and t == t else ""
    return {"AM": "Before open", "PM": "After close"}.get(t, "Time not set")


def reactions(sym, earnings, bars, today):
    """Past reports of `sym` with bars: [(earnings date, gap %, 5-day %)]. Gap = reaction-day open vs the close before
    it; 5-day = the close of the 5th trading day from the reaction day (inclusive) vs that same close."""
    b = bars[bars["Symbol"] == sym].sort_values("Date").reset_index(drop=True)
    if b.empty:
        return []
    out = []
    dates = b["Date"].values
    for _, r in earnings[(earnings["Symbol"] == sym) & (earnings["Earnings Date"] < pd.Timestamp(today))].iterrows():
        react = es.reaction_day(r["Earnings Date"], r.get("Time"))
        i = int(np.searchsorted(dates, np.datetime64(react)))
        if i <= 0 or i + 4 >= len(b) or b["Date"].iloc[i] != react:
            continue
        prev = float(b["Close"].iloc[i - 1])
        out.append((r["Earnings Date"], (float(b["Open"].iloc[i]) / prev - 1) * 100, (float(b["Close"].iloc[i + 4]) / prev - 1) * 100))
    return out


def plan(held, entry, earnings, bars, today, horizon=HORIZON_DAYS, show_next=SHOW_NEXT):
    """(table, in_range). held = {symbol: last price}; entry = {symbol: first buy date still held}. in_range False = no
    held stock reports within `horizon` days and the table lists the next `show_next` upcoming reports instead."""
    today = pd.Timestamp(today).normalize()
    e = earnings[earnings["Symbol"].isin(list(held)) & (earnings["Earnings Date"] >= today)]
    nxt = e.sort_values("Earnings Date").groupby("Symbol").head(1)
    soon = nxt[nxt["Earnings Date"] <= today + pd.Timedelta(days=horizon)]
    in_range = not soon.empty
    pick = soon if in_range else nxt.head(show_next)
    rows = []
    for _, r in pick.iterrows():
        s, d = r["Symbol"], r["Earnings Date"]
        t = r.get("Time")
        ev = es.active_event(s, earnings, today)
        react = es.reaction_day(d, t)
        window = (f"Active through {react:%a %b %-d}" if ev is not None else
                  f"Opens {d - pd.Timedelta(days=es.ARM_DAYS):%a %b %-d}")
        px = held.get(s)
        lvl = es.stop_level(bars[(bars["Symbol"] == s) & (bars["Date"] < today)], entry[s]) if s in entry else None
        stop = lvl[2] if lvl else np.nan
        past = reactions(s, earnings, bars, today)
        rows.append({"Stock": s, "Earnings": d, "Time": time_label(t), "Days": int((d - today).days), "Stop window": window,
                     "Last price": px, "Stop": stop,
                     "To stop %": (px - stop) / px * 100 if px and stop == stop else np.nan,
                     "Median gap %": float(np.median([g for _, g, _ in past])) if past else np.nan,
                     "Median 5-day %": float(np.median([m for _, _, m in past])) if past else np.nan,
                     "Reports": len(past)})
    return pd.DataFrame(rows, columns=COLS), in_range
