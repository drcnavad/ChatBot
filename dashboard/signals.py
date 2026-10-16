"""Plain-language signals (display only; the CSV values stay unchanged) and the next-rebalance plan."""
import re

import pandas as pd

from dashboard.data import read_report_csv
from dashboard.settings import CHANGES_CSV, EARNINGS, EXIT_BELOW, EXIT_TO_TOP, N_PICKS, PICKS_CSV

SIGNAL_BADGE = {"Buy": "sa-badge-bull", "Hold": "sa-badge-hold", "Sold": "sa-badge-bear",
                "Score below 0": "sa-badge-bear", "Watch": "sa-badge-hold", "Not ranked": "sa-badge-grey"}
PLAN_BADGE = {"Buy": "sa-badge-bull", "Keep": "sa-badge-hold", "Sell": "sa-badge-bear"}

def plain_reason(signal, reason, rank=None, score=None):
    """Engine reason string -> short plain-English explanation for the given display signal."""
    reason = "" if reason is None or (isinstance(reason, float) and pd.isna(reason)) else str(reason)
    r = f"{rank:.0f}" if rank is not None and pd.notna(rank) else "?"
    sc = f"{score:.1f}" if score is not None and pd.notna(score) else "?"
    m = re.search(r"rank (\d+)", reason)
    if reason.startswith("earnings in"):
        return f"rank {r}, not bought: {reason.split(': not bought')[0]} (no new buys within {EARNINGS or 5} days of earnings)"
    if reason.startswith("mid-week exit replace in"):
        rep_m = re.search(r"replaces (\S+)", reason)
        return (f"mid-week exit refill: into top-{EXIT_TO_TOP or 10} at rank {m.group(1) if m else r}"
                + (f", replaces {rep_m.group(1)}" if rep_m else ""))
    if reason.startswith("mid-week exit"):
        by = re.search(r"replaced by (\S+)", reason)
        if by:
            return (f"mid-week exit: {'rank ' + m.group(1) if m else 'no longer ranked'} is worse than {EXIT_BELOW or 30}; "
                    f"replaced by {by.group(1)}")
        return (f"mid-week exit: {'rank ' + m.group(1) if m else 'no longer ranked'} is worse than {EXIT_BELOW or 30}; "
                f"sold (no top-{EXIT_TO_TOP or 10} stock left to refill)")
    if signal == "Buy":
        rk = int(m.group(1)) if m else (int(rank) if rank is not None and pd.notna(rank) else None)
        if rk is not None and rk > N_PICKS:
            return (f"made the portfolio at rank {rk}"
                    + (" (higher-ranked stocks skipped by the earnings rule)" if EARNINGS else ""))
        return f"made the top {N_PICKS} at rank {rk if rk is not None else r}"
    if signal == "Hold":
        if rank is not None and pd.notna(rank) and rank <= N_PICKS:
            return f"in top {N_PICKS}, rank {r}"
        return f"still selected at rank {r} (held through a mid-week check)"
    if signal == "Sold":
        if reason.startswith("score"):
            return f"score fell below 0 ({sc})"
        if "outside top" in reason:
            return f"fell to rank {m.group(1) if m else r}, outside top {N_PICKS}"
        if reason.startswith("not eligible"):
            return "not enough data / not eligible"
        return reason or f"left the top {N_PICKS}"
    if signal == "Watch":
        return f"rank {r}, positive score but outside top {N_PICKS} — watch"
    if signal == "Score below 0":
        return f"score below 0 ({sc})"
    return "benchmark / not enough history to rank"

def decision_tag(signal, reason):
    """Short reason shown in brackets on the dated decision badge, e.g. 'Sold (mid-week exit)'."""
    reason = "" if reason is None or (isinstance(reason, float) and pd.isna(reason)) else str(reason)
    if signal == "Sold":
        for key, tag in (("mid-week exit", "mid-week exit"), ("outside top", f"outside top {N_PICKS}"), ("score", "score below 0"), ("not eligible", "not eligible")):
            if key in reason:
                return tag
    if signal == "Buy" and reason.startswith("mid-week exit replace in"):
        return "mid-week exit refill"
    return ""

def signal_board(df):
    """Every symbol's signal (Buy / Hold / Sold / Watch / Score below 0 / Not ranked), plain reason and tag for the
    decisions in force (Reports/strategy_changes.csv, else the last rebalance day's ranks). Returns (DataFrame Symbol,
    Signal, Why, Tag; decision date)."""
    ch = read_report_csv(CHANGES_CSV)
    sub = ch[ch["Symbol"].notna()] if ch is not None else pd.DataFrame()
    if not sub.empty:
        sub = sub.sort_values("Date", ascending=False)      # newest first: the decision in force per symbol
        date = pd.Timestamp(sub["Date"].iloc[0])
    else:
        reb = df.loc[df["Rebalance_Day"] == 1, "Date"]
        date = reb.max() if len(reb) else df["Date"].max()
    day = df[df["Date"] == date].drop_duplicates("Symbol").set_index("Symbol")
    dec = sub.drop_duplicates("Symbol").set_index("Symbol") if not sub.empty else pd.DataFrame()
    status_sig = {"add": "Buy", "hold": "Hold", "drop": "Sold"}
    rows = []
    for sym, r in day.iterrows():
        d = dec.loc[sym] if sym in dec.index else None
        score = d["Score"] if d is not None and pd.notna(d["Score"]) else r.get("Strategy_Score")
        rank = d["Rank"] if d is not None and pd.notna(d["Rank"]) else r.get("Strategy_Rank")
        sig = status_sig.get(d["Status"] if d is not None else None)
        if sig is None:
            sig = "Not ranked" if pd.isna(score) else ("Score below 0" if score <= 0 else "Watch")
        reason = d["Reason"] if d is not None else None
        rows.append({"Symbol": sym, "Signal": sig, "Why": plain_reason(sig, reason, rank, score),
                     "Tag": decision_tag(sig, reason)})
    return pd.DataFrame(rows, columns=["Symbol", "Signal", "Why", "Tag"]), date

def next_full_rebalance(day):
    """Date of the first full (Friday) rebalance after `day` (backtest_engine's decision calendar), or None."""
    try:
        from backtest_engine import next_decision
        for _ in range(6):
            day, kind, _fill = next_decision(pd.Timestamp(day))
            if kind == "full rebalance":
                return day
    except Exception:
        pass
    return None

def rebalance_plan(df):
    """(rebalance date, as-of date, {SYMBOL: (action, weight %)}) for the next full rebalance.

    Provisional_Weight in strategy_picks.csv = the full rebalance computed at the latest close - the numbers the trade
    step (paper_trade.py) trades on the rebalance day. Action vs the holdings going into that rebalance: Buy (new),
    Keep (stays, brought to the weight) or Sell (leaves); a stock not listed is not picked."""
    picks = read_report_csv(PICKS_CSV)
    if picks is None or picks.empty or "Provisional_Weight" not in picks.columns:
        return None, None, {}
    as_of, last_reb = pd.Timestamp(picks["As_Of"].iloc[0]), pd.Timestamp(picks["Last_Rebalance"].iloc[0])
    day = as_of if as_of == last_reb else next_full_rebalance(as_of)
    before = df[df["Date"] < day] if day is not None else df
    prev = before[before["Date"] == before["Date"].max()]
    held = set(prev.loc[prev["Strategy_Weight"].fillna(0) > 0, "Symbol"])
    plan = {s: ("Keep" if s in held else "Buy", w * 100)
            for s, w in zip(picks["Symbol"], picks["Provisional_Weight"].fillna(0.0)) if w > 0}
    plan.update({s: ("Sell", 0.0) for s in held if s not in plan})
    return day, as_of, plan

def plan_text(plan, sym):
    """'Buy 14.23%' / 'Keep 9.14%' / 'Sell' / 'not picked'."""
    action, w = plan.get(sym, (None, 0.0))
    return f"{action} {w:.2f}%" if action in ("Buy", "Keep") else (action or "not picked")
