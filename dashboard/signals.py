"""Plain-language signals (display only; the CSV values stay unchanged) and the next-rebalance plan."""
import re

import numpy as np
import pandas as pd

from dashboard.data import last_next_earnings, read_report_csv
from dashboard.settings import (CHANGES_CSV, EARNINGS, EXIT_BELOW, MAX_PICK, MIDWEEK, N_PICKS, PICKS_CSV, SECTOR_MAX,
    symbol_sector)


SIGNALS = ["Buy", "Hold", "Sold", "Score below 0", "Watch", "Watch (sector limit)", "Not ranked"]
SIGNAL_BADGE = {"Buy": "sa-badge-bull", "Hold": "sa-badge-hold", "Sold": "sa-badge-bear",
                "Score below 0": "sa-badge-bear", "Watch": "sa-badge-hold", "Watch (sector limit)": "sa-badge-hold",
                "Not ranked": "sa-badge-grey"}
PLAN_BADGE = {"Buy": "sa-badge-bull", "Keep": "sa-badge-hold", "Sell": "sa-badge-bear"}


def plain_reason(signal, reason, rank=None, score=None):
    """Engine reason string -> short plain-English explanation for the given display signal."""
    reason = "" if reason is None or (isinstance(reason, float) and pd.isna(reason)) else str(reason)
    r = f"{rank:.0f}" if rank is not None and pd.notna(rank) else "?"
    sc = f"{score:.1f}" if score is not None and pd.notna(score) else "?"
    m = re.search(r"rank (\d+)", reason)
    if reason.startswith("earnings in"):
        return f"rank {r}, not bought: {reason.split(': not bought')[0]} (no new buys within {EARNINGS or 5} days of earnings)"
    if reason.startswith("mid-week swap in"):
        rep_m = re.search(r"replaces (\S+)", reason)
        return (f"mid-week swap: jumped into the top {MIDWEEK['enter_top'] if MIDWEEK else 3} at rank "
                f"{m.group(1) if m else r}" + (f", replaces {rep_m.group(1)}" if rep_m else ""))
    if reason.startswith("mid-week exit"):
        return (f"mid-week exit: {'rank ' + m.group(1) if m else 'no longer ranked'} is worse than {EXIT_BELOW or 30}; "
                "sold, cash until the Friday rebalance")
    if reason.startswith("mid-week swap out"):
        by = re.search(r"replaced by (\S+)", reason)
        return (f"mid-week swap: fell to {'rank ' + m.group(1) if m else 'no longer qualifying'} "
                f"(below {MIDWEEK['exit_below'] if MIDWEEK else 15})" + (f", replaced by {by.group(1)}" if by else ""))
    if signal == "Buy":
        rk = int(m.group(1)) if m else (int(rank) if rank is not None and pd.notna(rank) else None)
        if "sector cap relaxed" in reason:
            return f"made the portfolio at rank {rk} (free slot filled from the top {MAX_PICK or 20}, sector limit relaxed)"
        if rk is not None and rk > N_PICKS:
            return (f"made the portfolio at rank {rk} (higher-ranked stocks were skipped by the {SECTOR_MAX}-per-sector limit"
                    + (" or the earnings rule)" if EARNINGS else ")"))
        return f"made the top {N_PICKS} at rank {rk if rk is not None else r}"
    if signal == "Hold":
        return f"in top {N_PICKS}, rank {r}" if rank is not None and pd.notna(rank) and rank <= N_PICKS else \
            f"still selected at rank {r} (higher-ranked stocks skipped by the sector limit)"
    if signal == "Sold":
        if reason.startswith("score"):
            return f"score fell below 0 ({sc})"
        if "picks only from ranks" in reason:
            return f"fell to rank {m.group(1) if m else r}, worse than {MAX_PICK} (picks only from ranks 1–{MAX_PICK})"
        if reason.startswith("skipped"):
            return f"skipped: already {SECTOR_MAX} stocks from this sector"
        if "outside top" in reason:
            return f"fell to rank {m.group(1) if m else r}, outside top {N_PICKS}"
        if reason.startswith("not eligible"):
            return "not enough data / not eligible"
        return reason or f"left the top {N_PICKS}"
    if signal == "Watch (sector limit)":
        return f"rank {r} but skipped: already {SECTOR_MAX} stocks from this sector"
    if signal == "Watch":
        return f"rank {r}, positive score but outside top {N_PICKS} — watch"
    if signal == "Score below 0":
        return f"score below 0 ({sc})"
    return "benchmark / not enough history to rank"


def decision_tag(signal, reason):
    """Short reason shown in brackets on the dated decision badge, e.g. 'Sold (sector limit)'."""
    reason = "" if reason is None or (isinstance(reason, float) and pd.isna(reason)) else str(reason)
    if signal == "Sold":
        for key, tag in (("mid-week exit", "mid-week exit"), ("mid-week swap out", "mid-week swap"), ("skipped", "sector limit"),
                         ("picks only from ranks", f"rank worse than {MAX_PICK}"), ("outside top", f"outside top {N_PICKS}"),
                         ("score", "score below 0"), ("not eligible", "not eligible")):
            if key in reason:
                return tag
    if signal == "Buy" and reason.startswith("mid-week swap in"):
        return "mid-week swap"
    return ""


def signal_board(df):
    """Every symbol's signal for the decisions in force (the last rebalance / mid-week check).

    Uses Reports/strategy_changes.csv (the engine's decisions) plus signal_analysis.csv for stocks not in it.
    Returns (board DataFrame, decision date)."""
    ch = read_report_csv(CHANGES_CSV)
    sub = ch[ch["Symbol"].notna()] if ch is not None else pd.DataFrame()
    if not sub.empty:
        # The decisions in force are the LATEST ones: sort newest-first so iloc[0]
        # and drop_duplicates(keep="first") both pick the latest decision per symbol.
        sub = sub.sort_values("Date", ascending=False)
        date = pd.Timestamp(sub["Date"].iloc[0])
    else:
        reb = df.loc[df["Rebalance_Day"] == 1, "Date"]
        date = reb.max() if len(reb) else df["Date"].max()
    day = df[df["Date"] == date].drop_duplicates("Symbol").set_index("Symbol")
    dec = sub.drop_duplicates("Symbol").set_index("Symbol") if not sub.empty else pd.DataFrame()
    next_ed = last_next_earnings(list(day.index)).set_index("Symbol")["Next ED"]
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    rows = []
    for sym, r in day.iterrows():
        d = dec.loc[sym] if sym in dec.index else None
        score = d["Score"] if d is not None and pd.notna(d["Score"]) else r.get("Strategy_Score")
        rank = d["Rank"] if d is not None and pd.notna(d["Rank"]) else r.get("Strategy_Rank")
        status = d["Status"] if d is not None else None
        if status == "add":
            sig, weight = "Buy", d["New_Weight"]
        elif status == "hold":
            sig, weight = "Hold", d["New_Weight"]
        elif status == "drop":
            sig, weight = "Sold", d["Old_Weight"]
        elif d is not None and str(d["Reason"]).startswith("skipped"):
            sig, weight = "Watch (sector limit)", 0.0
        elif pd.isna(score):
            sig, weight = "Not ranked", np.nan
        else:
            sig, weight = ("Score below 0" if score <= 0 else "Watch"), 0.0
        ned = next_ed.get(sym, "")
        soon = ""
        if ned:
            n_days = int(np.busday_count(today.date(), pd.Timestamp(ned).date()))
            soon = "⚠️ within 2 sessions" if 0 <= n_days <= 2 else ""
        sector = d["Sector"] if d is not None and "Sector" in d and pd.notna(d["Sector"]) else symbol_sector.get(sym)
        rows.append({"Symbol": sym, "Signal": sig, "Rank": rank, "Score": score,
                     "Portfolio weight %": weight * 100 if pd.notna(weight) else np.nan, "Sector": sector or "—",
                     "Why": plain_reason(sig, d["Reason"] if d is not None else None, rank, score),
                     "Tag": decision_tag(sig, d["Reason"] if d is not None else None),
                     "Next earnings": ned, "Earnings soon": soon})
    board = pd.DataFrame(rows, columns=["Symbol", "Signal", "Rank", "Score", "Portfolio weight %", "Sector", "Why",
                                        "Next earnings", "Earnings soon", "Tag"])
    board = board.sort_values(["Rank", "Symbol"], na_position="last").reset_index(drop=True)
    picked = board["Signal"].isin(["Buy", "Hold"])
    board.insert(2, "Portfolio slot", "—")
    board.loc[picked, "Portfolio slot"] = [str(i) for i in range(1, int(picked.sum()) + 1)]
    return board, date


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


def next_decision_date(latest_day):
    """(date, kind) of the next decision close: 'full rebalance' or 'mid-week check' (from backtest_engine)."""
    try:
        from backtest_engine import next_decision
        d, kind, _fill = next_decision(pd.Timestamp(latest_day))
        return d, kind
    except Exception:
        return None, None
