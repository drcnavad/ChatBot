"""Plain-language signals (display only; the CSV values stay unchanged) and the next-rebalance plan."""
import re

import pandas as pd
import streamlit as st

from dashboard.data import read_report_csv
from dashboard.settings import CHANGES_CSV, EARNINGS, EXIT_BELOW, EXIT_TO_TOP, N_PICKS, PICKS_CSV

SIGNAL_BADGE = {"Buy": "sa-badge-bull", "Hold": "sa-badge-hold", "Sold": "sa-badge-bear",
                "Score below 0": "sa-badge-bear", "Watch": "sa-badge-hold", "Not ranked": "sa-badge-grey"}
PLAN_BADGE = {"Buy": "sa-badge-bull", "Keep": "sa-badge-hold", "Take over": "sa-badge-hold", "Sell": "sa-badge-bear", "Blocked": "sa-badge-bear"}

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

@st.cache_data(ttl=60, max_entries=8, show_spinner=False)
def _live_rebalance_plan(slot, signal_mtime):
    """Cached next_plan for build_page: one Alpaca-light + sleeve compute per refresh slot, shared across tabs."""
    from dashboard.details.last_decision import next_plan
    return next_plan()


def rebalance_plan(df):
    """The Next rebalance preview (live RS short-term, Oct 27, 2026): (next rebalance, as-of, {SYMBOL: (action, % of account,
    day)}); falls back to strategy_picks.csv below when Alpaca cannot be read."""
    try:
        import db
        from dashboard.data import refresh_slot
        from dashboard.settings import SIGNAL_CSV
        np_ = _live_rebalance_plan(refresh_slot(), db.mtime(SIGNAL_CSV))
        if np_ is not None:
            return min(d for d in np_["days"].values() if d is not None), np_["asof"], np_["plan"]
    except Exception:
        pass
    return _pipeline_plan(df)

def _pipeline_plan(df):
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
    try:   # a buy the trade step skips shows as Blocked: a new do-not-buy pick, or any pick under a manual sell's block
        import manual_trade                # (the block exists only after the stock was sold, so it is not held)
        import paper_trade
        dnb, manual = set(paper_trade.manual_list("buy")), set(manual_trade.blocked_notes())
    except Exception:
        dnb, manual = set(), set()
    plan.update({s: ("Blocked", 0.0) for s, (a, _) in list(plan.items())
                 if (a == "Buy" and s in dnb) or (a in ("Buy", "Keep") and s in manual)})
    return day, as_of, plan

def plan_text(plan, sym):
    """'Buy 14.23%' / 'Keep 9.14%' / 'Sell' / 'not picked'."""
    action, w = plan.get(sym, (None, 0.0))[:2]
    return f"{action} {w:.2f}%" if action in ("Buy", "Keep", "Take over") else (action or "not picked")


# ----------------------------------------------------------------------------- live Short-term relative strength note
LIVE_STRATEGY = "Short-term relative strength"
LIVE_N = 10          # top-N equal weight (forward_test LIVE_NOW / sleeves)
LIVE_EXIT_BELOW = 20 # Mon-Thu: sell when worse than this rank


def _held_symbols():
    """Uppercase symbols currently held in live Alpaca (empty on failure)."""
    try:
        from dashboard.details.live_holdings import account_core
        data, _ = account_core()
        if not data:
            return set()
        return {str(r.get("symbol") or "").upper()
                for r in (data.get("positions") or []) if float(r.get("qty") or 0) > 0}
    except Exception:
        return set()


def _overlays(sym, held=False):
    """Extra chips: do_not_sell/buy, sold_block, target limit sell + order status, drop_guard."""
    import sector_mapping as sm
    chips = []  # (text, css_class)
    su = str(sym).upper()
    if su in {str(x).upper() for x in sm.do_not_sell}:
        chips.append(("do_not_sell", "sa-badge-hold"))
    if su in {str(x).upper() for x in sm.do_not_buy}:
        chips.append(("do_not_buy", "sa-badge-bear"))
    try:
        import sold_block as sb
        blk = sb.active_blocks()
        if su in blk:
            chips.append((blk[su], "sa-badge-bear"))
    except Exception:
        pass
    try:
        import target_sells as ts
        cfg = ts.load_config()
        entry = (cfg.get("targets") or {}).get(su)
        if entry and float(entry.get("limit") or 0) > 0:
            limit = float(entry["limit"])
            st = ts._load_state()
            o = (st.get("orders") or {}).get(su) or {}
            oid, day = o.get("order_id") or "", o.get("day") or ""
            if oid:
                status = f"open order {str(oid)[:8]}… · day {day}" if day else f"open order {str(oid)[:8]}…"
            else:
                status = "no open broker order yet (job refreshes 3 AM–7 PM CT session days)"
            chips.append((f"Limit sell: ${limit:,.2f} · {status}", "sa-badge-hold"))
    except Exception:
        pass
    try:
        import json, os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "Reports", "drop_guard_state.json")
        with open(path) as f:
            dg = json.load(f)
        sold = (dg.get("sold_today") or {}).get(su)
        if sold:
            chips.append((f"drop_guard sold today ({sold.get('at', '')[:16]})", "sa-badge-bear"))
        else:
            # only mention drop_guard when config is live-enabled — keep quiet otherwise
            cfg_path = os.path.join(os.path.dirname(path), "drop_guard_config.json")
            try:
                with open(cfg_path) as f:
                    dgc = json.load(f)
                if held and dgc.get("enabled") and not dgc.get("dry_run", True):
                    chips.append(("drop_guard armed (4.1% + AI)", "sa-badge-grey"))
            except Exception:
                pass
    except Exception:
        pass
    return chips


def live_stock_note(ticker, p):
    """Status for the Stock Analysis header under Short-term relative strength.

    Returns dict: status, badge, why, plan_text, plan_action, plan_tip, chips[(text,class)].
    Uses live ranks on `p.by_symbol` (build_page overwrites pipeline ranks) and `p.plan` from next_plan.
    """
    import math
    import pandas as pd
    from dashboard.details.last_decision import next_run
    import sleeves as sl

    sym = str(ticker).upper()
    held = sym in _held_symbols()
    rank = score = None
    if sym in getattr(p, "by_symbol", pd.DataFrame()).index:
        row = p.by_symbol.loc[sym]
        rank = row.get("Strategy_Rank")
        score = row.get("Strategy_Score")
        try:
            rank = float(rank) if rank == rank and rank is not None else None
        except (TypeError, ValueError):
            rank = None
        try:
            score = float(score) if score == score and score is not None else None
        except (TypeError, ValueError):
            score = None

    entry = (getattr(p, "plan", None) or {}).get(sym) or (getattr(p, "plan", None) or {}).get(ticker)
    action = entry[0] if entry else None
    weight = entry[1] if entry and len(entry) > 1 else 0.0
    plan_day = entry[2] if entry and len(entry) > 2 and entry[2] is not None else getattr(p, "plan_day", None)

    # Derive status from live plan + hold + rank (never from Oct-16 strategy_changes)
    if action == "Blocked":
        status = "Blocked"
    elif held and action == "Sell":
        status = "Sell"
    elif held and action in ("Keep", "Take over", "Buy"):
        status = "Hold"
    elif held and action is None:
        status = "Hold"  # held but not in next plan preview yet
    elif (not held) and action == "Buy":
        status = "Buy"
    elif rank is None and (score is None or (isinstance(score, float) and math.isnan(score))):
        status = "Not ranked"
    elif score is not None and score <= 0:
        status = "Score below 0"
    else:
        status = "Watch"

    rk = f"{int(rank)}" if rank is not None and rank == rank else "?"
    sc = f"{score:.1f}" if score is not None and score == score else "?"
    if status == "Hold":
        if rank is not None and rank <= LIVE_N:
            why = f"held · rank {rk} of top {LIVE_N} (score {sc})"
        elif rank is not None and rank <= LIVE_EXIT_BELOW:
            why = f"held · rank {rk} (inside Mon-Thu exit band ≤{LIVE_EXIT_BELOW}; still above sell line)"
        else:
            why = f"held · rank {rk} (score {sc})"
    elif status == "Sell":
        if rank is not None and rank > LIVE_EXIT_BELOW:
            why = f"held · rank {rk} worse than {LIVE_EXIT_BELOW} — Mon-Thu exit / Friday drop"
        elif rank is not None and rank > LIVE_N:
            why = f"held · rank {rk} outside top {LIVE_N} — next rebalance sells"
        else:
            why = f"held · plan sells (rank {rk}, score {sc})"
    elif status == "Buy":
        why = f"not held · rank {rk} in top {LIVE_N} — next run buys (~{weight:.1f}% of account)"
    elif status == "Blocked":
        why = f"rank {rk} would be bought but a buy block is active"
    elif status == "Watch":
        why = f"rank {rk}, score {sc} — outside top {LIVE_N}, not held"
    elif status == "Score below 0":
        why = f"score {sc} ≤ 0 — not eligible"
    else:
        why = "not enough history / not in the live ranking"

    asof = getattr(p, "plan_asof", None)
    asof_s = f"{asof:%a %b %-d}" if asof is not None else "latest close"
    badge = f"{LIVE_STRATEGY} · as of {asof_s}: {status}"
    if rank is not None and rank == rank:
        badge += f" · rank {int(rank)}"

    # Next run chip
    led = None
    try:
        led = sl.load_ledger()
    except Exception:
        pass
    slot, kind = next_run(ledger=led)
    if kind == "full rebalance":
        kind_label = "Friday rebalance"
    elif kind and "mid" in str(kind).lower():
        kind_label = "Mon-Thu check"
    elif kind:
        kind_label = str(kind)
    else:
        kind_label = "Next run"
    when = f"{slot:%a %b %-d %-I:%M %p CT}" if slot is not None else (
        f"{plan_day:%a %b %-d}" if plan_day is not None else "Next run")
    if action in ("Buy", "Keep", "Take over"):
        plan_txt = f"{kind_label} · {when}: {action} {weight:.2f}%"
    elif action:
        plan_txt = f"{kind_label} · {when}: {action}"
    else:
        plan_txt = f"{kind_label} · {when}: not picked"
    tip = ("Preview from the latest close under Short-term relative strength. "
           "Final orders go out at the 2:30 PM CT decision slot "
           "(or 11:00 AM CT on early-close days).")

    chips = _overlays(sym, held=held)
    if held:
        chips.insert(0, ("held in Alpaca", "sa-badge-bull"))
    return {
        "status": status, "badge": badge, "why": why,
        "plan_text": plan_txt, "plan_action": action or "not picked",
        "plan_tip": tip, "chips": chips, "rank": rank, "score": score, "held": held,
    }

