"""Strategy tab, Last decision (the two live halves from Oct 27, 2026: "RS short-term (weekly + Mon–Thu sells)" + "RS standard (Fridays only)"): a line per
half (stocks, value, next rebalance), each half's top-20 ranking at its latest decision (Reports/decision_ranks.csv, saved
by the trade run) with held stocks marked, then the next decision preview from the latest close. A stock both halves hold
is marked "Both". Weights from the live Alpaca positions; the ledger is never written here."""
import os

import pandas as pd
import streamlit as st

from dashboard.page import open_symbol
from dashboard.style import GOOD, INK, SCORE, table, toned

def _sleeves():
    import sleeves
    return sleeves

SLEEVE = {"rs_short": "RS short-term", "friday": "RS Fridays"}

def _state(positions, ledger, equity):
    """(reconciled state {half: {'shares', 'cash'}}, prices, market values) like the bot at trade time (read-only)."""
    sl = _sleeves()
    qty = {p["symbol"]: float(p.get("qty") or 0) for p in positions or []}
    px = {p["symbol"]: float(p.get("current_price") or 0) for p in positions or []}
    value = {p["symbol"]: float(p.get("market_value") or 0) for p in positions or []}
    state, _, _ = sl.reconcile(ledger, qty, px, equity)
    return state, px, value

def half_values(positions, ledger, equity):
    """{half: $ value} reconciled like the bot (sleeves.reconcile)."""
    sl = _sleeves()
    state, px, _ = _state(positions, ledger, equity)
    return {h: sl.half_value(state[h], px) for h in state}

def _mine(positions, ledger, equity):
    state, _, value = _state(positions, ledger, equity)
    return {h: sorted(s for s, q in state[h]["shares"].items() if q > 0 and s in value) for h in state}

def _next_full(after):
    sl = _sleeves()
    cal = sl.calendar(pd.Timestamp(after) + pd.Timedelta(days=1), 15)
    return cal[0][0] if cal else None

def sleeve_summary(day, positions, ledger, equity):
    """One line per half: its stocks, value, share of the account, next full rebalance."""
    sl = _sleeves()
    vals, mine, nxt = half_values(positions, ledger, equity), _mine(positions, ledger, equity), _next_full(day)
    out = []
    for h in sl.HALVES:
        v = vals[h]
        out.append(f"**{sl.LABEL[h]}**: {len(mine[h])} stocks · ${v:,.0f}"
                   + (f" ({100 * v / equity:.1f}% of the account)" if equity else "")
                   + (f" · next full rebalance {nxt:%a %b %-d}" if nxt is not None else "")
                   + (" · Mon-Thu: exit check (worse than rank 20)" if h == "rs_short" else ""))
    both = sorted(set(mine["rs_short"]) & set(mine["friday"]))
    if both:
        out.append("Held by both halves (each from its own budget): " + ", ".join(both))
    return out

def half_decision_day(half, path=None):
    """The latest day in Reports/decision_ranks.csv on which `half` decided (rs_short every session, friday on Fridays)."""
    import decision_ranks
    sl = _sleeves()
    path = path or decision_ranks.RANKS_CSV
    from dashboard.data import read_report_csv
    try:
        r = read_report_csv(path)
    except Exception:
        return None
    if r is None or "Date" not in r or decision_ranks.RANK_COL[half] not in r.columns:
        return None
    days = sorted({str(d)[:10] for d in r["Date"]}, reverse=True)
    return next((pd.Timestamp(d) for d in days if sl.half_day(half, d)), None)

def _scores(half, day):
    sl = _sleeves()
    try:
        s = sl.ranking(half, day)
    except Exception:
        return {}
    return s.to_dict()

def ranking_table(half, day, positions, ledger, top=20, path=None, values=None, equity=None):
    """(table, others): Rank, Stock, Sector, Score, Held by for the half's top `top` ranks at `day`; others = the stocks
    the half holds outside that top."""
    import decision_ranks
    from dashboard.settings import symbol_sector
    col = decision_ranks.RANK_COL[half]
    r = decision_ranks.ranks_on(day)[1] if path == "__live__" else decision_ranks.load(day, path=path)
    r = r.dropna(subset=[col]).sort_values(col) if r is not None else pd.DataFrame(columns=["Symbol", col])
    eq = equity if equity is not None else sum(float(p.get("market_value") or 0) for p in positions or [])
    mine = _mine(positions, ledger, eq)
    score = _scores(half, day)
    r = r[r[col] <= top]
    others = sorted(set(mine[half]) - set(r["Symbol"]))
    def held(x):
        hs = [h for h in mine if x in mine[h]]
        return "✓ Both" if len(hs) == 2 else ("✓ " + SLEEVE[hs[0]]) if hs else ""
    t = pd.DataFrame({"Rank": pd.array(pd.to_numeric(r[col], errors="coerce").round(), dtype="Int64"), "Stock": r["Symbol"].to_numpy(),
                      "Sector": [symbol_sector.get(x, "—") for x in r["Symbol"]],
                      "Score": [round(float(score.get(x)), 2) if score.get(x) is not None and pd.notna(score.get(x)) else None
                                for x in r["Symbol"]],
                      "Held by": [held(x) for x in r["Symbol"]]})
    return t, others

@st.fragment(run_every=30)
def render_last_decision(p):
    """Strategy tab: per-half summary, each half's ranking at its latest decision, then the next decision preview."""
    from dashboard.page import fresh
    p = fresh(p)
    sl = _sleeves()
    from dashboard.details.live_holdings import holdings_this_run
    led = sl.load_ledger()
    data, err = holdings_this_run()
    if data is None:
        st.info(f"Last decision: {err} (live positions are the only holdings source)")
        return
    positions, equity = data["positions"], data["equity"]
    if led is None:
        st.info("The two halves (RS short-term (weekly + Mon–Thu sells) + RS standard (Fridays only)) start at the first trade run from Tue Oct 27, 2026 "
                "2:30 PM CT (Reports/sleeve_ledger.json is written then).")
    else:
        for line in sleeve_summary(pd.Timestamp(led["day"]), positions, led, equity):
            st.markdown(line)
        cols = st.columns(2)
        for c, half in zip(cols, sl.HALVES):
            with c:
                day = half_decision_day(half)
                if day is None:
                    st.info(f"{sl.LABEL[half]}: no saved decision ranks yet.")
                    continue
                st.markdown(f"**{sl.LABEL[half]}** · ranks at its decision of {day:%a %b %-d}")
                t, others = ranking_table(half, day, positions, led, equity=equity)
                sty = toned(t, ["Held by"], fn=lambda v: GOOD if str(v).startswith("✓") else INK)
                event = table(sty, fit=True, cols={"Score": SCORE}, on_select="rerun", selection_mode="single-row",
                              key=f"rank_tbl_{half}")
                if others:
                    st.caption(f"Held, not in top 20: {', '.join(others)}")
                open_symbol(t.rename(columns={"Stock": "Symbol"}), event, f"rank_{half}")
    st.markdown("##### Next decision preview · based on the latest close")
    render_preview(positions, led, equity)

# ---------------------------------------------------------------- next decision preview (read-only, no orders)
@st.cache_data(ttl=600, show_spinner=False)
def _preview_targets(day):
    """{half: {symbol: target weight in the half}} at the close `day` = sleeves.half_weights (the forward test's own
    targets; manual buy blocks shown as Blocked)."""
    sl = _sleeves()
    return {h: {s: float(w) for s, w in sl.half_weights(h, day).items() if w > 0} for h in sl.HALVES}

def next_plan():
    """{"asof", "days": {half: next decision}, "targets", "plan": {sym: (action, % of account, day)}} or None.
    Action: Keep / Buy / Sell / Blocked (both halves' weights summed). Display only."""
    import paper_trade as pt
    sl = _sleeves()
    from dashboard.details.live_holdings import holdings_this_run
    try:
        led = sl.load_ledger()
        data, _ = holdings_this_run()
        if data is None:
            return None
        positions, equity = data["positions"], data["equity"]
        asof = sl.inputs()["close"].index[-1]
        t = _preview_targets(asof)
        vals = half_values(positions, led, equity)
        value = {p["symbol"]: float(p.get("market_value") or 0) for p in positions or []}
    except Exception:
        return None
    import backtest_engine as be
    nxt = be.next_sessions(pd.Timestamp.now().normalize() - pd.Timedelta(days=1), 1)[0]
    full = _next_full(pd.Timestamp.now().normalize() - pd.Timedelta(days=1))
    days = {"rs_short": nxt, "friday": full}
    tot = sum(vals.values()) or 1.0
    blocked = set(pt.manual_list("buy"))
    plan = {}
    for h in sl.HALVES:
        for s, w in t[h].items():
            a, pct, d = plan.get(s, ("Keep" if s in value else "Buy", 0.0, days[h]))
            plan[s] = (a, pct + 100 * w * vals[h] / tot, min(d, days[h]) if d is not None and days[h] is not None else d)
    for s in value:
        plan.setdefault(s, ("Sell", 0.0, nxt))
    for s in blocked:
        if s in plan and plan[s][0] == "Buy":
            plan[s] = ("Blocked", 0.0, plan[s][2])
    return {"asof": asof, "days": days, "targets": t, "plan": plan}

def render_preview(positions, led, equity):
    """Each half's top 20 at the latest close with its plan (Keep / Buy / Blocked) and what it would sell."""
    import decision_ranks
    import paper_trade as pt
    sl = _sleeves()
    try:
        day = sl.inputs()["close"].index[-1]
        targets = _preview_targets(day)
    except Exception as e:
        st.info(f"Next decision preview unavailable: {str(e)[:150]}")
        return
    blocked = set(pt.manual_list("buy"))
    mine = _mine(positions, led, equity) if led is not None else {h: [] for h in sl.HALVES}
    cols = st.columns(2)
    for c, half in zip(cols, sl.HALVES):
        with c:
            st.markdown(f"**{sl.LABEL[half]}** · targets at the {day:%a %b %-d} close"
                        + (" (Mon-Thu only the exit check trades)" if half == "rs_short" else " (traded on Fridays)"))
            t, _ = ranking_table(half, day, positions, led, path="__live__", equity=equity)
            tg = targets[half]
            plan = lambda s: ("Blocked" if s in blocked else "Keep" if s in mine[half] else "Buy") if s in tg else ""
            t.insert(1, "Plan", [plan(s) for s in t["Stock"]])
            table(toned(t, ["Plan"], fn=lambda v: GOOD if v == "Buy" else INK), fit=True, cols={"Score": SCORE},
                  key=f"preview_{half}")
            extra = sorted(set(tg) - set(t["Stock"]))
            if extra:
                st.caption(f"Also targeted (outside the top 20 / defensive stocks): {', '.join(extra)}")
            sells = sorted(set(mine[half]) - set(tg))
            st.markdown("Would sell: " + (", ".join(sells) if sells else "nothing"))
