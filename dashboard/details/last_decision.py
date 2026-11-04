"""Strategy tab, Decisions (Nov 4, 2026, Chirag): ONE table per stock - the last Friday rebalance (orders in
Reports/live_orders_log.csv with that As_Of, ranks in Reports/decision_ranks.csv) next to what a Friday rebalance at the
latest close would do (sleeves.plan through the live code path, read-only: no orders, the ledger is never written)."""

import pandas as pd
import streamlit as st

from dashboard.page import open_symbol
from dashboard.style import BAD, GOOD, INK, table, toned


def _positions(positions):
    return ({p["symbol"]: float(p.get("qty") or 0) for p in positions or [] if float(p.get("qty") or 0) > 0},
            {p["symbol"]: float(p.get("current_price") or 0) for p in positions or []})


@st.cache_data(ttl=600, show_spinner=False)
def _preview(key, positions_t, prices_t, equity):
    """(day, orders, info) of a Friday rebalance at the latest close: sleeves.plan with rebalance=True (the live run's
    plan, blocks and all - manual / earnings-stop buy blocks, stretched, earnings, sector cap, regime)."""
    import paper_trade as pt
    import sleeves as sl
    d, inp = sl.rebalance_inputs()
    no_buy = {**pt.manual_list("buy"), **pt.earnings_stop_blocked(d.date().isoformat())}
    o, info = sl.plan(d, dict(positions_t), equity, dict(prices_t), no_buy=no_buy, no_sell=pt.manual_list("sell"),
                      inp=inp, rebalance=True)
    return d, o, {"no_buy": no_buy, "targets": info["targets"]}


def preview(positions, equity):
    import sleeves as sl
    pos, px = _positions(positions)
    key = str(sl.inputs()["close"].index[-1].date())
    return _preview(key, tuple(sorted(pos.items())), tuple(sorted(px.items())), float(equity))


def last_rebalance(path=None):
    """(day, {SYMBOL: 'Buy Oct 30' | 'Sell ...'}) of the latest Friday rebalance in the order log, plus any trade after it
    (e.g. the Nov 2 mid-week run), each labelled with its decision day (fill-check rows count as their decision)."""
    import db
    import sleeves as sl
    from paper_trade import ORDER_LOG_CSV
    path = path or ORDER_LOG_CSV
    o = db.read_csv(path) if db.exists(path) else None
    if o is None or o.empty:
        return None, {}
    o = o[~o["Status"].astype(str).str.upper().str.startswith(("STAGED", "FAIL", "SKIP", "REFUSED"))]
    days = sorted({str(x)[:10] for x in o["As_Of"].dropna()}, reverse=True)
    day = next((pd.Timestamp(x) for x in days if sl.full_day(sl.HALVES[0], x)), None)
    if day is None:
        return None, {}
    o = o.assign(D=pd.to_datetime(o["As_Of"].astype(str).str[:10], errors="coerce"))
    mine = o[o["D"] >= day]
    return day, {s: f"{str(g['Side'].iloc[-1]).capitalize()} {g['D'].iloc[-1]:%b %-d}" for s, g in mine.groupby("Symbol")}


def decision_table(positions, equity):
    """(table, last day, rank-now day, next day): Stock, Sector, Held now (live Alpaca positions), Last week (latest
    action since the last Friday: Buy / Sell / Trim = partial sell, still held / Hold), Rank then, Rank now, Next, Why."""
    import decision_ranks
    import sleeves as sl
    from dashboard.settings import symbol_sector
    held, _ = _positions(positions)
    inp = sl.inputs()
    last, side = last_rebalance()
    side = {k: v.replace("Sell", "Trim", 1) if k in held else v for k, v in side.items()}   # a partial sell, still held
    col = decision_ranks.RANK_COL[sl.HALVES[0]]
    then = decision_ranks.load(last) if last is not None else None
    then = {} if then is None else dict(zip(then["Symbol"], then[col]))
    now = {s: r for s, (r, _) in sl.live_ranks(inp=inp).items()}
    d, o, info = preview(positions, equity)
    nxt = {r.Symbol: r.Side for r in o.itertuples()}
    book = set(info["targets"][sl.HALVES[0]])
    stretched = inp["stretched_live"].loc[d] if d in inp["stretched_live"].index else pd.Series(dtype=bool)
    soon = inp["soon"].loc[d] if d in inp["soon"].index else pd.Series(dtype=bool)
    n_sec = pd.Series([symbol_sector.get(x) for x in book]).value_counts()

    def why(s, act):
        r = now.get(s)
        if s in info["no_buy"] and s not in held:
            return f"buy block: {info['no_buy'][s]}"
        if act == "Sell":
            return "not ranked (score <= 0)" if r is None else f"rank {r}: out of the top 10"
        if act == "Buy" and str(side.get(s, "")).startswith("Sell"):
            return f"buy back (sold {side[s][5:]}, not held now)"
        if act in ("Buy", "Hold", "Trim", "Add") or s in book:
            return {"Buy": "new pick", "Add": "topped up to equal weight", "Trim": "trimmed to equal weight"}.get(act, "kept")
        if r is None or r > 20:
            return ""
        if bool(stretched.get(s, False)):
            return "stretched: >15% over its 20-day average"
        if bool(soon.get(s, False)):
            return "earnings within 5 days"
        if n_sec.get(symbol_sector.get(s), 0) >= 4:
            return f"sector cap: 4 {symbol_sector.get(s)} already"
        return f"rank {r}: top 10 already filled"

    def act(s):
        a = str(nxt.get(s, "")).upper()
        if a == "BUY":
            return "Add" if s in held else "Buy"
        if a == "SELL":
            return "Sell" if s not in book else "Trim"
        return "Hold" if s in held or s in book else "—"

    syms = sorted(set(held) | set(side) | book | {s for s, r in now.items() if r <= 20},
                  key=lambda s: (now.get(s, 999), s))
    rows = []
    for s in syms:
        a = act(s)
        rows.append({"Stock": s, "Sector": symbol_sector.get(s, "—"), "Held now": held.get(s, 0.0),
                     "Last week": (side[s] if s in side else "Hold" if s in held else "—"),
                     "Rank then": then.get(s), "Rank now": now.get(s), "Next": a, "Why": why(s, a)})
    t = pd.DataFrame(rows)
    t["Held now"] = t["Held now"].map(lambda q: f"{q:g} sh" if q else "—")
    for c in ("Rank then", "Rank now"):
        t[c] = pd.array(pd.to_numeric(t[c], errors="coerce").round(), dtype="Int64")
    return t, last, inp["close"].index[-1], d


@st.fragment(run_every=30)
def render_last_decision(p):
    """Strategy tab: next run line + the one decisions table (last Friday vs next Friday at the latest close)."""
    from dashboard.page import fresh
    from dashboard.details.live_holdings import holdings_this_run
    import sleeves as sl
    p = fresh(p)
    data, err = holdings_this_run()
    if data is None:
        st.info(f"Last decision: {err} (live positions are the only holdings source)")
        return
    try:
        slot, kind = next_run(ledger=sl.load_ledger())
        if slot is not None:
            st.markdown(f"**Next run: {slot:%a %b %-d, %-I:%M %p} CT** · {kind}")
        t, last, now_day, d = decision_table(data["positions"], data["equity"])
    except Exception as e:
        st.info(f"Decisions unavailable: {str(e)[:150]}")
        return
    st.caption(f"Last week = the {last:%a %b %-d} rebalance orders and any trade since (Rank then = {last:%b %-d})." if last is not None else "No rebalance yet.")
    st.caption(f"Rank now = the {now_day:%a %b %-d} close. Next = a Friday rebalance at that close through the live plan "
               "(top 10 equal ~9.9%, max 4 per sector, no new buys >15% over the 20-day average or within 5 days of "
               "earnings, replacements to rank 20, half weights when the regime is off). Prices move until Friday.")
    tone = lambda v: GOOD if v in ("Buy", "Add") else BAD if v in ("Sell", "Trim") or str(v).startswith("Trim") or str(v).startswith("Sell") else INK
    sty = toned(toned(t, ["Next"], fn=tone), ["Last week"], fn=lambda v: tone(str(v).split(" ")[0]))
    event = table(sty, fit=True, on_select="rerun", selection_mode="single-row", key="decisions_tbl",
                  cols={"Rank then": st.column_config.NumberColumn("Rank then", width=60),
                        "Rank now": st.column_config.NumberColumn("Rank now", width=60)})
    open_symbol(t.rename(columns={"Stock": "Symbol"}), event, "decisions")


def next_plan():
    """{"asof", "days": {half: next decision}, "plan": {sym: (action, % of account, day)}, "kind"} or None, from the same
    Friday preview as the Strategy tab (display only)."""
    import sleeves as sl
    from dashboard.details.live_holdings import account_core
    try:
        data, _ = account_core()
        if data is None:
            return None
        d, o, _ = preview(data["positions"], data["equity"])
        slot, kind = next_run(ledger=sl.load_ledger())
    except Exception:
        return None
    import backtest_engine as be
    nxt = next((x for x in [d, *be.next_sessions(d, 6)] if x > d and sl.full_day(sl.HALVES[0], x)), None)
    plan = {}
    for _, r in o.iterrows():
        held, tgt = float(r["Current_Shares"] or 0) > 0, float(r["Target_Shares"] or 0)
        if not held and tgt <= 0:
            continue
        plan[r["Symbol"]] = ("Sell" if tgt <= 0 else "Keep" if held else "Buy", float(r["Target_Weight_%"] or 0), nxt)
    return {"asof": d, "days": {h: nxt for h in sl.HALVES}, "plan": plan, "kind": kind}


def next_run(now=None, ledger=None):
    """(slot datetime CT, kind text) of the next 2:30 PM CT decision. With no ledger yet the first run is a full rebalance."""
    import backtest_engine as be
    now = now or pd.Timestamp.now(tz=be.CENTRAL).to_pydatetime()
    slot = be.next_decision_slot(now)
    try:   # a missed decision not yet run (run_all.decision_gate) trades first: now if the market is open, else 9:00 AM CT
        import run_all as ra
        D, how, _ = ra.decision_gate(now, ra.load_state())
        if D is not None and D.date() != now.date() and how in ("session", "wait"):
            start = now if how == "session" else ra.next_session_start(now)
            if slot is None or start < slot:
                return start, ("full rebalance" if ledger is None else f"catch-up of the {D:%a %b %-d} decision")
    except Exception:
        pass
    if slot is None:
        return None, None
    kind = be.decision_kind(slot.date())
    if ledger is None:
        kind = "full rebalance"
    return slot, kind



def plan_frames(positions, equity):
    """(orders table, targets table, cash after) of the next Friday rebalance preview (chatbot)."""
    _, orders, _ = preview(positions, equity)
    o = orders[orders["Side"].isin(["BUY", "SELL"])].copy()
    o = o.assign(Side=o["Side"].str.capitalize())
    o = o[["Side", "Symbol", "Shares", "Price", "Est_Value", "Current_Shares", "Target_Shares"]].rename(
        columns={"Est_Value": "Value $", "Current_Shares": "Now shares", "Target_Shares": "Target shares"})
    o = o.sort_values(["Side", "Value $"], ascending=[False, False]).reset_index(drop=True)
    t = orders[orders["Target_Weight_%"].fillna(0) > 0][["Symbol", "Target_Weight_%"]].rename(
        columns={"Target_Weight_%": "Target %"}).sort_values("Target %", ascending=False).reset_index(drop=True)
    cash = float(equity) * (1 - t["Target %"].sum() / 100)
    return o, t, cash
