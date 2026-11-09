"""Strategy tab, Decisions (Nov 4, 2026, Chirag): ONE table per stock - its rank at the last Friday rebalance (the saved
snapshot in Reports/decision_ranks.csv, orders in Reports/live_orders_log.csv with that As_Of) next to its rank for the
next Friday decision as a PREVIEW at the latest close (sleeves.plan through the live code path, read-only: no orders,
the ledger is never written).

Nov 8, 2026 (Chirag: "I don't see the table that showed rank at last decision and next decision"): the table no longer
depends on the full Alpaca holdings read (positions + ~2000 fill pages + portfolio history). It reads positions and
equity only (account_core), and the two rank columns are drawn from saved data even when Alpaca or the preview fails;
only the Held now / Next decision columns need the live account."""

import pandas as pd
import streamlit as st

from dashboard.page import open_symbol
from dashboard.style import BAD, GOOD, INK, table, toned

RANK_LAST, RANK_NEXT, NEXT = "Rank at last decision", "Rank at next decision (preview)", "Next decision (preview)"


def _positions(positions):
    return ({p["symbol"]: float(p.get("qty") or 0) for p in positions or [] if float(p.get("qty") or 0) > 0},
            {p["symbol"]: float(p.get("current_price") or 0) for p in positions or []})


@st.cache_data(ttl=600, show_spinner=False)
def _preview(key, positions_t, prices_t, equity):
    """(day, orders, info) of a Friday rebalance at the latest close: sleeves.plan with rebalance=True (the live run's
    plan, blocks and all - manual / earnings-stop buy blocks, stretched, earnings, sector cap)."""
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
    """(table, last day, rank-now day, preview day, preview error) - one row per stock: Stock, Sector, Held now (live
    Alpaca positions), Last week (latest action since the last Friday: Buy / Sell / Trim = partial sell, still held /
    Hold), Rank at last decision (saved snapshot), Rank at next decision (preview at the latest close), Next decision
    (preview), Why. positions=None (no live read): the two rank columns only, Held now / Next decision empty."""
    import decision_ranks
    import sleeves as sl
    from dashboard.settings import symbol_sector
    live = positions is not None
    held, _ = _positions(positions)
    inp = sl.inputs()
    last, side = last_rebalance()
    book_now = held if live else set(((sl.load_ledger() or {}).get("halves", {}).get(sl.HALVES[0], {}).get("shares") or {}))
    side = {k: v.replace("Sell", "Trim", 1) if k in book_now else v for k, v in side.items()}   # a partial sell, still held
    col = decision_ranks.RANK_COL[sl.HALVES[0]]
    then = decision_ranks.load(last) if last is not None else None
    then = {} if then is None else dict(zip(then["Symbol"], then[col]))
    now = {s: r for s, (r, _) in sl.live_ranks(inp=inp, all_stocks=True).items()}
    d, o, info, err = None, None, {"no_buy": {}, "targets": {sl.HALVES[0]: {}}}, None
    if live:
        try:
            d, o, info = preview(positions, equity)
        except Exception as e:                      # the ranks still show; the preview columns say why they are empty
            err = f"{type(e).__name__}: {str(e)[:200]}"
    nxt = {} if o is None else {r.Symbol: r.Side for r in o.itertuples()}
    book = set(info["targets"][sl.HALVES[0]])
    day = d if d is not None else inp["close"].index[-1]
    stretched = inp["stretched_live"].loc[day] if day in inp["stretched_live"].index else pd.Series(dtype=bool)
    soon = inp["soon"].loc[day] if day in inp["soon"].index else pd.Series(dtype=bool)
    n_sec = pd.Series([symbol_sector.get(x) for x in book]).value_counts()

    def why(s, act):
        r = now.get(s)
        if s in info["no_buy"] and s not in held:
            return f"buy block: {info['no_buy'][s]}"
        if o is None:                               # no preview: only the facts that need no live account
            return ""
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
        if o is None:
            return "—"
        a = str(nxt.get(s, "")).upper()
        if a == "BUY":
            return "Add" if s in held else "Buy"
        if a == "SELL":
            return "Sell" if s not in book else "Trim"
        return "Hold" if s in held or s in book else "—"

    syms = sorted(set(book_now) | set(side) | book | {s for s, r in now.items() if r <= 20},
                  key=lambda s: (now.get(s, 999), s))
    rows = []
    for s in syms:
        a = act(s)
        rows.append({"Stock": s, "Sector": symbol_sector.get(s, "—"), "Held now": held.get(s, 0.0) if live else None,
                     "Last week": (side[s] if s in side else "Hold" if s in book_now else "—"),
                     RANK_LAST: then.get(s), RANK_NEXT: now.get(s), NEXT: a, "Why": why(s, a)})
    t = pd.DataFrame(rows, columns=["Stock", "Sector", "Held now", "Last week", RANK_LAST, RANK_NEXT, NEXT, "Why"])
    t["Held now"] = t["Held now"].map(lambda q: "?" if q is None or q != q else f"{q:g} sh" if q else "—")
    for c in (RANK_LAST, RANK_NEXT):
        t[c] = pd.array(pd.to_numeric(t[c], errors="coerce").round(), dtype="Int64")
    return t, last, inp["close"].index[-1], d, err


def next_rebalance_slot(now=None):
    """The next Friday rebalance slot (tz-aware CT; Thursday when Friday is a holiday, 11:00 AM CT on early-close days)
    strictly after `now`, or None - the only decision that trades (Nov 4, 2026: no Mon-Thu trades)."""
    import backtest_engine as be
    import sleeves as sl
    now = now or pd.Timestamp.now(tz=be.CENTRAL).to_pydatetime()
    d = pd.Timestamp(now.astimezone(be.CENTRAL).date())
    for _ in range(be.DECISION_SEARCH_DAYS):
        if sl.full_day(sl.HALVES[0], d) and be.decision_slot(d) > now:
            return be.decision_slot(d)
        d += pd.Timedelta(days=1)
    return None


@st.fragment(run_every=30)
def render_last_decision(p):
    """Strategy tab: the next rebalance / next run lines + the one decisions table (rank at the last Friday decision vs
    the preview for the next Friday at the latest close)."""
    from dashboard.page import fresh
    from dashboard.details.live_holdings import account_core
    import sleeves as sl
    p = fresh(p)
    data, err = account_core()                      # positions + equity only (not the fills / history read)
    try:
        reb = next_rebalance_slot()
        slot, kind = next_run(ledger=sl.load_ledger())
        if reb is not None:
            line = f"**Next decision (rebalance): {reb:%a %b %-d, %-I:%M %p} CT**"
            if slot is not None and slot < reb:
                kind = "no rebalance Mon-Thu" if kind == "mid-week check" else kind
                line += f" · next scheduled run {slot:%a %b %-d, %-I:%M %p} CT ({kind})"
            st.markdown(line)
        t, last, now_day, d, perr = decision_table(data["positions"] if data else None, data["equity"] if data else 0.0)
    except Exception as e:
        st.info(f"Decisions unavailable: {type(e).__name__}: {str(e)[:150]}")
        return
    if data is None:
        st.info(f"Last decision: {err} The two rank columns below come from saved data; Held now and Next decision "
                "need the live account.")
    elif perr:
        st.info(f"Next decision preview unavailable ({perr}); the ranks below are still current.")
    if last is not None:
        saved = t[RANK_LAST].notna().any()
        st.caption(f"Last week = the {last:%a %b %-d} rebalance orders and any trade since. {RANK_LAST} = the snapshot "
                   f"saved at the {last:%a %b %-d} decision" + ("." if saved else " (not saved for that day: shown as empty)."))
    else:
        st.caption("No rebalance yet.")
    st.caption(f"{RANK_NEXT} = ranks at the {now_day:%a %b %-d} close; {NEXT} = a Friday rebalance at that close through "
               "the live plan (top 10 equal ~9.9%, max 4 per sector, no new buys >15% over the 20-day average or within "
               "5 days of earnings, replacements to rank 20). A preview: prices move until Friday.")
    tone = lambda v: GOOD if v in ("Buy", "Add") else BAD if v in ("Sell", "Trim") or str(v).startswith("Trim") or str(v).startswith("Sell") else INK
    sty = toned(toned(t, [NEXT], fn=tone), ["Last week"], fn=lambda v: tone(str(v).split(" ")[0]))
    event = table(sty, fit=True, on_select="rerun", selection_mode="single-row", key="decisions_tbl",
                  cols={RANK_LAST: st.column_config.NumberColumn(
                            f"Rank at last decision ({last:%b %-d})" if last is not None else RANK_LAST,
                            help="Saved snapshot of the live RS short-term rank at that decision (Reports/decision_ranks.csv)"),
                        RANK_NEXT: st.column_config.NumberColumn(
                            RANK_NEXT, help=f"Live RS short-term rank at the {now_day:%b %-d} close - what the next Friday "
                                            "decision would use if prices did not move (preview)")})
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
