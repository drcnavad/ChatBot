"""Strategy tab, Last decision (live from Oct 27, 2026: "RS short-term (weekly + Mon–Thu sells)", one sleeve = the account):
a summary line (stocks, value, next rebalance), the top-20 ranking at the latest decision (Reports/decision_ranks.csv,
saved by the trade run) with held stocks marked, then the next decision preview from the latest close. Weights from the
live Alpaca positions; the ledger is never written here."""

import pandas as pd
import streamlit as st

from dashboard.page import open_symbol
from dashboard.style import GOOD, INK, SCORE, table, toned

def _sleeves():
    import sleeves
    return sleeves


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
        out.append(f"**{sl.LABEL[h]}**: {len(mine[h])} stocks · \\${v:,.0f}"
                   + (f" ({100 * v / equity:.1f}% of the account)" if equity else "")
                   + (f" · next full rebalance {nxt:%a %b %-d}" if nxt is not None else "")
                   + (" · Mon-Thu: exit check (worse than rank 20)" if h == "rs_short" else ""))
    return out

def half_decision_day(half, path=None):
    """The latest day in Reports/decision_ranks.csv on which `half` decided (rs_short: every session)."""
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
    """(table, others): Rank, Stock, Sector, Score, Held (✓) for the half's top `top` ranks at `day`; others = the stocks
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
    t = pd.DataFrame({"Rank": pd.array(pd.to_numeric(r[col], errors="coerce").round(), dtype="Int64"), "Stock": r["Symbol"].to_numpy(),
                      "Sector": [symbol_sector.get(x, "—") for x in r["Symbol"]],
                      "Score": [round(float(score.get(x)), 2) if score.get(x) is not None and pd.notna(score.get(x)) else None
                                for x in r["Symbol"]],
                      "Held": ["✓" if x in mine[half] else "" for x in r["Symbol"]]})
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
        st.info("RS short-term (weekly + Mon–Thu sells) starts at its first full rebalance, Fri Oct 30, 2026 "
                "9:00 AM CT (Reports/sleeve_ledger.json is written then).")
    else:
        for line in sleeve_summary(pd.Timestamp(led["day"]), positions, led, equity):
            st.markdown(line)
        cols = st.columns(len(sl.HALVES))
        for c, half in zip(cols, sl.HALVES):
            with c:
                day = half_decision_day(half)
                if day is None:
                    st.info(f"{sl.LABEL[half]}: no saved decision ranks yet.")
                    continue
                st.markdown(f"**{sl.LABEL[half]}** · ranks at its decision of {day:%a %b %-d}")
                t, others = ranking_table(half, day, positions, led, equity=equity)
                sty = toned(t, ["Held"], fn=lambda v: GOOD if v == "✓" else INK)
                event = table(sty, fit=True, cols={"Score": SCORE}, on_select="rerun", selection_mode="single-row",
                              key=f"rank_tbl_{half}")
                if others:
                    st.caption(f"Held, not in top 20: {', '.join(others)}")
                open_symbol(t.rename(columns={"Stock": "Symbol"}), event, f"rank_{half}")
    st.markdown("##### Next decision")
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
    Action: Keep / Buy / Sell / Blocked. Display only."""
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
    days = {h: nxt for h in sl.HALVES}
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


@st.cache_data(ttl=600, show_spinner=False)
def _dry_plan(key, positions_t, equity):
    """The trade step's own plan (paper_trade.plan_orders, read-only, no broker calls, no orders) at the latest close."""
    import paper_trade as pt
    orders, meta, targets = pt.plan_orders("provisional", equity, positions=dict(positions_t), fractional=True)
    return orders, targets

def plan_frames(positions, equity):
    """(orders table, targets table, cash after) for the next decision from the current dry-run plan."""
    pos = tuple(sorted((p["symbol"], float(p.get("qty") or 0)) for p in positions or []))
    key = str(_sleeves().inputs()["close"].index[-1].date())
    orders, targets = _dry_plan(key, pos, float(equity))
    o = orders[orders["Side"].isin(["buy", "sell", "BUY", "SELL"])].copy() if len(orders) else orders
    o = o.assign(Side=o["Side"].str.capitalize())
    o = o[["Side", "Symbol", "Shares", "Price", "Est_Value", "Current_Shares", "Target_Shares"]].rename(
        columns={"Est_Value": "Value $", "Current_Shares": "Now shares", "Target_Shares": "Target shares"})
    o = o.sort_values(["Side", "Value $"], ascending=[False, False]).reset_index(drop=True)
    t = targets.assign(**{"Target %": targets["Weight"] * 100})[["Symbol", "Target %"]].sort_values(
        "Target %", ascending=False).reset_index(drop=True)
    cash = float(equity) * (1 - float(targets["Weight"].sum()))
    return o, t, cash

def render_preview(positions, led, equity):
    """Next run (date/time CT, kind), the rule schedule, and the planned trades from the trade step's dry-run plan."""
    try:
        slot, kind = next_run(ledger=led)
    except Exception as e:
        slot, kind = None, f"unavailable ({str(e)[:80]})"
    if slot is not None:
        st.markdown(f"**Next run: {slot:%a %b %-d, %-I:%M %p} CT** · {kind}")
    try:
        o, t, cash = plan_frames(positions, equity)
    except Exception as e:
        st.info(f"Planned trades unavailable: {str(e)[:150]}")
        return
    sl = _sleeves()
    day = sl.inputs()["close"].index[-1]
    sells, buys = o[o.Side == "Sell"]["Value $"].sum(), o[o.Side == "Buy"]["Value $"].sum()
    st.markdown(f"Sell \\${sells:,.0f} · Buy \\${buys:,.0f} · Cash after \\${cash:,.0f}")
    from dashboard.settings import symbol_sector
    held = {p_["symbol"] for p_ in positions or [] if float(p_.get("qty") or 0) > 0}
    tgt = dict(zip(t["Symbol"], t["Target %"]))
    side = dict(zip(o["Symbol"], o["Side"]))
    def earn_soon(syms):
        """{SYMBOL: 'Oct 28'} for names with earnings within the 5-day rule (the engine's own check: not newly bought)."""
        try:
            import backtest_engine as be, forward_test as ft
            inp = sl.inputs()
            a = be.earnings_days_ahead([day], list(syms), inp["earnings"], ft.EARN_DAYS).iloc[0].dropna()
            return {k: f"{day + pd.offsets.BDay(int(v)):%b %-d}" for k, v in a.items()}
        except Exception:
            return {}
    def plan(x):
        if x not in tgt:
            if x in earn:
                return f"{'Sell' if side.get(x) == 'Sell' else 'Skip'}: earn {earn[x]}"
            if side.get(x) == "Sell":
                return "Sell: not a pick"
            return "Skip: not Fri pick"   # entered the top 20 after the Friday rebalance; Mon-Thu only refills exits
        a_ = "Keep" if x in held else "Buy"
        if x in held and side.get(x) in ("Buy", "Sell"):
            a_ = "Add" if side[x] == "Buy" else "Trim"
        return a_
    for half in sl.HALVES:
        tab, _ = ranking_table(half, day, positions, led, path="__live__", equity=equity)
        tab = tab.drop(columns=["Held"])
        earn = earn_soon(tab["Stock"])
        mv = {p_["symbol"]: float(p_.get("market_value") or 0) for p_ in positions or []}
        try:
            from dashboard.data import index_quotes
            qq = index_quotes(tuple(tab["Stock"])) or {}
        except Exception:
            qq = {}
        tab["Weight now"] = [round(100 * mv[x] / equity, 1) if x in mv and equity else None for x in tab["Stock"]]
        tab["Today %"] = [round(float(qq[x][1]), 2) if x in qq and qq[x][1] == qq[x][1] else None for x in tab["Stock"]]
        tab.insert(1, "Plan", [plan(x) for x in tab["Stock"]])
        from dashboard.style import BAD
        tone_ = lambda v: (GOOD if v > 0 else BAD if v < 0 else INK) if isinstance(v, (int, float)) and v == v else INK
        sty = toned(tab, ["Plan"], fn=lambda v: GOOD if str(v).startswith(("Buy", "Add")) else BAD if str(v).startswith("Sell") else INK)
        sty = toned(sty, ["Today %"], fn=tone_)
        table(sty, fit=True, key=f"preview_{half}",
              cols={"Score": SCORE, "Rank": st.column_config.NumberColumn("#", width=30),
                    "Plan": st.column_config.TextColumn(width=96), "Stock": st.column_config.TextColumn(width=52),
                    "Sector": st.column_config.TextColumn(width=110),
                    "Weight now": st.column_config.NumberColumn("Weight now", format="%.1f%%", width=62),
                    "Today %": st.column_config.NumberColumn("Today %", format="%+.2f", width=58)})
        extra = [x for x in tgt if x not in set(tab["Stock"])]
        if extra:
            st.caption("Also targeted (outside the top 20 / defensive stocks): "
                       + ", ".join(f"{x} ({symbol_sector.get(x, '—')}) {plan(x)}" for x in extra))
        sold = o[(o.Side == "Sell") & ~o.Symbol.isin(set(tgt))]
        st.markdown("Would sell: " + (", ".join(f"{r.Symbol} ({symbol_sector.get(r.Symbol, '—')}) \\${r[4]:,.0f}"
                                                  for r in sold.itertuples(index=False)) if len(sold) else "nothing"))
