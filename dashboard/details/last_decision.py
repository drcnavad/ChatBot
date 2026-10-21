"""Strategy tab, Last decision: a line per half (stocks, value, next rebalance), then each half's top-20 ranking at its
own latest rebalance (Reports/decision_ranks.csv, saved by the trade run), side by side, held stocks marked. Sleeve =
the half that owns a stock in Reports/sleeve_ledger.json; weights from the live Alpaca positions."""
import os

import pandas as pd
import streamlit as st

from dashboard.page import open_symbol
from dashboard.settings import SIGNAL_CSV
from dashboard.style import GOOD, INK, PCT_COL, toned

SLEEVE = {"lowvol": "Low vol", "sector": "Sector rotation"}
def sleeve_summary(day, positions, ledger, equity):
    """One line per half: its live positions (by tag) + its cash, share of the account, next rebalance; a queued buy is
    'Pending order', never a holding."""
    import holdings
    import sleeves
    value = {p["symbol"]: float(p.get("market_value") or 0) for p in positions or []}
    pend = holdings.pending_symbols()
    cal = sleeves.calendar(pd.Timestamp(day) + pd.Timedelta(days=1), 45)
    out = []
    for h, label in SLEEVE.items():
        L = ledger["halves"][h]
        syms = [s for s, o in sleeves.owners(ledger).items() if o == h and s in value]
        v = sum(value.get(s, 0.0) for s in syms) + float(L.get("cash", 0.0))
        waiting = sorted(s for s, o in pend.items() if (o or sleeves.owners(ledger).get(s)) == h and s not in value)
        nxt = next((d for d, hs in cal if h in hs), None)
        out.append(f"**{sleeves.LABEL[h]}**: {len(syms)} stocks · ${v:,.0f}"
                   + (f" ({100 * v / equity:.1f}% of the account)" if equity else "")
                   + (f" · next rebalance {nxt:%a %b %-d}" if nxt is not None else "")
                   + (f" · Pending order: {', '.join(waiting)}" if waiting else ""))
    return out

def half_decision_day(half, path=None):
    """The latest day in Reports/decision_ranks.csv on which `half` rebalanced (sleeves.lowvol_day / sector_day)."""
    import decision_ranks
    import sleeves
    path = path or decision_ranks.RANKS_CSV
    if not os.path.exists(path):
        return None
    days = sorted(pd.read_csv(path, usecols=["Date"], dtype=str)["Date"].unique(), reverse=True)
    test = sleeves.lowvol_day if half == "lowvol" else sleeves.sector_day
    return next((pd.Timestamp(d) for d in days if test(d)), None)

def ranking_table(half, day, positions, ledger, top=20, path=None):
    """(table, others): Rank, Stock, Sector, Score, Held by, Weight in half % for the half's top `top` ranks at its
    decision `day`; others = the stocks the half holds outside that top. Score = Strategy_Score (low vol) / the sector-rotation score (sector)."""
    import decision_ranks
    import sleeves
    from dashboard.settings import symbol_sector
    col = "Rank_Vol" if half == "lowvol" else "Rank_Sector"
    r = decision_ranks.ranks_on(day)[1] if path == "__live__" else decision_ranks.load(day, path=path)
    r = r.dropna(subset=[col]).sort_values(col) if r is not None else pd.DataFrame(columns=["Symbol", col])
    owner = sleeves.owners(ledger)
    value = {p["symbol"]: float(p.get("market_value") or 0) for p in positions or []}
    mine = [x for x, h in owner.items() if h == half and x in value]
    half_v = sum(value[x] for x in mine) + float(ledger["halves"][half].get("cash", 0.0))
    if half == "lowvol":
        sig = pd.read_csv(SIGNAL_CSV, usecols=["Date", "Symbol", "Strategy_Score"], parse_dates=["Date"])
        score = sig[sig["Date"] == day].drop_duplicates("Symbol").set_index("Symbol")["Strategy_Score"]
    else:
        score = sleeves.sector_inputs()["scores"]["sector_rot"].loc[day]
    r = r[r[col] <= top]
    others = sorted(set(mine) - set(r["Symbol"]))                 # held by the half, not in its top `top` that day
    held = lambda x: ("✓ " + SLEEVE[owner[x]]) if x in value and owner.get(x) in SLEEVE else ("✓ (untagged)" if x in value else "")
    t = pd.DataFrame({"Rank": pd.array(pd.to_numeric(r[col], errors="coerce").round(), dtype="Int64"), "Stock": r["Symbol"].to_numpy(),
                         "Sector": [symbol_sector.get(x, "—") for x in r["Symbol"]],
                         "Score": [round(float(score.get(x)), 1) if pd.notna(score.get(x)) else None for x in r["Symbol"]],
                         "Held by": [held(x) for x in r["Symbol"]],
                         "Weight in half %": [round(100 * value[x] / half_v, 1) if owner.get(x) == half and x in value
                                              and half_v else None for x in r["Symbol"]]})
    return t, others

def render_last_decision(p):
    """Strategy tab: per-half summary, then each half's ranking at its own latest rebalance, side by side (held stocks
    marked). Click a row to open the stock."""
    import sleeves
    from dashboard.details.live_holdings import holdings_this_run
    led = sleeves.load_ledger()
    if led is None:
        st.info("No trade run yet (Reports/sleeve_ledger.json is written by the first one).")
        return
    data, err = holdings_this_run()
    if data is None:                        # Alpaca not readable: nothing is assumed held
        st.info(f"Last decision: {err} (live positions are the only holdings source)")
        return
    positions, equity = data["positions"], data["equity"]
    for line in sleeve_summary(pd.Timestamp(led["day"]), positions, led, equity):
        st.markdown(line)
    cols = st.columns(2)
    for c, half in zip(cols, ("sector", "lowvol")):
        with c:
            day = half_decision_day(half)
            if day is None:
                st.info(f"{sleeves.LABEL[half]}: no saved rebalance ranks yet.")
                continue
            st.markdown(f"**{sleeves.LABEL[half]}** · ranks at its rebalance of {day:%a %b %-d}")
            t, others = ranking_table(half, day, positions, led)
            sty = toned(t, ["Held by"], fn=lambda v: GOOD if str(v).startswith("✓") else INK)
            event = st.dataframe(sty, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                                 key=f"rank_tbl_{half}", height=35 * (len(t) + 1) + 3,
                                 column_config={"Weight in half %": PCT_COL,
                                                "Score": st.column_config.NumberColumn(format="%.1f")})
            if others:
                st.caption(f"Held, not in top 20: {', '.join(others)}")
            t = t.rename(columns={"Stock": "Symbol"})
            open_symbol(t, event, f"rank_{half}")
    st.caption("Ranks saved by the trade run (Reports/decision_ranks.csv). Sector rotation: rank of the sector score among "
               "the stocks of the 3 sectors with the best 63-day ETF return; the bot then skips stocks the low-vol half owns "
               "(no overlap), blocked stocks and a 5th stock of one sector, so a held stock can rank below a skipped one. "
               "Low vol: calmest first, above the 200-day only; Score = its strategy score. Top 20 ranks (Sector can have fewer: "
               "only stocks of its 3 sectors with a positive score are ranked); Weight in half = live value / the half's value (stocks + its cash).")
    st.markdown("##### Next rebalance preview · based on yesterday's close")
    render_preview(positions, led)

# ---------------------------------------------------------------- next rebalance preview (read-only, no orders)
@st.cache_data(ttl=600, show_spinner=False)
def _preview_targets(day, lv_held, sec_held):
    """{half: set of target symbols} at the close `day`, by the bot's own selection: the sector half =
    sleeves.sector_weights (the low-vol half's names excluded, as in sleeves.plan); the low-vol half = the pipeline's
    Provisional_Weight in strategy_picks.csv (as paper_trade.plan_sleeve_orders), stock list only."""
    import paper_trade as pt
    import sector_mapping as sm
    import sleeves
    w = sleeves.sector_weights(day, exclude=set(lv_held), held=dict(sec_held))
    picks = pd.read_csv(pt.PICKS_CSV)
    lv = picks[pd.to_numeric(picks["Provisional_Weight"], errors="coerce").fillna(0) > 0]["Symbol"].astype(str).str.upper()
    return {"sector": set(w[w > 0].index), "lowvol": set(lv) & set(sm.tradable_symbols)}

def render_preview(positions, led):
    """Next rebalance preview: each half's top 20 at the latest close with Keep / Buy / Blocked, and what it would sell."""
    import decision_ranks
    import paper_trade as pt
    import sleeves
    day, _ = decision_ranks.ranks_on()
    owner = sleeves.owners(led)
    value = {p["symbol"]: float(p.get("market_value") or 0) for p in positions or []}
    mine = {h: [x for x, o in owner.items() if o == h and x in value] for h in SLEEVE}
    vs = sum(value[x] for x in mine["sector"]) + float(led["halves"]["sector"].get("cash", 0.0))
    sec_held = tuple(sorted((x, value[x] / vs) for x in mine["sector"])) if vs else ()
    try:
        targets = _preview_targets(day, tuple(sorted(mine["lowvol"])), sec_held)
    except Exception as e:
        st.info(f"Next rebalance preview unavailable: {str(e)[:150]}")
        return
    blocked = set(pt.manual_list("buy"))
    cal = sleeves.calendar(pd.Timestamp.now().normalize() + pd.Timedelta(days=1), 45)
    cols = st.columns(2)
    for c, half in zip(cols, ("sector", "lowvol")):
        with c:
            nxt = next((d for d, hs in cal if half in hs), None)
            st.markdown(f"**{sleeves.LABEL[half]}** · next rebalance {nxt:%a %b %-d}" if nxt is not None
                        else f"**{sleeves.LABEL[half]}**")
            t, _ = ranking_table(half, day, positions, led, path="__live__")
            tg = targets[half]
            other = "lowvol" if half == "sector" else "sector"
            plan = lambda s: ("Keep" if s in mine[half] else f"Take over from {SLEEVE[other]}" if s in mine[other]
                              else "Blocked" if s in blocked else "Buy") if s in tg else ""
            t.insert(1, "Plan", [plan(s) for s in t["Stock"]])
            sty = toned(t, ["Plan"], fn=lambda v: GOOD if v == "Buy" else INK)
            st.dataframe(sty, hide_index=True, width="stretch", height=35 * (len(t) + 1) + 3, key=f"preview_{half}",
                         column_config={"Weight in half %": PCT_COL,
                                        "Score": st.column_config.NumberColumn(format="%.1f")})
            buys_out = sorted(tg - set(t["Stock"]) - set(mine[half]) - blocked)
            sells = sorted(set(mine[half]) - tg)
            st.caption(f"Would sell: {', '.join(sells) or 'nothing'}"
                       + (f" · would also buy (ranked below 20): {', '.join(buys_out)}" if buys_out else ""))
    st.caption(f"Ranks and targets at the {day:%a %b %-d} close, by the bot's own selection code (read-only; nothing is "
               "sent). Keep = target and held; Buy = target, not held; Take over = the low-vol half takes a stock the other "
               "half holds (no trade); Blocked = a target on a buy block / do-not-buy list (not bought). The real rebalance uses the data of its own day, so the list can still change.")
