"""Strategy tab: the ONE portfolio view (Oct 16, 2026; replaces the old Last decision, Two halves, Targets and Latest
signals views) - what was actually bought, sold and held at the last decision, straight from the Alpaca account, and
every stock's rank in both strategies.

Decision day = the day of the last trade run (Reports/sleeve_ledger.json "day"). Action from the account's fills since that
day (GET /account/activities, the Trading Account tab's read): Bought = not held before, held now; Sold = held before,
none now; Held = held before and now (a trim or top-up is noted). Sleeve = the half that owns it in the ledger. Ranks at
the decision from Reports/decision_ranks.csv (decision_ranks.py), today's from the latest signal data. Weight = the
position's market value / account equity now."""
import pandas as pd
import streamlit as st

from dashboard.page import open_symbol
from dashboard.settings import SIGNAL_CSV
from dashboard.style import BAD, GOOD, INK, PCT_COL, toned

SLEEVE = {"lowvol": "Low vol", "sector": "Sector rotation"}
ORDER = {"Bought": 0, "Held": 1, "—": 1, "Sold": 2, "Not": 3}
COLUMNS = ["Symbol", "Action", "Sleeve", "Rank Vol at decision", "Rank Vol today", "Rank Sector at decision",
           "Rank Sector today", "Current weight %"]

def decision_table(day, positions, fills, equity, ledger=None, ranks_then=None, ranks_now=None, all_stocks=()):
    """One row per stock bought, sold or held at decision day `day` (COLUMNS); positions / fills = Alpaca dicts.
    fills=None (account not read): actions unknown ("—"). all_stocks: every other symbol is added as "Not held"."""
    since = pd.Timestamp(day).tz_localize("America/Chicago")
    net = {}
    for f in fills or []:
        if pd.Timestamp(f["transaction_time"]) >= since:
            q = float(f.get("qty") or 0) * (1 if f.get("side") == "buy" else -1)
            net[f["symbol"]] = net.get(f["symbol"], 0.0) + q
    now = {p["symbol"]: float(p.get("qty") or 0) for p in positions or []}
    value = {p["symbol"]: float(p.get("market_value") or 0) for p in positions or []}
    import sleeves
    owner = {s: SLEEVE[h] for s, h in sleeves.owners(ledger).items() if h in SLEEVE}
    rank = lambda r, s, c: (r.set_index("Symbol")[c].get(s, pd.NA) if r is not None else pd.NA)
    rows = []
    for s in sorted(set(net) | set(now) | set(all_stocks)):
        after, traded = now.get(s, 0.0), net.get(s, 0.0)
        before = after - traded
        if after <= 1e-6 and before <= 1e-6:
            if s not in all_stocks:
                continue                                  # bought and sold again since the decision: not in the portfolio
            action = "Not held"
        elif fills is None:
            action = "—"
        elif before <= 1e-6:
            action = "Bought"
        elif after <= 1e-6:
            action = "Sold"
        else:
            action = "Held" + (" (trimmed)" if traded < -1e-6 else " (added)" if traded > 1e-6 else "")
        rows.append({"Symbol": s, "Action": action, "Sleeve": owner.get(s, "—"),
                     "Rank Vol at decision": rank(ranks_then, s, "Rank_Vol"), "Rank Vol today": rank(ranks_now, s, "Rank_Vol"),
                     "Rank Sector at decision": rank(ranks_then, s, "Rank_Sector"),
                     "Rank Sector today": rank(ranks_now, s, "Rank_Sector"),
                     "Current weight %": round(100 * value.get(s, 0.0) / equity, 2) if equity else None})
    t = pd.DataFrame(rows, columns=COLUMNS)
    for c in COLUMNS[3:7]:
        t[c] = pd.to_numeric(t[c], errors="coerce").round().astype("Int64")
    key = t["Action"].str.split().str[0].map(ORDER).fillna(3)
    return (t.assign(_k=key, _r=t["Rank Vol today"].astype("Float64").fillna(1e9))
            .sort_values(["_k", "Sleeve", "_r", "Symbol"]).drop(columns=["_k", "_r"]).reset_index(drop=True))

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

@st.cache_data(show_spinner=False)
def _ranks_now(mtime):
    import decision_ranks
    return decision_ranks.ranks_on()[1]

def render_last_decision(p):
    """Strategy tab, the one portfolio view: per-half summary, then the last decision's Bought / Sold / Held (or every
    stock with both strategies' ranks). Click a row to open the stock."""
    import os

    import decision_ranks
    import sleeves
    from dashboard.details.live_holdings import holdings_this_run
    led = sleeves.load_ledger()
    if led is None:
        st.info("No trade run yet (Reports/sleeve_ledger.json is written by the first one).")
        return
    day = pd.Timestamp(led["day"])
    data, err = holdings_this_run()
    if data is not None:
        positions, fills, equity = data["positions"], data["fills"], data["equity"]
    else:                                   # Alpaca not readable: nothing is assumed held
        st.info(f"Last decision · {day:%a %b %-d}: {err} (live positions are the only holdings source)")
        return
    for line in sleeve_summary(day, positions, led, equity):
        st.markdown(line)
    show = st.radio("Show", ["Portfolio (last decision)", "All stocks (both strategies' ranks)"], horizontal=True,
                    key="decision_filter", label_visibility="collapsed")
    then = decision_ranks.load(day)
    now = _ranks_now(os.path.getmtime(SIGNAL_CSV))
    every = list(now["Symbol"]) if show.startswith("All") else ()
    t = decision_table(day, positions, fills, equity, led, then, now, all_stocks=every)
    n = t["Action"].str.split().str[0].value_counts()
    st.caption(f"Decision of {day:%a %b %-d} (orders at 2:30 PM CT, any rest at 9 AM CT the next trading day), from the "
               f"account's fills: Bought {n.get('Bought', 0)} · Sold {n.get('Sold', 0)} · Held {n.get('Held', 0)}. "
               "Rank Vol = low-volatility rank (calmest first, above the 200-day average only); Rank Sector = "
               "sector-rotation rank (score among the 3 sectors with the best 63-day ETF return); blank = not ranked. "
               + ("" if then is not None else "Ranks at the decision were not saved for this day. ")
               + "Current weight = the position's value / equity now. Click a row to open the stock on the Home tab.")
    sty = toned(t, ["Action"], fn=lambda v: GOOD if v == "Bought" else BAD if v == "Sold" else INK)
    event = st.dataframe(sty, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                         key=f"last_decision_tbl_{show[:3]}", column_config={"Current weight %": PCT_COL})
    open_symbol(t, event, "last_decision")
