"""Mon/Wed spare-cash rule (live only, approved 2026-10-06): after the mid-week swaps/exits, the account's cash above 1%
of equity buys the best-ranked stocks NOT held (latest ranking, rank order, up to rank 20) at their rule weights; held
stocks get no order; earnings / pre-earnings-stop blocks; 20% x 99% cap; a rest under max($100, 1% of equity) stays cash.
Hand-worked cases with fake CSVs; no network, no orders.
Run: python tests/run_tests.py  (or python tests/test_midweek_cash_deploy.py)"""
import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import paper_trade as pt

FAIL = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)


def ranking(rows, regime=1):
    """rows: (symbol, rank, provisional weight, close)."""
    return pd.DataFrame([{"Symbol": s, "Rank": r, "Score": 80 - r, "Provisional_Weight": w, "Close": c,
                          "Regime_On": regime, "Midweek_Check": 1} for s, r, w, c in rows])


EMPTY = pd.DataFrame(columns=pt.ORDER_COLUMNS)
EQ = 100_000.0
R = ranking([("AAA", 1, 0.10, 100.0), ("BBB", 2, 0.10, 50.0), ("CCC", 3, 0.12, 20.0), ("DDD", 4, 0.08, 10.0),
             ("EEE", 11, 0.0, 25.0), ("FFF", 12, 0.0, 40.0), ("GGG", 25, 0.0, 10.0)])


def buys(orders):
    b = orders[orders["Side"] == "BUY"]
    return dict(zip(b["Symbol"], b["Est_Value"]))


# 1) held names get nothing; non-held top names in rank order at their weights; leftover -> rank 11+ at the median weight
o, info = pt.build_cash_deploy_orders(pt.build_hold_orders({"AAA": 100, "BBB": 200}, {"AAA": 100.0, "BBB": 50.0}),
                                      {"AAA": 100, "BBB": 200}, EQ, cash=30_000, ranking=R, fractional=True)
b = buys(o)
check("held AAA/BBB get no order (only HOLD rows)", "AAA" not in b and "BBB" not in b
      and set(o.loc[o["Symbol"].isin(["AAA", "BBB"]), "Side"]) == {"HOLD"}, o[["Symbol", "Side"]].values.tolist())
check("spare = cash - 1% of equity = $29,000", abs(info["spare"] - 29_000) < 0.01, info["spare"])
check("CCC (rank 3) gets its 12% weight, DDD (rank 4) its 8%",
      abs(b.get("CCC", 0) - 12_000) < 1 and abs(b.get("DDD", 0) - 8_000) < 1, b)
check("rest $9,000 -> EEE (rank 11) at most the median weight (10%) -> $9,000",
      abs(b.get("EEE", 0) - 9_000) < 1 and "FFF" not in b and "GGG" not in b, b)
check("account ends ~99% invested (total buys = spare)", abs(sum(b.values()) - 29_000) < 1, sum(b.values()))

# 2) not a single top-20 name to buy below the minimum; rank > 20 never bought
o, info = pt.build_cash_deploy_orders(EMPTY, {s: 1 for s in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]}, EQ,
                                      cash=50_000, ranking=R, fractional=True)
check("all top-20 held -> nothing bought (rank 25 is never used); cash stays", buys(o) == {}, buys(o))

# 3) small leftover stays in cash (below max($100, 1% of equity) = $1,000)
o, info = pt.build_cash_deploy_orders(EMPTY, {}, EQ, cash=1_900, ranking=R, fractional=True)
check("spare $900 < $1,000 minimum -> no buy", buys(o) == {} and abs(info["spare"] - 900) < 0.01, (buys(o), info["spare"]))

# 4) blocks: earnings / stop names are skipped with a reason, next name takes the cash
o, info = pt.build_cash_deploy_orders(EMPTY, {"AAA": 1, "BBB": 1}, EQ, cash=13_000, ranking=R, fractional=True,
                                      blocked={"CCC": "earnings Thu Oct 08, in 2 days"})
b = buys(o)
check("earnings-blocked CCC skipped, DDD bought instead", "CCC" not in b and "DDD" in b
      and any(s == "CCC" for s, _ in info["skipped"]), (b, info["skipped"]))

# 5) 20% x 99% cap, and the pro-rata top-up when every candidate got its weight and cash is left
R2 = ranking([("AAA", 1, 0.10, 100.0), ("BBB", 2, 0.30, 100.0)])
o, info = pt.build_cash_deploy_orders(EMPTY, {}, EQ, cash=60_000, ranking=R2, fractional=True)
b = buys(o)
check("BBB weight 30% capped at 19.8% of equity", b.get("BBB", 0) <= 19_800 + 0.01, b)
check("leftover tops AAA up to the 19.8% cap too (pro rata room)", abs(b.get("AAA", 0) - 19_800) < 100, b)

# 6) swaps run first: a name sold today is not bought back; the swap's buy counts as held; proceeds count as cash
sw = pd.DataFrame([{"Symbol": "AAA", "Side": "SELL", "Shares": 100, "Price": 100.0, "Est_Value": 10_000.0,
                    "Current_Shares": 100, "Target_Shares": 0, "Target_Weight_%": 0, "Target_Value": 0},
                   {"Symbol": "BBB", "Side": "BUY", "Shares": 200, "Price": 50.0, "Est_Value": 10_000.0,
                    "Current_Shares": 0, "Target_Shares": 200, "Target_Weight_%": 10, "Target_Value": 10_000}],
                  columns=pt.ORDER_COLUMNS)
o, info = pt.build_cash_deploy_orders(sw, {"AAA": 100}, EQ, cash=13_000, ranking=R, fractional=True)
b = buys(o)
check("sold-today AAA not bought back; swap-buy BBB not doubled; CCC gets the spare $12,000",
      list(o["Symbol"]).count("BBB") == 1 and list(o["Symbol"]).count("AAA") == 1 and abs(b.get("CCC", 0) - 12_000) < 1, b)

# 7) QQQ below its 200-day (Regime_On 0): invested level 49.5%, so less spare cash
o, info = pt.build_cash_deploy_orders(EMPTY, {}, EQ, cash=60_000, ranking=ranking([("AAA", 1, 0.05, 100.0)], regime=0),
                                      fractional=True)
check("regime off: keeps 50.5% of equity in cash -> spare $9,500", abs(info["spare"] - 9_500) < 0.01, info["spare"])

# 8) whole shares when not fractional, rounded down
o, _ = pt.build_cash_deploy_orders(EMPTY, {}, EQ, cash=5_000, ranking=ranking([("AAA", 1, 0.10, 333.0)]), fractional=False)
check("whole shares rounded down (4,010 / 333 -> 12)", list(o["Shares"]) == [12], list(o["Shares"]))

# --- end to end through plan_orders with fake report files ---
tmp = tempfile.mkdtemp()
picks, sig, mid = (os.path.join(tmp, n) for n in ("picks.csv", "sig.csv", "mid.csv"))
pd.DataFrame([{"As_Of": "2026-10-07", "Last_Rebalance": "2026-10-02", "Last_Decision": "2026-10-02", "Strategy": "C6",
               "Symbol": "AAA", "Strategy_Weight": 0.5, "Provisional_Weight": 0.10, "Close": 100.0}]).to_csv(picks, index=False)
rows = []
for d, chk in (("2026-10-06", 0), ("2026-10-07", 1)):
    for s, r, w, c in [("AAA", 1, 0.10, 100.0), ("BBB", 2, 0.10, 50.0), ("CCC", 3, 0.0, 20.0), ("NEG", 4, 0.0, 5.0)]:
        rows.append({"Date": d, "Symbol": s, "Close": c, "Strategy_Score": -1 if s == "NEG" else 70 - r,
                     "Strategy_Rank": r, "Strategy_Weight": 0, "Provisional_Weight": w, "Regime_On": 1, "Midweek_Check": chk})
pd.DataFrame(rows).to_csv(sig, index=False)
pd.DataFrame(columns=["As_Of", "Event", "Event_Date", "Action", "Sell", "Buy", "Message"]).to_csv(mid, index=False)

rk = pt.latest_ranking("2026-10-07", sig)
check("latest_ranking: score > 0 only, best rank first", list(rk["Symbol"]) == ["AAA", "BBB", "CCC"], list(rk["Symbol"]))
check("is_midweek_check_day: Wed yes, Tue no",
      pt.is_midweek_check_day("2026-10-07", sig) and not pt.is_midweek_check_day("2026-10-06", sig))

o, meta, _ = pt.plan_orders("auto", EQ, {"AAA": 500}, picks_csv=picks, signal_csv=sig, midweek_csv=mid,
                            fractional=True, cash=40_000)
b = buys(o)
check("plan_orders on a quiet Wed check: HOLD AAA, buy BBB and CCC (10% each, then the $19,000 rest pro rata up to the "
      "19.8% cap -> $19,500 each), source hold+cash",
      meta["source"] == "hold+cash" and set(b) == {"BBB", "CCC"} and o.loc[o.Symbol == "AAA", "Side"].tolist() == ["HOLD"]
      and abs(b["BBB"] - 19_500) < 1 and abs(b["CCC"] - 19_500) < 1, (meta["source"], b))
o, meta, _ = pt.plan_orders("auto", EQ, {"AAA": 500}, picks_csv=picks, signal_csv=sig, midweek_csv=mid, fractional=True)
check("no cash given -> unchanged hold behaviour (no buys)", meta["source"] == "hold" and buys(o) == {}, meta["source"])
pd.DataFrame([{**pd.read_csv(picks).iloc[0].to_dict(), "As_Of": "2026-10-06"}]).to_csv(picks, index=False)
o, meta, _ = pt.plan_orders("auto", EQ, {"AAA": 500}, picks_csv=picks, signal_csv=sig, midweek_csv=mid,
                            fractional=True, cash=40_000)
check("not a check day (Tue) -> no spare-cash buys", meta["source"] == "hold" and buys(o) == {}, (meta["source"], buys(o)))
o, meta, _ = pt.plan_orders("auto", EQ, {"AAA": 500}, picks_csv=picks, signal_csv=sig, midweek_csv=mid,
                            fractional=True, cash=40_000, force_deploy=True)
check("force_deploy (dry-run preview) runs it on any day", meta["source"] == "hold+cash" and buys(o), meta["source"])

# reconcile has nothing to compare on a +cash day (no false drift alarm)
rep, ok = pt.reconcile_positions("hold+cash")
check("reconcile_positions('midweek+cash'/'hold+cash') -> empty, ok", rep.empty and ok is True, (rep, ok))

# Friday rebalance untouched: provisional never gets spare-cash buys
pd.DataFrame([{**pd.read_csv(picks).iloc[0].to_dict(), "As_Of": "2026-10-02"}]).to_csv(picks, index=False)
o, meta, _ = pt.plan_orders("auto", EQ, {}, picks_csv=picks, signal_csv=sig, midweek_csv=mid, fractional=True,
                            cash=100_000, decision="2026-10-02")
check("Friday (provisional) is unchanged: no cash_deploy step", meta["source"] == "provisional" and "cash_deploy" not in meta,
      meta["source"])

print(f"\n{len(FAIL)} failed" if FAIL else "\nMIDWEEK CASH DEPLOY OK")
sys.exit(1 if FAIL else 0)
