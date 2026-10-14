"""Mon/Wed spare-cash rule (live only, approved 2026-10-06; leftover rule 2026-10-07): after the mid-week replacements/exits, the
account's cash above 1% of equity buys the top-10 stocks NOT held (latest ranking, rank order) at their rule weights; what
is left buys unheld ranks 11, 12, ... (score > 0) at most 10% of equity each; held stocks are never topped up; other
held stocks get no order; earnings / earnings-day-stop blocks; an order under max($100, 1% of equity) is not made; cash no
stock can take stays cash. Friday's rebalance (exactly the top-10 targets, every other stock sold) is unchanged.
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

# 1) unheld top-10 names in rank order at their weights; the leftover buys unheld rank 11+ names (<= 10% each)
o, info = pt.build_cash_deploy_orders(pt.build_hold_orders({"AAA": 100, "BBB": 200}, {"AAA": 100.0, "BBB": 50.0}),
                                      {"AAA": 100, "BBB": 200}, EQ, cash=30_000, ranking=R, fractional=True)
b = buys(o)
check("spare = cash - 1% of equity = $29,000", abs(info["spare"] - 29_000) < 0.01, info["spare"])
check("CCC (rank 3) gets its 12% weight, DDD (rank 4) its 8%",
      abs(b.get("CCC", 0) - 12_000) < 1 and abs(b.get("DDD", 0) - 8_000) < 1, b)
check("rest $9,000 buys unheld EEE (rank 11, under the 10% cap); held AAA is not topped up",
      b == {"CCC": 12_000.0, "DDD": 8_000.0, "EEE": 9_000.0} and info["topups"] == [], (b, info["topups"]))
a_row = o[o["Symbol"] == "AAA"]
check("held AAA and BBB stay HOLD (no top-up)",
      o.loc[o.Symbol.isin(["AAA", "BBB"]), "Side"].tolist() == ["HOLD", "HOLD"], o.values.tolist())
check("account ends ~99% invested (total buys = spare)", abs(sum(b.values()) - 29_000) < 1, sum(b.values()))

# 2) everything up to rank 12 held: the cash buys the next unheld name at 10%
o, info = pt.build_cash_deploy_orders(EMPTY, {s: 1 for s in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]}, EQ,
                                      cash=50_000, ranking=R, fractional=True)
b = buys(o)
check("all top 10 and ranks 11-12 held: GGG (rank 25) gets 10% ($10,000), the rest stays cash", b == {"GGG": 10_000.0}, b)

# 3) small leftover stays in cash (below max($100, 1% of equity) = $1,000)
o, info = pt.build_cash_deploy_orders(EMPTY, {}, EQ, cash=1_900, ranking=R, fractional=True)
check("spare $900 < $1,000 minimum -> no buy", buys(o) == {} and abs(info["spare"] - 900) < 0.01, (buys(o), info["spare"]))

# 4) blocks: earnings / stop names are skipped with a reason, next name takes the cash
o, info = pt.build_cash_deploy_orders(EMPTY, {"AAA": 1, "BBB": 1}, EQ, cash=13_000, ranking=R, fractional=True,
                                      blocked={"CCC": "earnings Thu Oct 08, in 2 days"})
b = buys(o)
check("earnings-blocked CCC skipped (not bought, not topped up), DDD bought instead", "CCC" not in b and "DDD" in b
      and [s for s, _ in info["skipped"]] == ["CCC"], (b, info["skipped"]))

# 5) top 10 held: the leftover buys ranks 11, 12, ... at 10% each; held names are never topped up
full = {"AAA": 198, "BBB": 396, "CCC": 990}                      # each $19,800 = the cap
o, info = pt.build_cash_deploy_orders(pt.build_hold_orders(full, {"AAA": 100.0, "BBB": 50.0, "CCC": 20.0}), {**full, "DDD": 1},
                                      EQ, cash=20_000, ranking=R, fractional=True)
check("top 10 held: EEE (rank 11) $10,000, FFF (rank 12) the last $9,000", buys(o) == {"EEE": 10_000.0, "FFF": 9_000.0}, buys(o))
o, info = pt.build_cash_deploy_orders(EMPTY, {"AAA": 1, "BBB": 1, "CCC": 1, "DDD": 1}, EQ, cash=30_000, ranking=R,
                                      fractional=True, blocked={"AAA": "earnings Fri Oct 09, in 2 days"})
b = buys(o)
check("held names never topped up: EEE, FFF at 10% each, GGG the last $9,000", b == {"EEE": 10_000.0, "FFF": 10_000.0, "GGG": 9_000.0}, b)
near = {"AAA": 193, "BBB": 1, "CCC": 1, "DDD": 1}                # AAA $19,300: room $500 < the $1,000 minimum
o, info = pt.build_cash_deploy_orders(EMPTY, near, EQ, cash=5_000, ranking=R, fractional=True)
check("all top 10 held: the $4,000 goes to EEE (rank 11)", buys(o) == {"EEE": 4_000.0}, buys(o))

# 5b) per-stock cap 19.8% for a new name; an unheld rank-1-3 name gets its weight, then the top-up on top (one row)
R2 = ranking([("AAA", 1, 0.10, 100.0), ("BBB", 2, 0.30, 100.0)])
o, info = pt.build_cash_deploy_orders(EMPTY, {}, EQ, cash=60_000, ranking=R2, fractional=True)
b = buys(o)
check("BBB weight 30% capped at 19.8% of equity", abs(b.get("BBB", 0) - 19_800) < 0.01, b)
check("no rank 11+ in the ranking: AAA gets its 10% weight, no top-up, the rest stays cash",
      b == {"AAA": 10_000.0, "BBB": 19_800.0} and abs(info["spent"] - 29_800) < 0.01, b)

# 6) replacements run first: a name sold today is not bought back or topped up; the replacement buy counts as held; proceeds count as cash
sw = pd.DataFrame([{"Symbol": "AAA", "Side": "SELL", "Shares": 100, "Price": 100.0, "Est_Value": 10_000.0,
                    "Current_Shares": 100, "Target_Shares": 0, "Target_Weight_%": 0, "Target_Value": 0},
                   {"Symbol": "BBB", "Side": "BUY", "Shares": 200, "Price": 50.0, "Est_Value": 10_000.0,
                    "Current_Shares": 0, "Target_Shares": 200, "Target_Weight_%": 10, "Target_Value": 10_000}],
                  columns=pt.ORDER_COLUMNS)
o, info = pt.build_cash_deploy_orders(sw, {"AAA": 100}, EQ, cash=13_000, ranking=R, fractional=True)
b = buys(o)
check("sold-today AAA not bought back; replacement buy BBB not doubled; CCC gets the spare $12,000",
      list(o["Symbol"]).count("BBB") == 1 and list(o["Symbol"]).count("AAA") == 1 and abs(b.get("CCC", 0) - 12_000) < 1, b)
o, info = pt.build_cash_deploy_orders(sw, {"AAA": 100, "CCC": 1, "DDD": 1}, EQ, cash=13_000, ranking=R, fractional=True)
bb = o[o["Symbol"] == "BBB"]
check("leftover skips sold-today AAA, leaves the replacement BBB buy at $10,000, then EEE $10,000 and FFF the last $2,000",
      len(bb) == 1 and abs(bb["Est_Value"].iloc[0] - 10_000) < 1 and o.loc[o.Symbol == "AAA", "Side"].tolist() == ["SELL"]
      and buys(o) == {"BBB": 10_000.0, "EEE": 10_000.0, "FFF": 2_000.0}, o.values.tolist())

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
check("plan_orders on a quiet Wed check: HOLD AAA, buy BBB and CCC (10% each), no top-up, source hold+cash",
      meta["source"] == "hold+cash" and b == {"BBB": 10_000.0, "CCC": 10_000.0}
      and o.loc[o.Symbol == "AAA", "Side"].tolist() == ["HOLD"], (meta["source"], b))
o, meta, _ = pt.plan_orders("auto", EQ, {"AAA": 100, "BBB": 200, "CCC": 500}, picks_csv=picks, signal_csv=sig,
                            midweek_csv=mid, fractional=True, cash=12_000)
check("plan_orders, everything held and no rank 11+: nothing bought, the cash stays",
      buys(o) == {} and len(o) == 3, (meta["source"], buys(o)))
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

# Friday rebalance: exactly the top-10 targets are kept, every other stock is sold completely (e.g. MSFT, U, BE)
top10 = [f"T{i}" for i in range(1, 11)]
targets = pd.DataFrame({"Symbol": top10, "Weight": [0.099] * 10, "Price": [100.0] * 10})
held = {**{s: 99 for s in top10[:8]}, "MSFT": 18.34, "U": 25.53, "BE": 12, "LITE": 3.82, "TWLO": 19}
prices = {**{s: 100.0 for s in top10}, "MSFT": 530.0, "U": 45.0, "BE": 290.0, "LITE": 1100.0, "TWLO": 273.0}
fri = pt.build_orders(targets, EQ, held, prices, fractional=True, statuses={s: "hold" if s in held else "add" for s in top10})
after = {r.Symbol: r.Target_Shares for r in fri.itertuples() if r.Side in ("BUY", "SELL", "HOLD")}
for s, q in held.items():
    after.setdefault(s, q)
kept = sorted(s for s, q in after.items() if q and q > 0)
sold = fri[(fri["Side"] == "SELL") & (fri["Target_Shares"] == 0)]
check("Friday: the 5 stocks outside the top 10 (MSFT, U, BE, LITE, TWLO) are sold completely (exact shares)",
      sorted(sold["Symbol"]) == sorted(set(held) - set(top10)) and len(sold) == 5
      and all(r.Shares == held[r.Symbol] for r in sold.itertuples()), fri[["Symbol", "Side", "Shares"]].values.tolist())
check("Friday: the account ends with exactly the 10 targets (2 new ones bought)", kept == sorted(top10)
      and set(fri.loc[fri.Side == "BUY", "Symbol"]) >= {"T9", "T10"}, kept)

print(f"\n{len(FAIL)} failed" if FAIL else "\nMIDWEEK CASH DEPLOY OK")
sys.exit(1 if FAIL else 0)
