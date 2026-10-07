"""Mon/Wed spare-cash rule (live only, approved 2026-10-06; leftover rule 2026-10-07): after the mid-week replacements/exits, the
account's cash above 1% of equity buys the top-10 stocks NOT held (latest ranking, rank order) at their rule weights; what
is left tops up ranks 1-3 (held or not), rank 1 to the 19.8% cap first, then 2, then 3; ranks 11-20 are never bought; other
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

# 1) unheld top-10 names in rank order at their weights; the leftover tops up rank 1 (held) - never rank 11+
o, info = pt.build_cash_deploy_orders(pt.build_hold_orders({"AAA": 100, "BBB": 200}, {"AAA": 100.0, "BBB": 50.0}),
                                      {"AAA": 100, "BBB": 200}, EQ, cash=30_000, ranking=R, fractional=True)
b = buys(o)
check("spare = cash - 1% of equity = $29,000", abs(info["spare"] - 29_000) < 0.01, info["spare"])
check("CCC (rank 3) gets its 12% weight, DDD (rank 4) its 8%",
      abs(b.get("CCC", 0) - 12_000) < 1 and abs(b.get("DDD", 0) - 8_000) < 1, b)
check("rest $9,000 tops up held AAA (rank 1, $10,000 -> $19,000, under the $19,800 cap); EEE / FFF (ranks 11-12) get nothing",
      abs(b.get("AAA", 0) - 9_000) < 1 and "EEE" not in b and "FFF" not in b and "GGG" not in b
      and info["topups"] == [("AAA", 1, 90.0, 9_000.0)], (b, info["topups"]))
a_row = o[o["Symbol"] == "AAA"]
check("the top-up turns AAA's HOLD row into one BUY row (100 -> 190 shares); BBB (rank 2) stays HOLD",
      len(a_row) == 1 and a_row["Side"].iloc[0] == "BUY" and a_row["Current_Shares"].iloc[0] == 100
      and a_row["Target_Shares"].iloc[0] == 190 and o.loc[o.Symbol == "BBB", "Side"].tolist() == ["HOLD"], o.values.tolist())
check("account ends ~99% invested (total buys = spare)", abs(sum(b.values()) - 29_000) < 1, sum(b.values()))

# 2) every top-10 name held: the cash tops up rank 1 to the cap, then rank 2, then rank 3; never ranks 11+
o, info = pt.build_cash_deploy_orders(EMPTY, {s: 1 for s in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]}, EQ,
                                      cash=50_000, ranking=R, fractional=True)
b = buys(o)
check("all top 10 held: AAA to $19,800 (+$19,700), BBB (+$19,750), CCC gets the last $9,550; EEE/FFF/GGG nothing",
      abs(b.get("AAA", 0) - 19_700) < 1 and abs(b.get("BBB", 0) - 19_750) < 1 and abs(b.get("CCC", 0) - 9_550) < 1
      and set(b) == {"AAA", "BBB", "CCC"}, b)

# 3) small leftover stays in cash (below max($100, 1% of equity) = $1,000)
o, info = pt.build_cash_deploy_orders(EMPTY, {}, EQ, cash=1_900, ranking=R, fractional=True)
check("spare $900 < $1,000 minimum -> no buy", buys(o) == {} and abs(info["spare"] - 900) < 0.01, (buys(o), info["spare"]))

# 4) blocks: earnings / stop names are skipped with a reason, next name takes the cash
o, info = pt.build_cash_deploy_orders(EMPTY, {"AAA": 1, "BBB": 1}, EQ, cash=13_000, ranking=R, fractional=True,
                                      blocked={"CCC": "earnings Thu Oct 08, in 2 days"})
b = buys(o)
check("earnings-blocked CCC skipped (not bought, not topped up), DDD bought instead", "CCC" not in b and "DDD" in b
      and [s for s, _ in info["skipped"]] == ["CCC"], (b, info["skipped"]))

# 5) ranks 1-3 at the cap or blocked: the leftover stays in cash (no rank 4+ top-up, no rank 11+ buy)
full = {"AAA": 198, "BBB": 396, "CCC": 990}                      # each $19,800 = the cap
o, info = pt.build_cash_deploy_orders(pt.build_hold_orders(full, {"AAA": 100.0, "BBB": 50.0, "CCC": 20.0}), {**full, "DDD": 1},
                                      EQ, cash=20_000, ranking=R, fractional=True)
check("ranks 1-3 already at 19.8% and DDD held: nothing bought, the $19,000 stays in cash", buys(o) == {}, buys(o))
o, info = pt.build_cash_deploy_orders(EMPTY, {"AAA": 1, "BBB": 1, "CCC": 1, "DDD": 1}, EQ, cash=30_000, ranking=R,
                                      fractional=True, blocked={"AAA": "earnings Fri Oct 09, in 2 days"})
b = buys(o)
check("rank 1 blocked by earnings: rank 2 filled to the cap first, then rank 3; AAA untouched",
      "AAA" not in b and abs(b.get("BBB", 0) - 19_750) < 1 and abs(b.get("CCC", 0) - 9_250) < 1, b)
near = {"AAA": 193, "BBB": 1, "CCC": 1, "DDD": 1}                # AAA $19,300: room $500 < the $1,000 minimum
o, info = pt.build_cash_deploy_orders(EMPTY, near, EQ, cash=5_000, ranking=R, fractional=True)
check("a top-up under the minimum order is not made: AAA skipped (room $500), BBB takes the $4,000",
      "AAA" not in buys(o) and abs(buys(o).get("BBB", 0) - 4_000) < 1, buys(o))

# 5b) per-stock cap 19.8% for a new name; an unheld rank-1-3 name gets its weight, then the top-up on top (one row)
R2 = ranking([("AAA", 1, 0.10, 100.0), ("BBB", 2, 0.30, 100.0)])
o, info = pt.build_cash_deploy_orders(EMPTY, {}, EQ, cash=60_000, ranking=R2, fractional=True)
b = buys(o)
check("BBB weight 30% capped at 19.8% of equity", abs(b.get("BBB", 0) - 19_800) < 0.01, b)
check("leftover tops AAA (rank 1) up to the 19.8% cap: one AAA row, the rest ($19,400) stays cash",
      abs(b.get("AAA", 0) - 19_800) < 0.01 and list(o["Symbol"]).count("AAA") == 1 and abs(info["spent"] - 39_600) < 0.01, b)

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
check("leftover skips sold-today AAA (rank 1), grows the replacement's BBB buy (rank 2: $10,000 -> $19,800) in its own row, "
      "then rank 3 CCC gets the last $2,200",
      len(bb) == 1 and abs(bb["Est_Value"].iloc[0] - 19_800) < 1 and bb["Target_Shares"].iloc[0] == 396
      and list(o["Symbol"]).count("AAA") == 1 and o.loc[o.Symbol == "AAA", "Side"].tolist() == ["SELL"]
      and abs(buys(o).get("CCC", 0) - 2_200) < 1, o.values.tolist())

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
check("plan_orders on a quiet Wed check: HOLD AAA (50%, over the cap), buy BBB and CCC (10% each), the $19,000 rest "
      "tops up rank 2 BBB to $19,800 then rank 3 CCC (+$9,200), source hold+cash",
      meta["source"] == "hold+cash" and set(b) == {"BBB", "CCC"} and o.loc[o.Symbol == "AAA", "Side"].tolist() == ["HOLD"]
      and abs(b["BBB"] - 19_800) < 1 and abs(b["CCC"] - 19_200) < 1, (meta["source"], b))
o, meta, _ = pt.plan_orders("auto", EQ, {"AAA": 100, "BBB": 200, "CCC": 500}, picks_csv=picks, signal_csv=sig,
                            midweek_csv=mid, fractional=True, cash=12_000)
check("plan_orders, everything held: the spare $11,000 tops up held AAA (rank 1) to the cap (+$9,800), then BBB "
      "(rank 2) +$1,200; source hold+cash",
      meta["source"] == "hold+cash" and buys(o) == {"AAA": 9_800.0, "BBB": 1_200.0} and len(o) == 3, (meta["source"], buys(o)))
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
