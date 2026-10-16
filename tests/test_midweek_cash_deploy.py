"""Mon-Thu spare-cash rule (live only, approved 2026-10-06; leftover rule Chirag 2026-10-15): after the mid-week
replacements/exits, the account's cash above 1% of equity buys the top-10 stocks NOT held (latest ranking, rank order) at
their rule weights; what is left is split equally over the held stocks at/above their Alpaca avg entry price (none in
profit -> all held; blocked / sold-today ones skipped; a stock at the 15% x 99% = 14.85% cap, 5% for the defensive sleeve,
passes its share to the others); earnings / earnings-day-stop blocks; an order under max($100, 1% of equity) is not made; cash no
stock can take stays cash. Friday's rebalance (exactly the top-10 targets, every other stock sold) is unchanged.
Hand-worked cases with fake CSVs; no network, no orders.
Run: python tests/run_tests.py  (or python tests/test_midweek_cash_deploy.py)"""
import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import c6_rules  # noqa: E402,F401  the Oct 15 weekly rules this test checks (live: monthly low volatility)

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

HOLD2 = pt.build_hold_orders({"AAA": 100, "BBB": 200}, {"AAA": 100.0, "BBB": 50.0})

# 1) unheld top-10 names in rank order at their weights; the leftover goes to the held stocks at/above break-even (cap 14.85%)
o, info = pt.build_cash_deploy_orders(HOLD2, {"AAA": 100, "BBB": 200}, EQ, cash=30_000, ranking=R, fractional=True,
                                      entry={"AAA": 90.0, "BBB": 60.0})
b = buys(o)
check("spare = cash - 1% of equity = $29,000", abs(info["spare"] - 29_000) < 0.01, info["spare"])
check("unheld CCC (rank 3) gets its 12% weight, DDD (rank 4) its 8%; rest $9,000 -> AAA only (in profit) up to the $14,850 "
      "cap (+$4,850); BBB (below entry) nothing; the $4,150 no one can take stays cash",
      b == {"AAA": 4_850.0, "CCC": 12_000.0, "DDD": 8_000.0} and info["in_profit"] == ["AAA"]
      and [t[0] for t in info["topups"]] == ["AAA"], (b, info["in_profit"], info["topups"]))
check("held AAA becomes a BUY top-up (one row); BBB stays HOLD; EEE (rank 11) is not bought",
      o.loc[o.Symbol == "AAA", "Side"].tolist() == ["BUY"] and o.loc[o.Symbol == "BBB", "Side"].tolist() == ["HOLD"]
      and "EEE" not in b and list(o["Symbol"]).count("AAA") == 1, o.values.tolist())

# 2) fallback: no held stock in profit (or no entry prices) -> split equally over all held stocks
for ent, name in (({"AAA": 110.0, "BBB": 60.0}, "both below entry"), (None, "no entry prices")):
    o, info = pt.build_cash_deploy_orders(HOLD2, {"AAA": 100, "BBB": 200}, EQ, cash=30_000, ranking=R, fractional=True, entry=ent)
    check(f"fallback ({name}): the $9,000 rest is split over AAA and BBB ($4,500 each)",
          buys(o) == {"AAA": 4_500.0, "BBB": 4_500.0, "CCC": 12_000.0, "DDD": 8_000.0} and info["in_profit"] == [], buys(o))

o, info = pt.build_cash_deploy_orders(HOLD2, {"AAA": 100, "BBB": 200}, EQ, cash=1_500, ranking=R, entry={"AAA": 90.0})
check("spare under the minimum: no step 2, so in_profit is None (the preview prints no 'held in profit')",
      buys(o) == {} and info["in_profit"] is None, info)

# 2b) cap spill among the in-profit holdings; a loser and a stock with room under the minimum get nothing
pos = {"AAA": 140, "BBB": 250, "CCC": 500, "DDD": 100}               # $14,000 (room $850 < $1,000), $12,500, $10,000, $1,000
o, info = pt.build_cash_deploy_orders(EMPTY, pos, EQ, cash=8_000, ranking=R, fractional=True,
                                      entry={"AAA": 90.0, "BBB": 40.0, "CCC": 15.0, "DDD": 20.0})
check("$7,000 over BBB and CCC (in profit): BBB to the cap (+$2,350), its spill to CCC (+$4,650); DDD (loss) and AAA "
      "(room under the minimum) none", buys(o) == {"BBB": 2_350.0, "CCC": 4_650.0}, buys(o))
o, info = pt.build_cash_deploy_orders(EMPTY, {"LMT": 30}, EQ, cash=5_000, ranking=ranking([("ZZZ", 15, 0.0, 10.0)]),
                                      fractional=True, prices={"LMT": 100.0}, entry={"LMT": 90.0})
check("defensive-sleeve LMT tops up only to its 5% x 99% cap ($4,950: +$1,950); the rest stays cash",
      buys(o) == {"LMT": 1_950.0}, buys(o))

# 3) small leftover stays in cash (below max($100, 1% of equity) = $1,000)
o, info = pt.build_cash_deploy_orders(EMPTY, {}, EQ, cash=1_900, ranking=R, fractional=True)
check("spare $900 < $1,000 minimum -> no buy", buys(o) == {} and abs(info["spare"] - 900) < 0.01, (buys(o), info["spare"]))

# 4) sold today / blocked holdings are skipped
sell_a = pd.DataFrame([{"Symbol": "AAA", "Side": "SELL", "Shares": 100, "Price": 100.0, "Est_Value": 10_000.0,
                        "Current_Shares": 100, "Target_Shares": 0, "Target_Weight_%": 0, "Target_Value": 0}], columns=pt.ORDER_COLUMNS)
o, info = pt.build_cash_deploy_orders(sell_a, {"AAA": 100, "BBB": 200, "CCC": 500}, EQ, cash=2_000, ranking=R, fractional=True,
                                      blocked={"CCC": "earnings Thu Oct 08, in 2 days"}, entry={"AAA": 90.0, "BBB": 40.0, "CCC": 10.0})
b = buys(o)
check("sold-today AAA not bought back, earnings-blocked CCC not topped up: DDD (unheld rank 4) 8%, the last $3,000 to BBB",
      b == {"BBB": 3_000.0, "DDD": 8_000.0} and dict(info["skipped"]).get("CCC", "").startswith("held: earnings")
      and o.loc[o.Symbol == "AAA", "Side"].tolist() == ["SELL"], (b, info["skipped"]))

# 5) per-stock cap 14.85% (15% x 99%) for a new name; unheld names get only their weight (no top-up in the rest step)
R2 = ranking([("AAA", 1, 0.10, 100.0), ("BBB", 2, 0.30, 100.0)])
o, info = pt.build_cash_deploy_orders(EMPTY, {}, EQ, cash=60_000, ranking=R2, fractional=True)
b = buys(o)
check("BBB weight 30% capped at 14.85% of equity; AAA its 10%; nothing held: the rest stays cash",
      b == {"AAA": 10_000.0, "BBB": 14_850.0} and abs(info["spent"] - 24_850) < 0.01, b)

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
check("leftover skips sold-today AAA and today's replacement BBB: $12,000 split over held CCC and DDD (no entry -> all held)",
      len(bb) == 1 and abs(bb["Est_Value"].iloc[0] - 10_000) < 1 and o.loc[o.Symbol == "AAA", "Side"].tolist() == ["SELL"]
      and buys(o) == {"BBB": 10_000.0, "CCC": 6_000.0, "DDD": 6_000.0}, o.values.tolist())

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
check("plan_orders on a quiet Wed check: HOLD AAA (over the cap), BBB and CCC 10% each, the rest stays cash, source hold+cash",
      meta["source"] == "hold+cash" and b == {"BBB": 10_000.0, "CCC": 10_000.0}
      and o.loc[o.Symbol == "AAA", "Side"].tolist() == ["HOLD"], (meta["source"], b))
o, meta, _ = pt.plan_orders("auto", EQ, {"AAA": 100, "BBB": 200, "CCC": 500}, picks_csv=picks, signal_csv=sig,
                            midweek_csv=mid, fractional=True, cash=12_000)
check("plan_orders, everything held: the $11,000 spare tops up all held equally (no entry prices)",
      buys(o) == {"AAA": 3_666.0, "BBB": 3_666.5, "CCC": 3_666.6} and len(o) == 3, (meta["source"], buys(o)))
o, meta, _ = pt.plan_orders("auto", EQ, {"AAA": 100, "BBB": 200, "CCC": 500}, picks_csv=picks, signal_csv=sig,
                            midweek_csv=mid, fractional=True, cash=8_000, entry={"AAA": 90.0, "BBB": 60.0, "CCC": 10.0})
check("plan_orders with entry prices: the $7,000 goes to AAA and CCC (in profit), not BBB",
      buys(o) == {"AAA": 3_500.0, "CCC": 3_500.0} and meta["cash_deploy"]["in_profit"] == ["AAA", "CCC"], buys(o))
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
