"""Tax view (tax_lots.py + app.render_tax_view), hand-worked cases. Pure functions with fake activities, then the app
with a FAKE Alpaca account (GET only, no network, no keys):
  - FIFO lots (HIFO / LIFO comparison), partial and fractional fills, REG / TAF / CAT fees in proceeds / basis
  - wash sales: replacement bought after / before the loss sale, the rest of the same purchase, 31-day edge, partial share
    matching (oldest first, each share once), disallowed loss into the replacement basis, holding period carried over,
    a cascade, bot rebuys flagged
  - splits (forward / reverse, by ratio text or share count), transfers in with / without basis, transfers out
  - short / long-term boundary incl. a Feb 29 purchase; realized by year; unrealized by lot
  - dividends (qualified estimate, 61-day test), sweep / margin interest
  - Schedule D netting, $3,000 ($1,500 MFS) limit, carryforward, 2026 brackets, LTCG stacking, NIIT, state
  - harvest list exclusions (recent buy / bot plan), lots turning long-term within 30 / 60 days, Form 8949 rows
  - data checks: unmatched sell, missing basis, position mismatch, needs-review activity
  - fresh start (TAX_START = Oct 2, 2026): older activity, prior years and wash matching against older trades ignored;
    shares held from before the start left out (sold first under every lot method) and listed; paging stops at the start
  (sections 1-14 test the engine on the full history: build(..., start=None))
Run: python tests/run_tests.py  (or python tests/test_tax_lots.py)"""
import math
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import pandas as pd

import tax_lots as tl

FAIL = []
TODAY = date(2026, 10, 4)


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)


def close(a, b, tol=0.005):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


_n = [0]


def F(sym, side, qty, price, day, hh=15, order=None, kind="fill"):
    _n[0] += 1
    return {"activity_type": "FILL", "id": f"f{_n[0]:05d}", "symbol": sym, "side": side, "qty": str(qty), "price": str(price),
            "transaction_time": f"{day}T{hh:02d}:00:00Z", "order_id": order or f"o{_n[0]}", "type": kind}


def NTA(t, day, **kw):
    _n[0] += 1
    return {"activity_type": t, "id": f"n{_n[0]:05d}", "date": day, "status": "executed", **kw}


def run(acts, positions=None, method="FIFO", bots=None, start=None):
    acts = sorted(acts, key=lambda a: a.get("transaction_time") or a["date"] + "T00")
    return tl.build(acts, positions, method, bots or {}, today=TODAY, start=start)


def lots(r, sym="AAA"):
    l = r["lots"]
    return l[l["Symbol"] == sym].reset_index(drop=True)


# ---------------------------------------------------------------- 1. FIFO / HIFO / LIFO
A = [F("AAA", "buy", 10, 100, "2025-01-02"), F("AAA", "buy", 10, 120, "2025-03-03"), F("AAA", "sell", 15, 130, "2025-06-02")]
r = run(A)
s = r["sales"]
check("FIFO: 10 @ $100 then 5 @ $120 sold at $130 -> gains $300 + $50, short-term",
      len(s) == 2 and close(s["Gain"][0], 300) and close(s["Gain"][1], 50) and set(s["Term"]) == {"Short"}, s.to_dict("list"))
check("FIFO: 5 shares @ $120 left", len(lots(r)) == 1 and close(lots(r)["Shares"][0], 5) and close(lots(r)["Basis / share"][0], 120))
by = tl.realized_by_year(r["sales"])
check("realized by year: 2025 short-term $350, long-term $0", close(by.loc[by.Year == 2025, "Short-term gain"].iloc[0], 350)
      and close(by.loc[by.Year == 2025, "Long-term gain"].iloc[0], 0), by.to_dict("list"))
for m, want, left in (("HIFO", 250, 100), ("LIFO", 250, 100)):
    rm = run(A, method=m)
    check(f"{m}: the $120 lot first -> $100 + $150 = $250, 5 @ $100 left", close(rm["sales"]["Gain"].sum(), want)
          and close(lots(rm)["Basis / share"][0], left))

# ---------------------------------------------------------------- 2. wash sale, replacement bought after
W = [F("AAA", "buy", 10, 50, "2025-01-02"), F("AAA", "sell", 10, 40, "2025-02-03"), F("AAA", "buy", 10, 45, "2025-02-20", order="rep1")]
r = run(W, bots={"rep1": "live-fri-20250220-AAA"})
s, l, w = r["sales"], lots(r), r["washes"]
check("wash after: $100 loss fully disallowed (adjustment $100, gain $0)", close(s["Wash adj"][0], 100) and close(s["Gain"][0], 0), s.to_dict("list"))
check("wash after: replacement basis $45 + $10 = $55 / share", close(l["Basis / share"][0], 55) and close(l["Wash adj"][0], 100))
check("wash after: holding period carried: Feb 20 - 32 days = Jan 19, 2025", l["Holding from"][0] == date(2025, 1, 19), l["Holding from"][0])
check("wash after: one wash row, bought after the sale, bot rebuy flagged",
      len(w) == 1 and w["Before / after"][0] == "bought after the sale" and w["Replacement order"][0].startswith("bot rebuy")
      and close(w["Disallowed loss"][0], 100), w.to_dict("list"))
f = tl.form_8949(s, 2025)
check("Form 8949: '10 sh AAA', 01/02/2025 -> 02/03/2025, proceeds 400, basis 500, code W 100, gain 0, box A",
      f.iloc[0].to_dict() == {"Description": "10 sh AAA", "Date acquired": "01/02/2025", "Date sold": "02/03/2025", "Proceeds": 400.0,
                              "Cost basis": 500.0, "Adjustment code": "W", "Adjustment amount": 100.0, "Gain or loss": 0.0,
                              "Term": "Short", "Box": "A"}, f.iloc[0].to_dict())
r2 = run(W + [F("AAA", "sell", 10, 60, "2025-03-10")])
s2 = r2["sales"]
check("the replacement sold later at $60: gain $600 - $550 = $50; total = economic result (-$100 + $150)",
      close(s2["Gain"].iloc[-1], 50) and close(s2["Gain"].sum(), 50))
check("other order (not live-) is labelled as such", run(W)["washes"]["Replacement order"][0].startswith("other order"))

# ---------------------------------------------------------------- 3. partial wash (fewer replacement shares)
r = run([F("AAA", "buy", 10, 50, "2025-01-02"), F("AAA", "sell", 10, 40, "2025-02-03"), F("AAA", "buy", 4, 42, "2025-02-10")])
s, l = r["sales"], lots(r)
check("partial wash: 4 of 10 shares replaced -> $40 disallowed, $60 loss allowed", close(s["Wash adj"][0], 40) and close(s["Gain"][0], -60))
check("partial wash: replacement 4 sh at $42 + $10 = $52, holding from Jan 9", close(l["Basis / share"][0], 52) and close(l["Shares"][0], 4)
      and l["Holding from"][0] == date(2025, 1, 9))

# ---------------------------------------------------------------- 4. replacement bought BEFORE the loss sale
r = run([F("AAA", "buy", 10, 50, "2025-01-02"), F("AAA", "buy", 10, 45, "2025-03-20"), F("AAA", "sell", 10, 40, "2025-03-25")])
s, l, w = r["sales"], lots(r), r["washes"]
check("wash before: Jan lot sold (FIFO) at a $100 loss, the Mar 20 purchase (5 days before) is the replacement",
      close(s["Wash adj"][0], 100) and close(l["Basis / share"][0], 55) and w["Before / after"][0] == "bought before the sale")
check("wash before: holding from Mar 20 - 82 days = Dec 28, 2024", l["Holding from"][0] == date(2024, 12, 28), l["Holding from"][0])

# ---------------------------------------------------------------- 5. 30-day edges
base = [F("AAA", "buy", 10, 50, "2025-01-02"), F("AAA", "sell", 10, 40, "2025-02-03")]
check("rebuy on day 31 (Mar 6): no wash sale", run(base + [F("AAA", "buy", 10, 45, "2025-03-06")])["washes"].empty)
check("rebuy on day 30 (Mar 5): wash sale", len(run(base + [F("AAA", "buy", 10, 45, "2025-03-05")])["washes"]) == 1)
check("other stock bought: no wash sale", run(base + [F("BBB", "buy", 10, 45, "2025-02-10")])["washes"].empty)

# ---------------------------------------------------------------- 6. the rest of the same purchase is replacement stock
r = run([F("AAA", "buy", 10, 50, "2025-03-03"), F("AAA", "sell", 4, 40, "2025-03-10")])
l = lots(r).sort_values("Basis / share").reset_index(drop=True)
check("same purchase: 4 of the 6 remaining shares absorb the $40 loss (basis $60, holding from Feb 24), 2 stay at $50",
      close(r["sales"]["Wash adj"][0], 40) and len(l) == 2 and close(l["Shares"][0], 2) and close(l["Basis / share"][0], 50)
      and close(l["Shares"][1], 4) and close(l["Basis / share"][1], 60) and l["Holding from"][1] == date(2025, 2, 24), l.to_dict("list"))

# ---------------------------------------------------------------- 7. cascade + each share used once + oldest first
r = run([F("AAA", "buy", 10, 50, "2025-01-02"), F("AAA", "sell", 10, 40, "2025-02-03"), F("AAA", "buy", 10, 45, "2025-02-10"),
         F("AAA", "sell", 10, 44, "2025-02-20")])
check("cascade: the replacement sold at $44 has basis $55 -> $110 loss allowed; total -$110 = economic result",
      close(r["sales"]["Gain"].iloc[-1], -110) and close(r["sales"]["Gain"].sum(), -110) and len(r["washes"]) == 1)
r = run([F("AAA", "buy", 10, 50, "2025-01-02"), F("AAA", "buy", 3, 46, "2025-01-25"), F("AAA", "sell", 10, 40, "2025-02-03"),
         F("AAA", "buy", 4, 44, "2025-02-12"), F("AAA", "buy", 5, 43, "2025-02-14")])
w = r["washes"]
l = lots(r)
check("partial share matching, oldest first: 3 shares bought before + 4 + 3 of the 5 Feb 14 shares replace all 10 -> $100 "
      "disallowed; 2 Feb 14 shares keep their $43 basis", list(w["Shares"]) == [3, 4, 3] and close(r["sales"]["Wash adj"][0], 100)
      and close(r["sales"]["Gain"][0], 0) and close(l.loc[(l["Bought"] == date(2025, 2, 14)) & (l["Wash adj"] == 0), "Shares"].sum(), 2)
      and close(l.loc[(l["Bought"] == date(2025, 2, 14)) & (l["Wash adj"] > 0), "Basis / share"].iloc[0], 53), w.to_dict("list"))
r = run([F("AAA", "buy", 10, 50, "2025-01-02"), F("AAA", "sell", 5, 40, "2025-02-03"), F("AAA", "sell", 5, 40, "2025-02-04"),
         F("AAA", "buy", 5, 41, "2025-02-10")])
check("each replacement share used once: two loss sales, 5 replacement shares -> only the first sale is washed",
      [round(v, 2) for v in r["sales"]["Wash adj"]] == [50.0, 0.0] and len(r["washes"]) == 1,
      r["sales"][["Shares", "Wash adj"]].to_dict("list"))

# ---------------------------------------------------------------- 8. splits, transfers
r = run([F("AAA", "buy", 10, 100, "2025-01-02"), NTA("SSP", "2025-06-02", symbol="AAA", qty="10", description="2 for 1 forward split"),
         F("AAA", "sell", 20, 60, "2025-07-01")])
check("2-for-1 split: 20 sh at $50 basis, sold at $60 -> gain $200, holding period from Jan 2", close(r["sales"]["Gain"].sum(), 200)
      and r["sales"]["Acquired"][0] == date(2025, 1, 2) and r["issues"].empty, r["issues"].to_dict("list"))
r = run([F("AAA", "buy", 10, 100, "2025-01-02"), NTA("SSP", "2025-06-02", symbol="AAA", qty="20")])
check("split from the share count only (+20 on 10 -> 3:1): 30 sh at $33.33", close(lots(r)["Shares"][0], 30) and close(lots(r)["Basis / share"][0], 33.333))
r = run([F("AAA", "buy", 8, 10, "2025-01-02"), NTA("SSP", "2025-06-02", symbol="AAA", description="1 for 4 reverse split")])
check("1-for-4 reverse split: 2 sh at $40", close(lots(r)["Shares"][0], 2) and close(lots(r)["Basis / share"][0], 40))
r = run([NTA("ACATS", "2025-02-01", symbol="TTT", qty="10", cost_basis="500", acquired_date="2024-01-05"),
         F("TTT", "sell", 10, 80, "2025-03-01")])
check("transfer in with basis + purchase date: sold at $80 -> $300 long-term gain, box D", close(r["sales"]["Gain"][0], 300)
      and r["sales"]["Term"][0] == "Long" and tl.form_8949(r["sales"])["Box"][0] == "D" and r["issues"].empty)
r = run([NTA("ACATS", "2025-02-01", symbol="TTT", qty="10")])
check("transfer in without basis -> Missing basis check", "Missing basis" in set(r["issues"]["Check"]))
r = run([F("AAA", "buy", 10, 10, "2025-01-02"), NTA("ACATS", "2025-02-01", symbol="AAA", qty="-4")])
check("transfer out: 6 shares left, no sale", close(lots(r)["Shares"][0], 6) and r["sales"].empty)

# ---------------------------------------------------------------- 9. short / long-term boundary
check("bought Mar 14, 2025: sold Mar 14, 2026 = short-term, Mar 15, 2026 = long-term",
      not tl.is_long_term(date(2025, 3, 14), date(2026, 3, 14)) and tl.is_long_term(date(2025, 3, 14), date(2026, 3, 15)))
check("bought Feb 29, 2024: Feb 28, 2025 short, Mar 1, 2025 long; long_term_on = Mar 1, 2025",
      not tl.is_long_term(date(2024, 2, 29), date(2025, 2, 28)) and tl.is_long_term(date(2024, 2, 29), date(2025, 3, 1))
      and tl.long_term_on(date(2024, 2, 29)) == date(2025, 3, 1))
r = run([F("AAA", "buy", 1, 10, "2025-03-14"), F("AAA", "buy", 1, 10, "2025-03-17"), F("AAA", "sell", 2, 20, "2026-03-16")])
check("one sale over the boundary: Mar 14 lot long-term, Mar 17 lot short-term", list(r["sales"]["Term"]) == ["Long", "Short"])

# ---------------------------------------------------------------- 10. fees, fractional and partial fills
r = run([F("AAA", "buy", 10, 100, "2025-01-02"), F("AAA", "sell", 10, 110, "2025-01-10"),
         NTA("FEE", "2025-01-11", activity_sub_type="REG", net_amount="-0.03", description="REG fee for proceed of $1100 on 2025-01-10"),
         NTA("FEE", "2025-01-11", activity_sub_type="TAF", net_amount="-0.02", description="TAF fee for proceed of 10 shares on 2025-01-10"),
         NTA("FEE", "2025-01-03", activity_sub_type="CAT", net_amount="-0.01", description="CAT fee for proceed of 1 trades on 2025-01-02")])
s = r["sales"]
check("fees: proceeds $1,100 - $0.05 REG/TAF, basis $1,000 + $0.01 CAT -> gain $99.94", close(s["Proceeds"][0], 1099.95, 1e-6)
      and close(s["Basis"][0], 1000.01, 1e-6) and close(s["Gain"][0], 99.94, 1e-6), s.to_dict("list"))
check("fees by year listed (3 rows, $0.06)", close(r["fees"]["Amount"].sum(), 0.06, 1e-9) and len(r["fees"]) == 3)
r = run([F("AAA", "buy", 0.5, 100, "2025-01-02", kind="partial_fill"), F("AAA", "buy", 1.25, 120, "2025-01-02", hh=16),
         F("AAA", "sell", 1.0, 130, "2025-01-10")])
check("fractional / partial fills: 0.5 @ $100 + 0.5 @ $120 sold at $130 -> $20 gain; 0.75 @ $120 left",
      close(r["sales"]["Gain"].sum(), 20) and close(lots(r)["Shares"][0], 0.75) and close(lots(r)["Basis / share"][0], 120))

# ---------------------------------------------------------------- 11. data checks
r = run([F("AAA", "sell", 5, 10, "2025-01-02")])
check("sell without a buy -> Unmatched sell, no basis", "Unmatched sell" in set(r["issues"]["Check"]) and r["sales"]["Basis"].isna().all())
r = run([F("AAA", "buy", 5, "", "2025-01-02")])
check("buy without a price -> Missing basis", "Missing basis" in set(r["issues"]["Check"]))
r = run([F("AAA", "buy", 5, 10, "2025-01-02")], positions=[{"symbol": "AAA", "qty": "7", "current_price": "12"}])
check("lots 5 vs Alpaca 7 -> Position mismatch", "Position mismatch" in set(r["issues"]["Check"]))
r = run([F("AAA", "buy", 5, 10, "2025-01-02")], positions=[{"symbol": "AAA", "qty": "5", "current_price": "12"}])
check("lots = positions: no issue; unrealized $10, long-term on Oct 4, 2026 (bought Jan 2, 2025)",
      r["issues"].empty and close(tl.ytd(r)["unrealized_lt"], 10) and close(tl.ytd(r)["unrealized_st"], 0))
r = run([F("AAA", "buy", 5, 10, "2025-01-02"), NTA("MA", "2025-05-01", symbol="AAA", description="merger")])
check("merger -> Needs review", "Needs review" in set(r["issues"]["Check"]))
check("no negative lots in any case above", not any(c == "Negative lot" for c in r["issues"]["Check"]))

# ---------------------------------------------------------------- 12. dividends and interest
r = run([F("AAA", "buy", 10, 10, "2025-01-02"),
         NTA("DIV", "2025-03-15", symbol="AAA", net_amount="5", activity_sub_type="CDIV", description="Cash DIV @ 0.5, Pos QTY: 10, Rec Date: 2025-03-01"),
         F("BBB", "buy", 10, 10, "2025-02-20"), F("BBB", "sell", 10, 11, "2025-03-10"),
         NTA("DIV", "2025-03-20", symbol="BBB", net_amount="2", activity_sub_type="CDIV", description="Cash DIV @ 0.2, Pos QTY: 10, Rec Date: 2025-03-01"),
         NTA("INT", "2025-09-30", activity_sub_type="SWP", net_amount="0", qty="2.89", symbol="SWEEPFDIC", description="Sweep"),
         NTA("INT", "2025-10-02", activity_sub_type="MGN", net_amount="-6.28", description="Monthly Margin Interest Charge")])
inc = r["income"].set_index("Symbol")
check("dividend on shares held 118 days in the window -> qualified $5", close(inc.loc["AAA", "Qualified (est.)"], 5))
check("dividend on shares held 18 days -> ordinary (qualified $0)", close(inc.loc["BBB", "Qualified (est.)"], 0))
y = tl.ytd(r, 2025)
check("2025: dividends $7 (qualified $5), sweep interest $2.89, margin interest $6.28 paid",
      close(y["dividends"], 7) and close(y["qualified"], 5) and close(y["interest"], 2.89) and close(y["margin_interest"], 6.28), y)

# ---------------------------------------------------------------- 13. netting, limit, carryforward, brackets, NIIT
check("net ST -$10,000: $3,000 deducted, $7,000 short-term carryforward", tl.net_capital(-10000, 0) == (-10000, 0, 3000, 7000, 0))
check("ST +$5,000, LT -$8,000 -> net long-term loss $3,000, all deducted", tl.net_capital(5000, -8000) == (0.0, -3000, 3000, 0.0, 0.0))
check("ST -$2,000, LT -$4,000 -> $3,000 deducted (ST first), $3,000 long-term carryforward",
      tl.net_capital(-2000, -4000) == (-2000, -4000, 3000, 0.0, 3000))
check("married filing separately: $1,500 limit", tl.net_capital(-10000, 0, filing="mfs")[2] == 1500)
check("carryover input: ST +$1,000 with $4,000 carryover -> $3,000 deducted", tl.net_capital(1000, 0, carry_st=4000)[2] == 3000)
check("bracket tax: single $50,000 -> $5,752", close(tl.bracket_tax(50000, tl.ORDINARY_2026["single"]), 5752))
e = tl.estimate_tax(10000, 0, filing="single", other_agi=100000, state_rate=0.0495)
check("single, AGI $100k, ST +$10k: federal $2,200 (22%), state $495, no NIIT", close(e["federal"], 2200) and close(e["state"], 495) and e["niit"] == 0, e)
check("single, AGI $100k, LT +$10k: federal $1,500 (15%)", close(tl.estimate_tax(0, 10000, other_agi=100000, state_rate=0)["federal"], 1500))
e = tl.estimate_tax(-10000, 0, other_agi=100000, state_rate=0.0495)
check("ST -$10k: $3,000 offset saves $660 federal, $148.50 state, $7,000 carried", close(e["federal"], -660) and close(e["state"], -148.5)
      and close(e["carry_short"], 7000), e)
e = tl.estimate_tax(10000, 0, other_agi=250000, state_rate=0)
check("AGI $250k, ST +$10k: 32% bracket $3,200 + NIIT 3.8% x $10,000 = $380", close(e["federal"], 3200) and close(e["niit"], 380), e)
e = tl.estimate_tax(0, 20000, other_agi=56100, state_rate=0)
check("LTCG stacking: taxable $40,000 + LT $20,000 -> 0% on $9,450 + 15% on $10,550 = $1,582.50", close(e["federal"], 1582.5), e)
check("married filing jointly, AGI $200k, ST +$10k -> $2,200 (22%)", close(tl.estimate_tax(10000, 0, filing="mfj", other_agi=200000, state_rate=0)["federal"], 2200))

# ---------------------------------------------------------------- 14. harvest list + turning long-term
acts = [F("AAA", "buy", 10, 50, "2026-08-01"), F("BBB", "buy", 10, 50, "2026-09-25"), F("CCC", "buy", 10, 50, "2026-07-01"),
        F("DDD", "buy", 10, 50, "2025-10-20"), F("EEE", "buy", 10, 50, "2025-11-20")]
pos = [{"symbol": s, "qty": "10", "current_price": px} for s, px in (("AAA", "40"), ("BBB", "45"), ("CCC", "48"), ("DDD", "60"), ("EEE", "45"))]
r = run(acts, positions=pos)
ok, ex = tl.harvest(r, planned_buys={"CCC"})
check("harvest: AAA and EEE ok; BBB (bought Sep 25, within 30 days) and CCC (in the bot's plan) excluded; DDD (gain) not listed",
      set(ok["Symbol"]) == {"AAA", "EEE"} and set(ex["Symbol"]) == {"BBB", "CCC"} and "DDD" not in set(ok["Symbol"]) | set(ex["Symbol"]),
      (ok.to_dict("list"), ex.to_dict("list")))
check("harvest reasons in plain words", "within 30 days" in ex.set_index("Symbol").loc["BBB", "Why"]
      and "bot plans" in ex.set_index("Symbol").loc["CCC", "Why"])
t = tl.turning_long_term(r, 60).set_index("Symbol")
check("DDD (bought Oct 20, 2025) turns long-term Oct 21, 2026: in 17 days, within 30 days, gain -> wait",
      t.loc["DDD", "Days to long-term"] == 17 and t.loc["DDD", "Window"] == "within 30 days" and "wait" in t.loc["DDD", "Hint"])
check("EEE (Nov 20, 2025): 48 days, within 60 days, loss -> sell before keeps it short-term",
      t.loc["EEE", "Days to long-term"] == 48 and t.loc["EEE", "Window"] == "within 60 days" and "short-term" in t.loc["EEE", "Hint"]
      and ok.set_index("Symbol").loc["EEE", "Next long-term date"] == date(2026, 11, 21))

# ---------------------------------------------------------------- 15. fresh start at TAX_START (Oct 2, 2026)
check("TAX_START is Fri Oct 2, 2026 and build() uses it by default", tl.TAX_START == date(2026, 10, 2)
      and tl.build.__defaults__[-1] == tl.TAX_START)
OLD = [F("EEE", "buy", 10, 10, "2025-03-03"), F("EEE", "sell", 10, 15, "2025-04-01"),           # a 2025 gain: ignored
       F("AAA", "buy", 10, 50, "2026-09-01"), F("AAA", "sell", 10, 40, "2026-09-20"),           # loss 18 days before the start
       F("BBB", "buy", 5, 20, "2026-08-01"), F("CCC", "buy", 4, 30, "2026-09-10"),
       NTA("DIV", "2026-09-15", symbol="BBB", net_amount="1", description="Cash DIV, Rec Date: 2026-09-10"),
       NTA("INT", "2026-09-30", activity_sub_type="SWP", net_amount="0.5")]
NEW = [F("AAA", "buy", 10, 45, "2026-10-02"),                     # would replace the Sep 20 loss under the full history
       F("CCC", "sell", 4, 35, "2026-10-02"),                     # sells shares bought before the start
       F("BBB", "buy", 5, 22, "2026-10-02"), F("BBB", "sell", 3, 24, "2026-10-02", hh=18),
       F("DDD", "buy", 10, 10, "2026-10-02"), F("DDD", "sell", 4, 12, "2026-10-02", hh=18)]
FPOS = [{"symbol": "AAA", "qty": "10", "current_price": "46", "cost_basis": "450"},
        {"symbol": "BBB", "qty": "7", "current_price": "25", "cost_basis": "150"},
        {"symbol": "DDD", "qty": "6", "current_price": "11", "cost_basis": "60"}]
for m in tl.METHODS:
    r = run(OLD + NEW, positions=FPOS, method=m, start=tl.TAX_START)
    s, l, pre, ps = r["sales"], r["lots"], r["pre_open"], r["pre_sales"]
    check(f"fresh start ({m}): the 8 older activities are dropped, no data issue (lots + left-out shares = positions)",
          r["dropped"] == 8 and r["issues"].empty, (r["dropped"], r["issues"].to_dict("list")))
    check(f"fresh start ({m}): only DDD counts as realized: 4 sh bought $10, sold $12 -> +$8 short-term, 2026 only",
          len(s) == 1 and s["Symbol"][0] == "DDD" and close(s["Gain"][0], 8) and list(tl.realized_by_year(s)["Year"]) == [2026]
          and close(tl.ytd(r)["realized_st"], 8), s.to_dict("list"))
    check(f"fresh start ({m}): shares from before the start are sold first: CCC 4 sh ($140) and BBB 3 sh ($72) left out of the gains",
          sorted(zip(ps["Symbol"], ps["Shares"])) == [("BBB", 3), ("CCC", 4)] and close(ps["Proceeds"].sum(), 212), ps.to_dict("list"))
    check(f"fresh start ({m}): BBB 2 sh held from before the start left out (Alpaca avg cost 2 x $150/7 = $42.86); "
          "the Oct 2 BBB lot keeps 5 sh at $22", list(pre["Symbol"]) == ["BBB"] and close(pre["Shares"][0], 2)
          and close(pre["Alpaca avg cost"][0], 42.857) and close(l.loc[l.Symbol == "BBB", "Shares"].sum(), 5)
          and close(l.loc[l.Symbol == "BBB", "Basis / share"].iloc[0], 22), (pre.to_dict("list"), l.to_dict("list")))
r = run(OLD + NEW, positions=FPOS, start=tl.TAX_START)
a = lots(r)
check("fresh start: no wash sale against the Sep 20 loss: AAA Oct 2 lot keeps $45 basis, holding from Oct 2, 2026",
      r["washes"].empty and close(a["Basis / share"][0], 45) and a["Holding from"][0] == date(2026, 10, 2) and close(a["Wash adj"][0], 0))
check("fresh start: older dividend and interest ignored; unrealized = AAA $10 + BBB $15 + DDD $6 = $31",
      r["income"].empty and close(tl.ytd(r)["unrealized_st"], 31) and close(tl.ytd(r)["dividends"], 0))
check("fresh start: Form 8949 has the one DDD row, no 2025 rows", len(tl.form_8949(r["sales"])) == 1
      and not tl.form_8949(r["sales"])["Date sold"].str.endswith("2025").any())
full = run(OLD + NEW, positions=FPOS)
check("full history (start=None) for comparison: Sep 20 loss washed into the Oct 2 AAA lot ($55), 2025 gain +$50 present",
      close(lots(full)["Basis / share"][0], 55) and close(tl.realized_by_year(full["sales"]).set_index("Year").loc[2025, "Total gain"], 50))
r = run(OLD + NEW, positions=None, start=tl.TAX_START)
check("fresh start without positions: the CCC sale has no basis -> Unmatched sell check", "Unmatched sell" in set(r["issues"]["Check"]))


class PageAcct:
    def __init__(self, acts):
        self.acts, self.calls = acts[::-1], []

    def _get(self, path, params=None):
        self.calls.append(path)
        if path == "/orders":
            return []
        i = 0 if "page_token" not in params else [a["id"] for a in self.acts].index(params["page_token"]) + 1
        return self.acts[i:i + params["page_size"]]


many = [F("ZZZ", "buy", 1, 10, "2026-09-0" + str(1 + i % 9)) for i in range(250)] + [F("ZZZ", "buy", 1, 10, "2026-10-02") for _ in range(50)]
acct = PageAcct(many)
got = tl.fetch_inputs(acct)
check("fetch_inputs stops paging at the start date: 1 activities page (the newest 100), not 4",
      acct.calls.count("/account/activities") == 1 and len(got["activities"]) == 100, acct.calls)
acct = PageAcct(many)
check("fetch_inputs(start=None) reads the whole history (4 pages, 300 activities)",
      len(tl.fetch_inputs(acct, start=None)["activities"]) == 300 and acct.calls.count("/account/activities") == 4)

# ---------------------------------------------------------------- 16. the app renders it (fake account, GET only)
from streamlit.testing.v1 import AppTest  # noqa: E402
import streamlit as st  # noqa: E402
import alpaca_paper as ap  # noqa: E402

ACTS = sorted(W + [F("AAA", "buy", 5, 50, "2026-01-05"), NTA("FEE", "2026-01-06", activity_sub_type="CAT", net_amount="-0.01",
                                                              description="CAT fee for proceed of 1 trades on 2026-01-05"),
                   F("AAA", "buy", 4, 50, "2026-10-02", order="new1")],     # the only purchase on/after TAX_START
              key=lambda a: a.get("transaction_time") or a["date"])
POS = [{"symbol": "AAA", "qty": "19", "avg_entry_price": "48.42", "cost_basis": "920", "market_value": "988", "unrealized_pl": "68",
        "unrealized_plpc": "0.074", "current_price": "52", "change_today": "0.01", "lastday_price": "51.5"}]


class FakeAccount(ap.PaperAccount):
    calls = []

    def __init__(self, *a, **k):
        pass

    def _get(self, path, params=None):
        assert path in ap.ALLOWED_PATHS, path
        FakeAccount.calls.append((path, dict(params or {})))
        if path == "/positions":
            return POS
        if path == "/account":
            return {"equity": "1000", "cash": "220", "buying_power": "220", "long_market_value": "780", "last_equity": "990"}
        if path == "/orders":
            return [{"id": "rep1", "client_order_id": "live-fri-20250220-AAA", "submitted_at": "2025-02-20T15:00:00Z"}]
        items = [a for a in ACTS if params.get("activity_types") != "FILL" or a["activity_type"] == "FILL"][::-1]
        start = 0 if "page_token" not in params else [a["id"] for a in items].index(params["page_token"]) + 1
        return items[start:start + params["page_size"]]


real, real_key = ap.PaperAccount, ap.holdings_refresh_key
ap.PaperAccount, ap.holdings_refresh_key = FakeAccount, (lambda now=None: "k1")
os.environ.pop("STOCK_ANALYSIS_LIVE_HOLDINGS", None)
try:
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file("app.py", default_timeout=180).run()
    labels = [e.label for e in at.expander]
    check("app: no exceptions", not at.exception, [str(e) for e in at.exception])
    check("app: earnings planner + tax view sections, rules still last",
          any(l.startswith("Earnings planner") for l in labels) and "Tax view · live account since Oct 2, 2026 (estimate, not tax advice)" in labels
          and labels[-1] == "Strategy rules", labels)
    md = " ".join(m.value for m in at.markdown)
    check("app: 'Estimate, not tax advice' note and card numbers (realized since Oct 2)",
          "Estimate, not tax advice." in md and "Realized since Oct 2 short-term" in md and "Wash-sale loss deferred since Oct 2" in md
          and "Realized 2026" not in md and "deferred 2025" not in md, md[:400])
    infos = " ".join(i.value for i in at.info)
    check("app: fresh-start note: start date, older activities ignored, AAA 15 sh from before the start left out",
          "Fresh start: only activity on or after Fri Oct 2, 2026 counts" in infos and "AAA 15 sh" in infos
          and "not carried in" in infos, infos[:400])
    caps = " ".join(c.value for c in at.caption)
    check("app: dated captions (As of ... CT) and the 1099-B note", "lot method FIFO" in caps and "1099-B check" in caps
          and "Wash-sale rule" in caps)
    check("app: data checks passed caption (lots = Alpaca positions)", "Data checks passed" in caps, caps[:300])
    frames = [d.value for d in at.dataframe]
    check("app: by-year, open-lots, 8949 and lot-method tables", any("Short-term gain" in f.columns for f in frames)
          and any("Holding from" in f.columns for f in frames) and any("Adjustment code" in f.columns for f in frames)
          and any("Method" in f.columns for f in frames))
    lot_tbl = next(f for f in frames if "Holding from" in f.columns)
    check("app: open lots = only the Oct 2, 2026 purchase (4 sh at $50, no wash adjustment from the 2025 trades)",
          list(lot_tbl["Bought"]) == ["Oct 2, 2026"] and close(lot_tbl["Basis / share"].iloc[0], 50)
          and close(lot_tbl["Shares"].iloc[0], 4) and not lot_tbl["Note"].str.contains("wash sale").any(), lot_tbl.to_dict("list"))
    by_tbl = next(f for f in frames if "Short-term gain" in f.columns)
    check("app: by-year table and Form 8949 have no 2025 (or other prior-year) rows", by_tbl.empty
          and all(not f.astype(str).apply(lambda c: c.str.contains("2025")).any().any() for f in frames
                  if {"Adjustment code", "Holding from", "Method", "Disallowed loss", "Qualified (est.)"} & set(f.columns)),
          by_tbl.to_dict("list"))
    paths = {p for p, _ in FakeAccount.calls}
    check("app: only allow-listed GET paths, incl. /orders for the bot flag", paths <= {"/positions", "/account", "/account/activities", "/orders", "/clock"}
          and "/orders" in paths, paths)
    n_hist = sum(1 for p, q in FakeAccount.calls if p == "/account/activities" and q.get("activity_types") is None)
    at.run()
    check("app: the account history is cached for the hour (no new full read on rerun)",
          sum(1 for p, q in FakeAccount.calls if p == "/account/activities" and q.get("activity_types") is None) == n_hist == 1, n_hist)
    check("app: estimated tax inputs with labelled defaults", any("default" in n.label for n in at.number_input)
          and any(s.label.startswith("Filing status (default: Single)") for s in at.selectbox))
    at.number_input(key="tax_agi").set_value(300000.0).run()
    check("app: editing an input reruns without errors", not at.exception, [str(e) for e in at.exception])
finally:
    ap.PaperAccount, ap.holdings_refresh_key = real, real_key
    st.cache_data.clear()
    st.cache_resource.clear()

src = open(os.path.join(ROOT, "tax_lots.py")).read()
check("tax_lots.py never places or cancels orders (reads only through PaperAccount._get)",
      not any(w in src for w in ("submit_order", "cancel_order", "method=\"POST\"", "TradingClient", "requests.post")))
print(f"\n{len(FAIL)} failed" if FAIL else "\nTAX LOTS OK")
sys.exit(1 if FAIL else 0)
