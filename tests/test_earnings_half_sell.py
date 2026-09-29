"""Strategy rule: earnings half-sell (WINNER['earnings_sell_fraction'] = 0.5, window WINNER['earnings_block_days'] = 5).

Synthetic data, no network: a HELD stock with earnings within 5 calendar days of a decision is cut by half ONCE per
earnings event (Friday rebalance or Mon/Wed check), is not bought back up before its earnings, and the normal rules
resume after the earnings date. Setting the option to None reverts to the old behavior.
Run: cd <folder> && python3 tests/test_earnings_half_sell.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import pandas as pd

import backtest_engine as be

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f" ({detail})" if not cond else ""))


# ---- unit: earnings_half_sell
w, held, done = np.array([0.10, 0.10, 0.10]), np.array([0.10, 0.10, 0.0]), {}
days = np.array([3.0, np.nan, 2.0])                       # A: earnings in 3 days; B: none; C: not held
out = be.earnings_half_sell(w, held, days, "2026-09-28", 0.5, done)
check("unit: held stock with earnings in the window is cut in half", w[0] == 0.05 and done == {0: pd.Timestamp("2026-10-01")})
check("unit: plain-word reason", out and out[0][3] == "earnings Thu Oct 01: sold half before earnings", out)
check("unit: no earnings -> untouched; not held -> untouched (the buy block handles it)", w[1] == 0.10 and w[2] == 0.10)
w2 = np.array([0.12, 0.10, 0.10])                         # next decision, same event, target back at 12%
be.earnings_half_sell(w2, np.array([0.05, 0.10, 0.0]), np.array([1.0, np.nan, np.nan]), "2026-09-30", 0.5, done)
check("unit: same earnings event again -> not sold again, only capped at the held 5% (no top-up)", w2[0] == 0.05, list(w2))

# ---- integration: winner_targets on synthetic scores (3 stocks, all picked: 3/10 invested, 10% each)
dates = pd.bdate_range("2026-09-14", "2026-10-09")
syms = ["AAA", "BBB", "CCC", "DDD"]
score = pd.DataFrame({"AAA": 3.0, "BBB": 2.0, "CCC": 1.0, "DDD": np.nan}, index=dates)
score.loc["2026-09-24":, "DDD"] = 0.5                     # DDD qualifies from Thu 9/24 (not held yet)
elig = pd.DataFrame(True, index=dates, columns=syms)
vol = pd.DataFrame(0.3, index=dates, columns=syms)
regime = pd.Series(True, index=dates)
weekly = be.weekly_rebalance_days(dates)
earn = pd.DataFrame({"Symbol": ["AAA", "BBB", "DDD"],
                     "Earnings Date": pd.to_datetime(["2026-10-01", "2026-09-29", "2026-09-29"])})
MW = {"enter_top": 3, "exit_below": 15, "days": ["Mon", "Wed"]}


def run(frac):
    chk, dec = [], []
    t, _ = be.winner_targets(score, elig, vol, regime, weekly, check_log=chk, decision_log=dec, midweek=MW,
                             exit_all_below=None, earnings_block_days=5, earnings=earn, earnings_sell_fraction=frac)
    return t, pd.DataFrame(chk), pd.DataFrame(dec)


t, chk, dec = run(0.5)
wt = lambda d, s: round(float(t.at[pd.Timestamp(d), s]), 4)
check("Fri 9/18: AAA, BBB, CCC held at 10% each (3 of 10 slots), DDD not qualified yet",
      [wt("2026-09-18", s) for s in syms] == [0.1, 0.1, 0.1, 0.0], t.loc["2026-09-18"].round(4).to_dict())
base_a = wt("2026-09-25", "AAA")
check("Fri 9/25: BBB (earnings Tue 9/29) held -> sold half at the Friday rebalance",
      wt("2026-09-25", "BBB") == round(wt("2026-09-18", "BBB") / 2, 4), (wt("2026-09-18", "BBB"), wt("2026-09-25", "BBB")))
r = dec[(dec.Date == "2026-09-25") & (dec.Symbol == "BBB")]
check("Fri 9/25: reason says 'earnings Tue Sep 29: sold half before earnings'",
      len(r) == 1 and r.Reason.iloc[0] == "earnings Tue Sep 29: sold half before earnings", r.Reason.tolist())
check("Fri 9/25: DDD (not held, earnings Tue 9/29) is not newly bought", wt("2026-09-25", "DDD") == 0)
tr = chk[chk.Action == "TRIM"] if len(chk) else chk
check("Mon 9/28: AAA (earnings Thu 10/1) sold half once at the check (TRIM, 50%)",
      len(tr[(tr.Date == "2026-09-28") & (tr.Sell == "AAA")]) == 1 and wt("2026-09-28", "AAA") == round(base_a / 2, 4),
      (tr[["Date", "Sell"]].to_dict("records") if len(tr) else [], wt("2026-09-28", "AAA")))
check("Mon 9/28: BBB already halved for this earnings -> not sold again, not topped up",
      len(tr[tr.Sell == "BBB"]) == 0 and wt("2026-09-28", "BBB") == wt("2026-09-25", "BBB"))
check("Wed 9/30: AAA still before earnings -> no second sale", len(tr[tr.Sell == "AAA"]) == 1
      and wt("2026-09-30", "AAA") == wt("2026-09-28", "AAA"))
check("TRIM note in plain words", len(tr) and tr.Note.iloc[0].startswith("earnings Thu Oct 01: sold half before earnings"),
      tr.Note.tolist() if len(tr) else [])
check("Fri 10/2 (after both earnings): AAA and BBB back to normal target weight, DDD bought",
      wt("2026-10-02", "AAA") == wt("2026-10-02", "CCC") == wt("2026-10-02", "BBB") and wt("2026-10-02", "DDD") > 0,
      t.loc["2026-10-02"].round(4).to_dict())
check("freed weight stays cash until Friday (invested drops after the Mon trim)",
      t.loc["2026-09-28"].sum() < t.loc["2026-09-25"].sum())

t0, chk0, _ = run(None)
check("revert: earnings_sell_fraction=None -> no TRIM rows, no halving",
      (len(chk0) == 0 or not (chk0.Action == "TRIM").any()) and
      float(t0.at[pd.Timestamp("2026-09-28"), "AAA"]) == float(t0.at[pd.Timestamp("2026-09-18"), "AAA"]))
check("revert: held set is the same with and without the half-sell (weights only)",
      ((t > 0) == (t0 > 0)).all().all())
check("live WINNER has the option on (0.5) and the window (5 days)",
      be.WINNER.get("earnings_sell_fraction") == 0.5 and be.WINNER.get("earnings_block_days") == 5)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
