"""The live rules from Fri Oct 16, 2026 (Chirag): Low volatility in uptrend (monthly) = forward_test.py's "Low volatility in
uptrend" (score low_vol, monthly, equal weight, max 4 per sector, no earnings rule, no sleeve) + do_not_buy +
the optional Friday check (WINNER['friday_falling_check'], OFF live; tested with the flag on): sell a holding below its entry price, at/below its 200-day
average or out of the top 10, same dollars into the best pick not held; no Mon-Thu trading; the earnings-day stop off.
Mocked: no broker calls, temporary files only."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
os.chdir(ROOT)
import pandas as pd  # noqa: E402

import backtest_engine as be  # noqa: E402
import forward_test as ft  # noqa: E402
import paper_trade as pt  # noqa: E402
import sector_mapping as sm  # noqa: E402

FAIL = []

def check(ok, what, got=""):
    print(("PASS " if ok else "FAIL ") + what + ("" if ok else f"  ({got})"))
    if not ok:
        FAIL.append(what)

W = be.WINNER
cfg = next(c for c in ft.STRATEGIES if c["name"] == "Low volatility in uptrend")
sel = {**ft.PIN, **cfg["select"]}
check(cfg["score"] == W["score"] == "low_vol" and cfg["calendar"] == W["calendar"] == "monthly", "score / calendar = forward strategy")
for k in ("sector_cap", "max_pick_rank", "cap_soft"):
    check(sel[k] == W[k], f"selection {k} = forward strategy", (sel[k], W[k]))
check(cfg["earnings"] is None and not W["earnings_block_days"], "no earnings rule (forward strategy: earnings None)")
check(not W["midweek_swap"] and not W["midweek_exit_below"], "no Mon-Thu checks")
check(not W["sleeve"] and "sleeve" not in be.winner_rank_args(pd.Series(dtype=bool)) and sm.do_not_sell == [], "no sleeve, do_not_sell empty")
check(not W["keep_profit_at_rebalance"] and not W["earnings_day_stop"] and not W["friday_falling_check"], "keep rule off, stop off, Friday check off (plain monthly)")
check(pt.MAX_ORDER_PCT == 0.20, "order backstop = 20% of equity (safety only)")

# calendar of the low-volatility rules alone (the sector half's Fridays: tests/test_sleeves.py): Oct 16 switch, then the
# last trading day of each month; nothing in between
days = pd.date_range("2026-10-12", "2027-01-31", freq="D")
sleeves_cfg = W.pop("sleeves", None)
W["friday_falling_check"] = False
dec = {d.date().isoformat(): be.decision_kind(d) for d in days if be.decision_kind(d)}
W["friday_falling_check"] = True   # the flag's own calendar
check(dec == {"2026-10-16": "full rebalance", "2026-10-30": "full rebalance", "2026-11-30": "full rebalance",
              "2026-12-31": "full rebalance", "2027-01-29": "full rebalance"}, "decision days Oct 16, Oct 30, Nov 30, Dec 31, Jan 29", dec)
fridays = [d for d in days if d.dayofweek == 4 and d.date().isoformat() not in dec and be.is_session(d)]
check(all(be.decision_kind(d) == "Friday check" for d in fridays) and be.decision_kind(pd.Timestamp("2026-10-30")) == "full rebalance"
      and be.decision_kind(pd.Timestamp("2026-10-20")) is None, "flag on: every other Friday is a 'Friday check', Mon-Thu still none")
W["friday_falling_check"] = False
check(be.decision_kind(pd.Timestamp("2026-10-23")) is None, "flag off: Fri Oct 23 is no decision")
if sleeves_cfg:
    W["sleeves"] = sleeves_cfg


print("LOW VOL MONTHLY OK" if not FAIL else f"LOW VOL MONTHLY FAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
