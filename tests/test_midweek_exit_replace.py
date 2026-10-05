"""Live Mon/Wed exit refill (WINNER['midweek_exit_to_top']=10): a holding worse than rank 30 is swapped for the best
non-held top-10 name (same weight); cash only when nothing qualifies. Matches forward_test's former
"Live, exit replaced by top-10" rule. Hand-worked cases; no network, no orders.
Run: python tests/run_tests.py  (or python tests/test_midweek_exit_replace.py)"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest_engine as be

FAIL = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)


# columns 0..4 = ranks 1..5 by score order; hold 0 and 4 (ranks 1 and 5); exit_below for exit = 3 means rank>3 exits
# so holding 4 (rank 5) should leave; best non-held in top-3 is column 1 (rank 2).
cur = np.array([0.10, 0.0, 0.0, 0.0, 0.08])
order = [0, 1, 2, 3, 4]
rank = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
reps = be.midweek_exit_replacements(cur, order, rank, exit_all_below=3, exit_to_top=3)
check("replace: worst held rank 5 -> best non-held top-3 (col 1), same weight",
      reps == [(1, 4, 0.08)] and np.isclose(cur[1], 0.08) and cur[4] == 0.0 and np.isclose(cur[0], 0.10), (reps, cur))
exits = be.midweek_exit_sells(cur, rank, 3)
check("after a successful refill nothing is left to sell to cash", exits == [] and cur.sum() == cur[0] + cur[1], exits)

# earnings skip: col 1 blocked -> take col 2
cur = np.array([0.10, 0.0, 0.0, 0.0, 0.08])
reps = be.midweek_exit_replacements(cur, order, rank, 3, 3, skip={1})
check("replace: earnings skip on the best candidate -> next top-3 name",
      reps == [(2, 4, 0.08)] and np.isclose(cur[2], 0.08) and cur[1] == 0.0, reps)

# no refill left in top-N -> no replacement; midweek_exit_sells takes the cash
cur = np.array([0.10, 0.09, 0.08, 0.0, 0.07])   # top-3 all held
reps = be.midweek_exit_replacements(cur, order, rank, 3, 3)
check("replace: top-3 already held -> no refill", reps == [] and np.isclose(cur[4], 0.07), reps)
exits = be.midweek_exit_sells(cur, rank, 3)
check("no refill -> cash exit of the rank>3 holding", exits == [(4, 0.07)] and cur[4] == 0.0, exits)

# live defaults
check("live WINNER: midweek_exit_below=30 and midweek_exit_to_top=10",
      be.WINNER.get("midweek_exit_below") == 30 and be.WINNER.get("midweek_exit_to_top") == 10)
check("live tag names the refill (MW30R10)", "MW30R10" in be.WINNER["tag"] and "replace with best top-10" in be.WINNER["name"])
check("Fri pure top-10 (no sector / no rank-20 gate); 20% / earnings / top-3 swap kept",
      be.WINNER["max_weight"] == 0.2 and be.WINNER["earnings_block_days"] == 5
      and be.WINNER["midweek_swap"] == {"enter_top": 3, "exit_below": 15, "days": ["Mon", "Wed"]}
      and be.WINNER.get("max_pick_rank") is None and be.WINNER["sector_cap"] >= 1.0 - 1e-12
      and "NS" in be.WINNER["tag"] and "no sector limit" in be.WINNER["name"])
check("exit_to_top=None still means cash-until-Friday (midweek_exit_replacements no-ops)",
      be.midweek_exit_replacements(np.array([0.1, 0.0, 0.08]), [0, 1, 2], {0: 1, 1: 2, 2: 5}, 3, None) == [])

print(f"\n{len(FAIL)} failed" if FAIL else "\nMIDWEEK EXIT REPLACE OK")
sys.exit(1 if FAIL else 0)
