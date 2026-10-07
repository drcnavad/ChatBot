"""Live Mon/Wed sell rule (since 2026-10-07, t187u): no top-3 swap; a holding worse than rank 20 is always sold and
replaced 1-for-1 by the best non-held top-10 name (same weight); cash only when nothing qualifies (was rank 30 + swap). Matches forward_test's former
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
check("live WINNER: midweek_exit_below=20 and midweek_exit_to_top=10",
      be.WINNER.get("midweek_exit_below") == 20 and be.WINNER.get("midweek_exit_to_top") == 10)
check("live tag / name / rules version: exit-only (X20R10), no swap",
      "-X20R10-" in be.WINNER["tag"] and "MW" not in be.WINNER["tag"] and "worse than rank 20" in be.WINNER["name"]
      and "replace with best top-10" in be.WINNER["name"] and "swap" not in be.WINNER["name"]
      and be.rules_version() == "v5-x20r10-ns-e5", (be.WINNER["tag"], be.rules_version()))
check("Fri pure top-10 (no sector / no rank-20 gate); 20% / earnings kept; Mon/Wed checks kept, top-3 swap off",
      be.WINNER["max_weight"] == 0.2 and be.WINNER["earnings_block_days"] == 5
      and be.WINNER["midweek_swap"] == {"enter_top": None, "exit_below": None, "days": ["Mon", "Wed"]}
      and be.WINNER.get("max_pick_rank") is None and be.WINNER["sector_cap"] >= 1.0 - 1e-12
      and "NS" in be.WINNER["tag"] and "no sector limit" in be.WINNER["name"])
check("exit_to_top=None still means cash-until-Friday (midweek_exit_replacements no-ops)",
      be.midweek_exit_replacements(np.array([0.1, 0.0, 0.08]), [0, 1, 2], {0: 1, 1: 2, 2: 5}, 3, None) == [])

# --- the live Mon/Wed rule through the engine (apply_midweek_swaps with the WINNER settings): 25 names, S00 = rank 1 ...
import pandas as pd

_cols = [f"S{i:02d}" for i in range(25)]
_dates = pd.to_datetime(["2026-10-02", "2026-10-05"])                 # Fri rebalance, Mon check
_score = pd.DataFrame([[100.0 - i for i in range(25)]] * 2, index=_dates, columns=_cols)
_ok = pd.DataFrame(True, index=_dates, columns=_cols)


def mw_run(held, rules):
    """Holdings (column numbers, 9% each) after the Monday check; and the check-log actions."""
    base = pd.DataFrame(0.0, index=_dates, columns=_cols)
    base.loc[:, [_cols[i] for i in held]] = 0.09
    log = []
    t = be.apply_midweek_swaps(base, _score, _ok, _ok * 0.2, pd.Series([True, False], index=_dates),
                               pd.Series([False, True], index=_dates), sector_cap=1.0, n=10, cap_soft=True, check_log=log,
                               **rules)
    return sorted(int(c[1:]) for c in t.columns[t.iloc[1] > 0]), [(r["Action"], r["Sell"], r["Buy"]) for r in log]


_mw = be.WINNER["midweek_swap"]
LIVE = dict(enter_top=_mw["enter_top"], exit_below=_mw["exit_below"], exit_all_below=be.WINNER["midweek_exit_below"],
            exit_to_top=be.WINNER["midweek_exit_to_top"])
OLD = dict(enter_top=3, exit_below=15, exit_all_below=30, exit_to_top=10)          # the rules until 2026-10-07
A = [1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 24]                                           # ranks 2,4-11, 17, 25; S00 / S02 not held
check("live: rank 25 is sold and replaced by the best top-10 not held (rank 1); rank 17 stays (no top-3 swap)",
      mw_run(A, LIVE) == ([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 16], [("REPLACE", "S24", "S00")]), mw_run(A, LIVE))
check("old rules (forward test pins) on the same day: two top-3 swaps, rank 17 out",
      mw_run(A, OLD)[1] == [("SWAP", "S24", "S00"), ("SWAP", "S16", "S02")], mw_run(A, OLD))
B = [2, 3, 4, 5, 6, 7, 8, 9, 21, 24]                                               # ranks 22, 25; S00 / S01 not held
check("live: several worse than 20 -> worst first, best top-10 first, 1-for-1",
      mw_run(B, LIVE)[1] == [("REPLACE", "S24", "S00"), ("REPLACE", "S21", "S01")], mw_run(B, LIVE))
C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 24]                                            # only S00 left in the top 10
check("live: refills run out -> the next worse-than-20 holding is sold to cash",
      mw_run(C, LIVE)[1] == [("REPLACE", "S24", "S00"), ("SELL", "S21", "")], mw_run(C, LIVE))
D = list(range(10)) + [19]                                                         # rank 20 = not worse than 20
check("live: rank 20 is kept; nothing to do -> one NO CHANGE row ('no holding is worse than rank 20')",
      mw_run(D, LIVE)[1] == [("NO CHANGE", "", "")] and 19 in mw_run(D, LIVE)[0], mw_run(D, LIVE))

# --- paper_trade must treat REPLACE like a mid-week trade (Sell+Buy), not ignore it ---
import tempfile
import paper_trade as pt

_csv = (
    "As_Of,Event,Event_Date,Applies_To_Open,Action,Sell,Buy,Message,Sell_Rank,Buy_Rank,Weight_%,Is_Latest\n"
    "2026-10-05,mid-week check,2026-10-05,2026-10-06,REPLACE,BE,U,exit refill,40,10,6.0,1\n"
    "2026-10-05,mid-week check,2026-10-05,2026-10-06,SELL,ZZ,,cash leftover,55,,5.0,1\n"
    "2026-10-05,mid-week check,2026-10-05,2026-10-06,NO CHANGE,,,quiet,,,,1\n"
)
with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
    f.write(_csv)
    path = f.name
rows = pt.latest_midweek_changes(path, "2026-10-05")
check("paper_trade.latest_midweek_changes includes REPLACE and SELL, skips NO CHANGE",
      list(rows["Action"]) == ["REPLACE", "SELL"] and list(rows["Buy"].fillna("")) == ["U", ""],
      rows.to_dict("list"))
orders = pt.build_replace_orders(rows, account_size=100_000, positions={"BE": 10, "ZZ": 5},
                              prices={"BE": 50.0, "U": 25.0, "ZZ": 40.0}, fractional=True)
sides = orders.set_index("Symbol")["Side"].to_dict()
check("REPLACE builds SELL BE + BUY U; SELL builds SELL ZZ only",
      sides.get("BE") == "SELL" and sides.get("U") == "BUY" and sides.get("ZZ") == "SELL",
      orders[["Symbol", "Side", "Shares"]].to_dict("list"))

print(f"\n{len(FAIL)} failed" if FAIL else "\nMIDWEEK EXIT REPLACE OK")
sys.exit(1 if FAIL else 0)
