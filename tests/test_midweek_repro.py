"""Regression: the LIVE engine (backtest_engine.winner_targets) reproduces the tested backtests exactly.
  1. plain C6-U96-MW (Mon/Wed top 3 in / below 15 out):  +470.31%, Sharpe 1.4393
  2. C6-U96-MW30 (+ sell anything worse than rank 30 at the Mon/Wed checks): +457.14%, Sharpe 1.4451 (sell-rule test S3)
  3. C6-U96-T20-MW30 (picks only from ranks 1-20, sector cap relaxed to fill 10 slots, top-3 swaps ignore the cap):
     +452.96%, Sharpe 1.3172, never-seen 0.5942
  4. C6-U96-T20-MW30-E5 (the live rules from 2026-09-25: + no new buys with earnings within 5 days; PARTIAL - the earnings
     dates on disk start in late 2024): engine == the independent re-implementation, numbers pinned as a regression
     (pinned values re-baselined 2026-09-25 after simulate() was fixed to trade only on target changes)
  (3 and 4 were user decisions, not pre-registered tests.)
Compares targets, swap and sell logs with the independent re-implementation in tests/backtest_setup.py, and the live
pipeline's decision history (Reports/strategy_decisions.csv) with the test over the overlapping window.
Run: python tests/run_tests.py  (or PYTHONPATH=. python tests/test_midweek_repro.py)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import backtest_engine as be
import backtest_setup as g

# (exit_below, t20, earnings rule) -> (total return %, Sharpe) at 0.1%/side, and the never-seen 2022-04 -> 2024-09 Sharpe
EXPECTED = {(None, False, False): (470.31, 1.4393), (30, False, False): (457.14, 1.4451), (30, True, False): (452.96, 1.3172),
            (30, True, True): (387.57, 1.2249)}
EXPECTED_NEVER_SEEN = {(None, False, False): 0.8511, (30, False, False): 0.8211, (30, True, False): 0.5942,
                       (30, True, True): 0.5942}
MW = {"enter_top": 3, "exit_below": 15, "days": ["Mon", "Wed"]}


def check(exit_all, t20=False, e5=False):
    chk = []
    kw = {"exit_all_below": exit_all, "earnings_block_days": 5 if e5 else None,
          "selection": dict(max_pick_rank=20, cap_soft=True) if t20 else g.PLAIN}
    t_live, checks = be.winner_targets(g.sc, g.el, g.vol, g.reg, g.weekly, tiebreak_w=g.rs, check_log=chk, midweek=MW, **kw)
    t_test, swaps_test, sells_test = g.buffered_midweek(3, 15, exit_all, t20=t20, block=g.block_matrix(5) if e5 else None)
    assert (checks.to_numpy(bool) == g.midweek.to_numpy(bool)).all(), "check-day calendars differ"
    diff = np.abs(t_live.to_numpy() - t_test.reindex_like(t_live).to_numpy()).max()
    sw = pd.DataFrame([c for c in chk if c["Action"] == "SWAP"])
    se = pd.DataFrame([c for c in chk if c["Action"] == "SELL"])
    same_swaps = len(sw) == len(swaps_test) and all(
        a.Date == b["Date"] and a.In == b["Buy"] and a.Out == b["Sell"] and a.In_rank == b["Buy_Rank"]
        and (a.Out_rank == b["Sell_Rank"] or (pd.isna(a.Out_rank) and pd.isna(b["Sell_Rank"])))
        for a, (_, b) in zip(swaps_test.itertuples(), sw.iterrows()))
    same_sells = len(se) == len(sells_test) and all(
        a.Date == b["Date"] and a.Sell == b["Sell"] and (a.Rank == b["Sell_Rank"] or (pd.isna(a.Rank) and pd.isna(b["Sell_Rank"])))
        for a, (_, b) in zip(sells_test.itertuples(), se.iterrows()))
    res = be.simulate(g.O, g.C, g.full(t_live), g.WF, rebalance=g.weekly, cost=be.COST)
    m, ns = be.metrics(res), g.em(res["equity"].loc[:g.NEVER_END])
    label = ("C6-U96-T20-MW" if t20 else "C6-U96-MW") + ("" if exit_all is None else str(exit_all)) + ("-E5" if e5 else "")
    print(f"[{label}] check days {int(checks.sum())} | swaps {len(sw)} (same as test: {same_swaps}) | rank-{exit_all} sells "
          f"{len(se)} (same as test: {same_sells}) | max |target diff| {diff:.3g}")
    print(f"[{label}] total {m['Total Return %']:.2f}%  Sharpe {m['Sharpe']:.4f}  max DD {m['Max DD %']:.2f}%  "
          f"never-seen Sharpe {ns['Sharpe']:.4f}")
    assert diff < (1e-12 if t20 else 1e-300) and same_swaps and same_sells, (diff, same_swaps, same_sells)
    if e5:
        skips = [c for c in chk if "earnings in" in str(c.get("Note", ""))]
        print(f"[{label}] mid-week checks where an earnings block stopped a swap: {len(skips)}")
    if EXPECTED[(exit_all, t20, e5)] is not None:
        assert (round(m["Total Return %"], 2), round(m["Sharpe"], 4)) == EXPECTED[(exit_all, t20, e5)], m
        assert round(ns["Sharpe"], 4) == EXPECTED_NEVER_SEEN[(exit_all, t20, e5)], ns
    if exit_all is not None:
        assert len(se) >= 2, "need at least 2 historical rank sells"
        print(f"[{label}] last rank-{exit_all} sells:\n" + se[["Date", "Sell", "Sell_Rank", "Weight"]].tail(4).to_string(index=False))
    print(f"[{label}] REPRODUCED EXACTLY")
    return swaps_test, sells_test


plain = check(None)
live_exit = be.WINNER.get("midweek_exit_below")
live_t20 = bool(be.WINNER.get("max_pick_rank") == 20 and be.WINNER.get("cap_soft"))
mw30 = check(30)
t20 = check(30, t20=True)
e5 = check(30, t20=True, e5=True)
live_e5 = be.WINNER.get("earnings_block_days") == 5
swaps_test, sells_test = {(None, False, False): plain, (30, False, False): mw30, (30, True, False): t20,
                          (30, True, True): e5}[(live_exit, live_t20, live_e5)]

# --- the live pipeline output (shorter data window) agrees with the test over the overlap ---
if be.WINNER.get("midweek_swap"):
    dec = pd.read_csv(os.path.join(be.REPORTS_DIR, "strategy_decisions.csv"), parse_dates=["Date"])
    first_live = dec.Date.min() + pd.Timedelta(days=7)          # after the first live weekly decision
    adds = dec[(dec.Status == "add") & dec.Reason.astype(str).str.startswith("mid-week swap in")]
    live_pairs = {(d, s, r.split("replaces ")[1].split(" ")[0]) for d, s, r in zip(adds.Date, adds.Symbol, adds.Reason) if d >= first_live}
    test_pairs = {(d, i, o) for d, i, o in zip(swaps_test.Date, swaps_test.In, swaps_test.Out) if d >= first_live}
    print(f"live pipeline swaps since {first_live.date()}: {len(live_pairs)} | test: {len(test_pairs)} | identical: {live_pairs == test_pairs}")
    assert live_pairs == test_pairs, sorted(live_pairs ^ test_pairs)[:10]
    if live_exit:
        ex = dec[(dec.Status == "drop") & dec.Reason.astype(str).str.startswith("mid-week exit")]
        live_sells = {(d, s) for d, s in zip(ex.Date, ex.Symbol) if d >= first_live}
        test_sells = {(d, s) for d, s in zip(sells_test.Date, sells_test.Sell) if d >= first_live}
        print(f"live pipeline rank-{live_exit} sells since {first_live.date()}: {len(live_sells)} | test: {len(test_sells)} | "
              f"identical: {live_sells == test_sells}")
        assert live_sells == test_sells and len(live_sells) >= 2, sorted(live_sells ^ test_sells)[:10]
print("MIDWEEK REPRO OK")
