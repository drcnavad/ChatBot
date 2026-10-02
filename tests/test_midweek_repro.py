"""Regression: the LIVE engine (backtest_engine.winner_targets) reproduces the tested backtests exactly, on the live universe
(sector_mapping.tradable_symbols; pinned for the 91-stock list of 2026-10-01 = tag C6-U91, see PINNED_UNIVERSE):
  1. plain C6-U91-MW (Mon/Wed top 3 in / below 15 out):  +463.17%, Sharpe 1.4156, never-seen 0.7617
  2. C6-U91-MW30 (+ sell anything worse than rank 30 at the Mon/Wed checks): +466.67%, Sharpe 1.4423, never-seen 0.7584
  3. C6-U91-T20-MW30 (picks only from ranks 1-20, sector cap relaxed to fill 10 slots, top-3 swaps ignore the cap):
     +451.51%, Sharpe 1.3227, never-seen 0.5887
  4. C6-U91-T20-MW30-E5 (the live rules from 2026-09-25: + no new buys with earnings within 5 days; PARTIAL - the earnings
     dates on disk start in late 2024): +414.17%, Sharpe 1.2700
  (3 and 4 were user decisions, not pre-registered tests.) Pins re-baselined 2026-10-01 for the universe change (user decision:
  drop ADBE AFRM MU SOFI MDB MSTR, add TTWO; TTWO bars added to Reports/cache/bars_daily_long.pkl), after the engine still
  matched the independent re-implementation exactly and the live decision history over the overlap. Previous pins (U96, 96
  stocks): 1. +470.31% / 1.4393 / 0.8511, 2. +457.14% / 1.4451 / 0.8211, 3. +452.96% / 1.3172 / 0.5942, 4. +387.57% / 1.2249 / 0.5942
  (4 re-baselined 2026-09-25 after simulate() was fixed to trade only on target changes).
Compares targets, swap and sell logs with the independent re-implementation in tests/backtest_setup.py, and the live
pipeline's decision history (Reports/strategy_decisions.csv) with the test over the overlapping window.
Run: python tests/run_tests.py  (or PYTHONPATH=. python tests/test_midweek_repro.py)"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import backtest_engine as be
import backtest_setup as g

# (exit_below, t20, earnings rule) -> (total return %, Sharpe) at 0.1%/side, and the never-seen 2022-04 -> 2024-09 Sharpe
# pinned for ONE stock list (C6-U91 of 2026-10-01, fingerprint below). After a stock is added to / removed from
# sector_mapping.py the pins are skipped (loudly) - the exactness checks against the independent re-implementation and the
# live history still run; re-baseline the numbers deliberately once they pass.
PINNED_UNIVERSE = "82c08da0d768"   # sha1 of the sorted SCORED universe (tradable_symbols minus short-history stocks), 12 hex
FINGERPRINT = hashlib.sha1(",".join(sorted(g.U)).encode()).hexdigest()[:12]
PINNED = FINGERPRINT == PINNED_UNIVERSE
EXPECTED = {(None, False, False): (463.17, 1.4156), (30, False, False): (466.67, 1.4423), (30, True, False): (451.51, 1.3227),
            (30, True, True): (414.17, 1.2700)}
EXPECTED_NEVER_SEEN = {(None, False, False): 0.7617, (30, False, False): 0.7584, (30, True, False): 0.5887,
                       (30, True, True): 0.5887}
if not PINNED:
    print(f"SKIP PINNED NUMBERS: the stock list changed ({len(g.U)} stocks, fingerprint {FINGERPRINT} != {PINNED_UNIVERSE}); "
          f"exactness checks still run - re-baseline EXPECTED / PINNED_UNIVERSE once they pass")
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
    label = (f"C6-U{len(g.U)}-T20-MW" if t20 else f"C6-U{len(g.U)}-MW") + ("" if exit_all is None else str(exit_all)) + ("-E5" if e5 else "")
    print(f"[{label}] check days {int(checks.sum())} | swaps {len(sw)} (same as test: {same_swaps}) | rank-{exit_all} sells "
          f"{len(se)} (same as test: {same_sells}) | max |target diff| {diff:.3g}")
    print(f"[{label}] total {m['Total Return %']:.2f}%  Sharpe {m['Sharpe']:.4f}  max DD {m['Max DD %']:.2f}%  "
          f"never-seen Sharpe {ns['Sharpe']:.4f}")
    assert diff < (1e-12 if t20 else 1e-300) and same_swaps and same_sells, (diff, same_swaps, same_sells)
    if e5:
        skips = [c for c in chk if "earnings in" in str(c.get("Note", ""))]
        print(f"[{label}] mid-week checks where an earnings block stopped a swap: {len(skips)}")
    if PINNED and EXPECTED[(exit_all, t20, e5)] is not None:
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
_dec_path = os.path.join(be.REPORTS_DIR, "strategy_decisions.csv")
_sa = pd.read_csv(os.path.join(be.REPORTS_DIR, "signal_analysis.csv"), usecols=["Date", "Symbol"])
_run_list = set(_sa.loc[_sa.Date == _sa.Date.max(), "Symbol"]) - set(be.BENCHMARKS)   # the list of the last pipeline run
_list_changed = _run_list != set(g.U)
if _list_changed:
    print(f"SKIP live-history check: the stock list changed since the last main_signal_analysis run (added "
          f"{sorted(set(g.U) - _run_list)}, removed {sorted(_run_list - set(g.U))}) - it runs again after the next pipeline run")
if be.WINNER.get("midweek_swap") and not _list_changed:
    dec = pd.read_csv(_dec_path, parse_dates=["Date"])
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
