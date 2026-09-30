"""Order-signal audit (2026-09-28): the backtest fills follow the live planner, and the gaps found in the audit stay fixed.

Synthetic data and mocks only - no network, no broker calls, no pop-ups.
  * simulate(band=...) = the live rule: Friday brings every pick back to its weight outside the 1-point band (bought up
    or trimmed), a held pick with earnings soon is not bought up, a Mon/Wed weight change alone is never traded (no
    earnings half-sell), and band=None keeps the old adds/exits-only fills.
  * paper_trade.NO_TRADE_BAND equals WINNER['rebalance_band'] (live and backtest use one number).
  * paper_trade --submit (manual CLI) caps buys at buying power + sells, less the 1% cushion.
  * run_all --fill-check never runs twice at once (lock file).
  * the watchdog falls back to another Gemini model on HTTP 404.
Run: cd <folder> && python3 tests/test_live_rules_audit.py
"""
import io
import os
import sys
import tempfile
import urllib.error
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("STOCK_ANALYSIS_RUN_LOG", os.path.join(tempfile.gettempdir(), "sa_test_run_log.csv"))  # never the real run log
import numpy as np
import pandas as pd

import backtest_engine as be
import paper_trade as pt

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f" ({detail})" if not cond else ""))


# ---- simulate(band=): two stocks, flat prices except where noted; decisions at the close, fills at the next open
dates = pd.bdate_range("2026-09-21", "2026-10-09")                 # Mon Sep 21 .. Fri Oct 9
cols = ["AAA", "BBB"]
px = pd.DataFrame(100.0, index=dates, columns=cols)
px.loc["2026-09-23":, "AAA"] = 150.0                               # AAA jumps +50% from Wed Sep 23 (drifts overweight)
tgt = pd.DataFrame(0.0, index=dates, columns=cols)
tgt.loc[:, ["AAA", "BBB"]] = 0.4                                   # both held at 40% from the first decision
weekly = pd.Series(dates.dayofweek == 4, index=dates)             # Fridays
res_live = be.simulate(px, px, tgt, "2026-09-22", rebalance=weekly, cost=0.0, band=0.01)
res_old = be.simulate(px, px, tgt, "2026-09-22", rebalance=weekly, cost=0.0)
kinds_live = res_live["trades"]["Kind"].tolist()
check("band: Friday trims the overweight pick back to its weight (a 'trim' trade)", "trim" in kinds_live, kinds_live)
check("band=None: the old rule never resizes a hold (no trim)", "trim" not in res_old["trades"]["Kind"].tolist(),
      res_old["trades"]["Kind"].tolist())
# after the Fri Sep 25 rebalance (filled Mon Sep 28 open) AAA should be ~40% of equity again
eq, expo = res_live["equity"], res_live["exposure"]
check("band: invested share stays ~80% after the rebalance", abs(expo.loc["2026-09-28"] - 0.8) < 0.01, expo.loc["2026-09-28"])

# earnings: a held pick is not bought up at the Friday rebalance when its earnings are within the window
px2 = pd.DataFrame(100.0, index=dates, columns=cols)
px2.loc["2026-09-23":, "AAA"] = 60.0                               # AAA falls: underweight at the Friday rebalance
block = pd.DataFrame(np.nan, index=dates, columns=cols)
block.loc["2026-09-25", "AAA"] = 3.0                               # earnings Mon Sep 28 (3 days after Fri Sep 25)
r_blk = be.simulate(px2, px2, tgt, "2026-09-22", rebalance=weekly, cost=0.0, band=0.01, block=block)
r_free = be.simulate(px2, px2, tgt, "2026-09-22", rebalance=weekly, cost=0.0, band=0.01)
n_blk = (r_blk["trades"]["Symbol"] == "AAA").sum()
check("band: without the earnings block the underweight pick is bought up on Monday",
      r_free["exposure"].loc["2026-09-28"] > r_blk["exposure"].loc["2026-09-28"] + 0.05,
      (r_free["exposure"].loc["2026-09-28"], r_blk["exposure"].loc["2026-09-28"]))
check("band: with earnings in the window it is NOT bought up before earnings", n_blk == 0, n_blk)

# a mid-week weight change alone (no swap, no exit) is never traded: the band applies only at the Friday rebalance
tgt3 = tgt.copy()
tgt3.loc["2026-09-30":"2026-10-01", "BBB"] = 0.2
t3 = be.simulate(px, px, tgt3, "2026-09-22", rebalance=weekly, cost=0.0, band=0.01)["trades"]
check("band: a Mon/Wed weight change alone is not traded (no earnings half-sell)",
      not ((t3["Symbol"] == "BBB") & (t3["Exit"] == pd.Timestamp("2026-10-01"))).any(), t3.to_string())

# ---- one band number for live and backtest
check("live band == backtest band (paper_trade.NO_TRADE_BAND == WINNER['rebalance_band'])",
      pt.NO_TRADE_BAND == be.WINNER.get("rebalance_band"), (pt.NO_TRADE_BAND, be.WINNER.get("rebalance_band")))


# ---- paper_trade CLI --submit: buys capped at buying power + sells, 1% cushion (mocked client, nothing sent)
class _Pos:
    def __init__(self, s, q):
        self.symbol, self.qty = s, q


class _Acct:
    equity, buying_power = 10_000.0, 1_000.0


class _Client:
    def get_all_positions(self):
        return [_Pos("OLD", 10)]

    def get_account(self):
        return _Acct()


plan = pd.DataFrame([{"Symbol": "OLD", "Side": "SELL", "Shares": 10, "Price": 50.0, "Est_Value": 500.0},
                     {"Symbol": "NEW", "Side": "BUY", "Shares": 30, "Price": 100.0, "Est_Value": 3000.0}],
                    columns=pt.ORDER_COLUMNS)
sent = {}
orig = (pt.paper_trading_client, pt.plan_orders, pt.submit_paper)
pt.paper_trading_client = lambda: _Client()
pt.plan_orders = lambda *a, **k: (plan.copy(), {"source": "provisional", "as_of": "2026-10-02", "strategy": "x",
                                               "invested": 1.0, "last_rebalance": "2026-10-02",
                                               "last_decision": "2026-10-02", "swaps": pd.DataFrame()}, plan)
pt.submit_paper = lambda orders, positions=None: sent.setdefault("o", orders)
try:
    with redirect_stdout(io.StringIO()):
        pt.main(["--submit"])
finally:
    pt.paper_trading_client, pt.plan_orders, pt.submit_paper = orig
o = sent.get("o")
buy = o[o["Side"] == "BUY"] if o is not None else pd.DataFrame()
check("CLI --submit: the buy is cut to fit $1,000 + $500 of sells, less 1% (14 shares of $100)",
      len(buy) == 1 and int(buy["Shares"].iloc[0]) == 14, None if o is None else o[["Symbol", "Side", "Shares"]].to_dict("records"))

# ---- watchdog: a 404 (model shut down) falls back to the next Gemini model
import pipeline_watchdog as wd

calls = []


def fake_complete(provider, model, key, prompt):
    calls.append(model)
    if model == "gemini-flash-latest":
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    return '{"diagnosis": "ok"}'


orig_c, orig_p = wd._llm_complete, wd._llm_provider
wd._llm_complete, wd._llm_provider = fake_complete, (lambda: ("gemini", "gemini-flash-latest", "k"))
try:
    with redirect_stdout(io.StringIO()):
        d = wd.diagnose_with_llm("main", "steps", "tail", {})
finally:
    wd._llm_complete, wd._llm_provider = orig_c, orig_p
check("watchdog: HTTP 404 -> tries gemini-2.5-flash and gets a diagnosis",
      calls == ["gemini-flash-latest", "gemini-2.5-flash"] and d == {"diagnosis": "ok"}, (calls, d))
check("watchdog: default Gemini model is the maintained alias (gemini-flash-latest)",
      wd.LLM_PROVIDERS[0][1] == "gemini-flash-latest")

# ---- run_all --fill-check: a second fill check exits while one is running (mocked phase, no broker)
import json
import time

import run_all as ra

with tempfile.TemporaryDirectory() as d:
    lock = os.path.join(d, ".fill_check.lock")
    ran = []
    orig_fc, orig_lock, orig_logs = ra._fill_check, ra.FILL_LOCK, ra.LOG_DIR
    ra._fill_check, ra.FILL_LOCK, ra.LOG_DIR = (lambda *a, **k: ran.append(1) or 0), lock, d   # run logs go to the temp dir
    try:
        with open(lock, "w") as f:
            json.dump({"pid": os.getpid(), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)   # a live run holds it
        with redirect_stdout(io.StringIO()):
            rc = ra.main(["--fill-check"])
        check("fill-check lock: held by a live run -> exits without checking", rc == 0 and ran == [], (rc, ran))
        os.remove(lock)
        with redirect_stdout(io.StringIO()):
            ra.main(["--fill-check"])
        check("fill-check lock: free -> the check runs once and the lock is released", ran == [1] and not os.path.exists(lock),
              (ran, os.path.exists(lock)))
    finally:
        ra._fill_check, ra.FILL_LOCK, ra.LOG_DIR = orig_fc, orig_lock, orig_logs

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
