"""Live-bot safety guards (mocks and temp files only: no broker, no network; the real run log / run_state are never
written):
  * backtest_engine WINNER["max_weight"]: each target weight is clipped to 20% after the vol weights and the regime halving;
    the extra stays in cash (19.8% live after the 99% round-down)
  * paper_trade.MAX_ORDER_PCT (backstop): a BUY worth more than 20% of equity is refused - not sent, not staged for 9 AM, one
    plain run-log row; sells are never capped; today's plans stay under it
  * run_all: a scheduled decision that missed its 2:30 PM slot and waits for the market gets ONE run-log warning row
    (then the launchd runs stay quiet; the watchdog still sees 'idle:' as the last line); --dry-run writes nothing
Run: python tests/run_tests.py  (or python tests/test_safety_guards.py)"""
import csv
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp()
os.environ["STOCK_ANALYSIS_RUN_LOG"] = os.path.join(tmp, "run_log.csv")
import pandas as pd

import fake_quotes
import paper_trade as pt
fake_quotes.install(pt)   # never a real quote
import run_all as ra

FAIL = []


def check(name, ok, detail=None):
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or detail is None else f"   -> {detail}"))
    if not ok:
        FAIL.append(name)


def t(s):
    return datetime.fromisoformat(s).replace(tzinfo=ra.CT)


# ------------------------------------------------------------------ 20% max weight per stock (engine)
import backtest_engine as be
check("max weight: WINNER says 20% and the rank args pass it on",
      be.WINNER["max_weight"] == 0.20 and be.winner_rank_args(None)["max_weight"] == 0.20)
d, cols = pd.bdate_range("2026-01-05", periods=2), [f"S{i}" for i in range(10)]
score = pd.DataFrame([[10.0 - i for i in range(10)]] * 2, index=d, columns=cols)
vol = pd.DataFrame([[0.005] + [0.02] * 9] * 2, index=d, columns=cols)   # S0: 1/4 the vol -> 4/13 = 30.8% unclipped
kw = dict(n=10, rebalance_days=pd.Series([True, False], index=d), sector_cap=1.0, regime_scale=0.5)
for on in (True, False):
    reg = pd.Series(on, index=d)
    raw = be.rank_targets(score, score.notna(), vol, regime=reg, **kw).iloc[0]
    cut = be.rank_targets(score, score.notna(), vol, regime=reg, max_weight=0.20, **kw).iloc[0]
    want = raw.clip(upper=0.20)
    check(f"max weight (regime {'on' if on else 'off'}): {raw['S0']:.2%} -> {cut['S0']:.2%}, the others unchanged, the extra in cash",
          (cut - want).abs().max() < 1e-15 and cut.sum() <= raw.sum() and cut.max() <= 0.20, (raw.round(4).tolist(), cut.round(4).tolist()))
live = be.live_weights(cut)
check("max weight: after the 99% round-down the biggest live weight is 19.8% and the total stays <= 99%",
      live.max() <= 0.198 and live.sum() <= 0.99, live.tolist())

# ------------------------------------------------------------------ order cap (pure backstop)
check("cap constant is 20% of equity", pt.MAX_ORDER_PCT == 0.20)
o = pd.DataFrame([{"Symbol": "OK", "Side": "BUY", "Shares": 19.0, "Price": 100.0},
                  {"Symbol": "BIG", "Side": "BUY", "Shares": 21.0, "Price": 100.0},
                  {"Symbol": "OUT", "Side": "SELL", "Shares": 50.0, "Price": 100.0}], columns=pt.ORDER_COLUMNS)
capped, refused = pt.apply_order_cap(o, 10000.0)
check("cap: a 19% buy passes, a 21% buy is refused, a 50% sell is never capped",
      list(capped["Side"]) == ["BUY", "SKIP (over order cap)", "SELL"] and list(refused["Symbol"]) == ["BIG"]
      and refused["Est_Value"].iloc[0] == 2100.0, capped.to_dict("records"))
check("cap: unusable equity refuses every buy (fail closed), sells untouched",
      list(pt.apply_order_cap(o, float("nan"))[0]["Side"]) == ["SKIP (over order cap)"] * 2 + ["SELL"])
pos = pt.read_positions_csv(os.path.join(ROOT, "my_positions.csv")) if os.path.exists(os.path.join(ROOT, "my_positions.csv")) else {}
for src in ("auto", "provisional", "current"):
    plan, _m, _t = pt.plan_orders(src, 58235.07, pos, fractional=True)
    check(f"cap: today's {src} plan is unchanged by it", pt.apply_order_cap(plan, 58235.07)[1].empty, plan)

# ------------------------------------------------------------------ order cap inside auto_trade (staged path, no broker)
pend, picks = os.path.join(tmp, "pending.json"), os.path.join(tmp, "picks.csv")
pd.DataFrame({"Symbol": ["AAA", "BIG"]}).to_csv(picks, index=False)
names = ("check_signal_freshness", "get_live_positions_and_equity", "plan_orders", "current_prices", "paper_trading_client",
         "PENDING_ORDERS_JSON", "PICKS_CSV", "log_event", "_today_ct")
saved, seen = {k: getattr(pt, k) for k in names}, []
plan = pd.DataFrame([{"Symbol": "AAA", "Side": "BUY", "Shares": 10.0, "Price": 100.0, "Est_Value": 1000.0},
                     {"Symbol": "BIG", "Side": "BUY", "Shares": 40.0, "Price": 100.0, "Est_Value": 4000.0}], columns=pt.ORDER_COLUMNS)
try:
    pt.check_signal_freshness = lambda **k: "2026-10-05"
    pt.get_live_positions_and_equity = lambda: ({}, 10000.0, 10000.0, 10000.0)
    pt.current_prices = lambda syms: {"AAA": 100.0, "BIG": 100.0}
    pt.plan_orders = lambda *a, **k: (plan.copy(), {"source": "provisional", "as_of": "2026-10-05"}, None)
    pt.paper_trading_client = lambda: (_ for _ in ()).throw(AssertionError("no broker client"))
    pt.PENDING_ORDERS_JSON, pt.PICKS_CSV = pend, picks
    pt.log_event = lambda *a, **k: seen.append(a)
    pt._today_ct = lambda: t("2026-10-05 14:43")
    with redirect_stdout(io.StringIO()):
        _o, _meta, res = pt.auto_trade(log_csv=None, decision=pd.Timestamp("2026-10-05"), session=True)
    staged = [x["symbol"] for x in json.load(open(pend))["orders"]]
    big = res[res["Symbol"] == "BIG"]
    check("auto_trade: the 40% buy is not staged / sent (not carried to 9 AM), the normal one is",
          staged == ["AAA"] and len(big) == 1 and big["Status"].iloc[0].startswith("SKIPPED: over the 20%"), (staged, res.to_dict("records")))
    cap_rows = [a for a in seen if "Order cap" in a[3]]
    check("auto_trade: one plain run-log warning row for the refused order",
          len(cap_rows) == 1 and cap_rows[0][:3] == ("Trade", "warning", "no") and "BIG 40 shares" in cap_rows[0][3], seen)
finally:
    for k, v in saved.items():
        setattr(pt, k, v)

# ------------------------------------------------------------------ missed scheduled decision: one run-log row
state_path = os.path.join(tmp, "run_state.json")
saved_ra = {k: getattr(ra, k) for k in ("STATE_FILE", "TRADE_LOCK", "LOG_DIR", "_pipeline", "RUN_LOG")}
try:
    ra.STATE_FILE, ra.TRADE_LOCK, ra.LOG_DIR = state_path, os.path.join(tmp, ".trade.lock"), os.path.join(tmp, "logs")
    ra.RUN_LOG = os.path.join(tmp, "run_log_missed.csv")
    ra._pipeline = lambda *a, **k: 0
    json.dump({"last_decision": "2026-09-30"}, open(state_path, "w"))          # Fri Oct 2 never ran
    with redirect_stdout(io.StringIO()):
        ra.main(["--trade", "--scheduled", "--dry-run", "--now", "2026-10-02 20:00"])
    check("missed decision: a --dry-run writes nothing", not os.path.exists(ra.RUN_LOG) and "last_missed_notice" not in json.load(open(state_path)))
    outs = []
    for when in ("2026-10-02 20:00", "2026-10-03 11:00", "2026-10-04 09:30"):             # Fri night, Sat, Sun: waiting
        with redirect_stdout(io.StringIO()) as out:
            ra.main(["--trade", "--scheduled", "--now", when])
        outs.append(out.getvalue().strip().splitlines()[-1])
    rows = list(csv.DictReader(open(ra.RUN_LOG)))
    check("missed decision: ONE warning row while it waits (no repeat on every launchd run)",
          len(rows) == 1 and rows[0]["run"] == "Missed decision" and rows[0]["status"] == "warning"
          and rows[0]["money_moved"] == "no" and "Fri Oct 02 full rebalance did not run at its 2:30 PM CT slot" in rows[0]["message"]
          and "Mon Oct 05 9:00 AM CT" in rows[0]["message"], rows)
    check("missed decision: every launchd run still ends with its 'idle:' line (the watchdog stays quiet)",
          all(x.startswith("idle:") for x in outs), outs)
    check("missed decision: remembered in run_state (last_missed_notice)", json.load(open(state_path)).get("last_missed_notice") == "2026-10-02")
    json.dump({"last_decision": "2026-10-02"}, open(state_path, "w"))          # ran on time: nothing to report
    os.remove(ra.RUN_LOG)
    with redirect_stdout(io.StringIO()):
        ra.main(["--trade", "--scheduled", "--now", "2026-10-03 11:00"])
    check("decision that ran: no warning row", not os.path.exists(ra.RUN_LOG))
finally:
    for k, v in saved_ra.items():
        setattr(ra, k, v)
print(f"\n{len(FAIL)} failed" if FAIL else "\nSAFETY GUARDS OK")
sys.exit(1 if FAIL else 0)
