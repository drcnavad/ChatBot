"""Missed-decision catch-up (2026-09-28): every decision runs exactly once - at its 3:15 PM CT slot or, if the Mac was
asleep/off, at the next regular session - and is superseded (skipped + alert) once the next decision slot arrives.

Mocks and temp files only - no network, no broker calls, no pop-ups, the real Reports/run_state.json is never written.
  * decision_gate: missed Fri -> Sat waits -> Mon 9:30 catch-up -> Mon 3:15 PM superseded; missed Mon -> Tue; missed
    Wed -> Thu/Fri; already done -> nothing; launchd retries limited; holidays (Thu Dec 24 rebalance, Labor Day) and the
    early close (Fri Nov 27).
  * the catch-up trades the missed decision's own picks (load_targets(decision)) and the freshness check accepts its
    files; a catch-up in regular hours stages every order with send_now at current prices (nothing sent by auto_trade).
  * run_all: a second run never trades a decision another run finished (re-check under the lock); an idle launchd run
    writes no log; update_state merges instead of overwriting; the watchdog stays quiet on idle runs.
Run: python tests/run_tests.py  (or python tests/test_catch_up.py)
"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd

import paper_trade as pt
import run_all as ra

CT = ra.CT
PASS, FAIL = [], []


def check(name, ok, detail=None):
    (PASS if ok else FAIL).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or detail is None else f"   -> {detail}"))


def t(s):
    return datetime.fromisoformat(s).replace(tzinfo=CT)


def gate(when, done, **kw):
    """(decision date or None, how) at CT time `when` with `done` as the last decision that ran."""
    D, how, why = ra.decision_gate(t(when), {"last_decision": done} if done else {}, **kw)
    return (D.date().isoformat() if D is not None else None), how


def superseded(when, done):
    s = ra.superseded_decision(t(when), {"last_decision": done})
    return s.date().isoformat() if s is not None else None


# ------------------------------------------------------------------ missed Friday rebalance (Oct 2)
check("Fri 10/2 3:40 PM: the rebalance runs in the evening", gate("2026-10-02 15:40", "2026-09-30") == ("2026-10-02", "evening"))
check("Fri 10/2 8 PM (extended hours over): waits for Monday", gate("2026-10-02 20:00", "2026-09-30") == ("2026-10-02", "wait"))
check("Sat 10/3: waits for the Monday open (no weekend orders)", gate("2026-10-03 11:00", "2026-09-30") == ("2026-10-02", "wait"))
check("Mon 10/5 8:45 AM: still waits (catch-up from 9:00 AM CT)", gate("2026-10-05 08:45", "2026-09-30") == ("2026-10-02", "wait"))
check("Mon 10/5 9:30 AM: caught up now with market orders", gate("2026-10-05 09:30", "2026-09-30") == ("2026-10-02", "session"))
check("Mon 10/5 2:50 PM: too close to the close and the next slot -> nothing", gate("2026-10-05 14:50", "2026-09-30")[1] is None)
check("Mon 10/5 3:15 PM: Monday's check runs instead ...", gate("2026-10-05 15:15", "2026-09-30") == ("2026-10-05", "evening"))
check("... and Friday is reported superseded (once)", superseded("2026-10-05 15:15", "2026-09-30") == "2026-10-02")
check("superseded is reported only once", ra.superseded_decision(t("2026-10-05 15:20"), {"last_decision": "2026-09-30",
                                                                    "last_superseded": "2026-10-02"}) is None)
check("nothing superseded when Friday ran", superseded("2026-10-05 15:15", "2026-10-02") is None)

# ------------------------------------------------------------------ missed Monday / Wednesday checks
check("missed Mon 9/28 -> Tue 9/29 10 AM catch-up", gate("2026-09-29 10:00", "2026-09-25") == ("2026-09-28", "session"))
check("missed Mon 9/28 -> Tue 8 PM waits for Wednesday", gate("2026-09-29 20:00", "2026-09-25") == ("2026-09-28", "wait"))
check("missed Mon 9/28 -> Wed 9/30 10 AM still catches up", gate("2026-09-30 10:00", "2026-09-25") == ("2026-09-28", "session"))
check("missed Mon 9/28 -> Wed 3:15 PM superseded, Wednesday runs",
      gate("2026-09-30 15:15", "2026-09-25") == ("2026-09-30", "evening") and superseded("2026-09-30 15:15", "2026-09-25") == "2026-09-28")
check("missed Wed 9/30 -> Thu 10/1 10 AM catch-up", gate("2026-10-01 10:00", "2026-09-28") == ("2026-09-30", "session"))
check("missed Wed 9/30 -> Fri 10/2 11 AM catch-up", gate("2026-10-02 11:00", "2026-09-28") == ("2026-09-30", "session"))
check("missed Wed 9/30 -> Fri 3:15 PM superseded", superseded("2026-10-02 15:15", "2026-09-28") == "2026-09-30")

# ------------------------------------------------------------------ never twice
for when in ("2026-10-02 16:00", "2026-10-02 20:00", "2026-10-03 11:00", "2026-10-05 09:30", "2026-10-05 15:14"):
    check(f"Friday done -> nothing at {when}", gate(when, "2026-10-02")[1] is None)
att = {"last_decision": "2026-09-30", "decision_attempts": {"decision": "2026-10-02", "n": 3, "at": "2026-10-05T10:00:00-05:00"}}
check("launchd: 3 failed attempts -> no more automatic retries", ra.decision_gate(t("2026-10-05 12:00"), att, scheduled=True)[1] is None)
check("... a manual run may still try", ra.decision_gate(t("2026-10-05 12:00"), att)[1] == "session")

# ------------------------------------------------------------------ holidays and early closes
check("Thu 12/24 rebalance (Christmas Friday) 3:20 PM: evening run", gate("2026-12-24 15:20", "2026-12-23") == ("2026-12-24", "evening"))
check("Thu 12/24 4:30 PM (early close: extended hours end 4 PM CT): waits", gate("2026-12-24 16:30", "2026-12-23") == ("2026-12-24", "wait"))
check("Fri 12/25 holiday: waits", gate("2026-12-25 10:00", "2026-12-23") == ("2026-12-24", "wait"))
check("Mon 12/28 10 AM: Thursday's rebalance caught up", gate("2026-12-28 10:00", "2026-12-23") == ("2026-12-24", "session"))
check("Mon 12/28 3:15 PM: superseded by Monday's check", superseded("2026-12-28 15:15", "2026-12-23") == "2026-12-24")
check("Fri 9/4 rebalance, Mon 9/7 Labor Day: Mon waits", gate("2026-09-07 10:00", "2026-09-02") == ("2026-09-04", "wait"))
check("... Tue 9/8 10 AM caught up", gate("2026-09-08 10:00", "2026-09-02") == ("2026-09-04", "session"))
check("... Tue 9/8 3:15 PM superseded by the Tuesday check",
      gate("2026-09-08 15:15", "2026-09-02") == ("2026-09-08", "evening") and superseded("2026-09-08 15:15", "2026-09-02") == "2026-09-04")
check("missed Wed 11/25, Thanksgiving Thu: waits for Fri 11/27", gate("2026-11-26 10:00", "2026-11-23") == ("2026-11-25", "wait"))
check("Fri 11/27 (1 PM ET close) 11:30 AM: caught up", gate("2026-11-27 11:30", "2026-11-23") == ("2026-11-25", "session"))
check("Fri 11/27 11:50 AM (within 15 min of the close): nothing", gate("2026-11-27 11:50", "2026-11-23")[1] is None)
check("Fri 11/27 rebalance 3:40 PM: evening (extended hours to 4 PM CT)", gate("2026-11-27 15:40", "2026-11-25") == ("2026-11-27", "evening"))
check("Fri 11/27 rebalance 4:10 PM: waits for Mon 11/30", gate("2026-11-27 16:10", "2026-11-25") == ("2026-11-27", "wait"))

# ------------------------------------------------------------------ mode of a catch-up run
D = pd.Timestamp("2026-10-02")
check("catch-up: full update when none ran since the decision's slot",
      ra.choose_mode(t("2026-10-05 09:30"), {"last_full_at": "2026-09-30T15:40:00-05:00"}, decision=D)[0] == "full")
check("catch-up: quick when the full update already ran after the slot",
      ra.choose_mode(t("2026-10-05 09:30"), {"last_full_at": "2026-10-02T15:40:00-05:00"}, decision=D)[0] == "quick")

# ------------------------------------------------------------------ the missed decision's own picks and files
tmp = tempfile.mkdtemp()
picks, mid, chg = (os.path.join(tmp, f) for f in ("picks.csv", "mid.csv", "chg.csv"))


def write(as_of, last_reb, mid_rows=(), chg_date=None):
    pd.DataFrame([{"As_Of": as_of, "Last_Rebalance": last_reb, "Last_Decision": last_reb, "Strategy": "x", "Symbol": s,
                   "Strategy_Weight": 0.1, "Provisional_Weight": 0.1, "Close": 100.0} for s in ("AAA", "BBB")]).to_csv(picks, index=False)
    pd.DataFrame(list(mid_rows), columns=["Event", "Event_Date", "Action", "Sell", "Buy", "Weight_%"]).to_csv(mid, index=False)
    pd.DataFrame([{"Date": chg_date or as_of, "Symbol": "AAA", "Status": "hold"}]).to_csv(chg, index=False)


write("2026-10-02", "2026-10-02")
check("Mon morning catch-up of Friday: its rebalance weights", pt.load_targets("auto", picks, mid, decision="2026-10-02")[1]["source"] == "provisional")
swap = ("mid-week check", "2026-09-28", "SWAP", "BBB", "CCC", 10.0)
write("2026-09-29", "2026-09-25", [swap])
check("Tue catch-up of Monday: Monday's swap", pt.load_targets("auto", picks, mid, decision="2026-09-28")[1]["source"] == "midweek")
write("2026-09-29", "2026-09-25", [("mid-week check", "2026-09-28", "NO SWAP", None, None, None)])
check("Tue catch-up of a quiet Monday: hold (nothing to trade)",
      pt.load_targets("auto", picks, mid, decision="2026-09-28")[1]["source"] == "hold")
write("2026-09-30", "2026-09-25", [swap])
check("Wed-morning catch-up of Monday (data as of Tue): still Monday's swap",
      len(pt.load_targets("auto", picks, mid, decision="2026-09-28")[1]["swaps"]) == 1)
write("2026-10-05", "2026-10-02")
try:
    pt.load_targets("auto", picks, mid, decision="2026-10-02")
    check("a rebalance is never traded from later data", False)
except ValueError:
    check("a rebalance is never traded from later data", True)


def fresh(as_of, decision, now, chg_date=None):
    write(as_of, as_of, chg_date=chg_date)
    try:
        pt.check_signal_freshness(picks, chg, decision=decision, now=t(now))
        return True
    except ValueError:
        return False


check("freshness: Mon 10 AM accepts Friday's files for Friday's decision", fresh("2026-10-02", "2026-10-02", "2026-10-05 10:00"))
check("freshness: Mon 10 AM refuses Thursday's files", not fresh("2026-10-01", "2026-10-02", "2026-10-05 10:00"))
check("freshness: Wed 10 AM needs Tuesday's files for Monday's decision", fresh("2026-09-29", "2026-09-28", "2026-09-30 10:00")
      and not fresh("2026-09-28", "2026-09-28", "2026-09-30 10:00"))
check("freshness: after 4:30 PM ET the evening needs today's files", not fresh("2026-10-02", "2026-10-02", "2026-10-05 16:00"))
check("freshness: a changes file older than the picks is refused",
      not fresh("2026-10-02", "2026-10-02", "2026-10-05 10:00", chg_date="2026-10-01"))

# ------------------------------------------------------------------ a catch-up in regular hours stages everything (send_now)
pend = os.path.join(tmp, "live_pending_orders.json")
saved = {k: getattr(pt, k) for k in ("check_signal_freshness", "get_live_positions_and_equity", "plan_orders", "current_prices",
                                     "paper_trading_client", "PENDING_ORDERS_JSON", "PICKS_CSV", "log_event", "_today_ct")}
seen = {}
plan = pd.DataFrame([{"Symbol": "OLD", "Side": "SELL", "Shares": 10.0, "Price": 50.0, "Est_Value": 500.0},
                     {"Symbol": "AAA", "Side": "BUY", "Shares": 4.5, "Price": 110.0, "Est_Value": 495.0}], columns=pt.ORDER_COLUMNS)
try:
    write("2026-10-02", "2026-10-02")
    pt.check_signal_freshness = lambda **k: seen.setdefault("fresh", k) and "2026-10-02"
    pt.get_live_positions_and_equity = lambda: ({"OLD": 10.0}, 10000.0, 5000.0, 5000.0)
    pt.current_prices = lambda syms: seen.setdefault("quotes", sorted(syms)) and {"AAA": 110.0}
    pt.plan_orders = lambda *a, **k: (seen.setdefault("plan", k), (plan.copy(), {"source": "provisional", "as_of": "2026-10-02"}, None))[1]
    pt.paper_trading_client = lambda: (_ for _ in ()).throw(AssertionError("no broker client in a session catch-up"))
    pt.PENDING_ORDERS_JSON, pt.PICKS_CSV = pend, picks
    pt.log_event = lambda *a, **k: seen.setdefault("notify", a)
    pt._today_ct = lambda: t("2026-10-05 10:05")
    with redirect_stdout(io.StringIO()):
        _o, meta, res = pt.auto_trade(log_csv=None, decision=pd.Timestamp("2026-10-02"), session=True)
    data = json.load(open(pend))
    check("session catch-up: freshness checked for the missed decision", seen["fresh"].get("decision") == pd.Timestamp("2026-10-02"))
    check("session catch-up: sized at current prices, the decision's picks", seen["plan"].get("live_prices") == {"AAA": 110.0}
          and seen["plan"].get("decision") == pd.Timestamp("2026-10-02"))
    check("session catch-up: every order STAGED (catch-up), nothing sent by auto_trade",
          res["Status"].str.contains("catch-up").all() and len(res) == 2, res.to_dict("records"))
    check("session catch-up: pending file marked send_now, rows carry the decision and time",
          data.get("send_now") is True and all(o["decision"] == "2026-10-02" and o["recorded_at"].startswith("2026-10-05T10:05")
                                                 for o in data["orders"]), data)
    check("session catch-up: run-log row says it is traded now", "Catch-up trades" in (seen.get("notify") or ("",) * 4)[3], seen.get("notify"))
    check("fill gate: the staged catch-up orders may go at once",
          ra.fill_check_allowed(t("2026-10-05 10:06"), pending_path=pend)[0])
finally:
    for k, v in saved.items():
        setattr(pt, k, v)

# ------------------------------------------------------------------ run_all: never twice, idle runs are silent
state_path = os.path.join(tmp, "run_state.json")
saved_ra = {k: getattr(ra, k) for k in ("STATE_FILE", "TRADE_LOCK", "LOG_DIR", "_pipeline", "load_state")}
calls = []
try:
    ra.STATE_FILE, ra.TRADE_LOCK, ra.LOG_DIR = state_path, os.path.join(tmp, ".trade.lock"), os.path.join(tmp, "logs")
    ra._pipeline = lambda *a, **k: calls.append(a[-2:]) or 0
    json.dump({"last_decision": "2026-09-30"}, open(state_path, "w"))
    with redirect_stdout(io.StringIO()) as out:
        rc = ra.main(["--trade", "--scheduled", "--now", "2026-10-03 11:00"])
    check("idle launchd run (Saturday): one 'idle:' line, no run log, no lock, nothing runs",
          rc == 0 and out.getvalue().startswith("idle:") and not os.path.exists(ra.LOG_DIR) and not calls, out.getvalue())
    real_load = saved_ra["load_state"]
    seq = [{"last_decision": "2026-09-30"}, {"last_decision": "2026-10-02"}]   # another run finishes Friday meanwhile
    ra.load_state = lambda path=None: seq.pop(0) if seq else real_load(path)
    with redirect_stdout(io.StringIO()):
        rc = ra.main(["--trade", "--now", "2026-10-02 16:00"])
    check("no double run: the decision is re-checked under the lock (lock released)",
          rc == 0 and not calls and not os.path.exists(ra.TRADE_LOCK))
    ra.load_state = real_load
    with redirect_stdout(io.StringIO()):
        ra.main(["--trade", "--scheduled", "--now", "2026-10-05 09:30"])
    ra.release_trade_lock()                               # the mocked _pipeline does not release it
    st = json.load(open(state_path))
    check("Mon 9:30 AM launchd run: Friday's catch-up starts (session) and the attempt is counted",
          calls and calls[-1][0] == pd.Timestamp("2026-10-02") and calls[-1][1] == "session"
          and st["decision_attempts"]["n"] == 1, (calls, st))
finally:
    for k, v in saved_ra.items():
        setattr(ra, k, v)

# update_state merges: a stale in-memory copy can't wipe another job's key
p = os.path.join(tmp, "merge.json")
json.dump({"last_full_at": "2026-10-02T15:40:00-05:00"}, open(p, "w"))
ra.update_state(p, last_decision="2026-10-02")
ra.update_state(p, last_fill_check_at="2026-10-05T09:01:00-05:00")
check("update_state merges keys from both jobs", ra.load_state(p) == {"last_full_at": "2026-10-02T15:40:00-05:00",
                                                                      "last_decision": "2026-10-02",
                                                                      "last_fill_check_at": "2026-10-05T09:01:00-05:00"})

# watchdog: an idle scheduled run adds no lines of its own
import pipeline_watchdog as wd

orig = wd.run_pipeline
wd.run_pipeline = lambda argv: (0, "idle: Sat Oct 03 11:00 AM CT - nothing due\n")
try:
    with redirect_stdout(io.StringIO()) as out:
        wd.main(["--trade", "--scheduled"])
    check("watchdog: idle scheduled run -> no watchdog lines", out.getvalue() == "", out.getvalue())
finally:
    wd.run_pipeline = orig

# a failed --trade: one plain run-log row (what happened, money moved?, what next) and the watchdog adds nothing
import paper_trade as _pt
_saved_rec = _pt._todays_recorded_orders
_pt._todays_recorded_orders = lambda *a, **k: set()
try:
    D0 = pd.Timestamp("2026-09-30")
    t, m = ra.trade_failure_text(D0, ["trade"], "can't reach Alpaca", {"decision_attempts": {"decision": "2026-09-30", "n": 1}}, True)
    check("trade failed: says no money moved and that it retries (try 2 of 3)",
          "No orders went out, no money moved" in m and "try 2 of 3" in m and t == "no", (t, m))
    t, m = ra.trade_failure_text(D0, ["trade"], "x", {"decision_attempts": {"decision": "2026-09-30", "n": 3}}, True)
    check("trade failed 3rd time: says no more automatic tries + the command", "No more automatic tries" in m
          and "run_all.py --trade" in m, m)
    t, m = ra.trade_failure_text(D0, ["main", "trade_skipped"], None, {}, True)
    check("pipeline broke before the trade: 'not placed ... No money moved'", "not placed" in m and "No money moved" in m, m)
    _pt._todays_recorded_orders = lambda *a, **k: {("AMD", "BUY"), ("MU", "SELL")}
    t, m = ra.trade_failure_text(D0, ["trade"], "network", {}, False)
    check("trade failed after 2 orders went out: says so, never sent twice, money_moved yes", "2 order(s) went out" in m
          and "never sent twice" in m and t == "yes", (t, m))
finally:
    _pt._todays_recorded_orders = _saved_rec

wd.run_pipeline = lambda argv: (ra.EXIT_REPORTED, "Trade not placed ...\n")
_saved_log = ra.log_event
ra.log_event = lambda *a, **k: NOTIFIED.append(a)
NOTIFIED = []
try:
    with redirect_stdout(io.StringIO()):
        rc = wd.main(["--trade", "--scheduled"])
    check("watchdog: a failure run_all already reported -> exit 1, no retry, no second row", rc == 1 and not NOTIFIED, NOTIFIED)
finally:
    wd.run_pipeline, ra.log_event = orig, _saved_log

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
