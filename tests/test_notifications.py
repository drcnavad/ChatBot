"""Mocked tests (zero broker calls, zero network, zero real pop-ups) for the trade alerts.

Covers: pop-ups are short (title <= 40, body <= 200) and plain (no 'pp', no 'DRIFT');
STOCK_ANALYSIS_NO_POPUPS=1 skips the macOS pop-up but still prints; the evening trade alert
names symbols, share counts and dollars; the morning check sends no alert when nothing is
pending; holdings are compared only for the symbols just traded (hand-bought stocks never
false-alarm); 'couldn't check' is its own alert, not drift.

Run: cd <folder> && python3 tests/test_notifications.py
"""
import io
import json
import os
import sys
import tempfile
import types
from contextlib import redirect_stdout
from enum import Enum
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["STOCK_ANALYSIS_NO_POPUPS"] = "1"   # belt and braces: never a real pop-up from this file


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class _E(Enum):
    ALL = "all"
    OPEN = "open"
    DESC = "desc"
    DAY = "day"


class _Req:
    def __init__(self, **kw):
        self.__dict__.update(kw)


mods = {n: types.ModuleType(n) for n in ("alpaca", "alpaca.trading", "alpaca.trading.client",
                                          "alpaca.trading.enums", "alpaca.trading.requests", "alpaca.common",
                                          "alpaca.common.enums", "dotenv")}
mods["alpaca.trading.enums"].__dict__.update(OrderSide=OrderSide, TimeInForce=_E, QueryOrderStatus=_E)
mods["alpaca.common.enums"].Sort = _E
mods["alpaca.trading.requests"].__dict__.update(MarketOrderRequest=_Req, LimitOrderRequest=_Req,
                                                GetOrdersRequest=_Req)
mods["alpaca.trading.client"].TradingClient = type("TradingClient", (), {})
mods["dotenv"].load_dotenv = lambda *a, **k: None
sys.modules.update(mods)

import pandas as pd
import paper_trade as pt
import run_all

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""))


NOTES = []
_real_notify = pt._notify


def _record(title, message, details=None):
    NOTES.append((title, message, details))


def plain_and_short(t, m):
    return len(t) <= 40 and len(m) <= 200 and " pp" not in m and "DRIFT" not in (t + m)


# ------------------------------------------------------------------ pop-up switch + length limits
with mock.patch("subprocess.run") as sp, redirect_stdout(io.StringIO()) as out:
    os.environ["STOCK_ANALYSIS_NO_POPUPS"] = "1"
    _real_notify("T", "M", details="full detail")
check("NO_POPUPS=1: no osascript call", not sp.called)
check("NO_POPUPS=1: alert and details still printed to the log", "ALERT: T" in out.getvalue()
      and "Details: full detail" in out.getvalue())

with mock.patch("subprocess.run") as sp, redirect_stdout(io.StringIO()):
    os.environ["STOCK_ANALYSIS_NO_POPUPS"] = ""
    _real_notify("x" * 80, "y" * 500)
    os.environ["STOCK_ANALYSIS_NO_POPUPS"] = "1"
script = sp.call_args[0][0][2] if sp.called else ""
check("pop-up on: osascript called once (mocked)", sp.call_count == 1)
check("pop-up title cut to 40 chars", 'title "' + "x" * 39 + "\u2026" + '"' in script)
check("pop-up body cut to 200 chars", '"' + "y" * 199 + "\u2026" + '"' in script)

with mock.patch("subprocess.run") as sp:
    os.environ["STOCK_ANALYSIS_NO_POPUPS"] = "1"
    run_all._notify("T", "M")
check("run_all NO_POPUPS=1: no osascript call", not sp.called)
with mock.patch("subprocess.run") as sp:
    os.environ["STOCK_ANALYSIS_NO_POPUPS"] = ""
    run_all._notify("z" * 80, "w" * 500, details="d")
    os.environ["STOCK_ANALYSIS_NO_POPUPS"] = "1"
script = sp.call_args[0][0][2] if sp.called else ""
check("run_all pop-up clipped to 40/200", 'title "' + "z" * 39 + "\u2026" in script and "w" * 200 not in script)

check("share counts: 2 decimals, bad values shown as '?'",
      [pt._fmt_shares(x) for x in (4, 1.054222297, 0.82, float("nan"), "abc", None)] == ["4", "1.05", "0.82", "?", "?", "?"])

# ------------------------------------------------------------------ evening trade alert
sells = [("ENPH", 4), ("FIG", 6), ("HIMS", 15), ("IONQ", 5), ("ORCL", 1.054222297), ("UMAC", 7)]
buys = [("ANET", 3), ("BIIB", 4), ("FTNT", 4), ("MRK", 6), ("PANW", 1), ("RBRK", 4), ("SMCI", 9),
        ("TEM", 4), ("TWLO", 1)]
rows = [(s, "SELL", q, "id", "submitted (ext-hours limit)") for s, q in sells] + \
       [(s, "BUY", q, "id", "submitted (ext-hours limit)") for s, q in buys] + \
       [("AMD", "BUY", 0.82, None, "STAGED for the 9 AM check")]
res = pd.DataFrame(rows, columns=["Symbol", "Side", "Shares", "Order_ID", "Status"])
prices = {"ENPH": 40, "FIG": 30, "HIMS": 50, "IONQ": 45, "ORCL": 280, "UMAC": 12, "ANET": 140, "BIIB": 150,
          "FTNT": 100, "MRK": 90, "PANW": 190, "RBRK": 80, "SMCI": 45, "TEM": 70, "TWLO": 120, "AMD": 160}
t, m, d = pt._trade_notice(res, prices)
print("   sample:", t, "|", m)
check("trade alert: short and plain", plain_and_short(t, m), (len(t), len(m)))
check("trade alert: counts in title", t == "Trades sent: 6 sell, 10 buy", t)
check("trade alert: names symbols with share counts", "ENPH 4" in m and "ANET 3" in m, m)
check("trade alert: dollar estimate", "~$" in m, m)
check("trade alert: says what to do and where to look", "Nothing to do" in m and "live_orders_log" in m, m)
check("trade alert: every order listed in the log details", all(s in d for s, _ in sells + buys) and "AMD" in d)
q = res[res["Symbol"] == "AMD"]
t, m, _ = pt._trade_notice(q, prices)
check("queued-only alert says nothing was sent", t.startswith("Trades queued") and "no money moved" in m, (t, m))

# ------------------------------------------------------------------ morning check: nothing pending
pt._notify = _record
NOTES.clear()
missing = os.path.join(tempfile.mkdtemp(), "live_pending_orders.json")
r = pt.complete_unfilled_orders(pending_path=missing, log_csv=None, dry_run=False)
check("nothing pending: no alert at all", r.empty and not NOTES, NOTES)
empty = os.path.join(tempfile.mkdtemp(), "live_pending_orders.json")
json.dump({"evening_date": "2026-10-02", "orders": []}, open(empty, "w"))
r = pt.complete_unfilled_orders(pending_path=empty, log_csv=None, dry_run=False)
check("empty order list: no alert at all", r.empty and not NOTES, NOTES)

# ------------------------------------------------------------------ holdings check (reconcile)
targets = pd.DataFrame({"Symbol": ["ANET", "AMD"], "Weight": [0.05, 0.05], "Price": [100.0, 100.0]})
manual = {"ENPH": 4, "FIG": 6, "HIMS": 15, "IONQ": 5, "ORCL": 1.054222297, "UMAC": 7}


def patch_account(held):
    pt.load_targets = lambda src="auto": (targets, {"source": "provisional"})
    pt.get_live_positions_and_equity = lambda: (dict(held), 10000.0, 5000.0, 5000.0)
    pt.latest_prices = lambda syms: {s: 50.0 for s in syms}


patch_account({**manual, "ANET": 5})
rep, ok = pt.reconcile_positions("auto", symbols={"ANET"})
check("reconcile: only traded symbols checked (hand-bought stocks ignored)",
      ok is True and list(rep["Symbol"]) == ["ANET"], rep.to_string())
rep, ok = pt.reconcile_positions("auto")
check("reconcile without symbols= still sees everything (the old false alarm)", ok is False)
pt.get_live_positions_and_equity = lambda: (_ for _ in ()).throw(ConnectionError("down"))
rep, ok = pt.reconcile_positions("auto", symbols={"ANET"})
check("reconcile: unreadable account -> None (not drift)", ok is None and rep.empty)


class Broker:
    def __init__(self, held):
        self.held = dict(held)

    def get_clock(self): return types.SimpleNamespace(is_open=True)
    def get_account(self): return types.SimpleNamespace(buying_power="5000", equity="10000", cash="5000")
    def get_all_positions(self): return [types.SimpleNamespace(symbol=s, qty=str(q)) for s, q in self.held.items()]
    def get_order(self, oid): return self.orders()[0]

    def orders(self):
        return [types.SimpleNamespace(id="e1", symbol="ANET", side="BUY", qty=5, status="filled", filled_qty=5,
                                      client_order_id="live-20261002-BUY-ANET-5-1")]

    def get_orders(self, req=None): return self.orders()
    def submit_order(self, req): raise AssertionError("no order may be sent in this test")
    def cancel_order(self, oid): raise AssertionError("no cancel in this test")


def morning(held, reconcile_ok=True):
    patch_account(held)
    if not reconcile_ok:
        pt.get_live_positions_and_equity = lambda: (_ for _ in ()).throw(ConnectionError("down"))
    pt.paper_trading_client = lambda: Broker(held)
    p = os.path.join(tempfile.mkdtemp(), "live_pending_orders.json")
    json.dump({"evening_date": "2026-10-02", "target_source": "auto", "orders": [
        {"symbol": "ANET", "side": "BUY", "qty": 5, "order_qty": 5, "limit_price": 100.0, "order_id": "e1"}]},
        open(p, "w"))
    NOTES.clear()
    with redirect_stdout(io.StringIO()):
        pt.complete_unfilled_orders(pending_path=p, log_csv=None, dry_run=False)
    return [n[0] for n in NOTES]


titles = morning({**manual, "ANET": 5})
check("filled + on target, with hand-bought stocks held: only the 'filled' alert",
      titles == ["Fill check: orders filled"], titles)
check("'filled' alert names the symbol and says money moved",
      "buy ANET 5" in NOTES[0][1] and "money moved" in NOTES[0][1] and "Nothing to do" in NOTES[0][1], NOTES)
titles = morning({**manual, "ANET": 2})
drift = [n for n in NOTES if "off target" in n[0]]
check("traded symbol really off target: alert fires", len(drift) == 1 and drift[0][0].startswith("ANET"), titles)
check("off-target alert shows the percentages, not 'pp'", drift and "2.0%" in drift[0][1] and "5.0%" in drift[0][1]
      and " pp" not in drift[0][1], drift)
titles = morning({**manual, "ANET": 5}, reconcile_ok=False)
check("couldn't read holdings: its own wording, no drift alert",
      "Couldn't double-check holdings" in titles and not any("off target" in t for t in titles), titles)

# ------------------------------------------------------------------ run_all start / finish alerts (plain words)
from datetime import datetime
from zoneinfo import ZoneInfo

CT = ZoneInfo("America/Chicago")
wed, fri = pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-02")
check("labels: Wed check / Fri rebalance / catch-up / data refresh",
      [run_all.run_label(wed, "evening"), run_all.run_label(fri, "evening"), run_all.run_label(fri, "session"),
       run_all.run_label(None, None)] == ["Wed check", "Fri rebalance", "Catch-up: Fri Oct 2 rebalance", "Data refresh"])
ran7 = [(n, True, 1.0) for n in "abcdefg"]
rows = lambda *r: pd.DataFrame(list(r), columns=["Symbol", "Side", "Shares", "Order_ID", "Status"])
end = datetime(2026, 9, 30, 15, 31, tzinfo=CT)
t, m = run_all.finished_text("Wed check", end, ran7, [], True, "evening", rows())
check("Wed, no swap: done + all OK + no money moved + next + nothing to do",
      t == "Wed check done" and m == ("Wed check done 3:31 PM. All 7 steps OK. No trades needed - no orders, no money "
                                      "moved. Next: Fri Oct 2 rebalance at 3:15 PM. Nothing to do."), m)
sent = rows(("ENPH", "SELL", 4, "1", "submitted (accepted)"), ("FIG", "SELL", 6, "2", "submitted (new)"),
            ("AMD", "BUY", 3, "3", "submitted (new) + 0.58 fractional rest next morning"),
            ("TEM", "BUY", 0.8, None, "STAGED for morning market (<1 whole share after hours)"))
t, m = run_all.finished_text("Fri rebalance", datetime(2026, 10, 2, 15, 32, tzinfo=CT), ran7, [], True, "evening", sent)
check("Fri, orders sent: counts, money moves as they fill, 9 AM remainders, next = Mon check",
      "Orders sent: 2 sells, 2 buys - money moves as they fill; small remainders go out at 9 AM." in m
      and "Next: Mon Oct 5 check at 3:15 PM." in m and m.endswith("Nothing to do."), m)
t, m = run_all.finished_text("Fri rebalance", datetime(2026, 10, 2, 19, 5, tzinfo=CT), ran7, [], True, "evening",
                             rows(("AMD", "BUY", 3, None, "STAGED for morning market (past 7 PM CT - not submitted)")))
check("past 7 PM: queued for 9 AM, no money moved yet", "Orders queued for 9 AM: 1 buy - no money moved yet." in m, m)
done = rows(("ENPH", "SELL", 4, "m1", "COMPLETED via market (staged, filled 0/4)"),
            ("AMD", "BUY", 3.58, "m2", "COMPLETED via market (staged, filled 0/3.58)"))
t, m = run_all.finished_text("Catch-up: Fri Oct 2 rebalance", datetime(2026, 10, 5, 9, 31, tzinfo=CT), ran7, [], True,
                             "session", rows(("AMD", "BUY", 3.58, None, "STAGED ... (catch-up)")), done)
check("catch-up: sent at market now, counts, money moves", t == "Catch-up: Fri Oct 2 rebalance done"
      and "Catch-up orders sent at market: 1 sell, 1 buy - money moves as they fill." in m, m)
t, m = run_all.finished_text("Data refresh", end, [("main", True, 1), ("validate", True, 1)], [], False, None, None)
check("refresh only: says no trades, no money moved", "Data refresh only - no trades, no money moved." in m, m)
ran_bad = ran7[:6] + [("sentiment", False, 1.0)]
t, m = run_all.finished_text("Wed check", end, ran_bad, ["sentiment"], True, "evening", rows())
check("optional step failed: named, last good data used", "6 of 7 steps OK (sentiment failed - last good data used)." in m, m)
for args in [("Catch-up: Wed Sep 30 check", datetime(2026, 10, 1, 9, 5, tzinfo=CT), ran_bad,
              ["sentiment", "earnings", "fundamentals"], True, "session", rows(), done)]:
    t, m = run_all.finished_text(*args)
    check("long finish text still fits a pop-up and keeps 'Nothing to do'", len(m) <= 200 and len(t) <= 40
          and m.endswith("Nothing to do."), (len(m), m))

with mock.patch("subprocess.run", return_value=types.SimpleNamespace(returncode=1, stderr="execution error: -1743")), \
        mock.patch.object(run_all.logging, "warning") as warn:
    os.environ["STOCK_ANALYSIS_NO_POPUPS"] = ""
    run_all._notify("T", "M")
    os.environ["STOCK_ANALYSIS_NO_POPUPS"] = "1"
check("osascript error is logged, not swallowed", warn.called and "-1743" in str(warn.call_args), warn.call_args)
with mock.patch("subprocess.run", return_value=types.SimpleNamespace(returncode=1, stderr="boom")), \
        redirect_stdout(io.StringIO()) as out:
    os.environ["STOCK_ANALYSIS_NO_POPUPS"] = ""
    _real_notify("T", "M")
    os.environ["STOCK_ANALYSIS_NO_POPUPS"] = "1"
check("paper_trade: osascript error is printed to the log", "notification not shown" in out.getvalue(), out.getvalue())

bad = [(t, m) for t, m, _ in NOTES if not plain_and_short(t, m)]
check("every alert in this file is short and plain", not bad, bad)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
