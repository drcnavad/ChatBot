"""Mocked tests (zero broker calls, zero network) for the run log, Reports/run_log.csv (it replaced the pop-ups).

Covers: log_event writes one plain row (time_ct, run, status, money_moved, message, details), keeps the newest 1000 rows
and never raises; the trade row names symbols, share counts and dollars; the fill check writes nothing when nothing
is pending; holdings are compared only for the symbols just traded (hand-bought stocks never false-alarm); the
finished row says which run, whether all steps were OK, whether money moved, what's next and whether to act.

Run: cd <folder> && python3 tests/test_run_log.py
"""
import csv
import io
import json
import os
import sys
import tempfile
import types
from contextlib import redirect_stdout
from datetime import datetime
from enum import Enum
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["STOCK_ANALYSIS_RUN_LOG"] = os.path.join(tempfile.mkdtemp(), "run_log.csv")   # never the real one

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
import fake_quotes
fake_quotes.install(pt)   # never a real quote

PASS, FAIL = [], []

def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""))

# ------------------------------------------------------------------ log_event: the CSV itself
path = run_all.RUN_LOG
with redirect_stdout(io.StringIO()) as out:
    run_all.log_event("Wed check", "ok", "no", "Wed check done 3:31 PM.\n  All 7 steps OK.", details="")
rows = list(csv.reader(open(path)))
check("run log: header + one row with the 6 columns", rows[0] == run_all.RUN_LOG_COLUMNS and len(rows) == 2
      and rows[1][1:5] == ["Wed check", "ok", "no", "Wed check done 3:31 PM. All 7 steps OK."], rows)
check("run log: time_ct looks like 2026-09-30 15:31:00", len(rows[1][0]) == 19 and rows[1][0][4] == "-", rows[1][0])
check("run log: the row is also printed to the run log", "RUN LOG: Wed check | ok | money moved: no" in out.getvalue())
with redirect_stdout(io.StringIO()):
    for i in range(1150):
        run_all.log_event("Fill check", "ok", "yes", f"row {i}")
rows = list(csv.reader(open(path)))
check("run log: trimmed to the newest 1000 (+ at most 100 before the next trim), header kept",
      rows[0] == run_all.RUN_LOG_COLUMNS and run_all.RUN_LOG_KEEP < len(rows) - 1 <= run_all.RUN_LOG_KEEP + 100
      and rows[-1][4] == "row 1149", len(rows))
saved = run_all.RUN_LOG
run_all.RUN_LOG = os.path.join(tempfile.mkdtemp(), "missing_dir", "run_log.csv")
try:
    with redirect_stdout(io.StringIO()) as out:
        run_all.log_event("Trade", "failed", "no", "x")
    check("run log: an unwritable file never raises (a full disk must not stop a trade)",
          "run log not written" in out.getvalue())
finally:
    run_all.RUN_LOG = saved

ROWS = []
pt.log_event = lambda run, status, moved, message, details="": ROWS.append((run, status, moved, message, details))

check("share counts: 2 decimals, bad values shown as '?'",
      [pt._fmt_shares(x) for x in (4, 1.054222297, 0.82, float("nan"), "abc", None)] == ["4", "1.05", "0.82", "?", "?", "?"])

# ------------------------------------------------------------------ evening trade row
sells = [("ENPH", 4), ("FIG", 6), ("HIMS", 15), ("IONQ", 5), ("ORCL", 1.054222297), ("UMAC", 7)]
buys = [("ANET", 3), ("BIIB", 4), ("FTNT", 4), ("MRK", 6), ("PANW", 1), ("RBRK", 4), ("SMCI", 9),
        ("TEM", 4), ("TWLO", 1)]
res = pd.DataFrame([(s, "SELL", q, "id", "submitted (ext-hours limit)") for s, q in sells] +
                   [(s, "BUY", q, "id", "submitted (ext-hours limit)") for s, q in buys] +
                   [("AMD", "BUY", 0.82, None, "STAGED for the 9 AM check")],
                   columns=["Symbol", "Side", "Shares", "Order_ID", "Status"])
prices = {"ENPH": 40, "FIG": 30, "HIMS": 50, "IONQ": 45, "ORCL": 280, "UMAC": 12, "ANET": 140, "BIIB": 150,
          "FTNT": 100, "MRK": 90, "PANW": 190, "RBRK": 80, "SMCI": 45, "TEM": 70, "TWLO": 120, "AMD": 160}
moved, m, d = pt._trade_notice(res, prices)
print("   sample:", moved, "|", m)
check("trade row: money moved, counts first", moved == "yes" and m.startswith("Trades sent: 6 sell, 10 buy."), m)
check("trade row: names every symbol with share counts", all(f"{s} " in m for s, _ in sells + buys), m)
check("trade row: dollar estimate", "~$" in m, m)
check("trade row: says what to do and where to look", "Nothing to do" in m and "live_orders_log" in m, m)
check("trade row: every order listed in the details", all(s in d for s, _ in sells + buys) and "AMD" in d)
moved, m, _ = pt._trade_notice(res[res["Symbol"] == "AMD"], prices)
check("queued-only row: nothing sent, no money moved", moved == "no" and m.startswith("Trades queued")
      and "no money moved" in m, m)

# ------------------------------------------------------------------ fill check: nothing pending -> no row
ROWS.clear()
missing = os.path.join(tempfile.mkdtemp(), "live_pending_orders.json")
r = pt.complete_unfilled_orders(pending_path=missing, log_csv=None)
check("nothing pending: no row at all", r.empty and not ROWS, ROWS)
empty = os.path.join(tempfile.mkdtemp(), "live_pending_orders.json")
json.dump({"evening_date": "2026-10-02", "orders": []}, open(empty, "w"))
r = pt.complete_unfilled_orders(pending_path=empty, log_csv=None)
check("empty order list: no row at all", r.empty and not ROWS, ROWS)

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
    def get_order_by_id(self, oid): return self.orders()[0]

    def orders(self):
        return [types.SimpleNamespace(id="e1", symbol="ANET", side="BUY", qty=5, status="filled", filled_qty=5,
                                      client_order_id="live-20261002-BUY-ANET-5-1")]

    def get_orders(self, req=None): return self.orders()
    def submit_order(self, req): raise AssertionError("no order may be sent in this test")
    def cancel_order_by_id(self, oid): raise AssertionError("no cancel in this test")

def morning(held, reconcile_ok=True):
    patch_account(held)
    if not reconcile_ok:
        pt.get_live_positions_and_equity = lambda: (_ for _ in ()).throw(ConnectionError("down"))
    pt.paper_trading_client = lambda: Broker(held)
    p = os.path.join(tempfile.mkdtemp(), "live_pending_orders.json")
    json.dump({"evening_date": "2026-10-02", "target_source": "auto", "orders": [
        {"symbol": "ANET", "side": "BUY", "qty": 5, "order_qty": 5, "limit_price": 100.0, "order_id": "e1"}]},
        open(p, "w"))
    ROWS.clear()
    with redirect_stdout(io.StringIO()):
        pt.complete_unfilled_orders(pending_path=p, log_csv=None)
    return list(ROWS)

got = morning({**manual, "ANET": 5})
check("filled + on target, hand-bought stocks held: one 'Fill check' ok row, money moved",
      [(r[0], r[1], r[2]) for r in got] == [("Fill check", "ok", "yes")], got)
check("'filled' row names the symbol and says nothing to do",
      "buy ANET 5" in got[0][3] and "money moved" in got[0][3] and "Nothing to do" in got[0][3], got)
got = morning({**manual, "ANET": 2})
drift = [r for r in got if "2.0% of your account" in r[3]]
check("traded symbol really off target: a warning row with the percentages",
      len(drift) == 1 and drift[0][1] == "warning" and "5.0%" in drift[0][3] and " pp" not in drift[0][3], got)
got = morning({**manual, "ANET": 5}, reconcile_ok=False)
check("couldn't read holdings: its own warning row, no drift row",
      any("couldn't be read to compare holdings" in r[3] for r in got) and not any("% of your account" in r[3] for r in got), got)

# ------------------------------------------------------------------ run_all finished rows (plain words)
CT = ZoneInfo("America/Chicago")
wed, fri = pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-02")
check("labels: Wed check / Fri rebalance (2:30 PM, regular hours) / catch-up / data refresh",
      [run_all.run_label(wed, "evening"), run_all.run_label(fri, "session", datetime(2026, 10, 2, 14, 43, tzinfo=CT)),
       run_all.run_label(fri, "session", datetime(2026, 10, 5, 9, 31, tzinfo=CT)), run_all.run_label(None, None)]
      == ["Wed check", "Fri rebalance", "Catch-up: Fri Oct 2 rebalance", "Data refresh"])
ran7 = [(n, True, 1.0) for n in "abcdefg"]
table = lambda *r: pd.DataFrame(list(r), columns=["Symbol", "Side", "Shares", "Order_ID", "Status"])
end = datetime(2026, 9, 30, 15, 31, tzinfo=CT)
moved, m = run_all.finished_text("Wed check", end, ran7, [], True, "evening", table())
check("Wed, quiet check: done + all OK + no money moved + next + nothing to do",
      moved == "no" and m == ("Wed check done 3:31 PM. All 7 steps OK. No trades needed - no orders, no money "
                              "moved. Next: Fri Oct 2 rebalance at 2:30 PM. Nothing to do."), m)
sent = table(("ENPH", "SELL", 4, "1", "submitted (accepted)"), ("FIG", "SELL", 6, "2", "submitted (new)"),
             ("AMD", "BUY", 3, "3", "submitted (new) + 0.58 fractional rest next morning"),
             ("TEM", "BUY", 0.8, None, "STAGED for morning market (<1 whole share after hours)"))
moved, m = run_all.finished_text("Fri rebalance", datetime(2026, 10, 2, 15, 32, tzinfo=CT), ran7, [], True, "evening", sent)
check("Fri, orders sent: money moved, counts, 9 AM remainders, next = Mon check", moved == "yes"
      and "Orders sent: 2 sells, 2 buys - money moves as they fill; small remainders go out at 9 AM." in m
      and "Next: Mon Oct 5 check at 2:30 PM." in m and m.endswith("Nothing to do."), m)
moved, m = run_all.finished_text("Fri rebalance", datetime(2026, 10, 2, 19, 5, tzinfo=CT), ran7, [], True, "evening",
                                 table(("AMD", "BUY", 3, None, "STAGED for morning market (past 7 PM CT - not submitted)")))
check("past 7 PM: queued for 9 AM, no money moved yet", moved == "no"
      and "Orders queued for 9 AM: 1 buy - no money moved yet." in m, m)
done = table(("ENPH", "SELL", 4, "m1", "COMPLETED via limit $33.4 (staged, filled 0/4)"),
             ("AMD", "BUY", 3.58, "m2", "COMPLETED via limit $634.2 (staged, filled 0/3.58)"))
moved, m = run_all.finished_text("Catch-up: Fri Oct 2 rebalance", datetime(2026, 10, 5, 9, 31, tzinfo=CT), ran7, [], True,
                                 "session", table(("AMD", "BUY", 3.58, None, "STAGED ... (catch-up)")), done)
check("catch-up: sent now as limit orders, counts, money moved", moved == "yes"
      and "Catch-up orders sent as limit orders at the live bid/ask (sells first): 1 sell, 1 buy - money moves as they fill; any rest goes out at 9 AM CT."
      in m, m)
moved, m = run_all.finished_text("Fri rebalance", datetime(2026, 10, 2, 14, 46, tzinfo=CT), ran7, [], True,
                                 "session", table(("AMD", "BUY", 3.58, None, "STAGED for regular-hours limit orders now")), done)
check("2:30 PM run: sent in regular hours (not called a catch-up), money moved", moved == "yes"
      and m.startswith("Fri rebalance done 2:46 PM.")
      and "Orders sent as limit orders at the live bid/ask (sells first): 1 sell, 1 buy - money moves as they fill; any rest goes out at 9 AM CT." in m, m)
now = table(("SNOW", "SELL", 23, None, "STAGED for regular-hours limit orders now"),
            ("GTLB", "BUY", 147.49, None, "STAGED for regular-hours limit orders now"))
line = run_all.trade_summary(now, table(("SNOW", "SELL", 23, "m1", "COMPLETED via limit $333.29 (sent now, nothing was sent before)"),
                                        ("GTLB", "BUY", 147.49, "m2", "COMPLETED via limit $52.04 (sent now, nothing was sent before)")))
check("summary, 2:30 PM run: orders sent at once are not called 'STAGED for the 9 AM fill check'",
      line == "Trade: 2 sent at once as regular-hours limit orders (2 completed), 0 skipped/failed "
              "(see Reports/live_orders_log.csv)" and "9 AM" not in line, line)
line = run_all.trade_summary(table(("ENPH", "SELL", 4, "1", "submitted (accepted)"),
                                   ("TEM", "BUY", 0.8, None, "STAGED for morning market (<1 whole share after hours)"),
                                   ("BAD", "BUY", 1, None, "FAILED: bad quote")))
check("summary, after hours: submitted, queued for 9 AM and failed counted apart",
      line == "Trade: 1 submitted after hours, 1 queued for the 9 AM fill check, 1 skipped/failed "
              "(see Reports/live_orders_log.csv)", line)
moved, m = run_all.finished_text("Data refresh", end, [("main", True, 1), ("validate", True, 1)], [], False, None, None)
check("refresh only: no trades, no money moved", moved == "no" and "Data refresh only - no trades, no money moved." in m, m)
moved, m = run_all.finished_text("Wed check", end, ran7[:6] + [("sentiment", False, 1.0)], ["sentiment"], True,
                                 "evening", table())
check("optional step failed: named, last good data used", "6 of 7 steps OK (sentiment failed - last good data used)." in m, m)

bad = [r for r in ROWS if r[1] not in ("ok", "warning", "failed") or r[2] not in ("yes", "no", "unknown")]
check(f"every row written here has a valid status and money_moved ({len(ROWS)} rows)", ROWS and not bad, bad)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
