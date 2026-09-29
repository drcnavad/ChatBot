"""Mocked regression tests for the LIVE paper_trade.py (zero broker calls, zero network).

Covers the live-specific safety properties:
1. live- / live-fill- client order id namespace (never pa-)
2. evening broker reconciliation matches live- ids only
3. submit_paper_extended: extended-hours limit orders, whole shares, SELL-first,
   SELL clamped to held, incremental record callback
4. submit_paper: regular-hours DAY market orders, 2-decimal rounding
5. auto_trade fails closed on stale signals
6. auto_trade stages (never submits) past 7 PM CT
7. morning fill check completes remainders, never retries rejects, drops malformed rows
8. morning crash recovery: prior completion order never duplicated
9. morning fill check aborts when live positions unreadable (pending kept)
10. live pending/log paths are used (never the paper paths)
11. paper_trading_client builds paper=False client from LIVE keys; missing keys -> SystemExit
12. CLI: --paper retired+refused, --fill-check needs --live

Run: cd <folder> && python3 tests/test_paper_trade_live_safety.py
"""
import json
import math
import os
os.environ.setdefault("STOCK_ANALYSIS_NO_POPUPS", "1")  # never pop real macOS alerts from tests
import sys
import tempfile
import types
from enum import Enum as _Enum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root: paper_trade.py lives here
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# --- stub alpaca.trading before any in-function import runs -------------------
class OrderSide(_Enum):
    BUY = "buy"
    SELL = "sell"

class TimeInForce(_Enum):
    DAY = "day"

class QueryOrderStatus(_Enum):
    ALL = "all"
    OPEN = "open"
    CLOSED = "closed"

class SortDirection(_Enum):
    DESCENDING = "desc"
    ASCENDING = "asc"

class _Req:
    def __init__(self, **kw):
        self.__dict__.update(kw)

class MarketOrderRequest(_Req): pass
class LimitOrderRequest(_Req): pass
class GetOrdersRequest(_Req): pass

_alpaca = types.ModuleType("alpaca")
_trading = types.ModuleType("alpaca.trading")
_client_mod = types.ModuleType("alpaca.trading.client")
_enums = types.ModuleType("alpaca.trading.enums")
_requests = types.ModuleType("alpaca.trading.requests")
_enums.OrderSide = OrderSide
_enums.TimeInForce = TimeInForce
_enums.QueryOrderStatus = QueryOrderStatus
_enums.SortDirection = SortDirection
_requests.MarketOrderRequest = MarketOrderRequest
_requests.LimitOrderRequest = LimitOrderRequest
_requests.GetOrdersRequest = GetOrdersRequest
_client_mod.TradingClient = type("TradingClient", (), {})
_alpaca.trading = _trading
_trading.client = _client_mod
_trading.enums = _enums
_trading.requests = _requests
sys.modules.update({"alpaca": _alpaca, "alpaca.trading": _trading,
                    "alpaca.trading.client": _client_mod,
                    "alpaca.trading.enums": _enums,
                    "alpaca.trading.requests": _requests})

import pandas as pd
_dotenv_stub = types.ModuleType("dotenv")
_dotenv_stub.load_dotenv = lambda *a, **k: None
sys.modules["dotenv"] = _dotenv_stub
import paper_trade
import paper_trade as pt

PASS, FAIL = [], []

def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""))


class FakeOrder:
    _n = 0
    def __init__(self, **kw):
        FakeOrder._n += 1
        self.id = kw.pop("id", f"fake-{FakeOrder._n}")
        self.status = kw.pop("status", "accepted")
        self.filled_qty = kw.pop("filled_qty", 0)
        self.client_order_id = kw.pop("client_order_id", "")
        self.__dict__.update(kw)

class FakeAccount:
    def __init__(self, equity=100000.0, cash=50000.0, bp=50000.0):
        self.equity, self.cash, self.buying_power = equity, cash, bp

class FakePos:
    def __init__(self, symbol, qty):
        self.symbol, self.qty = symbol, qty

class FakeClient:
    """Fake Alpaca trading client. Set fail_submit / fail_positions to simulate errors."""
    def __init__(self, positions=None, account=None, orders_by_id=None, orders_list=None):
        self.positions = positions or {}
        self.account = account or FakeAccount()
        self.orders_by_id = orders_by_id or {}
        self.orders_list = orders_list or []
        self.submitted = []
        self.canceled = []
        self.paper_kwarg = None
    def get_account(self): return self.account
    def get_all_positions(self): return [FakePos(s, q) for s, q in self.positions.items()]
    def submit_order(self, req):
        self.submitted.append(req)
        o = FakeOrder(id=f"ord-{len(self.submitted)}", status="accepted",
                      client_order_id=getattr(req, "client_order_id", ""))
        self.orders_by_id[o.id] = o
        return o
    def get_order(self, oid): return self.orders_by_id[oid]
    def cancel_order(self, oid): self.canceled.append(oid)
    def get_orders(self, req): return list(self.orders_list)
    def get_clock(self): return types.SimpleNamespace(is_open=True)


def fake_orders_df():
    return pd.DataFrame([
        {"Symbol": "AAA", "Side": "SELL", "Shares": 10, "Price": 100.0, "Est_Value": 1000.0,
         "Target_Shares": 0},
        {"Symbol": "BBB", "Side": "BUY", "Shares": 5, "Price": 50.0, "Est_Value": 250.0,
         "Target_Shares": 5},
    ])

def patch_live(monkey, **kw):
    old = {}
    for k, v in kw.items():
        old[k] = getattr(paper_trade, k)
        setattr(paper_trade, k, v)
    monkey.append(old)

# ---------------------------------------------------------------- 1: id namespace
cid = paper_trade._client_order_id("BRK.B", "BUY", 10, 123.45, "20260928")
check("evening id has live- prefix", cid.startswith("live-") and not cid.startswith("pa-"), cid)
check("evening id <= 48 chars", len(cid) <= 48, cid)
check("evening id deterministic",
      cid == paper_trade._client_order_id("BRK.B", "BUY", 10, 123.45, "20260928"))
cidf = paper_trade._client_order_id("AAA", "SELL", 3, 10.0, "20260928", kind="fill")
check("morning id has live-fill- prefix", cidf.startswith("live-fill-"), cidf)

# ------------------------------------------------- 2: evening broker reconciliation
fc = FakeClient(orders_list=[
    FakeOrder(client_order_id="live-20260928-SELL-AAA-10-10000"),
    FakeOrder(client_order_id="live-fill-20260928-BUY-BBB-5-25000"),
    FakeOrder(client_order_id="pa-20260928-SELL-CCC-7-7000"),
    FakeOrder(client_order_id="live-20260927-SELL-DDD-1-1000"),
])
got = paper_trade._evening_submitted_on_broker(fc, "20260928")
check("reconciliation matches live- ids only", got == {("AAA", "SELL")}, str(got))

# ------------------------------------------------- 3: submit_live_extended
fc = FakeClient(positions={"AAA": 10})
recorded = []
res = paper_trade.submit_paper_extended(fake_orders_df(), positions={"AAA": 10},
                                      record=recorded.append, client=fc, order_date="20260928")
check("extended: 2 rows submitted", len(res) == 2 and len(fc.submitted) == 2)
r0, r1 = fc.submitted
check("extended: SELL first", getattr(r0, "side", None) == OrderSide.SELL)
check("extended: limit orders w/ extended_hours",
      isinstance(r0, LimitOrderRequest) and r0.extended_hours is True and r0.time_in_force == TimeInForce.DAY)
check("extended: whole shares", r0.qty == 10 and isinstance(r0.qty, int))
check("extended: limit at close", r0.limit_price == 100.0)
check("extended: live- client order id", r0.client_order_id.startswith("live-"), r0.client_order_id)
check("extended: rows recorded incrementally", len(recorded) == 2 and recorded[0]["order_id"] is not None)
# SELL clamp: plan says 10 but only 4 held
res2 = paper_trade.submit_paper_extended(fake_orders_df(), positions={"AAA": 4}, client=fc, order_date="20260928")
check("extended: SELL clamped to held", fc.submitted[-2].qty == 4)
# SELL not held -> skipped
res3 = paper_trade.submit_paper_extended(fake_orders_df(), positions={}, client=fc, order_date="20260928")
check("extended: unheld SELL skipped", "SKIPPED" in res3.iloc[0]["Status"] and len(fc.submitted) == 5, res3.iloc[0]["Status"])

# ------------------------------------------------- 4: submit_live (market)
fc = FakeClient(positions={"AAA": 10.5})
df = fake_orders_df()
df.loc[df.Symbol == "AAA", "Shares"] = 10.5
res = paper_trade.submit_paper(df, positions={"AAA": 10.5}, client=fc, order_date="20260928")
m0 = fc.submitted[0]
check("market: MarketOrderRequest DAY", isinstance(m0, MarketOrderRequest) and m0.time_in_force == TimeInForce.DAY)
check("market: SELL clears fractional holding", m0.qty == 10.5, str(m0.qty))
check("market: live- id", m0.client_order_id.startswith("live-"))

# ------------------------------------------------- 5: stale signals fail closed
monkey = []
patch_live(monkey, check_signal_freshness=lambda: (_ for _ in ()).throw(ValueError("STALE DATA")))
try:
    paper_trade.auto_trade(dry_run=True)
    check("stale signals raise", False)
except ValueError:
    check("stale signals raise", True)
except Exception as e:
    check("stale signals raise", False, repr(e))
for k, v in monkey.pop().items(): setattr(paper_trade, k, v)

# ------------------------------------------------- 6: past-7PM staging
tmp = tempfile.mkdtemp()
pend, log = os.path.join(tmp, "live_pending_orders.json"), os.path.join(tmp, "live_orders_log.csv")
monkey = []
patch_live(monkey,
           check_signal_freshness=lambda: "2026-09-28",
           get_live_positions_and_equity=lambda: ({"AAA": 10}, 100000.0, 50000.0, 50000.0),
           _past_evening_cutoff=lambda: True,
           plan_orders=lambda *a, **k: (fake_orders_df(), {"source": "auto", "as_of": "2026-09-28"}, pd.DataFrame()),
           apply_buying_power_guard=lambda o, bp, **k: o,
           _todays_recorded_orders=lambda *a, **k: set())
orig_pend, orig_log = paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV
paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV = pend, log
fc = FakeClient()
patch_live(monkey, paper_trading_client=lambda: fc)
orders, meta, results = paper_trade.auto_trade(dry_run=False, log_csv=log)
paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV = orig_pend, orig_log
for k, v in monkey.pop().items(): setattr(paper_trade, k, v)
check("past-7PM: nothing submitted", len(fc.submitted) == 0)
check("past-7PM: rows STAGED", (results["Status"].str.contains("STAGED")).all())
check("past-7PM: staged in LIVE pending file", os.path.exists(pend) and not os.path.exists(
    os.path.join(tmp, "paper_pending_orders.json")))
pj = json.load(open(pend))
check("past-7PM: staged rows have no broker id", all(o["order_id"] is None for o in pj["orders"]))
check("past-7PM: logged to LIVE csv", os.path.exists(log))

# ------------------------------------------------- 7: morning fill check
tmp = tempfile.mkdtemp()
pend, log = os.path.join(tmp, "live_pending_orders.json"), os.path.join(tmp, "live_orders_log.csv")
eve = {"evening_date": "2026-09-28", "submitted_at_ct": "2026-09-28 16:00:00",
       "target_source": "auto", "as_of": "2026-09-28", "orders": [
    {"symbol": "AAA", "side": "SELL", "qty": 10, "limit_price": 100.0, "order_id": "eve-1"},
    {"symbol": "BBB", "side": "BUY", "qty": 5, "limit_price": 50.0, "order_id": "eve-2"},
    {"symbol": "CCC", "side": "BUY", "qty": 2, "limit_price": 20.0, "order_id": "eve-3"},
    {"symbol": "DDD", "side": "SELL", "qty": 7, "limit_price": 70.0, "order_id": None},  # staged past 7PM
    {"nope": True},  # malformed row
]}
json.dump(eve, open(pend, "w"))
orders_by_id = {
    "eve-1": FakeOrder(id="eve-1", status="expired", filled_qty=4, client_order_id="live-20260928-SELL-AAA-10-10000"),
    "eve-2": FakeOrder(id="eve-2", status="filled", filled_qty=5, client_order_id="live-20260928-BUY-BBB-5-25000"),
    "eve-3": FakeOrder(id="eve-3", status="rejected", filled_qty=0, client_order_id="live-20260928-BUY-CCC-2-2000"),
}
fc = FakeClient(positions={"AAA": 10, "DDD": 7}, orders_by_id=orders_by_id, orders_list=[])
monkey = []
patch_live(monkey, paper_trading_client=lambda: fc,
           _wait_terminal=lambda c, oid, **k: True,
           _broker_orders_by_client_id=lambda c: {})
orig_pend, orig_log = paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV
paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV = pend, log
res = paper_trade.complete_unfilled_orders(pending_path=pend, log_csv=log, dry_run=False)
paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV = orig_pend, orig_log
for k, v in monkey.pop().items(): setattr(paper_trade, k, v)
by_sym = {r.Symbol: r.Status for r in res.itertuples()}
check("morning: partial SELL completed", "COMPLETED" in by_sym.get("AAA", ""), by_sym.get("AAA"))
check("morning: filled BUY untouched", by_sym.get("BBB", "").startswith("FILLED"), by_sym.get("BBB"))
check("morning: rejected never retried", by_sym.get("CCC", "").startswith("REJECTED"), by_sym.get("CCC"))
check("morning: staged row sent full qty", "COMPLETED" in by_sym.get("DDD", ""), by_sym.get("DDD"))
check("morning: malformed row dropped loudly", by_sym.get("?", "").startswith("FAILED"), by_sym.get("?"))
mkt = [s for s in fc.submitted if isinstance(s, MarketOrderRequest)]
check("morning: market orders only", len(mkt) == len(fc.submitted) == 2, str(len(fc.submitted)))
check("morning: SELL remainder clamped (10-4=6)", any(s.qty == 6 and s.side == OrderSide.SELL for s in mkt),
      str([(s.symbol, s.qty) for s in mkt]))
check("morning: completion ids live-fill-", all(s.client_order_id.startswith("live-fill-") for s in mkt),
      str([s.client_order_id for s in mkt]))
check("morning: SELL completed before BUY", [s.side for s in mkt][0] == OrderSide.SELL)
check("morning: logged to LIVE csv", os.path.exists(log))

# ------------------------------------------------- 8: crash recovery (no duplicate)
tmp = tempfile.mkdtemp()
pend, log = os.path.join(tmp, "live_pending_orders.json"), os.path.join(tmp, "live_orders_log.csv")
prior = FakeOrder(id="morn-1", status="accepted", filled_qty=6, client_order_id="live-fill-20260928-SELL-AAA-6-10000")
eve = {"evening_date": "2026-09-28", "submitted_at_ct": "2026-09-28 16:00:00",
       "target_source": "auto", "as_of": "2026-09-28", "orders": [
    {"symbol": "AAA", "side": "SELL", "qty": 10, "limit_price": 100.0, "order_id": "eve-1"},
]}
json.dump(eve, open(pend, "w"))
fc = FakeClient(positions={"AAA": 10},
                orders_by_id={"eve-1": FakeOrder(id="eve-1", status="expired", filled_qty=4,
                                                 client_order_id="live-20260928-SELL-AAA-10-10000")},
                orders_list=[prior])
monkey = []
patch_live(monkey, paper_trading_client=lambda: fc,
           _wait_terminal=lambda c, oid, **k: True,
           _broker_orders_by_client_id=lambda c: {"live-fill-20260928-SELL-AAA-6-10000": prior},
           _today_ct=lambda: __import__("datetime").datetime(2026, 9, 29, 10, 0,
                        tzinfo=__import__("zoneinfo").ZoneInfo("America/Chicago")))
orig_pend, orig_log = paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV
paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV = pend, log
res = paper_trade.complete_unfilled_orders(pending_path=pend, log_csv=log, dry_run=False)
paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV = orig_pend, orig_log
for k, v in monkey.pop().items(): setattr(paper_trade, k, v)
check("crash recovery: no duplicate submit", len(fc.submitted) == 0, str(len(fc.submitted)))
check("crash recovery: marked ALREADY COMPLETED", "ALREADY COMPLETED" in res.iloc[0]["Status"])
check("crash recovery: pending file removed", not os.path.exists(pend))

# ------------------------------------------------- 9: fail closed on unreadable positions
tmp = tempfile.mkdtemp()
pend = os.path.join(tmp, "live_pending_orders.json")
json.dump({"evening_date": "2026-09-28", "orders": [
    {"symbol": "AAA", "side": "SELL", "qty": 10, "limit_price": 100.0, "order_id": "eve-1"}]}, open(pend, "w"))
class DeadClient(FakeClient):
    def get_all_positions(self): raise RuntimeError("broker down")
monkey = []
patch_live(monkey, paper_trading_client=lambda: DeadClient())
orig_pend = paper_trade.PENDING_ORDERS_JSON
paper_trade.PENDING_ORDERS_JSON = pend
try:
    paper_trade.complete_unfilled_orders(pending_path=pend, log_csv=os.path.join(tmp, "x.csv"), dry_run=False)
    check("unreadable positions aborts", False)
except RuntimeError:
    check("unreadable positions aborts", True)
except Exception as e:
    check("unreadable positions aborts", False, repr(e))
paper_trade.PENDING_ORDERS_JSON = orig_pend
for k, v in monkey.pop().items(): setattr(paper_trade, k, v)
check("pending file kept for retry", os.path.exists(pend))

# ------------------------------------------------- 10: dry run changes nothing
tmp = tempfile.mkdtemp()
pend, log = os.path.join(tmp, "live_pending_orders.json"), os.path.join(tmp, "live_orders_log.csv")
json.dump({"evening_date": "2026-09-28", "orders": [
    {"symbol": "AAA", "side": "SELL", "qty": 10, "limit_price": 100.0, "order_id": "eve-1"}]}, open(pend, "w"))
fc = FakeClient(positions={"AAA": 10},
                orders_by_id={"eve-1": FakeOrder(id="eve-1", status="expired", filled_qty=0,
                                                 client_order_id="live-20260928-SELL-AAA-10-10000")})
monkey = []
patch_live(monkey, paper_trading_client=lambda: fc,
           _broker_orders_by_client_id=lambda c: {})
orig_pend, orig_log = paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV
paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV = pend, log
res = paper_trade.complete_unfilled_orders(pending_path=pend, log_csv=log, dry_run=True)
paper_trade.PENDING_ORDERS_JSON, paper_trade.ORDER_LOG_CSV = orig_pend, orig_log
for k, v in monkey.pop().items(): setattr(paper_trade, k, v)
check("dry run: nothing submitted", len(fc.submitted) == 0)
check("dry run: pending file kept", os.path.exists(pend))
check("dry run: nothing logged", not os.path.exists(log))
check("dry run: WOULD COMPLETE shown", "WOULD COMPLETE" in res.iloc[0]["Status"])

# ------------------------------------------------- 11: client is paper=False; missing keys
seen = {}
class TC:
    def __init__(self, key, secret, paper=True):
        seen["key"], seen["secret"], seen["paper"] = key, secret, paper
sys.modules["alpaca.trading.client"].TradingClient = TC
os.environ["ALPACA_LIVE_KEY_ID"] = "K"
os.environ["ALPACA_LIVE_SECRET_KEY"] = "S"
paper_trade.paper_trading_client()
check("client built with paper=False", seen.get("paper") is False, str(seen))
check("client uses LIVE key names only", seen.get("key") == "K" and seen.get("secret") == "S")
# wrong key names are never consulted: delete live keys, set paper-style keys
del os.environ["ALPACA_LIVE_KEY_ID"]; del os.environ["ALPACA_LIVE_SECRET_KEY"]
os.environ["ALPACA_KEY_ID"] = "PAPERK"; os.environ["ALPACA_SECRET_KEY"] = "PAPERS"
try:
    paper_trade.paper_trading_client()
    check("missing live keys -> SystemExit", False)
except SystemExit:
    check("missing live keys -> SystemExit", True)
except Exception as e:
    check("missing live keys -> SystemExit", False, repr(e))
del os.environ["ALPACA_KEY_ID"]; del os.environ["ALPACA_SECRET_KEY"]

# ------------------------------------------------- 12: CLI guards
try:
    paper_trade.main(["--paper"])
    check("CLI --paper refused (retired flag)", False)
except SystemExit:
    check("CLI --paper refused (retired flag)", True)
try:
    paper_trade.main(["--submit", "--paper"])
    check("CLI --submit --paper refused", False)
except SystemExit:
    check("CLI --submit --paper refused", True)
try:
    paper_trade.main(["--fill-check"])
    check("CLI --fill-check without --live refused", False)
except SystemExit:
    check("CLI --fill-check without --live refused", True)
# --live --fill-check with no pending file: clean no-op, no broker client created
tmp = tempfile.mkdtemp()
ghost = os.path.join(tmp, "ghost_pending.json")
orig_pend, orig_ptc = paper_trade.PENDING_ORDERS_JSON, paper_trade.paper_trading_client
paper_trade.PENDING_ORDERS_JSON = ghost
paper_trade.paper_trading_client = lambda: (_ for _ in ()).throw(AssertionError("client must not be created"))
try:
    r = paper_trade.main(["--live", "--fill-check"])
    check("CLI --live --fill-check no-op without pending file", r is not None and r.empty)
except AssertionError as e:
    check("CLI --live --fill-check no-op without pending file", False, str(e))
finally:
    paper_trade.PENDING_ORDERS_JSON, paper_trade.paper_trading_client = orig_pend, orig_ptc
# ledger constants point at the live files
check("PENDING_ORDERS_JSON is the live ledger",
      paper_trade.PENDING_ORDERS_JSON.endswith("live_pending_orders.json"))
check("ORDER_LOG_CSV is the live ledger",
      paper_trade.ORDER_LOG_CSV.endswith("live_orders_log.csv"))

# ------------------------------------------------- isolation: no paper paths inside paper_trade
src = open(os.path.join(ROOT, "paper_trade.py")).read()
check("no paper pending path constant", "paper_pending_orders" not in src)
check("no paper log path constant", "paper_orders_log" not in src)
check("no pa- client id prefix", '"pa-' not in src and "'pa-" not in src)


# ------------------------------------------------- 13: morning BUY buying-power guards
def _morning_bp_case(bp, price, qty, expect_submit):
    tmp = tempfile.mkdtemp()
    pend = os.path.join(tmp, "live_pending_orders.json")
    json.dump({"evening_date": "2026-09-28", "submitted_at_ct": "x", "target_source": "auto",
               "as_of": "2026-09-28", "orders": [
        {"symbol": "BBB", "side": "BUY", "qty": qty, "limit_price": price, "order_id": "eve-9"}]},
              open(pend, "w"))
    fc = FakeClient(positions={},
                    orders_by_id={"eve-9": FakeOrder(id="eve-9", status="expired", filled_qty=0,
                                                    client_order_id="live-20260928-BUY-BBB")})
    monkey = []
    patch_live(monkey, paper_trading_client=lambda: fc,
               _wait_terminal=lambda c, oid, **k: True,
               _broker_orders_by_client_id=lambda c: {},
               _read_buying_power=lambda c: bp)
    res = paper_trade.complete_unfilled_orders(pending_path=pend,
                                                   log_csv=os.path.join(tmp, "x.csv"),
                                                   dry_run=False)
    for k, v in monkey.pop().items(): setattr(paper_trade, k, v)
    return res, fc, pend

res, fc, pend = _morning_bp_case(100.0, 50.0, 5, False)  # need ~$252.50 (1% cushion), have $100
check("morning: short on cash -> buys the part that fits (100 / 50.50 = 1.98)",
      [getattr(o, "qty", None) for o in fc.submitted] == [1.98], [getattr(o, "qty", None) for o in fc.submitted])
check("morning: short on cash -> partial noted, row done", "only 1.98 of 5 fit" in res.iloc[0]["Status"]
      and not os.path.exists(pend), res.iloc[0]["Status"])
res, fc, pend = _morning_bp_case(0.5, 50.0, 5, False)  # not even $1 free
check("morning: no cash -> nothing sent", len(fc.submitted) == 0)
check("morning: no cash -> NO FILL, not retried", res.iloc[0]["Status"].startswith("NO FILL")
      and not os.path.exists(pend), res.iloc[0]["Status"])

res, fc, pend = _morning_bp_case(100000.0, 50.0, 5, True)  # plenty
check("morning: sufficient BP -> submitted", len(fc.submitted) == 1)
check("morning: sufficient BP -> pending removed", not os.path.exists(pend))

import math as _m
res, fc, pend = _morning_bp_case(float("nan"), 50.0, 5, False)  # unreadable BP
check("morning: unreadable BP -> not submitted", len(fc.submitted) == 0)
check("morning: unreadable BP -> kept for retry", os.path.exists(pend))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
