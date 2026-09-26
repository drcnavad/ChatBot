"""Mocked regression tests for the paper_trade.py safety fixes (zero broker calls).

Covers:
1. fail-closed unknown signal status: _is_buy_signal / build_orders / latest_signal_status
2. fail-closed buying-power guard: apply_buying_power_guard with unknown/invalid buying power
3. stable client order ids + broker reconciliation: _client_order_id,
   _broker_orders_by_client_id, _evening_submitted_on_broker, _morning_completion_plan
4. sequenced evening submit: SELLs settle before BUYs are sized (submit_paper_extended_sequenced)
5. morning fill check aborts when live positions cannot be read (complete_unfilled_orders)
6. morning crash recovery: a prior attempt's completion order is never duplicated

No test touches the network or the broker: alpaca.trading is stubbed in sys.modules
and paper_trading_client is monkeypatched with a fake client.
Run: PYTHONPATH=. python tests/test_paper_trade_safety.py
"""
import json
import math
import os
import sys
import tempfile
from enum import Enum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# --- stub alpaca.trading before paper_trade's in-function imports run -----------
import types

from enum import Enum as _Enum


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


class MarketOrderRequest(_Req):
    pass


class LimitOrderRequest(_Req):
    pass


class GetOrdersRequest(_Req):
    pass


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
sys.modules["alpaca"] = _alpaca
sys.modules["alpaca.trading"] = _trading
sys.modules["alpaca.trading.client"] = _client_mod
sys.modules["alpaca.trading.enums"] = _enums
sys.modules["alpaca.trading.requests"] = _requests

import pandas as pd

import paper_trade

FAIL = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        FAIL.append(what)


# --- fakes --------------------------------------------------------------------
class FakePosition:
    def __init__(self, symbol, qty):
        self.symbol = symbol
        self.qty = qty


class FakeAccount:
    def __init__(self, buying_power, cash=None):
        self.buying_power = buying_power
        # cash defaults to buying_power when not given (ample for tests)
        self.cash = buying_power if cash is None else cash


class FakeOrder:
    def __init__(self, id, symbol="", side="BUY", qty=0, status="new", filled_qty=0,
                 client_order_id="", limit_price=None):
        self.id = id
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.status = status  # plain string: the code handles .value or str()
        self.filled_qty = filled_qty
        self.client_order_id = client_order_id
        self.limit_price = limit_price


class FakeClient:
    """Stand-in for alpaca TradingClient. Records everything; never touches the network."""

    def __init__(self):
        self.submitted = []      # request objects, in submission order
        self.orders = {}         # id -> FakeOrder (for get_order)
        self.all_orders = []     # for get_orders
        self.positions = []      # [(symbol, qty)]
        self.buying_power = 100000.0
        self.fail_positions = False
        self.fail_orders = False
        self._next_id = 1

    def submit_order(self, req):
        self.submitted.append(req)
        oid = f"fake-{self._next_id}"
        self._next_id += 1
        o = FakeOrder(id=oid, symbol=req.symbol, qty=req.qty,
                      client_order_id=getattr(req, "client_order_id", "") or "")
        self.orders[oid] = o
        self.all_orders.append(o)
        return o

    def get_order(self, oid):
        return self.orders[oid]

    def cancel_order(self, oid):
        if oid in self.orders:
            self.orders[oid].status = "canceled"

    def get_all_positions(self):
        if self.fail_positions:
            raise ConnectionError("broker unreachable")
        return [FakePosition(s, q) for s, q in self.positions]

    def get_account(self):
        return FakeAccount(self.buying_power)

    def get_orders(self, req=None):
        if self.fail_orders:
            raise ConnectionError("broker unreachable")
        return list(self.all_orders)


def _use_fake(fake):
    orig = paper_trade.paper_trading_client
    paper_trade.paper_trading_client = lambda: fake
    return orig


def _orders_frame():
    cols = paper_trade.ORDER_COLUMNS
    rows = [
        {"Symbol": "AAA", "Side": "BUY", "Shares": 10, "Price": 100.0, "Est_Value": 1000.0,
         "Current_Shares": 0, "Target_Shares": 10, "Target_Weight_%": 10.0, "Target_Value": 1000.0},
        {"Symbol": "BBB", "Side": "BUY", "Shares": 5, "Price": 100.0, "Est_Value": 500.0,
         "Current_Shares": 0, "Target_Shares": 5, "Target_Weight_%": 5.0, "Target_Value": 500.0},
        {"Symbol": "CCC", "Side": "SELL", "Shares": 3, "Price": 100.0, "Est_Value": 300.0,
         "Current_Shares": 3, "Target_Shares": 0, "Target_Weight_%": 0.0, "Target_Value": 0.0},
    ]
    return pd.DataFrame(rows, columns=cols)


def _write_pending(path, orders):
    payload = {"evening_date": paper_trade._today_ct().date().isoformat(),
               "submitted_at_ct": "2026-09-25 18:00:00",
               "target_source": "test", "as_of": "2026-09-25",
               "orders": orders}
    with open(path, "w") as f:
        json.dump(payload, f)


# --- 1. fail-closed unknown signal status ---------------------------------------
def test_unknown_status_is_not_a_buy_signal():
    check(paper_trade._is_buy_signal(None) is False, "unknown status: _is_buy_signal(None) is False")
    check(paper_trade._is_buy_signal("add") is True, "unknown status: 'add' is a buy signal")
    check(paper_trade._is_buy_signal("BUY") is True, "unknown status: 'BUY' is a buy signal")
    check(paper_trade._is_buy_signal("bullish") is True, "unknown status: 'bullish' is a buy signal")
    check(paper_trade._is_buy_signal("hold") is False, "unknown status: 'hold' is not a buy signal")
    check(paper_trade._is_buy_signal("drop") is False, "unknown status: 'drop' is not a buy signal")
    check(paper_trade._is_buy_signal("") is False, "unknown status: '' is not a buy signal")


def test_build_orders_unknown_status_hold():
    targets = pd.DataFrame([
        {"Symbol": "AAA", "Price": 100.0, "Weight": 0.10},
        {"Symbol": "BBB", "Price": 50.0, "Weight": 0.10},
    ])
    orders = paper_trade.build_orders(targets, 10000, positions={}, statuses={})
    check(not (orders["Side"] == "BUY").any(),
          "unknown status: no BUY rows when strategy_changes.csv has no rows")
    check(set(orders["Side"]) == {"HOLD"},
          "unknown status: every target becomes HOLD (fail closed)")

    orders = paper_trade.build_orders(targets, 10000, positions={}, statuses={"AAA": "add"})
    aaa = orders[orders["Symbol"] == "AAA"].iloc[0]
    bbb = orders[orders["Symbol"] == "BBB"].iloc[0]
    check(aaa["Side"] == "BUY" and aaa["Shares"] == 10,
          "unknown status: confirmed 'add' symbol still gets its BUY (10 sh)")
    check(bbb["Side"] == "HOLD",
          "unknown status: symbol with no row is HOLD, never BUY")


def test_latest_signal_status_missing_file():
    check(paper_trade.latest_signal_status(changes_csv="/nonexistent/changes.csv") == {},
          "unknown status: missing strategy_changes.csv -> {} (callers treat as HOLD)")


# --- 2. fail-closed buying-power guard -------------------------------------------
def test_buying_power_guard_unknown_value():
    for bp, label in [(None, "None"), (float("nan"), "NaN"),
                      (float("inf"), "inf"), (-5, "negative"), ("junk", "non-numeric")]:
        orders = paper_trade.apply_buying_power_guard(_orders_frame(), bp)
        buys = orders[orders["Symbol"].isin(["AAA", "BBB"])]
        ok = ((buys["Side"] == "SKIP (no buying power)").all()
              and (buys["Shares"] == 0).all()
              and (orders[orders["Symbol"] == "CCC"]["Side"] == "SELL").all())
        check(ok, f"buying-power guard: bp={label} -> BUYs SKIP (no buying power), SELL untouched")


def test_buying_power_guard_valid_still_caps():
    orders = paper_trade.apply_buying_power_guard(_orders_frame(), 1000.0)
    aaa = orders[orders["Symbol"] == "AAA"].iloc[0]
    bbb = orders[orders["Symbol"] == "BBB"].iloc[0]
    total = aaa["Shares"] * aaa["Price"] + bbb["Shares"] * bbb["Price"]
    check(aaa["Side"] == "BUY" and bbb["Side"] == "BUY" and total <= 1000.0,
          f"buying-power guard: valid BP still caps buys proportionally (spend {total} <= 1000)")


# --- 3. stable client order ids + broker reconciliation --------------------------
def test_client_order_id_stable():
    cid = paper_trade._client_order_id("aapl", "BUY", 10, 150.25, "20260925")
    check(cid == "pa-20260925-BUY-AAPL-10-15025", f"client id: deterministic format ({cid})")
    check(len(cid) <= 48, "client id: <= 48 chars (Alpaca-safe)")
    check(paper_trade._client_order_id("aapl", "BUY", 10, 150.25, "20260925") == cid,
          "client id: stable across retries")
    check(paper_trade._client_order_id("aapl", "BUY", 11, 150.25, "20260925") != cid,
          "client id: changes with quantity")
    check(paper_trade._client_order_id("aapl", "BUY", 10, 150.25, "20260925", kind="fill")
          == "pa-fill-20260925-BUY-AAPL-10-15025",
          "client id: fill kind marks morning orders")


def test_evening_reconciliation():
    fake = FakeClient()
    fake.all_orders = [
        FakeOrder(id="1", client_order_id="pa-20260925-SELL-AMD-10-16542"),
        FakeOrder(id="2", client_order_id="pa-fill-20260926-BUY-AAA-5-10000"),
        FakeOrder(id="3", client_order_id="pa-20260924-BUY-AAA-5-10000"),
        FakeOrder(id="4", client_order_id=""),
    ]
    found = paper_trade._evening_submitted_on_broker(fake, "20260925")
    check(found == {("AMD", "SELL")},
          "reconciliation: finds this evening's broker order, ignores fill-orders/other days/blank ids")


def test_broker_orders_unreadable_degrades_safely():
    fake = FakeClient()
    fake.fail_orders = True
    check(paper_trade._broker_orders_by_client_id(fake) == {},
          "reconciliation: unreadable broker orders -> {} (local record stays primary)")


def test_morning_completion_plan():
    cid, qty, prior = paper_trade._morning_completion_plan({}, "AAA", "BUY", 10, 100.0, "20260926")
    check(prior is None and qty == 10 and cid == "pa-fill-20260926-BUY-AAA-10-10000",
          "morning plan: no prior attempt -> fresh id, full qty")
    done = FakeOrder(id="m-1", filled_qty=10, client_order_id=cid)
    c2, q2, p2 = paper_trade._morning_completion_plan({cid: done}, "AAA", "BUY", 10, 100.0, "20260926")
    check(c2 is None and q2 == 0 and p2 is done,
          "morning plan: prior attempt covered it -> mark completed, never resubmit")
    part = FakeOrder(id="m-1", filled_qty=4, client_order_id=cid)
    c3, q3, p3 = paper_trade._morning_completion_plan({cid: part}, "AAA", "BUY", 10, 100.0, "20260926")
    check(p3 is None and q3 == 6 and c3 == (cid + "-r2")[:48],
          "morning plan: prior attempt partial -> remainder only, suffixed id")


# --- 4. sequenced evening submit -------------------------------------------------
def test_sequenced_sells_settle_before_buys():
    cols = paper_trade.ORDER_COLUMNS
    orders = pd.DataFrame([
        {"Symbol": "AAA", "Side": "SELL", "Shares": 3, "Price": 100.0, "Est_Value": 300.0,
         "Current_Shares": 3, "Target_Shares": 0, "Target_Weight_%": 0.0, "Target_Value": 0.0},
        {"Symbol": "BBB", "Side": "BUY", "Shares": 10, "Price": 100.0, "Est_Value": 1000.0,
         "Current_Shares": 0, "Target_Shares": 10, "Target_Weight_%": 10.0, "Target_Value": 1000.0},
        {"Symbol": "CCC", "Side": "BUY", "Shares": 5, "Price": 100.0, "Est_Value": 500.0,
         "Current_Shares": 0, "Target_Shares": 5, "Target_Weight_%": 5.0, "Target_Value": 500.0},
    ], columns=cols)
    fake = FakeClient()
    fake.buying_power = 1000.0  # fresh broker number after the sells: buys must shrink to fit
    seen = {}
    orig_wait = paper_trade._wait_for_terminal_all

    def fake_wait(client, ids, **kw):
        seen["wait_ids"] = list(ids)
        return {i: ("filled", 3.0) for i in ids}

    paper_trade._wait_for_terminal_all = fake_wait
    recorded = []
    try:
        res = paper_trade.submit_paper_extended_sequenced(
            orders, positions={"AAA": 3}, record=recorded.append,
            client=fake, order_date="20260925")
    finally:
        paper_trade._wait_for_terminal_all = orig_wait
    sides = [r.side for r in fake.submitted]
    check(sides and sides[0] == OrderSide.SELL and all(s == OrderSide.BUY for s in sides[1:]),
          "sequenced: SELL submitted before any BUY")
    check(bool(seen.get("wait_ids")), "sequenced: sells settle (bounded wait) before buys are sized")
    check(all(getattr(r, "client_order_id", "") for r in fake.submitted),
          "sequenced: every submitted order carries a deterministic client_order_id")
    buy_spend = sum(r.qty * 100.0 for r in fake.submitted[1:])
    check(buy_spend <= 1000.0,
          f"sequenced: buys re-guarded against fresh buying power (spend {buy_spend} <= 1000)")
    check(len(recorded) == len(fake.submitted),
          "sequenced: every submitted order recorded incrementally")
    check(isinstance(res, pd.DataFrame) and not res.empty,
          "sequenced: returns the combined results frame")


# --- 5. morning abort on unreadable positions ------------------------------------
def test_morning_aborts_when_positions_unreadable():
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pending.json")
        _write_pending(pp, [{"symbol": "AAA", "side": "BUY", "qty": 10,
                             "limit_price": 100.0, "order_id": "oid-1"}])
        fake = FakeClient()
        fake.fail_positions = True
        fake.orders["oid-1"] = FakeOrder(id="oid-1", status="expired", filled_qty=0)
        orig = _use_fake(fake)
        try:
            try:
                paper_trade.complete_unfilled_orders(pending_path=pp, log_csv=None, dry_run=False)
                raised = False
            except RuntimeError as e:
                raised = "ABORTED" in str(e)
        finally:
            paper_trade.paper_trading_client = orig
        check(raised, "morning: unreadable positions -> ABORTED (fail closed), never unclamped")
        check(os.path.exists(pp), "morning: pending file kept for retry after abort")


def test_morning_dry_run_never_aborts():
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pending.json")
        _write_pending(pp, [{"symbol": "AAA", "side": "BUY", "qty": 10,
                             "limit_price": 100.0, "order_id": "oid-1"}])
        fake = FakeClient()
        fake.fail_positions = True
        fake.orders["oid-1"] = FakeOrder(id="oid-1", status="expired", filled_qty=0)
        orig = _use_fake(fake)
        try:
            res = paper_trade.complete_unfilled_orders(pending_path=pp, log_csv=None, dry_run=True)
        finally:
            paper_trade.paper_trading_client = orig
        check(not res.empty and "WOULD COMPLETE" in res["Status"].iloc[0],
              "morning: dry run previews instead of aborting (nothing is submitted)")


# --- 6. morning crash recovery ----------------------------------------------------
def test_morning_crash_recovery_no_duplicate():
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pending.json")
        _write_pending(pp, [{"symbol": "AAA", "side": "BUY", "qty": 10,
                             "limit_price": 100.0, "order_id": "oid-1"}])
        fake = FakeClient()
        fake.orders["oid-1"] = FakeOrder(id="oid-1", status="expired", filled_qty=0)
        fill_date = paper_trade._today_ct().date().strftime("%Y%m%d")
        exp_cid = paper_trade._client_order_id("AAA", "BUY", 10, 100.0, fill_date, kind="fill")
        # a previous attempt submitted the morning order, then crashed before recording it
        fake.all_orders = [FakeOrder(id="m-1", status="filled", filled_qty=10,
                                     client_order_id=exp_cid)]
        orig = _use_fake(fake)
        try:
            res = paper_trade.complete_unfilled_orders(pending_path=pp, log_csv=None, dry_run=False)
        finally:
            paper_trade.paper_trading_client = orig
        check(not res.empty and "ALREADY COMPLETED" in res["Status"].iloc[0],
              "morning: prior attempt's order found on broker -> ALREADY COMPLETED")
        check(fake.submitted == [], "morning: no duplicate order submitted")
        check(not os.path.exists(pp), "morning: pending file removed (row reached terminal state)")


def test_morning_partial_prior_orders_remainder():
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pending.json")
        _write_pending(pp, [{"symbol": "AAA", "side": "BUY", "qty": 10,
                             "limit_price": 100.0, "order_id": "oid-1"}])
        fake = FakeClient()
        fake.orders["oid-1"] = FakeOrder(id="oid-1", status="expired", filled_qty=0)
        fill_date = paper_trade._today_ct().date().strftime("%Y%m%d")
        exp_cid = paper_trade._client_order_id("AAA", "BUY", 10, 100.0, fill_date, kind="fill")
        fake.all_orders = [FakeOrder(id="m-1", status="new", filled_qty=4, client_order_id=exp_cid)]
        orig = _use_fake(fake)
        try:
            res = paper_trade.complete_unfilled_orders(pending_path=pp, log_csv=None, dry_run=False)
        finally:
            paper_trade.paper_trading_client = orig
        check(len(fake.submitted) == 1 and fake.submitted[0].qty == 6,
              "morning: prior attempt partial -> only the 6-share remainder is ordered")
        check(fake.submitted[0].client_order_id == (exp_cid + "-r2")[:48],
              "morning: remainder uses a suffixed deterministic id")
        check("COMPLETED" in res["Status"].iloc[0], "morning: remainder completion reported")


def test_morning_sell_clamped_to_live_positions():
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pending.json")
        _write_pending(pp, [{"symbol": "AAA", "side": "SELL", "qty": 10,
                             "limit_price": 100.0, "order_id": "oid-1"}])
        fake = FakeClient()
        fake.orders["oid-1"] = FakeOrder(id="oid-1", status="expired", filled_qty=0)
        fake.positions = [("AAA", 6)]  # only 6 shares actually held
        orig = _use_fake(fake)
        try:
            res = paper_trade.complete_unfilled_orders(pending_path=pp, log_csv=None, dry_run=False)
        finally:
            paper_trade.paper_trading_client = orig
        check(len(fake.submitted) == 1 and fake.submitted[0].qty == 6,
              "morning: SELL remainder clamped to the 6 shares actually held")


def test_trade_summary_lists_buys_and_sells():
    df = pd.DataFrame([
        ("NVDA", "BUY", 10, "x1", "submitted"),
        ("MSFT", "BUY", 5, "x2", "STAGED for morning market (past 7 PM CT - not submitted)"),
        ("INTC", "SELL", 8, "x3", "submitted"),
        ("AMD", "BUY", 3, "x4", "SKIPPED: <1 whole share"),
        ("TSLA", "SELL", 2, "x5", "FAILED: no price"),
        ("META", "SKIP (no buying power)", 0, None, "SKIP (no buying power)"),
    ], columns=["Symbol", "Side", "Shares", "Order_ID", "Status"])
    s = paper_trade._trade_summary(df)
    check(s == "Bought: NVDA x10, MSFT x5 | Sold: INTC x8",
          f"trade summary lists submitted/staged buys and sells (got {s!r})")
    check(paper_trade._trade_summary(df.iloc[0:0]) == "Bought: none | Sold: none",
          "trade summary on empty results says none/none")


if __name__ == "__main__":
    test_unknown_status_is_not_a_buy_signal()
    test_build_orders_unknown_status_hold()
    test_latest_signal_status_missing_file()
    test_buying_power_guard_unknown_value()
    test_buying_power_guard_valid_still_caps()
    test_client_order_id_stable()
    test_evening_reconciliation()
    test_broker_orders_unreadable_degrades_safely()
    test_morning_completion_plan()
    test_sequenced_sells_settle_before_buys()
    test_morning_aborts_when_positions_unreadable()
    test_morning_dry_run_never_aborts()
    test_morning_crash_recovery_no_duplicate()
    test_morning_partial_prior_orders_remainder()
    test_morning_sell_clamped_to_live_positions()
    test_trade_summary_lists_buys_and_sells()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURES:")
        for f in FAIL:
            print(" -", f)
        sys.exit(1)
    print("PASS test_paper_trade_safety.py")
