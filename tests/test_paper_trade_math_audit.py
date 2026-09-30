"""Mocked regression tests for the paper_trade.py math/safety audit (zero broker calls).

Covers the audit fixes:
- +inf prices rejected everywhere (build_orders / swap / hold builders / client ids)
- duplicate symbols and >100% total weights raise (fail closed)
- per-symbol weight >100% becomes SKIP (bad weight)
- apply_buying_power_guard recomputes shares x price, never trusts Est_Value
- strategy_changes.csv freshness enforced alongside strategy_picks.csv
- morning fill check: SELLs complete before BUYs (explicit order + settle wait),
  BUYs with unusable prices are dropped loudly (never submitted blind),
  BUY cash checks reserve spend cumulatively, malformed pending rows are dropped
  without crashing the run
- reconcile_positions on a hold day returns OK with an empty report (no false DRIFT)

Self-contained: stubs alpaca.trading in sys.modules and defines its own
FakeClient helpers (the old test_paper_trade_safety.py was retired; its
LIVE successor is test_paper_trade_live_safety.py).
Run: PYTHONPATH=. python tests/test_paper_trade_math_audit.py
"""
import json
import os
import sys
import tempfile
os.environ.setdefault("STOCK_ANALYSIS_RUN_LOG", os.path.join(tempfile.gettempdir(), "sa_test_run_log.csv"))  # never the real run log

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

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


class Sort(_Enum):             # alpaca.common.enums.Sort
    DESC = "desc"
    ASC = "asc"


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
_common = types.ModuleType("alpaca.common")
_common_enums = types.ModuleType("alpaca.common.enums")
_common_enums.Sort = Sort
sys.modules.update({"alpaca.common": _common, "alpaca.common.enums": _common_enums})
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

    def get_clock(self):
        return type("Clock", (), {"is_open": True})()

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


def test_build_orders_rejects_inf_price():
    # +inf price: `inf > 0` is True, so a naive positivity check would let it
    # through and the submit would crash on round(inf) (OverflowError).
    targets = pd.DataFrame([
        {"Symbol": "AAA", "Price": float("inf"), "Weight": 0.10},
        {"Symbol": "BBB", "Price": 50.0, "Weight": 0.10},
    ])
    orders = paper_trade.build_orders(targets, 10000, positions={},
                                      statuses={"AAA": "add", "BBB": "add"})
    aaa = orders[orders["Symbol"] == "AAA"].iloc[0]
    check(aaa["Side"] == "SKIP (no price)", f"+inf price -> SKIP (no price), got {aaa['Side']}")
    check(aaa["Shares"] == 0, "+inf price row orders 0 shares")
    check(orders[orders["Symbol"] == "BBB"].iloc[0]["Side"] == "BUY",
          "finite-price symbol still buys")


def test_build_orders_duplicate_symbols_raise():
    targets = pd.DataFrame([
        {"Symbol": "AAA", "Price": 100.0, "Weight": 0.10},
        {"Symbol": "AAA", "Price": 100.0, "Weight": 0.20},
    ])
    try:
        paper_trade.build_orders(targets, 10000, positions={})
        check(False, "duplicate symbols should raise ValueError (ambiguous weights)")
    except ValueError:
        check(True, "duplicate symbols raise ValueError")


def test_build_orders_total_weight_over_100_raises():
    targets = pd.DataFrame([
        {"Symbol": "AAA", "Price": 100.0, "Weight": 0.70},
        {"Symbol": "BBB", "Price": 100.0, "Weight": 0.60},
    ])
    try:
        paper_trade.build_orders(targets, 10000, positions={})
        check(False, "weights summing to 130% should raise ValueError")
    except ValueError:
        check(True, "weights summing to >100% raise ValueError")


def test_build_orders_single_weight_over_100_skip():
    targets = pd.DataFrame([
        {"Symbol": "AAA", "Price": 100.0, "Weight": 1.50},
        {"Symbol": "BBB", "Price": 100.0, "Weight": -0.60},
        {"Symbol": "CCC", "Price": 100.0, "Weight": 0.10},
    ])
    orders = paper_trade.build_orders(targets, 10000, positions={},
                                      statuses={"AAA": "add", "BBB": "add", "CCC": "add"})
    by = {r["Symbol"]: r["Side"] for r in orders.to_dict("records")}
    check(by["AAA"] == "SKIP (bad weight)", f"weight 150% -> SKIP (bad weight), got {by['AAA']}")
    check(by["BBB"] == "SKIP (bad weight)", f"weight -60% -> SKIP (bad weight), got {by['BBB']}")
    check(by["CCC"] == "BUY", f"valid weight still buys, got {by['CCC']}")


def test_buying_power_guard_recomputes_cost_not_est_value():
    # Est_Value understated (stale): the guard must recompute shares x price instead
    # of trusting Est_Value for the early return. Real cost $1,500 > $600 cash, but
    # the fake Est_Value sums to $2 - the old code would have returned uncapped.
    df = _orders_frame()
    df.loc[0, "Est_Value"] = 1.0
    df.loc[1, "Est_Value"] = 1.0
    out = paper_trade.apply_buying_power_guard(df, 600.0, cushion=0)  # pure scaling (cushion tested separately)
    aaa = out[out["Symbol"] == "AAA"].iloc[0]
    bbb = out[out["Symbol"] == "BBB"].iloc[0]
    check((aaa["Shares"], bbb["Shares"]) == (4, 2),
          f"guard scales on recomputed cost (got AAA x{aaa['Shares']}, BBB x{bbb['Shares']})")
    check((aaa["Est_Value"], bbb["Est_Value"]) == (400.0, 200.0),
          "guard recomputes Est_Value from shares x price")


def test_buying_power_guard_invalid_row_skipped_upfront():
    df = _orders_frame()
    df.loc[0, "Price"] = float("inf")
    df.loc[0, "Est_Value"] = 5.0
    out = paper_trade.apply_buying_power_guard(df, 100000.0)
    check(out[out["Symbol"] == "AAA"]["Side"].iloc[0] == "SKIP (no buying power)",
          "+inf-price buy with tiny Est_Value becomes SKIP (no buying power), never submitted")


def test_client_order_id_inf_price_no_crash():
    cid = paper_trade._client_order_id("AAA", "BUY", 10, float("inf"), "20260926", kind="fill")
    check(cid == "live-fill-20260926-BUY-AAA-10-0",
          f"+inf price -> price-cents fall back to 0, no OverflowError ({cid})")


def test_check_signal_freshness_rejects_stale_changes():
    from datetime import date, timedelta
    with tempfile.TemporaryDirectory() as td:
        pk = os.path.join(td, "picks.csv")
        ch = os.path.join(td, "changes.csv")
        today = date.today().isoformat()
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        pd.DataFrame([{"As_Of": today}]).to_csv(pk, index=False)
        pd.DataFrame([{"Date": yesterday, "Symbol": "AAA", "Status": "add"}]).to_csv(ch, index=False)
        try:
            paper_trade.check_signal_freshness(pk, ch)
            check(False, "stale strategy_changes.csv should raise ValueError")
        except ValueError:
            check(True, "stale strategy_changes.csv raises ValueError (fail closed)")
        pd.DataFrame([{"Date": today, "Symbol": "AAA", "Status": "add"}]).to_csv(ch, index=False)
        check(paper_trade.check_signal_freshness(pk, ch) == today,
              "fresh picks + fresh changes pass the freshness check")


def test_morning_buy_no_price_dropped_not_retried():
    # A morning BUY whose price cannot be verified must NOT be submitted blind
    # (the old code skipped the cash check entirely when limit_price was missing).
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pending.json")
        _write_pending(pp, [{"symbol": "AAA", "side": "BUY", "qty": 10, "order_id": None}])
        fake = FakeClient()
        orig = _use_fake(fake)
        try:
            res = paper_trade.complete_unfilled_orders(pending_path=pp, log_csv=None)
        finally:
            paper_trade.paper_trading_client = orig
        check(fake.submitted == [], "morning: BUY with no price is never submitted")
        check(not res.empty and "FAILED: no usable price" in res["Status"].iloc[0],
              "morning: BUY with no price is FAILED/dropped loudly")
        check(not os.path.exists(pp), "morning: unpriceable row dropped, not kept for retry")


def test_morning_sells_complete_before_buys():
    # The BUY is listed first in the pending file: SELLs must still complete first,
    # and this run's SELL completions are waited on before the BUY's cash check.
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pending.json")
        _write_pending(pp, [
            {"symbol": "BBB", "side": "BUY", "qty": 5, "limit_price": 100.0, "order_id": None},
            {"symbol": "AAA", "side": "SELL", "qty": 10, "limit_price": 50.0, "order_id": None},
        ])
        fake = FakeClient()
        fake.positions = [("AAA", 10)]
        wait_calls = []
        orig_wait = paper_trade._wait_for_terminal_all
        paper_trade._wait_for_terminal_all = (
            lambda client, ids, **kw: wait_calls.append(list(ids)) or {})
        orig = _use_fake(fake)
        try:
            paper_trade.complete_unfilled_orders(pending_path=pp, log_csv=None)
        finally:
            paper_trade.paper_trading_client = orig
            paper_trade._wait_for_terminal_all = orig_wait
        syms = [r.symbol for r in fake.submitted]
        check(syms == ["AAA", "BBB"],
              f"morning: SELL completes before BUY regardless of file order (got {syms})")
        check(wait_calls == [["fake-1"]],
              f"morning: this run's SELL completion is waited on before the BUY (got {wait_calls})")


def test_morning_buy_cumulative_cash_reserved():
    # Two BUYs each fit in cash alone but not together: the second buys only the part that
    # still fits after the first (1% cushion), never more than the cash.
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pending.json")
        _write_pending(pp, [
            {"symbol": "AAA", "side": "BUY", "qty": 60, "limit_price": 100.0, "order_id": None},
            {"symbol": "BBB", "side": "BUY", "qty": 60, "limit_price": 100.0, "order_id": None},
        ])
        fake = FakeClient()
        fake.buying_power = 10000.0  # cash defaults to buying_power in the fake
        orig = _use_fake(fake)
        try:
            paper_trade.complete_unfilled_orders(pending_path=pp, log_csv=None)
        finally:
            paper_trade.paper_trading_client = orig
        got = [(r.symbol, r.qty) for r in fake.submitted]
        check(got == [("AAA", 60), ("BBB", 39.0)],
              f"morning: first BUY in full ($6,060 with cushion), second cut to what is left: 3,940 / 101 = 39 (got {got})")
        check(sum(q * 100.0 * 1.01 for _, q in got) <= 10000.0, "morning: both BUYs together stay within buying power")
        check(not os.path.exists(pp), "morning: the part that did not fit is not retried (pending file removed)")


def test_morning_malformed_row_dropped():
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pending.json")
        _write_pending(pp, [
            {"symbol": "AAA", "side": "BUY", "qty": "abc", "order_id": None},
            {"symbol": "BBB", "side": "SELL", "qty": 5, "limit_price": 50.0, "order_id": None},
        ])
        fake = FakeClient()
        fake.positions = [("BBB", 5)]
        orig = _use_fake(fake)
        try:
            res = paper_trade.complete_unfilled_orders(pending_path=pp, log_csv=None)
        finally:
            paper_trade.paper_trading_client = orig
        check(any("FAILED: malformed" in s for s in res["Status"]),
              "morning: malformed qty row FAILED loudly instead of crashing")
        check(any("COMPLETED" in s for s in res["Status"]),
              "morning: valid SELL row still completes alongside the bad row")
        check(len(fake.submitted) == 1 and fake.submitted[0].symbol == "BBB",
              "morning: only the valid row is submitted")
        check(not os.path.exists(pp), "morning: malformed row dropped, pending file cleaned up")


def test_reconcile_hold_returns_ok_empty():
    # On a hold day nothing was ordered: reconciliation must return OK with an empty
    # report, not flag every held position as DRIFT against a 0% target.
    orig_targets = paper_trade.load_targets
    orig_live = paper_trade.get_live_positions_and_equity
    paper_trade.load_targets = lambda source="auto": (
        pd.DataFrame(columns=["Symbol", "Weight", "Price"]),
        {"source": "hold", "as_of": "2026-09-26"})
    paper_trade.get_live_positions_and_equity = lambda: ({"AAA": 10}, 100000.0, 50000.0, 100000.0)
    try:
        report, ok = paper_trade.reconcile_positions("hold")
    finally:
        paper_trade.load_targets = orig_targets
        paper_trade.get_live_positions_and_equity = orig_live
    check(ok is True, "reconcile on a hold day returns ok=True")
    check(report.empty, "reconcile on a hold day returns an empty report (no false DRIFT)")


if __name__ == "__main__":
    test_build_orders_rejects_inf_price()
    test_build_orders_duplicate_symbols_raise()
    test_build_orders_total_weight_over_100_raises()
    test_build_orders_single_weight_over_100_skip()
    test_buying_power_guard_recomputes_cost_not_est_value()
    test_buying_power_guard_invalid_row_skipped_upfront()
    test_client_order_id_inf_price_no_crash()
    test_check_signal_freshness_rejects_stale_changes()
    test_morning_buy_no_price_dropped_not_retried()
    test_morning_sells_complete_before_buys()
    test_morning_buy_cumulative_cash_reserved()
    test_morning_malformed_row_dropped()
    test_reconcile_hold_returns_ok_empty()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURES:")
        for f in FAIL:
            print(" -", f)
        sys.exit(1)
    print("PASS test_paper_trade_math_audit.py")
