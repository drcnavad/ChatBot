"""Mocked tests (zero broker calls, zero network) for fractional sizing and the morning fill check.

Sizing: 2-decimal shares rounded DOWN for buys/trims, exact held qty for full exits, WHOLE shares
after hours (the fractional rest goes to the next morning's regular-hours market order).
Rebalance: unowned 'hold' picks are bought like 'add'; owned 'hold' picks are not topped up;
Mon/Wed auto mode trades only replacements/exits (never a mass buy).
Fill check: partial fill, full fill, no fill, still-open order (cancel confirmed / not confirmed),
crash between cancel and replace, crash after submit, rerun of the same day, market closed,
open order already on the broker, unreadable broker orders, $1 minimum.

Run: cd <folder> && python3 tests/test_fill_check_fractional.py
"""
import json
import os
import sys
import tempfile
os.environ.setdefault("STOCK_ANALYSIS_RUN_LOG", os.path.join(tempfile.gettempdir(), "sa_test_run_log.csv"))  # never the real run log
import types
from enum import Enum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class TimeInForce(Enum):
    DAY = "day"

class QueryOrderStatus(Enum):
    ALL = "all"
    OPEN = "open"

class Sort(Enum):              # alpaca.common.enums.Sort
    ASC = "asc"
    DESC = "desc"

class _Req:
    def __init__(self, **kw):
        self.__dict__.update(kw)

class MarketOrderRequest(_Req): pass
class LimitOrderRequest(_Req): pass
class GetOrdersRequest(_Req): pass

mods = {n: types.ModuleType(n) for n in ("alpaca", "alpaca.trading", "alpaca.trading.client",
                                          "alpaca.trading.enums", "alpaca.trading.requests", "alpaca.common",
                                          "alpaca.common.enums", "dotenv")}
mods["alpaca.trading.enums"].__dict__.update(OrderSide=OrderSide, TimeInForce=TimeInForce,
                                             QueryOrderStatus=QueryOrderStatus)
mods["alpaca.common.enums"].Sort = Sort
mods["alpaca.trading.requests"].__dict__.update(MarketOrderRequest=MarketOrderRequest,
                                                LimitOrderRequest=LimitOrderRequest,
                                                GetOrdersRequest=GetOrdersRequest)
mods["alpaca.trading.client"].TradingClient = type("TradingClient", (), {})
mods["dotenv"].load_dotenv = lambda *a, **k: None
sys.modules.update(mods)

import pandas as pd
import paper_trade as pt
import fake_quotes
QUOTES = fake_quotes.install(pt)   # every symbol quoted 99.99 / 100.01 (tight, fresh)

NOTES = []  # every run-log row (status, message), checked at the end
pt.log_event = lambda run, status, moved, message, details="": NOTES.append((status, message))
PASS, FAIL = [], []

def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""))

class Crash(BaseException):
    """Simulates the process dying (not caught by the code's `except Exception`)."""

class Order:
    def __init__(self, id, symbol, side, qty, status, filled_qty=0, client_order_id=""):
        self.id, self.symbol, self.side, self.qty = id, symbol, side, qty
        self.status, self.filled_qty, self.client_order_id = status, filled_qty, client_order_id

class Broker:
    """Fake Alpaca client. crash = None | ("before", n) | ("after", n): die on the n-th submit
    (0-based) before it reaches the broker, or right after the broker accepted it."""
    def __init__(self, positions=None, orders=(), is_open=True, cancel_works=True, crash=None,
                 orders_readable=True):
        self.positions = dict(positions or {})
        self.orders = {o.id: o for o in orders}
        self.is_open, self.cancel_works, self.crash = is_open, cancel_works, crash
        self.orders_readable = orders_readable
        self.submitted, self.canceled, self.n_submit = [], [], 0

    def get_clock(self): return types.SimpleNamespace(is_open=self.is_open)
    bp = 1000000.0  # drops when a BUY is submitted, like Alpaca's buying power

    def get_account(self): return types.SimpleNamespace(buying_power=str(self.bp), equity="1000000", cash=str(self.bp))
    def get_all_positions(self): return [types.SimpleNamespace(symbol=s, qty=str(q)) for s, q in self.positions.items()]
    def get_order_by_id(self, oid): return self.orders[oid]

    def get_orders(self, req=None):
        if not self.orders_readable:
            raise ConnectionError("broker unreachable")
        return list(self.orders.values())

    def cancel_order_by_id(self, oid):
        self.canceled.append(oid)
        if self.cancel_works and self.orders[oid].status not in pt.TERMINAL_STATUSES:
            self.orders[oid].status = "canceled"

    def submit_order(self, req):
        n, self.n_submit = self.n_submit, self.n_submit + 1
        if self.crash == ("before", n):
            raise Crash()
        side = "SELL" if req.side == OrderSide.SELL else "BUY"
        o = Order(f"m{n}", req.symbol, side, req.qty, "filled", req.qty, req.client_order_id)
        self.orders[o.id] = o
        self.submitted.append(req)
        self.positions[req.symbol] = self.positions.get(req.symbol, 0) + (req.qty if side == "BUY" else -req.qty)
        if side == "BUY":
            self.bp -= req.qty * 100.0  # fake fill price
        if self.crash == ("after", n):
            raise Crash()
        return o

def write_pending(rows, evening_date="2026-10-02"):
    path = os.path.join(tempfile.mkdtemp(), "live_pending_orders.json")
    json.dump({"evening_date": evening_date, "submitted_at_ct": "x", "target_source": "auto",
               "as_of": evening_date, "orders": rows}, open(path, "w"))
    return path

def run(broker, path):
    pt.paper_trading_client = lambda: broker
    return pt.complete_unfilled_orders(pending_path=path, log_csv=None)

def status_of(res, sym):
    return " | ".join(res.loc[res["Symbol"] == sym, "Status"].astype(str))

# ------------------------------------------------------------------ sizing
check("floor2: 500/139.66 -> 3.58", pt._floor2(500 / 139.66) == 3.58, pt._floor2(500 / 139.66))
check("floor2: 3.5799 -> 3.57 (never up)", pt._floor2(3.5799) == 3.57)
check("floor2: 3.58 float noise kept at 3.58", pt._floor2(0.1 * 35.8) == 3.58, pt._floor2(0.1 * 35.8))

targets = pd.DataFrame({"Symbol": ["ANET", "AMD", "MRK", "TWLO", "XYZ"], "Weight": [0.05, 0.08, 0.05, 0.20, 0.05],
                        "Price": [139.66, 200.0, 90.0, 100.0, 50.0]})
orders = pt.build_orders(targets, 10000, positions={"ORCL": 1.054, "MRK": 10, "TWLO": 1}, prices={"ORCL": 280.0},
                         fractional=True, statuses={"ANET": "add", "AMD": "hold", "MRK": "add", "TWLO": "hold"})
row = {r.Symbol: r for r in orders.itertuples()}
check("plan: BUY 2-decimal rounded down (ANET 3.58)", row["ANET"].Side == "BUY" and row["ANET"].Shares == 3.58,
      (row["ANET"].Side, row["ANET"].Shares))
check("plan: unowned 'hold' pick is bought up to its weight (AMD 800/200 = 4.00)",
      row["AMD"].Side == "BUY" and row["AMD"].Shares == 4.0, (row["AMD"].Side, row["AMD"].Shares))
check("plan: owned underweight 'hold' pick is topped up to target (TWLO 1 -> 20 shares, buy 19)",
      row["TWLO"].Side == "BUY" and row["TWLO"].Shares == 19.0, (row["TWLO"].Side, row["TWLO"].Shares))
check("plan: unknown status is never bought (fail closed)", row["XYZ"].Side == "HOLD")
whole = pt.build_orders(targets, 10000, positions={}, statuses={"AMD": "hold"})
check("plan: unowned 'hold' bought in whole-share mode too", whole.set_index("Symbol").loc["AMD", "Shares"] == 4)
check("plan: full exit sells exact held qty (ORCL 1.054)", row["ORCL"].Side == "SELL" and row["ORCL"].Shares == 1.054,
      row["ORCL"].Shares)
check("plan: trim rounded down to 2 decimals (MRK 10 -> 5.55, sell 4.45)",
      row["MRK"].Side == "SELL" and row["MRK"].Shares == 4.45, row["MRK"].Shares)

# ---- Mon/Wed auto mode: only the strategy's replacements/exits - never a mass buy of unowned picks
def picks_csv(as_of, last_reb):
    rows = [{"As_Of": as_of, "Last_Rebalance": last_reb, "Last_Decision": last_reb, "Strategy": "test",
             "Symbol": f"S{i}", "Strategy_Weight": 0.1, "Provisional_Weight": 0.1, "Close": 100.0}
            for i in range(10)]
    d = tempfile.mkdtemp()
    pd.DataFrame(rows).to_csv(os.path.join(d, "picks.csv"), index=False)
    pd.DataFrame([{"Date": as_of, "Symbol": "OWN", "Close": 50.0}]).to_csv(os.path.join(d, "sig.csv"), index=False)
    return d

def midweek_csv(d, as_of, rows):
    p = os.path.join(d, "mid.csv")
    pd.DataFrame([{"Event": "mid-week check", "Event_Date": as_of, "Action": a, "Sell": s, "Buy": b,
                   "Weight_%": 10.0, "Message": "m"} for a, s, b in rows] or
                 [{"Event": "mid-week check", "Event_Date": as_of, "Action": "HOLD", "Sell": None, "Buy": None,
                   "Weight_%": 0.0, "Message": "nothing to trade"}]).to_csv(p, index=False)
    return p

pt.latest_signal_status = lambda *a, **k: {f"S{i}": "hold" for i in range(10)}  # every pick 'hold'
d = picks_csv("2026-09-30", "2026-09-25")
kw = dict(picks_csv=os.path.join(d, "picks.csv"), signal_csv=os.path.join(d, "sig.csv"), fractional=True)
o, m = pt.plan_orders("auto", 10000, {"OWN": 5}, midweek_csv=midweek_csv(d, "2026-09-30", []), **kw)[:2]
check("Wednesday quiet check: auto -> 'hold', nothing bought or sold",
      m["source"] == "hold" and set(o["Side"]) == {"HOLD"}, (m["source"], list(o["Side"])))
o, m = pt.plan_orders("auto", 10000, {"OWN": 5}, midweek_csv=midweek_csv(d, "2026-09-30", [("REPLACE", "OWN", "S3")]), **kw)[:2]
buys = list(o.loc[o.Side == "BUY", "Symbol"])
check("Wednesday replacement: only it trades (sell OWN, buy S3), no other picks bought",
      m["source"] == "midweek" and buys == ["S3"] and list(o.loc[o.Side == "SELL", "Symbol"]) == ["OWN"],
      (m["source"], buys))
d = picks_csv("2026-10-02", "2026-10-02")
o, m = pt.plan_orders("auto", 10000, {}, midweek_csv=midweek_csv(d, "2026-10-02", []),
                      picks_csv=os.path.join(d, "picks.csv"), signal_csv=os.path.join(d, "sig.csv"), fractional=True)[:2]
check("Friday rebalance: auto -> 'provisional', all 10 unowned 'hold' picks bought",
      m["source"] == "provisional" and (o["Side"] == "BUY").sum() == 10, (m["source"], list(o["Side"])))

g = pd.DataFrame([{"Symbol": "A", "Side": "BUY", "Shares": 3.58, "Price": 100.0, "Est_Value": 358.0,
                   "Current_Shares": 0, "Target_Shares": 3.58, "Target_Weight_%": 5, "Target_Value": 358},
                  {"Symbol": "B", "Side": "BUY", "Shares": 2.0, "Price": 100.0, "Est_Value": 200.0,
                   "Current_Shares": 0, "Target_Shares": 2, "Target_Weight_%": 5, "Target_Value": 200}])
out = pt.apply_buying_power_guard(g, 300.0, fractional=True, cushion=0)  # pure scaling (cushion tested separately)
check("guard: fractional scale-down rounds DOWN to 2 decimals",
      list(out["Shares"]) == [1.92, 1.07] and (out["Shares"] * out["Price"]).sum() <= 300, list(out["Shares"]))
check("guard: whole-share mode unchanged", list(pt.apply_buying_power_guard(g, 300.0)["Shares"]) == [1, 1])

ext = pd.DataFrame([{"Symbol": "ORCL", "Side": "SELL", "Shares": 1.054, "Price": 280.0},
                    {"Symbol": "ANET", "Side": "BUY", "Shares": 3.58, "Price": 139.66},
                    {"Symbol": "TEM", "Side": "BUY", "Shares": 0.5, "Price": 60.0}])
b, rec = Broker(positions={"ORCL": 1.054}), []
res = pt.submit_paper_extended(ext, positions={"ORCL": 1.054}, record=rec.append, client=b, order_date="20261002")
lim = {r.symbol: r for r in b.submitted}
check("after hours: limit orders in WHOLE shares", lim["ORCL"].qty == 1 and lim["ANET"].qty == 3
      and all(isinstance(r.qty, int) and isinstance(r, LimitOrderRequest) and r.extended_hours for r in b.submitted),
      [(r.symbol, r.qty) for r in b.submitted])
recs = {r["symbol"]: r for r in rec}
check("after hours: pending keeps planned qty + whole qty sent",
      recs["ANET"]["qty"] == 3.58 and recs["ANET"]["order_qty"] == 3 and recs["ORCL"]["exit"] is True
      and recs["ORCL"]["qty"] == 1.054, recs)
check("after hours: <1 whole share is staged for the morning, not sent",
      "TEM" not in lim and recs["TEM"]["order_id"] is None and "STAGED" in status_of(res, "TEM"))

b = Broker(positions={"ORCL": 1.054})
pt.submit_paper(ext.iloc[:2], positions={"ORCL": 1.054}, client=b, order_date="20261005")
mk = {r.symbol: r.qty for r in b.submitted}
check("regular hours: market orders fractional (exit exact 1.054, buy 3.58)",
      mk == {"ORCL": 1.054, "ANET": 3.58} and all(isinstance(r, MarketOrderRequest) for r in b.submitted), mk)

# ------------------------------------------------------------------ morning fill check
def eve(sym, side, qty, order_qty, status, filled, **extra):
    o = Order(f"e-{sym}", sym, side, order_qty, status, filled, f"live-20261002-{side}-{sym}-{order_qty}-1")
    row = {"symbol": sym, "side": side, "qty": qty, "order_qty": order_qty, "limit_price": 100.0,
           "order_id": o.id, **extra}
    return o, row

# partial fill
o, r = eve("ANET", "BUY", 3.58, 3, "expired", 2)
b = Broker(orders=[o]); p = write_pending([r]); res = run(b, p)
check("partial fill: only the unfilled 1.58 sent, as a DAY limit at the ask + 0.05%",
      [(x.symbol, x.qty) for x in b.submitted] == [("ANET", 1.58)] and isinstance(b.submitted[0], LimitOrderRequest)
      and b.submitted[0].limit_price == 100.07 and not getattr(b.submitted[0], "extended_hours", False),
      [(x.symbol, x.qty, getattr(x, "limit_price", None)) for x in b.submitted])
check("partial fill: deterministic id uses the evening date",
      b.submitted[0].client_order_id.startswith("live-fill-20261002-BUY-ANET-1-"), b.submitted[0].client_order_id)
check("partial fill: pending file removed when done", not os.path.exists(p))

# full fill (whole plan)
o, r = eve("MRK", "BUY", 6, 6, "filled", 6)
b = Broker(orders=[o]); res = run(b, write_pending([r]))
check("full fill: nothing sent", not b.submitted and status_of(res, "MRK").startswith("FILLED"), status_of(res, "MRK"))

# full fill of the whole-share part -> fractional rest completed
o, r = eve("ANET", "BUY", 3.58, 3, "filled", 3)
b = Broker(orders=[o]); run(b, write_pending([r]))
check("whole part filled: fractional rest 0.58 sent", [x.qty for x in b.submitted] == [0.58], [x.qty for x in b.submitted])

# no fill
o, r = eve("ANET", "BUY", 3.58, 3, "expired", 0)
b = Broker(orders=[o]); run(b, write_pending([r]))
check("no fill: full planned 3.58 sent", [x.qty for x in b.submitted] == [3.58])

# full exit: evening sold the whole share, morning sells the exact fractional rest
o, r = eve("ORCL", "SELL", 1.054, 1, "filled", 1, exit=True)
b = Broker(positions={"ORCL": 0.054}, orders=[o]); run(b, write_pending([r]))
check("exit: exact held rest 0.054 sold (no dust)", [(x.symbol, x.qty) for x in b.submitted] == [("ORCL", 0.054)],
      [(x.symbol, x.qty) for x in b.submitted])

# still-open evening order: cancel confirmed first
o, r = eve("ANET", "BUY", 3.58, 3, "partially_filled", 1)
b = Broker(orders=[o]); run(b, write_pending([r]))
check("open order: canceled + confirmed before replacing, only 2.58 sent",
      b.canceled == ["e-ANET"] and [x.qty for x in b.submitted] == [2.58], (b.canceled, [x.qty for x in b.submitted]))

o, r = eve("ANET", "BUY", 3.58, 3, "partially_filled", 1)
b = Broker(orders=[o], cancel_works=False); p = write_pending([r]); res = run(b, p)
check("open order: cancel NOT confirmed -> nothing sent, kept for retry",
      not b.submitted and os.path.exists(p) and "FAILED" in status_of(res, "ANET"), status_of(res, "ANET"))

# crash between cancel and replace, then rerun
o, r = eve("ANET", "BUY", 3.58, 3, "partially_filled", 1)
b = Broker(orders=[o], crash=("before", 0)); p = write_pending([r])
try:
    run(b, p)
except Crash:
    pass
check("crash before replace: evening order canceled, nothing sent", o.status == "canceled" and not b.submitted)
b.crash = None
run(b, p)
check("crash before replace: rerun sends the 2.58 rest exactly once", [x.qty for x in b.submitted] == [2.58],
      [x.qty for x in b.submitted])
run(b, p)
check("crash before replace: third run sends nothing", len(b.submitted) == 1)

# crash right after the broker accepted the replacement (before any local record)
o, r = eve("ANET", "BUY", 3.58, 3, "expired", 1)
b = Broker(orders=[o], crash=("after", 0)); p = write_pending([r])
try:
    run(b, p)
except Crash:
    pass
b.crash = None
res = run(b, p)
check("crash after submit: rerun finds it by client id, no duplicate",
      len(b.submitted) == 1 and "ALREADY COMPLETED" in status_of(res, "ANET"), status_of(res, "ANET"))

# rerun of the same day after a crash mid-batch: completed row never repeated
o1, r1 = eve("MU", "SELL", 5, 5, "expired", 0)
o2, r2 = eve("ANET", "BUY", 3.58, 3, "expired", 0)
b = Broker(positions={"MU": 5}, orders=[o1, o2], crash=("before", 1)); p = write_pending([r1, r2])
try:
    run(b, p)
except Crash:
    pass
saved = {x["symbol"]: x.get("completed_order_id") for x in json.load(open(p))["orders"]}
check("rerun: first completion saved right away", saved.get("MU") and not saved.get("ANET"), saved)
b.crash = None
res = run(b, p)
check("rerun: MU not repeated, ANET sent once", [(x.symbol, x.qty) for x in b.submitted] == [("MU", 5), ("ANET", 3.58)]
      and "ALREADY COMPLETED" in status_of(res, "MU"), [(x.symbol, x.qty) for x in b.submitted])
res = run(b, p)
check("rerun: a third run finds nothing to do", res.empty and len(b.submitted) == 2)

# Alpaca rejects the market order every time: retried at most MAX_FILL_TRIES checks, then dropped (never forever)
class Rejecting(Broker):
    def submit_order(self, req):
        raise RuntimeError("insufficient buying power")

o, r = eve("ANET", "BUY", 3.58, 3, "expired", 0)
b = Rejecting(orders=[o]); p = write_pending([r])
NOTES.clear()
res1 = run(b, p)
check("rejected market order: kept for retry (try 1), alert says it retries - don't place by hand",
      os.path.exists(p) and "retrying" in status_of(res1, "ANET")
      and any(t == "warning" and "don't place" in m for t, m in NOTES), (status_of(res1, "ANET"), NOTES))
run(b, p)
NOTES.clear()
res3 = run(b, p)
check(f"rejected market order: dropped after {pt.MAX_FILL_TRIES} tries, alert says place by hand",
      not os.path.exists(p) and "gave up" in status_of(res3, "ANET")
      and any(t == "failed" and "by hand" in m and "No money" in m for t, m in NOTES), (status_of(res3, "ANET"), NOTES))

# market closed (holiday / Mac woke after the close)
o, r = eve("ANET", "BUY", 3.58, 3, "expired", 0)
b = Broker(orders=[o], is_open=False); p = write_pending([r]); res = run(b, p)
check("market closed: nothing sent, pending kept", not b.submitted and os.path.exists(p)
      and status_of(res, "ANET").startswith("WAITING"), status_of(res, "ANET"))

# an open order for the symbol already on the broker
o, r = eve("ANET", "BUY", 3.58, 3, "expired", 0)
manual = Order("x1", "ANET", "BUY", 2, "new", 0, "manual")
b = Broker(orders=[o, manual]); p = write_pending([r]); res = run(b, p)
check("open order on broker: nothing sent, kept for retry",
      not b.submitted and os.path.exists(p) and "open ANET order" in status_of(res, "ANET"), status_of(res, "ANET"))

# broker orders unreadable -> abort (cannot rule out duplicates)
o, r = eve("ANET", "BUY", 3.58, 3, "expired", 0)
b = Broker(orders=[o], orders_readable=False); p = write_pending([r])
try:
    run(b, p)
    check("unreadable broker orders: abort", False)
except RuntimeError:
    check("unreadable broker orders: abort, nothing sent, pending kept", not b.submitted and os.path.exists(p))

# buying power: Alpaca lowers it for each submitted BUY - earlier BUYs must not be counted twice
o1, r1 = eve("AAA", "BUY", 4, 4, "expired", 0)
o2, r2 = eve("BBB", "BUY", 4, 4, "expired", 0)
b = Broker(orders=[o1, o2]); b.bp = 1000.0  # 2 x 4 x $100 x 1.05 = $840 fits in $1,000
run(b, write_pending([r1, r2]))
check("buying power: two BUYs that fit together are both sent (no double counting)",
      [x.symbol for x in b.submitted] == ["AAA", "BBB"], [x.symbol for x in b.submitted])

# $1 minimum for a fractional BUY rest
o, r = eve("ANET", "BUY", 3.01, 3, "filled", 3)
r["limit_price"] = 50.0
QUOTES.quotes["ANET"] = (49.99, 50.01)     # 0.01 share x $50.04 limit = $0.50
b = Broker(orders=[o]); p = write_pending([r]); res = run(b, p)
check("rest under $1: not ordered, not retried", not b.submitted and not os.path.exists(p)
      and "under $1" in status_of(res, "ANET"), status_of(res, "ANET"))

# ------------------------------------------------------------------ earnings-day stop: no buy back until after its last earnings-day session
stop_state = os.path.join(tempfile.mkdtemp(), "earnings_stop_state.json")
json.dump({"sold": {"S1|2026-10-01": {"at": "2026-09-30T10:00:00-05:00", "react": "2026-10-02"},
                    "S3|2026-10-01": {"at": "2026-09-29T10:00:00-05:00", "react": "2026-10-02"},
                    "ANET|2099-01-01": {"at": "2026-09-29T10:00:00-05:00", "react": "2099-01-02"}}}, open(stop_state, "w"))
saved_state, pt.EARNINGS_STOP_STATE = pt.EARNINGS_STOP_STATE, stop_state
check("stop block: through the reaction day only", set(pt.earnings_stop_blocked("2026-10-02")) == {"S1", "S3", "ANET"}
      and set(pt.earnings_stop_blocked("2026-10-03")) == {"ANET"} and pt.earnings_stop_blocked("2026-10-02", "/nonexistent") == {})
d = picks_csv("2026-10-02", "2026-10-02")
o = pt.plan_orders("auto", 10000, {}, midweek_csv=midweek_csv(d, "2026-10-02", []),
                   picks_csv=os.path.join(d, "picks.csv"), signal_csv=os.path.join(d, "sig.csv"), fractional=True)[0]
check("Friday rebalance: the stopped S1 and S3 are not bought back (SKIP), the other 8 picks are",
      all(o.set_index("Symbol").Side[x].startswith("SKIP (sold by the earnings-day stop") for x in ("S1", "S3"))
      and (o["Side"] == "BUY").sum() == 8, list(o["Side"]))
o = pt.plan_orders("auto", 10000, {"S1": 0.4}, midweek_csv=midweek_csv(d, "2026-10-02", []),
                   picks_csv=os.path.join(d, "picks.csv"), signal_csv=os.path.join(d, "sig.csv"), fractional=True)[0]
check("Friday rebalance: a leftover S1 fraction is not topped up", o.set_index("Symbol").Side["S1"] == "HOLD")
d = picks_csv("2026-09-30", "2026-09-25")
o = pt.plan_orders("auto", 10000, {"OWN": 5}, midweek_csv=midweek_csv(d, "2026-09-30", [("REPLACE", "OWN", "S3")]),
                   picks_csv=os.path.join(d, "picks.csv"), signal_csv=os.path.join(d, "sig.csv"), fractional=True)[0]
check("Wednesday replacement into a stopped stock: no BUY (SKIP), its sell still goes",
      not (o["Side"] == "BUY").any() and o.set_index("Symbol").Side["S3"].startswith("SKIP") and list(o.loc[o.Side == "SELL", "Symbol"]) == ["OWN"],
      list(zip(o.Symbol, o.Side)))
o, r = eve("ANET", "BUY", 3.58, 3, "expired", 0)
b = Broker(orders=[o]); p = write_pending([r]); res = run(b, p)
check("9 AM fill check: a BUY rest of a stopped stock is dropped, nothing sent", not b.submitted and "earnings-day stop" in status_of(res, "ANET"),
      status_of(res, "ANET"))
saved_rec = (pt.load_targets, pt.get_live_positions_and_equity, pt.latest_prices)
pt.load_targets = lambda *a, **k: (pd.DataFrame({"Symbol": ["ANET", "MRK"], "Weight": [0.1, 0.1], "Price": [100.0, 100.0]}), {"source": "current"})
pt.get_live_positions_and_equity = lambda: ({"MRK": 100}, 100000.0, None, None)
pt.latest_prices = lambda syms, *a, **k: {}
rep, ok = pt.reconcile_positions(symbols={"ANET", "MRK"})
check("reconciliation: a stock sold by the stop is expected at 0% (no false 'below planned weight' warning)",
      ok is True and rep.set_index("Symbol").Status["ANET"] == "OK", rep.to_dict("records"))
pt.load_targets, pt.get_live_positions_and_equity, pt.latest_prices = saved_rec
pt.EARNINGS_STOP_STATE = saved_state

# ------------------------------------------------------------------ your do-not-buy / do-not-sell lists (sector_mapping)
import sector_mapping as sm
sm.do_not_buy, sm.do_not_sell = ["s2"], ["OWN", "S5"]   # lower case is fine
d = picks_csv("2026-10-02", "2026-10-02")
o, m = pt.plan_orders("auto", 10000, {"OWN": 20, "S5": 1}, midweek_csv=midweek_csv(d, "2026-10-02", []),
                      picks_csv=os.path.join(d, "picks.csv"), signal_csv=os.path.join(d, "sig.csv"), fractional=True)[:2]
sd = o.set_index("Symbol").Side
check("Friday: kept OWN ($1,000 = 10%) comes out of the picks: each 10% pick scaled to 9% ($900), source +kept",
      m["kept_outside"] == {"OWN": 0.1} and m["source"].endswith("+kept") and abs(m["invested"] - 0.9) < 1e-9
      and abs(o.set_index("Symbol").Est_Value["S0"] - 900) < 1, (m["kept_outside"], m["source"], m["invested"],
                                                                 o[["Symbol", "Side", "Est_Value"]].values.tolist()))
check("reconcile skips a +kept Friday (the account differs from Provisional_Weight on purpose)",
      pt.reconcile_positions("provisional+kept")[0].empty and pt.reconcile_positions("provisional+kept")[1] is True)
check("Friday: do-not-buy S2 is a SKIP row; non-pick OWN (do-not-sell) is not sold; S5 (do-not-sell) can still be topped up",
      sd["S2"] == "SKIP (on your do-not-buy list)" and sd["OWN"] == "SKIP (on your do-not-sell list)" and sd["S5"] == "BUY"
      and not (o["Side"] == "SELL").any() and (o["Side"] == "BUY").sum() == 9, list(zip(o.Symbol, o.Side)))
d = picks_csv("2026-09-30", "2026-09-25")
o = pt.plan_orders("auto", 10000, {"OWN": 5}, midweek_csv=midweek_csv(d, "2026-09-30", [("REPLACE", "OWN", "S3")]),
                   picks_csv=os.path.join(d, "picks.csv"), signal_csv=os.path.join(d, "sig.csv"), fractional=True)[0]
check("Wednesday replacement out of a do-not-sell stock: neither its sell nor its paired buy is sent",
      not o["Side"].isin(["BUY", "SELL"]).any(), list(zip(o.Symbol, o.Side)))
o = pt.plan_orders("auto", 10000, {"S9": 5}, midweek_csv=midweek_csv(d, "2026-09-30", [("REPLACE", "S9", "S2")]),
                   picks_csv=os.path.join(d, "picks.csv"), signal_csv=os.path.join(d, "sig.csv"), fractional=True)[0]
check("Wednesday replacement into a do-not-buy stock: its sell goes, the buy is a SKIP row",
      list(o.loc[o.Side == "SELL", "Symbol"]) == ["S9"] and o.set_index("Symbol").Side["S2"] == "SKIP (on your do-not-buy list)",
      list(zip(o.Symbol, o.Side)))
o1, r1 = eve("S2", "BUY", 3.58, 3, "expired", 0)
o2, r2 = eve("OWN", "SELL", 5, 5, "expired", 0)
b = Broker(orders=[o1, o2], positions={"OWN": 5}); p = write_pending([r1, r2]); res = run(b, p)
check("9 AM fill check: a do-not-buy BUY rest and a do-not-sell SELL rest are dropped, nothing sent",
      not b.submitted and "do-not-buy" in status_of(res, "S2") and "do-not-sell" in status_of(res, "OWN"),
      (status_of(res, "S2"), status_of(res, "OWN")))
sm.do_not_buy, sm.do_not_sell = [], []

bad = [n for n in NOTES if n[0] not in ("ok", "warning", "failed") or " pp" in n[1] or "DRIFT" in n[1]]
check(f"all {len(NOTES)} run-log rows written here have a valid status and plain words", NOTES and not bad, bad)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
