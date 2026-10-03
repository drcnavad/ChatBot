"""Mocked tests (zero broker calls, zero network, no pop-ups) for the Friday rebalance and cash rules (2026-09-28):
- 1-percentage-point no-trade band; owned 'hold' picks trimmed / topped up to target;
- buys sized from free cash with a 1% cushion: the part that fits is bought, and in sequence never more than the cash;
- earnings: an unowned pick with earnings within 5 days is not bought, an owned one is not topped up (a trim still goes;
  nothing is ever sold because of earnings);
- the fill check sends only in regular hours, a daytime catch-up's orders at once, and drops orders whose decision was
  superseded (next decision slot passed); no duplicate orders on a retry.
Run: cd <folder> && python3 tests/test_rebalance_rules.py
"""
import json
import os
import sys
import tempfile
import types
from datetime import datetime
from enum import Enum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("STOCK_ANALYSIS_RUN_LOG", os.path.join(tempfile.gettempdir(), "sa_test_run_log.csv"))  # never the real run log


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


class MarketOrderRequest(_Req): pass
class LimitOrderRequest(_Req): pass


mods = {n: types.ModuleType(n) for n in ("alpaca", "alpaca.trading", "alpaca.trading.client",
                                          "alpaca.trading.enums", "alpaca.trading.requests", "alpaca.common",
                                          "alpaca.common.enums", "dotenv")}
mods["alpaca.trading.enums"].__dict__.update(OrderSide=OrderSide, TimeInForce=_E, QueryOrderStatus=_E)
mods["alpaca.common.enums"].Sort = _E
mods["alpaca.trading.requests"].__dict__.update(MarketOrderRequest=MarketOrderRequest, LimitOrderRequest=LimitOrderRequest,
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
    print(("PASS " if cond else "FAIL ") + name + (f" ({detail})" if not cond else ""))


def plan(weight, held, price=100.0, status="hold", account=10000, blocked=None):
    t = pd.DataFrame({"Symbol": ["AMD"], "Weight": [weight], "Price": [price]})
    o = pt.build_orders(t, account, positions={"AMD": held} if held else {}, fractional=True,
                        statuses={"AMD": status}, blocked=blocked)
    r = o.iloc[0]
    return r.Side, float(r.Shares)


# ------------------------------------------------------------------ no-trade band, trim, top-up
check("band: 9.5% held vs 10% target -> no trade", plan(0.10, 9.5)[0] == "HOLD")
check("band: exactly 1 point off (9% vs 10%) -> no trade", plan(0.10, 9.0)[0] == "HOLD")
check("band: 8.5% vs 10% -> buy the difference (1.5 shares)", plan(0.10, 8.5) == ("BUY", 1.5), plan(0.10, 8.5))
check("trim overweight hold: AMD 10% -> 6% target sells 4 shares (~4% of the account)", plan(0.06, 10) == ("SELL", 4.0),
      plan(0.06, 10))
check("top up underweight hold: 3% -> 6% buys 3 shares", plan(0.06, 3) == ("BUY", 3.0), plan(0.06, 3))
check("add pick behaves the same (band applies)", plan(0.10, 9.5, status="add")[0] == "HOLD")
check("unknown status: never traded (fail closed)", plan(0.06, 10, status=None)[0] == "HOLD")
t = pd.DataFrame({"Symbol": ["AMD"], "Weight": [0.10], "Price": [100.0]})
o = pt.build_orders(t, 10000, positions={"AMD": 10, "ENPH": 0.3}, prices={"ENPH": 30.0}, fractional=True,
                    statuses={"AMD": "hold"}).set_index("Symbol")
check("non-target sold in full even when tiny (no band): ENPH 0.3", o.loc["ENPH", "Side"] == "SELL"
      and o.loc["ENPH", "Shares"] == 0.3)

# ------------------------------------------------------------------ earnings (live planner)
bl = {"AMD": "earnings Wed Sep 30, in 2 days"}
check("earnings: unowned pick not newly bought", plan(0.10, 0, blocked=bl)[0].startswith("SKIP (earnings Wed Sep 30"),
      plan(0.10, 0, blocked=bl))
check("earnings: owned underweight pick not topped up", plan(0.10, 3, blocked=bl)[0] == "HOLD")
check("earnings: owned overweight pick can still be trimmed", plan(0.05, 10, blocked=bl) == ("SELL", 5.0))
be_days = pt.earnings_blocked(["AAA", "BBB"], "2026-09-28",
                              earnings_csv=(lambda p: (pd.DataFrame({"Symbol": ["AAA", "BBB"], "Earnings Date":
                                            ["2026-10-02", "2026-10-09"]}).to_csv(p, index=False), p)[1])(
                                  os.path.join(tempfile.mkdtemp(), "e.csv")))
check("earnings_blocked: within 5 days blocked, 11 days out not", be_days == {"AAA": "earnings Fri Oct 02, in 4 days"},
      be_days)

# ------------------------------------------------------------------ cash: 1% cushion, part that fits
g = pd.DataFrame([{"Symbol": "A", "Side": "BUY", "Shares": 10.0, "Price": 100.0, "Est_Value": 1000.0,
                   "Current_Shares": 0, "Target_Shares": 10, "Target_Weight_%": 10, "Target_Value": 1000}])
out = pt.apply_buying_power_guard(g, 505.0, fractional=True)
check("guard: $505 free -> buys what fits after the 1% cushion (505/1.01/100 = 5.00)", out.iloc[0].Shares == 5.0,
      out.iloc[0].Shares)


class Order:
    def __init__(self, id, symbol, side, qty, status, filled_qty=0, client_order_id=""):
        self.id, self.symbol, self.side, self.qty = id, symbol, side, qty
        self.status, self.filled_qty, self.client_order_id = status, filled_qty, client_order_id


class Broker:
    """Fake Alpaca: market BUYs fill 1% ABOVE the planned price and cost real buying power; a buy that does not fit
    is rejected (as Alpaca would)."""
    def __init__(self, bp, positions=None, orders=(), slip=1.01):
        self.bp, self.slip, self.positions = bp, slip, dict(positions or {})
        self.orders = {o.id: o for o in orders}
        self.submitted, self.rejected, self.n = [], [], 0
        self.min_bp = bp

    def get_clock(self): return types.SimpleNamespace(is_open=True)
    def get_account(self): return types.SimpleNamespace(buying_power=str(self.bp), equity="10000", cash=str(self.bp))
    def get_all_positions(self): return [types.SimpleNamespace(symbol=s, qty=str(q)) for s, q in self.positions.items()]
    def get_order_by_id(self, oid): return self.orders[oid]
    def get_orders(self, req=None): return list(self.orders.values())
    def cancel_order_by_id(self, oid): self.orders[oid].status = "canceled"

    def submit_order(self, req):
        price = req.limit_price if hasattr(req, "limit_price") else PRICES[req.symbol] * self.slip
        buy = req.side == OrderSide.BUY
        if buy and req.qty * price > self.bp + 1e-9:
            self.rejected.append(req)
            raise RuntimeError("insufficient buying power")
        self.n += 1
        o = Order(f"m{self.n}", req.symbol, "BUY" if buy else "SELL", req.qty, "filled", req.qty, req.client_order_id)
        self.orders[o.id] = o
        self.submitted.append(req)
        self.bp += -req.qty * price if buy else req.qty * price
        self.min_bp = min(self.min_bp, self.bp)
        self.positions[req.symbol] = self.positions.get(req.symbol, 0) + (req.qty if buy else -req.qty)
        return o


PRICES = {"AAA": 100.0, "BBB": 50.0, "CCC": 20.0, "DDD": 10.0}
import fake_quotes  # noqa: E402
fake_quotes.install(pt, prices=PRICES)   # the ask at the planned price; limits fill at ask + 0.05%


def pending(rows, evening_date="2026-10-02"):
    p = os.path.join(tempfile.mkdtemp(), "live_pending_orders.json")
    json.dump({"evening_date": evening_date, "target_source": "auto", "as_of": evening_date, "orders": rows}, open(p, "w"))
    return p


def morning(b, p):
    pt.paper_trading_client = lambda: b
    pt.reconcile_positions = lambda *a, **k: (pd.DataFrame(), True)
    return pt.complete_unfilled_orders(pending_path=p, log_csv=None)


rows = [{"symbol": s, "side": "BUY", "qty": q, "limit_price": PRICES[s], "order_id": None}
        for s, q in (("AAA", 20), ("BBB", 30), ("CCC", 40), ("DDD", 100))]           # wants ~$5,300, only $4,000 free
b = Broker(4000.0)
res = morning(b, pending(rows))
check("sequence: no buy ever rejected, cash never below zero with fills 1% above plan",
      not b.rejected and b.min_bp >= 0, (len(b.rejected), round(b.min_bp, 2)))
check("sequence: ~99% of the cash invested (the part that fits is bought)", b.bp < 4000 * 0.02, round(b.bp, 2))
check("sequence: the buy that did not fully fit was cut, not skipped",
      any("fit in free cash" in s for s in res["Status"]), res["Status"].tolist())

# evening: only what fits in the cash free NOW is sent (whole shares); the pending entry keeps the full plan
rec = []
b = Broker(1000.0)
buys = pd.DataFrame([{"Symbol": "AAA", "Side": "BUY", "Shares": 15.0, "Price": 100.0, "Est_Value": 1500.0,
                      "Current_Shares": 0, "Target_Shares": 15, "Target_Weight_%": 15, "Target_Value": 1500}])
pt._wait_for_terminal_all = lambda *a, **k: {}
out = pt.submit_paper_extended_sequenced(buys, positions={}, record=rec.append, client=b, order_date="20261002")
check("evening partial buy: 9 whole shares sent ($1,000 / 1.01 = 9.90 fits), no reject",
      [(x.symbol, x.qty) for x in b.submitted] == [("AAA", 9)] and not b.rejected, [(x.symbol, x.qty) for x in b.submitted])
check("evening partial buy: pending keeps the full 15-share plan for 9 AM (order_qty 9)",
      rec and rec[0]["qty"] == 15.0 and rec[0]["order_qty"] == 9, rec)

# ------------------------------------------------------------------ fill-check gate: regular hours, until the next decision
CT = run_all.CT
p = pending([{"symbol": "AAA", "side": "BUY", "qty": 1, "limit_price": 100.0, "order_id": None}], "2026-10-02")
gate = lambda y, m, d, h=10, mi=0: run_all.fill_check_allowed(datetime(y, m, d, h, mi, tzinfo=CT), pending_path=p)
check("gate: Fri 10/2 evening orders -> Mon 10/5 8 AM not yet", not gate(2026, 10, 5, 8)[0])
check("gate: Mon 10/5 10 AM -> runs", gate(2026, 10, 5)[0])
check("gate: Mon 10/5 2:50 PM (15 min before the close) -> waits, no alert", not gate(2026, 10, 5, 14, 50)[0]
      and not gate(2026, 10, 5, 14, 50)[2])
check("gate: Sat 10/10 -> no run, no alert", not gate(2026, 10, 10)[0] and not gate(2026, 10, 10)[2])
exp = lambda when: [o["symbol"] for o in pt.superseded_orders(datetime.fromisoformat(when).replace(tzinfo=CT), p)]
check("superseded: Friday's leftovers can be completed until Mon 2:30 PM", exp("2026-10-05 14:29") == [])
check("superseded: from Mon 2:30 PM (the next decision) they are dropped, never sent", exp("2026-10-05 14:30") == ["AAA"])
p2 = pending([{"symbol": "AAA", "side": "BUY", "qty": 1, "limit_price": 100.0, "order_id": None,
               "recorded_at": "2026-10-05T10:02:00-05:00"}], "2026-10-05")
json.dump({**json.load(open(p2)), "send_now": True}, open(p2, "w"))
g = run_all.fill_check_allowed(datetime(2026, 10, 5, 10, 5, tzinfo=CT), pending_path=p2)
check("gate: a daytime catch-up's orders (send_now) go at once, not next morning", g[0], g)
check("superseded: a Monday-morning catch-up (Friday's decision) expires at Mon 2:30 PM",
      [o["symbol"] for o in pt.superseded_orders(datetime(2026, 10, 5, 14, 30, tzinfo=CT), p2)] == ["AAA"])
pt.log_event = lambda *a, **k: None
dropped = pt.drop_superseded_orders(datetime(2026, 10, 5, 15, 16, tzinfo=CT), p2)
check("drop_superseded_orders: removes the rows (file gone when empty)", len(dropped) == 1 and not os.path.exists(p2))

# no duplicates on retry: Monday's completion reached the broker but the Mac died before recording it
cid = pt._client_order_id("AAA", "BUY", 20, 100.0, "20261002", kind="fill")
prior = Order("mon-1", "AAA", "BUY", 20, "filled", 20, cid)
b = Broker(10000.0, orders=[prior])
res = morning(b, pending([{"symbol": "AAA", "side": "BUY", "qty": 20, "limit_price": 100.0, "order_id": None}]))
check("retry Tuesday: Monday's order found by its fixed id -> nothing sent again",
      not b.submitted and "ALREADY COMPLETED" in res["Status"].iloc[0], res["Status"].tolist())
open_o = Order("x-1", "AAA", "BUY", 5, "new", 0, "manual")
b = Broker(10000.0, orders=[open_o])
res = morning(b, pending([{"symbol": "AAA", "side": "BUY", "qty": 20, "limit_price": 100.0, "order_id": None}]))
check("retry: an open AAA order on Alpaca -> nothing sent", not b.submitted and "open AAA order" in res["Status"].iloc[0],
      res["Status"].tolist())

# leftovers kept across evenings: the Mac slept through Monday, Monday evening trades again
tmpd = tempfile.mkdtemp()
pp = os.path.join(tmpd, "live_pending_orders.json")
json.dump({"evening_date": "2026-10-02", "orders": [
    {"symbol": "AAA", "side": "BUY", "qty": 0.5, "limit_price": 100.0, "order_id": None},
    {"symbol": "BBB", "side": "BUY", "qty": 3, "limit_price": 50.0, "order_id": None}]}, open(pp, "w"))
pt._today_ct = lambda: datetime(2026, 10, 5, 15, 30, tzinfo=CT)
pt.record_pending_order({"symbol": "BBB", "side": "SELL", "qty": 3, "limit_price": 50.0, "order_id": "o1"}, {"source": "midweek"}, pp)
kept = json.load(open(pp))
syms = [(o["symbol"], o["side"], o.get("evening_date")) for o in kept["orders"]]
check("Monday evening: Friday's AAA leftover kept with its own date; BBB superseded by the new plan",
      kept["evening_date"] == "2026-10-05" and ("AAA", "BUY", "2026-10-02") in syms and ("BBB", "SELL", None) in syms
      and ("BBB", "BUY", "2026-10-02") not in syms, syms)
check("Monday evening retry: Friday leftover is not treated as already sent this evening",
      pt._todays_recorded_orders(pp) == {("BBB", "SELL")}, pt._todays_recorded_orders(pp))
b = Broker(10000.0, positions={"BBB": 3})
b.orders["o1"] = Order("o1", "BBB", "SELL", 3, "filled", 3, "x")
res = morning(b, pp)
cids = [x.client_order_id for x in b.submitted]
check("Tuesday: Friday's leftover completes with Friday's fixed id (live-fill-20261002-...)",
      any(c.startswith("live-fill-20261002-BUY-AAA") for c in cids), cids)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
