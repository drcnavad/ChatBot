"""Smart limit orders from the latest quote - fake broker + fake quotes only (no Alpaca account, no network).

The REAL alpaca-py request classes are used (they validate the order fields); the broker and the market-data client are
fakes. Covered: buy at the ask + 0.05% / sell at the bid - 0.05%; normal, wide, stale and missing quotes (asked again,
then skipped); a skipped order waits for the next business day's 9 AM CT check (one retry slot, never sent twice), and a
bad quote there goes to the normal retries; partial fills; cancel + replace once at a fresh quote; a replacement left
working; cancel not confirmed; buying-power cap; after-hours whole-share limits; Friday's old-format pending rows on
Monday 9 AM; the SIP / IEX feed check; the order-log price columns and the run-log cost row.

Run: cd <folder> && python3 tests/test_smart_orders.py"""
import json
import os
import sys
import tempfile
import types
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

os.environ.setdefault("STOCK_ANALYSIS_RUN_LOG", os.path.join(tempfile.mkdtemp(), "run_log.csv"))  # never the real run log
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
os.chdir(ROOT)
sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)   # .env is never read

import pandas as pd
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

import fake_quotes
import paper_trade as pt

CT = ZoneInfo("America/Chicago")
PASS, FAIL, ROWS = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""))


pt.quote_client = lambda: (_ for _ in ()).throw(AssertionError("real market-data client used in a test"))
pt.log_event = lambda run, status, moved, message, details="": ROWS.append((run, status, message))
pt.reconcile_positions = lambda *a, **k: (pd.DataFrame(), True)
NOW = [datetime(2026, 10, 2, 14, 45, tzinfo=CT)]          # the 2:30 PM run on Fri Oct 2 (sends at ~2:45)
pt._today_ct = lambda: NOW[0]
TERMINAL = pt.TERMINAL_STATUSES


class Order:
    def __init__(self, id, symbol, side, qty, status, filled_qty=0, client_order_id="", filled_avg_price=None):
        self.id, self.symbol, self.side, self.qty, self.status = id, symbol, side, qty, status
        self.filled_qty, self.client_order_id, self.filled_avg_price = filled_qty, client_order_id, filled_avg_price


class Broker:
    """Fake Alpaca. behave[symbol] = list of what each new order of that symbol does: "fill", "none" (stays open) or
    ("part", qty) (fills qty, stays open). The last item repeats. Fills are at the limit price."""
    def __init__(self, positions=None, bp=1_000_000.0, behave=None, orders=(), cancel_works=True):
        self.positions, self.bp, self.behave = dict(positions or {}), bp, {k: list(v) for k, v in (behave or {}).items()}
        self.orders = {o.id: o for o in orders}
        self.submitted, self.canceled, self.cancel_works = [], [], cancel_works

    def get_clock(self): return types.SimpleNamespace(is_open=True)
    def get_account(self): return types.SimpleNamespace(buying_power=str(self.bp), equity="100000", cash=str(self.bp))
    def get_all_positions(self): return [types.SimpleNamespace(symbol=s, qty=str(q)) for s, q in self.positions.items() if q]
    def get_order_by_id(self, oid): return self.orders[oid]
    def get_orders(self, req=None): return list(self.orders.values())

    def cancel_order_by_id(self, oid):
        self.canceled.append(oid)
        if self.cancel_works and self.orders[oid].status not in TERMINAL:
            self.orders[oid].status = "canceled"

    def submit_order(self, req):
        assert isinstance(req, LimitOrderRequest), f"not a limit order: {type(req).__name__}"
        if req.client_order_id in {o.client_order_id for o in self.orders.values()}:
            raise RuntimeError("client_order_id must be unique")       # what Alpaca does
        self.submitted.append(req)
        b = self.behave.get(req.symbol, ["fill"])
        kind = b.pop(0) if len(b) > 1 else b[0]
        filled = req.qty if kind == "fill" else (kind[1] if isinstance(kind, tuple) else 0)
        status = "filled" if kind == "fill" else ("partially_filled" if filled else "new")
        side = "BUY" if req.side == OrderSide.BUY else "SELL"
        o = Order(f"o{len(self.submitted)}", req.symbol, side, req.qty, status, filled, req.client_order_id,
                  req.limit_price if filled else None)
        self.orders[o.id] = o
        self.positions[req.symbol] = self.positions.get(req.symbol, 0) + (filled if side == "BUY" else -filled)
        self.bp -= req.qty * req.limit_price if side == "BUY" else 0
        return o


def pending(rows, evening_date="2026-10-02", send_now=True, recorded_at="2026-10-02T14:44:00-05:00"):
    p = os.path.join(tempfile.mkdtemp(), "live_pending_orders.json")
    rows = [{"limit_price": 100.0, "order_id": None, "exit": False, "recorded_at": recorded_at, **r} for r in rows]
    json.dump({"evening_date": evening_date, "target_source": "provisional", "as_of": evening_date, "orders": rows,
               **({"send_now": True} if send_now else {})}, open(p, "w"))
    return p


def run(b, p, log=None):
    pt.paper_trading_client = lambda: b
    return pt.complete_unfilled_orders(pending_path=p, log_csv=log)


def status_of(res, sym):
    return " | ".join(res.loc[res["Symbol"] == sym, "Status"].astype(str))


def left(p):
    return json.load(open(p))["orders"] if os.path.exists(p) else []


GOOD = (99.90, 100.00)            # spread 0.10%: buy limit 100.00 + 0.05% = 100.05, sell 99.90 - 0.05% = 99.85
WIDE = (99.00, 100.00)            # spread 1.0% > 0.5%
STALE = (99.90, 100.00, 120)      # 2 minutes old
REAL_LATEST = pt._latest_quote             # the real one, used with a fake data client in the feed test
Q = fake_quotes.install(pt, default=GOOD)

# ------------------------------------------------------------------ prices from one quote
check("buy limit = ask + 0.05%, rounded UP to the cent", pt.smart_quote("AAA", "BUY")["limit"] == 100.05)
check("sell limit = bid - 0.05%, rounded DOWN to the cent", pt.smart_quote("AAA", "SELL")["limit"] == 99.85)
Q.quotes["PENNY"] = (0.5120, 0.5124)
check("under $1: 4 decimals (0.5124 + 0.05% -> 0.5127)", pt.smart_quote("PENNY", "BUY")["limit"] == 0.5127,
      pt.smart_quote("PENNY", "BUY")["limit"])
for name, quote, word in (("wide spread", WIDE, "spread too wide"), ("stale quote", STALE, "stale"),
                          ("missing bid", (0, 100.0), "missing"), ("unreadable", ConnectionError("down"), "no quote")):
    Q.quotes["BAD"], Q.calls[:] = [quote], []
    q = pt.smart_quote("BAD", "BUY")
    check(f"{name}: skipped after {pt.QUOTE_TRIES} tries ({word})",
          q.get("limit") is None and word in q["skip"] and Q.calls == ["BAD"] * pt.QUOTE_TRIES, (q, Q.calls))
Q.quotes["BAD"] = [WIDE, GOOD]
q = pt.smart_quote("BAD", "BUY")
check("wide, then normal on the retry: priced from the good quote", q["skip"] is None and q["limit"] == 100.05, q)

# ------------------------------------------------------------------ the 2:30 PM run: sells first, DAY limits, fractional
log = os.path.join(tempfile.mkdtemp(), "live_orders_log.csv")
pd.DataFrame([{"Symbol": "OLD", "Side": "BUY", "Shares": 1, "Status": "submitted", "Submitted_At_CT": "x",
               "Target_Source": "provisional", "As_Of": "2026-10-01", "Equity": 1.0}]).to_csv(log, index=False)
b = Broker(positions={"SSS": 4.0})
p = pending([{"symbol": "BBB", "side": "BUY", "qty": 3.58}, {"symbol": "SSS", "side": "SELL", "qty": 4.0, "exit": True}])
ROWS.clear()
res = run(b, p, log)
s = b.submitted
check("2:30 run: SELL sent before the BUY", [(x.symbol, x.side) for x in s] == [("SSS", OrderSide.SELL), ("BBB", OrderSide.BUY)],
      [(x.symbol, x.side) for x in s])
check("2:30 run: DAY limit orders, no extended hours, 2-decimal shares",
      all(x.time_in_force == TimeInForce.DAY and not x.extended_hours for x in s) and s[1].qty == 3.58, [x.qty for x in s])
check("2:30 run: sell at 99.85, buy at 100.05", (s[0].limit_price, s[1].limit_price) == (99.85, 100.05))
check("2:30 run: done, nothing kept", not os.path.exists(p) and "COMPLETED via limit" in status_of(res, "BBB"))
lg = pd.read_csv(log)
need = ["Bid", "Ask", "Spread_%", "Limit", "Fill_Price", "Slippage_%"]
check("order log: price columns added, the old row kept", all(c in lg.columns for c in need) and lg.Symbol.iloc[0] == "OLD",
      list(lg.columns))
bb = lg[lg.Symbol == "BBB"].iloc[0]
check("order log: BUY bid 99.90 / ask 100.00 / spread 0.1% / limit 100.05 / fill 100.05 / slippage +0.1%",
      (bb.Bid, bb.Ask, bb["Spread_%"], bb.Limit, bb.Fill_Price, round(bb["Slippage_%"], 2)) == (99.9, 100.0, 0.1, 100.05, 100.05, 0.1),
      bb.to_dict())
cost = [r for r in ROWS if r[2].startswith("Order prices")]
check("run log: one plain cost row (SIP quotes, $ cost vs the mid price)",
      len(cost) == 1 and "SIP" in cost[0][2] and "spread cost about $" in cost[0][2], cost)

# ------------------------------------------------------------------ wide spread at 2:30 -> one retry slot at Mon 9 AM
Q.quotes["WID"] = [WIDE]
b = Broker()
p = pending([{"symbol": "WID", "side": "BUY", "qty": 5}])
res = run(b, p)
row = left(p)
check("wide spread at 2:30: nothing sent", not b.submitted)
check("wide spread at 2:30: kept for the Mon Oct 5 9 AM CT check (retry_on 2026-10-05)",
      len(row) == 1 and row[0].get("retry_on") == "2026-10-05" and "bad quote" in status_of(res, "WID")
      and "Mon Oct 5" in status_of(res, "WID"), (row, status_of(res, "WID")))
NOW[0] = datetime(2026, 10, 2, 14, 50, tzinfo=CT)
Q.quotes["WID"] = [GOOD]
res = run(b, p)
check("a second check the same afternoon: still nothing sent (waits for 9 AM)", not b.submitted and "WAITING" in status_of(res, "WID"))
NOW[0] = datetime(2026, 10, 5, 9, 0, tzinfo=CT)
res = run(b, p)
check("Mon 9 AM: sent once with a fresh quote", [(x.symbol, x.qty, x.limit_price) for x in b.submitted] == [("WID", 5, 100.05)]
      and not os.path.exists(p), [(x.symbol, x.qty) for x in b.submitted])
NOW[0] = datetime(2026, 10, 5, 9, 30, tzinfo=CT)
check("Mon 9:30: nothing left, never sent twice", run(b, p).empty and len(b.submitted) == 1)

# ------------------------------------------------------------------ still bad at 9 AM -> normal retries, then dropped
NOW[0] = datetime(2026, 10, 2, 14, 45, tzinfo=CT)
Q.quotes["STL"] = [STALE]
b = Broker()
p = pending([{"symbol": "STL", "side": "BUY", "qty": 2}])
run(b, p)
ROWS.clear()
NOW[0] = datetime(2026, 10, 5, 9, 0, tzinfo=CT)
res = run(b, p)
check("stale at 2:30 and again at Mon 9 AM: not sent, try 1 of 3 (retried in 30 min)",
      not b.submitted and "try 1 of 3" in status_of(res, "STL") and "stale" in status_of(res, "STL"), status_of(res, "STL"))
check("... and a plain run-log row says it tries again (don't place by hand)",
      any("STL" in m and "tries again" in m and "don't place" in m for _, _, m in ROWS), ROWS)
for t in (9 * 60 + 30, 10 * 60):
    NOW[0] = datetime(2026, 10, 5, t // 60, t % 60, tzinfo=CT)
    res = run(b, p)
check("still stale at the 3rd check: dropped (gave up), never sent", not b.submitted and not os.path.exists(p)
      and "gave up" in status_of(res, "STL"), status_of(res, "STL"))

# a Friday-evening row that meets a bad quote at Mon 9 AM: Tuesday 9 AM is after the Mon 2:30 decision -> normal retries
Q.quotes["MON"] = [WIDE]
b = Broker()
p = pending([{"symbol": "MON", "side": "BUY", "qty": 2}], send_now=False, recorded_at="2026-10-02T15:31:00-05:00")
NOW[0] = datetime(2026, 10, 5, 9, 0, tzinfo=CT)
res = run(b, p)
check("Friday's leftover, bad quote Mon 9 AM: not carried to Tuesday (Mon 2:30 decision first) - retried in 30 min",
      not b.submitted and "try 1 of 3" in status_of(res, "MON") and not left(p)[0].get("retry_on"), status_of(res, "MON"))

# ------------------------------------------------------------------ partial fill -> cancel + replace once at a fresh quote
NOW[0] = datetime(2026, 10, 2, 14, 45, tzinfo=CT)
Q.quotes["PRT"] = [GOOD, (99.95, 100.05)]
b = Broker(behave={"PRT": [("part", 2), "fill"]})
p = pending([{"symbol": "PRT", "side": "BUY", "qty": 5}])
res = run(b, p)
s = b.submitted
check("partial: first limit 5 @ 100.05 filled 2, canceled, the rest 3 replaced @ 100.11 (fresh ask 100.05 + 0.05%)",
      [(x.qty, x.limit_price) for x in s] == [(5, 100.05), (3, 100.11)] and b.canceled == ["o1"],
      [(x.qty, x.limit_price) for x in s])
check("partial: replacement id = first id + '-c' (deterministic, never a duplicate id)",
      s[1].client_order_id == s[0].client_order_id[:46] + "-c", (s[0].client_order_id, s[1].client_order_id))
check("partial: done after the replacement filled (nothing kept)", not os.path.exists(p) and "REPLACED" in status_of(res, "PRT"))
check("partial: shares bought = exactly 5", b.positions["PRT"] == 5, b.positions)

# replacement not filled either -> left working, looked at again Mon 9 AM; nothing sent before then
Q.quotes["WRK"] = [GOOD]
b = Broker(behave={"WRK": ["none"]})
p = pending([{"symbol": "WRK", "side": "BUY", "qty": 4}])
res = run(b, p)
row = left(p)
check("unfilled twice: one replacement only, left working (not canceled again)",
      len(b.submitted) == 2 and b.canceled == ["o1"] and b.orders["o2"].status == "new", (len(b.submitted), b.canceled))
check("unfilled twice: pending row = the working replacement, for Mon 9 AM",
      len(row) == 1 and row[0]["order_id"] == "o2" and row[0]["qty"] == 4 and row[0]["retry_on"] == "2026-10-05", row)
NOW[0] = datetime(2026, 10, 2, 14, 55, tzinfo=CT)
run(b, p)
check("unfilled twice: a later check the same day sends nothing", len(b.submitted) == 2)
b.orders["o2"].status, b.orders["o2"].filled_qty = "expired", 1          # the DAY order expired at the close, 1 filled
b.behave["WRK"] = ["fill"]
NOW[0] = datetime(2026, 10, 5, 9, 0, tzinfo=CT)
res = run(b, p)
check("Mon 9 AM: only the unfilled 3 sent once", [x.qty for x in b.submitted[2:]] == [3] and not os.path.exists(p),
      [x.qty for x in b.submitted])

# cancel not confirmed -> no replacement (never two working orders)
NOW[0] = datetime(2026, 10, 2, 14, 45, tzinfo=CT)
b = Broker(behave={"CNX": ["none"]}, cancel_works=False)
p = pending([{"symbol": "CNX", "side": "SELL", "qty": 3, "exit": True}])
b.positions["CNX"] = 3
res = run(b, p)
row = left(p)
check("cancel not confirmed: no replacement sent, the open order is carried to Mon 9 AM",
      len(b.submitted) == 1 and len(row) == 1 and row[0]["order_id"] == "o1", (len(b.submitted), row))

# fresh ask ran up more than the 1% cushion -> not replaced, the rest goes to Mon 9 AM
Q.quotes["RUN"] = [GOOD, (101.50, 101.60)]
b = Broker(behave={"RUN": [("part", 1), "fill"]})
p = pending([{"symbol": "RUN", "side": "BUY", "qty": 4}])
res = run(b, p)
row = left(p)
check("price ran up >1%: no replacement, the rest 3 kept for Mon 9 AM (no order id)",
      len(b.submitted) == 1 and len(row) == 1 and row[0]["qty"] == 3 and row[0]["order_id"] is None
      and row[0]["retry_on"] == "2026-10-05", (len(b.submitted), row))

# ------------------------------------------------------------------ buying-power cap at the limit price + 1% cushion
Q.quotes.clear()
b = Broker(bp=1000.0)
p = pending([{"symbol": "AAA", "side": "BUY", "qty": 6}, {"symbol": "BBB", "side": "BUY", "qty": 6}])
res = run(b, p)
got = [(x.symbol, x.qty) for x in b.submitted]
spent = sum(x.qty * x.limit_price for x in b.submitted)
check("buying power $1,000: AAA 6 fits (6 x 100.05 x 1.01 = $606.30), BBB cut to 3.89 (= $393.70 / 101.05)",
      got == [("AAA", 6), ("BBB", 3.89)], got)
check("buying power: never more than the cash at the limit price", spent <= 1000.0, spent)
check("buying power: BBB's rest 2.11 waits for the next fill check", [(r["symbol"], r["qty"]) for r in left(p)] == [("BBB", 2.11)],
      left(p))

# ------------------------------------------------------------------ after hours: whole shares, extended hours, quote-priced
Q.quotes.update({"EXT": [WIDE]})
b = Broker(positions={"SSS": 3.5})
rec = []
orders = pd.DataFrame([{"Symbol": "SSS", "Side": "SELL", "Shares": 3.5, "Price": 100.0},
                       {"Symbol": "AAA", "Side": "BUY", "Shares": 2.6, "Price": 100.0},
                       {"Symbol": "EXT", "Side": "BUY", "Shares": 4.0, "Price": 100.0}])
NOW[0] = datetime(2026, 10, 2, 15, 10, tzinfo=CT)
out = pt.submit_paper_extended(orders, positions={"SSS": 3.5}, record=rec.append, client=b, order_date="20261002")
lim = {x.symbol: x for x in b.submitted}
check("after hours: whole-share limits with extended hours (SSS 3 @ 99.85, AAA 2 @ 100.05)",
      {k: (v.qty, v.limit_price, v.extended_hours) for k, v in lim.items()} == {"SSS": (3, 99.85, True), "AAA": (2, 100.05, True)},
      {k: (v.qty, v.limit_price) for k, v in lim.items()})
ext = [r for r in rec if r["symbol"] == "EXT"]
check("after hours: wide spread -> not sent, STAGED for Mon Oct 5 9 AM CT (no order id)",
      "EXT" not in lim and ext and ext[0]["order_id"] is None and ext[0]["retry_on"] == "2026-10-05"
      and "bad quote" in status_of(out, "EXT"), (ext, status_of(out, "EXT")))
check("after hours: the client order id still uses the planned price (stable across retries)",
      lim["AAA"].client_order_id == pt._client_order_id("AAA", "BUY", 2, 100.0, "20261002"), lim["AAA"].client_order_id)
check("after hours: the quote is saved with the pending row (fill cost shows at 9 AM)",
      [r for r in rec if r["symbol"] == "AAA"][0]["quote"]["limit"] == 100.05)
check("after hours: price columns in the results", out.loc[out.Symbol == "AAA", "Limit"].iloc[0] == 100.05)
Q.quotes["CUT"] = [(102.0, 102.1)]
out = pt.submit_paper_extended(pd.DataFrame([{"Symbol": "CUT", "Side": "BUY", "Shares": 2.0, "Price": 100.0}]),
                               positions={}, record=rec.append, client=b, order_date="20261002")
check("after hours: a buy whose ask is >1% above the plan is staged for 9 AM (it might not fit the cash)",
      "CUT" not in {x.symbol for x in b.submitted} and "STAGED" in status_of(out, "CUT"), status_of(out, "CUT"))

# the staged after-hours row is placed at its 9 AM retry slot, once
p = os.path.join(tempfile.mkdtemp(), "live_pending_orders.json")
json.dump({"evening_date": "2026-10-02", "target_source": "provisional", "as_of": "2026-10-02",
           "orders": [{**ext[0], "recorded_at": "2026-10-02T15:10:00-05:00"}]}, open(p, "w"))
Q.quotes["EXT"] = [GOOD]
b = Broker()
NOW[0] = datetime(2026, 10, 5, 9, 0, tzinfo=CT)
run(b, p)
run(b, p)
check("after hours -> Mon 9 AM: the skipped EXT buy is placed once (4 @ 100.05, fractional-capable DAY limit)",
      [(x.symbol, x.qty, x.limit_price, bool(x.extended_hours)) for x in b.submitted] == [("EXT", 4.0, 100.05, False)],
      [(x.symbol, x.qty) for x in b.submitted])

# ------------------------------------------------------------------ Friday's old-format rows (before this change) on Mon 9 AM
FRI = [  # the shape of Reports/live_pending_orders.json written Fri Oct 2 3:31 PM (no quote / retry_on fields)
    {"symbol": "ENPH", "side": "SELL", "qty": 4.0, "limit_price": 33.47, "order_id": "e-ENPH", "exit": True, "order_qty": 4,
     "recorded_at": "2026-10-02T15:31:09-05:00", "decision": "2026-10-02"},
    {"symbol": "ORCL", "side": "SELL", "qty": 1.054222297, "limit_price": 142.3, "order_id": "e-ORCL", "exit": True,
     "order_qty": 1, "recorded_at": "2026-10-02T15:31:10-05:00", "decision": "2026-10-02"},
    {"symbol": "AMD", "side": "BUY", "qty": 9.59, "limit_price": 633.91, "order_id": "e-AMD", "exit": False, "order_qty": 9,
     "recorded_at": "2026-10-02T15:31:10-05:00", "decision": "2026-10-02"},
    {"symbol": "RBRK", "side": "BUY", "qty": 46.45, "limit_price": 118.58, "order_id": "e-RBRK", "exit": False, "order_qty": 45,
     "recorded_at": "2026-10-02T15:31:10-05:00", "decision": "2026-10-02"}]
eve = [Order("e-ENPH", "ENPH", "SELL", 4, "filled", 4, "live-20261002-SELL-ENPH-4-3347", 33.40),
       Order("e-ORCL", "ORCL", "SELL", 1, "filled", 1, "live-20261002-SELL-ORCL-1-14230", 142.10),
       Order("e-AMD", "AMD", "BUY", 9, "filled", 9, "live-20261002-BUY-AMD-9-63391", 634.00),
       Order("e-RBRK", "RBRK", "BUY", 45, "expired", 40, "live-20261002-BUY-RBRK-45-11858", 118.60)]
p = os.path.join(tempfile.mkdtemp(), "live_pending_orders.json")
json.dump({"evening_date": "2026-10-02", "submitted_at_ct": "2026-10-02 15:31:10", "target_source": "provisional",
           "as_of": "2026-10-02", "orders": FRI}, open(p, "w"))
Q.quotes.update({"ORCL": [(141.0, 141.1)], "AMD": [(630.0, 630.3)], "RBRK": [(118.0, 118.05)]})
b = Broker(positions={"ORCL": 0.054222297, "AMD": 9, "RBRK": 40}, orders=eve)
NOW[0] = datetime(2026, 10, 5, 9, 0, tzinfo=CT)
log = os.path.join(tempfile.mkdtemp(), "live_orders_log.csv")
res = run(b, p, log)
got = {x.symbol: (x.qty, x.limit_price, x.client_order_id) for x in b.submitted}
check("Friday rows, Mon 9 AM: ORCL exact 0.054222297 rest sold @ 141.0 - 0.05% = 140.92",
      got.get("ORCL", (0,))[:2] == (0.054222297, 140.92), got.get("ORCL"))
check("Friday rows, Mon 9 AM: AMD 0.59 rest @ 630.3 + 0.05% = 630.62; RBRK 6.45 (46.45 - 40 filled) @ 118.11",
      got.get("AMD", (0,))[:2] == (0.59, 630.62) and got.get("RBRK", (0,))[:2] == (6.45, 118.11), got)
check("Friday rows: completion ids keep Friday's date + planned price (old scheme, crash-safe)",
      got["AMD"][2] == "live-fill-20261002-BUY-AMD-0-63391" and got["RBRK"][2] == "live-fill-20261002-BUY-RBRK-6-11858",
      (got["AMD"][2], got["RBRK"][2]))
check("Friday rows: ENPH (fully filled Friday) not touched; file cleared", "ENPH" not in got and not os.path.exists(p))
lg = pd.read_csv(log)
check("Friday rows: the evening fill price is logged (ENPH 33.40)", lg.loc[lg.Symbol == "ENPH", "Fill_Price"].iloc[0] == 33.4)

# ------------------------------------------------------------------ SIP when it works, else IEX; neither -> skip (no keys read)
class DataClient:
    def __init__(self, sip_ok=True, iex_ok=True):
        self.ok, self.feeds = {"sip": sip_ok, "iex": iex_ok}, []

    def get_stock_latest_quote(self, req):
        self.feeds.append(req.feed.value)
        if not self.ok[req.feed.value]:
            raise RuntimeError(f"{req.feed.value}: subscription does not permit querying recent data")
        sym = req.symbol_or_symbols if isinstance(req.symbol_or_symbols, str) else req.symbol_or_symbols[0]
        return {sym: types.SimpleNamespace(bid_price=99.9, ask_price=100.0, timestamp=datetime.now(timezone.utc))}


pt._latest_quote = REAL_LATEST                # the real feed logic below, with fake data clients
for sip_ok, want in ((True, "sip"), (False, "iex")):
    dc = DataClient(sip_ok)
    pt.quote_client = lambda dc=dc: dc
    pt._QUOTES.clear()
    q = pt.smart_quote("AAA", "BUY")
    check(f"feed: {'SIP works -> SIP' if sip_ok else 'SIP raises -> IEX'}, same limit price (ask 100.00 + 0.05% = 100.05)",
          q["feed"] == want and q["limit"] == 100.05 and q["skip"] is None
          and dc.feeds == (["sip"] if sip_ok else ["sip", "iex"]), (q, dc.feeds))

dc = DataClient(sip_ok=False, iex_ok=False)
pt.quote_client = lambda: dc
pt._QUOTES.clear()
q = pt.smart_quote("AAA", "BUY")
check("feed: SIP and IEX both raise -> no crash, re-quoted 3 times, then skipped as a bad quote",
      q["skip"] and q["skip"].startswith("no quote") and dc.feeds == ["sip", "iex"] * 3, (q, dc.feeds))

pt._QUOTES.clear()
pt.quote_client = lambda: (_ for _ in ()).throw(OSError("no market-data client"))
q = pt.smart_quote("AAA", "SELL")
check("feed: the data client can't even be made -> no crash, skipped as a bad quote", q["skip"].startswith("no quote"), q)

# both feeds down in a real 2:30 fill check: nothing sent, no crash, carried to the next 9 AM CT check
dc = DataClient(sip_ok=False, iex_ok=False)
pt.quote_client = lambda: dc
pt._QUOTES.clear()
NOW[0] = datetime(2026, 10, 2, 14, 45, tzinfo=CT)
b, p = Broker(), pending([{"symbol": "DWN", "side": "BUY", "qty": 5}])
res = run(b, p, os.path.join(tempfile.mkdtemp(), "live_orders_log.csv"))
check("feed: both down at 2:30 -> nothing sent, run finishes, DWN kept for the Mon Oct 5 9 AM CT check",
      not b.submitted and "Mon Oct 5" in status_of(res, "DWN") and left(p)[0].get("retry_on") == "2026-10-05",
      status_of(res, "DWN"))

for r in [r for r in ROWS if r[2].startswith("Order prices")][-3:]:
    print("   run-log cost row:", r[1], "|", r[2])
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
