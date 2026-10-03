"""Mocked tests (zero broker calls, zero network) for the live pre-earnings 3.5 x ATR stop (earnings_stop.py).

Window: 7 calendar days before the earnings date through the reaction day (AM: that day; PM/unknown: next session).
Sessions: pre-market 4:00, regular 9:30, after hours until 8 PM ET (5 PM on early-close days); weekends/holidays idle.
Stop = highest close since entry - 3.5 x ATR(14) (the engine's ATR). At or below: one whole-share limit SELL at the
bid - 0.05%, extended_hours outside regular hours, the fraction to the 9 AM fill check; never twice (state + broker
client id); fails closed on unreadable reads, open orders and running jobs. Dry run: no trading client, no files.

Run: cd <folder> && python3 tests/test_earnings_stop.py
"""
import csv
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from types import SimpleNamespace as NS

TMP = tempfile.mkdtemp()
os.environ["STOCK_ANALYSIS_RUN_LOG"] = os.path.join(TMP, "run_log.csv")      # never the real run log
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.modules["dotenv"] = NS(load_dotenv=lambda *a, **k: None)                  # never reads .env

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import backtest_engine as be  # noqa: E402
import earnings_stop as es  # noqa: E402
import paper_trade as pt  # noqa: E402
import run_all  # noqa: E402

FAIL = []


def check(what, ok, got=""):
    print(("PASS " if ok else "FAIL ") + what + ("" if ok else f"  -> {got}"))
    if not ok:
        FAIL.append(what)


ET = be.EASTERN
at = lambda s: pd.Timestamp(s).tz_localize(ET).to_pydatetime()                # an ET wall time
D = lambda s: pd.Timestamp(s)

# ----------------------------------------------------------------------------- window and sessions
E = pd.DataFrame({"Symbol": ["XYZ", "XYZ", "AMX", "FRI", "NOT"],
                  "Earnings Date": [D("2026-10-29"), D("2026-07-30"), D("2026-10-29"), D("2026-10-30"), D("2026-10-29")],
                  "Time": ["PM", "PM", "AM", "PM", np.nan]})
ev = lambda s, d: es.active_event(s, E, d)
check("window opens exactly 7 calendar days before (Thu Oct 22 for Thu Oct 29)", ev("XYZ", "2026-10-21") is None and ev("XYZ", "2026-10-22") is not None)
check("PM report: active through the next session (Fri Oct 30), off after", ev("XYZ", "2026-10-30")[2] == D("2026-10-30") and ev("XYZ", "2026-10-31") is None)
check("AM report: the report day is the reaction day", ev("AMX", "2026-10-29")[2] == D("2026-10-29") and ev("AMX", "2026-10-30") is None)
check("Friday PM report reacts on Monday", ev("FRI", "2026-11-02")[2] == D("2026-11-02") and ev("FRI", "2026-11-03") is None)
check("the following Friday counts (Fri Oct 23 -> Fri Oct 30)", ev("FRI", "2026-10-23") is not None)
check("unknown time reacts the next session", ev("NOT", "2026-10-30") is not None and ev("NOT", "2026-10-30")[1] == "")
check("a past report does not open a window", ev("XYZ", "2026-07-25") is not None and ev("XYZ", "2026-08-05") is None)

sess = es.session_now
check("sessions on a weekday: 3:59 none, 4:00 pre, 9:30 regular, 16:00 after, 19:59 after, 20:00 none",
      [sess(at(f"2026-10-26 {t}")) for t in ("03:59", "04:00", "09:29", "09:30", "15:59", "16:00", "19:59", "20:00")]
      == [None, "pre", "pre", "regular", "regular", "after", "after", None])
check("weekend and holiday are idle", sess(at("2026-10-24 10:00")) is None and sess(at("2026-11-26 10:00")) is None)
check("early close (Nov 27): regular to 13:00, after hours to 17:00 ET",
      [sess(at(f"2026-11-27 {t}")) for t in ("12:59", "13:00", "16:59", "17:00")] == ["regular", "after", "after", None])

# ----------------------------------------------------------------------------- stop math
rng = np.random.default_rng(7)
days = pd.date_range("2026-08-03", "2026-10-21", freq=be.NYSE_SESSION)
close = 100 + np.cumsum(rng.normal(0, 1.5, len(days)))
close[10] = 200                                                                # a spike BEFORE the entry date
bars = pd.DataFrame({"Symbol": "XYZ", "Date": days, "Open": close, "High": close + rng.uniform(0.5, 2, len(days)),
                     "Low": close - rng.uniform(0.5, 2, len(days)), "Close": close, "Volume": 1e6})
entry = days[20].date()
peak, atr, stop = es.stop_level(bars, entry)
ref = be.calculate_technical_indicators(bars.copy())["atr"].iloc[-1]
check("ATR(14) equals the engine's", abs(atr - ref) < 1e-9, (atr, ref))
check("peak = highest close since entry (an earlier spike is ignored)", peak == close[20:].max() and peak < 200)
check("stop = peak - 3.5 x ATR", abs(stop - (peak - 3.5 * atr)) < 1e-9)
check("too few bars -> no stop", es.stop_level(bars.head(14), days[0].date()) is None)

# ----------------------------------------------------------------------------- the live check, fully mocked


class Account:
    def __init__(self, held):
        self.held, self.calls = held, 0

    def position_dicts(self):
        self.calls += 1
        return [{"symbol": s, "qty": str(q)} for s, q in self.held.items()]

    def fills(self):
        return [{"symbol": s, "side": "buy", "qty": str(q), "transaction_time": f"{entry}T15:00:00Z"} for s, q in self.held.items()]


class Client:
    def __init__(self, held, open_orders=(), prior=(), fail=False):
        self.held, self.open, self.prior, self.fail, self.sent = dict(held), list(open_orders), list(prior), fail, []

    def get_all_positions(self):
        return [NS(symbol=s, qty=str(q)) for s, q in self.held.items()]

    def get_orders(self, req):
        if self.fail:
            raise RuntimeError("broker down")
        if getattr(req.status, "value", req.status) == "open":
            return [o for o in self.open if o.symbol in (req.symbols or [])]
        return self.prior + [NS(id=f"oid-{i}", client_order_id=r.client_order_id, status=NS(value="new"), filled_qty="0", symbol=r.symbol)
                             for i, r in enumerate(self.sent)]

    def submit_order(self, req):
        self.sent.append(req)
        return NS(id=f"oid-{len(self.sent) - 1}", status=NS(value="new"), filled_qty="0")


def setup(held=None, client=None, quote=None, age=0, earnings=E):
    d = tempfile.mkdtemp(dir=TMP)
    es.STATE_JSON, es.STATUS_CSV = os.path.join(d, "state.json"), os.path.join(d, "stops.csv")
    run_all.TRADE_LOCK, run_all.FILL_LOCK = os.path.join(d, ".trade.lock"), os.path.join(d, ".fill.lock")
    pt.PENDING_ORDERS_JSON = os.path.join(d, "pending.json")
    acct = Account(held if held is not None else {"XYZ": 10.4, "QQQ": 3})
    es.account = lambda: acct
    es.be.load_earnings = lambda path=None: earnings
    es.daily_bars = lambda syms, now: bars[bars["Symbol"].isin(syms)]
    made = []
    es.trading_client = lambda: made.append(1) or client
    q = quote or (stop + 5, stop + 5.04)
    es.latest_quote = lambda s: (q[0], q[1], CLOCK[0] - timedelta(seconds=age), "iex")   # age vs the simulated time
    return d, acct, made


def runlog():
    p = os.environ["STOCK_ANALYSIS_RUN_LOG"]
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


REG = at("2026-10-22 11:00")
CLOCK = [REG]
_run_check = es.run_check


def run_check(now, **kw):
    CLOCK[0] = now
    return _run_check(now=now, **kw)


es_run = run_check

# above the stop: no order, armed once, status written
c = Client({"XYZ": 10.4})
d, acct, made = setup(client=c)
n0 = len(runlog())
es_run(REG)
es_run(REG + timedelta(minutes=10))
st = pd.read_csv(es.STATUS_CSV)
check("above the stop: no order and no trading client", c.sent == [] and made == [])
check("status file lists the held stock in the window (QQQ has no report)", list(st.Symbol) == ["XYZ"] and st.Status[0] == "watching", st.to_dict("records"))
check("status stop = the computed stop", abs(st.Stop[0] - round(stop, 2)) < 1e-9)
check("'armed' run-log row written once", sum("armed for XYZ" in r["message"] for r in runlog()[n0:]) == 1, runlog()[n0:])

# at the stop in regular hours: one whole-share sale
bid = stop - 1
c = Client({"XYZ": 10.4})
d, acct, made = setup(client=c, quote=(bid, bid + 0.04))
n0 = len(runlog())
es_run(REG)
r = c.sent[0] if c.sent else None
check("at the stop: one SELL of the 10 whole shares", len(c.sent) == 1 and r.qty == 10 and r.side.value == "sell", c.sent)
check("limit = bid - 0.05% on the tick, DAY, regular hours (no extended_hours)",
      r and r.limit_price == pt._tick(bid * (1 - pt.LIMIT_OFFSET), False) and r.time_in_force.value == "day" and not r.extended_hours)
check("fixed client order id live-stop-20261029-XYZ", r and r.client_order_id == "live-stop-20261029-XYZ")
pend = json.load(open(pt.PENDING_ORDERS_JSON))
row = pend["orders"][0]
check("pending row for the 9 AM check: exact qty 10.4, order_qty 10, exit, order id",
      len(pend["orders"]) == 1 and row["qty"] == 10.4 and row["order_qty"] == 10 and row["exit"] is True and row["order_id"] == "oid-0"
      and row["side"] == "SELL" and pend["evening_date"] == "2026-10-22", pend)
log = [x for x in runlog()[n0:] if "Pre-earnings stop: XYZ" in x["message"]]
check("one run-log row for the sale (money moved unknown until filled)", len(log) == 1 and log[0]["run"] == "Earnings stop" and log[0]["money_moved"] == "unknown", log)
check("the fill-check lock is released", not os.path.exists(run_all.FILL_LOCK))
es_run(REG + timedelta(minutes=10))
check("never twice: the next check sends nothing", len(c.sent) == 1)
os.remove(es.STATE_JSON)
es_run(REG + timedelta(minutes=20))
check("never twice even without the state file (the client id is on the broker)", len(c.sent) == 1)

# pre-market and after hours: extended_hours
for t, name in (("2026-10-23 07:00", "pre-market"), ("2026-10-23 17:30", "after hours")):
    c = Client({"XYZ": 10.4})
    setup(client=c, quote=(bid, bid + 0.04), age=300)
    es_run(at(t))
    check(f"{name}: extended_hours limit DAY order (a 5-minute-old quote is fine)", len(c.sent) == 1 and c.sent[0].extended_hours is True
          and c.sent[0].time_in_force.value == "day")
    ev_day = json.load(open(pt.PENDING_ORDERS_JSON))["evening_date"]
    check(f"{name}: the fraction goes to the next 9 AM CT check", ev_day == ("2026-10-22" if name == "pre-market" else "2026-10-23"), ev_day)
    nine = datetime(2026, 10, 23 if name == "pre-market" else 26, 9, 5, tzinfo=run_all.CT)
    okd = run_all.fill_check_allowed(nine, pt.PENDING_ORDERS_JSON)[0] and not run_all.fill_check_allowed(nine - timedelta(minutes=10), pt.PENDING_ORDERS_JSON)[0]
    check(f"{name}: the fill check opens at that 9 AM ({nine:%a %b %-d}) and the row is not superseded by then",
          okd and pt.superseded_orders(nine, pt.PENDING_ORDERS_JSON) == [])

# fail closed / skips
cases = [("stale quote in regular hours", dict(age=120), {}, None),
         ("open order already working", {}, dict(open_orders=[NS(symbol="XYZ", status=NS(value="new"))]), "open order"),
         ("broker orders unreadable", {}, dict(fail=True), "could not be read"),
         ("wide spread", dict(quote=(bid, bid * 1.02)), {}, None)]
for name, kw, ckw, msg in cases:
    c = Client({"XYZ": 10.4}, **ckw)
    setup(client=c, quote=kw.get("quote", (bid, bid + 0.04)), age=kw.get("age", 0))
    n0 = len(runlog())
    es_run(REG)
    es_run(REG + timedelta(minutes=10))
    check(f"{name}: nothing sent" + (", logged once" if msg else ""), c.sent == [] and
          (msg is None or sum(msg in x["message"] for x in runlog()[n0:]) == 1), runlog()[n0:])
c = Client({"XYZ": 10.4})
setup(client=c, quote=(bid, bid + 0.04))
open(run_all.TRADE_LOCK, "w").write(json.dumps({"pid": os.getpid(), "at": datetime.now(run_all.CT).isoformat()}))
es_run(REG)
check("the 2:30 PM trade job is running: the stop waits (no order)", c.sent == [] and "waiting" in pd.read_csv(es.STATUS_CSV).Status[0])
c = Client({"XYZ": 0.4})
setup(held={"XYZ": 0.4}, client=c, quote=(bid, bid + 0.04))
es_run(REG)
p = json.load(open(pt.PENDING_ORDERS_JSON))["orders"][0]
check("only a fraction held: no order now, the 9 AM fill check sells it", c.sent == [] and p["order_qty"] == 0 and p["order_id"] is None and p["qty"] == 0.4)
c = Client({"XYZ": 10.4}, prior=[NS(id="x", client_order_id="live-stop-20261029-XYZ", status=NS(value="rejected"), filled_qty="0", symbol="XYZ")])
setup(client=c, quote=(bid, bid + 0.04))
es_run(REG)
check("a rejected stop sale is retried once with -r2", len(c.sent) == 1 and c.sent[0].client_order_id == "live-stop-20261029-XYZ-r2")
c = Client({"XYZ": 10.4})
setup(client=c, quote=(bid, bid + 0.04))
json.dump({"evening_date": "2026-10-21", "submitted_at_ct": "x", "target_source": "auto", "as_of": "2026-10-21",
           "orders": [{"symbol": "AAA", "side": "BUY", "qty": 1.5}]}, open(pt.PENDING_ORDERS_JSON, "w"))
es_run(REG)
p = json.load(open(pt.PENDING_ORDERS_JSON))
check("an existing pending file keeps its dates and rows (the stop row is added)",
      p["evening_date"] == "2026-10-21" and [o["symbol"] for o in p["orders"]] == ["AAA", "XYZ"], p)

# idle / dry run
c = Client({"XYZ": 10.4})
d, acct, made = setup(client=c, quote=(bid, bid + 0.04))
es_run(at("2026-10-24 11:00"))
check("weekend: idle, no account read", acct.calls == 0 and not os.path.exists(es.STATUS_CSV))
far = E.assign(**{"Earnings Date": E["Earnings Date"] + pd.Timedelta(days=60)})
d, acct, made = setup(client=c, quote=(bid, bid + 0.04), earnings=far)
es_run(REG)
check("no report near: no account read, an empty status file", acct.calls == 0 and pd.read_csv(es.STATUS_CSV).empty)
d, acct, made = setup(client=c, quote=(bid, bid + 0.04))
n0 = len(runlog())
import io  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
buf = io.StringIO()
with redirect_stdout(buf):
    es_run(at("2026-10-24 11:00"), dry_run=True)
out = buf.getvalue()
check("dry run: shows the stock at its stop, no trading client, no order, no files, no run-log row",
      "AT STOP - would sell 10 whole shares" in out and "XYZ" in out and made == [] and c.sent == [] and len(runlog()) == n0
      and not any(os.path.exists(x) for x in (es.STATE_JSON, es.STATUS_CSV, pt.PENDING_ORDERS_JSON)), out)

src = open(os.path.join(ROOT, "earnings_stop.py")).read()
check("the only order call is the one stop SELL (no cancel, replace, close or buy)",
      src.count("submit_order(") == 1 and not any(w in src for w in ("cancel_order", "replace_order", "close_position", "OrderSide.BUY", "MarketOrderRequest")))
print(f"\n{len(FAIL)} failed" if FAIL else "\nEARNINGS STOP OK")
sys.exit(1 if FAIL else 0)
