"""Mocked tests (zero broker calls, zero network) for the live earnings-day 5% drop stop (earnings_stop.py, t192u).

Earnings day: AM report -> that day (pre, regular, after); PM / unknown -> that day (all sessions) + the next trading
day's pre-market and regular session; weekend date -> the next trading day. Reference = the previous trading day's close
(for the PM next morning: the earnings-day close). At 5% or more below: one whole-share limit SELL at the bid - 0.05%,
extended_hours, DAY; the fraction to the 9 AM fill check; never twice (state + broker client id); fails closed on
unreadable reads, missing / stale / wide quotes, open orders and running jobs. The 30-second loop runs only while a held
stock is watched (one loop at a time). Dry run: no trading client, no files.

Run: cd <folder> && python3 tests/test_earnings_stop.py
"""
import csv
import io
import json
import os
import plistlib
import sys
import tempfile
from contextlib import redirect_stdout
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
ALL, AMS = ("pre", "regular", "after"), ("pre", "regular")

# ----------------------------------------------------------------------------- earnings days and sessions
check("rule: 5% drop, checked every 30 seconds", es.DROP == 0.05 and es.INTERVAL_S == 30)
check("AM report on a trading day: that day, all three sessions", es.watch_days(D("2026-10-29"), "AM") == [(D("2026-10-29"), ALL)])
check("PM report: the report day (all sessions) + the next day's pre-market and regular session",
      es.watch_days(D("2026-10-29"), "PM") == [(D("2026-10-29"), ALL), (D("2026-10-30"), AMS)])
check("Friday PM report: Friday + Monday morning", es.watch_days(D("2026-10-30"), "PM") == [(D("2026-10-30"), ALL), (D("2026-11-02"), AMS)])
check("unknown time = PM (covers both reactions)", es.watch_days(D("2026-10-29"), "") == es.watch_days(D("2026-10-29"), "PM"))
check("a report dated on a weekend: the next trading day, all sessions", es.watch_days(D("2026-10-31"), "AM") == [(D("2026-11-02"), ALL)])
E = pd.DataFrame({"Symbol": ["XYZ", "XYZ", "AMX", "NOT"],
                  "Earnings Date": [D("2026-10-29"), D("2026-07-30"), D("2026-10-29"), D("2026-10-29")],
                  "Time": ["PM", "PM", "AM", np.nan]})
ev = lambda s, d, ses=None: es.active_event(s, E, d, ses)
check("PM: not watched the day before; watched on the day and the next morning; not the next after-hours",
      ev("XYZ", "2026-10-28") is None and ev("XYZ", "2026-10-29", "after") is not None and ev("XYZ", "2026-10-30", "pre") is not None
      and ev("XYZ", "2026-10-30", "regular") is not None and ev("XYZ", "2026-10-30", "after") is None and ev("XYZ", "2026-11-02") is None)
check("AM: after hours of the report day watched, the next day not", ev("AMX", "2026-10-29", "after") is not None and ev("AMX", "2026-10-30") is None)
check("last watched day = the buy-back block end (PM: next day, AM: the day)", ev("XYZ", "2026-10-29")[4] == D("2026-10-30")
      and ev("AMX", "2026-10-29")[4] == D("2026-10-29"))
sess = es.session_now
check("sessions on a weekday: 3:59 none, 4:00 pre, 9:30 regular, 16:00 after, 19:59 after, 20:00 none",
      [sess(at(f"2026-10-26 {t}")) for t in ("03:59", "04:00", "09:29", "09:30", "15:59", "16:00", "19:59", "20:00")]
      == [None, "pre", "pre", "regular", "regular", "after", "after", None])
check("weekend and holiday are idle", sess(at("2026-10-24 10:00")) is None and sess(at("2026-11-26 10:00")) is None)
check("early close (Nov 27): regular to 13:00, after hours to 17:00 ET",
      [sess(at(f"2026-11-27 {t}")) for t in ("12:59", "13:00", "16:59", "17:00")] == ["regular", "after", "after", None])

# ----------------------------------------------------------------------------- reference close
days = pd.date_range("2026-10-19", "2026-10-30", freq=be.NYSE_SESSION)
closes = {d: 100.0 for d in days}
closes[D("2026-10-29")] = 90.0                                                 # the earnings-day close
bars = pd.DataFrame({"Symbol": "XYZ", "Date": list(closes), "Close": list(closes.values())})
bars = pd.concat([bars, bars.assign(Symbol="AMX", Close=100.0)], ignore_index=True)   # AMX: flat 100
check("reference on the report day = the previous day's close (Oct 28: 100)", es.prev_close(bars[bars.Symbol == "XYZ"], D("2026-10-29")) == 100.0)
check("reference the next morning = the earnings-day close (Oct 29: 90)", es.prev_close(bars[bars.Symbol == "XYZ"], D("2026-10-30")) == 90.0)
check("no bar before the day -> no reference", es.prev_close(bars[bars.Symbol == "XYZ"], D("2026-10-01")) is None)

# ----------------------------------------------------------------------------- the live check, fully mocked

class Account:
    def __init__(self, held):
        self.held, self.calls = held, 0

    def position_dicts(self):
        self.calls += 1
        return [{"symbol": s, "qty": str(q)} for s, q in self.held.items()]

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

CLOCK = [at("2026-10-29 11:00")]

def setup(held=None, client=None, quote=(96.0, 96.04), age=0, earnings=E, quote_err=None):
    d = tempfile.mkdtemp(dir=TMP)
    es.STATE_JSON, es.STATUS_CSV, es.LOOP_LOCK = (os.path.join(d, n) for n in ("state.json", "stops.csv", ".loop.lock"))
    run_all.TRADE_LOCK, run_all.FILL_LOCK = os.path.join(d, ".trade.lock"), os.path.join(d, ".fill.lock")
    pt.PENDING_ORDERS_JSON = os.path.join(d, "pending.json")
    acct = Account(held if held is not None else {"XYZ": 10.4, "QQQ": 3})
    es.account = lambda: acct
    es.be.load_earnings = lambda path=None: earnings
    es.daily_bars = lambda syms, now: bars[bars["Symbol"].isin(syms) & (bars["Date"] < D(now.astimezone(ET).date()))]
    made = []
    es.trading_client = lambda: made.append(1) or client

    def lq(s):
        if quote_err:
            raise RuntimeError(quote_err)
        return quote[0], quote[1], CLOCK[0] - timedelta(seconds=age), "sip"
    es.latest_quote = lq
    return d, acct, made

def runlog():
    p = os.environ["STOCK_ANALYSIS_RUN_LOG"]
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []

def run(now, **kw):
    CLOCK[0] = now
    return es.run_check(now=now, **kw)

REG = at("2026-10-29 11:00")                                                   # report day (PM), reference 100, trigger 95
# above the trigger: no order, one 'watching' row, status written
c = Client({"XYZ": 10.4})
d, acct, made = setup(client=c)
n0 = len(runlog())
run(REG)
run(REG + timedelta(seconds=30))
st = pd.read_csv(es.STATUS_CSV)
check("4% down (96 vs 100): no order and no trading client", c.sent == [] and made == [])
check("status: XYZ watched (QQQ has no report), reference 100, trigger 95", list(st.Symbol) == ["XYZ"] and st.Prev_Close[0] == 100
      and st.Trigger[0] == 95 and st.Status[0] == "watching", st.to_dict("records"))
check("'watching' run-log row written once", sum("stop watching XYZ" in r["message"] for r in runlog()[n0:]) == 1, runlog()[n0:])

# exactly 5% down in regular hours: one whole-share sale, extended-hours eligible
c = Client({"XYZ": 10.4})
d, acct, made = setup(client=c, quote=(94.98, 95.02))                          # mid 95.00 = exactly 5% below
n0 = len(runlog())
run(REG)
r = c.sent[0] if c.sent else None
check("exactly 5% below (mid 95.00): one SELL of the 10 whole shares", len(c.sent) == 1 and r.qty == 10 and r.side.value == "sell", c.sent)
check("limit = bid - 0.05% on the tick, DAY, extended_hours (works in every session)",
      r and r.limit_price == pt._tick(94.98 * (1 - pt.LIMIT_OFFSET), False) and r.time_in_force.value == "day" and r.extended_hours is True)
check("fixed client order id live-stop-20261029-XYZ", r and r.client_order_id == "live-stop-20261029-XYZ")
pend = json.load(open(pt.PENDING_ORDERS_JSON))
row = pend["orders"][0]
check("pending row: qty 10.4 (the 0.4 fraction for the 9 AM check), order_qty 10, exit, order id",
      len(pend["orders"]) == 1 and row["qty"] == 10.4 and row["order_qty"] == 10 and row["exit"] is True and row["order_id"] == "oid-0"
      and row["side"] == "SELL" and pend["evening_date"] == "2026-10-29", pend)
log = [x for x in runlog()[n0:] if "Earnings-day stop: XYZ" in x["message"]]
check("one run-log row for the sale (money moved unknown until filled)", len(log) == 1 and log[0]["run"] == "Earnings stop"
      and log[0]["money_moved"] == "unknown" and "fractional" in log[0]["message"], log)
check("the fill-check lock is released", not os.path.exists(run_all.FILL_LOCK))
check("no buy back through the last watched day (PM: Fri Oct 30), allowed after",
      set(pt.earnings_stop_blocked("2026-10-30", es.STATE_JSON)) == {"XYZ"} and pt.earnings_stop_blocked("2026-10-31", es.STATE_JSON) == {})
run(REG + timedelta(seconds=30))
check("never twice: the next check sends nothing", len(c.sent) == 1)
os.remove(es.STATE_JSON)
run(REG + timedelta(minutes=1))
check("never twice even without the state file (the client id is on the broker)", len(c.sent) == 1)

# the next morning (PM report): reference = the earnings-day close 90 -> trigger 85.50
for t, q, n_sent, name in (("2026-10-30 07:00", (86.0, 86.04), 0, "pre-market 4.4% down: no sale"),
                           ("2026-10-30 07:00", (85.40, 85.44), 1, "pre-market 5.1% below the earnings-day close: sale"),
                           ("2026-10-30 10:15", (85.40, 85.44), 1, "next regular session: sale"),
                           ("2026-10-30 17:30", (50.0, 50.04), 0, "next after-hours: not watched for a PM report")):
    c = Client({"XYZ": 10.4})
    setup(client=c, quote=q, age=300 if t.endswith("07:00") or t.endswith("17:30") else 0)
    run(at(t))
    check(f"{name}", len(c.sent) == n_sent and all(x.extended_hours is True and x.qty == 10 for x in c.sent), c.sent)
c = Client({"AMX": 5})
setup(held={"AMX": 5}, client=c, quote=(90.0, 90.04), age=300)
run(at("2026-10-29 17:30"))                                                    # AM report: its after hours are watched
check("AM report: after hours of the report day watched (10% down vs 100 -> sale)", [x.qty for x in c.sent] == [5], c.sent)
c = Client({"XYZ": 10.4})
setup(client=c, quote=(90.0, 90.04))
run(at("2026-10-28 11:00"))
check("the day before a PM report: not watched (no order even 10% down)", c.sent == [])
ev_day = None
c = Client({"XYZ": 10.4})
setup(client=c, quote=(85.4, 85.44), age=300)
run(at("2026-10-30 07:00"))
ev_day = json.load(open(pt.PENDING_ORDERS_JSON))["evening_date"]
nine = datetime(2026, 10, 30, 9, 5, tzinfo=run_all.CT)
check("pre-market sale: the fraction goes to that morning's 9 AM CT check (dated the previous session)",
      ev_day == "2026-10-29" and run_all.fill_check_allowed(nine, pt.PENDING_ORDERS_JSON)[0]
      and pt.superseded_orders(nine, pt.PENDING_ORDERS_JSON) == [], ev_day)

# fail closed / skips
bid = 94.0
cases = [("stale quote in regular hours", dict(age=120), {}, None),
         ("missing quote (no SIP and no IEX quote)", dict(quote_err="no quote from sip or iex"), {}, None),
         ("one-sided quote", dict(quote=(0.0, 94.04)), {}, None),
         ("wide spread", dict(quote=(bid, bid * 1.02)), {}, None),
         ("open order already working", {}, dict(open_orders=[NS(symbol="XYZ", status=NS(value="new"))]), "open order"),
         ("broker orders unreadable", {}, dict(fail=True), "could not be read")]
for name, kw, ckw, msg in cases:
    c = Client({"XYZ": 10.4}, **ckw)
    setup(client=c, quote=kw.get("quote", (bid, bid + 0.04)), age=kw.get("age", 0), quote_err=kw.get("quote_err"))
    n0 = len(runlog())
    run(REG)
    run(REG + timedelta(seconds=30))
    check(f"{name}: nothing sent" + (", logged once" if msg else ""), c.sent == [] and
          (msg is None or sum(msg in x["message"] for x in runlog()[n0:]) == 1), runlog()[n0:])
c = Client({"XYZ": 10.4})
setup(client=c, quote=(bid, bid + 0.04))
open(run_all.TRADE_LOCK, "w").write(json.dumps({"pid": os.getpid(), "at": datetime.now(run_all.CT).isoformat()}))
run(REG)
check("the 2:30 PM trade job is running: the stop waits (no order)", c.sent == [] and "waiting" in pd.read_csv(es.STATUS_CSV).Status[0])
c = Client({"XYZ": 0.4})
setup(held={"XYZ": 0.4}, client=c, quote=(bid, bid + 0.04))
run(REG)
check("only a fraction held, regular hours: the 0.4 is sold now (no extended hours)", [(x.qty, x.extended_hours) for x in c.sent] == [(0.4, False)], c.sent)
c = Client({"XYZ": 0.4})
setup(held={"XYZ": 0.4}, client=c, quote=(bid, bid + 0.04), age=60)
run(at("2026-10-29 17:30"))
p = json.load(open(pt.PENDING_ORDERS_JSON))["orders"][0]
check("only a fraction held, after hours: no order now, the 9 AM fill check sells it", c.sent == [] and p["order_qty"] == 0 and p["order_id"] is None and p["qty"] == 0.4)
c = Client({"XYZ": 10.4}, prior=[NS(id="x", client_order_id="live-stop-20261029-XYZ", status=NS(value="rejected"), filled_qty="0", symbol="XYZ")])
setup(client=c, quote=(bid, bid + 0.04))
run(REG)
check("a rejected stop sale is retried once with -r2", len(c.sent) == 1 and c.sent[0].client_order_id == "live-stop-20261029-XYZ-r2")
c = Client({"XYZ": 10.4})
setup(client=c, quote=(bid, bid + 0.04))
json.dump({"evening_date": "2026-10-28", "submitted_at_ct": "x", "target_source": "auto", "as_of": "2026-10-28",
           "orders": [{"symbol": "AAA", "side": "BUY", "qty": 1.5}]}, open(pt.PENDING_ORDERS_JSON, "w"))
run(REG)
p = json.load(open(pt.PENDING_ORDERS_JSON))
check("an existing pending file keeps its dates and rows (the stop row is added)",
      p["evening_date"] == "2026-10-28" and [o["symbol"] for o in p["orders"]] == ["AAA", "XYZ"], p)
c = Client({"XYZ": 10.4})
setup(client=c, quote=(bid, bid + 0.04))
es.trading_client = lambda: (_ for _ in ()).throw(SystemExit("No Alpaca LIVE keys"))
n0 = len(runlog())
run(REG)
run(REG + timedelta(seconds=30))
check("no trading client (e.g. keys missing): no crash, one failed row a day, retried later",
      c.sent == [] and sum("stop check failed" in x["message"] for x in runlog()[n0:]) == 1 and "error" in pd.read_csv(es.STATUS_CSV).Status[0])

c = Client({"XYZ": 10.4})
setup(client=c, quote=(80.0, 80.04))
full = es.daily_bars
es.daily_bars = lambda syms, now: full(syms, now)[lambda b: b["Date"] != D("2026-10-28")]   # the Oct 28 bar is missing
n0 = len(runlog())
run(REG)
run(REG + timedelta(seconds=30))
st = pd.read_csv(es.STATUS_CSV)
check("the previous day's bar missing: no stop on an older close (20% down, nothing sent), warned once",
      c.sent == [] and "not available yet" in st.Status[0] and sum("has no earnings-day stop" in x["message"] for x in runlog()[n0:]) == 1,
      (st.to_dict("records"), runlog()[n0:]))
check("previous trading day: Mon -> Fri, after a holiday -> the day before it",
      es.prev_session(D("2026-11-02")) == D("2026-10-30") and es.prev_session(D("2026-11-27")) == D("2026-11-25"))

# the 30-second loop
c = Client({"XYZ": 10.4})
setup(client=c, quote=(96.0, 96.04))
clock = iter([REG + timedelta(seconds=30 * i) for i in range(10)])
naps, quotes = [], iter([(96.0, 96.04), (95.5, 95.54), (94.0, 94.04), (93.0, 93.04)])

def nap(s):
    naps.append(s)
    q = next(quotes)
    es.latest_quote = lambda sym, q=q: (q[0], q[1], CLOCK[0], "sip")

def now_fn():
    CLOCK[0] = next(clock)
    return CLOCK[0]

n = es.loop(now_fn=now_fn, sleep_fn=nap, max_checks=10)
check("loop: a check every 30 s while watched; stops after the sale (4 checks: 96, 96, 95.5, 94 -> sold)",
      n == 4 and naps == [30, 30, 30] and len(c.sent) == 1 and not os.path.exists(es.LOOP_LOCK), (n, naps, len(c.sent)))
setup(client=Client({"XYZ": 10.4}), earnings=E.assign(**{"Earnings Date": E["Earnings Date"] + pd.Timedelta(days=60)}))
naps = []
check("loop on a quiet day: one check, no sleep, exits", es.loop(now_fn=lambda: REG, sleep_fn=naps.append) == 1 and naps == [])
open(es.LOOP_LOCK, "w").write(json.dumps({"pid": os.getpid()}))
check("a second loop exits at once while one is running (pid lock)", es.loop(now_fn=lambda: REG, sleep_fn=naps.append) == 0)
os.remove(es.LOOP_LOCK)

# idle / dry run
c = Client({"XYZ": 10.4})
d, acct, made = setup(client=c, quote=(bid, bid + 0.04))
run(at("2026-10-24 11:00"))
check("weekend: idle, no account read", acct.calls == 0 and not os.path.exists(es.STATUS_CSV))
far = E.assign(**{"Earnings Date": E["Earnings Date"] + pd.Timedelta(days=60)})
d, acct, made = setup(client=c, quote=(bid, bid + 0.04), earnings=far)
run(REG)
check("no report near: no account read, an empty status file", acct.calls == 0 and pd.read_csv(es.STATUS_CSV).empty)
d, acct, made = setup(client=c, quote=(bid, bid + 0.04))
n0 = len(runlog())
buf = io.StringIO()
with redirect_stdout(buf):
    run(REG, dry_run=True)
out = buf.getvalue()
check("dry run: shows the stock at its trigger, no trading client, no order, no files, no run-log row",
      "AT TRIGGER - would sell 10 whole shares" in out and "XYZ" in out and made == [] and c.sent == [] and len(runlog()) == n0
      and not any(os.path.exists(x) for x in (es.STATE_JSON, es.STATUS_CSV, pt.PENDING_ORDERS_JSON)), out)

src = open(os.path.join(ROOT, "earnings_stop.py")).read()
check("the only order call is the one stop SELL (no cancel, replace, close or buy)",
      src.count("submit_order(") == 1 and not any(w in src for w in ("cancel_order", "replace_order", "close_position", "OrderSide.BUY", "MarketOrderRequest")))
check("the ATR stop code is gone from the live job", not any(w in src for w in ("atr_wilder", "stop_level", "K_ATR", "ARM_DAYS", "first_buy_dates")))
pl = plistlib.load(open(os.path.join(ROOT, "launchd", "com.stockanalysis.earningsstop.plist"), "rb"))
check("launchd job: starts the --loop every 5 minutes and at login", pl["ProgramArguments"][-1] == "--loop"
      and pl["StartInterval"] == 300 and pl["RunAtLoad"] is True, pl)
print(f"\n{len(FAIL)} failed" if FAIL else "\nEARNINGS STOP OK")
sys.exit(1 if FAIL else 0)
