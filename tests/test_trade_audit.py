"""Trade audit (trade_audit.py + the Details panel), hand-worked cases. Pure functions with fake orders / fills, then the
app with a FAKE Alpaca account (GET only, no network, no keys):
  - who sent an order (bot decision / fill check / rest / replacement / earnings-day stop / manual) and its plan price
    from the client order id (ids cut at 48 characters not trusted)
  - slippage vs the plan price in % and $ (+ = it cost money) for buys and sells; the ledger and its summary
  - round trips: FIFO buy price / sell price per sale, partial lots, a sale with no purchase in the history
  - reconciliation: blocked account, a manual open order, an untracked bot order, a pending row Alpaca does not list,
    holdings outside the strategy with / without a queued sale, targets not held, weight drift
  - alerts: daily loss 3% / 5%, Mac clock drift, NYSE calendar vs Alpaca (holiday, unscheduled closure, early close),
    a rejected order, a fill over 1% worse than plan (last 7 days only), Alpaca unreadable
  - each alert reported once (daily ones once a day) to the run log (a test file), the CSVs written atomically
  - only allow-listed GET paths; no order code in trade_audit.py
Run: python tests/run_tests.py  (or python tests/test_trade_audit.py)"""
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
TMP = tempfile.mkdtemp(prefix="trade_audit_test_")
os.environ["STOCK_ANALYSIS_RUN_LOG"] = os.path.join(TMP, "run_log.csv")          # never the real run log
os.environ["STOCK_ANALYSIS_AUDIT_STATE"] = os.path.join(TMP, "audit_state.json")
import pandas as pd  # noqa: E402

import trade_audit as ta  # noqa: E402

CT = ZoneInfo("America/Chicago")
FAIL = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)


def close(a, b, tol=0.005):
    return a is not None and b is not None and a == a and abs(float(a) - float(b)) <= tol


def O(cid, sym, side, qty, filled, fill_px, status="filled", at="2026-10-02T20:31:10Z", limit=None, oid=None, kind="limit"):
    return {"id": oid or f"id-{cid}", "client_order_id": cid, "symbol": sym, "side": side, "qty": str(qty), "filled_qty": str(filled),
            "filled_avg_price": None if fill_px is None else str(fill_px), "status": status, "submitted_at": at,
            "filled_at": at if filled else None, "type": kind, "time_in_force": "day", "extended_hours": True,
            "limit_price": None if limit is None else str(limit)}


# ---------------------------------------------------------------- 1. client order ids
check("source: decision / fill check / rest / replacement / stop / manual",
      [ta.order_source(c) for c in ("live-20261002-BUY-TWLO-19-29458", "live-fill-20261002-SELL-FIG-5-2136",
                                    "live-rest-20261002-BUY-AMD-1-63391", "live-fill-20261002-BUY-AMD-1-63391-c",
                                    "live-stop-20261029-BE-1", "fa38dbe4-aa57-4b2d-8fbe-1621a63967d9", "")]
      == ["bot: decision", "bot: fill check", "bot: rest that did not fit", "bot: replacement at a fresh quote",
          "earnings-day stop", "manual / other", "manual / other"])
check("plan price from the id: TWLO 29458 -> $294.58, FIG fill check 2136 -> $21.36, -r2 / -c suffixes kept",
      ta.plan_price("live-20261002-BUY-TWLO-19-29458") == 294.58 and ta.plan_price("live-fill-20261002-SELL-FIG-5-2136") == 21.36
      and ta.plan_price("live-fill-20261002-SELL-FIG-5-2136-r2") == 21.36 and ta.plan_price("live-fill-20261002-SELL-FIG-5-2136-c") == 21.36)
long_id = ("live-fill-20261002-SELL-ABCDEFGHIJK-123456-98765")[:48]
check("no plan price: a 48-character (possibly cut) id, a stop id, a manual id, a zero price",
      len(long_id) == 48 and ta.plan_price(long_id) is None and ta.plan_price("live-stop-20261029-BE-1") is None
      and ta.plan_price("fa38dbe4-aa57") is None and ta.plan_price("live-20261002-BUY-X-1-0") is None)

# ---------------------------------------------------------------- 2. slippage, ledger, summary
check("buy plan $100 filled $101 x 10 -> +1% / +$10 (cost)", ta.slippage("buy", 100, 101, 10) == (1.0, 10.0))
check("sell plan $100 filled $99 x 5 -> +1% / +$5 (cost); filled $101 -> -1% / -$5 (gain)",
      ta.slippage("sell", 100, 99, 5) == (1.0, 5.0) and ta.slippage("sell", 100, 101, 5) == (-1.0, -5.0))
check("no plan price or fill -> nan; an exact fill is 0.0 (not -0.0)", all(math.isnan(v) for v in ta.slippage("buy", None, 101, 1))
      and all(math.isnan(v) for v in ta.slippage("buy", 100, None, 1)) and str(ta.slippage("sell", 22.37, 22.37, 7)[0]) == "0.0")
ORD = [O("live-20261002-BUY-AAA-10-10000", "AAA", "buy", 10, 10, 101, limit=100.05, at="2026-10-02T20:31:00Z"),
       O("live-20261002-SELL-BBB-5-5000", "BBB", "sell", 5, 5, 49.5, limit=49.97, at="2026-10-02T20:30:00Z"),
       O("live-20261002-SELL-CCC-6-2000", "CCC", "sell", 6, 1, 20, status="expired", limit=19.99, at="2026-10-02T20:30:30Z"),
       O("manual-1", "DDD", "buy", 2, 2, 30, kind="market", at="2026-10-02T19:00:00Z")]
led = ta.ledger(ORD)
check("ledger: one row per order, newest first, CT times", list(led["Symbol"]) == ["AAA", "CCC", "BBB", "DDD"]
      and led["Submitted_CT"].iloc[0] == pd.Timestamp("2026-10-02 15:31:00") and list(led.columns) == ta.LEDGER_COLS)
r = led.set_index("Symbol")
check("ledger: AAA +1% / +$10, BBB sell $0.50 under plan x 5 = +$2.50 (+1%), CCC 1 of 6 at plan, DDD manual (no plan)",
      close(r.loc["AAA", "Slippage_vs_Plan_$"], 10) and close(r.loc["BBB", "Slippage_vs_Plan_%"], 1.0) and close(r.loc["BBB", "Slippage_vs_Plan_$"], 2.5)
      and close(r.loc["CCC", "Slippage_vs_Plan_$"], 0) and r.loc["CCC", "Status"] == "expired" and pd.isna(r.loc["DDD", "Plan_Price"])
      and r.loc["DDD", "Source"] == "manual / other", r.to_dict("index"))
s = ta.slippage_summary(led)
check("summary: 3 bot fills with a plan price, traded $1,010 + $247.50 + $20 = $1,277.50, cost $12.50 (0.978%)",
      s["filled_with_plan"] == 3 and close(s["traded"], 1277.5) and close(s["cost"], 12.5) and close(s["cost_pct"], 12.5 / 1277.5 * 100)
      and s["worst"]["Symbol"].iloc[0] == "AAA", s)

# ---------------------------------------------------------------- 3. round trips
def FL(sym, side, qty, px, t, oid="x"):
    return {"id": f"{t}-{sym}", "symbol": sym, "side": side, "qty": str(qty), "price": str(px), "transaction_time": t, "order_id": oid}


# Dates on/after AUDIT_START (2026-10-02); span kept so Days_Held still 56 / etc.
fills = [FL("AAA", "buy", 10, 100, "2026-10-02T15:00:00Z", "b1"), FL("AAA", "buy", 5, 110, "2026-10-30T15:00:00Z", "b2"),
         FL("AAA", "sell", 12, 120, "2026-11-27T15:00:00Z", "s1"), FL("AAA", "sell", 5, 90, "2026-12-04T15:00:00Z", "s2")]
rt = ta.round_trips(fills, [{"id": "s1", "client_order_id": "live-20261127-SELL-AAA-12-12000"}]).sort_values(["Sold_CT", "Bought_CT"])
rows = rt.to_dict("records")
check("FIFO: the Nov 27 sale of 12 = 10 bought $100 (+$200, +20%, 56 days) + 2 bought $110 (+$20)",
      close(rows[0]["Shares"], 10) and close(rows[0]["Buy_Price"], 100) and close(rows[0]["P/L $"], 200) and close(rows[0]["P/L %"], 20)
      and rows[0]["Days_Held"] == 56 and close(rows[1]["Shares"], 2) and close(rows[1]["P/L $"], 20)
      and rows[0]["Sell_Source"] == "bot: decision", rows)
check("the Dec 4 sale of 5 = the 3 left at $110 (-$60) + 2 with no purchase in the history (noted, no P/L)",
      close(rows[2]["Shares"], 3) and close(rows[2]["P/L $"], -60) and close(rows[3]["Shares"], 2) and pd.isna(rows[3]["Buy_Price"])
      and "before the account history" in rows[3]["Note"], rows[2:])

# ---------------------------------------------------------------- 4. reconciliation
picks = pd.DataFrame({"Symbol": ["AAA", "BBB", "EEE"], "Strategy_Weight": [0.5, 0.2, 0.1]})
pos = [{"symbol": "AAA", "qty": "10", "market_value": "520"}, {"symbol": "BBB", "qty": "4", "market_value": "200"},
       {"symbol": "FIG", "qty": "5", "market_value": "100"}, {"symbol": "ZZZ", "qty": "1", "market_value": "50"}]
open_orders = [O("manual-9", "QQQ", "buy", 1, 0, None, status="new", oid="m9"),
               O("live-20261005-BUY-AAA-1-10000", "AAA", "buy", 1, 0, None, status="accepted", oid="t1"),
               O("live-20261005-BUY-BBB-1-5000", "BBB", "buy", 1, 0, None, status="new", oid="t2")]
pend = {"orders": [{"symbol": "FIG", "side": "SELL", "qty": 5, "order_id": "gone-1"}, {"symbol": "AAA", "side": "BUY", "order_id": "t1"}]}
data = {"account": {"status": "ACTIVE", "trading_blocked": False, "equity": "1000"}, "positions": pos, "orders": ORD + open_orders}
f = {x["Key"].split("|")[0]: x for x in ta.reconcile(data, pend, {"sold": {}}, picks)}
check("account active -> ok", f["account-flags"]["Level"] == "ok")
check("manual open order -> warning; untracked open bot order (BBB) -> warning; tracked one (AAA, in pending) -> none",
      f["manual-open"]["Level"] == "warning" and "QQQ" in f["manual-open"]["Check"] and f["orphan-open"]["Level"] == "warning"
      and "BBB" in f["orphan-open"]["Check"] and sum(1 for x in ta.reconcile(data, pend, {}, picks) if x["Key"].startswith("orphan")) == 1)
check("pending FIG row whose order Alpaca does not list -> warning", f["pending-missing"]["Level"] == "warning" and "gone-1" in f["pending-missing"]["Check"])
check("FIG not in the strategy but its sale is queued -> info; ZZZ not in it and no sale -> warning",
      f["leftover"]["Level"] == "info" and "FIG" in f["leftover"]["Check"] and f["untracked"]["Level"] == "warning" and "ZZZ" in f["untracked"]["Check"])
check("EEE in the strategy, not held -> info; AAA 52% vs 50% is within 2 points, BBB 20% = 20% -> no drift row",
      "EEE" in f["targets-not-held"]["Check"] and "drift" not in f)
data2 = {**data, "positions": [{"symbol": "AAA", "qty": "10", "market_value": "600"}], "orders": [],
         "account": {"status": "ACTIVE", "trade_suspended_by_user": True, "equity": "1000"}}
f2 = {x["Key"].split("|")[0]: x for x in ta.reconcile(data2, {}, {}, picks)}
check("trading suspended -> failed; AAA 60% vs 50% -> drift info; no open orders -> ok; every holding in the strategy -> ok",
      f2["account-flags"]["Level"] == "failed" and "trade_suspended_by_user" in f2["account-flags"]["Check"] and "AAA 60.0% vs 50.0%" in f2["drift"]["Check"]
      and f2["open-orders"]["Level"] == "ok" and f2["holdings"]["Level"] == "ok")

# ---------------------------------------------------------------- 5. alerts
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=CT)          # Saturday


def A(acct=None, clock=None, local=None, led_=None, errors=(), now=NOW):
    return {x["Key"].split("|")[0]: x for x in ta.alerts({"account": acct, "clock": clock, "clock_local": local, "errors": list(errors)},
                                                         led_, now)}


SAT_CLOCK = {"is_open": False, "timestamp": "2026-10-03T13:00:00-04:00", "next_open": "2026-10-05T09:30:00-04:00",
             "next_close": "2026-10-05T16:00:00-04:00"}
utc_at = datetime(2026, 10, 3, 17, 0, tzinfo=timezone.utc)
check("quiet Saturday: no alerts (next open Mon 9:30 ET as the calendar says, clock in sync)",
      A({"equity": "1000", "last_equity": "1000"}, SAT_CLOCK, utc_at) == {})
check("equity -3.5% vs the last close -> warning; -6% -> failed (alert only)",
      A({"equity": "965", "last_equity": "1000"})["day-loss-3"]["Level"] == "warning"
      and A({"equity": "940", "last_equity": "1000"})["day-loss-5"]["Level"] == "failed" and "day-loss-3" not in A({"equity": "940", "last_equity": "1000"}))
check("Mac clock 2 min ahead of Alpaca -> warning; 30 s -> fine",
      "+120 s" in A(None, SAT_CLOCK, datetime(2026, 10, 3, 17, 2, tzinfo=timezone.utc))["clock"]["Check"]
      and "clock" not in A(None, SAT_CLOCK, datetime(2026, 10, 3, 17, 0, 30, tzinfo=timezone.utc)))
closed_mon = {**SAT_CLOCK, "next_open": "2026-10-06T09:30:00-04:00", "next_close": "2026-10-06T16:00:00-04:00"}
check("Alpaca says next open Tue (an unscheduled Monday closure) -> calendar warning naming both days",
      "Tue Oct 6" in A(None, closed_mon, utc_at)["calendar-open"]["Check"] and "Mon Oct 5" in A(None, closed_mon, utc_at)["calendar-open"]["Check"])
tg = datetime(2026, 11, 26, 10, 0, tzinfo=CT)          # Thanksgiving: closed; Fri Nov 27 closes 1 PM ET
tg_clock = {"is_open": False, "timestamp": "2026-11-26T11:00:00-05:00", "next_open": "2026-11-27T09:30:00-05:00",
            "next_close": "2026-11-27T13:00:00-05:00"}
check("Thanksgiving: next open Fri, early close 1 PM ET = the calendar -> no alert; a 4 PM close there -> early-close warning",
      A(None, tg_clock, datetime(2026, 11, 26, 16, 0, tzinfo=timezone.utc), now=tg) == {}
      and "calendar-close" in A(None, {**tg_clock, "next_close": "2026-11-27T16:00:00-05:00"}, datetime(2026, 11, 26, 16, 0, tzinfo=timezone.utc), now=tg))
pre = datetime(2026, 10, 5, 7, 0, tzinfo=CT)           # Monday before the open: next open is today
check("Monday 7 AM CT: next open today 9:30 ET -> no alert",
      A(None, {**SAT_CLOCK, "timestamp": "2026-10-05T08:00:00-04:00"}, datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc), now=pre) == {})
recent = ta.ledger([O("live-20261002-BUY-AAA-10-10000", "AAA", "buy", 10, 10, 101.5, at="2026-10-02T20:31:00Z", oid="s1"),
                    O("live-20261002-BUY-BBB-10-10000", "BBB", "buy", 10, 0, None, status="rejected", at="2026-10-02T20:31:00Z", oid="r1"),
                    O("live-20260920-BUY-CCC-10-10000", "CCC", "buy", 10, 10, 103, at="2026-09-20T20:31:00Z", oid="old")])
al = A(led_=recent)
check("last 7 days: AAA filled 1.5% over plan -> slippage warning ($15); BBB rejected -> warning; Sep 20 order (13 days) -> none",
      "1.50% worse" in al["slippage"]["Check"] and "$15.00" in al["slippage"]["Check"] and "BBB" in al["rejected"]["Check"]
      and sum(1 for x in ta.alerts({"errors": []}, recent, NOW) if "CCC" in x["Check"]) == 0, al)
check("Alpaca unreadable -> failed alert naming the part", "orders: ConnectionError" in A(errors=["orders: ConnectionError: down"])["unreadable"]["Check"])

# ---------------------------------------------------------------- 6. once-only run-log rows, files
items = ta.alerts({"account": {"equity": "940", "last_equity": "1000"}, "errors": []}, recent, NOW)
first = ta.log_alerts(items, NOW)
again = ta.log_alerts(items, NOW)
nxt = ta.log_alerts(ta.alerts({"account": {"equity": "940", "last_equity": "1000"}, "errors": []}, recent,
                              datetime(2026, 10, 4, 12, 0, tzinfo=CT)), datetime(2026, 10, 4, 12, 0, tzinfo=CT))
log = pd.read_csv(os.environ["STOCK_ANALYSIS_RUN_LOG"])
check("run log: 3 rows the first time (loss, slippage, reject), none on a rerun, only the daily loss again the next day",
      len(first) == 3 and again == [] and [x["Key"].split("|")[0] for x in nxt] == ["day-loss-5"] and len(log) == 4
      and set(log["run"]) == {"Trade audit"} and log["money_moved"].eq("no").all() and log["message"].str.contains("alert only").all(),
      (len(first), again, nxt, len(log)))
check("info / ok findings never go to the run log", ta.new_alerts([{"Level": "info", "Key": "x", "Check": "y"}], NOW) == [])
rep = ta.report({"account": {"status": "ACTIVE", "equity": "1000", "last_equity": "1000"}, "positions": pos[:2], "orders": ORD,
                 "fills": fills, "clock": None, "errors": [], "as_of": NOW}, NOW, pend, {}, picks)
lp, tp = os.path.join(TMP, "ledger.csv"), os.path.join(TMP, "trips.csv")
ta.write_files(rep, lp, tp)
check("report + CSVs: ledger 4 rows, round trips 4 rows, no .tmp left", len(pd.read_csv(lp)) == 4 and len(pd.read_csv(tp)) == 4
      and not any(n.endswith(".tmp") for n in os.listdir(TMP)))

# ---------------------------------------------------------------- 7. fetch: GET paths only, parts fail on their own
import alpaca_paper as ap  # noqa: E402


class Fake(ap.PaperAccount):
    calls, fail = [], set()

    def __init__(self, *a, **k):
        pass

    def _get(self, path, params=None):
        assert path in ap.ALLOWED_PATHS, path
        Fake.calls.append(path)
        if path in Fake.fail:
            raise ap.PaperAccountError(f"GET {path} failed: HTTP 500")
        return {"/account": {"status": "ACTIVE"}, "/positions": pos, "/orders": ORD, "/clock": SAT_CLOCK}.get(path, fills[::-1])


d = ta.fetch(Fake())
check("fetch: account, positions, orders, fills, clock; nothing failed", d["errors"] == [] and len(d["orders"]) == 4 and len(d["fills"]) == 4
      and d["clock"]["next_open"].startswith("2026-10-05") and set(Fake.calls) <= set(ap.ALLOWED_PATHS))
Fake.calls, Fake.fail = [], {"/orders"}
d = ta.fetch(Fake(), positions=pos[:1])
check("fetch(positions=...): /positions not read; a failing /orders is named, the rest still read",
      "/positions" not in Fake.calls and d["orders"] is None and d["positions"] == pos[:1] and d["errors"][0].startswith("orders: PaperAccountError")
      and d["account"] == {"status": "ACTIVE"})
Fake.fail = set()

# ---------------------------------------------------------------- 8. the app renders the panel (fake account, GET only)
from streamlit.testing.v1 import AppTest  # noqa: E402
import streamlit as st  # noqa: E402

APOS = [{"symbol": "AAA", "qty": "10", "avg_entry_price": "100", "cost_basis": "1000", "market_value": "1010", "unrealized_pl": "10",
         "unrealized_plpc": "0.01", "current_price": "101", "change_today": "0", "lastday_price": "101"}]


class AppFake(Fake):
    def _get(self, path, params=None):
        assert path in ap.ALLOWED_PATHS, path
        Fake.calls.append(path)
        if path == "/positions":
            return APOS
        if path == "/account":
            return {"status": "ACTIVE", "equity": "2000", "last_equity": "2000", "cash": "990", "buying_power": "990",
                    "long_market_value": "1010"}
        if path == "/orders":
            return ORD[:1]
        if path == "/clock":                                    # in sync with this Mac (no next open: no calendar check)
            return {"is_open": False, "timestamp": datetime.now(timezone.utc).isoformat()}
        items = [FL("AAA", "buy", 10, 101, "2026-10-02T20:31:05Z", "id-live-20261002-BUY-AAA-10-10000")][::-1]
        return [] if params.get("page_token") else items


real, real_key = ap.PaperAccount, ap.holdings_refresh_key
ap.PaperAccount, ap.holdings_refresh_key = AppFake, (lambda now=None: "k1")
os.environ.pop("STOCK_ANALYSIS_LIVE_HOLDINGS", None)
try:
    Fake.calls = []
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file("app.py", default_timeout=180).run()
    labels = [e.label for e in at.expander]
    check("app: no exceptions, trade audit section before the data settings, rules still last", not at.exception
          and "Trade audit · every order vs its plan price, buy / sell prices, checks (read-only)" in labels and labels[-1] == "Strategy rules",
          ([str(e) for e in at.exception], labels))
    frames = [x.value for x in at.dataframe]
    led_tbl = next((x for x in frames if "Plan_Price" in x.columns), None)
    check("app: order ledger with plan price $100, fill $101, +$10 vs plan", led_tbl is not None and close(led_tbl["Plan_Price"].iloc[0], 100)
          and close(led_tbl["Slippage_vs_Plan_$"].iloc[0], 10), None if led_tbl is None else led_tbl.to_dict("list"))
    check("app: reconciliation table and dated caption", any("Result" in x.columns for x in frames)
          and any("plan price" in c.value and c.value.startswith("As of ") for c in at.caption))
    check("app: AAA held but not in the strategy -> a check line above the cards; no Alpaca-unreadable alert",
          any("AAA 10 sh is held but not in the strategy" in x.value for x in at.warning)
          and not any("could not be read" in x.value for x in at.warning), [x.value for x in at.warning])
    check("app: only allow-listed GET paths; /positions read once (the holdings read is reused)",
          set(Fake.calls) <= set(ap.ALLOWED_PATHS) and Fake.calls.count("/positions") == 1, Fake.calls)
finally:
    ap.PaperAccount, ap.holdings_refresh_key = real, real_key
    st.cache_data.clear()
    st.cache_resource.clear()

src = open(os.path.join(ROOT, "trade_audit.py")).read()
check("trade_audit.py never places or cancels orders (reads only through PaperAccount._get)",
      not any(w in src for w in ("submit_order", "cancel_order", "method=\"POST\"", "TradingClient", "requests.post", "paper_trading_client")))
check("the real run log was not touched", os.environ["STOCK_ANALYSIS_RUN_LOG"].startswith(TMP))
print(f"\n{len(FAIL)} failed" if FAIL else "\nTRADE AUDIT OK")
sys.exit(1 if FAIL else 0)
