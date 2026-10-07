"""The Details tab's live holdings table (alpaca_paper.holdings_table + app.render_live_holdings) with a FAKE Alpaca client:
  - only allow-listed read-only GETs (positions, account, fills with paging), never an order call
  - the fill history is kept: a full read once a day, then only new fills (one request)
  - First bought = the earliest buy still in the position (FIFO: sells use the oldest shares; a full exit restarts)
  - per-stock numbers from Alpaca's fields, the Total row, Weight % of equity; the account vs index ETFs table below it
    (its math: test_benchmark_compare.py)
  - refresh: a new read each minute in market hours (8:30 AM-3:00 PM CT), each hour otherwise; the table reruns itself
  - the dashboard renders it (and a plain message when Alpaca fails, never cached) without exceptions
No network to Alpaca, no keys. Run: python tests/run_tests.py  (or python tests/test_live_holdings.py)"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

import alpaca_paper as ap

FAIL = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)


def fill(sym, side, qty, day, i=0):
    return {"id": f"{day}-{sym}-{i}", "symbol": sym, "side": side, "qty": str(qty), "transaction_time": f"{day}T19:30:00Z"}


FILLS = [fill("AAA", "buy", 10, "2026-09-18"), fill("AAA", "buy", 5, "2026-09-25"), fill("AAA", "sell", 10, "2026-09-28"),
         fill("BBB", "buy", 5, "2026-09-11"), fill("BBB", "sell", 5, "2026-09-14"), fill("BBB", "buy", 3, "2026-09-25"),
         fill("CCC", "buy", 2, "2026-09-18"), fill("CCC", "buy", 2, "2026-09-21"), fill("CCC", "sell", 1, "2026-09-22")]
FILLS += [fill("ZZZ", "buy" if i % 2 == 0 else "sell", 1, "2026-09-01", i) for i in range(120)]   # 120 fills -> 2 pages, flat
FILLS.sort(key=lambda f: f["transaction_time"])                       # Alpaca keeps time order
POS = [{"symbol": "AAA", "qty": "5", "avg_entry_price": "100", "cost_basis": "500", "market_value": "550",
        "unrealized_pl": "50", "unrealized_plpc": "0.1", "current_price": "110", "change_today": "0.02", "lastday_price": "107.84"},
       {"symbol": "BBB", "qty": "3", "avg_entry_price": "50", "cost_basis": "150", "market_value": "135",
        "unrealized_pl": "-15", "unrealized_plpc": "-0.1", "current_price": "45", "change_today": "-0.01", "lastday_price": "45.4545"},
       {"symbol": "CCC", "qty": "3", "avg_entry_price": "20", "cost_basis": "60", "market_value": "66",
        "unrealized_pl": "6", "unrealized_plpc": "0.1", "current_price": "22", "change_today": "0", "lastday_price": "22"}]


DEPOSITS = [{"id": "d1", "activity_type": "CSD", "date": "2026-10-05", "created_at": "2026-10-05T21:15:38Z", "net_amount": "100",
             "status": "executed"}]


class FakeAccount(ap.PaperAccount):
    """No network: answers the allow-listed GET paths from memory and records every call."""
    calls, fail = [], None

    def __init__(self, *a, **k):
        pass

    def _get(self, path, params=None):
        assert path in ap.ALLOWED_PATHS, path
        FakeAccount.calls.append((path, dict(params or {})))
        if FakeAccount.fail:
            raise ap.PaperAccountError(FakeAccount.fail)
        if path == "/positions":
            return POS
        if path == "/orders":                                   # the tax view's order list (client ids)
            return []
        if path == "/account":
            return {"equity": "1000", "cash": "249", "buying_power": "249", "long_market_value": "751", "last_equity": "990"}
        if params.get("activity_types") != "FILL":                # deposits / withdrawals (cash_flows)
            return DEPOSITS
        newest_first = [f for f in FILLS[::-1] if f["transaction_time"] > params.get("after", "")]
        start = 0 if "page_token" not in params else [f["id"] for f in newest_first].index(params["page_token"]) + 1
        return newest_first[start:start + params["page_size"]]


acct = FakeAccount()
got = acct.fills()
check("fills: every page read (page_token), oldest first", len(got) == len(FILLS) and got[0]["transaction_time"] <= got[-1]["transaction_time"]
      and sum(1 for p, q in FakeAccount.calls if "page_token" in q) == 1, len(got))
first = ap.first_buy_dates(got)
check("First bought, FIFO after a partial sell: AAA = Sep 25 (the Sep 18 shares were sold first)", str(first.get("AAA")) == "2026-09-25", first)
check("First bought restarts after a full exit: BBB = Sep 25", str(first.get("BBB")) == "2026-09-25", first)
check("First bought keeps the oldest shares still held: CCC = Sep 18", str(first.get("CCC")) == "2026-09-18", first)
check("a closed position has no date (ZZZ)", "ZZZ" not in first, first)

t = ap.holdings_table(acct.position_dicts(), got, 1000.0)
check("columns", list(t.columns) == ap.HOLDING_COLS, list(t.columns))
check("rows: 3 stocks (largest first) + Total", list(t["Stock"]) == ["AAA", "BBB", "CCC", "Total (3 stocks)"], list(t["Stock"]))
a = t.iloc[0]
check("AAA from Alpaca's fields", (a["Shares"], a["Avg price"], a["Cost basis"], a["Market value"], a["P/L $"], a["Price"])
      == (5, 100, 500, 550, 50, 110) and math.isclose(a["P/L %"], 10) and math.isclose(a["Today %"], 2) and math.isclose(a["Weight %"], 55))
tot = t.iloc[3]
check("Total row: sums, P/L % on cost, weight of equity",
      (tot["Cost basis"], tot["Market value"], tot["P/L $"]) == (710, 751, 41) and math.isclose(tot["P/L %"], 41 / 710 * 100)
      and math.isclose(tot["Weight %"], 75.1), tot.to_dict())
prev = 5 * 107.84 + 3 * 45.4545 + 3 * 22
check("Total Today % = value now vs last close", math.isclose(tot["Today %"], (751 - prev) / prev * 100), tot["Today %"])
check("no positions: empty table", ap.holdings_table([], [], 1000.0).empty)

# ---------------------------------------------------------------- fill history kept between reads (request count stays flat)
from datetime import datetime  # noqa: E402
day1, day2 = (datetime(2026, 10, 6, h, tzinfo=ap.CT) for h in (10, 23))
h, FakeAccount.calls = ap.FillHistory(), []
first_read = h.update(acct, now=day1)
full_pages = len(FakeAccount.calls)
check("first read of the day pages the whole history", len(first_read) == len(FILLS) and full_pages == 2, full_pages)
FILLS.append(fill("DDD", "buy", 1, "2026-10-06"))
FakeAccount.calls = []
second = h.update(acct, now=day1)
check("later read: ONE request, only fills after the newest kept (1 min overlap), new fill added once",
      len(FakeAccount.calls) == 1 and "after" in FakeAccount.calls[0][1] and len(second) == len(FILLS)
      and sum(f["symbol"] == "DDD" for f in second) == 1, (FakeAccount.calls, len(second)))
check("kept history gives the same First bought dates as a full re-read", ap.first_buy_dates(second) == ap.first_buy_dates(acct.fills()))
FakeAccount.calls = []
check("nothing new: still one request, no duplicates", len(h.update(acct, now=day1)) == len(FILLS) and len(FakeAccount.calls) == 1)
FakeAccount.calls = []
h.update(acct, now=day2.replace(day=7))
check("next day: a full re-read (safety net)", len(FakeAccount.calls) == 2 and all("after" not in q for _, q in FakeAccount.calls))
FILLS.pop()

# ---------------------------------------------------------------- refresh: every minute in market hours, hourly otherwise
key = lambda s: ap.holdings_refresh_key(datetime.fromisoformat(s).replace(tzinfo=ap.CT))
check("market hours: a new key each minute (Tue 10:31:05 = 10:31:59, 10:32 differs)",
      key("2026-10-06 10:31:05") == key("2026-10-06 10:31:59") != key("2026-10-06 10:32:00"))
check("market hours start 8:30 AM CT and end 3:00 PM CT",
      key("2026-10-06 08:29") == key("2026-10-06 08:01") != key("2026-10-06 08:30") != key("2026-10-06 08:31")
      and key("2026-10-06 15:00") == key("2026-10-06 15:59") and key("2026-10-06 14:59") != key("2026-10-06 14:58"))
check("after hours / nights: one key per hour", key("2026-10-06 20:05") == key("2026-10-06 20:59") != key("2026-10-06 21:00"))
check("weekend: one key per hour, also at 10 AM", key("2026-10-03 10:01") == key("2026-10-03 10:45"))
check("holiday (Thanksgiving): hourly", key("2026-11-26 10:01") == key("2026-11-26 10:45"))
check("early close (day after Thanksgiving): per minute until 12:00 PM CT, hourly after",
      key("2026-11-27 11:58") != key("2026-11-27 11:59") and key("2026-11-27 12:01") == key("2026-11-27 12:40"))
app_src = open(os.path.join(ROOT, "dashboard", "details", "live_holdings.py")).read()   # the panel (app.py is the entry point)
check("the table reruns by itself (st.fragment run_every=60), not the whole page",
      "@st.fragment(run_every=60)" in app_src.split("def render_live_holdings")[0][-200:])

# ---------------------------------------------------------------- the dashboard renders it (fake client, no Alpaca)
import dashboard.details.live_holdings as lh  # noqa: E402
real, real_key = ap.PaperAccount, ap.holdings_refresh_key
real_lh = lh.read_report_csv, lh.etf_prices
DAILY = pd.DataFrame({"Date": ["2026-10-02", "2026-10-06"], "Time_CT": ["16:15", "16:15"], "Equity": [880.0, 990.0],
                      "Net_Deposits": [0.0, 100.0]})
lh.read_report_csv = lambda path: DAILY if path.endswith("forward_test_daily.csv") else real_lh[0](path)
lh.etf_prices = lambda symbols, start: (pd.DataFrame({s: [100.0, 105.0] for s in symbols},       # no yfinance call
                                                     index=pd.to_datetime(["2026-10-02", "2026-10-05"])), dict.fromkeys(symbols, 110.0))
ap.PaperAccount = FakeAccount
os.environ.pop("STOCK_ANALYSIS_LIVE_HOLDINGS", None)
reads = lambda: sum(1 for p, _ in FakeAccount.calls if p == "/positions")
try:
    FakeAccount.calls = []
    ap.holdings_refresh_key = lambda now=None: "k1"
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file("app.py", default_timeout=180).run()
    frames = [d.value for d in at.dataframe if "First bought" in d.value.columns]
    check("app: no exceptions, holdings table on the Details tab", not at.exception and len(frames) == 1, [str(e) for e in at.exception])
    check("app: 3 stocks + Total rows", len(frames) == 1 and len(frames[0]) == 4 and frames[0]["Stock"].iloc[-1] == "Total (3 stocks)")
    bm = [d.value for d in at.dataframe if "Compared with" in d.value.columns]
    check("app: the account vs QQQ / SPY / IWM / DIA table, each ETF +10% (110 vs the Oct 2 close 100)",
          len(bm) == 1 and list(bm[0]["Compared with"]) == ["Your account"] + [f"{s} ({n})" for s, n in ap.BENCHMARK_ETFS.items()]
          and all(math.isclose(r, 10) for r in bm[0]["Return %"].iloc[1:]), bm)
    check("app: the account's return nets out the $100 deposit (990 on 880 + 100 = +0%, then 1000 / 990)",
          len(bm) == 1 and math.isclose(bm[0]["Return %"].iloc[0], (990 / 980 * 1000 / 990 - 1) * 100), bm)
    check("app: the comparison caption explains the same deposits and time-weighting",
          any("each deposit or withdrawal bought or sold at the close of its date" in c.value and "time-weighted" in c.value
              for c in at.caption))
    check("app: as-of time and how-to-read note", any(c.value.startswith("As of ") and "every hour otherwise" in c.value for c in at.caption))
    check("app: only read-only GET paths used", FakeAccount.calls and all(p in ("/positions", "/account", "/account/activities", "/orders", "/clock")
                                                                          for p, _ in FakeAccount.calls), FakeAccount.calls)
    at.run()
    check("app: same refresh key -> cached, no new Alpaca reads", reads() == 1, reads())
    ap.holdings_refresh_key = lambda now=None: "k2"
    at.run()
    check("app: new refresh key (next minute / hour) -> one new read", reads() == 2, reads())
    acts = [q for p, q in FakeAccount.calls if p == "/account/activities" and q.get("activity_types") == "FILL"]
    check("app: the second read asks only for new fills (after=...)", "after" in acts[-1] and sum("after" in q for q in acts) == 1, acts)
    FakeAccount.fail = "GET /positions failed: HTTP 401 Unauthorized"
    ap.holdings_refresh_key = lambda now=None: "k3"
    at.run()
    msg = [i.value for i in at.info if "Live holdings unavailable" in i.value]
    check("app: Alpaca failure -> plain message, page still renders", not at.exception and len(msg) == 1, [str(e) for e in at.exception])
    FakeAccount.fail = None
    at.run()
    check("app: a failure is not cached (same key, next run reads again and shows the table)",
          reads() == 4 and any("First bought" in d.value.columns for d in at.dataframe), reads())
finally:
    ap.PaperAccount, ap.holdings_refresh_key, FakeAccount.fail = real, real_key, None
    lh.read_report_csv, lh.etf_prices = real_lh
    st.cache_data.clear()
    st.cache_resource.clear()

src = open(os.path.join(ROOT, "alpaca_paper.py")).read()
check("alpaca_paper.py stays read-only (GET only, no order calls)",
      'method="GET"' in src and not any(w in src for w in ('method="POST"', 'method="DELETE"', 'method="PATCH"', "submit_order", "cancel_order")))
print(f"\n{len(FAIL)} failed" if FAIL else "\nLIVE HOLDINGS OK")
sys.exit(1 if FAIL else 0)
