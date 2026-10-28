"""Regression tests for the Oct 2026 logic-audit fixes: naive news timestamps, the support-window text, watchdog
classification (HTTP 5xx, SQLite missing-table), rest-order audit ids, sleeve reconcile, fresh-DB autofetch, the
headerless-append guard, trade-lock ownership, the shared float helpers, and the holdings cache copy.
Run: python tests/run_tests.py  (or python tests/test_bugfix_audit.py)"""
import io
import json
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["STOCK_ANALYSIS_DB"] = os.path.join(tempfile.mkdtemp(), "audit.db")
os.environ.setdefault("ALPHAVANTAGE_API_KEY", "test-dummy-key")
import pandas as pd

import db
import run_all as ra
import pipeline_watchdog as pw
import trade_audit as ta
import paper_trade as pt
import sleeves
import holdings
from tax_lots import _num as tax_num
import company_report_autofetch as cra

FAIL = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        FAIL.append(name)

CT = ZoneInfo("America/Chicago")

# 1. news_allowed accepts a naive last_news_at (assumed CT) instead of raising TypeError.
try:
    allowed, _ = ra.news_allowed(datetime(2026, 9, 28, 20, 0, tzinfo=CT), {"last_news_at": "2026-09-28T15:41:00"})
    naive_ok = allowed is False
except TypeError:
    naive_ok = False
check("news_allowed: naive stamp = 4 h gap, refused", naive_ok)
try:
    invalid_ok = ra.news_allowed(datetime(2026, 9, 28, 20, 0, tzinfo=CT),
                                 {"last_news_at": "not-a-date"})[0] is True
except (TypeError, ValueError):
    invalid_ok = False
check("news_allowed: invalid stamp allows (fail open on corrupt state)", invalid_ok)

# 2. Support-window text renders noon as 12:30, not 0:30.
msg = ra.support_idle(datetime(2026, 10, 27, 8, 0, tzinfo=CT), real=False)
check("support_idle: '12:30-2:00 PM CT' window text", isinstance(msg, str) and "12:30-2:00 PM CT" in msg)

# 3. Watchdog classification: HTTP 5xx (any case) is transient; SQLite's missing-table error resumes upstream.
check("watchdog: 'HTTP 500' is transient", pw.classify_failure("boom: HTTP 500")[0] == "transient")
check("watchdog: 'http 503' is transient", pw.classify_failure("http 503 Service Unavailable")[0] == "transient")
check("watchdog: db missing-table resumes the writer",
      pw.classify_failure("FileNotFoundError: no table 'signal_analysis' in /x/stock_analysis.db")
      == ("missing_upstream", "main"))

# 4. Rest-order ids (live-rest<parent>-...) audit as rest completions with a plan price.
rest = "live-rest1a2b3c-20261002-BUY-AMD-1-63391"
check("audit: rest id source", ta.order_source(rest) == "bot: rest that did not fit")
check("audit: rest id plan price", ta.plan_price(rest) == 633.91)
check("audit: legacy rest id still works",
      ta.order_source("live-rest-20261002-BUY-AMD-1-63391") == "bot: rest that did not fit"
      and ta.plan_price("live-rest-20261002-BUY-AMD-1-63391") == 633.91)

# 5. Sleeve runs have nothing to reconcile against the picks file (no spurious warning).
rep, ok = pt.reconcile_positions("sleeve")
check("reconcile: 'sleeve' source returns empty/True", rep.empty and ok is True)

# 6. A headerless write to a NEW table raises instead of eating the first row as the header.
try:
    db.save_table("zz_audit_hdr", pd.DataFrame([["a", "b"], ["c", "d"]]), header=False)
    hdr_ok = False
except ValueError:
    hdr_ok = True
check("db: header=False on a new table raises ValueError", hdr_ok)

# 7. Releasing a lock owned by another process leaves it alone; your own is removed.
other = os.path.join(tempfile.mkdtemp(), ".l1")
json.dump({"pid": 1, "at": datetime.now(CT).isoformat()}, open(other, "w"))
ra.release_trade_lock(other)
check("trade lock: another owner's lock is kept", os.path.exists(other))
mine = os.path.join(tempfile.mkdtemp(), ".l2")
json.dump({"pid": os.getpid(), "at": datetime.now(CT).isoformat()}, open(mine, "w"))
ra.release_trade_lock(mine)
check("trade lock: your own lock is released", not os.path.exists(mine))

# 8. One float parser (tax_lots._num), thin adapters elsewhere.
import math
check("helpers: tax_lots._num", tax_num("3.5") == 3.5 and math.isnan(tax_num("x"))
      and math.isnan(tax_num(float("inf"))) and math.isnan(tax_num(None)))
check("helpers: sleeves._num = positive-or-None",
      sleeves._num("3.5") == 3.5 and sleeves._num("-1") is None and sleeves._num("x") is None
      and sleeves._num(float("inf")) is None)
check("helpers: holdings._f = float-or-None", holdings._f("3.5") == 3.5 and holdings._f("x") is None
      and holdings._f(float("inf")) is None)
check("helpers: paper_trade._safe_number default", pt._safe_number("3.5") == 3.5 and pt._safe_number("x") == 0.0
      and pt._safe_number(float("nan"), default=9.0) == 9.0)

# 9. Caffeinate is macOS-only and never raises (the watchdog must survive Linux).
keep = getattr(pw, "_keep_awake", None)
with mock.patch.object(pw.sys, "platform", "linux"):
    linux_ok = keep is not None and keep() is None
with mock.patch.object(pw.os, "spawnlp", side_effect=FileNotFoundError):
    with mock.patch.object(pw.sys, "platform", "darwin"):
        mac_missing_ok = keep is not None and keep() is None
check("watchdog: no caffeinate off macOS, missing binary tolerated", linux_ok and mac_missing_ok)

# 10. The LLM prompt describes the chronological tail honestly.
with mock.patch.object(pw, "_llm_provider", return_value=("groq", "m", "k")):
    with mock.patch.object(pw, "_llm_complete", return_value="{}") as comp:
        pw.diagnose_with_llm("sentiment", "steps", "line1\nline2", {})
check("watchdog: diagnose prompt says chronological",
      comp.called and "chronological" in comp.call_args[0][3])

# 11. Fundamentals rows with unparseable dates are dropped loudly, not silently.
inc = pd.DataFrame({"Symbol": ["AAA", "BBB"], "FiscalDateEnding": ["2026-01-01", "not-a-date"],
                    **{c: [1, 2] for c in cra.INCOME_FIELDS}})
bal = pd.DataFrame({"Symbol": ["AAA", "BBB"], "FiscalDateEnding": ["2026-01-01", "not-a-date"],
                    **{c: [3, 4] for c in cra.BALANCE_FIELDS}})
buf = io.StringIO()
with redirect_stdout(buf):
    out = cra.build_fundamentals(inc, bal)
check("autofetch: bad dates dropped with a warning",
      len(out) == 1 and "FiscalDateEnding" in buf.getvalue())

# 12. First-ever fundamentals save works with no table and no file yet.
saved_csv, tmp_csv = cra.BALANCE_CSV, os.path.join(ROOT, "Reports", "zz_audit_fresh.csv")
cra.BALANCE_CSV = tmp_csv
try:
    new = pd.DataFrame([["AAA", "2026-01-01"] + [1.0] * (len(cra.OUTPUT_COLUMNS) - 2)], columns=cra.OUTPUT_COLUMNS)
    cra.save_fundamentals(new)
    fresh_ok = len(db.load_table("zz_audit_fresh")) == 1
except FileNotFoundError:
    fresh_ok = False
finally:
    cra.BALANCE_CSV = saved_csv
    db.remove(tmp_csv)
check("autofetch: save works on a fresh DB", fresh_ok)

# 13. Borrowed holding rows are copies: mutating them cannot poison the 60 s cache.
saved_cache = dict(holdings._cache)
holdings._cache["rows"] = (time.time(), [{"symbol": "AAA", "qty": "5"}])
r1 = holdings.position_rows()
r1[0]["qty"] = "MUTATED"
r2 = holdings.position_rows()
holdings._cache.clear()
holdings._cache.update(saved_cache)
check("holdings: position_rows returns copies", r2[0]["qty"] == "5")

print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
sys.exit(1 if FAIL else 0)
