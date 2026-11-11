"""Record-keeping helpers (trade_records, sleeves.finalize_book skipped_out, save_ledger archive):
skip reasons, ledger snapshots, ops banner, plan-vs-fill recap, contribution vs QQQ.
No broker calls, no orders. Temp files only.
Run: python tests/run_tests.py  (or python tests/test_trade_records.py)
"""
import json
import os
import sys
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

TMP = tempfile.mkdtemp(prefix="sa_rec_")
os.environ["STOCK_ANALYSIS_DECISION_SKIPS"] = os.path.join(TMP, "decision_skips.csv")
os.environ["STOCK_ANALYSIS_LEDGER_HISTORY"] = os.path.join(TMP, "ledger_history")
os.environ["STOCK_ANALYSIS_FILL_RECAP"] = os.path.join(TMP, "fill_recap.csv")
os.environ["STOCK_ANALYSIS_FILL_RECAP_SUMMARY"] = os.path.join(TMP, "fill_recap_summary.csv")
os.environ["STOCK_ANALYSIS_SLEEVE_LEDGER"] = os.path.join(TMP, "sleeve_ledger.json")
os.environ["STOCK_ANALYSIS_SLEEVE_PLAN"] = os.path.join(TMP, "sleeve_plan.csv")
os.environ["STOCK_ANALYSIS_RUN_LOG"] = os.path.join(TMP, "run_log.csv")
os.environ["STOCK_ANALYSIS_PENDING_ORDERS"] = os.path.join(TMP, "pending.json")
os.environ["STOCK_ANALYSIS_LIVE_HOLDINGS"] = "off"

import pandas as pd  # noqa: E402
import sleeves  # noqa: E402
import trade_records as tr  # noqa: E402
import sector_mapping as sm  # noqa: E402

# Re-bind module paths in case they were imported before env was set (they weren't).
tr.SKIP_CSV = os.environ["STOCK_ANALYSIS_DECISION_SKIPS"]
tr.LEDGER_HISTORY = os.environ["STOCK_ANALYSIS_LEDGER_HISTORY"]
tr.RECAP_CSV = os.environ["STOCK_ANALYSIS_FILL_RECAP"]
tr.RECAP_SUM_CSV = os.environ["STOCK_ANALYSIS_FILL_RECAP_SUMMARY"]
tr.RUN_LOG = os.environ["STOCK_ANALYSIS_RUN_LOG"]
tr.PENDING_JSON = os.environ["STOCK_ANALYSIS_PENDING_ORDERS"]
tr.PLAN_CSV = os.environ["STOCK_ANALYSIS_SLEEVE_PLAN"]
tr.LEDGER_JSON = os.environ["STOCK_ANALYSIS_SLEEVE_LEDGER"]
sleeves.LEDGER_JSON = os.environ["STOCK_ANALYSIS_SLEEVE_LEDGER"]
sleeves.LEDGER_HISTORY = os.environ["STOCK_ANALYSIS_LEDGER_HISTORY"]
sleeves.SKIP_CSV = os.environ["STOCK_ANALYSIS_DECISION_SKIPS"]

FAIL = []
CT = ZoneInfo("America/Chicago")


def check(ok, what, got=""):
    print(("PASS " if ok else "FAIL ") + what + ("" if ok else f"  ({got})"))
    if not ok:
        FAIL.append(what)


# --- finalize_book skipped_out does not change the book ---
hc = ["HC1", "HC2", "HC3", "HC4", "HC5"]
oth = ["O1", "O2", "O3", "O4", "O5", "O6"]
all_n = hc + oth
rank_of = {s: i + 1 for i, s in enumerate(all_n)}
orig = dict(sm.symbol_sector)
try:
    sm.symbol_sector.update({s: "Health Care" for s in hc})
    sm.symbol_sector.update({s: f"Sec{s}" for s in oth})
    skipped = []
    book = sleeves.finalize_book(
        names=hc[:4] + oth[:5], held=set(), rank_of=rank_of, blocked={}, stretched={"O1"},
        soon=set(), priced=set(all_n), n_picks=10, invested=0.99, max_rank=20, sector_cap=0.4,
        skipped_out=skipped)
    book2 = sleeves.finalize_book(
        names=hc[:4] + oth[:5], held=set(), rank_of=rank_of, blocked={}, stretched={"O1"},
        soon=set(), priced=set(all_n), n_picks=10, invested=0.99, max_rank=20, sector_cap=0.4)
    check(book == book2, "skipped_out does not change the book", (book, book2))
    reasons = {r["Symbol"]: r["Reason"] for r in skipped}
    check("HC5" in reasons and "sector cap" in reasons["HC5"],
          "HC5 skip reason is the sector cap", reasons)
    check("O1" in reasons and reasons["O1"] == "stretched",
          "O1 skip reason is stretched", reasons)
    check(all(r["Rank"] == rank_of[r["Symbol"]] for r in skipped), "skip rows include ranks")
finally:
    sm.symbol_sector.clear()
    sm.symbol_sector.update(orig)

# --- write_skips is dated and replace-in-place for one day ---
tr.write_skips("2026-11-06", [{"Symbol": "APP", "Rank": 9, "Reason": "stretched"},
                               {"Symbol": "XYZ", "Rank": 12, "Reason": "earnings within 5 days"}])
tr.write_skips("2026-10-30", [{"Symbol": "OLD", "Rank": 8, "Reason": "stretched"}])
tr.write_skips("2026-11-06", [{"Symbol": "APP", "Rank": 9, "Reason": "stretched"}])  # replace that day
nov = tr.load_skips("2026-11-06")
all_s = tr.load_skips()
check(list(nov["Symbol"]) == ["APP"] and nov.iloc[0]["Reason"] == "stretched",
      "write_skips replaces one day only", nov.to_dict("records"))
check(set(all_s["Date"].astype(str).str[:10]) == {"2026-11-06", "2026-10-30"} and "OLD" in set(all_s["Symbol"]),
      "other days are kept", all_s.to_dict("records"))

# --- ledger archive only on a same-day rebalance ---
today = datetime.now(CT).date().isoformat()
led = {"version": 4, "day": today,
       "halves": {"rs_short": {"shares": {"EL": 1}, "cash": 10, "rebalanced": True, "value": 100}}}
sleeves.save_ledger(led, skipped=[{"Symbol": "APP", "Rank": 9, "Reason": "stretched"}])
arch = os.path.join(tr.LEDGER_HISTORY, f"sleeve_ledger_{today}.json")
check(os.path.exists(arch), "same-day rebalance ledger is archived", arch)
with open(arch) as f:
    copied = json.load(f)
check(copied["halves"]["rs_short"]["shares"] == {"EL": 1}, "archive matches the ledger")
sk = tr.load_skips(today)
check(len(sk) == 1 and sk.iloc[0]["Symbol"] == "APP", "save_ledger writes skip rows for that day", sk.to_dict("records"))

# A later-dated ledger (manual edit on a later day) must not overwrite the Friday archive
old_day = "2026-10-16"
led2 = {"version": 4, "day": old_day,
        "halves": {"rs_short": {"shares": {"KO": 1}, "cash": 0, "rebalanced": True, "value": 50}}}
sleeves.save_ledger(led2)
check(not os.path.exists(os.path.join(tr.LEDGER_HISTORY, f"sleeve_ledger_{old_day}.json")),
      "a ledger whose day is not today is not archived")
with open(arch) as f:
    still = json.load(f)
check(still["halves"]["rs_short"]["shares"] == {"EL": 1}, "later save_ledger does not overwrite today's archive")

# A failing recorder must not break save_ledger
real = tr.after_ledger_save
tr.after_ledger_save = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
try:
    path = sleeves.save_ledger({"version": 4, "day": today,
                                "halves": {"rs_short": {"shares": {"MA": 2}, "cash": 1, "rebalanced": True, "value": 80}}})
    check(os.path.exists(path) and json.load(open(path))["halves"]["rs_short"]["shares"] == {"MA": 2},
          "save_ledger still writes the ledger when record-keeping fails")
finally:
    tr.after_ledger_save = real

# --- ops banner: clean when empty; hides resolved failures; leftovers still show ---
check(tr.ops_banner_lines(now="2026-11-11 12:00:00") == [], "banner is empty when the log and pending file are empty")
# write a fake run log as a plain CSV (temp path is outside the project, so db keeps it as a file)
pd.DataFrame([
    {"time_ct": "2026-10-30 14:30:15", "run": "Fri rebalance", "status": "failed", "money_moved": "no",
     "message": "The Fri Oct 30 trade was not placed.", "details": ""},
    {"time_ct": "2026-10-30 14:32:15", "run": "Watchdog", "status": "failed", "money_moved": "no",
     "message": "Step main failed.", "details": ""},
    {"time_ct": "2026-11-06 14:30:26", "run": "Fri rebalance", "status": "ok", "money_moved": "yes",
     "message": "Fri rebalance done 2:30 PM.", "details": ""},
    {"time_ct": "2026-11-10 09:00:00", "run": "Fill check", "status": "failed", "money_moved": "no",
     "message": "Not traded: FDS 41.", "details": ""},
    {"time_ct": "2026-11-10 09:30:00", "run": "Trade audit", "status": "failed", "money_moved": "no",
     "message": "API outage", "details": ""},
    {"time_ct": "2026-11-10 15:00:00", "run": "Fill check", "status": "ok", "money_moved": "yes",
     "message": "Filled.", "details": ""},
    {"time_ct": "2026-10-20 14:30:00", "run": "Fri rebalance", "status": "failed", "money_moved": "no",
     "message": "too old", "details": ""},
    {"time_ct": "2026-11-11 09:00:00", "run": "Fill check", "status": "failed", "money_moved": "no",
     "message": "Not traded: XYZ 10.", "details": ""},
]).to_csv(tr.RUN_LOG, index=False)
lines = tr.ops_banner_lines(days=14, now="2026-11-11 12:00:00")
text = "\n".join(lines)
check(all("Oct 30" not in x and "Watchdog" not in x for x in lines),
      "banner hides Oct 30 Fri rebalance / Watchdog failures after a later successful Friday", lines)
check(all("FDS" not in x for x in lines),
      "banner hides a Fill check failure once a later Fill check succeeds", lines)
check(any("Fill check failed" in x and "XYZ 10" in x for x in lines),
      "banner still shows an unresolved recent Fill check failure", lines)
check("Trade audit" not in text, "banner ignores trade-audit failures", lines)
check("too old" not in text, "banner ignores failures older than 14 days", lines)
# No leftover pending file yet: only the unresolved fill check should show
check(len(lines) == 1, "banner has only the unresolved Fill check when there are no leftovers", lines)
with open(tr.PENDING_JSON, "w") as f:
    json.dump({"evening_date": "2026-11-06", "orders": [{"symbol": "V", "side": "SELL", "qty": 42.04}]}, f)
lines2 = tr.ops_banner_lines(days=14, now="2026-11-11 12:00:00")
check(any("Pending leftover" in x and "SELL V" in x for x in lines2),
      "banner mentions leftover pending orders", lines2)
check(any("XYZ 10" in x for x in lines2),
      "unresolved failure still shows alongside leftover orders", lines2)

# --- plan-vs-fill recap from fixtures (offline) ---
plan_path = os.path.join(TMP, "plan.csv")
log_path = os.path.join(TMP, "orders.csv")
led_path = os.path.join(TMP, "ledger_asof.json")
recap_path = os.path.join(TMP, "recap_out.csv")
sum_path = os.path.join(TMP, "recap_sum.csv")
pd.DataFrame([
    {"Half": "live", "Symbol": "EL", "Held_Shares": 0, "Target_Shares": 160.11, "Price": 110.35,
     "Target_Value": 17667, "Target_in_half_%": 9.9, "Target_of_account_%": 9.9},
    {"Half": "live", "Symbol": "V", "Held_Shares": 42.37, "Target_Shares": 42.04, "Price": 422.36,
     "Target_Value": 17756, "Target_in_half_%": 9.95, "Target_of_account_%": 9.95},
    {"Half": "live", "Symbol": "FDS", "Held_Shares": 0, "Target_Shares": 41, "Price": 280.0,
     "Target_Value": 11480, "Target_in_half_%": 9.9, "Target_of_account_%": 9.9},
]).to_csv(plan_path, index=False)
pd.DataFrame([
    {"Symbol": "EL", "Side": "BUY", "Shares": 160.11, "Status": "COMPLETED via limit $110.46",
     "As_Of": "2026-11-06", "Fill_Price": 110.39, "Limit": 110.46, "Cost_$": 7.53},
    {"Symbol": "V", "Side": "SELL", "Shares": 0.33, "Status": "COMPLETED via limit $422.09",
     "As_Of": "2026-11-06", "Fill_Price": 422.33, "Limit": 422.09, "Cost_$": 0.01},
    {"Symbol": "FDS", "Side": "BUY", "Shares": 41, "Status": "FAILED (bad quote)",
     "As_Of": "2026-11-06", "Fill_Price": None, "Limit": 280.0, "Cost_$": 0.0},
]).to_csv(log_path, index=False)
json.dump({"version": 4, "day": "2026-11-06",
           "halves": {"rs_short": {"shares": {"EL": 160.11, "V": 42.04}, "cash": 1788.92, "rebalanced": True}}},
          open(led_path, "w"))
pend = os.path.join(TMP, "pend_fds.json")
json.dump({"evening_date": "2026-11-06", "as_of": "2026-11-06",
           "orders": [{"symbol": "FDS", "side": "BUY", "qty": 41}]}, open(pend, "w"))
# point ledger history so In_Book uses this ledger
os.makedirs(tr.LEDGER_HISTORY, exist_ok=True)
json.dump(json.load(open(led_path)), open(os.path.join(tr.LEDGER_HISTORY, "sleeve_ledger_2026-11-06.json"), "w"))

detail, summary = tr.build_fill_recap(
    as_of="2026-11-06", plan_path=plan_path, log_path=log_path, ledger_path=led_path, pending_path=pend)
by = detail.set_index("Symbol")
check(abs(by.at["EL", "Filled_Shares"] - 160.11) < 1e-6 and abs(by.at["EL", "Slippage_$"] - 7.53) < 1e-9,
      "recap: EL is filled with its slippage", by.loc["EL"].to_dict() if "EL" in by.index else detail)
check(by.at["FDS", "Leftover"] == 41 and by.at["FDS", "Filled_Shares"] == 0,
      "recap: FDS leftover is 41 unfilled shares", by.loc["FDS"].to_dict() if "FDS" in by.index else detail)
check(summary["Leftover_N"] == 1 and "FDS" in summary["Leftover"] and abs(summary["Slippage_$"] - 7.54) < 1e-9,
      "recap summary counts leftover and sums slippage", summary)
check(int(by.at["EL", "In_Book"]) == 1, "recap: EL is in the resulting book")

# write + load round trip, and a second day does not wipe the first
tr.write_fill_recap(as_of="2026-11-06", plan_path=plan_path, log_path=log_path, ledger_path=led_path,
                    pending_path=pend, recap_path=recap_path, summary_path=sum_path)
# a write that raises inside the helper must be catchable and not be the default
d2, s2 = tr.load_fill_recap(as_of="2026-11-06", path=recap_path, summary_path=sum_path)
check(len(d2) == len(detail) and s2.get("As_Of") == "2026-11-06",
      "write_fill_recap / load_fill_recap round trip", (len(d2), s2))

# Recap builder itself never raises on empty inputs
empty_d, empty_s = tr.build_fill_recap(as_of="2020-01-01", plan_path=os.path.join(TMP, "no_plan.csv"),
                                       log_path=os.path.join(TMP, "no_log.csv"),
                                       ledger_path=os.path.join(TMP, "no_led.json"),
                                       pending_path=os.path.join(TMP, "no_pend.json"))
check(isinstance(empty_d, pd.DataFrame) and empty_s.get("As_Of") == "2020-01-01",
      "recap on missing files still returns a frame and a dated summary", empty_s)

# --- contribution vs QQQ (synthetic prices) ---
idx = pd.to_datetime(["2026-10-29", "2026-10-30", "2026-11-06", "2026-11-10"])
prices = pd.DataFrame({
    "AAA": [100.0, 100.0, 100.0, 110.0],   # +10% since week start (Fri Nov 6) and +10% since Oct 29 from 100
    "BBB": [50.0, 50.0, 80.0, 80.0],       # 0% this week, +60% since Oct 29
}, index=idx)
qqq = pd.Series([200.0, 200.0, 200.0, 210.0], index=idx)  # +5% this week and since Oct 29 (200->210 from Oct 29 too)
# week_start on Wed Nov 11 / Tue Nov 10 = Fri Nov 6
positions = [{"symbol": "AAA", "qty": 10, "current_price": 110.0, "sector": "Tech"},
             {"symbol": "BBB", "qty": 5, "current_price": 80.0, "sector": "Health"}]
# first_bought BBB on Nov 6 so since-Oct-29 start is Nov 6 (80), not Oct 29 (50)
hold, sec = tr.contribution_table(positions, prices, qqq, as_of="2026-11-10", since="2026-10-29",
                                  first_bought={"BBB": "2026-11-06"})
h = hold.set_index("Symbol")
check(abs(h.at["AAA", "This week $"] - 100.0) < 1e-6 and abs(h.at["AAA", "This week %"] - 10.0) < 1e-6,
      "AAA this week is +$100 / +10%", h.loc["AAA"].to_dict())
check(abs(h.at["AAA", "vs QQQ this week pp"] - 5.0) < 1e-6,
      "AAA vs QQQ this week is +5 pp (10% - 5%)", h.loc["AAA"].to_dict())
check(abs(h.at["BBB", "This week $"] - 0.0) < 1e-6 and abs(h.at["BBB", "Since Oct 29 $"] - 0.0) < 1e-6,
      "BBB bought at the Friday close: $0 this week and $0 since first close after purchase",
      h.loc["BBB"].to_dict() if "BBB" in h.index else hold)
check(set(sec["Sector"]) == {"Tech", "Health"}, "sector table has both sectors", sec.to_dict("records"))
check("QQQ" in set(h.index) and "Portfolio (current book)" in set(h.index), "summary rows for Portfolio and QQQ")

# week_start: a Friday as_of still uses the previous Friday
ws = tr.week_start("2026-11-13")
check(str(pd.Timestamp(ws).date()) == "2026-11-06", "week_start on Friday Nov 13 is the prior Friday Nov 6", ws)
ws2 = tr.week_start("2026-11-11")
check(str(pd.Timestamp(ws2).date()) == "2026-11-06", "week_start on Wednesday Nov 11 is Friday Nov 6", ws2)

# --- dashboard helpers import without running the app ---
from dashboard.page import render_ops_banner  # noqa: E402
from dashboard.details.last_decision import _render_skips_and_recap  # noqa: E402
check(callable(render_ops_banner) and callable(_render_skips_and_recap),
      "dashboard banner and last-decision recap helpers import")

print("TRADE RECORDS OK" if not FAIL else f"TRADE RECORDS FAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
