"""run_all.py decision logic (no steps run, no API calls): full vs quick by the clock, holidays, once-a-day full update,
NewsAPI 24 h guard, Alpha Vantage once-a-day guard, --dry-run plan. Run: python tests/run_tests.py"""
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_all as r

CT = r.CT
FAIL = []


def t(s):
    return datetime.fromisoformat(s).replace(tzinfo=CT)


def expect(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAIL.append(what)


cases = [("2026-09-28 15:40", {}, "full"),        # Monday after 3:15 PM CT
         ("2026-09-28 15:10", {}, "quick"),       # Monday before 3:15 PM CT
         ("2026-09-29 16:00", {}, "quick"),       # Tuesday
         ("2026-09-30 17:00", {}, "full"),        # Wednesday evening
         ("2026-10-02 15:15", {}, "full"),        # Friday exactly 3:15 PM CT
         ("2026-09-26 16:00", {}, "quick"),       # Saturday
         ("2026-11-26 16:00", {}, "quick"),       # Thanksgiving (Thursday anyway)
         ("2026-12-25 16:00", {}, "quick"),       # Christmas Friday = holiday
         ("2026-09-07 16:00", {}, "quick"),       # Labor Day Monday = holiday
         ("2026-12-24 16:00", {}, "full"),        # Thursday = the week's rebalance (Friday Christmas is a holiday)
         ("2026-07-02 16:00", {}, "full"),        # Thursday = rebalance (Fri Jul 3 2026 is the Independence Day holiday)
         ("2026-09-08 16:00", {}, "full"),        # Tuesday = the Monday check moved by the Labor Day holiday
         ("2026-10-01 16:00", {}, "quick"),       # a normal Thursday: no decision
         ("2026-09-28 18:00", {"last_full_date": "2026-09-28"}, "quick"),   # full update already done today
         ("2026-09-30 15:30", {"last_full_date": "2026-09-28"}, "full")]
for when, state, want in cases:
    mode, why = r.choose_mode(t(when), state)
    expect(mode == want, f"{when} {state or ''} -> {mode} ({why})")
expect(r.choose_mode(t("2026-09-29 10:00"), {}, force_full=True)[0] == "full", "--full forces full")
expect(r.choose_mode(t("2026-09-28 16:00"), {}, force_quick=True)[0] == "quick", "--quick forces quick")

# decision days follow the strategy calendar (backtest_engine.next_decision), not fixed weekdays
for d, want in [("2026-10-02", "full rebalance"), ("2026-09-30", "mid-week check"), ("2026-09-28", "mid-week check"),
                ("2026-09-29", None), ("2026-12-24", "full rebalance"), ("2026-12-25", None), ("2026-09-08", "mid-week check"),
                ("2026-09-07", None)]:
    got = r.decision_day(datetime.fromisoformat(d).date())
    expect(got == want, f"decision_day {d} -> {got}")
nxt = r.next_scheduled_trade(t("2026-12-23 16:00"))
expect(nxt is not None and nxt.date().isoformat() == "2026-12-24", f"next trade after Wed Dec 23 = Thu Dec 24 ({nxt})")
expect(r.already_attempted(t("2026-10-02 16:30"), {"last_trade_attempt_at": "2026-10-02T15:15:07-05:00"}),
       "a scheduled run skips a window already attempted today")
expect(not r.already_attempted(t("2026-10-02 16:30"), {"last_trade_attempt_at": "2026-09-30T15:15:07-05:00"}),
       "a scheduled run proceeds when the last attempt was an earlier day")
expect(not r.already_attempted(t("2026-10-02 16:30"), {}), "a scheduled run proceeds with no attempt on record")
att = {"last_trade_attempt_at": "2026-10-02T15:15:07-05:00"}
expect(r.trade_gate(t("2026-10-02 16:30"), {})[0], "trade gate: Friday after 3:15 PM with nothing on record -> trade")
expect(not r.trade_gate(t("2026-10-02 14:00"), {})[0], "trade gate: before 3:15 PM -> nothing (login at 2 PM does not trade)")
expect(not r.trade_gate(t("2026-10-01 16:30"), {})[0], "trade gate: Thursday (no decision) -> nothing")
expect(not r.trade_gate(t("2026-10-02 16:30"), att, scheduled=True)[0], "trade gate: launchd re-run after an attempt -> nothing")
expect(r.trade_gate(t("2026-10-02 16:30"), att, scheduled=False)[0], "trade gate: a manual/resumed run is not blocked by the attempt")
expect(not r.trade_gate(t("2026-10-02 18:00"), {"last_trade_at": "2026-10-02T15:40:00-05:00"})[0],
       "trade gate: already traded this window -> nothing")

st = {"last_news_at": "2026-09-28T15:41:00-05:00"}
expect(not r.news_allowed(t("2026-09-28 20:00"), st)[0], "NewsAPI refused 4 h after the last run")
expect(not r.news_allowed(t("2026-09-29 15:00"), st)[0], "NewsAPI refused 23 h after the last run (rolling 24 h)")
expect(r.news_allowed(t("2026-09-30 15:40"), st)[0], "NewsAPI allowed 2 days later")
expect(r.news_allowed(t("2026-09-28 20:00"), st, force_news=True)[0], "--force-news overrides")
expect(r.news_allowed(t("2026-09-28 20:00"), {})[0], "NewsAPI allowed with no history")
expect(not r.fundamentals_allowed(t("2026-09-28 20:00"), {"last_fundamentals_date": "2026-09-28"})[0], "Alpha Vantage once a day")
expect(r.fundamentals_allowed(t("2026-09-30 16:00"), {"last_fundamentals_date": "2026-09-28"})[0], "Alpha Vantage next check day")

env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
out = subprocess.run([sys.executable, "run_all.py", "--dry-run", "--full", "--now", "2026-09-28 15:40"], cwd=ROOT, env=env,
                     capture_output=True, text=True).stdout
expect("mode FULL" in out and "sentiment" in out and "DRY RUN" in out, "--dry-run --full prints the full plan")
out = subprocess.run([sys.executable, "run_all.py", "--dry-run", "--now", "2026-09-29 11:00"], cwd=ROOT, env=env,
                     capture_output=True, text=True).stdout
expect("mode QUICK" in out and "sentiment" not in out and "main" in out, "--dry-run on a Tuesday = quick plan (no quota steps)")
print("\nRUNNER TESTS OK" if not FAIL else f"\nRUNNER TEST FAILURES: {FAIL}")
sys.exit(1 if FAIL else 0)
