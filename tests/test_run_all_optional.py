"""Optional-step resilience: an upstream API failure must not stop main or block trading.

Covers: OPTIONAL_STEPS covers exactly the five upstream API steps (main/validate stay
critical), _stop_on_failure (pipeline keeps going past optional failures, still stops on
main/validate without --keep-going), _critical_failures (only critical failures block the
evening trade / set a nonzero exit).
Pure logic - no notebooks run, no broker calls, no quota APIs.
Run: PYTHONPATH=. python tests/test_run_all_optional.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import run_all as r

FAIL = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        FAIL.append(what)


def test_optional_steps_cover_the_upstream_api_steps():
    check(r.OPTIONAL_STEPS == {"fundamentals", "processing", "scoring", "sentiment", "earnings"},
          f"OPTIONAL_STEPS is exactly the five upstream API steps (got {sorted(r.OPTIONAL_STEPS)})")


def test_main_and_validate_are_critical():
    check("main" not in r.OPTIONAL_STEPS and "validate" not in r.OPTIONAL_STEPS,
          "main and validate are NOT optional")


def test_every_pipeline_step_is_classified():
    step_names = {s[0] for s in r.STEPS}
    known = set(r.OPTIONAL_STEPS) | {"main", "validate", "visualization", "backtest"}
    check(step_names <= known,
          f"every STEPS entry is optional, critical, or opt-in (unclassified: {sorted(step_names - known)})")


def test_stop_on_failure():
    check(r._stop_on_failure("sentiment", False) is False,
          "sentiment failure never stops the pipeline")
    check(r._stop_on_failure("fundamentals", False) is False,
          "fundamentals failure never stops the pipeline")
    check(r._stop_on_failure("earnings", False) is False,
          "earnings failure never stops the pipeline")
    check(r._stop_on_failure("main", False) is True,
          "main failure stops the pipeline without --keep-going")
    check(r._stop_on_failure("validate", False) is True,
          "validate failure stops the pipeline without --keep-going")
    check(r._stop_on_failure("main", True) is False,
          "--keep-going keeps going even past main")


def test_critical_failures():
    check(r._critical_failures([]) == [],
          "no failures -> nothing critical")
    check(r._critical_failures(["sentiment"]) == [],
          "sentiment-only failure is not critical (trade may proceed)")
    check(r._critical_failures(["fundamentals", "processing", "scoring", "sentiment", "earnings"]) == [],
          "all-optional failures are not critical")
    check(r._critical_failures(["sentiment", "main"]) == ["main"],
          "main failure stays critical alongside optional ones")
    check(r._critical_failures(["sync_paper"]) == [],
          "sync_paper (read-only refresh) is not critical")
    check(r._critical_failures(["trade"]) == ["trade"],
          "a crashed trade stays critical")
    check(r._critical_failures(["trade_partial"]) == ["trade_partial"],
          "partial trade failures stay critical")


def test_trade_decision():
    # Optional-only failures must NOT block the evening trade (the trade guard
    # used to log "proceeding" but silently skip auto_trade - _trade_decision
    # pins the fixed behavior).
    proceed, reason = r._trade_decision([])
    check(proceed is True and reason == "",
          "no failures -> trade proceeds")
    proceed, reason = r._trade_decision(["sentiment"])
    check(proceed is True and "non-critical" in reason,
          "sentiment-only failure -> trade proceeds (not silently skipped)")
    proceed, _ = r._trade_decision(["fundamentals", "processing", "scoring", "sentiment", "earnings"])
    check(proceed is True,
          "all-optional failures -> trade still proceeds")
    proceed, _ = r._trade_decision(["sync_paper"])
    check(proceed is True,
          "sync_paper failure -> trade proceeds")
    proceed, reason = r._trade_decision(["main"])
    check(proceed is False and "critical" in reason,
          "main failure -> trade blocked")
    proceed, _ = r._trade_decision(["sentiment", "main"])
    check(proceed is False,
          "main failure stays blocking alongside optional ones")
    proceed, _ = r._trade_decision(["validate"])
    check(proceed is False,
          "validate failure -> trade blocked")
    proceed, _ = r._trade_decision(["trade_partial"])
    check(proceed is False,
          "partial trade failure -> trade blocked")


if __name__ == "__main__":
    test_optional_steps_cover_the_upstream_api_steps()
    test_main_and_validate_are_critical()
    test_every_pipeline_step_is_classified()
    test_stop_on_failure()
    test_critical_failures()
    test_trade_decision()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURES:")
        for f in FAIL:
            print(" -", f)
        sys.exit(1)
    print("PASS test_run_all_optional.py")
