"""Run the regression tests:  python tests/run_tests.py  [--fast = skip the app / dashboard tests]
  test_midweek_repro.py  live engine reproduces the pinned backtests exactly
                         ((None,F,F) 470.31/1.4393, (30,F,F) 457.14/1.4451,
                          (30,T,F) 452.96/1.3172, (30,T,T) 387.57/1.2249)
                         and an independent re-implementation of the earnings rule
  test_rank_audit.py     saved ranks, scores, picks, mid-week decisions and earnings skips re-derived independently
  test_runner.py         run_all.py mode choice + NewsAPI once-a-day guard (pure logic, no API calls)
  test_app.py            Streamlit AppTest: page, charts, displayed ranks, Details widgets, captions, holdings alert
  test_paper_account.py  alpaca_paper.py + alpaca_paper_account.ipynb + run_all --sync-paper against a local MOCK server
  test_dashboard_http.py the app on test port 8599 answers 200 for / and /?symbol=NVDA (never touches 8501)
  test_paper_trade_safety.py  mocked (zero broker calls) regression tests for the paper_trade.py
                         safety fixes: fail-closed signal status + cash guard, stable client order
                         ids + broker reconciliation, sequenced SELL-then-BUY submit, morning
                         abort on unreadable positions, morning crash recovery
Read-only for the project (temporary files only, deleted afterwards). No quota APIs, no Alpaca account calls."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = ["test_midweek_repro.py", "test_rank_audit.py", "test_runner.py", "test_app.py", "test_paper_account.py",
         "test_dashboard_http.py", "test_paper_trade_safety.py"]


def main():
    fast = "--fast" in sys.argv
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1",
               PYTHONWARNINGS="ignore")
    results = []
    for t in TESTS:
        if fast and t in ("test_app.py", "test_dashboard_http.py"):
            continue
        if not os.path.exists(os.path.join(HERE, t)):
            continue
        t0 = time.time()
        p = subprocess.run([sys.executable, os.path.join(HERE, t)], cwd=ROOT, env=env, capture_output=True, text=True)
        ok = p.returncode == 0
        results.append((t, ok, time.time() - t0))
        tail = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()][-4 if ok else -25:]
        print(f"{'PASS' if ok else 'FAIL'}  {t}  ({time.time() - t0:.0f}s)")
        for line in tail:
            print("      " + line)
    n_ok = sum(ok for _, ok, _ in results)
    print(f"\n{n_ok}/{len(results)} test files passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
