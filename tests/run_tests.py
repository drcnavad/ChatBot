"""Run every tests/test_*.py (each file's docstring says what it checks):  python tests/run_tests.py  [--fast = skip the
app / dashboard tests]. Read-only for the project (temporary files only); no quota APIs, no Alpaca account calls."""
import os
import subprocess
import tempfile
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = sorted(f for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))   # every test file

def main():
    fast = "--fast" in sys.argv
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1", STOCK_ANALYSIS_LIVE_HOLDINGS="off",
               PYTHONWARNINGS="ignore",   # tests never write the real Reports/run_log.csv
               STOCK_ANALYSIS_RUN_LOG=os.path.join(tempfile.mkdtemp(), "run_log.csv"),
               STOCK_ANALYSIS_EARNINGS_STOP_STATE=os.path.join(tempfile.mkdtemp(), "earnings_stop_state.json"))   # never the real stop sales
    results = []
    for t in TESTS:
        if fast and t in ("test_app.py", "test_dashboard_http.py"):
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
