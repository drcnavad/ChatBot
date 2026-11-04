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

# Paths that must not touch the real Reports/ files. Fresh per test file so one file's
# writes (manual blocks, …) cannot poison the next.
_ISOLATE = (
    ("STOCK_ANALYSIS_RUN_LOG", "run_log.csv"),
    ("STOCK_ANALYSIS_EARNINGS_STOP_STATE", "earnings_stop_state.json"),
    ("STOCK_ANALYSIS_SLEEVE_LEDGER", "sleeve_ledger.json"),
    ("STOCK_ANALYSIS_SLEEVE_PLAN", "sleeve_plan.csv"),
    ("STOCK_ANALYSIS_DECISION_RANKS", "decision_ranks.csv"),
    ("STOCK_ANALYSIS_MANUAL_ORDERS", "manual_orders.json"),
    ("STOCK_ANALYSIS_MANUAL_BLOCK", "manual_block.json"),
    ("STOCK_ANALYSIS_TARGET_SELLS", "target_sells.json"),
)

def _env_for_test():
    """Build a child env with brand-new temp paths (files must not exist yet — empty
    mktemp files break JSON loaders with JSONDecodeError)."""
    d = tempfile.mkdtemp(prefix="sa_test_")
    env = dict(os.environ,
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONDONTWRITEBYTECODE="1",
               STOCK_ANALYSIS_LIVE_HOLDINGS="off",
               STOCK_ANALYSIS_LAZY_EXPANDERS="off",
               PYTHONWARNINGS="ignore")
    for key, name in _ISOLATE:
        env[key] = os.path.join(d, name)  # path only; file created by the module when needed
    return env

def main():
    fast = "--fast" in sys.argv
    results = []
    for t in TESTS:
        if fast and t in ("test_app.py", "test_dashboard_http.py"):
            continue
        t0 = time.time()
        p = subprocess.run([sys.executable, os.path.join(HERE, t)], cwd=ROOT, env=_env_for_test(),
                           capture_output=True, text=True)
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
