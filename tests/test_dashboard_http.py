"""Dashboard smoke test over HTTP on the TEST port 8599 (never 8501, the user's running app).

Starts `streamlit run app.py --server.port 8599` as a child process, checks HTTP 200 for / and /?symbol=NVDA and the
health endpoint, then stops only that child process. Fails (without touching anything) if 8599 is already in use.
Run: python tests/run_tests.py  (or python tests/test_dashboard_http.py)"""
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8599
BASE = f"http://localhost:{PORT}"


def port_busy(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def status(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:
        return getattr(e, "code", None) or str(e), ""


if port_busy(PORT):
    print(f"port {PORT} is already in use - not starting a second server (nothing was stopped)")
    sys.exit(1)
proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
                         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
fail = []
try:
    for _ in range(60):
        if status(BASE + "/_stcore/health")[0] == 200:
            break
        time.sleep(1)
    for path in ("/_stcore/health", "/", "/?symbol=NVDA"):
        code, body = status(BASE + path)
        ok = code == 200 and (path != "/_stcore/health" or body.strip() == "ok")
        print(f"   {'ok' if ok else '!! FAIL'}: GET {path} -> {code}")
        if not ok:
            fail.append(path)
finally:
    os.killpg(proc.pid, signal.SIGTERM)       # only the process group this test started
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
print("\nDASHBOARD HTTP OK" if not fail else f"\nDASHBOARD HTTP FAILURES: {fail}")
sys.exit(1 if fail else 0)
