"""alpaca_paper.py / alpaca_paper_account.ipynb / run_all.py --sync-paper against a MOCK Alpaca server on 127.0.0.1.

No request ever goes to Alpaca: the client is pointed at a local mock (allowed only with the test-only flag), dummy keys are
passed explicitly (.env is not read), and every file is written to a temporary folder (the real my_positions.csv and
Reports/ are never touched). Checks: URL guard, read-only GET requests, headers, parsing, files, notebook top-to-bottom.
Run: python tests/run_tests.py  (or python tests/test_paper_account.py)"""
import functools
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import pandas as pd

import alpaca_paper as ap
import holdings_alert

FAIL = []
KEY, SECRET = "test-key-id", "test-secret-value"
REQUESTS = []
FIXTURES = {
    "/v2/account": {"equity": "101234.50", "cash": "12000.25", "buying_power": "24000.50", "last_equity": "100000",
                    "long_market_value": "89234.25", "status": "ACTIVE", "currency": "USD"},
    "/v2/positions": [
        {"symbol": "NVDA", "qty": "10", "avg_entry_price": "180.00", "current_price": "190.00", "market_value": "1900.00",
         "unrealized_pl": "100.00", "unrealized_plpc": "0.0555", "side": "long"},
        {"symbol": "MRK", "qty": "25", "avg_entry_price": "80.00", "current_price": "78.00", "market_value": "1950.00",
         "unrealized_pl": "-50.00", "unrealized_plpc": "-0.025", "side": "long"}],
    "/v2/orders": [{"submitted_at": "2026-09-22T13:30:05.123Z", "symbol": "NVDA", "side": "buy", "qty": "10",
                    "filled_qty": "10", "type": "market", "status": "filled", "filled_avg_price": "180.00"}],
}


def expect(ok, what):
    if not ok:
        FAIL.append(what)
        print("   !! FAIL:", what)
    else:
        print("   ok:", what)


class Mock(BaseHTTPRequestHandler):
    def _record(self):
        REQUESTS.append({"method": self.command, "path": self.path, "key": self.headers.get("APCA-API-KEY-ID"),
                         "secret": self.headers.get("APCA-API-SECRET-KEY")})

    def do_GET(self):
        self._record()
        path = self.path.split("?")[0]
        if path == "/v2/forbidden":
            self.send_response(403); self.end_headers(); return
        body = json.dumps(FIXTURES.get(path, {"message": "not found"})).encode()
        self.send_response(200 if path in FIXTURES else 404)
        self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    def do_POST(self):                     # would reveal an order attempt
        self._record(); self.send_response(405); self.end_headers()

    do_DELETE = do_PATCH = do_PUT = do_POST

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), Mock)
threading.Thread(target=server.serve_forever, daemon=True).start()
MOCK = f"http://127.0.0.1:{server.server_port}/v2"
tmp = tempfile.mkdtemp(prefix="paper_test_")


def real_positions_state():
    p = os.path.join(ROOT, "my_positions.csv")
    return os.path.getmtime(p) if os.path.exists(p) else None


REAL_BEFORE = real_positions_state()
try:
    # ------------------------------------------------------------ URL guard (no request is made for refused URLs)
    for bad in ["https://api.alpaca.markets/v2", "http://paper-api.alpaca.markets/v2", "https://paper-api.alpaca.markets/v1",
                "https://paper-api.alpaca.markets.evil.com/v2", "https://example.com/v2", MOCK]:
        try:
            ap.PaperAccount(KEY, SECRET, base_url=bad)
            expect(False, f"refuses {bad}")
        except ap.PaperAccountError:
            expect(True, f"refuses {bad}")
    expect(ap.PaperAccount(KEY, SECRET).base_url == "https://paper-api.alpaca.markets/v2", "accepts the paper URL (no request made)")
    try:
        ap.PaperAccount(KEY, SECRET, base_url="http://10.0.0.5:8080/v2", _allow_local_test=True)
        expect(False, "test flag only allows localhost")
    except ap.PaperAccountError:
        expect(True, "test flag only allows localhost")
    try:
        ap.PaperAccount("", "", base_url=ap.PAPER_BASE_URL)
        expect(False, "missing keys raise")
    except ap.PaperAccountError as e:
        expect("ALPACA_PAPER_KEY_ID" in str(e), "missing keys raise a clear error")

    # ------------------------------------------------------------ read-only views against the mock
    acct = ap.PaperAccount(KEY, SECRET, base_url=MOCK, _allow_local_test=True)
    s = acct.account_summary()
    expect(abs(s["Equity"] - 101234.50) < 1e-9 and abs(s["Cash"] - 12000.25) < 1e-9 and abs(s["Buying power"] - 24000.50) < 1e-9,
           f"account summary parsed {s}")
    pos = acct.positions()
    expect(list(pos["Symbol"]) == ["MRK", "NVDA"] and pos.loc[pos.Symbol == "NVDA", "Qty"].item() == 10
           and abs(pos.loc[pos.Symbol == "MRK", "Unrealized P/L"].item() + 50) < 1e-9, "positions parsed (sorted by market value)")
    orders = acct.recent_orders(5)
    expect(len(orders) == 1 and orders["Submitted (CT)"].iloc[0] == "2026-09-22 08:30 AM", "orders parsed, time in CT")
    try:
        acct._get("/orders/abc")
        expect(False, "non-whitelisted path refused")
    except ap.PaperAccountError:
        expect(True, "non-whitelisted path refused")
    # a server error (HTTP 403) must not leak the keys
    req_fail = None
    ap.ALLOWED_PATHS = ap.ALLOWED_PATHS + ("/forbidden",)
    try:
        acct._get("/forbidden")
    except ap.PaperAccountError as e:
        req_fail = str(e)
    finally:
        ap.ALLOWED_PATHS = ("/account", "/positions", "/orders")
    expect(bool(req_fail) and "403" in req_fail and SECRET not in req_fail and KEY not in req_fail, f"HTTP error without keys: {req_fail}")
    expect(SECRET not in repr(acct) and KEY not in repr(acct), "repr hides the keys")

    # ------------------------------------------------------------ files (temporary paths only)
    p_csv, snap, hist = (os.path.join(tmp, n) for n in ("my_positions.csv", "snapshot.csv", "history.csv"))
    r = ap.sync_paper_account(acct, positions_csv=p_csv, snapshot_csv=snap, history_csv=hist)
    ap.sync_paper_account(acct, positions_csv=p_csv, snapshot_csv=snap, history_csv=hist)
    mp = pd.read_csv(p_csv)
    expect(list(mp.columns) == ["Symbol", "Shares"] and dict(zip(mp.Symbol, mp.Shares)) == {"MRK": 25, "NVDA": 10}, "my_positions.csv = Symbol,Shares")
    rp = holdings_alert.read_positions(p_csv)
    expect(set(rp.Symbol) == {"MRK", "NVDA"}, "holdings_alert reads the synced positions file")
    alert = holdings_alert.build_alert(positions_path=p_csv)
    expect(alert["uses_positions"] and alert["lines"], "holdings alert builds from the synced file")
    sn = pd.read_csv(snap)
    expect({"CASH", "TOTAL EQUITY", "NVDA", "MRK"} <= set(sn.Symbol), "snapshot has positions + cash/equity rows")
    expect(len(pd.read_csv(hist)) == 2 and r["Positions"] == 2, "history appends one row per sync")

    # ------------------------------------------------------------ run_all --sync-paper hook (mocked, temp paths)
    import run_all
    real = ap.PaperAccount, ap.POSITIONS_CSV, ap.SNAPSHOT_CSV, ap.HISTORY_CSV
    try:
        ap.PaperAccount = functools.partial(real[0], KEY, SECRET, base_url=MOCK, _allow_local_test=True)
        ap.POSITIONS_CSV, ap.SNAPSHOT_CSV, ap.HISTORY_CSV = (os.path.join(tmp, n) for n in ("rp.csv", "rs.csv", "rh.csv"))
        run_all.sync_paper()
        expect(os.path.exists(ap.POSITIONS_CSV) and os.path.exists(ap.SNAPSHOT_CSV), "run_all.sync_paper writes the positions file")
    finally:
        ap.PaperAccount, ap.POSITIONS_CSV, ap.SNAPSHOT_CSV, ap.HISTORY_CSV = real
    expect(run_all.main(["--dry-run", "--quick", "--sync-paper"]) == 0, "run_all --dry-run --sync-paper plans without calling")

    # ------------------------------------------------------------ the notebook, top to bottom, against the mock
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(os.path.join(ROOT, "alpaca_paper_account.ipynb"), as_version=4)
    nb_tmp = os.path.join(tmp, "nb")
    os.makedirs(nb_tmp)
    inject = (f"import functools, alpaca_paper\n"
              f"alpaca_paper.PaperAccount = functools.partial(alpaca_paper.PaperAccount, {KEY!r}, {SECRET!r}, "
              f"base_url={MOCK!r}, _allow_local_test=True)\n"
              f"alpaca_paper.POSITIONS_CSV = {os.path.join(nb_tmp, 'my_positions.csv')!r}\n"
              f"alpaca_paper.SNAPSHOT_CSV = {os.path.join(nb_tmp, 'snapshot.csv')!r}\n"
              f"alpaca_paper.HISTORY_CSV = {os.path.join(nb_tmp, 'history.csv')!r}\n")
    nb.cells.insert(0, nbformat.v4.new_code_cell(inject))
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": ROOT}}).execute()
    text = json.dumps([c.get("outputs", []) for c in nb.cells])
    expect("Connected to" in text and "Saved 2 positions" in text and '"error"' not in text, "notebook runs top to bottom (mock)")
    expect(os.path.exists(os.path.join(nb_tmp, "my_positions.csv")), "notebook wrote the (temporary) positions file")
    expect(SECRET not in text, "notebook output never shows the secret")

    # ------------------------------------------------------------ read-only: only GETs, correct headers, no order code
    expect(REQUESTS and all(q["method"] == "GET" for q in REQUESTS), f"only GET requests ({len(REQUESTS)} made)")
    expect(all(q["key"] == KEY and q["secret"] == SECRET for q in REQUESTS), "auth headers sent on every request")
    expect(all(q["path"].split("?")[0] in ("/v2/account", "/v2/positions", "/v2/orders", "/v2/forbidden") for q in REQUESTS),
           "only account/positions/orders endpoints requested")
    src = open(os.path.join(ROOT, "alpaca_paper.py")).read()
    expect(all(w not in src for w in ('"POST"', '"DELETE"', '"PATCH"', "submit_order", "TradingClient")) and src.count('method="GET"') == 1,
           "alpaca_paper.py contains no order-placing code")
    expect(real_positions_state() == REAL_BEFORE, "real my_positions.csv untouched")
finally:
    server.shutdown()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

print("\nPAPER ACCOUNT TESTS OK" if not FAIL else f"\nPAPER ACCOUNT TEST FAILURES ({len(FAIL)}): {FAIL}")
sys.exit(1 if FAIL else 0)
