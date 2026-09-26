"""Read-only view of your Alpaca PAPER account: account summary, positions and recent orders.

Safety rules built into this module:
  * Only the paper endpoint https://paper-api.alpaca.markets/v2 is accepted; any other URL (the live api.alpaca.markets,
    plain http, another version or host) raises PaperAccountError before a request is made.
  * Only HTTP GET requests to /account, /positions and /orders are possible. There is no code here that places,
    changes or cancels orders.
  * The keys come from .env (ALPACA_PAPER_KEY_ID, ALPACA_PAPER_SECRET_KEY). They are sent only in the request headers
    and are never printed, logged or written to a file.

Outputs of sync_paper_account():
  my_positions.csv                        Symbol,Shares of the paper positions (read by holdings_alert.py and the app alert)
  Reports/paper_portfolio_snapshot.csv    positions + cash/equity rows at the time of the sync
  Reports/paper_account_history.csv       one row per sync (equity, cash, buying power, number of positions)

    python alpaca_paper.py            # print the summary, positions and recent orders (no files written)
    python alpaca_paper.py --sync     # ... and write the three files above
"""
import argparse
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
POSITIONS_CSV = os.path.join(ROOT, "my_positions.csv")
SNAPSHOT_CSV = os.path.join(REPORTS, "paper_portfolio_snapshot.csv")
HISTORY_CSV = os.path.join(REPORTS, "paper_account_history.csv")

PAPER_BASE_URL = "https://paper-api.alpaca.markets/v2"       # the ONLY accepted endpoint
KEY_ENV, SECRET_ENV = "ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET_KEY"
# Also accepted, in this order, if the names above are empty: Alpaca's standard names, then the market-data names
# ALPACA_API_KEY / ALPACA_SECRET_KEY - but only if that key ID is a PAPER key (paper key IDs start with "PK").
FALLBACK_NAMES = [("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"), ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")]
ALLOWED_PATHS = ("/account", "/positions", "/orders")          # read-only endpoints
_LOCAL_TEST_URL = re.compile(r"http://(127\.0\.0\.1|localhost):\d{2,5}/v2")   # tests only (mock server on this machine)
CT = ZoneInfo("America/Chicago")


class PaperAccountError(Exception):
    """Refused URL, missing keys or a failed request (the message never contains the keys)."""


def check_base_url(url, allow_local_test=False):
    """Return the URL if it is the Alpaca paper endpoint (or, in tests only, a localhost mock); otherwise raise."""
    url = str(url).rstrip("/")
    if url == PAPER_BASE_URL:
        return url
    if allow_local_test and _LOCAL_TEST_URL.fullmatch(url):
        return url
    raise PaperAccountError(f"refused base URL {url!r}: only {PAPER_BASE_URL} (Alpaca PAPER) is allowed")


def paper_keys():
    """(key_id, secret, source names) of the Alpaca PAPER keys from .env; ("", "", note) if none. Values are never printed."""
    load_dotenv(os.path.join(ROOT, ".env"))
    for key_name, secret_name in [(KEY_ENV, SECRET_ENV)] + FALLBACK_NAMES:
        key, secret = os.getenv(key_name, "").strip(), os.getenv(secret_name, "").strip()
        if key and secret and (key_name == KEY_ENV or key.startswith("PK")):
            return key, secret, f"{key_name} / {secret_name}"
    other = os.getenv("ALPACA_API_KEY", "").strip()
    why = " (ALPACA_API_KEY is not a paper key, not used)" if other and not other.startswith("PK") else ""
    return "", "", f"{KEY_ENV} / {SECRET_ENV} are missing or empty in .env{why}"


class PaperAccount:
    """Read-only client for the Alpaca PAPER trading API."""

    def __init__(self, key_id=None, secret_key=None, base_url=PAPER_BASE_URL, timeout=15, _allow_local_test=False):
        self.base_url = check_base_url(base_url, allow_local_test=_allow_local_test)
        note = "keys passed in"
        if key_id is None or secret_key is None:
            env_key, env_secret, note = paper_keys()
            key_id = key_id if key_id is not None else env_key
            secret_key = secret_key if secret_key is not None else env_secret
        if not key_id or not secret_key:
            raise PaperAccountError(note if "missing" in note else f"{KEY_ENV} / {SECRET_ENV} are missing or empty in .env")
        self.key_source = note
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key, "Accept": "application/json"}
        self.timeout = timeout

    def __repr__(self):  # never show the keys
        return f"PaperAccount(base_url={self.base_url!r})"

    def _get(self, path, params=None):
        """GET one of the read-only endpoints and return the parsed JSON."""
        if path not in ALLOWED_PATHS:
            raise PaperAccountError(f"refused path {path!r}: only {', '.join(ALLOWED_PATHS)} are allowed (read-only)")
        url = self.base_url + path + ("?" + urllib.parse.urlencode(params) if params else "")
        req = urllib.request.Request(url, headers=self._headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise PaperAccountError(f"GET {path} failed: HTTP {e.code} {e.reason}") from None
        except urllib.error.URLError as e:
            raise PaperAccountError(f"GET {path} failed: {e.reason}") from None

    # --- the three read-only views -----------------------------------------------------------------------------------
    def account_summary(self):
        """Equity, cash and buying power (plus today's change) as a dict of floats."""
        a = self._get("/account")
        f = lambda k: float(a[k]) if a.get(k) not in (None, "") else float("nan")
        return {"Equity": f("equity"), "Cash": f("cash"), "Buying power": f("buying_power"),
                "Long market value": f("long_market_value"), "Change today": f("equity") - f("last_equity"),
                "Status": a.get("status", ""), "Currency": a.get("currency", "")}

    def positions(self):
        """Open positions: Symbol, Qty, Avg entry, Market value, Unrealized P/L ($ and %), Current price."""
        rows = [{"Symbol": p["symbol"], "Qty": float(p["qty"]), "Avg entry": float(p["avg_entry_price"]),
                 "Current price": float(p.get("current_price") or "nan"), "Market value": float(p["market_value"]),
                 "Unrealized P/L": float(p["unrealized_pl"]), "Unrealized P/L %": float(p.get("unrealized_plpc") or 0) * 100,
                 "Side": p.get("side", "long")} for p in self._get("/positions")]
        cols = ["Symbol", "Qty", "Avg entry", "Current price", "Market value", "Unrealized P/L", "Unrealized P/L %", "Side"]
        return pd.DataFrame(rows, columns=cols).sort_values("Market value", ascending=False).reset_index(drop=True)

    def recent_orders(self, limit=20):
        """The most recent orders (any status), newest first; times in US Central."""
        rows = []
        for o in self._get("/orders", {"status": "all", "limit": int(limit), "direction": "desc"}):
            t = pd.to_datetime(o.get("submitted_at"), utc=True, errors="coerce")
            rows.append({"Submitted (CT)": t.tz_convert(CT).strftime("%Y-%m-%d %I:%M %p") if pd.notna(t) else "",
                         "Symbol": o.get("symbol"), "Side": o.get("side"), "Qty": o.get("qty") or o.get("notional"),
                         "Filled qty": o.get("filled_qty"), "Type": o.get("type"), "Status": o.get("status"),
                         "Filled avg price": o.get("filled_avg_price")})
        return pd.DataFrame(rows, columns=["Submitted (CT)", "Symbol", "Side", "Qty", "Filled qty", "Type", "Status",
                                           "Filled avg price"])


# --- files for the rest of the pipeline --------------------------------------------------------------------------------
def write_positions_csv(positions, path=POSITIONS_CSV):
    """my_positions.csv (Symbol,Shares) in the format holdings_alert.py and paper_trade.py read."""
    out = positions.loc[positions["Qty"] != 0, ["Symbol", "Qty"]].rename(columns={"Qty": "Shares"})
    out.to_csv(path, index=False)
    return path


def write_snapshot(summary, positions, snapshot_csv=SNAPSHOT_CSV, history_csv=HISTORY_CSV, now=None):
    """Positions + cash/equity rows (overwritten each sync) and one appended history row."""
    as_of = (now or datetime.now(CT)).strftime("%Y-%m-%d %H:%M")
    snap = positions.assign(As_Of=as_of)
    extra = pd.DataFrame([{"As_Of": as_of, "Symbol": "CASH", "Market value": summary["Cash"]},
                          {"As_Of": as_of, "Symbol": "TOTAL EQUITY", "Market value": summary["Equity"]}])
    snap = pd.concat([snap, extra], ignore_index=True)
    snap[["As_Of"] + [c for c in snap.columns if c != "As_Of"]].to_csv(snapshot_csv, index=False)
    row = pd.DataFrame([{"As_Of": as_of, "Equity": summary["Equity"], "Cash": summary["Cash"],
                         "Buying_Power": summary["Buying power"], "Positions": int((positions["Qty"] != 0).sum())}])
    row.to_csv(history_csv, mode="a", header=not os.path.exists(history_csv), index=False)
    return snapshot_csv, history_csv


def sync_paper_account(account=None, positions_csv=None, snapshot_csv=None, history_csv=None):
    """Read the paper account (3 GET calls) and write my_positions.csv + the snapshot files. Returns the summary dict.
    Paths default to the module settings (POSITIONS_CSV, SNAPSHOT_CSV, HISTORY_CSV), looked up at call time."""
    account = account or PaperAccount()
    positions_csv, snapshot_csv = positions_csv or POSITIONS_CSV, snapshot_csv or SNAPSHOT_CSV
    history_csv = history_csv or HISTORY_CSV
    summary, positions = account.account_summary(), account.positions()
    write_positions_csv(positions, positions_csv)
    write_snapshot(summary, positions, snapshot_csv, history_csv)
    return {**summary, "Positions": int((positions["Qty"] != 0).sum()), "positions_csv": positions_csv}


def main(argv=None):
    """Command line: print the paper account (read-only); --sync also writes the files."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sync", action="store_true", help="also write my_positions.csv and the Reports snapshot files")
    p.add_argument("--orders", type=int, default=10, help="how many recent orders to show (default 10)")
    a = p.parse_args(argv)
    acct = PaperAccount()
    s = acct.account_summary()
    print(f"PAPER account: equity ${s['Equity']:,.2f} | cash ${s['Cash']:,.2f} | buying power ${s['Buying power']:,.2f}")
    print(acct.positions().round(2).to_string(index=False))
    print(acct.recent_orders(a.orders).to_string(index=False))
    if a.sync:
        r = sync_paper_account(acct)
        print(f"wrote {os.path.relpath(r['positions_csv'], ROOT)} ({r['Positions']} positions) and the Reports snapshot files")


if __name__ == "__main__":
    main()
