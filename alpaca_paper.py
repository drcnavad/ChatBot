"""Read-only view of your Alpaca LIVE account: account summary, positions, orders, market clock and equity history.

NOTE (2026-09-28): this module used to read the Alpaca PAPER account. Paper trading is retired from this
pipeline, so the module now reads the LIVE account. The filename and the class/function names
(PaperAccount, paper_keys(), sync_paper_account(), ...) are kept unchanged so that pipeline_watchdog.py,
run_all.py and the dashboard keep working without edits.

Safety rules built into this module:
  * Only the live endpoint https://api.alpaca.markets/v2 is accepted; any other URL (the paper
    paper-api.alpaca.markets host, plain http, another version or host) raises PaperAccountError
    before a request is made.
  * Only HTTP GET requests to /account, /positions, /orders, /clock and /account/activities (deposits/withdrawals, fills)
    are possible.
    There is no code here that places, changes or cancels orders.
  * The keys come from .env (ALPACA_LIVE_KEY_ID, ALPACA_LIVE_SECRET_KEY). They are sent only in the request headers
    and are never printed, logged or written to a file.

Outputs of sync_paper_account():
  my_positions.csv                        Symbol,Shares of the live positions (paper_trade.py --positions reads it)
  Reports/live_portfolio_snapshot.csv    positions + cash/equity rows at the time of the sync
  Reports/live_account_history.csv       one row per sync (equity, cash, buying power, number of positions,
                                         lifetime net deposits)

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
SNAPSHOT_CSV = os.path.join(REPORTS, "live_portfolio_snapshot.csv")
HISTORY_CSV = os.path.join(REPORTS, "live_account_history.csv")

LIVE_BASE_URL = "https://api.alpaca.markets/v2"       # the ONLY accepted endpoint
KEY_ENV, SECRET_ENV = "ALPACA_LIVE_KEY_ID", "ALPACA_LIVE_SECRET_KEY"
# Also accepted, in this order, if the names above are empty: Alpaca's standard key names.
FALLBACK_NAMES = [("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")]
ALLOWED_PATHS = ("/account", "/positions", "/orders", "/clock", "/account/activities")   # read-only endpoints (GET only, live host only)
_LOCAL_TEST_URL = re.compile(r"http://(127\.0\.0\.1|localhost):\d{2,5}/v2")   # tests only (mock server on this machine)
CT = ZoneInfo("America/Chicago")


class PaperAccountError(Exception):
    """Refused URL, missing keys or a failed request (the message never contains the keys)."""


def check_base_url(url, allow_local_test=False):
    """Return the URL if it is the Alpaca live endpoint (or, in tests only, a localhost mock); otherwise raise."""
    url = str(url).rstrip("/")
    if url == LIVE_BASE_URL:
        return url
    if allow_local_test and _LOCAL_TEST_URL.fullmatch(url):
        return url
    raise PaperAccountError(f"refused base URL {url!r}: only {LIVE_BASE_URL} (Alpaca LIVE) is allowed")


def paper_keys():
    """(key_id, secret, source names) of the Alpaca LIVE keys from .env; ("", "", note) if none. Values are never printed."""
    load_dotenv(os.path.join(ROOT, ".env"))
    for key_name, secret_name in [(KEY_ENV, SECRET_ENV)] + FALLBACK_NAMES:
        key, secret = os.getenv(key_name, "").strip(), os.getenv(secret_name, "").strip()
        if key and secret:
            return key, secret, f"{key_name} / {secret_name}"
    return "", "", f"{KEY_ENV} / {SECRET_ENV} are missing or empty in .env"


class PaperAccount:
    """Read-only client for the Alpaca LIVE trading API."""

    def __init__(self, key_id=None, secret_key=None, base_url=LIVE_BASE_URL, timeout=15, _allow_local_test=False):
        self.base_url = check_base_url(base_url, allow_local_test=_allow_local_test)
        note = "keys passed in"
        if key_id is None or secret_key is None:
            env_key, env_secret, note = paper_keys()
            key_id = key_id if key_id is not None else env_key
            secret_key = secret_key if secret_key is not None else env_secret
        if not key_id or not secret_key:
            raise PaperAccountError(note if "missing" in note else f"{KEY_ENV} / {SECRET_ENV} are missing or empty in .env")
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

    # --- the read-only views ---------------------------------------------------------------------------------------
    @staticmethod
    def _order_rows(order_dicts):
        """Shared row format for recent_orders() and open_orders()."""
        rows = []
        for o in order_dicts:
            t = pd.to_datetime(o.get("submitted_at"), utc=True, errors="coerce")
            rows.append({"Submitted (CT)": t.tz_convert(CT).strftime("%Y-%m-%d %I:%M %p") if pd.notna(t) else "",
                         "Symbol": o.get("symbol"), "Side": o.get("side"), "Qty": o.get("qty") or o.get("notional"),
                         "Filled qty": o.get("filled_qty"), "Type": o.get("type"), "Status": o.get("status"),
                         "Filled avg price": o.get("filled_avg_price")})
        return pd.DataFrame(rows, columns=["Submitted (CT)", "Symbol", "Side", "Qty", "Filled qty", "Type", "Status",
                                           "Filled avg price"])

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
        return self._order_rows(self._get("/orders", {"status": "all", "limit": int(limit), "direction": "desc"}))

    def open_orders(self, limit=50):
        """Orders that are still open (not filled/canceled/expired), newest first; times in US Central.

        This is what the pipeline watchdog reads to answer "are the evening's orders still hanging?". """
        return self._order_rows(self._get("/orders", {"status": "open", "limit": int(limit), "direction": "desc"}))

    def market_clock(self):
        """Is the market open right now? Next open/close times, in US Central.

        Lets the watchdog distinguish "the market is closed" from a real failure. """
        c = self._get("/clock")

        def ct(ts):
            t = pd.to_datetime(ts, utc=True, errors="coerce")
            return t.tz_convert(CT).strftime("%Y-%m-%d %I:%M %p") if pd.notna(t) else ""

        return {"Is open": bool(c.get("is_open")), "Next open (CT)": ct(c.get("next_open")),
                "Next close (CT)": ct(c.get("next_close")), "Timestamp (CT)": ct(c.get("timestamp"))}

    def net_deposits(self):
        """Lifetime deposits minus withdrawals (CSD/CSW cash transfers + JNLC cash journals; Alpaca signs withdrawals
        negative; canceled transfers excluded), so the strategy's P&L can leave them out."""
        total, params = 0.0, {"activity_types": "CSD,CSW,JNLC", "page_size": 100, "direction": "desc"}
        while True:
            page = self._get("/account/activities", params)
            total += sum(float(a.get("net_amount") or 0) for a in page if a.get("status") != "canceled")
            if len(page) < 100:
                return total
            params = {**params, "page_token": page[-1]["id"]}

    def position_dicts(self):
        """Open positions as Alpaca returns them (GET /positions), for the dashboard's holdings table."""
        return self._get("/positions")

    def fills(self, max_pages=50):
        """Every order fill on the account (GET /account/activities, type FILL), oldest first."""
        out, params = [], {"activity_types": "FILL", "page_size": 100, "direction": "desc"}
        for _ in range(max_pages):
            page = self._get("/account/activities", params)
            out += page
            if len(page) < 100:
                break
            params = {**params, "page_token": page[-1]["id"]}
        return out[::-1]


# --- the dashboard's live holdings table (pure: no requests here) -------------------------------------------------------
QQQ_BASE_DATE, QQQ_BASE_CLOSE = "2026-10-02", 749.58   # the QQQ comparison row starts at QQQ's close on Fri Oct 2, 2026
QQQ_LABEL = "QQQ since Oct 2, 2026 — not held, comparison only"
HOLDING_COLS = ["Stock", "Shares", "Avg price", "First bought", "Cost basis", "Market value", "P/L $", "P/L %",
                "Price", "Today %", "Weight %"]


def holdings_refresh_key(now=None):
    """Cache key for the dashboard's live holdings: a new key every minute in regular market hours (8:30 AM-3:00 PM CT,
    12:00 PM on early-close days, NYSE sessions only), else every hour (after hours, nights, weekends, holidays)."""
    import backtest_engine as be
    now = (now or datetime.now(CT)).astimezone(CT)
    close = (12, 0) if be.is_early_close(now.date()) else (15, 0)
    if be.is_session(now.date()) and (8, 30) <= (now.hour, now.minute) < close:
        return f"{now:%Y-%m-%d %H:%M}"
    return f"{now:%Y-%m-%d %H}h"


def first_buy_dates(fills):
    """Symbol -> date (CT) of the earliest buy fill still part of the current position. Sells use up the oldest shares
    first (FIFO), so after a partial sell the date moves to the oldest shares still held; after a full exit it restarts."""
    lots = {}
    for f in sorted(fills, key=lambda f: str(f.get("transaction_time"))):
        q, qty = lots.setdefault(f["symbol"], []), float(f.get("qty") or 0)
        if f.get("side") == "buy":
            q.append([pd.to_datetime(f["transaction_time"], utc=True).tz_convert(CT).date(), qty])
            continue
        while qty > 1e-9 and q:
            used = min(qty, q[0][1])
            q[0][1] -= used
            qty -= used
            if q[0][1] <= 1e-9:
                q.pop(0)
    return {s: q[0][0] for s, q in lots.items() if q}


def holdings_table(positions, fills, equity, qqq_now=None):
    """One row per held stock, a Total row and a QQQ comparison row (not held). positions: GET /positions dicts;
    fills: GET /account/activities FILL dicts; equity: account equity (Weight % = market value / equity);
    qqq_now: latest QQQ price. The QQQ row (only with qqq_now) invests the same total cost basis in QQQ at its fixed
    Oct 2, 2026 close (QQQ_BASE_CLOSE) and values it at qqq_now: a hypothetical P/L, not a holding."""
    num = lambda d, k: float(d.get(k)) if d.get(k) not in (None, "") else float("nan")
    first = first_buy_dates(fills)
    rows = [{"Stock": p["symbol"], "Shares": num(p, "qty"), "Avg price": num(p, "avg_entry_price"),
             "First bought": first.get(p["symbol"]), "Cost basis": num(p, "cost_basis"),
             "Market value": num(p, "market_value"), "P/L $": num(p, "unrealized_pl"),
             "P/L %": num(p, "unrealized_plpc") * 100, "Price": num(p, "current_price"),
             "Today %": num(p, "change_today") * 100, "Weight %": num(p, "market_value") / equity * 100 if equity else float("nan"),
             "_prev": num(p, "qty") * num(p, "lastday_price")} for p in positions]
    if not rows:
        return pd.DataFrame(columns=HOLDING_COLS)
    t = pd.DataFrame(rows).sort_values("Market value", ascending=False)
    cost, pl, prev = t["Cost basis"].sum(), t["P/L $"].sum(), t["_prev"].sum()
    total = {"Stock": f"Total ({len(t)} stocks)", "Cost basis": cost, "Market value": t["Market value"].sum(), "P/L $": pl,
             "P/L %": pl / cost * 100 if cost else float("nan"), "Weight %": t["Weight %"].sum(),
             "Today %": (t["Market value"].sum() - prev) / prev * 100 if prev else float("nan")}
    out = [t.drop(columns="_prev"), pd.DataFrame([total])]
    if qqq_now:
        worth = cost * qqq_now / QQQ_BASE_CLOSE
        out.append(pd.DataFrame([{"Stock": QQQ_LABEL, "Avg price": QQQ_BASE_CLOSE, "First bought": pd.Timestamp(QQQ_BASE_DATE).date(),
                                  "Cost basis": cost, "Market value": worth, "P/L $": worth - cost,
                                  "P/L %": (qqq_now / QQQ_BASE_CLOSE - 1) * 100, "Price": qqq_now}]))
    return pd.concat(out, ignore_index=True)[HOLDING_COLS]


# --- files for the rest of the pipeline --------------------------------------------------------------------------------
def write_positions_csv(positions, path=POSITIONS_CSV):
    """my_positions.csv (Symbol,Shares) in the format paper_trade.py reads."""
    out = positions.loc[positions["Qty"] != 0, ["Symbol", "Qty"]].rename(columns={"Qty": "Shares"})
    out.to_csv(path, index=False)
    return path


def write_snapshot(summary, positions, snapshot_csv=SNAPSHOT_CSV, history_csv=HISTORY_CSV, now=None, net_deposits=0.0):
    """Positions + cash/equity rows (overwritten each sync) and one appended history row."""
    as_of = (now or datetime.now(CT)).strftime("%Y-%m-%d %H:%M")
    snap = positions.assign(As_Of=as_of)
    extra = pd.DataFrame([{"As_Of": as_of, "Symbol": "CASH", "Market value": summary["Cash"]},
                          {"As_Of": as_of, "Symbol": "TOTAL EQUITY", "Market value": summary["Equity"]}])
    snap = pd.concat([snap, extra], ignore_index=True)
    snap[["As_Of"] + [c for c in snap.columns if c != "As_Of"]].to_csv(snapshot_csv, index=False)
    row = pd.DataFrame([{"As_Of": as_of, "Equity": summary["Equity"], "Cash": summary["Cash"],
                         "Buying_Power": summary["Buying power"], "Positions": int((positions["Qty"] != 0).sum()),
                         "Net_Deposits": net_deposits}])
    row.to_csv(history_csv, mode="a", header=not os.path.exists(history_csv), index=False)
    return snapshot_csv, history_csv


def sync_paper_account(account=None, positions_csv=None, snapshot_csv=None, history_csv=None):
    """Read the LIVE account (3 GET calls) and write my_positions.csv + the snapshot files. Returns the summary dict.
    Paths default to the module settings (POSITIONS_CSV, SNAPSHOT_CSV, HISTORY_CSV), looked up at call time."""
    account = account or PaperAccount()
    positions_csv, snapshot_csv = positions_csv or POSITIONS_CSV, snapshot_csv or SNAPSHOT_CSV
    history_csv = history_csv or HISTORY_CSV
    summary, positions = account.account_summary(), account.positions()
    write_positions_csv(positions, positions_csv)
    write_snapshot(summary, positions, snapshot_csv, history_csv, net_deposits=account.net_deposits())
    return {**summary, "Positions": int((positions["Qty"] != 0).sum()), "positions_csv": positions_csv}


def main(argv=None):
    """Command line: print the live account (read-only); --sync also writes the files."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sync", action="store_true", help="also write my_positions.csv and the Reports snapshot files")
    p.add_argument("--orders", type=int, default=10, help="how many recent orders to show (default 10)")
    a = p.parse_args(argv)
    acct = PaperAccount()
    s = acct.account_summary()
    clock = acct.market_clock()
    print(f"LIVE account: equity ${s['Equity']:,.2f} | cash ${s['Cash']:,.2f} | buying power ${s['Buying power']:,.2f}")
    print(f"Market: {'OPEN' if clock['Is open'] else 'CLOSED'} (next open {clock['Next open (CT)'] or 'n/a'})")
    print(acct.positions().round(2).to_string(index=False))
    print(acct.recent_orders(a.orders).to_string(index=False))
    if a.sync:
        r = sync_paper_account(acct)
        print(f"wrote {os.path.relpath(r['positions_csv'], ROOT)} ({r['Positions']} positions) and the Reports snapshot files")


if __name__ == "__main__":
    main()
