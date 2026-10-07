"""Read-only view of the Alpaca LIVE account (the Paper* names are historical: paper trading was retired 2026-09-28).
Safety: only https://api.alpaca.markets/v2 and only GET requests to ALLOWED_PATHS (account, positions, orders, clock,
activities, daily portfolio history); nothing here places, changes or cancels orders. Keys come from .env
(ALPACA_LIVE_KEY_ID, ALPACA_LIVE_SECRET_KEY), go only into request headers and are never printed or saved.

    python alpaca_paper.py          # print the summary, positions and recent orders (writes nothing)
    python alpaca_paper.py --sync   # ... and write my_positions.csv, Reports/live_portfolio_snapshot.csv and
                                    #     Reports/live_account_history.csv (also run_all.py --sync-live)
"""
import argparse
import json
import os
import re
import threading
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
ALLOWED_PATHS = ("/account", "/positions", "/orders", "/clock", "/account/activities",   # read-only endpoints
                 "/account/portfolio/history")                                       # (GET only, live host only)
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
        except (TimeoutError, OSError, ValueError) as e:   # a read timeout / dropped connection / non-JSON reply
            raise PaperAccountError(f"GET {path} failed: {type(e).__name__}: {e}") from None

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

    def cash_flows(self):
        """Every deposit (+) and withdrawal (-), newest first: [{"date": "YYYY-MM-DD", "time": ISO UTC, "amount": $}] from
        the CSD/CSW cash transfers + JNLC cash journals (Alpaca signs withdrawals negative; canceled ones left out)."""
        out, params = [], {"activity_types": "CSD,CSW,JNLC", "page_size": 100, "direction": "desc"}
        while True:
            page = self._get("/account/activities", params)
            out += [{"date": str(a.get("date") or "")[:10], "time": str(a.get("created_at") or a.get("transaction_time") or ""),
                     "amount": float(a.get("net_amount") or 0)} for a in page if a.get("status") != "canceled"]
            if len(page) < 100:
                return out
            params = {**params, "page_token": page[-1]["id"]}

    def net_deposits(self):
        """Lifetime deposits minus withdrawals (cash_flows), so the strategy's P&L can leave them out."""
        return sum(f["amount"] for f in self.cash_flows())

    def snapshot(self):
        """{"equity", "flows", "at"}: the equity now and only the deposits already in it. The balance is read first, then
        the deposits, and a deposit booked after the balance read (Alpaca books them around 4:15 PM CT) is left out until
        the next read, so the equity and the deposits always describe the same moment."""
        equity = self.account_summary()["Equity"]
        at = pd.Timestamp.now(tz="UTC")
        return {"equity": equity, "flows": [f for f in self.cash_flows() if _flow_time(f) <= at], "at": at.tz_convert(CT)}

    def daily_history(self, start):
        """Alpaca's daily account history from `start` (GET /account/portfolio/history, 1D): a DataFrame of Date (the
        trading day), Equity (end of day, that day's deposits included) and Cashflow (that day's deposits - withdrawals)."""
        h = self._get("/account/portfolio/history", {"timeframe": "1D", "start": str(start)[:10], "pnl_reset": "no_reset",
                                                     "cashflow_types": "CSD,CSW,JNLC"})
        ts, n = h.get("timestamp") or [], len(h.get("timestamp") or [])
        cash = [sum(float((v or [])[i] or 0) if i < len(v or []) else 0.0 for v in (h.get("cashflow") or {}).values())
                for i in range(n)]
        days = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
        out = pd.DataFrame({"Date": days, "Equity": pd.to_numeric(pd.Series(h.get("equity") or [None] * n, dtype=object),
                                                                  errors="coerce").to_numpy(), "Cashflow": cash})
        return out.dropna(subset=["Equity"]).reset_index(drop=True)

    def position_dicts(self):
        """Open positions as Alpaca returns them (GET /positions), for the dashboard's holdings table."""
        return self._get("/positions")

    def fills(self, after=None, max_pages=50):
        """Every order fill on the account (GET /account/activities, type FILL), oldest first; only those after the
        time `after` (ISO, UTC) when given."""
        out, params = [], {"activity_types": "FILL", "page_size": 100, "direction": "desc", **({"after": after} if after else {})}
        for _ in range(max_pages):
            page = self._get("/account/activities", params)
            out += page
            if len(page) < 100:
                break
            params = {**params, "page_token": page[-1]["id"]}
        return out[::-1]

class FillHistory:
    """The fill history kept between the dashboard's reads, so the request count does not grow with the history: the
    first read of each day (CT) pages through every fill; later reads ask only for fills after the newest one kept (one
    minute of overlap, de-duplicated by id). On any problem it re-reads everything."""

    def __init__(self):
        self.fills, self.day, self._lock = [], None, threading.Lock()

    def update(self, account, now=None):
        today = (now or datetime.now(CT)).astimezone(CT).date()
        with self._lock:
            try:
                if self.day != today or not self.fills:
                    raise LookupError("full read")
                last = max(pd.to_datetime(f["transaction_time"], utc=True) for f in self.fills) - pd.Timedelta(minutes=1)
                seen = {f["id"] for f in self.fills}
                self.fills += [f for f in account.fills(after=last.strftime("%Y-%m-%dT%H:%M:%SZ")) if f["id"] not in seen]
            except Exception:
                self.fills, self.day = account.fills(), today
            return list(self.fills)

# --- the dashboard's live holdings table (pure: no requests here) -------------------------------------------------------
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

def holdings_table(positions, fills, equity):
    """One row per held stock and a Total row. positions: GET /positions dicts; fills: GET /account/activities FILL dicts;
    equity: account equity (Weight % = market value / equity). The ETF comparison is benchmark_table."""
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
    return pd.concat([t.drop(columns="_prev"), pd.DataFrame([total])], ignore_index=True)[HOLDING_COLS]

# --- the account vs index ETFs, deposits handled fairly (pure: no requests here) ----------------------------------------
BENCHMARK_ETFS = {"QQQ": "Nasdaq 100", "SPY": "S&P 500", "IWM": "Russell 2000", "DIA": "Dow Jones"}

def account_twr_index(daily):
    """Time-weighted growth of the account (1.0 on the first row) from daily rows (Date, Equity, Net_Deposits; e.g.
    Reports/forward_test_daily.csv, where each row is the account after the close, before that evening's deposit): each
    row's growth = Equity / (previous Equity + the net deposit since the previous row), so deposits and withdrawals are
    never returns. A deposit counts from the session after it arrives (Alpaca books them around 4:15 PM CT, after the
    close), the same day the ETFs in account_vs_etfs buy it (at its date's close)."""
    d = daily.assign(Date=pd.to_datetime(daily["Date"])).sort_values("Date")
    eq, nd = d["Equity"].astype(float), d["Net_Deposits"].astype(float)
    growth = eq / (eq.shift(1) + nd.diff())
    return pd.Series(growth.fillna(1.0).cumprod().to_numpy(), index=d["Date"].to_numpy())

def _close_on(closes, day, price_now):
    """The close of `day`, or of the next session with a close; price_now when there is none yet (today, open market)."""
    c = closes.dropna()
    c = c[c.index >= pd.Timestamp(day).normalize()]
    return float(c.iloc[0]) if len(c) else float(price_now)

def _flow_time(f):
    """When a cash_flows() entry happened, in CT (end of its date when Alpaca gave no time)."""
    t = pd.Timestamp(f["time"]) if f.get("time") else pd.NaT
    return pd.Timestamp(f"{f['date']} 23:59", tz=CT) if pd.isna(t) else (t.tz_localize("UTC") if t.tz is None else t).tz_convert(CT)

def account_vs_etfs(history, snap, closes, prices_now, start):
    """The account vs each BENCHMARK_ETFS fund since the `start` close, from one source: Alpaca's daily history
    (PaperAccount.daily_history) plus one live read (PaperAccount.snapshot). Deposits book after the close, so each day's
    account growth = (that day's end equity - that day's deposits) / the previous day's end equity; today's step uses the
    live equity less the deposits already in it that came after the last daily bar (a bar for today, if any, is left out:
    the live read replaces it). Each ETF buys the start balance at the start close and every later deposit (withdrawal)
    at the close of its date (the latest price when that close does not exist yet), valued at the latest price.
    Return % is time-weighted (for an ETF its price change since the start close); Gain $ = value - money put in.
    Returns (table, start date, money put in), or None when the history does not reach back to `start`."""
    start = pd.Timestamp(start).normalize()
    today = pd.Timestamp(snap["at"]).tz_convert(CT).tz_localize(None).normalize()
    h = history.assign(Date=pd.to_datetime(history["Date"]).dt.normalize()).sort_values("Date")
    h = h[(h["Date"] >= start) & (h["Date"] < today)].reset_index(drop=True)
    if h.empty or h["Date"].iloc[0] != start:
        return None
    eq, cash = h["Equity"].astype(float), h["Cashflow"].astype(float)
    last = h["Date"].iloc[-1]
    since = [f for f in snap["flows"] if pd.Timestamp(f["date"]).normalize() > last]
    before_today = sum(f["amount"] for f in since if pd.Timestamp(f["date"]).normalize() < today)   # in today's session
    today_in = sum(f["amount"] for f in since if pd.Timestamp(f["date"]).normalize() >= today)      # booked after it
    equity_now = float(snap["equity"])
    growth = ((eq - cash) / eq.shift(1)).iloc[1:].prod() * (equity_now - today_in) / (eq.iloc[-1] + before_today)
    acct_ret = growth - 1.0
    deposits = [(d, c) for d, c in zip(h["Date"].iloc[1:], cash.iloc[1:]) if c] + [(f["date"], f["amount"]) for f in since]
    put_in = float(eq.iloc[0]) + sum(c for _, c in deposits)
    rows = [{"Compared with": "Your account", "Return %": acct_ret * 100, "Value now": equity_now, "Gain $": equity_now - put_in,
             "Account ahead by (pts)": float("nan")}]
    for sym, name in BENCHMARK_ETFS.items():
        px_now = prices_now.get(sym)
        col = closes[sym] if closes is not None and sym in closes else pd.Series(dtype=float)
        if not px_now or col.dropna().empty:
            continue
        col = col.set_axis(pd.to_datetime(col.index).normalize())
        p0 = _close_on(col, start, px_now)
        units = float(eq.iloc[0]) / p0 + sum(c / _close_on(col, d, px_now) for d, c in deposits)
        ret = (px_now / p0 - 1) * 100
        rows.append({"Compared with": f"{sym} ({name})", "Return %": ret, "Value now": units * px_now,
                     "Gain $": units * px_now - put_in, "Account ahead by (pts)": acct_ret * 100 - ret})
    return pd.DataFrame(rows), start, put_in

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
    summary = account.account_summary()
    at = pd.Timestamp.now(tz="UTC")                       # deposits booked after the balance read are not in it yet
    positions = account.positions()
    write_positions_csv(positions, positions_csv)
    net = sum(f["amount"] for f in account.cash_flows() if _flow_time(f) <= at)
    write_snapshot(summary, positions, snapshot_csv, history_csv, net_deposits=net)
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
