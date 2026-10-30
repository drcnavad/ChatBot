"""Sold-symbol buy block: after ANY live sell, that symbol is not bought again for N
business days (default 5, skipping weekends/holidays via backtest_engine.is_session).

Covers drop_guard, target_sells, Mon-Thu checks, trims, Friday rebalance — every path
that records a sell. Does NOT block sales of stocks already held (buy-side only).

Persisted in Reports/sold_block.json. Shown on the dashboard. Wired into
paper_trade.manual_list("buy") so spare-cash / sleeves / Friday substitutes / drop_guard
replacements all see it.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

import backtest_engine as be

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
BLOCK_JSON = os.environ.get("STOCK_ANALYSIS_SOLD_BLOCK") or os.path.join(REPORTS, "sold_block.json")
LOG_CSV = os.path.join(REPORTS, "sold_block_log.csv")
DEFAULT_BUSINESS_DAYS = 5
LOG_COLS = ["At_CT", "Symbol", "Action", "Sold_On", "Blocked_Through", "Source", "Note"]


def _load(path=None):
    try:
        with open(path or BLOCK_JSON) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    data.setdefault("blocks", {})          # {SYM: {sold_on, through, source, at, note}}
    data.setdefault("business_days", DEFAULT_BUSINESS_DAYS)
    return data


def _save(data, path=None):
    path = path or BLOCK_JSON
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def business_days_ahead(start, n, is_session=None):
    """Nth session day strictly after `start` (start itself not counted). Holidays skipped."""
    is_session = is_session or be.is_session
    d = start if isinstance(start, date) else date.fromisoformat(str(start)[:10])
    left = int(n)
    cur = d
    # guard against infinite loop
    for _ in range(max(n, 1) * 10 + 30):
        cur = cur + timedelta(days=1)
        if is_session(cur):
            left -= 1
            if left <= 0:
                return cur
    raise RuntimeError(f"could not find {n} session days after {d}")


def blocked_through(sold_on, n=None):
    n = DEFAULT_BUSINESS_DAYS if n is None else int(n)
    d = sold_on if isinstance(sold_on, date) else date.fromisoformat(str(sold_on)[:10])
    return business_days_ahead(d, n)


def _append_log(row):
    try:
        import db
        full = {c: row.get(c, "") for c in LOG_COLS}
        db.append_rows(LOG_CSV, [[full[c] for c in LOG_COLS]], LOG_COLS)
    except Exception:
        pass


def record_sell(sym, sold_on=None, source="", note="", n=None, now=None):
    """Record a live sell of `sym`. Extends an existing block if the new through-date is later.
    Returns the block entry. Does nothing for blank symbols."""
    sym = str(sym or "").strip().upper()
    if not sym:
        return None
    data = _load()
    n = data.get("business_days", DEFAULT_BUSINESS_DAYS) if n is None else int(n)
    now = now or datetime.now(be.CENTRAL)
    sold_on = sold_on or now.astimezone(be.CENTRAL).date()
    if isinstance(sold_on, datetime):
        sold_on = sold_on.astimezone(be.CENTRAL).date()
    elif not isinstance(sold_on, date):
        sold_on = date.fromisoformat(str(sold_on)[:10])
    through = blocked_through(sold_on, n)
    prev = data["blocks"].get(sym)
    entry = {
        "sold_on": sold_on.isoformat(),
        "through": through.isoformat(),
        "source": str(source or "")[:80],
        "note": str(note or "")[:200],
        "at": now.astimezone(be.CENTRAL).isoformat(timespec="seconds"),
        "business_days": n,
    }
    if prev and str(prev.get("through", "")) > entry["through"]:
        # keep the later expiry; update note/source
        entry["through"] = prev["through"]
        entry["sold_on"] = prev.get("sold_on", entry["sold_on"])
    data["blocks"][sym] = entry
    _save(data)
    _append_log({
        "At_CT": now.astimezone(be.CENTRAL).strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": sym, "Action": "BLOCK",
        "Sold_On": entry["sold_on"], "Blocked_Through": entry["through"],
        "Source": entry["source"], "Note": entry["note"],
    })
    return entry


def prune(as_of=None):
    """Drop expired blocks (as_of > through). Returns list of removed symbols."""
    as_of = as_of or datetime.now(be.CENTRAL).date()
    if isinstance(as_of, datetime):
        as_of = as_of.astimezone(be.CENTRAL).date()
    elif not isinstance(as_of, date):
        as_of = date.fromisoformat(str(as_of)[:10])
    data = _load()
    gone = []
    for sym, e in list(data["blocks"].items()):
        try:
            through = date.fromisoformat(str(e.get("through"))[:10])
        except (TypeError, ValueError):
            del data["blocks"][sym]
            gone.append(sym)
            continue
        if as_of > through:
            del data["blocks"][sym]
            gone.append(sym)
            _append_log({
                "At_CT": datetime.now(be.CENTRAL).strftime("%Y-%m-%d %H:%M:%S"),
                "Symbol": sym, "Action": "EXPIRE",
                "Sold_On": e.get("sold_on", ""), "Blocked_Through": e.get("through", ""),
                "Source": e.get("source", ""), "Note": "expired",
            })
    if gone:
        _save(data)
    return gone


def active_blocks(as_of=None):
    """{SYMBOL: note} for buy gates. Prunes expired first."""
    prune(as_of)
    as_of = as_of or datetime.now(be.CENTRAL).date()
    if isinstance(as_of, datetime):
        as_of = as_of.astimezone(be.CENTRAL).date()
    elif not isinstance(as_of, date):
        as_of = date.fromisoformat(str(as_of)[:10])
    data = _load()
    out = {}
    for sym, e in data["blocks"].items():
        try:
            through = date.fromisoformat(str(e.get("through"))[:10])
        except (TypeError, ValueError):
            continue
        if as_of <= through:
            src = e.get("source") or "live sell"
            out[sym] = f"sold-block until {through:%a %b %d} ({src})"
    return out


def is_blocked(sym, as_of=None):
    return str(sym).upper() in active_blocks(as_of)


def set_business_days(n):
    data = _load()
    data["business_days"] = max(1, int(n))
    _save(data)
    return data["business_days"]
