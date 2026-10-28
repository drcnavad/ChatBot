"""The ONE source of what the account owns (Chirag, Oct 19, 2026): live Alpaca positions. Nothing is ever recorded as owned
before a fill - the sleeve ledger only tags which half a symbol belongs to, and a queued buy is a pending order, not a holding.

    get_positions()            {SYMBOL: {qty, market_value, current_price, avg_entry_price}} (GET /positions, 60 s cache)
    get_positions(client=c)    the same read through an alpaca-py TradingClient (order code: always fresh)
    held_qty(...)              {SYMBOL: qty}
"""
import time

from tax_lots import _num   # float or nan (one helper for every account view)

CACHE_SECS = 60
_cache = {}


def _f(x):
    """float(x), or None when x is missing, non-numeric or non-finite."""
    v = _num(x)
    return None if v != v else v


def position_rows(fresh=False):
    """Alpaca's raw position dicts (GET /positions), cached CACHE_SECS - the dashboard's holdings table.
    Returns copies: callers cannot mutate the cached rows."""
    hit = _cache.get("rows")
    if hit and not fresh and time.time() - hit[0] < CACHE_SECS:
        return [dict(r) for r in hit[1]]
    import alpaca_paper as ap
    rows = ap.PaperAccount().position_dicts()
    _cache["rows"] = (time.time(), rows)
    return [dict(r) for r in rows]


def get_positions(client=None, fresh=False, account=None):
    """Live LONG positions with qty > 0 (long-only account: zero/negative quantities are dropped, so a short leg
    would be invisible here). client = an alpaca-py TradingClient (orders: read right before trading, no cache);
    default the read-only REST account, cached CACHE_SECS. Raises when Alpaca cannot be read (callers fail closed)."""
    if client is not None:
        rows = [{"symbol": p.symbol, "qty": p.qty, "market_value": getattr(p, "market_value", None),
                 "current_price": getattr(p, "current_price", None), "avg_entry_price": getattr(p, "avg_entry_price", None)}
                for p in client.get_all_positions()]
    elif account is not None:                           # an alpaca_paper.PaperAccount (or a test double)
        rows = account.position_dicts()
    else:
        rows = position_rows(fresh)
    out = {}
    for p in rows:
        q = _f(p.get("qty"))
        if q and q > 0:
            out[str(p["symbol"]).upper()] = {"qty": q, "market_value": _f(p.get("market_value")),
                                              "current_price": _f(p.get("current_price")),
                                              "avg_entry_price": _f(p.get("avg_entry_price"))}
    return out


def held_qty(client=None, fresh=False, account=None):
    return {s: p["qty"] for s, p in get_positions(client, fresh, account).items()}
