#!/usr/bin/env python3
"""After-hours SELL reprice loop (Chirag, Oct 30 2026).

Every INTERVAL_SECS until STOP_CT (default 18:58 America/Chicago), for each open
extended-hours SELL from today's evening rebalance: fetch SIP ask; if sane and
different from the working limit, replace_order_by_id with limit=ask and remaining
unfilled qty. Updates Reports/live_pending_orders.json order_id so Mon 9 AM
fill-check does not double-sell. Never touches BUYs, do_not_sell, or unrelated orders.

  python ah_reprice.py                  # run until 18:58 CT today
  python ah_reprice.py --once           # one cycle then exit (test)
  python ah_reprice.py --stop 18:58 --interval 120
  python ah_reprice.py --dry-run        # log what would change; no replace
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

CT = ZoneInfo("America/Chicago")
ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
LOG_PATH = os.path.join(REPORTS, "logs", "ah_reprice.log")
PID_PATH = os.path.join(REPORTS, "logs", "ah_reprice.pid")
PENDING = os.path.join(REPORTS, "live_pending_orders.json")
MAX_ASK_VS_LAST = 0.10          # ask must be within 10% of last trade
INTERVAL_DEFAULT = 120          # seconds
STOP_DEFAULT = "18:58"


def _now():
    return datetime.now(CT)


def log(msg: str):
    line = f"{_now():%Y-%m-%d %H:%M:%S %Z}  {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _load_pending():
    if not os.path.exists(PENDING):
        return {"orders": []}
    with open(PENDING) as f:
        return json.load(f)


def _save_pending(data):
    tmp = PENDING + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, PENDING)


def _update_pending_order_id(old_id: str, new_id: str, new_limit: float, new_qty: float | None = None):
    """Point the pending SELL entry at the replaced broker order id."""
    data = _load_pending()
    changed = False
    for o in data.get("orders") or []:
        if str(o.get("order_id") or "") == str(old_id):
            o["order_id"] = str(new_id)
            o["limit_price"] = float(new_limit)
            if new_qty is not None:
                o["order_qty"] = int(new_qty) if float(new_qty) == int(new_qty) else float(new_qty)
            o["repriced_at"] = _now().isoformat(timespec="seconds")
            o.setdefault("prior_order_ids", [])
            if old_id not in o["prior_order_ids"]:
                o["prior_order_ids"].append(str(old_id))
            changed = True
            break
    if changed:
        _save_pending(data)
    return changed


def _tick(price: float) -> float:
    """Floor to Alpaca tick (cents >= $1, else 4 decimals) — sell limit at/below ask."""
    step = 100 if price >= 1 else 10_000
    return math.floor(round(price * step, 6)) / step


def _last_trade(sym: str, client_data) -> float | None:
    try:
        from alpaca.data.requests import StockLatestTradeRequest
        t = client_data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sym, feed="sip"))
        tr = t.get(sym) if isinstance(t, dict) else t
        if tr is None and hasattr(t, "get"):
            tr = t.get(sym)
        px = float(getattr(tr, "price", 0) or 0)
        return px if px > 0 else None
    except Exception:
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import StockLatestTradeRequest
            t = client_data.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=sym, feed=DataFeed.SIP))
            tr = t[sym] if sym in t else None
            px = float(getattr(tr, "price", 0) or 0) if tr else 0
            return px if px > 0 else None
        except Exception as e:
            log(f"  {sym}: last trade unavailable ({e})")
            return None


def _sip_quote(sym: str):
    import paper_trade as pt
    bid, ask, ts, feed = pt._latest_quote(sym)
    return bid, ask, ts, feed


def _sane_ask(sym: str, bid: float, ask: float, last: float | None) -> str | None:
    """None if ask is usable; else reason to keep current limit."""
    if not (ask and ask > 0):
        return "ask missing/<=0"
    if not (bid and bid > 0):
        return "bid missing/<=0"
    if ask < bid:
        return f"ask {ask} < bid {bid}"
    if last and last > 0:
        if abs(ask / last - 1) > MAX_ASK_VS_LAST:
            return f"ask {ask:g} >{MAX_ASK_VS_LAST:.0%} from last {last:g}"
    return None


def _today_prefix() -> str:
    return f"live-{_now():%Y%m%d}-SELL-"


def cycle(client, data_client, do_not_sell: set, dry_run: bool) -> int:
    """One reprice pass. Returns number of replaces submitted."""
    from alpaca.trading.enums import OrderSide, QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest

    prefix = _today_prefix()
    pending = _load_pending()
    pending_ids = {
        str(o.get("order_id"))
        for o in (pending.get("orders") or [])
        if str(o.get("side", "")).upper() == "SELL" and o.get("order_id")
    }
    open_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
    n_chg = 0
    for o in open_orders:
        side = getattr(getattr(o, "side", None), "value", str(getattr(o, "side", ""))).upper()
        if side != "SELL":
            continue
        if not getattr(o, "extended_hours", False):
            continue
        sym = str(o.symbol).upper()
        if sym in do_not_sell:
            log(f"  skip {sym}: do_not_sell")
            continue
        cid = getattr(o, "client_order_id", "") or ""
        oid = str(o.id)
        # Only today's rebalance sells (client id prefix) or pending-tracked ids.
        if not (cid.startswith(prefix) or oid in pending_ids):
            continue
        raw_st = getattr(o.status, "value", str(o.status)).lower()
        if raw_st not in ("new", "accepted", "pending_new", "partially_filled", "held"):
            continue
        filled = float(getattr(o, "filled_qty", 0) or 0)
        qty_total = float(getattr(o, "qty", 0) or 0)
        remaining = qty_total - filled
        if remaining < 1 - 1e-9:
            log(f"  skip {sym}: remaining {remaining:g} < 1")
            continue
        rem_whole = int(math.floor(remaining + 1e-9))
        if rem_whole < 1:
            continue
        cur_limit = float(o.limit_price) if o.limit_price is not None else None
        try:
            bid, ask, ts, feed = _sip_quote(sym)
        except Exception as e:
            log(f"  skip {sym}: quote error {e}")
            continue
        last = _last_trade(sym, data_client)
        why = _sane_ask(sym, float(bid or 0), float(ask or 0), last)
        if why:
            log(f"  keep {sym}: insane/unusable ask ({why}); limit={cur_limit}")
            continue
        new_limit = _tick(float(ask))
        if cur_limit is not None and abs(new_limit - cur_limit) < 1e-9:
            log(f"  ok   {sym}: already at ask {new_limit:g} (feed={feed})")
            continue
        log(f"  REPRICE {sym}: {cur_limit} -> {new_limit} (ask={ask:g} bid={bid:g} last={last} "
            f"rem={rem_whole} feed={feed}) oid={oid[:8]}…")
        if dry_run:
            n_chg += 1
            continue
        try:
            req = ReplaceOrderRequest(qty=rem_whole, limit_price=new_limit)
            new_o = client.replace_order_by_id(oid, req)
            new_id = str(new_o.id)
            new_lim = float(new_o.limit_price) if new_o.limit_price is not None else new_limit
            updated = _update_pending_order_id(oid, new_id, new_lim, rem_whole)
            log(f"  REPLACED {sym}: new_oid={new_id} limit={new_lim} pending_updated={updated} "
                f"status={getattr(new_o.status, 'value', new_o.status)}")
            n_chg += 1
        except Exception as e:
            log(f"  FAIL {sym} replace {oid}: {e}")
    return n_chg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="one cycle then exit")
    ap.add_argument("--dry-run", action="store_true", help="log changes only; do not replace")
    ap.add_argument("--interval", type=int, default=INTERVAL_DEFAULT, help="seconds between cycles (default 120)")
    ap.add_argument("--stop", default=STOP_DEFAULT, help="CT HH:MM to stop (default 18:58)")
    args = ap.parse_args(argv)

    hh, mm = map(int, args.stop.split(":"))
    stop_at = _now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    if stop_at <= _now() and not args.once:
        log(f"stop time {args.stop} CT already passed — exiting")
        return 0

    os.makedirs(os.path.dirname(PID_PATH), exist_ok=True)
    with open(PID_PATH, "w") as f:
        f.write(str(os.getpid()))

    import paper_trade as pt
    import sector_mapping as sm
    client = pt.paper_trading_client()
    data_client = pt.quote_client()
    do_not = {str(x).upper() for x in sm.do_not_sell}

    log(f"START pid={os.getpid()} stop={stop_at:%H:%M %Z} interval={args.interval}s "
        f"dry_run={args.dry_run} once={args.once} max_ask_vs_last={MAX_ASK_VS_LAST:.0%}")
    cycle_n = 0
    try:
        while True:
            cycle_n += 1
            now = _now()
            if not args.once and now >= stop_at:
                log(f"STOP at {now:%H:%M:%S %Z} (reached {args.stop} CT) after {cycle_n - 1} cycles")
                break
            log(f"--- cycle {cycle_n} ---")
            try:
                n = cycle(client, data_client, do_not, args.dry_run)
                log(f"cycle {cycle_n} done: {n} reprice(s)")
            except Exception:
                log(f"cycle {cycle_n} ERROR:\n{traceback.format_exc()}")
            if args.once:
                break
            # sleep in chunks so we can stop promptly
            wake = min(args.interval, max(1, int((stop_at - _now()).total_seconds())))
            for _ in range(wake):
                if _now() >= stop_at:
                    break
                time.sleep(1)
            if _now() >= stop_at:
                log(f"STOP at {_now():%H:%M:%S %Z} (reached {args.stop} CT)")
                break
    finally:
        try:
            if os.path.exists(PID_PATH):
                with open(PID_PATH) as f:
                    if f.read().strip() == str(os.getpid()):
                        os.remove(PID_PATH)
        except Exception:
            pass
        log(f"EXIT pid={os.getpid()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
