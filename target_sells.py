"""Target limit sells (Chirag Oct 29 2026): hold a DAY+extended_hours limit sell at a
chosen price so a position can fill in pre-market / regular / after-hours.

Alpaca allows extended_hours only with time_in_force=DAY, so a launchd job
(com.stockanalysis.target_sells) refreshes the order each trading day from ~3:00 AM CT
through 7:00 PM CT. Idempotent: same day + qty + limit leaves the open order alone.

Config: Reports/target_sells_config.json (generic list — add any symbol):
  {"enabled": true, "dry_run": false,
   "targets": {"CRL": {"limit": 387.68, "qty": "full", "added": "2026-10-29"}},
   "window_start_ct": [3, 0], "window_end_ct": [19, 0]}
CLI: python target_sells.py add CRL 387.68
     python target_sells.py remove CRL
     python target_sells.py list
     python target_sells.py --dry-run | --scheduled

When the limit fills (or another live path sells the name): record sold_block, remove the
target entry. drop_guard / 2:30 must call cancel_for_symbol() before selling the same name.
If the daily/weekly plan wants to exit the name (out of top list), cancel the target and
sell via the normal bid-based path — sold_block still applies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import backtest_engine as be
import sold_block

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
CONFIG_JSON = os.path.join(REPORTS, "target_sells_config.json")
STATE_JSON = os.path.join(REPORTS, "target_sells_state.json")
LOCK_PATH = os.path.join(REPORTS, ".target_sells.lock")
LOG_CSV = os.path.join(REPORTS, "target_sells_log.csv")

DEFAULT_CONFIG = {
    "enabled": True,
    "dry_run": True,
    "targets": {},                          # {SYM: {"limit": float, "qty": "full"|number}}
    "window_start_ct": [3, 0],              # ~Alpaca pre-market open (4 AM ET)
    "window_end_ct": [19, 0],               # ~Alpaca after-hours end (8 PM ET)
}
LOG_COLS = [
    "At_CT", "Symbol", "Action", "Qty", "Limit", "Order_Id", "Client_Order_Id",
    "Dry_Run", "Note",
]
CID_PREFIX = "live-tsell"


def load_config(path=None):
    cfg = dict(DEFAULT_CONFIG)
    cfg["targets"] = {}
    try:
        with open(path or CONFIG_JSON) as f:
            raw = json.load(f)
        for k, v in raw.items():
            if k == "targets" and isinstance(v, dict):
                cfg["targets"] = {str(s).upper(): dict(e) for s, e in v.items() if isinstance(e, dict)}
            elif k in DEFAULT_CONFIG or k.startswith("_"):
                cfg[k] = v
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg, path=None):
    path = path or CONFIG_JSON
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)


def _load_state(path=None):
    try:
        with open(path or STATE_JSON) as f:
            s = json.load(f)
    except (OSError, ValueError):
        s = {}
    s.setdefault("orders", {})   # {SYM: {order_id, cid, day, qty, limit, at}}
    return s


def _save_state(s, path=None):
    path = path or STATE_JSON
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, path)


def _append_log(row):
    try:
        import db
        full = {c: row.get(c, "") for c in LOG_COLS}
        db.append_rows(LOG_CSV, [[full[c] for c in LOG_COLS]], LOG_COLS)
    except Exception as e:
        print(f"target_sells log failed: {e}")


def _acquire_lock():
    os.makedirs(REPORTS, exist_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        return True, "ok"
    except FileExistsError:
        try:
            with open(LOCK_PATH) as f:
                pid = int(f.read().strip() or 0)
            os.kill(pid, 0)
            return False, f"held by pid {pid}"
        except (OSError, ValueError):
            try:
                os.remove(LOCK_PATH)
            except OSError:
                pass
            return _acquire_lock()


def _release_lock():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


def in_window(now_ct, cfg):
    d = now_ct.date()
    if not be.is_session(d):
        return False, "market closed (weekend/holiday)"
    sh, sm = cfg.get("window_start_ct", [3, 0])
    eh, em = cfg.get("window_end_ct", [19, 0])
    start = now_ct.replace(hour=int(sh), minute=int(sm), second=0, microsecond=0)
    end = now_ct.replace(hour=int(eh), minute=int(em), second=0, microsecond=0)
    if be.is_early_close(d):
        # early-close after-hours end earlier (~5 PM ET = 4 PM CT)
        end = min(end, now_ct.replace(hour=16, minute=0, second=0, microsecond=0))
    if now_ct < start:
        return False, f"before window ({int(sh):02d}:{int(sm):02d} CT)"
    if now_ct >= end:
        return False, f"after window (ends {int(eh):02d}:{int(em):02d} CT)"
    return True, "in window"


def trading_client():
    import paper_trade as pt
    return pt.paper_trading_client()


def account():
    import alpaca_paper as ap
    return ap.PaperAccount()


def held_qty(sym, acct=None):
    acct = acct or account()
    pos = acct.positions()
    import pandas as pd
    if isinstance(pos, pd.DataFrame) and not pos.empty and "Symbol" in pos.columns:
        row = pos[pos["Symbol"].astype(str).str.upper() == sym.upper()]
        if row.empty:
            return 0.0
        return float(row.iloc[0]["Qty"])
    return 0.0


def target_qty(sym, entry, held):
    """Whole shares for the limit (Alpaca extended-hours prefers whole). 'full' = floor(held)."""
    raw = entry.get("qty", "full")
    if raw == "full" or raw is None or raw == "":
        return math.floor(held + 1e-9)
    try:
        q = float(raw)
    except (TypeError, ValueError):
        return math.floor(held + 1e-9)
    return math.floor(min(q, held) + 1e-9)


def cid_for(sym, day):
    import pandas as pd
    d = pd.Timestamp(day).strftime("%Y%m%d")
    return f"{CID_PREFIX}-{d}-{str(sym).upper()}"[:48]


def is_target_cid(cid):
    return bool(cid) and str(cid).startswith(CID_PREFIX + "-")


def find_open_target(client, sym, broker=None):
    """Return (order|None) for an open target-sell of sym (any day)."""
    import paper_trade as pt
    sym = str(sym).upper()
    if broker is None:
        broker = pt._broker_orders_by_client_id(client) or {}
    best = None
    for cid, o in broker.items():
        if not is_target_cid(cid):
            continue
        parts = str(cid).split("-")
        # live-tsell-YYYYMMDD-SYM
        if len(parts) < 4 or parts[-1].upper() != sym:
            continue
        st = getattr(o.status, "value", str(o.status)).lower()
        if st in ("new", "accepted", "pending_new", "partially_filled", "held"):
            best = o
            break
        if st == "filled" and best is None:
            best = o  # remember fill for caller
    return best


def cancel_for_symbol(client, sym, reason=""):
    """Cancel any open target-sell for sym. Returns (canceled_order_id|None, note)."""
    import paper_trade as pt
    sym = str(sym).upper()
    broker = pt._broker_orders_by_client_id(client)
    if broker is None:
        return None, "broker unreadable"
    canceled = []
    for cid, o in list(broker.items()):
        if not is_target_cid(cid):
            continue
        parts = str(cid).split("-")
        if len(parts) < 4 or parts[-1].upper() != sym:
            continue
        st = getattr(o.status, "value", str(o.status)).lower()
        if st in ("filled", "canceled", "cancelled", "expired", "rejected", "done_for_day"):
            continue
        try:
            client.cancel_order_by_id(o.id)
            canceled.append(str(o.id))
        except Exception as e:
            return None, f"cancel failed: {str(e)[:100]}"
    # also clear state
    state = _load_state()
    if sym in state.get("orders", {}):
        state["orders"].pop(sym, None)
        _save_state(state)
    if canceled:
        return canceled[0], f"canceled {','.join(canceled)}" + (f" ({reason})" if reason else "")
    return None, "no open target order"


def remove_target(sym, cfg=None, reason=""):
    cfg = cfg or load_config()
    sym = str(sym).upper()
    if sym in cfg.get("targets", {}):
        del cfg["targets"][sym]
        save_config(cfg)
        return True
    return False


def _on_filled(sym, qty, limit, order_id, source, cfg, now):
    """After a target fill (or external sell of a target name): sold_block + drop target."""
    sold_block.record_sell(sym, sold_on=now.date(), source=source,
                           note=f"target limit {limit} qty {qty} order {order_id}")
    remove_target(sym, cfg, reason="filled")
    state = _load_state()
    state.get("orders", {}).pop(sym, None)
    _save_state(state)


def _submit(client, sym, qty, limit, cid, dry_run):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest
    if dry_run:
        return None, f"dry-run would SELL {qty:g} {sym} @ {limit} cid={cid}"
    req = LimitOrderRequest(
        symbol=sym, qty=qty, side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY, limit_price=float(limit),
        extended_hours=True, client_order_id=cid,
    )
    o = client.submit_order(req)
    return o, "submitted"


def refresh_one(sym, entry, cfg, state, now, client, broker, acct, dry_run):
    """Ensure one target has a working DAY+extended limit sell. Returns log row dict."""
    import paper_trade as pt
    sym = str(sym).upper()
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    day = now.date()
    limit = float(entry.get("limit") or 0)
    base = {"At_CT": stamp, "Symbol": sym, "Action": "", "Qty": "", "Limit": limit,
            "Order_Id": "", "Client_Order_Id": "", "Dry_Run": "yes" if dry_run else "no", "Note": ""}
    if limit <= 0:
        base["Action"] = "SKIP"
        base["Note"] = "limit missing/<=0"
        return base
    held = held_qty(sym, acct)
    qty = target_qty(sym, entry, held)
    base["Qty"] = qty
    if held <= 0 or qty < 1:
        # Not held: cancel any leftover target order, never place a new one.
        o = find_open_target(client, sym, broker) if client and not dry_run else None
        if o is not None:
            st = getattr(o.status, "value", str(o.status)).lower()
            filled_qty = pt._safe_number(getattr(o, "filled_qty", 0))
            if st == "filled" or filled_qty > 0:
                _on_filled(sym, filled_qty or qty, limit, str(o.id), "target_sell_fill", cfg, now)
                base["Action"] = "FILLED"
                base["Order_Id"] = str(o.id)
                base["Note"] = "filled; sold_block recorded; target removed"
                return base
            # Open leftover (partial/unfilled) while flat — cancel and drop target
            if not dry_run:
                try:
                    client.cancel_order_by_id(o.id)
                except Exception as e:
                    base["Note"] = f"cancel leftover failed: {str(e)[:100]}"
            base["Order_Id"] = str(getattr(o, "id", ""))
        remove_target(sym, cfg, reason="no longer held")
        state.get("orders", {}).pop(sym, None)
        _save_state(state)
        base["Action"] = "REMOVED"
        base["Note"] = (base.get("Note") + "; " if base.get("Note") else "") + "not held — canceled leftover, removed target, no new order"
        return base

    cid = cid_for(sym, day)
    base["Client_Order_Id"] = cid
    # Check existing open target
    open_o = None
    if broker:
        for c, o in broker.items():
            if not is_target_cid(c):
                continue
            parts = str(c).split("-")
            if len(parts) < 4 or parts[-1].upper() != sym:
                continue
            st = getattr(o.status, "value", str(o.status)).lower()
            if st in ("new", "accepted", "pending_new", "partially_filled", "held"):
                open_o = o
                break
            if st == "filled":
                _on_filled(sym, pt._safe_number(getattr(o, "filled_qty", 0)), limit,
                           str(o.id), "target_sell_fill", cfg, now)
                base["Action"] = "FILLED"
                base["Order_Id"] = str(o.id)
                base["Note"] = "filled"
                return base

    if open_o is not None:
        open_qty = pt._safe_number(getattr(open_o, "qty", 0))
        open_limit = pt._safe_number(getattr(open_o, "limit_price", 0))
        open_cid = getattr(open_o, "client_order_id", "") or ""
        # same day cid + same qty + same limit → leave alone
        if open_cid == cid and abs(open_qty - qty) < 1e-9 and abs(open_limit - limit) < 1e-9:
            base["Action"] = "KEEP"
            base["Order_Id"] = str(open_o.id)
            base["Note"] = "open order matches"
            state["orders"][sym] = {"order_id": str(open_o.id), "cid": cid, "day": str(day),
                                    "qty": qty, "limit": limit, "at": stamp}
            _save_state(state)
            return base
        # qty/limit/day mismatch → cancel and replace
        try:
            if not dry_run:
                client.cancel_order_by_id(open_o.id)
                time.sleep(0.5)
            base["Note"] = f"replaced prior {open_o.id} (qty/limit/day changed)"
        except Exception as e:
            base["Action"] = "SKIP"
            base["Note"] = f"could not cancel prior: {str(e)[:100]}"
            return base

    # Also cancel any non-target open sells? No — leave other jobs alone; they cancel us first.
    # Check non-target open orders that would block (earnings_stop / drop_guard pattern)
    held_now, open_now = (None, [])
    try:
        held_now, open_now = pt._symbol_state(client, sym)
    except Exception:
        pass
    foreign = [o for o in (open_now or []) if not is_target_cid(getattr(o, "client_order_id", "") or "")]
    if foreign and not dry_run:
        base["Action"] = "SKIP"
        base["Note"] = f"other open order(s) present: {[str(o.id) for o in foreign][:3]}"
        return base

    try:
        order, note = _submit(client, sym, qty, limit, cid, dry_run)
    except Exception as e:
        base["Action"] = "FAIL"
        base["Note"] = f"submit: {str(e)[:150]}"
        return base
    if dry_run:
        base["Action"] = "WOULD_PLACE"
        base["Note"] = note
        return base
    base["Action"] = "PLACED"
    base["Order_Id"] = str(getattr(order, "id", ""))
    base["Note"] = note
    state["orders"][sym] = {"order_id": base["Order_Id"], "cid": cid, "day": str(day),
                            "qty": qty, "limit": limit, "at": stamp}
    _save_state(state)
    return base


def run_check(now=None, dry_run=None, cfg=None):
    """Refresh all target sells. Returns list of log rows."""
    cfg = cfg or load_config()
    now = now or datetime.now(be.CENTRAL)
    if now.tzinfo is None:
        now = now.replace(tzinfo=be.CENTRAL)
    now_ct = now.astimezone(be.CENTRAL)
    if dry_run is None:
        dry_run = bool(cfg.get("dry_run", True))
    ok, why = in_window(now_ct, cfg)
    if not ok:
        print(f"{now_ct:%Y-%m-%d %H:%M:%S} CT: idle — {why}")
        return []
    if not cfg.get("enabled", True):
        print(f"{now_ct:%Y-%m-%d %H:%M:%S} CT: kill switch off — idle")
        return []
    targets = cfg.get("targets") or {}
    if not targets:
        print(f"{now_ct:%Y-%m-%d %H:%M:%S} CT: no targets configured")
        return []
    got, lock_why = _acquire_lock()
    if not got:
        print(f"{now_ct:%Y-%m-%d %H:%M:%S} CT: idle — lock {lock_why}")
        return []
    rows = []
    try:
        # Idle while the 2:30 trade job or drop_guard/earnings fill-check holds a live lock
        # (stale lock files with dead PIDs are ignored so we don't stall forever).
        import run_all
        def _pid_lock_busy(path):
            try:
                with open(path) as f:
                    meta = json.load(f)
                pid = int(meta.get("pid", 0))
                if pid <= 0:
                    return False
                os.kill(pid, 0)
                return True
            except (OSError, ValueError, TypeError):
                return False
        if _pid_lock_busy(run_all.TRADE_LOCK):
            print(f"{now_ct:%Y-%m-%d %H:%M:%S} CT: idle — trade job running")
            return []
        if _pid_lock_busy(run_all.FILL_LOCK):
            print(f"{now_ct:%Y-%m-%d %H:%M:%S} CT: idle — fill-check / drop_guard lock held")
            return []
        client = None if dry_run else trading_client()
        import paper_trade as pt
        broker = {} if dry_run else (pt._broker_orders_by_client_id(client) or {})
        acct = account()
        state = _load_state()
        # reload cfg inside loop so remove_target persists
        for sym, entry in list(targets.items()):
            cfg = load_config()
            if sym.upper() not in {s.upper() for s in (cfg.get("targets") or {})}:
                continue
            entry = (cfg.get("targets") or {}).get(sym.upper()) or entry
            row = refresh_one(sym, entry, cfg, state, now_ct, client, broker, acct, dry_run)
            _append_log(row)
            rows.append(row)
            print(f"  {sym}: {row['Action']} qty={row['Qty']} limit={row['Limit']} "
                  f"oid={row['Order_Id'] or '-'} — {row['Note']}")
        print(f"{now_ct:%Y-%m-%d %H:%M:%S} CT: refreshed {len(rows)} target(s); dry_run={dry_run}")
        return rows
    finally:
        _release_lock()


def ensure_config():
    if not os.path.exists(CONFIG_JSON):
        save_config(dict(DEFAULT_CONFIG))
        print(f"wrote {CONFIG_JSON}")
    return load_config()


def add_target(sym, limit, qty="full", cfg=None):
    """Add/update a target. Returns the entry."""
    from datetime import date as _date
    cfg = cfg or load_config()
    sym = str(sym).upper()
    entry = {
        "limit": float(limit),
        "qty": qty if qty == "full" else float(qty),
        "added": _date.today().isoformat(),
    }
    cfg.setdefault("targets", {})[sym] = entry
    save_config(cfg)
    return entry


def remove_target_cli(sym):
    """Cancel open order (if any) and remove config entry. Returns note."""
    cfg = load_config()
    sym = str(sym).upper()
    note = []
    try:
        client = trading_client()
        oid, n = cancel_for_symbol(client, sym, reason="cli remove")
        if oid:
            note.append(n)
    except Exception as e:
        note.append(f"cancel skipped: {e}")
    if remove_target(sym, cfg, reason="cli remove"):
        note.append("removed from config")
    else:
        note.append("not in config")
    state = _load_state()
    state.get("orders", {}).pop(sym, None)
    _save_state(state)
    return "; ".join(note)


def list_targets(cfg=None):
    cfg = cfg or load_config()
    state = _load_state()
    rows = []
    for sym, e in sorted((cfg.get("targets") or {}).items()):
        o = (state.get("orders") or {}).get(sym.upper(), {})
        rows.append({
            "symbol": sym, "limit": e.get("limit"), "qty": e.get("qty", "full"),
            "added": e.get("added", ""), "order_id": o.get("order_id", ""),
            "day": o.get("day", ""),
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Target limit sells (LIVE DAY+extended). Config: Reports/target_sells_config.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scheduled", action="store_true", help="one launchd tick")
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--status", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p_add = sub.add_parser("add", help="add/update a target sell")
    p_add.add_argument("symbol")
    p_add.add_argument("limit", type=float)
    p_add.add_argument("--qty", default="full", help='share qty or "full" (default)')
    p_rm = sub.add_parser("remove", help="cancel open order + drop target")
    p_rm.add_argument("symbol")
    sub.add_parser("list", help="list configured targets")
    args = ap.parse_args(argv)
    cfg = ensure_config()
    if args.write_config:
        return 0
    if args.cmd == "add":
        e = add_target(args.symbol, args.limit, qty=args.qty, cfg=cfg)
        print(f"added {args.symbol.upper()}: {json.dumps(e)}")
        return 0
    if args.cmd == "remove":
        print(remove_target_cli(args.symbol))
        return 0
    if args.cmd == "list":
        rows = list_targets(cfg)
        if not rows:
            print("(no targets)")
        for r in rows:
            print(f"{r['symbol']:6} limit={r['limit']} qty={r['qty']} added={r['added']} "
                  f"order={r['order_id'] or '-'} day={r['day'] or '-'}")
        return 0
    if args.status:
        print(json.dumps({"config": cfg, "state": _load_state(),
                          "sold_block": sold_block.active_blocks()}, indent=2))
        return 0
    dry = True if args.dry_run else None
    if args.scheduled and not args.dry_run:
        dry = None
    run_check(dry_run=dry, cfg=cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
