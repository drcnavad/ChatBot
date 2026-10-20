"""Manual orders on the LIVE Alpaca account, outside the pipeline (Chirag, Oct 16, 2026).

  python manual_trade.py sell KO [--block-until-drop 10] [--no-replace] [--dry-run]   # sell the whole position now
  python manual_trade.py buy XYZ [--amount 1000] [--dry-run]           # buy $1000 now (default: 5% of equity = one slot)
  python manual_trade.py blocks                                        # list the active buy blocks

Orders go out at once as DAY limit orders in regular hours, priced like the pipeline (paper_trade.smart_quote: buy at the
ask + 0.05%, sell at the bid - 0.05%, skipped on a stale / wide quote); outside market hours it refuses. Every action
(also a dry run or a refusal) is appended to Reports/manual_orders.json.

Sell + replace (Chirag, Oct 16): once the sell FILLS (waits up to FILL_WAIT_SECS), the freed cash buys the next-in-line
stock of the same half (low volatility: best Strategy-ranked low_vol stock; sector rotation: best sector_rot stock in the
top-3 sectors) that is not held, not blocked, not do-not-buy and not stopped out - a DAY limit at min(quote limit, ask),
never above the ask - and the stock joins that half of the ledger. --no-replace keeps the cash. An unfilled sell buys
nothing (logged).
Sell: the symbol leaves the sleeve ledger (Reports/sleeve_ledger.json), so the pipeline neither rebuys nor "repairs" it.
--block-until-drop N adds Reports/manual_block.json[SYMBOL] = {reference_close, block_until_price = close x (1 - N%),
added}: the reference close is that day's close, filled in by the after-close step (run_all.daily_step ->
`manual_trade.py update-blocks`); the pipeline never buys a blocked symbol (paper_trade.manual_list("buy")), and the
after-close step removes the block once the close is at or below block_until_price.
Buy: the position belongs to no half of the ledger, so the next pipeline trade run sells it (sleeves.plan) - a manual
buy is for between rebalances; say so in the log."""
import argparse
import json
import math
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
ORDERS_JSON = os.environ.get("STOCK_ANALYSIS_MANUAL_ORDERS") or os.path.join(REPORTS, "manual_orders.json")
BLOCK_JSON = os.environ.get("STOCK_ANALYSIS_MANUAL_BLOCK") or os.path.join(REPORTS, "manual_block.json")
FILL_WAIT_SECS = 30        # a manual sell is polled this long for its fill before the replacement buy
DEFAULT_PCT = 0.05          # buy size without --amount: 5% of equity (a pipeline slot: 10 stocks per half x 50%, 99%)
CT = ZoneInfo("America/Chicago")

def _now():
    return datetime.now(CT)

def _read(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)

def _write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)

def log(entry, path=None):
    path = path or ORDERS_JSON
    rows = _read(path, [])
    rows.append({"time": _now().strftime("%Y-%m-%d %H:%M:%S CT"), **entry})
    _write(path, rows)
    return rows[-1]

def load_blocks(path=None):
    """{SYMBOL: {reference_close, block_until_price, drop_pct, added}} (reference_close None until the after-close step)."""
    return _read(path or BLOCK_JSON, {})

def add_block(sym, drop_pct, day, path=None):
    b = load_blocks(path)
    b[sym] = {"reference_close": None, "block_until_price": None, "drop_pct": float(drop_pct), "added": str(day)}
    _write(path or BLOCK_JSON, b)
    return b[sym]

def update_blocks(closes, path=None):
    """After-close step: closes = {SYMBOL: [(date, close), ...]} official daily closes, oldest first. A new block gets the
    close of the day it was added (or the first session after) as reference close and block_until_price = close x
    (1 - drop%); an active block is removed once the latest close is at or below block_until_price.
    Returns (set, removed) symbol lists."""
    b, done, gone = load_blocks(path), [], []
    for sym, e in list(b.items()):
        bars = closes.get(sym) or []
        if not bars:
            continue
        if e.get("reference_close") is None:
            ref = [c for d, c in bars if str(d) >= e["added"]]
            if ref:
                e["reference_close"] = round(float(ref[0]), 4)
                e["block_until_price"] = round(float(ref[0]) * (1 - e.get("drop_pct", 10.0) / 100), 4)
                done.append(sym)
        elif float(bars[-1][1]) <= e["block_until_price"]:
            del b[sym]
            gone.append(sym)
    _write(path or BLOCK_JSON, b)
    return done, gone

def blocked_notes(path=None):
    """{SYMBOL: note} for paper_trade.manual_list('buy'): every active block (also one still waiting for its close)."""
    out = {}
    for s, e in load_blocks(path).items():
        until = e.get("block_until_price")
        out[s] = f"manual block until ${until:,.2f}" if until else "manual block (reference close at today's close)"
    return out

def drop_from_ledger(sym, ledger=None, proceeds=None):
    """Remove `sym` from both halves of the sleeve ledger (shares and prices); the sale proceeds (default shares x ledger
    price) become that half's cash. Returns the half it was in, or None."""
    import sleeves
    led = sleeves.load_ledger() if ledger is None else ledger
    if led is None:
        return None
    was = None
    for h in sleeves.HALVES:
        L = led["halves"][h]
        if sym in (L.get("shares") or {}):
            was = h
            q = float(L["shares"].pop(sym))
            px = (L.get("prices") or {}).pop(sym, None)
            L["cash"] = round(float(L.get("cash", 0.0)) + (proceeds if proceeds is not None else q * float(px or 0)), 2)
    if was and ledger is None:
        sleeves.save_ledger(led)
    return was

def next_in_line(half, exclude, rankings=None):
    """The best-ranked stock of `half` ('lowvol' / 'sector') not in `exclude` (held, blocked, do-not-buy, ...), or None.
    rankings: {half: [symbols best first]} (tests); default the latest close's rankings (dashboard.stock_groups)."""
    if rankings is None:
        from dashboard.stock_groups import _rankings
        _lv_t, _sec_t, lv, sr = _rankings()
        rankings = {"lowvol": lv, "sector": sr}
    return next((s for s in rankings.get(half, []) if s.upper() not in exclude), None)

LEFTOVER_MIN = 500.0        # leftover sleeve cash under max($500, 2% of the half) stays cash

def queue_leftover_cash(dry_run=False, rankings=None, prices=None, positions=None, pending_path=None, min_cash=LEFTOVER_MIN):
    """Standing rule (Chirag, Oct 19, 2026): after a rebalance, a half's leftover cash (e.g. a blocked target such as KO
    under a manual block) buys that half's next-in-line stock, like a manual sell's replacement: one BUY row per half in
    Reports/live_pending_orders.json (staged, at_ask = the limit is never above the ask) that the 9 AM CT fill check sends,
    and the shares go into that half of the sleeve ledger. Only halves that rebalanced on the ledger's day; never twice
    (a queued leftover buy for the half, or any queued BUY of the stock, skips it). Places no order itself.
    Returns [{half, symbol, qty, limit, cash, status}]."""
    import json
    import paper_trade as pt
    import sector_mapping as sm
    import sleeves
    pending_path = pending_path or pt.PENDING_ORDERS_JSON
    led = sleeves.load_ledger()
    if led is None:
        return []
    try:
        pend = json.load(open(pending_path)).get("orders") or []
    except (OSError, ValueError, AttributeError):
        pend = []
    if positions is None:
        import alpaca_paper as ap
        positions = [p["symbol"] for p in ap.PaperAccount().position_dicts()]
    queued = {str(o.get("symbol", "")).upper() for o in pend if str(o.get("side", "")).upper() == "BUY"}
    in_led = {s for h in sleeves.HALVES for s in (led["halves"][h].get("shares") or {})}
    stop = pt.earnings_stop_blocked(_now().date().isoformat())
    excl = ({str(s).upper() for s in positions} | in_led | queued | set(pt.manual_list("buy")) | set(stop)
            | {s for s in sm.symbol_sector if s not in set(sm.tradable_symbols)})
    out = []
    for h in sleeves.HALVES:
        L = led["halves"][h]
        cash, value = float(L.get("cash", 0.0)), float(L.get("value", 0.0) or 0.0)
        row = {"half": h, "cash": round(cash, 2)}
        if not L.get("rebalanced") or cash < max(min_cash, 0.02 * value):
            continue
        if any(o.get("source") == "leftover-cash" and o.get("half") == h for o in pend):
            out.append({**row, "status": "already queued (not doubled)"})
            continue
        lv_picks = set()
        if h == "sector":                                # no overlap: the low-volatility half's picks stay its own
            try:
                import pandas as pd
                p = pd.read_csv(pt.PICKS_CSV)
                lv_picks = set(p.loc[pd.to_numeric(p["Provisional_Weight"], errors="coerce").fillna(0) > 0, "Symbol"].str.upper())
            except (OSError, KeyError, ValueError):
                pass
        sym = next_in_line(h, excl | lv_picks, rankings)
        if not sym:
            out.append({**row, "status": "no next-in-line stock (cash stays)"})
            continue
        px = float((prices or {}).get(sym) or ((latest_closes([sym]).get(sym) or [(None, 0)])[-1][1]) or 0)
        if px <= 0:
            out.append({**row, "symbol": sym, "status": "no price (cash stays)"})
            continue
        limit = round(px * 1.005, 2)                     # ceiling for sizing; the 9 AM limit is the live ask or lower
        qty = math.floor(cash * 0.99 / limit * 100) / 100
        row = {**row, "symbol": sym, "qty": qty, "limit": limit}
        if qty <= 0:
            out.append({**row, "status": "buys no share"})
            continue
        excl.add(sym)
        if dry_run:
            out.append({**row, "status": "dry run (nothing queued)"})
            continue
        pt.record_pending_order({"symbol": sym, "side": "BUY", "qty": qty, "limit_price": limit, "order_id": None,
                                 "at_ask": True, "source": "leftover-cash", "half": h},
                                {"source": "leftover-cash"}, pending_path)
        L.setdefault("shares", {})[sym] = qty
        L.setdefault("prices", {})[sym] = px
        L["cash"] = round(cash - qty * px, 2)
        sleeves.save_ledger(led)
        out.append({**row, "status": "queued for the 9 AM CT fill check"})
        log({"action": "buy (leftover cash)", "half": h, "symbol": sym, "qty": qty, "limit": limit, "dry_run": False,
             "status": "queued for the 9 AM CT fill check"})
    return out

def _wait_fill(client, order_id, secs=None, sleep=None):
    """(filled qty, avg price) once the order is filled, else None after `secs`."""
    import time
    secs, sleep = FILL_WAIT_SECS if secs is None else secs, sleep or time.sleep
    for _ in range(max(1, int(secs / 2))):
        o = client.get_order_by_id(order_id)
        st = str(getattr(o.status, "value", o.status)).lower()
        if st == "filled":
            return float(o.filled_qty), float(o.filled_avg_price)
        if st in ("canceled", "cancelled", "expired", "rejected"):
            return None
        sleep(2)
    return None

def replace_buy(half, cash, client, quote, rankings=None, sold=None):
    """Buy the next-in-line stock of `half` with `cash` (limit <= ask), add it to that half of the ledger. Returns the log entry."""
    import paper_trade as pt
    import sleeves
    held = {str(p.symbol).upper() for p in client.get_all_positions()}
    led = sleeves.load_ledger()
    in_led = {s for h in sleeves.HALVES for s in (led["halves"][h].get("shares") or {})} if led else set()
    stop = pt.earnings_stop_blocked(_now().date().isoformat())
    excl = held | in_led | set(pt.manual_list("buy")) | set(stop) | {str(sold or "").upper()}
    sym = next_in_line(half, excl, rankings)
    base = {"action": "buy (replacement)", "replaces": sold, "half": half, "dry_run": False}
    if not sym:
        return log({**base, "symbol": None, "status": "no replacement: no next-in-line stock left (cash stays in the half)"})
    q = quote(sym, "BUY")
    if q.get("skip") or not q.get("limit") or not q.get("ask"):
        return log({**base, "symbol": sym, "status": f"no replacement: {q.get('skip') or 'no quote'} (cash stays in the half)"})
    limit = round(min(float(q["limit"]), float(q["ask"])), 2)
    qty = math.floor(cash / limit * 100) / 100
    if qty <= 0:
        return log({**base, "symbol": sym, "status": f"no replacement: ${cash:,.2f} buys no share"})
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest
    try:
        o = client.submit_order(LimitOrderRequest(symbol=sym, qty=qty, limit_price=limit, time_in_force=TimeInForce.DAY,
                                                  side=OrderSide.BUY, client_order_id=f"manual-{_now():%Y%m%d%H%M%S}-BUY-{sym}"[:48]))
    except Exception as e:
        return log({**base, "symbol": sym, "qty": qty, "limit": limit, "status": f"FAILED: {str(e)[:200]}"})
    if led:
        L = led["halves"][half]
        L.setdefault("shares", {})[sym] = qty
        L.setdefault("prices", {})[sym] = limit
        L["cash"] = round(float(L.get("cash", 0.0)) - qty * limit, 2)
        sleeves.save_ledger(led)
    return log({**base, "symbol": sym, "qty": qty, "limit": limit, "bid": q.get("bid"), "ask": q.get("ask"),
                "value": round(qty * limit, 2), "order_id": str(getattr(o, "id", "")),
                "status": f"submitted ({getattr(o.status, 'value', o.status)})", "ledger_half_added": half})

def _market_open(client):
    try:
        return bool(client.get_clock().is_open)
    except Exception as e:
        raise SystemExit(f"Could not read the market clock ({type(e).__name__}): nothing sent.") from None

def run(side, sym, amount=None, block_drop=None, dry_run=False, client=None, quote=None, replace=True, rankings=None,
        wait=None):
    """Place (or with dry_run only plan) one manual order. Returns the logged entry."""
    import paper_trade as pt
    sym = sym.upper()
    client = client or pt.paper_trading_client()
    base = {"action": side.lower(), "symbol": sym, "dry_run": bool(dry_run)}
    if not _market_open(client):
        return log({**base, "status": "refused: market closed (regular hours only, 8:30 AM-3:00 PM CT)"})
    if side == "BUY" and sym in blocked_notes():
        return log({**base, "status": f"refused: {blocked_notes()[sym]}"})
    q_fn = quote or pt.smart_quote
    q = q_fn(sym, side)
    if q.get("skip") or not q.get("limit"):
        return log({**base, "status": f"refused: {q.get('skip') or 'no limit price'}"})
    limit = float(q["limit"])
    if side == "SELL":
        pos = {str(p.symbol).upper(): float(p.qty) for p in client.get_all_positions()}
        qty = pos.get(sym, 0.0)
        if qty <= 0:
            return log({**base, "status": "refused: not held"})
    else:
        equity = float(client.get_account().equity)
        dollars = float(amount) if amount else DEFAULT_PCT * equity
        qty = math.floor(dollars / limit * 100) / 100                 # 2-decimal shares (regular hours)
        if qty <= 0:
            return log({**base, "status": f"refused: ${dollars:,.2f} buys less than 0.01 share at ${limit:,.2f}"})
    entry = {**base, "qty": qty, "limit": limit, "bid": q.get("bid"), "ask": q.get("ask"), "value": round(qty * limit, 2)}
    if not dry_run:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        req = LimitOrderRequest(symbol=sym, qty=qty, limit_price=limit, time_in_force=TimeInForce.DAY,
                                side=OrderSide.SELL if side == "SELL" else OrderSide.BUY,
                                client_order_id=f"manual-{_now():%Y%m%d%H%M%S}-{side}-{sym}"[:48])
        try:
            o = client.submit_order(req)
            entry.update(status=f"submitted ({getattr(o.status, 'value', o.status)})", order_id=str(getattr(o, "id", "")))
        except Exception as e:
            return log({**entry, "status": f"FAILED: {str(e)[:200]}"})
    else:
        entry["status"] = "dry run: not sent"
    fill = None
    if side == "SELL" and not dry_run and entry.get("order_id") is not None:
        fill = (wait or _wait_fill)(client, entry["order_id"])
        if fill:
            entry.update(status="filled", filled_qty=fill[0], fill_price=fill[1])
    if side == "SELL" and not dry_run:
        proceeds = round(fill[0] * fill[1], 2) if fill else round(qty * limit, 2)
        entry["ledger_half_removed"] = drop_from_ledger(sym, proceeds=proceeds)
    if side == "SELL" and block_drop:
        entry["block"] = add_block(sym, block_drop, _now().date().isoformat()) if not dry_run else \
            {"drop_pct": float(block_drop), "note": "dry run: not saved"}
    if side == "BUY":
        entry["note"] = "in no half of the ledger: the next pipeline trade run sells it"
    out = log(entry)
    if side == "SELL" and not dry_run and replace:
        half = entry.get("ledger_half_removed")
        if not fill:
            log({"action": "buy (replacement)", "replaces": sym, "symbol": None, "dry_run": False,
                 "status": f"no replacement: the sell was not filled within {FILL_WAIT_SECS}s (buy by hand if wanted)"})
        elif not half:
            log({"action": "buy (replacement)", "replaces": sym, "symbol": None, "dry_run": False,
                 "status": "no replacement: the sold stock was in no half of the ledger"})
        else:
            out = {**out, "replacement": replace_buy(half, round(fill[0] * fill[1], 2), client, q_fn, rankings, sold=sym)}
    return out

def latest_closes(symbols, now=None):
    """{SYMBOL: [(date, close), ...]}: official daily closes of the last 10 days (yfinance), oldest first. Today's bar
    counts only from 3:15 PM CT (the close is final); before that it is an intraday price and is dropped."""
    import yfinance as yf
    now = now or _now()
    out = {}
    for s in symbols:
        try:
            h = yf.Ticker(s).history(period="10d")["Close"].dropna()
        except Exception as e:
            print(f"  {s}: no closes ({type(e).__name__}) - block left as is")
            continue
        bars = [(d.strftime("%Y-%m-%d"), float(c)) for d, c in h.items()]
        today = now.strftime("%Y-%m-%d")
        out[s] = [b for b in bars if b[0] < today or (now.hour, now.minute) >= (15, 15)]
    return out

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["buy", "sell", "blocks", "update-blocks"])
    ap.add_argument("symbol", nargs="?")
    ap.add_argument("--amount", type=float, help=f"buy: dollars (default {DEFAULT_PCT:.0%} of equity)")
    ap.add_argument("--block-until-drop", type=float, metavar="PCT", help="sell: no pipeline buy until the price is PCT%% "
                    "below that day's close")
    ap.add_argument("--dry-run", action="store_true", help="plan and log, send nothing")
    ap.add_argument("--no-replace", action="store_true", help="sell: keep the cash (no next-in-line buy)")
    a = ap.parse_args(argv)
    if a.action == "blocks":
        print(json.dumps(load_blocks(), indent=1) if load_blocks() else "no active blocks")
        return 0
    if a.action == "update-blocks":
        b = load_blocks()
        done, gone = update_blocks(latest_closes(list(b))) if b else ([], [])
        print(f"manual blocks: reference close set {done or '-'}, removed (price reached) {gone or '-'}")
        return 0
    if not a.symbol:
        ap.error("a symbol is required")
    e = run(a.action.upper(), a.symbol, a.amount, a.block_until_drop, a.dry_run, replace=not a.no_replace)
    print(json.dumps(e, indent=1))
    return 0 if not str(e.get("status", "")).startswith(("refused", "FAILED")) else 1

if __name__ == "__main__":
    sys.exit(main())
