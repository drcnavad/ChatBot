"""Manual orders on the LIVE Alpaca account, outside the pipeline (Chirag, Oct 16, 2026).

  python manual_trade.py sell KO [--block-until-drop 10] [--dry-run]   # sell the whole position now
  python manual_trade.py buy XYZ [--amount 1000] [--dry-run]           # buy $1000 now (default: 5% of equity = one slot)
  python manual_trade.py blocks                                        # list the active buy blocks

Orders go out at once as DAY limit orders in regular hours, priced like the pipeline (paper_trade.smart_quote: buy at the
ask + 0.05%, sell at the bid - 0.05%, skipped on a stale / wide quote); outside market hours it refuses. Every action
(also a dry run or a refusal) is appended to Reports/manual_orders.json.

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

def _market_open(client):
    try:
        return bool(client.get_clock().is_open)
    except Exception as e:
        raise SystemExit(f"Could not read the market clock ({type(e).__name__}): nothing sent.") from None

def run(side, sym, amount=None, block_drop=None, dry_run=False, client=None, quote=None):
    """Place (or with dry_run only plan) one manual order. Returns the logged entry."""
    import paper_trade as pt
    sym = sym.upper()
    client = client or pt.paper_trading_client()
    base = {"action": side.lower(), "symbol": sym, "dry_run": bool(dry_run)}
    if not _market_open(client):
        return log({**base, "status": "refused: market closed (regular hours only, 8:30 AM-3:00 PM CT)"})
    if side == "BUY" and sym in blocked_notes():
        return log({**base, "status": f"refused: {blocked_notes()[sym]}"})
    q = (quote or pt.smart_quote)(sym, side)
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
    if side == "SELL" and not dry_run:
        entry["ledger_half_removed"] = drop_from_ledger(sym, proceeds=round(qty * limit, 2))
    if side == "SELL" and block_drop:
        entry["block"] = add_block(sym, block_drop, _now().date().isoformat()) if not dry_run else \
            {"drop_pct": float(block_drop), "note": "dry run: not saved"}
    if side == "BUY":
        entry["note"] = "in no half of the ledger: the next pipeline trade run sells it"
    return log(entry)

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
    e = run(a.action.upper(), a.symbol, a.amount, a.block_until_drop, a.dry_run)
    print(json.dumps(e, indent=1))
    return 0 if not str(e.get("status", "")).startswith(("refused", "FAILED")) else 1

if __name__ == "__main__":
    sys.exit(main())
