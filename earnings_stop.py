"""Pre-earnings ATR stop for the LIVE account (approved by Chirag, Sat Oct 3, 2026, 1:07 AM CT).

Rule: a stock the live account holds whose next earnings date (Reports/earnings_date.csv) is within the next 7 calendar
days gets a stop = its highest daily close since entry - 3 x ATR(14) (Wilder, as in the engine). The stop is active
from the day the earnings date enters the 7-day window through the first reaction day (an AM report reacts that day;
an after-close or unknown-time report reacts the next trading day). Entry = the oldest buy fill still held (FIFO).

Alpaca trailing stops do not work outside regular hours, so this job checks prices itself: launchd runs it every
10 minutes; it acts on NYSE trading days in the pre-market (4:00-9:30 AM ET), regular hours and after hours (until
8 PM ET, 5 PM ET on early-close days). Not overnight: the SIP/IEX quotes stop updating at 8 PM ET.
Quotes: the same SIP -> IEX fallback as the smart limit prices (paper_trade._latest_quote). The stop fires when the
quote's mid price is at or below the stop (a stale, one-sided or too-wide quote waits for the next check).
At the stop: SELL with a limit at the bid - 0.05% (time in force DAY): in regular hours every share held, fraction
included (nothing is left for later); outside regular hours the whole shares (extended_hours) and the fractional rest
goes to the 9 AM CT fill check. Any unfilled part is a pending row for that check too. A stock sold by the stop is not
bought back by any live run until after its reaction day (paper_trade.earnings_stop_blocked reads the state file).
Never twice: one sale per (stock, earnings date), kept in Reports/earnings_stop_state.json, and a fixed client order id
(live-stop-YYYYMMDD-SYMBOL) that is looked up on the broker before sending. Any read failure skips the sale (fails closed).
The cash stays in cash until the next scheduled run, which rebalances normally (its no-buy-before-earnings rule still
applies). Events go to Reports/run_log.csv; the dashboard reads Reports/earnings_stops.csv (latest check).

Usage:
  python earnings_stop.py --dry-run     # read-only: held stocks in the window, their stops and quotes; sends and writes nothing
  python earnings_stop.py --scheduled   # launchd job com.stockanalysis.earningsstop (LIVE: sells at the stop)
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import backtest_engine as be

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
EARNINGS_CSV = os.path.join(REPORTS, "earnings_date.csv")
STATE_JSON = (os.environ.get("STOCK_ANALYSIS_EARNINGS_STOP_STATE")    # sales done (+ reaction day: the buy-back block
              or os.path.join(REPORTS, "earnings_stop_state.json"))     # in paper_trade) + once-a-day log notes
STATUS_CSV = os.path.join(REPORTS, "earnings_stops.csv")           # the latest check, for the dashboard
K_ATR, ATR_LEN, ARM_DAYS = 3.0, 14, 7     # 3x ATR since Oct 3, 2026 (was 3.5x; the forward test keeps 3.5x)
REGULAR_MAX_AGE, EXTENDED_MAX_AGE = 60, 900                         # quote age limits (seconds)
REGULAR_MAX_SPREAD, EXTENDED_MAX_SPREAD = 0.005, 0.02               # spread limits (share of the mid price)
STATUS_COLS = ["Checked_At_CT", "Symbol", "Shares", "Entry_Date", "Earnings_Date", "Report_Time", "Reaction_Day",
               "Peak_Close", "ATR14", "Stop", "Price", "Vs_Stop_%", "Status"]
RUN = "Earnings stop"


# ----------------------------------------------------------------------------- the rule (pure)
def session_now(now):
    """'pre' / 'regular' / 'after' when `now` is in that Alpaca session of an NYSE trading day, else None."""
    et = now.astimezone(be.EASTERN)
    if not be.is_session(et.date()):
        return None
    close = 13 * 60 if be.is_early_close(et.date()) else 16 * 60
    m = et.hour * 60 + et.minute
    if 4 * 60 <= m < 9 * 60 + 30:
        return "pre"
    if 9 * 60 + 30 <= m < close:
        return "regular"
    if close <= m < close + 4 * 60:
        return "after"
    return None


def reaction_day(e_date, report_time):
    """First trading day that reacts to the report: the report day for an AM report on a trading day, otherwise the
    next trading day after it (PM or unknown time)."""
    e = pd.Timestamp(e_date).normalize()
    if str(report_time).strip().upper() == "AM" and be.is_session(e):
        return e
    return be.next_sessions(e, 1)[0]


def active_event(sym, earnings, today):
    """(earnings date, time, reaction day) of the report whose stop window holds `today`: from 7 calendar days before
    the earnings date through its reaction day. None when no window is open. `earnings` = be.load_earnings()."""
    today = pd.Timestamp(today).normalize()
    rows = earnings[earnings["Symbol"] == sym].sort_values("Earnings Date")
    for _, r in rows.iterrows():
        e = r["Earnings Date"]
        t = str(r["Time"]).strip().upper() if "Time" in r and pd.notna(r["Time"]) else ""
        react = reaction_day(e, t)
        if e - pd.Timedelta(days=ARM_DAYS) <= today <= react:
            return e, t, react
    return None


def atr_wilder(bars, n=ATR_LEN):
    """ATR(14), Wilder smoothing, exactly as be.calculate_technical_indicators."""
    prev = bars["Close"].shift(1)
    tr = pd.concat([bars["High"] - bars["Low"], (bars["High"] - prev).abs(), (bars["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def stop_level(bars, entry_date):
    """(peak close since entry, ATR(14), stop) from one stock's completed daily bars (Date, High, Low, Close; oldest
    first). None when there are too few bars or no close since entry."""
    bars = bars.sort_values("Date").reset_index(drop=True)
    if len(bars) <= ATR_LEN:
        return None
    atr = float(atr_wilder(bars).iloc[-1])
    since = bars.loc[bars["Date"] >= pd.Timestamp(entry_date).normalize(), "Close"]
    if since.empty or not np.isfinite(atr):
        return None
    peak = float(since.max())
    return peak, atr, peak - K_ATR * atr


def quote_check(q, session, now):
    """(mid, skip reason) for a (bid, ask, time, feed) quote."""
    bid, ask, ts, _ = q
    if not (bid > 0 and ask > 0 and ask >= bid):
        return None, "no two-sided quote"
    mid = (bid + ask) / 2
    age = (now - ts).total_seconds() if ts is not None else float("inf")
    max_age, max_spread = (REGULAR_MAX_AGE, REGULAR_MAX_SPREAD) if session == "regular" else (EXTENDED_MAX_AGE, EXTENDED_MAX_SPREAD)
    if age > max_age:
        return mid, f"quote is {age:.0f}s old"
    if (ask - bid) / mid > max_spread:
        return mid, f"spread {(ask - bid) / mid:.2%} too wide"
    return mid, None


def stop_cid(sym, e_date, attempt=1):
    """Fixed client order id of the stop sale for one report (a retry after a broker rejection adds -rN)."""
    return f"live-stop-{pd.Timestamp(e_date):%Y%m%d}-{sym}" + (f"-r{attempt}" if attempt > 1 else "")


# ----------------------------------------------------------------------------- inputs (tests replace these)
def account():
    import alpaca_paper as ap
    return ap.PaperAccount()


def trading_client():
    import paper_trade as pt
    return pt.paper_trading_client()


def latest_quote(sym):
    import paper_trade as pt
    return pt._latest_quote(sym)


def daily_bars(symbols, now):
    """Completed daily bars (today's bar only after 4:30 PM ET, when the pipeline treats it as final)."""
    import run_all
    et = now.astimezone(be.EASTERN)
    bars = be.fetch_daily_bars(symbols, start=(et - timedelta(days=400)).date())
    if (et.hour, et.minute) < run_all.BAR_FINAL_ET:
        bars = bars[bars["Date"] < pd.Timestamp(et.date())]
    return bars


def log_event(status, money, message, details=""):
    import run_all
    run_all.log_event(RUN, status, money, message, details)


# ----------------------------------------------------------------------------- state
def _load_state(path=None):
    try:
        with open(path or STATE_JSON) as f:
            s = json.load(f)
    except (OSError, ValueError):
        s = {}
    for k in ("sold", "notes", "armed"):
        s.setdefault(k, {})
    return s


def _save_state(s, path=None):
    path = path or STATE_JSON
    today = datetime.now(be.CENTRAL).date()
    s["notes"] = {k: v for k, v in s["notes"].items() if v >= str(today - timedelta(days=3))}   # keep a few days
    s["armed"] = {k: v for k, v in s["armed"].items() if k.split("|")[1] >= str(today - timedelta(days=30))}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, path)


def _once(state, key, today, status, money, message, details=""):
    """Log an event once per day per key (a failing check every 10 minutes writes one row)."""
    if state["notes"].get(key) != str(today):
        state["notes"][key] = str(today)
        log_event(status, money, message, details)


def _lock_busy(path):
    """True when a live process holds the pipeline lock at `path` (a stale or missing lock is not busy)."""
    try:
        with open(path) as f:
            pid = int(json.load(f).get("pid", 0))
        os.kill(pid, 0)
        return pid > 0
    except (OSError, ValueError, TypeError):
        return False


def _append_pending(row, path):
    """Add the stop sale to the morning fill check's pending file (the file's own dates and other rows are kept). A new
    file opens at the next 9 AM CT check: a pre-market sale (before 9 AM CT) is dated the previous trading day, so that
    morning's check takes it."""
    try:
        with open(path) as f:
            pend = json.load(f)
    except FileNotFoundError:
        pend = None
    if not isinstance(pend, dict) or not pend.get("orders"):
        t = datetime.fromisoformat(row["recorded_at"])
        day = pd.Timestamp(t.date()) - be.NYSE_SESSION if t.hour < 9 else pd.Timestamp(t.date())
        pend = {"evening_date": f"{day:%Y-%m-%d}", "submitted_at_ct": row["recorded_at"], "target_source": "auto",
                "as_of": None, "orders": []}
    pend["orders"] = [o for o in pend["orders"] if not (o.get("symbol") == row["symbol"] and o.get("source") == "earnings-stop")] + [row]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(pend, f, indent=2)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- one check
def candidates(now, acct=None, earnings=None, quick=True):
    """[{symbol, shares, entry, e_date, e_time, react, peak, atr, stop}] for held stocks in a stop window, plus
    [(symbol, reason)] held stocks in a window without a usable stop. Reads only."""
    import alpaca_paper as ap
    today = now.astimezone(be.EASTERN).date()
    earnings = be.load_earnings(EARNINGS_CSV) if earnings is None else earnings
    soon = earnings[(earnings["Earnings Date"] >= pd.Timestamp(today) - pd.Timedelta(days=5))
                    & (earnings["Earnings Date"] <= pd.Timestamp(today) + pd.Timedelta(days=ARM_DAYS))]
    if quick and soon.empty:
        return [], [], {}                       # no report near: no broker call at all
    acct = acct or account()
    held = {str(p["symbol"]).upper(): float(p.get("qty") or 0) for p in acct.position_dicts()}
    events = {s: ev for s in held if held[s] > 0 and (ev := active_event(s, earnings, today))}
    if not events:
        return [], [], held
    entry = ap.first_buy_dates(acct.fills())
    bars = daily_bars(sorted(events), now)
    out, bad = [], []
    for s, (e, t, react) in sorted(events.items()):
        if s not in entry:
            bad.append((s, "no buy fill found (entry date unknown)"))
            continue
        lvl = stop_level(bars[bars["Symbol"] == s], entry[s])
        if lvl is None:
            bad.append((s, "not enough daily bars for ATR(14)"))
            continue
        out.append({"symbol": s, "shares": held[s], "entry": entry[s], "e_date": e, "e_time": t or "?", "react": react,
                    "peak": lvl[0], "atr": lvl[1], "stop": lvl[2]})
    return out, bad, held


def _sell(c, q, session, now, state, pending_path):
    """Send the stop sale for one candidate (all checks first). Returns a status text."""
    import paper_trade as pt
    import run_all
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest
    s, e, today = c["symbol"], c["e_date"], now.astimezone(be.CENTRAL).date()
    key = f"{s}|{e:%Y-%m-%d}"
    if _lock_busy(run_all.TRADE_LOCK):
        return "at stop - waiting for the running trade job"
    ok, why = run_all.acquire_trade_lock(run_all.FILL_LOCK)
    if not ok:
        return "at stop - waiting for the running fill check"
    try:
        client = trading_client()
        broker = pt._broker_orders_by_client_id(client)
        if broker is None:
            _once(state, f"broker|{key}", today, "failed", "no", f"{s} is at its pre-earnings stop but the broker's orders "
                  "could not be read, so nothing was sent (fails closed). Next check in 10 minutes.")
            return "at stop - broker unreadable, not sent"
        attempt = 1
        while (o := broker.get(stop_cid(s, e, attempt))) is not None:
            st = getattr(o.status, "value", str(o.status)).lower()
            if st != "rejected" or pt._safe_number(getattr(o, "filled_qty", 0)) > 0:
                state["sold"][key] = {"at": now.astimezone(be.CENTRAL).isoformat(timespec="seconds"), "order_id": str(o.id),
                                      "react": f"{c['react']:%Y-%m-%d}", "note": f"found on the broker ({st})"}
                return f"sold earlier ({st})"
            attempt += 1
        if attempt > 3:
            _once(state, f"rejected|{key}", today, "failed", "no", f"The broker rejected the {s} stop sale 3 times; "
                  "not sent again. Check Alpaca.")
            return "at stop - rejected 3 times"
        held_now, open_now = pt._symbol_state(client, s)
        if held_now is None:
            _once(state, f"read|{key}", today, "failed", "no", f"{s} is at its pre-earnings stop but the position could not "
                  "be read, so nothing was sent (fails closed). Next check in 10 minutes.")
            return "at stop - position unreadable, not sent"
        if open_now:
            _once(state, f"open|{key}", today, "warning", "no", f"{s} is at its pre-earnings stop but already has an open "
                  "order, so the stop waits (it never stacks a second order).")
            return "at stop - open order already working"
        if held_now <= 0:
            return "no longer held"
        ext = session != "regular"
        whole = math.floor(held_now + 1e-9) if ext else held_now   # regular hours: the exact holding, fraction included
        limit = pt._tick(q[0] * (1 - pt.LIMIT_OFFSET), False)
        oid, filled = None, 0.0
        if whole >= 1 or (not ext and whole > 0):
            req = LimitOrderRequest(symbol=s, qty=whole, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                                    limit_price=limit, extended_hours=ext, client_order_id=stop_cid(s, e, attempt))
            try:
                order = client.submit_order(req)
            except Exception as err:
                _once(state, f"submit|{key}", today, "failed", "unknown", f"The {s} pre-earnings stop sale failed to send "
                      f"({str(err)[:120]}). It is tried again at the next check.")
                return "at stop - send failed, retrying"
            oid, filled = str(order.id), pt._safe_number(getattr(order, "filled_qty", 0))
        stamp = now.astimezone(be.CENTRAL).isoformat(timespec="seconds")
        state["sold"][key] = {"at": stamp, "order_id": oid, "qty": whole, "limit": limit, "held": held_now,
                              "react": f"{c['react']:%Y-%m-%d}"}
        _save_state(state)
        frac = round(held_now - whole, 6)
        try:
            _append_pending({"symbol": s, "side": "SELL", "qty": held_now, "limit_price": limit, "order_id": oid,
                             "order_qty": whole, "exit": True, "recorded_at": stamp, "decision": None,
                             "evening_date": str(today), "source": "earnings-stop"}, pending_path)
        except (OSError, ValueError) as err:
            log_event("failed", "unknown", f"The {s} pre-earnings stop sale was sent, but its row for the 9 AM CT check "
                      f"could not be written ({str(err)[:100]}). Check Alpaca: any unfilled rest or fraction must be sold by hand.")
        sent = (f"Sent a sell order for {whole:g} {'whole ' if ext else ''}shares with {'a pre-market' if session == 'pre' else 'an after-hours' if ext else 'a regular-hours'} "
                f"limit at ${limit:,.2f} (bid ${q[0]:,.2f})" if whole else "No whole share to sell")
        log_event("ok", "yes" if filled > 0 else "unknown",
                  f"Pre-earnings stop: {s} fell to ${(q[0] + q[1]) / 2:,.2f}, at or below its stop ${c['stop']:,.2f} "
                  f"(highest close since entry ${c['peak']:,.2f} - {K_ATR:g} x ATR ${c['atr']:,.2f}). {sent}."
                  + (f" The {frac:g} fractional share goes to the 9 AM CT fill check." if frac > 0 else "")
                  + f" Earnings {e:%a %b %-d} ({c['e_time']}). The cash waits for the next scheduled run; {s} is not bought back "
                  f"until after {c['react']:%a %b %-d}.",
                  f"order {oid or 'none'}; client id {stop_cid(s, e, attempt)}; held {held_now:g}")
        return ((f"SELL sent: {whole:g} shares at limit ${limit:,.2f}" if whole else "at stop - no whole share")
                + (f" (+{frac:g} at 9 AM CT)" if frac > 0 else ""))
    finally:
        run_all.release_trade_lock(run_all.FILL_LOCK)


def _sell_safe(c, q, session, now, state, pending_path):
    """_sell; an unexpected error (e.g. no trading client) is logged once a day and retried at the next check."""
    import paper_trade as pt
    try:
        return _sell(c, q, session, now, state, pending_path or pt.PENDING_ORDERS_JSON)
    except (Exception, SystemExit) as err:     # paper_trading_client raises SystemExit when the keys are missing
        _once(state, f"error|{c['symbol']}", now.astimezone(be.CENTRAL).date(), "failed", "unknown",
              f"The {c['symbol']} pre-earnings stop check failed ({str(err)[:150]}). Check Alpaca; next check in 10 minutes.")
        return "at stop - error, retrying"


def run_check(now=None, dry_run=False, pending_path=None):
    """One check. Dry run: prints the window and stops, sends and writes nothing, never makes a trading client.
    Returns the status rows."""
    import paper_trade as pt
    now = now or datetime.now(be.EASTERN)
    session = session_now(now)
    if not dry_run and session is None:
        print(f"{now.astimezone(be.CENTRAL):%Y-%m-%d %H:%M} CT: outside the pre-market/regular/after-hours sessions - idle")
        return []
    state = _load_state()
    today = now.astimezone(be.CENTRAL).date()
    try:
        cands, bad, held = candidates(now, quick=not dry_run)
    except Exception as err:
        if dry_run:
            raise
        _once(state, "inputs", today, "failed", "no", f"The pre-earnings stop check could not read its inputs ({str(err)[:150]}). "
              "No stop was checked; next check in 10 minutes.")
        _save_state(state)
        return []
    if not dry_run:
        for s, why in bad:
            _once(state, f"bad|{s}", today, "warning", "no", f"{s} has earnings within {ARM_DAYS} days but no pre-earnings stop: {why}.")
    rows, stamp = [], now.astimezone(be.CENTRAL).strftime("%Y-%m-%d %H:%M")
    for c in cands:
        s, key = c["symbol"], f"{c['symbol']}|{c['e_date']:%Y-%m-%d}"
        mid, status = float("nan"), "watching"
        try:
            q = latest_quote(s)
            m, skip = quote_check(q, session or "regular", now)
            mid = m if m is not None else float("nan")
        except Exception as err:
            q, skip = None, f"no quote ({str(err)[:60]})"
        if key in state["sold"]:
            status = f"sold {state['sold'][key]['at'][:16].replace('T', ' ')}"
        elif skip and not dry_run:
            status = f"watching (quote skipped: {skip})"
        elif q is not None and mid <= c["stop"]:
            whole = c["shares"] if (session or "regular") == "regular" else math.floor(c["shares"] + 1e-9)
            status = (f"AT STOP - would sell {whole:g} shares at ${pt._tick(q[0] * (1 - pt.LIMIT_OFFSET), False):,.2f}"
                      + (f" (quote note: {skip})" if skip else "")) if dry_run else _sell_safe(c, q, session, now, state, pending_path)
        elif dry_run and skip:
            status = f"watching (quote note: {skip})"
        if not dry_run and key not in state["armed"] and key not in state["sold"]:
            state["armed"][key] = str(today)    # first time in the window: one info row
            log_event("ok", "no", f"Pre-earnings stop armed for {s}: stop ${c['stop']:,.2f} (highest close since "
                      f"{c['entry']:%b %-d} ${c['peak']:,.2f} - {K_ATR:g} x ATR ${c['atr']:,.2f}); earnings {c['e_date']:%a %b %-d} "
                      f"({c['e_time']}), active through {c['react']:%a %b %-d}.")
        rows.append({"Checked_At_CT": stamp, "Symbol": s, "Shares": c["shares"], "Entry_Date": str(c["entry"]),
                     "Earnings_Date": f"{c['e_date']:%Y-%m-%d}", "Report_Time": c["e_time"], "Reaction_Day": f"{c['react']:%Y-%m-%d}",
                     "Peak_Close": round(c["peak"], 2), "ATR14": round(c["atr"], 2), "Stop": round(c["stop"], 2),
                     "Price": round(mid, 2), "Vs_Stop_%": round((mid / c["stop"] - 1) * 100, 2) if c["stop"] > 0 else float("nan"),
                     "Status": status})
    for s, why in bad:
        rows.append({"Checked_At_CT": stamp, "Symbol": s, "Shares": held.get(s), "Status": f"no stop: {why}"})
    table = pd.DataFrame(rows, columns=STATUS_COLS)
    if dry_run:
        _print_dry(now, session, table, held)
        return rows
    tmp = STATUS_CSV + ".tmp"
    table.to_csv(tmp, index=False)
    os.replace(tmp, STATUS_CSV)
    _save_state(state)
    print(f"{stamp} CT ({session}): {len(cands)} held stock(s) in a pre-earnings stop window" +
          "".join(f"\n  {r['Symbol']}: stop {r['Stop']} price {r['Price']} -> {r['Status']}" for r in rows))
    return rows


def _print_dry(now, session, table, held):
    print(f"DRY RUN {now.astimezone(be.CENTRAL):%a %b %-d %Y %H:%M} CT - nothing is sent or written. Session now: {session or 'closed'}.")
    print(f"Rule: held stock with earnings within {ARM_DAYS} calendar days -> stop = highest close since entry - "
          f"{K_ATR:g} x ATR({ATR_LEN}), active through the reaction day. Held: {len(held)} stocks.")
    if table.empty:
        print("No held stock is in a pre-earnings stop window.")
    else:
        with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 70):
            print(table.drop(columns="Checked_At_CT").to_string(index=False))
    e, today = be.load_earnings(EARNINGS_CSV), pd.Timestamp(now.astimezone(be.EASTERN).date())
    nxt = e[e["Symbol"].isin(list(held)) & (e["Earnings Date"] >= today)].sort_values("Earnings Date").groupby("Symbol").head(1)
    print("Next earnings of the held stocks (the stop window opens 7 days before):")
    for _, r in nxt.iterrows():
        t = str(r["Time"]).strip() if "Time" in r and pd.notna(r["Time"]) else "?"
        print(f"  {r['Symbol']:<6} {r['Earnings Date']:%a %b %-d} ({t})  window opens {r['Earnings Date'] - pd.Timedelta(days=ARM_DAYS):%a %b %-d}")
    for sym in sorted(set(held) - set(nxt["Symbol"])):
        print(f"  {sym:<6} no upcoming earnings date on file")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="read-only: show held stocks in the window and their stops")
    g.add_argument("--scheduled", action="store_true", help="LIVE check (launchd): sells at the stop")
    ap.add_argument("--as-of", help="dry run only: check as of this date (e.g. 2026-10-20 10:00, ET), with today's holdings and bars")
    a = ap.parse_args(argv)
    if a.as_of and not a.dry_run:
        ap.error("--as-of works with --dry-run only")
    now = pd.Timestamp(a.as_of).tz_localize(be.EASTERN).to_pydatetime() if a.as_of else None
    run_check(now=now, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
