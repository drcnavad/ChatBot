"""Earnings-day 5% drop stop for the LIVE account (approved by Chirag, Wed Oct 7, 2026, t192u; it replaced the
pre-earnings 3x ATR stop of Oct 3, 2026 - the forward test keeps its own ATR variants).

Rule: on a held stock's earnings day, its price is checked every 30 seconds in the pre-market (4:00-9:30 AM ET),
regular hours and after hours (until 8 PM ET, 5 PM ET on early-close days). If the price is ever 5% or more below the
PREVIOUS trading day's regular close, every whole share is sold once with a bid-based limit order (extended-hours
eligible).
"Earnings day" covers the session where the reaction happens (Reports/earnings_date.csv, Time column):
  - before-open report (AM) on a trading day: that day, all three sessions; the reference is the previous day's close.
  - after-close report (PM) or unknown time: the report day (all three sessions, reference = the previous day's close)
    AND the next trading day's pre-market and regular session (reference = the earnings-day close). An unknown time is
    treated like PM because that covers both possible reactions.
  - a report dated on a weekend / holiday: the next trading day, all three sessions.
The reference for a check is always the regular close of the trading day before the checked session's day (the last
completed daily bar before it, Alpaca market data).
Price = the quote's mid price (the same SIP -> IEX fallback as the smart limit prices, paper_trade._latest_quote); a
missing, stale, one-sided or too-wide quote is skipped and checked again 30 seconds later.
At the trigger: one SELL of the whole shares, limit at the bid - 0.05% (time in force DAY, extended_hours, so it works in
every session); a fractional rest (or an unfilled part) goes to the 9 AM CT fill check as a pending row; a holding below
one share is sold right away in regular hours, otherwise at that check. A stock sold by this stop is not bought back by
any live run until after its last earnings-day session (paper_trade.earnings_stop_blocked reads the state file; the
Friday rebalance gives its slot to the next eligible stock).
Never twice: one sale per (stock, earnings date), kept in Reports/earnings_stop_state.json, and a fixed client order id
(live-stop-YYYYMMDD-SYMBOL) that is looked up on the broker before sending. Any read failure skips the sale (fails closed).
The cash waits for the next scheduled run. Events go to Reports/run_log.csv; the latest check is in Reports/earnings_stops.csv.

Schedule: launchd com.stockanalysis.earningsstop starts `--loop` every 5 minutes (and at login); the loop runs only
while a held stock is in an earnings-day session (one loop at a time, a pid lock), checks every 30 seconds and exits when
nothing is left to watch. Quiet days: one quick look at the earnings file, no broker call.

Usage:
  python earnings_stop.py --dry-run [--as-of "2026-10-22 08:00"]  # read-only: watched stocks, references, quotes
  python earnings_stop.py --loop        # launchd job (LIVE: sells at the trigger), 30-second checks while needed
  python earnings_stop.py --scheduled   # one LIVE check
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

import backtest_engine as be

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
EARNINGS_CSV = os.path.join(REPORTS, "earnings_date.csv")
STATE_JSON = (os.environ.get("STOCK_ANALYSIS_EARNINGS_STOP_STATE")    # sales done (+ last earnings-day session: the
              or os.path.join(REPORTS, "earnings_stop_state.json"))     # buy-back block in paper_trade) + log notes
STATUS_CSV = os.path.join(REPORTS, "earnings_stops.csv")           # the latest check (status file)
LOOP_LOCK = os.path.join(REPORTS, ".earnings_stop_loop.lock")      # one 30-second loop at a time
DROP = 0.05                                                        # sell at 5% or more below the previous close
INTERVAL_S = 30                                                    # seconds between checks in the loop
ALL, MORNING = ("pre", "regular", "after"), ("pre", "regular")
REGULAR_MAX_AGE, EXTENDED_MAX_AGE = 60, 900                         # quote age limits (seconds)
REGULAR_MAX_SPREAD, EXTENDED_MAX_SPREAD = 0.005, 0.02               # spread limits (share of the mid price)
STATUS_COLS = ["Checked_At_CT", "Symbol", "Shares", "Earnings_Date", "Report_Time", "Watch_Day", "Sessions",
               "Prev_Close", "Prev_Close_Date", "Trigger", "Price", "Vs_Prev_%", "Status"]
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


def watch_days(e_date, report_time):
    """[(trading day, sessions)] watched for one report (see the module docstring)."""
    e = pd.Timestamp(e_date).normalize()
    if not be.is_session(e):
        return [(be.next_sessions(e, 1)[0], ALL)]
    if str(report_time).strip().upper() == "AM":
        return [(e, ALL)]
    return [(e, ALL), (be.next_sessions(e, 1)[0], MORNING)]


def active_event(sym, earnings, day, session=None):
    """(earnings date, time, watch day, sessions, last watch day) when `day` is an earnings day of `sym` (and `session`,
    if given, is one of its watched sessions); else None. `earnings` = be.load_earnings()."""
    day = pd.Timestamp(day).normalize()
    rows = earnings[earnings["Symbol"] == sym].sort_values("Earnings Date")
    for _, r in rows.iterrows():
        t = str(r["Time"]).strip().upper() if "Time" in r and pd.notna(r["Time"]) else ""
        days = watch_days(r["Earnings Date"], t)
        for d, sess in days:
            if d == day and (session is None or session in sess):
                return r["Earnings Date"], t, d, sess, days[-1][0]
    return None


def prev_close(bars, day, with_date=False):
    """Regular close of the last completed daily bar before `day` (one stock's bars: Date, Close), else None.
    with_date: (close, bar date) or (None, None)."""
    b = bars[bars["Date"] < pd.Timestamp(day).normalize()].sort_values("Date")
    c = float(b["Close"].iloc[-1]) if not b.empty else float("nan")
    ok = math.isfinite(c) and c > 0
    if with_date:
        return (c, pd.Timestamp(b["Date"].iloc[-1]).normalize()) if ok else (None, None)
    return c if ok else None


def prev_session(day):
    """The trading day before `day` (whose close is the reference)."""
    return (pd.Timestamp(day).normalize() - be.NYSE_SESSION).normalize()


def quote_check(q, session, now):
    """(mid, skip reason) for a (bid, ask, time, feed) quote."""
    bid, ask, ts, _ = q
    if not (bid and ask and bid > 0 and ask > 0 and ask >= bid):
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
    start = min(et, datetime.now(be.EASTERN)) - timedelta(days=15)    # a future --as-of (dry run) reads the latest bars
    bars = be.fetch_daily_bars(symbols, start=start.date())
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
    """Log an event once per day per key (a failing check every 30 seconds writes one row)."""
    if state["notes"].get(key) != str(today):
        state["notes"][key] = str(today)
        log_event(status, money, message, details)


def _lock_busy(path):
    """True when a live process holds the lock at `path` (a stale or missing lock is not busy)."""
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
def candidates(now, session, acct=None, earnings=None, quick=True, stand_in=False):
    """[{symbol, shares, e_date, e_time, day, sessions, last, prev, trigger}] for held stocks on an earnings day in
    `session` (any session when None), plus [(symbol, reason)] for those without a reference close, and the holdings.
    Reads only; no broker call when no report is near. The reference must be the close of the trading day right before
    the watched day (an older bar = no stop, warned); stand_in (dry run with a future --as-of only) uses the latest close
    available instead and says so."""
    today = pd.Timestamp(now.astimezone(be.EASTERN).date())
    earnings = be.load_earnings(EARNINGS_CSV) if earnings is None else earnings
    near = earnings[(earnings["Earnings Date"] >= today - pd.Timedelta(days=5)) & (earnings["Earnings Date"] <= today)]
    if quick and near.empty:
        return [], [], {}                       # no report near: no broker call at all
    acct = acct or account()
    held = {str(p["symbol"]).upper(): float(p.get("qty") or 0) for p in acct.position_dicts()}
    events = {s: ev for s in held if held[s] > 0 and (ev := active_event(s, earnings, today, session))}
    if not events:
        return [], [], held
    bars = daily_bars(sorted(events), now)
    out, bad = [], []
    for s, (e, t, day, sess, last) in sorted(events.items()):
        ref, ref_day = prev_close(bars[bars["Symbol"] == s], day, with_date=True)
        want = prev_session(day)
        if ref is None:
            bad.append((s, "no previous close (daily bars unavailable)"))
            continue
        if ref_day != want and not stand_in:
            bad.append((s, f"the {want:%a %b %-d} close is not available yet (latest bar {ref_day:%a %b %-d})"))
            continue
        out.append({"symbol": s, "shares": held[s], "e_date": e, "e_time": t or "?", "day": day, "sessions": sess,
                    "last": last, "prev": ref, "prev_date": ref_day, "stand_in": ref_day != want,
                    "trigger": ref * (1 - DROP)})
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
        return "at trigger - waiting for the running trade job"
    ok, why = run_all.acquire_trade_lock(run_all.FILL_LOCK)
    if not ok:
        return "at trigger - waiting for the running fill check"
    try:
        client = trading_client()
        broker = pt._broker_orders_by_client_id(client)
        if broker is None:
            _once(state, f"broker|{key}", today, "failed", "no", f"{s} fell 5% below its previous close on its earnings day "
                  "but the broker's orders could not be read, so nothing was sent (fails closed). Next check in 30 seconds.")
            return "at trigger - broker unreadable, not sent"
        attempt = 1
        while (o := broker.get(stop_cid(s, e, attempt))) is not None:
            st = getattr(o.status, "value", str(o.status)).lower()
            if st != "rejected" or pt._safe_number(getattr(o, "filled_qty", 0)) > 0:
                state["sold"][key] = {"at": now.astimezone(be.CENTRAL).isoformat(timespec="seconds"), "order_id": str(o.id),
                                      "react": f"{c['last']:%Y-%m-%d}", "note": f"found on the broker ({st})"}
                return f"sold earlier ({st})"
            attempt += 1
        if attempt > 3:
            _once(state, f"rejected|{key}", today, "failed", "no", f"The broker rejected the {s} earnings-day stop sale 3 times; "
                  "not sent again. Check Alpaca.")
            return "at trigger - rejected 3 times"
        held_now, open_now = pt._symbol_state(client, s)
        if held_now is None:
            _once(state, f"read|{key}", today, "failed", "no", f"{s} fell 5% below its previous close on its earnings day but "
                  "the position could not be read, so nothing was sent (fails closed). Next check in 30 seconds.")
            return "at trigger - position unreadable, not sent"
        if open_now:
            _once(state, f"open|{key}", today, "warning", "no", f"{s} fell 5% below its previous close on its earnings day "
                  "but already has an open order, so the stop waits (it never stacks a second order).")
            return "at trigger - open order already working"
        if held_now <= 0:
            return "no longer held"
        whole = math.floor(held_now + 1e-9)
        only_frac = whole < 1 and session == "regular"   # below one share: sold now in regular hours (no extended hours)
        qty = held_now if only_frac else whole
        limit = pt._tick(q[0] * (1 - pt.LIMIT_OFFSET), False)
        oid, filled = None, 0.0
        if qty > 0:
            req = LimitOrderRequest(symbol=s, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                                    limit_price=limit, extended_hours=not only_frac, client_order_id=stop_cid(s, e, attempt))
            try:
                order = client.submit_order(req)
            except Exception as err:
                _once(state, f"submit|{key}", today, "failed", "unknown", f"The {s} earnings-day stop sale failed to send "
                      f"({str(err)[:120]}). It is tried again at the next check.")
                return "at trigger - send failed, retrying"
            oid, filled = str(order.id), pt._safe_number(getattr(order, "filled_qty", 0))
        stamp = now.astimezone(be.CENTRAL).isoformat(timespec="seconds")
        state["sold"][key] = {"at": stamp, "order_id": oid, "qty": qty, "limit": limit, "held": held_now,
                              "react": f"{c['last']:%Y-%m-%d}"}
        _save_state(state)
        frac = round(held_now - qty, 6)
        try:
            _append_pending({"symbol": s, "side": "SELL", "qty": held_now, "limit_price": limit, "order_id": oid,
                             "order_qty": qty, "exit": True, "recorded_at": stamp, "decision": None,
                             "evening_date": str(today), "source": "earnings-stop"}, pending_path)
        except (OSError, ValueError) as err:
            log_event("failed", "unknown", f"The {s} earnings-day stop sale was sent, but its row for the 9 AM CT check "
                      f"could not be written ({str(err)[:100]}). Check Alpaca: any unfilled rest or fraction must be sold by hand.")
        where = {"pre": "pre-market", "regular": "regular-hours", "after": "after-hours"}[session]
        sent = (f"Sent a {where} sell order for {qty:g} {'' if only_frac else 'whole '}shares with a limit at ${limit:,.2f} "
                f"(bid ${q[0]:,.2f})" if qty else "No whole share to sell now")
        log_event("ok", "yes" if filled > 0 else "unknown",
                  f"Earnings-day stop: {s} fell to ${(q[0] + q[1]) / 2:,.2f}, {DROP:.0%} or more below the previous close "
                  f"${c['prev']:,.2f} (trigger ${c['trigger']:,.2f}). {sent}."
                  + (f" The {frac:g} fractional share goes to the 9 AM CT fill check." if frac > 0 else "")
                  + f" Earnings {e:%a %b %-d} ({c['e_time']}). The cash waits for the next scheduled run; {s} is not bought back "
                  f"until after {c['last']:%a %b %-d}.",
                  f"order {oid or 'none'}; client id {stop_cid(s, e, attempt)}; held {held_now:g}")
        return ((f"SELL sent: {qty:g} shares at limit ${limit:,.2f}" if qty else "at trigger - no whole share")
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
              f"The {c['symbol']} earnings-day stop check failed ({str(err)[:150]}). Check Alpaca; next check in 30 seconds.")
        return "at trigger - error, retrying"


def run_check(now=None, dry_run=False, pending_path=None):
    """One check. Dry run: prints the watched stocks, sends and writes nothing, never makes a trading client.
    Returns the status rows (one per watched held stock)."""
    import paper_trade as pt
    now = now or datetime.now(be.EASTERN)
    session = session_now(now)
    if not dry_run and session is None:
        print(f"{now.astimezone(be.CENTRAL):%Y-%m-%d %H:%M:%S} CT: outside the pre-market/regular/after-hours sessions - idle")
        return []
    state = _load_state()
    today = now.astimezone(be.CENTRAL).date()
    try:
        cands, bad, held = candidates(now, session, quick=not dry_run,
                                      stand_in=dry_run and now > datetime.now(be.EASTERN))
    except Exception as err:
        if dry_run:
            raise
        _once(state, "inputs", today, "failed", "no", f"The earnings-day stop check could not read its inputs ({str(err)[:150]}). "
              "No stop was checked; next check in 30 seconds.")
        _save_state(state)
        return []
    if not dry_run:
        for s, why in bad:
            _once(state, f"bad|{s}", today, "warning", "no", f"{s} is on its earnings day but has no earnings-day stop: {why}.")
    rows, stamp = [], now.astimezone(be.CENTRAL).strftime("%Y-%m-%d %H:%M:%S")
    for c in cands:
        s, key = c["symbol"], f"{c['symbol']}|{c['e_date']:%Y-%m-%d}"
        mid, status, m = float("nan"), "watching", None
        try:
            q = latest_quote(s)
            m, skip = quote_check(q, session or "regular", now)
            mid = m if m is not None else float("nan")
        except Exception as err:
            q, skip = None, f"no quote ({str(err)[:60]})"
        if key in state["sold"]:
            status = f"sold {state['sold'][key]['at'][:16].replace('T', ' ')}"
        elif skip and (not dry_run or m is None):
            status = f"watching (quote skipped: {skip})"
        elif q is not None and mid <= c["trigger"]:
            whole = math.floor(c["shares"] + 1e-9)
            status = (f"AT TRIGGER - would sell {whole:g} whole shares at ${pt._tick(q[0] * (1 - pt.LIMIT_OFFSET), False):,.2f}"
                      + (f" (quote note: {skip})" if skip else "")) if dry_run else _sell_safe(c, q, session, now, state, pending_path)
        elif dry_run and skip:
            status = f"watching (quote note: {skip})"
        if not dry_run and key not in state["armed"] and key not in state["sold"]:
            state["armed"][key] = str(today)    # first check of the report: one info row
            log_event("ok", "no", f"Earnings-day stop watching {s} (earnings {c['e_date']:%a %b %-d}, {c['e_time']}): sells the "
                      f"whole shares if the price falls to ${c['trigger']:,.2f} ({DROP:.0%} below the previous close "
                      f"${c['prev']:,.2f}); checked every {INTERVAL_S} seconds through {c['last']:%a %b %-d}.")
        rows.append({"Checked_At_CT": stamp, "Symbol": s, "Shares": c["shares"], "Earnings_Date": f"{c['e_date']:%Y-%m-%d}",
                     "Report_Time": c["e_time"], "Watch_Day": f"{c['day']:%Y-%m-%d}", "Sessions": "/".join(c["sessions"]),
                     "Prev_Close": round(c["prev"], 2), "Prev_Close_Date": f"{c['prev_date']:%Y-%m-%d}"
                     + (" (stand-in: latest available)" if c["stand_in"] else ""), "Trigger": round(c["trigger"], 2), "Price": round(mid, 2),
                     "Vs_Prev_%": round((mid / c["prev"] - 1) * 100, 2), "Status": status})
    table = pd.DataFrame(rows + [{"Checked_At_CT": stamp, "Symbol": s, "Shares": held.get(s), "Status": f"no stop: {why}"}
                                 for s, why in bad], columns=STATUS_COLS)
    if dry_run:
        _print_dry(now, session, table, held)
        return rows
    tmp = STATUS_CSV + ".tmp"
    table.to_csv(tmp, index=False)
    os.replace(tmp, STATUS_CSV)
    _save_state(state)
    print(f"{stamp} CT ({session}): {len(cands)} held stock(s) on an earnings day" +
          "".join(f"\n  {r['Symbol']}: prev close {r['Prev_Close']} trigger {r['Trigger']} price {r['Price']} -> {r['Status']}"
                  for r in rows))
    return rows


def watching(rows):
    """True while some checked stock is still watched (not sold, still held): the loop keeps going."""
    return any(not str(r["Status"]).startswith(("sold", "no longer held", "SELL sent")) for r in rows)


def loop(interval=INTERVAL_S, now_fn=None, sleep_fn=time.sleep, max_checks=None):
    """The launchd job: one loop at a time (pid lock); a check every `interval` seconds while a held stock is in an
    earnings-day session; exits when nothing is watched (launchd starts it again 5 minutes later). Returns the checks made."""
    if _lock_busy(LOOP_LOCK):
        print("another earnings-day stop loop is running - nothing to do")
        return 0
    with open(LOOP_LOCK, "w") as f:
        json.dump({"pid": os.getpid(), "at": datetime.now(be.CENTRAL).isoformat(timespec="seconds")}, f)
    n = 0
    try:
        while max_checks is None or n < max_checks:
            rows = run_check(now=now_fn() if now_fn else None)
            n += 1
            if not watching(rows):
                break
            sleep_fn(interval)
    finally:
        try:
            os.remove(LOOP_LOCK)
        except OSError:
            pass
    return n


def _print_dry(now, session, table, held):
    print(f"DRY RUN {now.astimezone(be.CENTRAL):%a %b %-d %Y %H:%M} CT - nothing is sent or written. Session now: {session or 'closed'}.")
    print(f"Rule: on a held stock's earnings day (AM report: that day; PM/unknown: that day + the next pre-market and regular "
          f"session) sell the whole shares once if the price is {DROP:.0%} or more below the previous close; checked every "
          f"{INTERVAL_S} s. Held: {len(held)} stocks.")
    if table.empty:
        print("No held stock is on an earnings day at this time.")
    else:
        with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 70):
            print(table.drop(columns="Checked_At_CT").to_string(index=False))
    e, today = be.load_earnings(EARNINGS_CSV), pd.Timestamp(now.astimezone(be.EASTERN).date())
    nxt = e[e["Symbol"].isin(list(held)) & (e["Earnings Date"] >= today)].sort_values("Earnings Date").groupby("Symbol").head(1)
    print("Next earnings of the held stocks and the days watched:")
    for _, r in nxt.iterrows():
        t = str(r["Time"]).strip() if "Time" in r and pd.notna(r["Time"]) else "?"
        days = "; ".join(f"{d:%a %b %-d} {'/'.join(s)}" for d, s in watch_days(r["Earnings Date"], t))
        print(f"  {r['Symbol']:<6} {r['Earnings Date']:%a %b %-d} ({t})  watched: {days}")
    for sym in sorted(set(held) - set(nxt["Symbol"])):
        print(f"  {sym:<6} no upcoming earnings date on file")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="read-only: show held stocks on an earnings day and their triggers")
    g.add_argument("--scheduled", action="store_true", help="one LIVE check: sells at the trigger")
    g.add_argument("--loop", action="store_true", help="launchd job: LIVE checks every 30 seconds while needed")
    ap.add_argument("--as-of", help="dry run only: check as of this time (e.g. 2026-10-22 08:00, ET), with today's holdings and bars")
    a = ap.parse_args(argv)
    if a.as_of and not a.dry_run:
        ap.error("--as-of works with --dry-run only")
    now = pd.Timestamp(a.as_of).tz_localize(be.EASTERN).to_pydatetime() if a.as_of else None
    if a.loop:
        loop()
    else:
        run_check(now=now, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
