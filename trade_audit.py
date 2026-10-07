"""Read-only trade audit of the LIVE account (approved by Chirag, Sun Oct 4, 2026): the record keeping and alerts a
professional trading bot keeps next to its order logic. It never places, changes or cancels an order and never touches
the trading files' state: GET only through alpaca_paper.PaperAccount (/account, /positions, /orders, /clock,
/account/activities), the keys stay inside alpaca_paper.

  1. Order ledger (Reports/live_trade_ledger.csv): every order on the account, filled or not, with its source (the bot's
     decision / fill check / rest / replacement / earnings-day stop, or manual), type, time in force, the INTENDED price
     (the plan price the bot stamps into its client order id, live-...-<cents>), the limit, the average fill price and the
     slippage vs the plan (+ = it cost money) in % and $.
  2. Round trips (Reports/live_round_trips.csv): every sale matched to its purchase (FIFO, as Alpaca): buy date and
     price, sell date and price, shares, P/L $ and %, days held, and which order bought / sold.
  3. Reconciliation with Alpaca: account flags (blocked / suspended), open orders the bot does not track (manual or
     orphaned), pending rows whose order is not on Alpaca, holdings outside the strategy with no sell queued, targets
     not held, weight drift.
  4. Alerts: a daily loss of 3% (warning) / 5% (alert) vs the last close, the Mac clock off Alpaca's clock by over 60 s,
     the NYSE calendar in backtest_engine disagreeing with Alpaca's next open / close (an unscheduled closure or early
     close), a fill over 1% worse than its plan price, a rejected order, and the account being unreadable (API outage).
     With --log each NEW alert is one Reports/run_log.csv row (daily alerts once a day, per-order alerts once ever;
     memory: Reports/trade_audit_state.json). Alerts only - nothing here stops or sends a trade.

    python trade_audit.py                  # print the audit (writes nothing)
    python trade_audit.py --write          # ... and write the ledger + round-trip CSVs
    python trade_audit.py --write --log    # ... and new alerts to the run log (launchd/com.stockanalysis.tradeaudit.plist)
"""
import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
LEDGER_CSV = os.path.join(REPORTS, "live_trade_ledger.csv")
ROUND_TRIPS_CSV = os.path.join(REPORTS, "live_round_trips.csv")
STATE_JSON = os.environ.get("STOCK_ANALYSIS_AUDIT_STATE") or os.path.join(REPORTS, "trade_audit_state.json")
PENDING_JSON = os.path.join(REPORTS, "live_pending_orders.json")
STOP_STATE_JSON = os.environ.get("STOCK_ANALYSIS_EARNINGS_STOP_STATE") or os.path.join(REPORTS, "earnings_stop_state.json")
PICKS_CSV = os.path.join(REPORTS, "strategy_picks.csv")
CT, ET = ZoneInfo("America/Chicago"), ZoneInfo("America/New_York")
RUN = "Trade audit"
AUDIT_START = "2026-10-02"  # fresh start: the audit only covers orders/fills from Friday Oct 2, 2026 onwards

DAY_LOSS_WARN, DAY_LOSS_ALERT = 0.03, 0.05     # daily loss vs the last close (alert only, nothing is stopped)
CLOCK_DRIFT_SECS = 60
SLIPPAGE_ALERT_PCT = 1.0                       # a fill this much worse than its plan price
DRIFT_PP = 2.0                                 # holding weight vs target (percentage points), info only
RECENT_DAYS = 7                                # per-order alerts look at orders of the last week
OPEN = {"new", "accepted", "pending_new", "partially_filled", "accepted_for_bidding", "pending_replace", "pending_cancel",
        "held", "calculated", "stopped", "suspended"}
LEDGER_COLS = ["Submitted_CT", "Filled_CT", "Symbol", "Side", "Source", "Type", "TIF", "Ext_Hours", "Qty", "Filled_Qty",
               "Status", "Plan_Price", "Limit", "Fill_Price", "Slippage_vs_Plan_%", "Slippage_vs_Plan_$", "Order_ID",
               "Client_Order_ID"]
TRIP_COLS = ["Symbol", "Shares", "Bought_CT", "Buy_Price", "Sold_CT", "Sell_Price", "P/L $", "P/L %", "Days_Held",
             "Buy_Source", "Sell_Source", "Note"]
_CID = re.compile(r"^live-(?:(fill|rest)-)?(\d{8})-(BUY|SELL)-([A-Z0-9]+)-(\d+)-(\d+)(?:-r\d+)?(-c)?$")


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _ct(ts):
    """ISO / Timestamp -> naive CT Timestamp (NaT when missing)."""
    if ts is None or ts == "":
        return pd.NaT
    t = pd.Timestamp(ts)
    return (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(CT).tz_localize(None)


# ----------------------------------------------------------------------------- client order ids (pure)
def order_source(cid):
    """Who sent an order, from its client order id (paper_trade._client_order_id / earnings_stop.stop_cid)."""
    cid = str(cid or "")
    if cid.startswith("live-stop-"):
        return "earnings-day stop"
    if not cid.startswith("live-"):
        return "manual / other"
    if cid.endswith("-c"):
        return "bot: replacement at a fresh quote"
    if cid.startswith("live-fill-"):
        return "bot: fill check"
    if cid.startswith("live-rest-"):
        return "bot: rest that did not fit"
    return "bot: decision"


def plan_price(cid):
    """The plan (intended) price in the bot's client order id (live-...-<cents>), or None. Ids cut at Alpaca's 48
    characters are not trusted (the cents may be cut)."""
    cid = str(cid or "")
    m = _CID.match(cid)
    if not m or len(cid) >= 48:
        return None
    cents = int(m.group(6))
    return cents / 100 if cents > 0 else None


def slippage(side, plan, fill, qty):
    """(% , $) of a fill vs its plan price; + = it cost money (a buy above / a sell below the plan). (nan, nan) when
    either price is missing."""
    plan, fill, qty = _num(plan), _num(fill), _num(qty)
    if not (plan > 0 and fill > 0):
        return math.nan, math.nan
    sign = 1 if str(side).lower() == "buy" else -1
    return (round(sign * (fill - plan) / plan * 100, 3) + 0.0,                  # + 0.0: no "-0.0"
            round(sign * (fill - plan) * (qty if qty == qty else 0.0), 2) + 0.0)


# ----------------------------------------------------------------------------- read (GET only)
def _orders(acct, max_pages=40):
    out, params = [], {"status": "all", "limit": 500, "direction": "desc"}
    for _ in range(max_pages):
        page = acct._get("/orders", params)
        if not isinstance(page, list):
            break
        out += page
        if len(page) < 500 or not page[-1].get("submitted_at"):
            break
        params = {**params, "until": page[-1]["submitted_at"]}
    return out


def fetch(acct=None, positions=None):
    """Everything the audit reads, each part on its own (a part that fails is None and named in 'errors').
    positions: already read by the caller (the dashboard's live holdings), so /positions is not read twice."""
    import alpaca_paper as ap
    acct = acct or ap.PaperAccount()
    data, errors = {}, []
    parts = [("account", lambda: acct._get("/account")), ("orders", lambda: _orders(acct)), ("fills", lambda: acct.fills())]
    if positions is None:
        parts.insert(1, ("positions", lambda: acct._get("/positions")))
    else:
        data["positions"] = list(positions)
    for name, fn in parts:
        try:
            data[name] = fn()
        except Exception as e:                      # the message never holds the keys (alpaca_paper)
            data[name] = None
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
    try:
        before = datetime.now(timezone.utc)
        clock = acct._get("/clock")
        after = datetime.now(timezone.utc)
        data["clock"] = clock if isinstance(clock, dict) and clock.get("timestamp") else None
        data["clock_local"] = before + (after - before) / 2
    except Exception as e:
        data["clock"], data["clock_local"] = None, None
        errors.append(f"clock: {type(e).__name__}: {str(e)[:120]}")
    data["errors"], data["as_of"] = errors, datetime.now(CT)
    return data


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------------------------- 1. order ledger
def ledger(orders):
    """One row per order (newest first) with the plan price, fill price and slippage vs the plan."""
    rows = []
    for o in orders or []:
        cid, side = o.get("client_order_id") or "", str(o.get("side") or "").lower()
        plan, fill, fq = plan_price(cid), _num(o.get("filled_avg_price")), _num(o.get("filled_qty"))
        pct, usd = slippage(side, plan, fill, fq)
        rows.append({"Submitted_CT": _ct(o.get("submitted_at")), "Filled_CT": _ct(o.get("filled_at")),
                     "Symbol": str(o.get("symbol") or ""), "Side": side.upper(), "Source": order_source(cid),
                     "Type": o.get("type") or o.get("order_type"), "TIF": o.get("time_in_force"),
                     "Ext_Hours": bool(o.get("extended_hours")), "Qty": _num(o.get("qty")), "Filled_Qty": fq,
                     "Status": o.get("status"), "Plan_Price": plan, "Limit": _num(o.get("limit_price")),
                     "Fill_Price": fill if fill == fill else None, "Slippage_vs_Plan_%": pct, "Slippage_vs_Plan_$": usd,
                     "Order_ID": o.get("id"), "Client_Order_ID": cid})
    df = pd.DataFrame(rows, columns=LEDGER_COLS)
    if not len(df):
        return df
    df = df[df["Submitted_CT"] >= pd.Timestamp(AUDIT_START)]  # fresh start: Friday Oct 2, 2026 onwards
    return df.sort_values("Submitted_CT", ascending=False, kind="stable").reset_index(drop=True)


def slippage_summary(led):
    """{orders, filled, traded $, cost vs plan $, cost %} over the bot's filled orders that carry a plan price."""
    f = led[(led["Filled_Qty"] > 0) & led["Plan_Price"].notna() & led["Fill_Price"].notna()] if len(led) else led
    traded = float((f["Fill_Price"] * f["Filled_Qty"]).sum()) if len(f) else 0.0
    cost = float(f["Slippage_vs_Plan_$"].sum()) if len(f) else 0.0
    return {"orders": int(len(led)), "filled_with_plan": int(len(f)), "traded": traded, "cost": cost,
            "cost_pct": cost / traded * 100 if traded else 0.0,
            "worst": f.sort_values("Slippage_vs_Plan_%", ascending=False).head(1) if len(f) else f}


# ----------------------------------------------------------------------------- 2. round trips (FIFO)
def round_trips(fills, orders=None):
    """Every sale matched to its purchase, oldest shares first (FIFO, Alpaca's default). Fees are not included."""
    src = {o.get("id"): order_source(o.get("client_order_id")) for o in orders or []}
    book, out = {}, []
    rows = sorted(fills or [], key=lambda a: str(a.get("transaction_time") or ""))
    rows = [a for a in rows if str(a.get("transaction_time") or "")[:10] >= AUDIT_START]  # fresh start: Friday Oct 2, 2026 onwards
    for a in rows:
        sym, side = str(a.get("symbol") or "").upper(), str(a.get("side") or "").lower()
        q, px, t, oid = _num(a.get("qty")), _num(a.get("price")), _ct(a.get("transaction_time")), a.get("order_id")
        if not (q > 0):
            continue
        lots = book.setdefault(sym, [])
        if side == "buy":
            lots.append([q, px, t, src.get(oid, "")])
            continue
        left = q
        while left > 1e-9 and lots:
            lot = lots[0]
            m = min(left, lot[0])
            out.append({"Symbol": sym, "Shares": m, "Bought_CT": lot[2], "Buy_Price": lot[1], "Sold_CT": t, "Sell_Price": px,
                        "P/L $": round(m * (px - lot[1]), 2), "P/L %": round((px / lot[1] - 1) * 100, 2) if lot[1] > 0 else math.nan,
                        "Days_Held": (t.normalize() - lot[2].normalize()).days, "Buy_Source": lot[3],
                        "Sell_Source": src.get(oid, ""), "Note": ""})
            lot[0] -= m
            left -= m
            if lot[0] <= 1e-9:
                lots.pop(0)
        if left > 1e-6:
            out.append({"Symbol": sym, "Shares": left, "Bought_CT": pd.NaT, "Buy_Price": math.nan, "Sold_CT": t,
                        "Sell_Price": px, "P/L $": math.nan, "P/L %": math.nan, "Days_Held": math.nan, "Buy_Source": "",
                        "Sell_Source": src.get(oid, ""), "Note": "bought before the account history (no purchase fill)"})
    df = pd.DataFrame(out, columns=TRIP_COLS)
    return df.sort_values("Sold_CT", ascending=False, kind="stable").reset_index(drop=True) if len(df) else df


# ----------------------------------------------------------------------------- 3. reconciliation
def _finding(level, key, text):
    return {"Level": level, "Key": key, "Check": text}


def reconcile(data, pending=None, stop_state=None, picks=None):
    """Findings (Level ok / info / warning / failed, Key, Check) comparing Alpaca with the bot's own records."""
    out, acct = [], data.get("account")
    if isinstance(acct, dict):
        flags = [k for k in ("trading_blocked", "account_blocked", "trade_suspended_by_user", "transfers_blocked") if acct.get(k)]
        if str(acct.get("status", "ACTIVE")).upper() != "ACTIVE" or flags:
            out.append(_finding("failed", f"account-flags|{datetime.now(CT):%Y-%m-%d}", f"Alpaca account status {acct.get('status')}"
                                + (f", flags: {', '.join(flags)}" if flags else "") + " - orders may be refused. Check Alpaca."))
        else:
            out.append(_finding("ok", "account-flags", "Account active, trading not blocked."))
    pending = pending if pending is not None else {}
    rows = pending.get("orders", []) if isinstance(pending, dict) else []
    tracked = {str(r.get(k)) for r in rows if isinstance(r, dict) for k in ("order_id", "completed_order_id") if r.get(k)}
    tracked |= {str(v.get("order_id")) for v in ((stop_state or {}).get("sold") or {}).values() if isinstance(v, dict) and v.get("order_id")}
    orders = data.get("orders")
    if orders is not None:
        on_broker = {str(o.get("id")) for o in orders}
        open_ = [o for o in orders if str(o.get("status") or "").lower() in OPEN]
        for o in open_:
            cid, label = o.get("client_order_id") or "", f"{str(o.get('side')).upper()} {o.get('symbol')} {_num(o.get('qty')):g}"
            if not str(cid).startswith("live-"):
                out.append(_finding("warning", f"manual-open|{o.get('id')}", f"Open order not sent by the bot: {label} "
                                    f"({o.get('type')}, {o.get('status')}). The bot does not track it; cancel it in Alpaca if it is not yours."))
            elif str(o.get("id")) not in tracked and not str(cid).startswith("live-stop-"):
                out.append(_finding("warning", f"orphan-open|{o.get('id')}", f"Open bot order {label} ({cid}) is not in "
                                    "Reports/live_pending_orders.json, so no fill check follows it up. Check it in Alpaca."))
        if not open_:
            out.append(_finding("ok", "open-orders", "No open orders on Alpaca."))
        for r in rows:
            oid = str(r.get("order_id") or "") if isinstance(r, dict) else ""
            if oid and oid not in on_broker and len(orders) < 500:
                out.append(_finding("warning", f"pending-missing|{oid}", f"Pending {r.get('side')} {r.get('symbol')} row points "
                                    f"to order {oid}, which Alpaca does not list. The 9 AM fill check will report it."))
    pos = data.get("positions")
    if pos is not None and picks is not None and len(picks):
        held = {str(p.get("symbol")).upper(): _num(p.get("qty")) for p in pos}
        value = {str(p.get("symbol")).upper(): _num(p.get("market_value")) for p in pos}
        equity = _num((acct or {}).get("equity")) if isinstance(acct, dict) else math.nan
        target = {str(s).upper(): float(w) for s, w in zip(picks["Symbol"], pd.to_numeric(picks["Strategy_Weight"], errors="coerce").fillna(0)) if w > 0}
        sells = {str(r.get("symbol")).upper() for r in rows if isinstance(r, dict) and str(r.get("side")).upper() == "SELL"}
        buys = {str(r.get("symbol")).upper() for r in rows if isinstance(r, dict) and str(r.get("side")).upper() == "BUY"}
        stopped = {k.split("|")[0] for k in ((stop_state or {}).get("sold") or {})}
        for s in sorted(set(held) - set(target)):
            if s in sells:
                out.append(_finding("info", f"leftover|{s}", f"{s} {held[s]:g} sh is not in the strategy; its sale is queued "
                                    "for the next fill check (Reports/live_pending_orders.json)."))
            else:
                out.append(_finding("warning", f"untracked|{s}", f"{s} {held[s]:g} sh is held but not in the strategy, and no "
                                    "sale is queued. The next Friday rebalance sells it; sell it by hand if you want it gone sooner."))
        missing = sorted(set(target) - set(held) - buys - stopped)
        if missing:
            out.append(_finding("info", "targets-not-held", "In the strategy but not held: " + ", ".join(missing)
                                + " (normal when the earnings rule skipped the buy or a buy is still to come)."))
        if equity > 0:
            drift = [(s, value[s] / equity * 100, target[s] * 100) for s in sorted(set(held) & set(target))
                     if value.get(s) == value.get(s) and abs(value[s] / equity * 100 - target[s] * 100) > DRIFT_PP]
            if drift:
                out.append(_finding("info", "drift", "Weight vs target over 2 points: " + "; ".join(
                    f"{s} {a:.1f}% vs {t:.1f}%" for s, a, t in drift) + " (prices move between rebalances; Friday resets it)."))
        if not any(f["Key"].startswith("untracked") for f in out):
            out.append(_finding("ok", "holdings", f"{len(held)} holdings: every one is in the strategy or has a sale queued."))
    return out


# ----------------------------------------------------------------------------- 4. alerts
def alerts(data, led=None, now=None):
    """Alert findings (warning / failed) from the account, the clock, the calendar and the recent orders."""
    import backtest_engine as be
    now = now or datetime.now(CT)
    out = []
    if data.get("errors"):
        out.append(_finding("failed", f"unreadable|{now:%Y-%m-%d}", "Alpaca could not be read fully (" + "; ".join(data["errors"])
                            + "). The trading jobs fail closed (send nothing) while it lasts; check Alpaca's status page."))
    acct = data.get("account")
    if isinstance(acct, dict):
        eq, last = _num(acct.get("equity")), _num(acct.get("last_equity"))
        if eq > 0 and last > 0:
            chg = eq / last - 1
            if chg <= -DAY_LOSS_ALERT:
                out.append(_finding("failed", f"day-loss-5|{now:%Y-%m-%d}", f"Daily loss alert: equity ${eq:,.0f} is {chg:.1%} vs the "
                                    f"last close (${last:,.0f}), over the {DAY_LOSS_ALERT:.0%} line. Nothing was stopped; decide "
                                    "whether to pause the bot."))
            elif chg <= -DAY_LOSS_WARN:
                out.append(_finding("warning", f"day-loss-3|{now:%Y-%m-%d}", f"Equity ${eq:,.0f} is {chg:.1%} vs the last close "
                                    f"(${last:,.0f}), over the {DAY_LOSS_WARN:.0%} warning line. Nothing was stopped."))
    clock = data.get("clock")
    if clock:
        server = pd.Timestamp(clock["timestamp"]).to_pydatetime()
        local = data.get("clock_local") or datetime.now(timezone.utc)
        drift = (local - server).total_seconds()
        if abs(drift) > CLOCK_DRIFT_SECS:
            out.append(_finding("warning", f"clock|{now:%Y-%m-%d}", f"This Mac's clock is {drift:+.0f} s off Alpaca's clock. "
                                "The jobs time their runs by the Mac's clock: turn on automatic time in System Settings."))
        out += _calendar_check(clock, be, now)
    since = pd.Timestamp(now.astimezone(CT).replace(tzinfo=None)) - pd.Timedelta(days=RECENT_DAYS)
    for _, r in (led if led is not None else pd.DataFrame(columns=LEDGER_COLS)).iterrows():
        t = r["Submitted_CT"]
        if pd.isna(t) or t < since:
            continue
        if str(r["Status"]).lower() == "rejected":
            out.append(_finding("warning", f"rejected|{r['Order_ID']}", f"Alpaca rejected {r['Side']} {r['Symbol']} {r['Qty']:g} "
                                f"({r['Source']}, sent {t:%a %b %-d %I:%M %p} CT). Check why in Alpaca."))
        s = r["Slippage_vs_Plan_%"]
        if s == s and s is not None and s > SLIPPAGE_ALERT_PCT:
            out.append(_finding("warning", f"slippage|{r['Order_ID']}", f"{r['Side']} {r['Symbol']} filled at ${r['Fill_Price']:,.2f}, "
                                f"{s:.2f}% worse than its plan price ${r['Plan_Price']:,.2f} (${r['Slippage_vs_Plan_$']:,.2f} on "
                                f"{r['Filled_Qty']:g} sh)."))
    return out


def _calendar_check(clock, be, now):
    """Alpaca's next open / close vs backtest_engine's NYSE calendar (holidays, early closes)."""
    out = []
    try:
        nopen = pd.Timestamp(clock["next_open"]).tz_convert(ET)
        nclose = pd.Timestamp(clock["next_close"]).tz_convert(ET)
    except Exception:
        return out
    n_et = pd.Timestamp(now).tz_convert(ET) if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz=CT).tz_convert(ET)
    d = n_et.normalize().tz_localize(None)
    if not clock.get("is_open"):
        if not (be.is_session(d) and n_et.hour * 60 + n_et.minute < 9 * 60 + 30):
            d += pd.Timedelta(days=1)
        while not be.is_session(d):
            d += pd.Timedelta(days=1)
        if d.date() != nopen.date():
            out.append(_finding("warning", f"calendar-open|{nopen.date()}", f"Calendar mismatch: Alpaca's next open is "
                                f"{nopen:%a %b %-d}, the bot's NYSE calendar says {d:%a %b %-d}. An unscheduled closure or a "
                                "holiday rule needs adding to backtest_engine (SPECIAL_CLOSURES) before the bot trades then."))
    early = be.is_early_close(nclose.date())
    if (nclose.hour == 13) != early:
        out.append(_finding("warning", f"calendar-close|{nclose.date()}", f"Early-close mismatch on {nclose:%a %b %-d}: Alpaca closes "
                            f"at {nclose:%I:%M %p} ET, the bot's calendar expects {'1:00 PM' if early else '4:00 PM'} ET."))
    return out


# ----------------------------------------------------------------------------- report + run-log rows
def report(data=None, now=None, pending=None, stop_state=None, picks=None):
    """The whole audit as a dict (ledger, round_trips, summary, findings, alerts). Reads only."""
    data = data if data is not None else fetch()
    pending = pending if pending is not None else _read_json(PENDING_JSON)
    stop_state = stop_state if stop_state is not None else _read_json(STOP_STATE_JSON)
    if picks is None:
        try:
            picks = pd.read_csv(PICKS_CSV, usecols=["Symbol", "Strategy_Weight"])
        except Exception:
            picks = None
    led = ledger(data.get("orders"))
    return {"ledger": led, "round_trips": round_trips(data.get("fills"), data.get("orders")), "summary": slippage_summary(led),
            "findings": reconcile(data, pending, stop_state, picks), "alerts": alerts(data, led, now), "as_of": data.get("as_of"),
            "errors": data.get("errors", [])}


def new_alerts(items, now=None, state_path=None):
    """The warning / failed items not reported before (state file: key -> first reported date; kept 60 days)."""
    now, state_path = now or datetime.now(CT), state_path or STATE_JSON
    state = _read_json(state_path)
    seen = state.get("reported", {}) if isinstance(state, dict) else {}
    cut = (now - timedelta(days=60)).date().isoformat()
    seen = {k: v for k, v in seen.items() if str(v) >= cut}
    fresh = [a for a in items if a["Level"] in ("warning", "failed") and a["Key"] not in seen]
    for a in fresh:
        seen[a["Key"]] = now.date().isoformat()
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    tmp = state_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"reported": seen, "last_run": now.isoformat(timespec="seconds")}, f, indent=1)
    os.replace(tmp, state_path)
    return fresh


def log_alerts(items, now=None, state_path=None):
    """One run-log row per new alert (run_all.log_event; never raises)."""
    from run_all import log_event
    fresh = new_alerts(items, now, state_path)
    for a in fresh:
        log_event(RUN, a["Level"], "no", a["Check"] + " (alert only, no order sent)", details=a["Key"])
    return fresh


def write_files(rep, ledger_csv=None, trips_csv=None):
    ledger_csv, trips_csv = ledger_csv or LEDGER_CSV, trips_csv or ROUND_TRIPS_CSV
    for df, path in ((rep["ledger"], ledger_csv), (rep["round_trips"], trips_csv)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write", action="store_true", help="write the ledger and round-trip CSVs")
    p.add_argument("--log", action="store_true", help="write new alerts to Reports/run_log.csv")
    a = p.parse_args(argv)
    rep = report()
    s = rep["summary"]
    print(f"Trade audit as of {rep['as_of']:%a %b %-d %I:%M %p} CT (read-only)")
    print(f"  Orders on Alpaca: {s['orders']}; filled bot orders with a plan price: {s['filled_with_plan']}; "
          f"traded ${s['traded']:,.2f}; cost vs plan {'-' if s['cost'] < 0 else ''}${abs(s['cost']):,.2f} "
          f"({s['cost_pct']:+.3f}%; + = it cost money)")
    print(f"  Round trips (FIFO sales): {len(rep['round_trips'])}")
    for f in rep["findings"] + rep["alerts"]:
        print(f"  [{f['Level']}] {f['Check']}")
    if not rep["alerts"]:
        print("  [ok] No alerts (daily loss, clock, calendar, slippage, rejects, API).")
    if a.write:
        write_files(rep)
        print(f"  Wrote {os.path.relpath(LEDGER_CSV, ROOT)} and {os.path.relpath(ROUND_TRIPS_CSV, ROOT)}")
    if a.log:
        fresh = log_alerts([f for f in rep["findings"] + rep["alerts"]])
        print(f"  Run log: {len(fresh)} new alert row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
