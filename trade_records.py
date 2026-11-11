"""Record-keeping and display helpers (no orders). Skip reasons, ledger snapshots,
plan-vs-fill recaps, ops failures, and contribution vs QQQ.

Every writer is fail-soft: callers should also wrap calls, but these functions never
raise into a trade or fill-check path.
"""
import json
import os
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import db

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
CT = ZoneInfo("America/Chicago")

SKIP_CSV = os.environ.get("STOCK_ANALYSIS_DECISION_SKIPS") or os.path.join(REPORTS, "decision_skips.csv")
SKIP_COLS = ["Date", "Symbol", "Rank", "Reason"]
LEDGER_HISTORY = os.environ.get("STOCK_ANALYSIS_LEDGER_HISTORY") or os.path.join(REPORTS, "ledger_history")
RECAP_CSV = os.environ.get("STOCK_ANALYSIS_FILL_RECAP") or os.path.join(REPORTS, "fill_recap.csv")
RECAP_COLS = ["As_Of", "Symbol", "Side", "Planned_Shares", "Filled_Shares", "Leftover",
              "Plan_Price", "Fill_Price", "Slippage_$", "Status", "In_Book"]
RECAP_SUM_CSV = os.environ.get("STOCK_ANALYSIS_FILL_RECAP_SUMMARY") or os.path.join(REPORTS, "fill_recap_summary.csv")
RECAP_SUM_COLS = ["As_Of", "Planned", "Filled", "Leftover_N", "Leftover", "Slippage_$", "Book", "Cash", "Message"]
RUN_LOG = os.environ.get("STOCK_ANALYSIS_RUN_LOG") or os.path.join(REPORTS, "run_log.csv")
PENDING_JSON = os.environ.get("STOCK_ANALYSIS_PENDING_ORDERS") or os.path.join(REPORTS, "live_pending_orders.json")
PLAN_CSV = os.environ.get("STOCK_ANALYSIS_SLEEVE_PLAN") or os.path.join(REPORTS, "sleeve_plan.csv")
ORDER_LOG = os.path.join(REPORTS, "live_orders_log.csv")
LEDGER_JSON = os.environ.get("STOCK_ANALYSIS_SLEEVE_LEDGER") or os.path.join(REPORTS, "sleeve_ledger.json")
MONEY_RUNS = {"Trade", "Fill check", "Watchdog", "Fri rebalance"}


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _day(x):
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return None
    t = pd.Timestamp(x)
    return t.tz_localize(None).normalize() if t.tzinfo is None else t.tz_convert(CT).tz_localize(None).normalize()


# ----------------------------------------------------------------------------- 1. skip reasons + ledger history
def write_skips(day, skipped, path=None):
    """Replace skip rows for `day` (other dates kept). `skipped` is a list of dicts
    with Symbol / Rank / Reason (Rank optional). Returns the path or None."""
    path = path or SKIP_CSV
    d = _day(day)
    if d is None:
        return None
    day_s = d.date().isoformat()
    rows = []
    for item in skipped or []:
        if isinstance(item, str):
            rows.append({"Date": day_s, "Symbol": item, "Rank": None, "Reason": ""})
            continue
        if not isinstance(item, dict):
            continue
        sym = str(item.get("Symbol") or item.get("symbol") or "").upper().strip()
        if not sym:
            continue
        rank = item.get("Rank", item.get("rank"))
        try:
            rank = int(rank) if rank is not None and str(rank) not in ("", "?", "nan") else None
        except (TypeError, ValueError):
            rank = None
        rows.append({"Date": day_s, "Symbol": sym, "Rank": rank,
                     "Reason": str(item.get("Reason") or item.get("reason") or "").strip()})
    prev = pd.DataFrame(columns=SKIP_COLS)
    if db.exists(path):
        try:
            prev = db.read_csv(path)
        except Exception:
            prev = pd.DataFrame(columns=SKIP_COLS)
    if len(prev):
        prev = prev[pd.to_datetime(prev["Date"], errors="coerce").dt.strftime("%Y-%m-%d") != day_s]
    out = pd.concat([prev, pd.DataFrame(rows, columns=SKIP_COLS)], ignore_index=True) if rows or len(prev) else pd.DataFrame(columns=SKIP_COLS)
    if len(out):
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out = out.sort_values(["Date", "Rank", "Symbol"], kind="stable")
    out[SKIP_COLS].pipe(db.to_csv, path, index=False)
    return path


def load_skips(day=None, path=None):
    """Skip rows for `day`, or all rows if day is None. Empty frame if none."""
    path = path or SKIP_CSV
    if not db.exists(path):
        return pd.DataFrame(columns=SKIP_COLS)
    df = db.read_csv(path)
    if df is None or not len(df):
        return pd.DataFrame(columns=SKIP_COLS)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if day is not None:
        d = _day(day)
        df = df[df["Date"] == d] if d is not None else df.iloc[0:0]
    return df.reset_index(drop=True)


def archive_ledger(led, src_path, dest_dir=None):
    """Copy the just-saved ledger to dest_dir/sleeve_ledger_YYYY-MM-DD.json when this
    save is a same-day rebalance (does not overwrite an older Friday after a later edit)."""
    dest_dir = dest_dir or LEDGER_HISTORY
    day = str((led or {}).get("day") or "")
    if not day:
        return None
    today = datetime.now(CT).date().isoformat()
    if day != today:
        return None
    halves = (led or {}).get("halves") or {}
    if not any((h or {}).get("rebalanced") for h in halves.values()):
        return None
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"sleeve_ledger_{day}.json")
    if src_path and os.path.exists(src_path):
        shutil.copyfile(src_path, dest)
    else:
        with open(dest, "w") as f:
            json.dump(led, f, indent=1, sort_keys=True)
    return dest


def after_ledger_save(led, path, skipped=None):
    """Called from sleeves.save_ledger (display / record keeping only)."""
    dest = archive_ledger(led, path)
    day = (led or {}).get("day")
    if skipped:
        write_skips(day, skipped)
    return dest


# ----------------------------------------------------------------------------- 2. ops banner (failed money jobs + leftovers)
def pending_leftovers(path=None):
    """List of pending order dicts, or []."""
    path = path or PENDING_JSON
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            pend = json.load(f)
    except (OSError, ValueError):
        return []
    if not isinstance(pend, dict):
        return []
    return [o for o in (pend.get("orders") or []) if isinstance(o, dict)]


def failed_money_runs(days=14, now=None, path=None):
    """run_log rows in the last `days` days whose run is a money-moving job and status is failed."""
    path = path or RUN_LOG
    cols = ["time_ct", "run", "status", "money_moved", "message"]
    empty = pd.DataFrame(columns=cols)
    if not db.exists(path):
        return empty
    try:
        log = db.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return empty
    if log is None or not len(log):
        return empty
    t = pd.to_datetime(log["time_ct"], errors="coerce")
    try:
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert(CT).dt.tz_localize(None)
    except (TypeError, AttributeError, ValueError):
        t = pd.to_datetime(log["time_ct"], errors="coerce")
    now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp(datetime.now(CT))
    if now_ts.tzinfo is not None:
        now_ts = now_ts.tz_convert(CT).tz_localize(None)
    else:
        now_ts = now_ts.tz_localize(None) if now_ts.tzinfo else now_ts
    cutoff = now_ts - pd.Timedelta(days=int(days))
    run = log["run"].astype(str).str.strip()
    status = log["status"].astype(str).str.strip().str.lower()
    mask = run.isin(MONEY_RUNS) & status.eq("failed") & t.ge(cutoff)
    keep = [c for c in cols if c in log.columns]
    return log.loc[mask, keep].copy()


def ops_issues(days=14, now=None, run_log=None, pending_path=None):
    """(failed DataFrame, leftover list). Empty/empty when everything is clean."""
    failed = failed_money_runs(days=days, now=now, path=run_log)
    left = pending_leftovers(pending_path)
    return failed, left


def ops_banner_lines(days=14, now=None, run_log=None, pending_path=None):
    """Plain-language lines for the dashboard banner, or [] when clean."""
    failed, left = ops_issues(days=days, now=now, run_log=run_log, pending_path=pending_path)
    lines = []
    for r in failed.itertuples(index=False):
        msg = " ".join(str(getattr(r, "message", "") or "").split())
        lines.append(f"{r.time_ct}  {r.run} failed: {msg}")
    if left:
        bits = [f"{str(o.get('side') or '').upper()} {str(o.get('symbol') or '?')} {o.get('qty')}"
                for o in left]
        lines.append("Pending leftover orders: " + ", ".join(bits) + ".")
    return lines


# ----------------------------------------------------------------------------- 4. plan vs fill recap
def _completed(status):
    s = str(status or "")
    return "COMPLETED" in s.upper() or "FILLED" in s.upper()


def build_fill_recap(as_of=None, plan_path=None, log_path=None, ledger_path=None, pending_path=None):
    """Build a (detail, summary_row) recap from existing plan / order-log / ledger data.
    No broker calls. as_of defaults to the latest As_Of in the order log (or the ledger day)."""
    plan_path = plan_path or PLAN_CSV
    log_path = log_path or ORDER_LOG
    ledger_path = ledger_path or LEDGER_JSON
    pending_path = pending_path or PENDING_JSON

    log = pd.DataFrame()
    if db.exists(log_path):
        try:
            log = db.read_csv(log_path)
        except Exception:
            log = pd.DataFrame()
    if log is not None and len(log) and "As_Of" in log.columns:
        asofs = pd.to_datetime(log["As_Of"], errors="coerce").dropna()
    else:
        asofs = pd.Series(dtype="datetime64[ns]")

    if as_of is None:
        if len(asofs):
            as_of = asofs.max()
        else:
            led = _safe(lambda: json.load(open(ledger_path))) if os.path.exists(ledger_path) else None
            as_of = (led or {}).get("day")
    d = _day(as_of)
    if d is None:
        return pd.DataFrame(columns=RECAP_COLS), {}
    day_s = d.date().isoformat()

    plan = pd.DataFrame()
    if db.exists(plan_path):
        try:
            plan = db.read_csv(plan_path)
        except Exception:
            plan = pd.DataFrame()

    day_log = pd.DataFrame()
    if log is not None and len(log) and "As_Of" in log.columns:
        day_log = log[pd.to_datetime(log["As_Of"], errors="coerce").dt.strftime("%Y-%m-%d") == day_s].copy()

    planned = {}
    if plan is not None and len(plan) and "Symbol" in plan.columns:
        for r in plan.itertuples(index=False):
            sym = str(getattr(r, "Symbol", "")).upper()
            held = float(getattr(r, "Held_Shares", 0) or 0)
            tgt = float(getattr(r, "Target_Shares", 0) or 0)
            px = getattr(r, "Price", None)
            diff = tgt - held
            side = "BUY" if diff > 1e-6 else "SELL" if diff < -1e-6 else "HOLD"
            planned[sym] = {"Side": side, "Planned_Shares": abs(diff) if side in ("BUY", "SELL") else 0.0,
                            "Plan_Price": px, "Target": tgt}

    book, cash = set(), None
    hist = os.path.join(os.environ.get("STOCK_ANALYSIS_LEDGER_HISTORY") or LEDGER_HISTORY,
                        f"sleeve_ledger_{day_s}.json")
    led_path = hist if os.path.exists(hist) else (ledger_path if os.path.exists(ledger_path) else None)
    if led_path:
        try:
            led = json.load(open(led_path))
            if str(led.get("day") or "") == day_s:
                for h in (led.get("halves") or {}).values():
                    book.update(str(s).upper() for s, q in ((h or {}).get("shares") or {}).items() if float(q or 0) > 0)
                    if cash is None and h.get("cash") is not None:
                        cash = float(h["cash"])
                    elif h.get("cash") is not None:
                        cash = (cash or 0) + float(h["cash"])
        except (OSError, ValueError):
            pass

    # Aggregate fills per symbol/side for that As_Of
    fills = {}
    def _col(row, *names):
        for n in names:
            if n in row.index and pd.notna(row[n]):
                return row[n]
        return None

    if day_log is not None and len(day_log):
        for _, row in day_log.iterrows():
            sym = str(row.get("Symbol") or "").upper()
            side = str(row.get("Side") or "").upper()
            if not sym or side not in ("BUY", "SELL"):
                continue
            key = (sym, side)
            rec = fills.setdefault(key, {"Filled_Shares": 0.0, "Slippage_$": 0.0, "Fill_Price": None,
                                         "Plan_Price": None, "Status": []})
            sh = float(row.get("Shares") or 0)
            st = str(row.get("Status") or "")
            rec["Status"].append(st)
            if _completed(st):
                rec["Filled_Shares"] += sh
            slip = _col(row, "Cost_$", "Slippage_$", "Slippage_vs_Plan_$")
            if slip is not None:
                try:
                    rec["Slippage_$"] += float(slip)
                except (TypeError, ValueError):
                    pass
            fp = _col(row, "Fill_Price")
            if fp is not None:
                rec["Fill_Price"] = fp
            pp = _col(row, "Limit", "Plan_Price")
            if rec["Plan_Price"] is None and pp is not None:
                rec["Plan_Price"] = pp

    leftovers = pending_leftovers(pending_path)
    leftover_by = {}
    for o in leftovers:
        ev = str(o.get("evening_date") or o.get("as_of") or "")
        if ev and ev[:10] not in (day_s, ""):
            # still count if as_of was not asked to filter strictly; include all current leftovers
            # only when their evening_date matches, or evening_date is missing
            if ev[:10] != day_s:
                continue
        key = (str(o.get("symbol") or "").upper(), str(o.get("side") or "").upper())
        leftover_by[key] = leftover_by.get(key, 0.0) + float(o.get("qty") or 0)

    keys = set(fills) | {(s, planned[s]["Side"]) for s in planned if planned[s]["Side"] in ("BUY", "SELL")} | set(leftover_by)
    rows = []
    for sym, side in sorted(keys, key=lambda k: (0 if k[1] == "SELL" else 1, k[0])):
        pl = planned.get(sym, {})
        if pl.get("Side") and pl["Side"] != side and (sym, pl["Side"]) in keys:
            # plan side differs; still emit this fill/leftover row
            pass
        rec = fills.get((sym, side), {})
        planned_sh = pl.get("Planned_Shares", 0.0) if pl.get("Side") == side else 0.0
        filled = float(rec.get("Filled_Shares") or 0.0)
        if planned_sh <= 1e-9 and filled > 0:
            planned_sh = filled  # sleeve_plan is the ending book: exits of dropped names live only in the order log
        left = leftover_by.get((sym, side))
        if left is None:
            left = max(round(planned_sh - filled, 4), 0.0) if planned_sh else 0.0
        else:
            left = max(float(left), 0.0)
        if left < 1e-6:
            left = 0.0
        planned_sh = round(planned_sh, 4)
        filled = round(filled, 4)
        status = "; ".join(rec.get("Status") or []) or ("pending" if left else "")
        rows.append({
            "As_Of": day_s, "Symbol": sym, "Side": side,
            "Planned_Shares": planned_sh, "Filled_Shares": filled, "Leftover": left,
            "Plan_Price": pl.get("Plan_Price") if pl.get("Side") == side else rec.get("Plan_Price"),
            "Fill_Price": rec.get("Fill_Price"),
            "Slippage_$": rec.get("Slippage_$") or 0.0,
            "Status": status, "In_Book": int(sym in book) if book else None,
        })
    detail = pd.DataFrame(rows, columns=RECAP_COLS)
    leftover_rows = detail[detail["Leftover"].fillna(0) > 1e-6] if len(detail) else detail
    slip = float(detail["Slippage_$"].sum()) if len(detail) else 0.0
    n_plan = int((detail["Planned_Shares"].fillna(0) > 1e-6).sum()) if len(detail) else 0
    n_fill = int((detail["Filled_Shares"].fillna(0) > 1e-6).sum()) if len(detail) else 0
    leftover_names = ", ".join(f"{r.Side} {r.Symbol}" for r in leftover_rows.itertuples()) if len(leftover_rows) else ""
    book_txt = ", ".join(sorted(book)) if book else ""
    if leftover_names:
        msg = f"{n_fill} filled, leftover: {leftover_names}. Slippage ${slip:,.2f}."
    else:
        msg = f"{n_fill} filled, no leftovers. Slippage ${slip:,.2f}. Book: {book_txt or '—'}."
    summary = {"As_Of": day_s, "Planned": n_plan, "Filled": n_fill, "Leftover_N": int(len(leftover_rows)),
               "Leftover": leftover_names, "Slippage_$": round(slip, 2), "Book": book_txt,
               "Cash": cash, "Message": msg}
    return detail, summary


def write_fill_recap(as_of=None, **paths):
    """Build and persist the recap for `as_of` (replaces rows for that day). Returns (detail, summary)."""
    recap_path = paths.pop("recap_path", None) or RECAP_CSV
    sum_path = paths.pop("summary_path", None) or RECAP_SUM_CSV
    detail, summary = build_fill_recap(as_of=as_of, **paths)
    if not summary:
        return detail, summary
    day_s = summary["As_Of"]
    prev = pd.DataFrame(columns=RECAP_COLS)
    if db.exists(recap_path):
        try:
            prev = db.read_csv(recap_path)
        except Exception:
            prev = pd.DataFrame(columns=RECAP_COLS)
    if len(prev) and "As_Of" in prev.columns:
        prev = prev[pd.to_datetime(prev["As_Of"], errors="coerce").dt.strftime("%Y-%m-%d") != day_s]
    frames = [x for x in (prev, detail) if x is not None and len(x)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RECAP_COLS)
    out[RECAP_COLS].pipe(db.to_csv, recap_path, index=False)
    prev_s = pd.DataFrame(columns=RECAP_SUM_COLS)
    if db.exists(sum_path):
        try:
            prev_s = db.read_csv(sum_path)
        except Exception:
            prev_s = pd.DataFrame(columns=RECAP_SUM_COLS)
    if len(prev_s) and "As_Of" in prev_s.columns:
        prev_s = prev_s[pd.to_datetime(prev_s["As_Of"], errors="coerce").dt.strftime("%Y-%m-%d") != day_s]
    row = pd.DataFrame([summary], columns=RECAP_SUM_COLS)
    frames_s = [x for x in (prev_s, row) if x is not None and len(x)]
    (pd.concat(frames_s, ignore_index=True) if frames_s else row)[RECAP_SUM_COLS].pipe(db.to_csv, sum_path, index=False)
    return detail, summary


def load_fill_recap(as_of=None, path=None, summary_path=None):
    """(detail, summary_row_dict or {}) for `as_of`, or the latest day if as_of is None."""
    path = path or RECAP_CSV
    summary_path = summary_path or RECAP_SUM_CSV
    detail = pd.DataFrame(columns=RECAP_COLS)
    if db.exists(path):
        try:
            detail = db.read_csv(path)
        except Exception:
            detail = pd.DataFrame(columns=RECAP_COLS)
    summary = pd.DataFrame(columns=RECAP_SUM_COLS)
    if db.exists(summary_path):
        try:
            summary = db.read_csv(summary_path)
        except Exception:
            summary = pd.DataFrame(columns=RECAP_SUM_COLS)
    if as_of is None:
        dates = []
        if len(detail) and "As_Of" in detail.columns:
            dates.append(pd.to_datetime(detail["As_Of"], errors="coerce").max())
        if len(summary) and "As_Of" in summary.columns:
            dates.append(pd.to_datetime(summary["As_Of"], errors="coerce").max())
        dates = [x for x in dates if pd.notna(x)]
        as_of = max(dates) if dates else None
    d = _day(as_of)
    if d is None:
        return pd.DataFrame(columns=RECAP_COLS), {}
    day_s = d.date().isoformat()
    if len(detail) and "As_Of" in detail.columns:
        detail = detail[pd.to_datetime(detail["As_Of"], errors="coerce").dt.strftime("%Y-%m-%d") == day_s]
    else:
        detail = pd.DataFrame(columns=RECAP_COLS)
    row = {}
    if len(summary) and "As_Of" in summary.columns:
        s = summary[pd.to_datetime(summary["As_Of"], errors="coerce").dt.strftime("%Y-%m-%d") == day_s]
        if len(s):
            row = s.iloc[-1].to_dict()
    return detail.reset_index(drop=True), row


# ----------------------------------------------------------------------------- 5. contribution vs QQQ
def week_start(as_of=None):
    """Last Friday session strictly before a Friday as_of, otherwise the Friday on or before as_of.
    So on a Friday this still means 'the week since the last rebalance'."""
    d = _day(as_of) or pd.Timestamp(datetime.now(CT).date())
    if d.weekday() == 4:
        d = d - pd.Timedelta(days=1)
    is_sess = None
    try:
        import backtest_engine as be
        is_sess = be.is_session
    except Exception:
        is_sess = lambda x: pd.Timestamp(x).weekday() < 5
    for _ in range(14):
        if d.weekday() == 4 and bool(is_sess(d)):
            return d
        d -= pd.Timedelta(days=1)
    return d


def _px_on_or_after(closes, start):
    """First close on or after start, or None."""
    if closes is None or not len(closes):
        return None
    s = closes[closes.index >= pd.Timestamp(start).normalize()]
    if not len(s):
        return None
    v = s.iloc[0]
    return float(v) if pd.notna(v) and float(v) > 0 else None


def contribution_table(positions, prices, qqq, as_of=None, since=None, first_bought=None):
    """This week and since-`since` contribution of current holdings vs QQQ.

    positions: iterable of dicts with symbol/Stock, qty/Shares, current_price/Price,
               optional market_value, sector.
    prices: DataFrame Date x Symbol of closes (or long form with Date, Symbol, Close).
    qqq: Series of QQQ closes indexed by Date.
    first_bought: optional {symbol: date} so a name bought mid-period does not use earlier prices.
    Returns (by_holding, by_sector) DataFrames.
    """
    import sector_mapping as sm
    try:
        import backtest_engine as be
        since = since or be.FORWARD_START
    except Exception:
        since = since or "2026-10-29"
    as_of = _day(as_of)
    since_d = _day(since)
    wk = week_start(as_of)
    first_bought = {str(k).upper(): _day(v) for k, v in (first_bought or {}).items() if v is not None}

    px = prices
    if px is not None and len(px) and "Symbol" in getattr(px, "columns", []):
        wide = px.copy()
        wide["Date"] = pd.to_datetime(wide["Date"])
        wide = wide.pivot_table(index="Date", columns="Symbol", values="Close", aggfunc="last")
        px = wide
    if px is not None and len(px):
        px = px.copy()
        px.index = pd.to_datetime(px.index).tz_localize(None)
        if as_of is None:
            as_of = pd.Timestamp(px.index.max()).normalize()
    if as_of is None:
        as_of = pd.Timestamp(datetime.now(CT).date())

    if qqq is not None and len(qqq):
        qqq = qqq.copy()
        qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
        qqq = qqq.sort_index()

    def qqq_ret(start):
        if qqq is None or not len(qqq):
            return None
        a = _px_on_or_after(qqq, start)
        b = _px_on_or_after(qqq, as_of)
        # prefer last close on or before as_of for the end
        end = qqq[qqq.index <= as_of]
        b = float(end.iloc[-1]) if len(end) and pd.notna(end.iloc[-1]) else b
        if not a or not b:
            return None
        return b / a - 1.0

    rows = []
    for p in positions or []:
        if not isinstance(p, dict):
            continue
        sym = str(p.get("symbol") or p.get("Stock") or p.get("Symbol") or "").upper()
        if not sym or sym.startswith("TOTAL"):
            continue
        qty = p.get("qty", p.get("Shares"))
        try:
            qty = float(qty or 0)
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        now_px = p.get("current_price", p.get("Price"))
        try:
            now_px = float(now_px) if now_px is not None and pd.notna(now_px) else None
        except (TypeError, ValueError):
            now_px = None
        if (now_px is None or not now_px) and px is not None and sym in getattr(px, "columns", []):
            end = px.loc[px.index <= as_of, sym].dropna()
            now_px = float(end.iloc[-1]) if len(end) else None
        if not now_px:
            continue
        series = px[sym].dropna() if px is not None and sym in getattr(px, "columns", []) else pd.Series(dtype=float)
        series.index = pd.to_datetime(series.index)

        def start_px(period):
            start = period
            fb = first_bought.get(sym)
            if fb is not None and fb > period:
                start = fb
            return _px_on_or_after(series, start), start

        p_w, w_start = start_px(wk)
        p_s, s_start = start_px(since_d)
        week_pnl = (now_px - p_w) * qty if p_w else None
        since_pnl = (now_px - p_s) * qty if p_s else None
        week_ret = (now_px / p_w - 1) if p_w else None
        since_ret = (now_px / p_s - 1) if p_s else None
        q_w = qqq_ret(w_start)
        q_s = qqq_ret(s_start)
        rows.append({
            "Symbol": sym,
            "Sector": p.get("sector") or sm.symbol_sector.get(sym, "—"),
            "Shares": qty,
            "Price": now_px,
            "This week $": None if week_pnl is None else round(week_pnl, 2),
            "This week %": None if week_ret is None else round(week_ret * 100, 2),
            "vs QQQ this week pp": None if week_ret is None or q_w is None else round((week_ret - q_w) * 100, 2),
            "Since Oct 29 $": None if since_pnl is None else round(since_pnl, 2),
            "Since Oct 29 %": None if since_ret is None else round(since_ret * 100, 2),
            "vs QQQ since Oct 29 pp": None if since_ret is None or q_s is None else round((since_ret - q_s) * 100, 2),
        })
    hold = pd.DataFrame(rows)
    if not len(hold):
        empty_s = pd.DataFrame(columns=["Sector", "This week $", "Since Oct 29 $"])
        return hold, empty_s
    hold = hold.sort_values("This week $", ascending=False, kind="stable", na_position="last").reset_index(drop=True)
    # Portfolio / QQQ summary rows
    q_w = qqq_ret(wk)
    q_s = qqq_ret(since_d)
    port_w = hold["This week $"].sum(min_count=1)
    port_s = hold["Since Oct 29 $"].sum(min_count=1)
    # approximate portfolio % from dollar P&L / current value (not starting equity; labeled as $ first)
    value = sum(r["Shares"] * r["Price"] for r in rows)
    port_w_pct = (port_w / (value - port_w) * 100) if port_w is not None and value and (value - port_w) else None
    port_s_pct = (port_s / (value - port_s) * 100) if port_s is not None and value and (value - port_s) else None
    summary_rows = [
        {"Symbol": "Portfolio (current book)", "Sector": "—", "Shares": None, "Price": None,
         "This week $": None if port_w is None else round(float(port_w), 2),
         "This week %": None if port_w_pct is None else round(float(port_w_pct), 2),
         "vs QQQ this week pp": None if port_w_pct is None or q_w is None else round(float(port_w_pct) - q_w * 100, 2),
         "Since Oct 29 $": None if port_s is None else round(float(port_s), 2),
         "Since Oct 29 %": None if port_s_pct is None else round(float(port_s_pct), 2),
         "vs QQQ since Oct 29 pp": None if port_s_pct is None or q_s is None else round(float(port_s_pct) - q_s * 100, 2)},
        {"Symbol": "QQQ", "Sector": "—", "Shares": None, "Price": None,
         "This week $": None, "This week %": None if q_w is None else round(q_w * 100, 2),
         "vs QQQ this week pp": 0.0 if q_w is not None else None,
         "Since Oct 29 $": None, "Since Oct 29 %": None if q_s is None else round(q_s * 100, 2),
         "vs QQQ since Oct 29 pp": 0.0 if q_s is not None else None},
    ]
    hold = pd.concat([hold, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)
    sec = (hold[~hold["Symbol"].isin(["Portfolio (current book)", "QQQ"])]
           .groupby("Sector", dropna=False)[["This week $", "Since Oct 29 $"]].sum()
           .reset_index().sort_values("This week $", ascending=False, kind="stable"))
    return hold, sec


def contribution_from_store(positions, as_of=None, first_bought=None):
    """Load closes and QQQ from the local DB and build the contribution tables."""
    import backtest_engine as be
    since = be.FORWARD_START
    bars = db.load_bars(start=since)
    if bars is None or not len(bars):
        return pd.DataFrame(), pd.DataFrame()
    bench = None
    try:
        from dashboard.data import load_benchmarks
        bench = load_benchmarks()
    except Exception:
        if db.exists(os.path.join(REPORTS, "benchmark_prices.csv")):
            bench = db.read_csv(os.path.join(REPORTS, "benchmark_prices.csv"))
    qqq = None
    if bench is not None and len(bench) and "QQQ" in bench.columns:
        b = bench.copy()
        b["Date"] = pd.to_datetime(b["Date"])
        qqq = b.set_index("Date")["QQQ"].dropna().sort_index()
    return contribution_table(positions, bars, qqq, as_of=as_of, since=since, first_bought=first_bought)


# ----------------------------------------------------------------------------- CLI (offline, no orders)
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Offline record-keeping builders (no orders).")
    p.add_argument("--recap", nargs="?", const="latest", metavar="YYYY-MM-DD",
                   help="build and print the plan-vs-fill recap (default: latest As_Of); pass a date to select one")
    p.add_argument("--write", action="store_true", help="also write the recap tables")
    a = p.parse_args(argv)
    if a.recap:
        as_of = None if a.recap == "latest" else a.recap
        fn = write_fill_recap if a.write else build_fill_recap
        detail, summary = fn(as_of=as_of)
        print(summary.get("Message") or summary)
        if len(detail):
            print(detail.to_string(index=False))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
