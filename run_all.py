"""ONE command for the whole Stock Analysis pipeline:   python run_all.py      (or open run_all.ipynb and Run All)

It picks the mode by itself (clock in US Central time):
  FULL   on a strategy decision day after 3:15 PM CT, when the full online update has not run yet today. Decision days =
         the Friday rebalance (the week's last session, e.g. Thursday when Friday is a holiday) and the Mon/Wed checks
         (the first session on/after Mon/Wed, e.g. Tuesday after a Monday holiday) - backtest_engine.next_decision:
           fundamentals  company_report_autofetch.py - Alpha Vantage rotation, max 12 stocks = max 24 calls (free limit 25/day)
           processing    company_report_processing.ipynb + scoring company_report_scoring.ipynb (company reports)
           sentiment     sentiment_analysis.ipynb online - NewsAPI 97 calls (free limit 100/day -> never twice within 24 h)
                         + Finnhub 97 calls
           earnings      earnings_date.ipynb online - Finnhub 97 calls (throttled below 60/min) + yfinance
           main          main_signal_analysis.ipynb - fresh Alpaca daily bars (market data only), ranks, picks, mid-week check
           validate      the report files the app reads
  QUICK  any other time: main + validate only. Zero quota APIs (only Alpaca market-data bars).
Both end with a short summary: what ran, API calls used, data date, the holdings ALERT line and the next check / rebalance.
The upstream online steps (fundamentals, processing, scoring, sentiment, earnings) are optional: if one
fails, the pipeline warns and continues - main_signal_analysis.ipynb reuses the last good upstream tables,
so the evening trade still runs on fresh signals. Only a main or validate failure stops the pipeline.

    python run_all.py --full          # force the online update now (NewsAPI still refused if it ran within 24 h)
    python run_all.py --full --force-news     # ... and allow a second NewsAPI run (may exceed the free 100/day)
    python run_all.py --quick         # force quick mode
    python run_all.py --dry-run       # show the plan (mode, steps, expected API calls) and exit - nothing runs
    python run_all.py --positions f.csv       # alert on another positions file (default: my_positions.csv if it exists)
    python run_all.py --only main | --from scoring | --list | --backtests | --visualization | --keep-going
    python run_all.py --sync-live     # also refresh my_positions.csv from your Alpaca LIVE account (off by default)
    python run_all.py --trade         # after the pipeline, auto-trade the LIVE account (REAL MONEY) for the due decision
                                      # (pulls live positions + equity; Friday rebalance brings every target
                                      #  pick ('add' or 'hold') to Weight * equity - bought up or trimmed -
                                      #  unless within 1 point of equity (no-trade band); no new buy / top-up
                                      #  of a pick with earnings within 5 days; Mon/Wed = swaps/exits only;
                                      #  non-targets sold entirely; sells sent first; logs to Reports/live_orders_log.csv).
                                      # A decision (Fri rebalance / Mon-Wed check, holiday-shifted) is due at 3:15 PM CT
                                      # on its day and runs ONCE (run_state last_decision, Reports/.trade.lock):
                                      #  - its evening (until 7 PM CT, 4 PM on early closes): WHOLE-share extended-hours
                                      #    DAY limit orders at the close; the 9 AM --fill-check completes the rest (2 decimals);
                                      #  - missed (Mac asleep/off): caught up at the next regular session (9:00 AM CT to
                                      #    15 min before the close) with the missed decision's own picks, sized at current
                                      #    prices, sent at once as regular-hours market orders (2 decimals); evenings,
                                      #    nights, weekends and holidays wait for the next session;
                                      #  - superseded (skipped, logged, alert) once the next decision slot arrives.
    python run_all.py --trade --scheduled   # what launchd runs (see launchd/; at 3:15 PM, login, wake and every 30 min):
                                      # like --trade, but a start with nothing due prints one "idle:" line (no log, no
                                      # alert), and a failed decision is retried at most 3 times, 60+ min apart.
    python run_all.py --fill-check    # send the pending orders (Reports/live_pending_orders.json) as regular-hours market
                                      # orders: the evening's unfilled rest from 9:00 AM CT the next trading day, a
                                      # daytime catch-up's at once; only in regular hours and only until the next decision
                                      # slot (then dropped with an alert). Runs no notebooks, uses no quota APIs.
                                      # --scheduled: idle starts print one "idle:" line.

State: Reports/run_state.json - the runner's memory (all values are ISO timestamps unless noted; merged under a file lock):
    last_full_date / last_full_at .. last successful full online update (date + time)
    last_news_at ................... NewsAPI 24 h guard; stamped only AFTER the sentiment step's NewsAPI
                                     calls complete, so a crash can never cause a false lockout
    last_fundamentals_date ......... last Alpha Vantage rotation (date; max once/day)
    last_decision .................. date of the last decision traded (orders placed/staged, or nothing to trade) -
                                     it never runs again
    decision_attempts .............. {decision, n, at}: launchd attempts of a decision that failed before trading
    last_superseded ................ last missed decision reported as skipped (alert sent once)
    last_trade_at .................. last --trade that actually submitted or staged orders
    last_fill_check_at ............. last fill check that sent orders
    last_run_at / last_mode ......... last invocation (any mode)
Logs: Reports/logs/run_*.log (last 30 kept); launchd output: Reports/logs/launchd_*.log.
Schedule (launchd, see launchd/install_schedule.sh): --trade --scheduled Mon-Fri 3:15 PM CT, --fill-check --scheduled
Mon-Fri 9:00 AM CT; both also at login (RunAtLoad), once on wake for a slot missed in sleep, and every 30 min.
By default no orders are placed. Alpaca account endpoints are only called with --sync-live (3 read-only GETs via
alpaca_paper.py), --trade (paper_trade.auto_trade(): reads positions + equity, then submits extended-hours DAY
limit orders to the LIVE account (REAL MONEY)) or --fill-check (paper_trade.complete_unfilled_orders(): checks the evening orders
and completes unfilled remainders with regular-hours market orders; keys ALPACA_LIVE_KEY_ID /
ALPACA_LIVE_SECRET_KEY in .env).
"""
import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
LOG_DIR = os.path.join(REPORTS, "logs")
STATE_FILE = os.path.join(REPORTS, "run_state.json")
CHECKPOINT_FILE = os.path.join(REPORTS, ".pipeline_checkpoint.json")   # watchdog resume point (see pipeline_watchdog.py)
CT, ET = ZoneInfo("America/Chicago"), ZoneInfo("America/New_York")
FULL_AFTER = (15, 15)                  # 3:15 PM CT - the scheduled pipeline/trade time
NEWS_MIN_GAP_H = 24                    # NewsAPI free tier: 100 requests/day, one run = 97
BAR_FINAL_ET = (16, 30)                # the pipeline treats the daily bar as final after 4:30 PM ET (3:30 PM CT)
NB_TIMEOUT = 3600
KEEP_LOGS = 30

# (name, kind, target, when): when = "full" (full mode only), "always", or an opt-in flag name
STEPS = [
    ("fundamentals", "script", "company_report_autofetch.py", "full"),
    ("processing", "notebook", "company_report_processing.ipynb", "full"),
    ("scoring", "notebook", "company_report_scoring.ipynb", "full"),
    ("sentiment", "notebook", "sentiment_analysis.ipynb", "full"),
    ("earnings", "notebook", "earnings_date.ipynb", "full"),
    ("main", "notebook", "main_signal_analysis.ipynb", "always"),
    ("visualization", "notebook", "company_report_visualization.ipynb", "visualization"),
    ("backtest", "notebook", "backtest.ipynb", "backtests"),
    ("validate", "check", None, "always"),
]
EXPECTED_CALLS = {"fundamentals": "Alpha Vantage <= 24", "sentiment": "NewsAPI 97 + Finnhub 97", "earnings": "Finnhub 97"}

# Steps whose failure must NOT stop the pipeline or block the evening trade. Their
# outputs degrade gracefully: main_signal_analysis.ipynb reuses the last good
# upstream tables (weighted_sentiment.csv, balance_sheet_weights.csv,
# earnings_date.csv), which are informational inputs to scoring, not load-bearing.
OPTIONAL_STEPS = frozenset({"fundamentals", "processing", "scoring", "sentiment", "earnings"})


def _stop_on_failure(name, keep_going):
    """True when a failed step must stop the pipeline right away.

    Optional steps never stop it; --keep-going never stops it; anything else
    (main, validate) stops it so broken picks can't reach the trade.
    """
    return name not in OPTIONAL_STEPS and not keep_going


def _critical_failures(failures):
    """Failures that block the evening trade and set a nonzero exit code.

    Optional-step failures degrade gracefully (the main signal notebook reuses
    the last good tables); 'sync_live' is a read-only refresh, so neither
    blocks trading.
    """
    return [f for f in failures if f not in OPTIONAL_STEPS and f != "sync_live"]


def _trade_decision(failures):
    """Evening-trade guard: (proceed, reason).

    Only critical failures block trading. Optional-step failures (and the
    read-only sync_live) degrade gracefully - the main signals are fresh,
    so the trade proceeds.
    """
    crit = _critical_failures(failures)
    if crit:
        return False, "critical pipeline failures (%s) - no orders sent" % ", ".join(crit)
    if failures:
        return True, "proceeding despite non-critical failures (%s) - signals are fresh" % ", ".join(failures)
    return True, ""

# Files the app / downstream steps read, with the columns they rely on
REQUIRED = {
    "signal_analysis.csv": ["Date", "Symbol", "Close", "ma_10", "ma_30", "ma_50", "ma_100", "ma_200", "RSI", "macd", "MACD Signal",
                            "Technical_Score", "SentimentScore", "Fundamental_Weight", "combined_signal", "final_trade",
                            "Buy Streak", "Sell Streak", "Hold Streak", "is_earnings_date", "RS_Score", "Strategy_Score",
                            "Strategy_Rank", "Strategy_Weight", "Provisional_Weight", "Regime_On", "Rebalance_Day"],
    "strategy_picks.csv": ["As_Of", "Last_Rebalance", "Strategy", "Symbol", "Strategy_Weight", "Provisional_Weight", "Close"],
    "strategy_changes.csv": ["Date", "Symbol", "Status", "Reason", "Rank", "Score", "Sector", "Old_Weight", "New_Weight", "View"],
    "strategy_holdings.csv": ["Symbol", "Weight", "Entry_Date", "Entry_Price", "Close", "PnL_%", "Days_Held", "Vol_63d_%", "ATR_Stop"],
    "strategy_tracking.csv": ["Date", "Strategy", "QQQ", "SPY"],
    "strategy_decisions.csv": ["Date", "Symbol", "Status", "Reason", "Rank", "Score", "Old_Weight", "New_Weight", "Regime_On"],
    "strategy_midweek_check.csv": ["As_Of", "Event", "Event_Date", "Applies_To_Open", "Action", "Sell", "Buy", "Message",
                                   "Next_Message", "Rules"],
    "benchmark_prices.csv": ["Date", "SPY", "QQQ"],
    "weighted_sentiment.csv": ["Symbol", "SentimentScore"],
    "news_cleaned_df.csv": ["symbol", "date", "headline", "summary", "source", "sentiment_label"],
    "earnings_date.csv": ["Symbol", "Earnings Date", "Time"],
    "balance_sheet_weights.csv": ["Symbol", "Fundamental_Weight"],
    "backtest_summary.csv": ["Strategy", "Period", "Total Return %", "CAGR %", "Sharpe", "Max DD %"],
}
XLSX_COLUMNS = ["Symbol", "Sector", "CurrentPrice", "FairValue_Composite", "PE_Ratio", "PB_Ratio", "RevenueGrowth_YoY",
                "TTM_ROE", "TTM_NetProfitMargin", "Debt_to_Equity"]


# ----------------------------------------------------------------------------- notifications
NO_POPUPS_ENV = "STOCK_ANALYSIS_NO_POPUPS"   # set to 1 to log alerts without macOS pop-ups (the tests do)


def _clip(text, limit):
    """Collapse whitespace and cut to `limit` characters (ending in an ellipsis). Pure."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "\u2026"


def _notify(title, message, details=None):
    """Tell the user the pipeline needs attention (macOS notification + loud log line).

    Pop-up: title <= 40 chars, body <= 200 (macOS cuts longer text); `details` and the
    full text only go to the log. Best effort: the log line is the primary channel (it
    lands in Reports/logs/run_*.log and the launchd logs). STOCK_ANALYSIS_NO_POPUPS=1
    skips the pop-up (the tests set it). Never raises."""
    logging.info("NOTIFY: %s - %s%s", title, message, f" | details: {details}" if details else "")
    if os.environ.get(NO_POPUPS_ENV, "").strip() not in ("", "0"):
        return
    try:
        import subprocess as _sp
        safe_title = _clip(title, 40).replace('"', "'").replace("\\", "")
        safe_msg = _clip(message, 200).replace('"', "'").replace("\\", "")
        _sp.run(["osascript", "-e",
                 f'display notification "{safe_msg}" with title "{safe_title}" sound name "Basso"'],
                timeout=5, capture_output=True)
    except Exception:
        pass  # the log line above is the fallback


def _notebook_error_summary(output, notebook):
    """One line describing a failed nbconvert run: 'main_signal_analysis.ipynb failed at
    cell 12, line 5: NameError: name 'x' is not defined'. Falls back to the last
    nbconvert ERROR line, then to a pointer at the run log. Pure: safe to unit-test."""
    text = output or ""
    cell = re.search(r"Cell In\[(\d+)\], line (\d+)", text)
    err = None
    for line in reversed(text.splitlines()):
        s = line.strip().lstrip(">").strip()
        if ":" in s and re.match(r"^[\w.]+(Error|Exception|Warning):", s):
            err = s[:200]
            break
    if err is None:
        for line in reversed(text.splitlines()):
            if "ERROR |" in line:
                err = line.split("ERROR |", 1)[1].strip()[:200]
                break
    where = f"cell {cell.group(1)}, line {cell.group(2)}" if cell else "unknown location"
    return f"{notebook} failed at {where}: {err or 'see the run log for the traceback'}"


# ----------------------------------------------------------------------------- state + mode (pure logic, unit-tested)
def load_state(path=None):
    """The runner's memory. First run: seeded from the evidence on disk (sentiment_history.csv is written only by online
    NewsAPI runs; fetch_run_log.csv logs every Alpha Vantage attempt) so a same-day repeat is still refused."""
    try:
        with open(path or STATE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        pass
    state = {}
    hist = os.path.join(REPORTS, "sentiment_history.csv")
    if os.path.exists(hist):
        state["last_news_at"] = datetime.fromtimestamp(os.path.getmtime(hist), CT).isoformat(timespec="seconds")
    runlog = os.path.join(REPORTS, "fetch_run_log.csv")
    if os.path.exists(runlog):
        with open(runlog) as f:
            dates = [line.split(",")[1] for line in f.read().splitlines()[1:] if line.count(",") >= 3]
        if dates:
            state["last_fundamentals_date"] = max(dates)[:10]
    return state


def save_state(state, path=None):
    """Write run_state.json atomically (temp file + rename)."""
    path = path or STATE_FILE
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def update_state(path=None, **changes):
    """Merge `changes` into run_state.json under a file lock (re-read first), so the evening and morning jobs never
    overwrite each other's keys. Returns the merged state."""
    import fcntl
    path = path or STATE_FILE
    with open(os.path.join(os.path.dirname(path), ".run_state.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = load_state(path)
        state.update(changes)
        save_state(state, path)
    return state


def write_checkpoint(run_id, argv, mode, steps_done, failed_step=None, phase="steps", path=CHECKPOINT_FILE):
    """Record pipeline progress for pipeline_watchdog.py: which steps finished and where it stopped.

    Written after every step and before the trade/fill-check phases; cleared on a clean run.
    Best effort - never raises, so a checkpoint failure can never break the pipeline itself. """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"run_id": run_id, "argv": list(argv), "mode": mode, "phase": phase,
                       "steps_done": list(steps_done), "failed_step": failed_step,
                       "updated_at": datetime.now(CT).isoformat(timespec="seconds")}, f)
        os.replace(tmp, path)
    except OSError:
        pass


def read_checkpoint(path=CHECKPOINT_FILE):
    """The last checkpoint dict, or {} when there is none (or it is corrupt)."""
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def clear_checkpoint(path=CHECKPOINT_FILE):
    """Remove the checkpoint after a clean run (exit 0). Best effort - never raises."""
    try:
        os.remove(path)
    except OSError:
        pass


def is_trading_day(d):
    """NYSE session (full-day holidays excluded), same calendar as the strategy."""
    import backtest_engine as be
    return be.is_session(d)


def decision_day(d):
    """'full rebalance' / 'mid-week check' when date d is a strategy decision session, else None.

    Same calendar as the strategy (backtest_engine.next_decision): the rebalance is the week's last session (Thursday
    when Friday is a holiday) and a check is the first session on/after each Mon/Wed (Tuesday after a Monday holiday)."""
    import backtest_engine as be
    return be.decision_kind(d)


def full_window(now_ct):
    """(True, why) when the automatic full update / trade is due by the clock: a decision day after 3:15 PM CT."""
    if not is_trading_day(now_ct.date()):
        return False, f"{now_ct:%a %b %d} is not a trading day"
    kind = decision_day(now_ct.date())
    if not kind:
        return False, f"{now_ct:%a %b %d} is not a decision day (no rebalance or mid-week check)"
    if (now_ct.hour, now_ct.minute) < FULL_AFTER:
        return False, f"before {FULL_AFTER[0] - 12}:{FULL_AFTER[1]:02d} PM CT"
    return True, f"{now_ct:%a} {kind} after {FULL_AFTER[0] - 12}:{FULL_AFTER[1]:02d} PM CT"


def choose_mode(now_ct, state, force_full=False, force_quick=False, decision=None):
    """'full' or 'quick' plus a plain reason. decision = a missed decision being caught up now (see decision_gate):
    full when no full update ran since that decision's 3:15 PM slot."""
    if force_quick:
        return "quick", "--quick"
    if force_full:
        return "full", "--full"
    due, why = full_window(now_ct)
    if decision is not None and not due:
        import backtest_engine as be
        label = f"catch-up of the {decision:%a %b %d} {be.decision_kind(decision)}"
        last = _parse_ct(state.get("last_full_at"))
        if last is not None and last >= be.decision_slot(decision):
            return "quick", f"{label}; the full update already ran {last:%a %I:%M %p}"
        return "full", label
    if not due:
        return "quick", why
    if state.get("last_full_date") == now_ct.date().isoformat():
        return "quick", f"the full update already ran today ({state.get('last_full_at', '')})"
    return "full", why


def news_allowed(now_ct, state, force_news=False):
    """(allowed, reason). NewsAPI runs at most once per NEWS_MIN_GAP_H hours unless forced."""
    last = state.get("last_news_at")
    if force_news or not last:
        return True, "--force-news" if (force_news and last) else "ok"
    gap_h = (now_ct - datetime.fromisoformat(last)).total_seconds() / 3600
    if gap_h < NEWS_MIN_GAP_H:
        return False, (f"NewsAPI already ran {gap_h:.1f} h ago ({last[:16]}); the free limit is 100 calls/day and one run "
                       f"uses 97 - skipped (use --force-news to override)")
    return True, "ok"


def fundamentals_allowed(now_ct, state, force=False):
    """(allowed, reason): the Alpha Vantage rotation runs at most once per day."""
    last = state.get("last_fundamentals_date")
    if force or last != now_ct.date().isoformat():
        return True, "ok"
    return False, "the Alpha Vantage rotation already ran today (free limit 25 calls/day) - skipped (use --force-fundamentals to override)"


# ----------------------------------------------------------------------------- decisions: run once, catch up if missed
SESSION_FROM = (9, 0)            # catch-up / completion market orders from 9:00 AM CT (30 min after the open) ...
SESSION_STOP_MIN = 15            # ... until 15 min before the close (2:45 PM CT; 11:45 AM on early-close days)
MAX_DECISION_ATTEMPTS = 3        # launchd retries a decision whose run failed at most 3 times ...
RETRY_GAP_MIN = 60               # ... at least 60 min apart


def _parse_ct(text):
    """ISO text -> tz-aware datetime (CT when it has no zone); None when missing or invalid."""
    try:
        t = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=CT)


def in_session(now_ct):
    """True in the regular-hours window for market orders: a trading day from 9:00 AM CT until 15 min before the close."""
    import backtest_engine as be
    d = now_ct.date()
    if not is_trading_day(d):
        return False
    close = datetime(d.year, d.month, d.day, 12 if be.is_early_close(d) else 15, 0, tzinfo=CT)
    start = datetime(d.year, d.month, d.day, *SESSION_FROM, tzinfo=CT)
    return start <= now_ct < close - timedelta(minutes=SESSION_STOP_MIN)


def next_session_start(now_ct):
    """The next 9:00 AM CT on a trading day at or after now_ct (today's if it is still ahead)."""
    d = now_ct.date()
    while True:
        start = datetime(d.year, d.month, d.day, *SESSION_FROM, tzinfo=CT)
        if is_trading_day(d) and start >= now_ct:
            return start
        d += timedelta(days=1)


def decision_gate(now_ct, state, scheduled=False):
    """(D, how, why): what a --trade run does now. Also shown by --dry-run, so the schedule can be tested safely.

    D = the latest decision (Timestamp) whose 3:15 PM CT slot has passed; None when it already ran (run_state
    'last_decision' - a decision never runs twice) or, for launchd runs, when its retries are used up or too recent.
    how = 'evening': D is today and extended hours are still open -> pipeline, then whole-share extended-hours limit
                     orders; the 9 AM check completes the fractional rest (the normal evening run).
          'session': D was missed (Mac asleep/off) and the market is open -> pipeline, then regular-hours market orders
                     in 2-decimal shares now, from D's picks (its point-in-time ranks) sized at current prices.
          'wait':    D was missed and the market is closed (evening, night, weekend, holiday) -> nothing now; it runs at
                     the next session's 9:00 AM CT, unless the next decision slot comes first (then it is superseded).
    """
    import backtest_engine as be
    import paper_trade
    D = be.last_decision_date(now_ct)
    if D is None:
        return None, None, "no decision found in the last 15 days"
    label = f"{D:%a %b %d} {be.decision_kind(D)}"
    done = str(state.get("last_decision") or "")
    if done >= D.date().isoformat():
        return None, None, f"the {label} already ran (last_decision {done})"
    att = state.get("decision_attempts") or {}
    if scheduled and att.get("decision") == D.date().isoformat():
        last = _parse_ct(att.get("at"))
        if int(att.get("n", 0)) >= MAX_DECISION_ATTEMPTS:
            return None, None, (f"the {label} failed {att.get('n')} times - no more automatic retries "
                                f"(run `python run_all.py --trade` by hand once fixed)")
        if last is not None and now_ct < last + timedelta(minutes=RETRY_GAP_MIN):
            return None, None, f"the {label} failed at {last:%I:%M %p}; next retry after {last + timedelta(minutes=RETRY_GAP_MIN):%I:%M %p} CT"
    until = be.next_decision_slot(now_ct)
    sup = f"superseded at {until:%a %b %d %I:%M %p} CT" if until else ""
    if now_ct.date() == D.date() and not paper_trade._past_evening_cutoff(now_ct):
        return D, "evening", f"{label} after 3:15 PM CT"
    if in_session(now_ct):
        return D, "session", f"catch-up of the missed {label} now, regular-hours market orders ({sup} if not run)"
    start = next_session_start(now_ct)
    if until and start >= until:
        return None, None, f"missed {label}: no session before the next decision ({sup})"
    return D, "wait", f"missed {label}: waiting for the market - runs {start:%a %b %d} 9:00 AM CT ({sup} if not run)"


def superseded_decision(now_ct, state):
    """The latest decision whose catch-up window closed without a run and was not reported yet (a Timestamp), else None.

    A decision may be caught up until the next decision slot; after that the newer decision replaces it."""
    import backtest_engine as be
    D = be.last_decision_date(now_ct)
    if D is None:
        return None
    prev = be.last_decision_date(be.decision_slot(D) - timedelta(minutes=1))
    if prev is None:
        return None
    p = prev.date().isoformat()
    if str(state.get("last_decision") or "") >= p or str(state.get("last_superseded") or "") >= p:
        return None
    return prev


TRADE_LOCK = os.path.join(REPORTS, ".trade.lock")
FILL_LOCK = os.path.join(REPORTS, ".fill_check.lock")
TRADE_LOCK_STALE_S = 6 * 3600              # a lock of a dead process, or older than this, is taken over (crashed run)


def acquire_trade_lock(path=None):
    """(True, None) when this process now owns the --trade lock; (False, reason) when another
    live --trade run holds it. The lock serializes overlapping --trade processes so two of them
    can never submit duplicate orders. A stale lock (dead PID or older than TRADE_LOCK_STALE_S)
    is taken over with a warning instead of blocking forever."""
    import errno
    path = path or TRADE_LOCK
    payload = json.dumps({"pid": os.getpid(), "at": datetime.now(CT).isoformat(timespec="seconds")})
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        return True, None
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        with open(path) as f:
            old = json.load(f)
        pid, at = int(old.get("pid", 0)), old.get("at", "")
        t = datetime.fromisoformat(at) if at else None
        if t is not None and t.tzinfo is None:
            t = t.replace(tzinfo=CT)
        age = (datetime.now(CT) - t).total_seconds() if t else float("inf")
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False
        if alive and age < TRADE_LOCK_STALE_S:
            return False, f"locked by live process {pid} since {at}"
        logging.warning("TRADE LOCK: taking over stale lock (pid %s, at %s)", pid, at or "?")
    except (OSError, ValueError):
        logging.warning("TRADE LOCK: taking over unreadable lock file")
    try:
        os.remove(path)
    except OSError:
        pass
    return acquire_trade_lock(path)


def release_trade_lock(path=None):
    """Release the --trade lock (best effort)."""
    try:
        os.remove(path or TRADE_LOCK)
    except OSError:
        pass


FILL_CHECK_AFTER = (9, 0)                # 9:00 AM CT - the morning fill check opens after this


def next_trading_day_after(d):
    """Next NYSE trading day strictly after date d."""
    n = d + timedelta(days=1)
    while not is_trading_day(n):
        n += timedelta(days=1)
    return n


def fill_check_allowed(now_ct, pending_path=None):
    """(allowed, reason, needs_investigation): may the fill check send its market orders now?

    Orders from an evening run open at 9:00 AM CT on the next trading day; orders staged by a daytime catch-up
    (send_now) at once. Either way only in regular hours (in_session: 9:00 AM CT to 15 min before the close). Rows
    whose decision was superseded (the next decision slot passed) are dropped before this check by
    paper_trade.drop_superseded_orders, so nothing old is ever replayed.
    needs_investigation is True when a human must look before re-running (corrupt file, invalid date) - the caller
    must warn, notify, and exit nonzero. It is an explicit flag, not a substring of the reason, so rewording a message
    cannot silently downgrade the safety behavior."""
    import paper_trade
    path = pending_path or paper_trade.PENDING_ORDERS_JSON
    # A corrupt backup from an earlier run must re-alert until it is investigated and removed: the first run moved it
    # aside and exited nonzero, but without this check the next run would see "no pending file" and exit 0, silently
    # orphaning the evening's orders.
    leftovers = sorted(glob.glob(path + ".corrupt_*"))
    if leftovers:
        return False, (f"uninvestigated corrupt pending file(s) from an earlier run: "
                       f"{', '.join(leftovers)} - investigate before re-running; "
                       f"the evening's orders were NOT completed"), True
    try:
        with open(path) as f:
            pend = json.load(f)
    except OSError:
        return False, "no pending fill check (no orders awaiting completion)", False
    except ValueError:
        # A corrupt pending file must never be silently treated as "no pending orders" - back it up loudly and stop.
        bad = path + f".corrupt_{datetime.now(CT):%Y%m%d_%H%M%S}"
        try:
            os.replace(path, bad)
        except OSError:
            bad = path
        return False, (f"the pending fill-check file is corrupt (moved to {bad}) - investigate it "
                       f"before re-running; the evening's orders were NOT completed"), True
    if not isinstance(pend, dict) or not pend.get("orders"):
        return False, "no pending fill check (order list is empty)", False
    try:
        evening = datetime.fromisoformat(pend["evening_date"]).date()
    except (KeyError, ValueError, TypeError):
        return False, ("the pending fill-check file has no valid evening_date - investigate "
                       "Reports/live_pending_orders.json before re-running"), True
    if not pend.get("send_now"):
        d = next_trading_day_after(evening)
        earliest = datetime(d.year, d.month, d.day, FILL_CHECK_AFTER[0], FILL_CHECK_AFTER[1], tzinfo=CT)
        if now_ct < earliest:
            return False, (f"fill check for the {evening:%a %b %d} evening run opens "
                           f"{earliest:%a %b %d} at 9:00 AM CT"), False
    if not in_session(now_ct):
        return False, (f"{now_ct:%a %b %d %I:%M %p} CT is outside regular hours (9:00 AM CT to 15 min before the close) "
                       f"- the orders wait for the next session"), False
    return True, "ok", False


def fill_check_idle(now_ct):
    """Plain reason when a scheduled fill check has nothing to do now (then it writes no log and sends no alert),
    else None (orders to send, superseded rows to drop, or a problem to report)."""
    import paper_trade
    if paper_trade.superseded_orders(now_ct):
        return None
    ok, reason, needs_investigation = fill_check_allowed(now_ct)
    return None if ok or needs_investigation else reason


def next_full_update(now_ct, state):
    """Plain text: when `python run_all.py` will next do the full online update by itself."""
    for k in range(0, 10):
        d = (now_ct + timedelta(days=k)).replace(hour=FULL_AFTER[0], minute=FULL_AFTER[1], second=0, microsecond=0)
        if full_window(d)[0] and d.date().isoformat() != state.get("last_full_date"):
            return ("today" if k == 0 else f"{d:%a %b} {d.day}") + " after 3:15 PM CT"
    return "the next decision day after 3:15 PM CT"


# ----------------------------------------------------------------------------- execution
def setup_logging():
    """Log to the console and to Reports/logs/run_<time>.log (keeps the newest KEEP_LOGS logs)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    for old in sorted(glob.glob(os.path.join(LOG_DIR, "run_*.log")))[:-KEEP_LOGS + 1]:
        os.remove(old)
    path = os.path.join(LOG_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True,
                        handlers=[logging.FileHandler(path), logging.StreamHandler(sys.stdout)])
    return path


def run_cmd(cmd, env):
    """Run a subprocess, stream its output into the log. Returns (exit code, captured output text)."""
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if line.strip():
            logging.info("    %s", line.rstrip())
    return proc.wait(), "".join(lines)


def run_notebook(path, env):
    """Execute a notebook in place with nbconvert."""
    return run_cmd([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    f"--ExecutePreprocessor.timeout={NB_TIMEOUT}", path], env)


def read_calls(path):
    """Sum the API request counts the steps appended to `path` (one JSON dict per line).

    A corrupt line (e.g. a notebook killed mid-write) is skipped, never fatal: this runs in the
    final summary, after the trade, so it must not turn a good run into a reported failure."""
    calls = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    items = json.loads(line).items()
                except ValueError:
                    continue
                for k, v in items:
                    try:
                        calls[k] = calls.get(k, 0) + int(v)
                    except (TypeError, ValueError):
                        continue
        os.remove(path)
    return calls


def validate():
    """Every report the app reads exists and has its columns. Returns a list of problems."""
    import pandas as pd
    problems = []
    for name, cols in REQUIRED.items():
        path = os.path.join(REPORTS, name)
        if not os.path.exists(path):
            problems.append(f"{name}: missing")
            continue
        df = pd.read_csv(path, nrows=200)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            problems.append(f"{name}: missing columns {missing}")
        elif df.empty and name != "news_cleaned_df.csv":
            problems.append(f"{name}: no rows")
    xlsx = os.path.join(REPORTS, "complete_company_analysis.xlsx")
    if not os.path.exists(xlsx):
        problems.append("complete_company_analysis.xlsx: missing")
    else:
        latest = pd.read_excel(xlsx, sheet_name="2_Latest_Quarter_Complete", nrows=5)
        missing = [c for c in XLSX_COLUMNS if c not in latest.columns]
        if missing:
            problems.append(f"complete_company_analysis.xlsx: missing columns {missing}")
    sig_path = os.path.join(REPORTS, "signal_analysis.csv")
    if os.path.exists(sig_path):
        sig = pd.read_csv(sig_path, usecols=["Date", "Symbol"])
        if sig.duplicated(["Date", "Symbol"]).any():
            problems.append("signal_analysis.csv: duplicate (Date, Symbol) rows")
    logging.info("  %d report files checked, %d problem(s)", len(REQUIRED) + 1, len(problems))
    return problems


def wait_for_final_bar(mode, dry_run=False):
    """Full mode on a decision day: the decision needs today's completed daily bar (final after 4:30 PM ET)."""
    now = datetime.now(ET)
    ready = now.replace(hour=BAR_FINAL_ET[0], minute=BAR_FINAL_ET[1] + 1, second=0, microsecond=0)
    if mode != "full" or now >= ready or not is_trading_day(now.date()):
        return
    wait = (ready - now).total_seconds()
    if wait > 20 * 60:
        return
    logging.info("  waiting %.0f min for today's final daily bar (after 4:30 PM ET = 3:30 PM CT) ...", wait / 60)
    if not dry_run:
        time.sleep(wait)


def data_summary():
    """(data date, decision lines, next-decision line) from the pipeline outputs."""
    import pandas as pd
    out = {"data_date": None, "decisions": [], "next": None, "rules": None}
    path = os.path.join(REPORTS, "strategy_midweek_check.csv")
    if os.path.exists(path):
        m = pd.read_csv(path)
        if len(m):
            # A malformed report must degrade the summary, never crash the pipeline (validate already flags it).
            get = lambda c: m[c].iloc[0] if c in m.columns else None
            as_of = get("As_Of")
            out.update(data_date=str(as_of)[:10] if as_of is not None else None,
                       next=get("Next_Message"), rules=get("Rules"))
            if "Message" in m.columns:
                tag = "Action" if "Action" in m.columns else ("Status" if "Status" in m.columns else None)
                out["decisions"] = [
                    f"{'>> ' if getattr(r, 'Is_Latest', False) else '   '}{getattr(r, 'Message', '')}"
                    + (f" [{getattr(r, tag)}]" if tag and getattr(r, tag, None) is not None else "")
                    for r in m.itertuples()]
    return out


def sync_live(positions_path=None):
    """Optional (--sync-live): refresh my_positions.csv + Reports/live_*.csv from the Alpaca LIVE account (read-only)."""
    import alpaca_paper
    kw = {"positions_csv": positions_path} if positions_path else {}
    r = alpaca_paper.sync_paper_account(**kw)
    logging.info("  live account: equity $%s, cash $%s, %d positions -> %s", f"{r['Equity']:,.2f}", f"{r['Cash']:,.2f}",
                 r["Positions"], os.path.relpath(r["positions_csv"], ROOT))


def plan_steps(a, mode):
    """The steps to run for this mode and the command-line filters (--only / --skip)."""
    names = [s[0] for s in STEPS]
    sel = []
    for i, step in enumerate(STEPS):
        name, _, _, when = step
        if a.only:
            if name == a.only:
                sel.append(step)
            continue
        if a.start and i < names.index(a.start):
            continue
        if when == "full" and mode != "full":
            continue
        if when not in ("full", "always") and not getattr(a, when):
            continue
        sel.append(step)
    return sel


def _hold_sleep_assertion():
    """Keep the Mac awake for the whole pipeline run (best-effort).

    A system sleep pauses the pipeline mid-step: on 2026-09-28 the sentiment
    step took 7415 s instead of the usual ~192 s, almost all of it frozen.
    `caffeinate -i` blocks idle system sleep, `-s` keeps the system awake on AC
    power (the display may still sleep), `-t 7200` caps the assertion at 2 h so
    an orphaned process can never hold the Mac awake forever. Released via atexit.
    """
    if sys.platform != "darwin":
        return
    try:
        import atexit
        p = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-s", "-t", "7200"])
        atexit.register(p.terminate)
        logging.info("  sleep assertion held (caffeinate pid %d, 2 h cap)", p.pid)
    except Exception as e:
        logging.info("  sleep assertion unavailable (%s) - keep the Mac awake manually", e)


def main(argv=None):
    """Command-line entry point: pick the mode, run the steps, validate the outputs, write the summary."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true", help="force the full online update now")
    p.add_argument("--quick", action="store_true", help="force quick mode (main signal analysis only, no quota APIs)")
    p.add_argument("--force-news", action="store_true", help="allow NewsAPI even if it ran within the last 24 h")
    p.add_argument("--force-fundamentals", action="store_true", help="allow a second Alpha Vantage rotation the same day")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit (nothing runs, nothing is written)")
    p.add_argument("--positions", help="positions CSV for the holdings alert (default: my_positions.csv if it exists)")
    p.add_argument("--only", choices=[s[0] for s in STEPS], help="run just this step")
    p.add_argument("--from", dest="start", choices=[s[0] for s in STEPS], help="start at this step (mode rules still apply)")
    p.add_argument("--visualization", action="store_true", help="also run company_report_visualization.ipynb")
    p.add_argument("--backtests", action="store_true", help="also run backtest.ipynb (≈10 s; rewrites Reports/backtest_*.csv)")
    p.add_argument("--keep-going", action="store_true", help="continue after a failed step (still exits 1)")
    p.add_argument("--list", action="store_true", help="list the steps and exit")
    p.add_argument("--sync-live", action="store_true",
                   help="refresh my_positions.csv from the Alpaca LIVE account before the alert (3 read-only GET calls)")
    p.add_argument("--trade", action="store_true",
                   help="after the pipeline, auto-trade the Alpaca LIVE account (REAL MONEY) from the fresh signals "
                        "(pulls live positions + equity, submits extended-hours DAY limit orders; LIVE only)")
    p.add_argument("--fill-check", action="store_true",
                   help="morning fill check: complete the previous evening's unfilled extended-hours orders "
                        "with regular-hours market orders (no notebooks run; LIVE - real money)")
    p.add_argument("--scheduled", action="store_true",
                   help="launchd runs: nothing due -> one 'idle:' line (no log); --trade retries a failed decision <= 3 times")
    p.add_argument("--now", help=argparse.SUPPRESS)          # tests: pretend it is this CT time ("2026-09-28 15:40")
    p.add_argument("--no-resume", action="store_true", help="do not resume from a previous failed run's checkpoint")
    a = p.parse_args(argv)
    if a.full and a.quick:
        p.error("--full and --quick exclude each other")
    if a.trade and a.fill_check:
        p.error("--trade and --fill-check exclude each other")
    if a.list:
        for name, kind, target, when in STEPS:
            print(f"{name:14s} {kind:9s} {target or 'report checks':40s} {when if when in ('full', 'always') else '--' + when}")
        return 0

    now = datetime.fromisoformat(a.now).replace(tzinfo=CT) if a.now else datetime.now(CT)
    t_wall = time.time()
    run_id = f"{now:%Y%m%d_%H%M%S}_{os.getpid()}"        # identifies this run in the watchdog checkpoint
    saved_argv = list(argv) if argv is not None else list(sys.argv[1:])
    _ckpt_on = a.now is None               # --now is the tests' fake clock: test runs never touch the real checkpoint

    def ckpt_write(*a_, **k_):
        if _ckpt_on:
            write_checkpoint(*a_, **k_)

    def ckpt_clear():
        if _ckpt_on:
            clear_checkpoint()

    def clock():                                            # the run's clock (= real time unless --now is given)
        """The run's clock as ISO text (= real time unless --now is given)."""
        return (now + timedelta(seconds=time.time() - t_wall)).isoformat(timespec="seconds")
    state = load_state()
    D, how, gate_why, gone = None, None, "", None
    if a.trade:
        D, how, gate_why = decision_gate(now, state, scheduled=a.scheduled and not a.start)
        gone = superseded_decision(now, state)
        if gone is not None and not a.dry_run:
            logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)   # before the run log
            state.update(report_superseded(gone, now))
        if a.scheduled and how in (None, "wait") and not a.dry_run:
            print(f"idle: {now:%a %b %d %I:%M %p} CT - {gate_why}", flush=True)   # launchd wake/interval run: no log
            return 0
    if a.fill_check and a.scheduled and not a.dry_run:
        idle = fill_check_idle(now)
        if idle:
            print(f"idle: {now:%a %b %d %I:%M %p} CT - {idle}", flush=True)
            return 0
    mode, why = choose_mode(now, state, a.full, a.quick, decision=D if how == "session" else None)
    if a.only in {"fundamentals", "sentiment", "earnings"}:
        if a.quick:
            p.error(f"--only {a.only} needs quota APIs, which --quick forbids - drop --quick or pick a non-quota step")
        mode, why = "full", f"--only {a.only}"
    steps = plan_steps(a, mode)
    skip = {}
    if any(s[0] == "sentiment" for s in steps):
        ok, reason = news_allowed(now, state, a.force_news)
        if not ok:
            skip["sentiment"] = reason
    if any(s[0] == "fundamentals" for s in steps):
        ok, reason = fundamentals_allowed(now, state, a.force_fundamentals)
        if not ok:
            skip["fundamentals"] = reason

    # Resume from checkpoint: if a previous run failed today at a critical step,
    # skip the steps that already succeeded and restart from the failed step.
    # This avoids re-running (and re-paying for) quota APIs after a transient failure.
    # Never resumes from a trade-phase failure (money-adjacent); the watchdog handles those.
    if _ckpt_on and not a.no_resume:
        ckpt = read_checkpoint()
        ckpt_failed = ckpt.get("failed_step")
        if ckpt_failed and ckpt.get("phase") == "steps":
            try:
                ckpt_time = datetime.fromisoformat(ckpt.get("updated_at", ""))
                ckpt_today = ckpt_time.date() == now.date()
            except (ValueError, TypeError):
                ckpt_today = False
            ckpt_mode = ckpt.get("mode")
            # Only resume same-day, same-mode checkpoints; stale or mismatched ones are ignored
            # (the plan printout below lists each resumed step as "SKIP - resumed ...")
            if ckpt_today and ckpt_mode == mode:
                for done_name in ckpt.get("steps_done", []):
                    if done_name not in skip:
                        skip[done_name] = f"resumed - already completed in failed {ckpt_time:%I:%M %p} run"

    if a.dry_run:
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True, stream=sys.stdout)
        log_path = None
    else:
        log_path = setup_logging()
    logging.info("run_all | %s CT | mode %s (%s)%s", f"{now:%a %b %d %I:%M %p}", mode.upper(), why,
                 " | DRY RUN - nothing runs" if a.dry_run else "")
    for name, kind, target, _ in steps:
        extra = f"  [{EXPECTED_CALLS[name]} calls]" if name in EXPECTED_CALLS and name not in skip else ""
        logging.info("  %-13s %s%s", name, f"SKIP - {skip[name]}" if name in skip else (target or "report checks"), extra)
    if a.sync_live:
        logging.info("  %-13s %s", "sync_live", "alpaca_paper.py  [Alpaca LIVE account: 3 read-only GET calls]")
    if a.trade:
        logging.info("  %-13s %s", "trade", "paper_trade.auto_trade  [Alpaca LIVE (REAL MONEY): reads positions+equity, submits extended-hours DAY limit orders]")
    if a.fill_check:
        logging.info("  %-13s %s", "fill-check", "paper_trade.complete_unfilled_orders  [Alpaca LIVE (REAL MONEY): completes unfilled evening orders]")
    if a.trade:
        if gone is not None:
            logging.info("SUPERSEDED: the %s decision never ran and its window closed - skipped (logged + alert)", gone.date())
        logging.info("TRADE: %s%s", {"evening": "due - ", "session": "due now - "}.get(how, "nothing to do - "), gate_why)
        if how not in ("evening", "session"):
            return 0
    if a.dry_run:
        return 0

    if a.trade:
        locked, lock_reason = acquire_trade_lock()
        if not locked:
            logging.info("TRADE LOCK: %s - another --trade run is already in progress; exiting.", lock_reason)
            logging.info("Nothing ran - the other --trade run owns this window.")
            return 0
        state = load_state()                               # re-read under the lock: never run a decision twice
        if str(state.get("last_decision") or "") >= D.date().isoformat():
            release_trade_lock()
            logging.info("TRADE: the %s decision already ran (another run finished it) - nothing to do", D.date())
            return 0
        if a.scheduled and not a.start:                    # launchd attempts count toward MAX_DECISION_ATTEMPTS
            att = state.get("decision_attempts") or {}
            n = int(att.get("n", 0)) + 1 if att.get("decision") == D.date().isoformat() else 1
            state.update(update_state(decision_attempts={"decision": D.date().isoformat(), "n": n, "at": clock()}))

    if a.fill_check:
        locked, lock_reason = acquire_trade_lock(FILL_LOCK)
        if not locked:
            logging.info("FILL CHECK LOCK: %s - another fill check is running; exiting.", lock_reason)
            return 0
        try:
            return _fill_check(now, run_id, saved_argv, mode, state, ckpt_write, ckpt_clear, clock, a.scheduled)
        finally:
            release_trade_lock(FILL_LOCK)

    return _pipeline(a, now, run_id, saved_argv, mode, why, steps, skip, state, log_path, ckpt_write, ckpt_clear, clock,
                     D, how)


def report_superseded(prev, now_ct):
    """Log + alert once that decision `prev` never ran and is now replaced by the newer one. Returns the new state."""
    import backtest_engine as be
    D = be.last_decision_date(now_ct)
    msg = (f"The {prev:%a %b %d} {be.decision_kind(prev)} never ran (the Mac was asleep or off). It was skipped at "
           f"{be.decision_slot(D):%a %I:%M %p} CT, when the {D:%a %b %d} {be.decision_kind(D)} replaced it. No money moved.")
    _notify("Missed decision skipped", msg)
    return update_state(last_superseded=prev.date().isoformat())


def _fill_check(now, run_id, saved_argv, mode, state, ckpt_write, ckpt_clear, clock, scheduled=False):
    """The --fill-check phase (under the fill-check lock). Returns the exit code."""
    import paper_trade
    paper_trade.drop_superseded_orders(now)               # a newer decision replaced them: never sent (alerted)
    ok, reason, needs_investigation = fill_check_allowed(now)
    if not ok and needs_investigation:
        logging.warning("FILL CHECK: %s", reason)
        if scheduled and state.get("fill_alert_on") == now.date().isoformat():
            return 1                                      # already alerted today (launchd runs every 30 min)
        _notify("9 AM check needs you", "The 9 AM order list is damaged, so nothing was sent (no money moved). Check "
                "Alpaca, then clear the live_pending_orders files in Reports. Log: Reports/logs", details=reason)
        state.update(update_state(fill_alert_on=now.date().isoformat()))
        return 1
    if not ok:
        logging.info("FILL CHECK: %s - nothing to do", reason)
        return 0
    logging.info("=== fill-check (paper_trade.complete_unfilled_orders, LIVE - real money)")
    ckpt_write(run_id, saved_argv, mode, [], phase="fill-check")
    try:
        import paper_trade
        results = paper_trade.complete_unfilled_orders(dry_run=False)
        if results.empty:
            logging.info("  fill check: no pending orders")
        else:
            for r in results.itertuples():
                logging.info("  %s %-6s %s shares -> %s", r.Side, r.Symbol, r.Shares, r.Status)
        state.update(update_state(last_fill_check_at=clock()))
    except Exception as e:
        logging.warning("  fill check failed: %s", e)
        return 1
    ckpt_clear()                                          # clean fill check - nothing to resume
    return 0


def _trade(a, D, how, failures, state, clock, ckpt):
    """The --trade phase for decision D (under the trade lock). how = 'evening' (extended-hours limit orders, completed
    by the 9 AM check) or 'session' (a missed decision caught up in regular hours: orders staged, then sent at once as
    market orders under the fill-check lock). D is marked done (run_state last_decision) as soon as auto_trade returns,
    even with FAILED rows, so a decision is never traded twice. Returns (results, meta)."""
    proceed, reason = _trade_decision(failures)
    if not proceed:
        logging.warning("=== trade skipped: %s", reason)
        failures.append("trade_skipped")
        return None, None
    ckpt()
    if reason:
        logging.warning("=== trade %s", reason)
    session = how == "session"
    logging.info("=== trade %s (paper_trade.auto_trade, LIVE - real money)",
                 f"catch-up of the {D:%a %b %d} decision" if session else f"{D:%a %b %d} evening")
    import paper_trade
    if session and not _wait_for_lock(FILL_LOCK):
        logging.warning("  a fill check has held its lock for 10 min - trade not started (retried by the next run)")
        failures.append("trade")
        return None, None
    try:
        try:
            _orders, meta, results = paper_trade.auto_trade(target="auto", dry_run=False, decision=D, session=session)
        except Exception as e:
            logging.warning("  trade failed: %s", e)
            failures.append("trade")
            return None, None
        st_, side = results["Status"].astype(str), results["Side"]
        n_fail = int(st_.str.startswith("FAILED").sum())
        n_sent = int((~st_.str.startswith(("SKIP", "FAILED"))).sum())
        logging.info("  trade: %d buys, %d sells, %d skipped, %d failed, %d submitted/staged (target=%s, as_of=%s) "
                     "-> Reports/live_orders_log.csv", int((side == "BUY").sum()), int((side == "SELL").sum()),
                     int(st_.str.startswith("SKIP").sum()), n_fail, n_sent, meta["source"], meta["as_of"])
        state.update(update_state(last_decision=D.date().isoformat(), **({"last_trade_at": clock()} if n_sent else {})))
        logging.info("  decision %s done (run_state last_decision) - it never runs again", D.date())
        if n_fail:
            failures.append("trade_partial")
        if session and n_sent:
            logging.info("=== send now (paper_trade.complete_unfilled_orders: regular-hours market orders, 2 decimals)")
            try:
                for r in paper_trade.complete_unfilled_orders(dry_run=False).itertuples():
                    logging.info("  %s %-6s %s shares -> %s", r.Side, r.Symbol, r.Shares, r.Status)
                state.update(update_state(last_fill_check_at=clock()))
            except Exception as e:                          # still pending (send_now): the fill-check job retries
                logging.warning("  sending failed: %s - the orders stay in Reports/live_pending_orders.json", e)
                failures.append("send_now")
        return results, meta
    finally:
        if session:
            release_trade_lock(FILL_LOCK)


def _wait_for_lock(path, minutes=10):
    """Take the lock at `path`, waiting up to `minutes` for a running holder (e.g. a fill check) to finish."""
    for _ in range(minutes * 6):
        if acquire_trade_lock(path)[0]:
            return True
        time.sleep(10)
    return False


def _pipeline(a, now, run_id, saved_argv, mode, why, steps, skip, state, log_path, ckpt_write, ckpt_clear, clock,
              D=None, how=None):
    """The notebook steps, the optional live sync and the --trade phase, then the summary. Returns the exit code."""
    _notify("Pipeline started",
            f"{mode} mode ({why}) - {len([s for s in steps if s[0] not in skip])} steps")
    if a.now is None:  # real runs only - the tests' fake clock never holds a sleep assertion
        _hold_sleep_assertion()
    env = dict(os.environ)
    env["PIPELINE_SENTIMENT_MODE"] = "online" if mode == "full" else "offline"
    env["PIPELINE_EARNINGS_MODE"] = "online" if mode == "full" else "offline"
    env["PIPELINE_NEWS_APPROVED"] = "1"                     # the runner's 24 h guard has been applied
    env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    env["PIPELINE_CALLS_FILE"] = calls_file = os.path.join(LOG_DIR, f"api_calls_{os.getpid()}.jsonl")
    ran, failures, t_start = [], [], time.time()
    for name, kind, target, _ in steps:
        if name in skip:
            logging.info("=== %s: skipped (%s)", name, skip[name])
            continue
        if name == "main" and how != "session":            # a daytime catch-up uses the last complete bar
            wait_for_final_bar("full" if how == "evening" else mode)
        if name == "fundamentals":
            state.update(update_state(last_fundamentals_date=now.date().isoformat()))
        t0 = time.time()
        logging.info("=== %s (%s)", name, target or "report checks")
        out, problems = "", []
        try:
            if kind == "script":
                rc, out = run_cmd([sys.executable, target], env)
            elif kind == "notebook":
                rc, out = run_notebook(target, env)
            else:
                problems = validate()
                for prob in problems:
                    logging.error("  %s", prob)
                rc = 1 if problems else 0
        except Exception as e:                              # e.g. jupyter missing
            logging.exception("step %s crashed: %s", name, e)
            rc = 1
        ran.append((name, rc == 0, time.time() - t0))
        ckpt_write(run_id, saved_argv, mode, [n for n, ok, _ in ran if ok],
                   failed_step=name if rc != 0 else None, phase="steps")
        if name == "sentiment" and rc == 0:
            # Stamp the NewsAPI 24 h guard on success. The notebook only stamps when run by hand
            # (PIPELINE_NEWS_APPROVED=1 under the pipeline, so it never stamps there) - without this,
            # last_news_at goes stale and the guard can't skip a same-day retry, burning 97 NewsAPI calls.
            state.update(update_state(last_news_at=clock()))
        logging.info("--- %s %s (%.0fs)", name, "OK" if rc == 0 else f"FAILED (exit {rc})", time.time() - t0)
        if rc != 0:
            failures.append(name)
            if kind == "notebook":
                _notify(f"Notebook failed: {target.replace('.ipynb', '')}", _notebook_error_summary(out, target))
            else:
                detail = "; ".join(problems) if problems else f"exit code {rc}"
                _notify(f"Pipeline step FAILED: {name}", f"{target or 'report checks'} - {detail}")
            if _stop_on_failure(name, a.keep_going):
                break
            elif name in OPTIONAL_STEPS:
                logging.warning("=== %s failed but is optional - continuing with the last good tables", name)
    # A full update counts when its critical steps (main/validate) succeeded, even if
    # optional upstream steps failed - the main signals are fresh.
    done = {"last_run_at": clock(), "last_mode": mode}
    if mode == "full" and not _critical_failures(failures) and not a.only and not a.start:
        done.update(last_full_date=now.date().isoformat(), last_full_at=clock())
    state.update(update_state(**done))
    live_synced = False
    if a.sync_live:                                        # opt-in: refresh the positions file before the alert
        logging.info("=== sync_live (alpaca_paper.py, read-only)")
        try:
            sync_live(a.positions)
            live_synced = True
        except Exception as e:                              # missing keys / network: the rest of the run still counts
            logging.warning("  live account sync failed: %s", e)
            failures.append("sync_live")

    trade_results, trade_meta = None, None
    if a.trade:
        try:
            trade_results, trade_meta = _trade(a, D, how, failures, state, clock, lambda: ckpt_write(
                run_id, saved_argv, mode, [n for n, ok, _ in ran if ok], phase="trade"))
        finally:
            release_trade_lock()

    # ------------------------------------------------------------------ summary
    info, calls = data_summary(), read_calls(calls_file)
    crit = _critical_failures(failures)
    logging.info("")
    logging.info("=" * 78)
    logging.info("SUMMARY  %s mode (%s), %.0f s%s", mode.upper(), why, time.time() - t_start,
                 f"  |  FAILED at: {', '.join(crit)}" if crit
                 else (f"  |  warnings: {', '.join(failures)}" if failures else ""))
    logging.info("  Ran:     %s", ", ".join(f"{n} {'ok' if ok else 'FAILED'}" for n, ok, _ in ran) or "nothing")
    if skip:
        logging.info("  Skipped: %s", "; ".join(f"{k} ({v.split(' - ')[0]})" for k, v in skip.items()))
    quota = {k: v for k, v in calls.items() if k in ("newsapi", "finnhub", "alphavantage")}
    logging.info("  API calls used: %s", ", ".join(f"{k} {v}" for k, v in quota.items()) if quota
                 else "none (Alpaca market-data bars only)")
    if live_synced:
        logging.info("  Alpaca LIVE account: 3 read-only GET calls (my_positions.csv refreshed)")
    if trade_results is not None and not trade_results.empty:
        st_ = trade_results["Status"].astype(str)
        logging.info("  Trade: %d submitted to LIVE, %d STAGED for the 9 AM fill check, %d skipped/failed "
                     "(see Reports/live_orders_log.csv)", int(st_.str.startswith("submitted").sum()),
                     int(st_.str.startswith("STAGED").sum()), int(st_.str.startswith(("SKIP", "FAILED")).sum()))
    if info["data_date"]:
        logging.info("  Data through %s  |  rules %s", info["data_date"], info["rules"])
        for line in info["decisions"][-3:]:
            logging.info("  %s", line)
    if any(n == "main" for n, ok, _ in ran if ok) or a.only == "validate":
        try:
            import holdings_alert
            for line in holdings_alert.alert_text(holdings_alert.build_alert(positions_path=a.positions)):
                logging.info("  ALERT %s", line)
        except Exception as e:                              # presentation only
            logging.warning("  holdings alert unavailable: %s", e)
    if info["next"]:
        logging.info("  %s", info["next"])
    logging.info("  Next full online update: %s  |  log %s", next_full_update(now, state), os.path.relpath(log_path, ROOT))
    logging.info("=" * 78)
    elapsed = time.time() - t_start
    if crit:
        _notify("Pipeline FAILED", f"{mode} mode, {elapsed:.0f}s - failed: {', '.join(crit)}")
    elif failures:
        _notify("Pipeline finished with warnings",
                f"{mode} mode, {elapsed:.0f}s - {', '.join(failures)} (main signals are fresh)")
    else:
        _notify("Pipeline finished", f"{mode} mode, {elapsed:.0f}s - {len(ran)} steps ok")
    if not crit:
        ckpt_clear()            # clean run (warnings ok) - the watchdog has nothing to resume
    return 1 if crit else 0


if __name__ == "__main__":
    sys.exit(main())
