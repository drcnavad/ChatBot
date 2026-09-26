"""ONE command for the whole Stock Analysis pipeline:   python run_all.py      (or open run_all.ipynb and Run All)

It picks the mode by itself (clock in US Central time):
  FULL   Mon / Wed / Fri after 3:15 PM CT on a trading day, when the full online update has not run yet today:
           fundamentals  company_report_autofetch.py - Alpha Vantage rotation, max 12 stocks = max 24 calls (free limit 25/day)
           processing    company_report_processing.ipynb + scoring company_report_scoring.ipynb (company reports)
           sentiment     sentiment_analysis.ipynb online - NewsAPI 97 calls (free limit 100/day -> never twice within 24 h)
                         + Finnhub 97 calls
           earnings      earnings_date.ipynb online - Finnhub 97 calls (throttled below 60/min) + yfinance
           main          main_signal_analysis.ipynb - fresh Alpaca daily bars (market data only), ranks, picks, mid-week check
           validate      the report files the app reads
  QUICK  any other time: main + validate only. Zero quota APIs (only Alpaca market-data bars).
Both end with a short summary: what ran, API calls used, data date, the holdings ALERT line and the next check / rebalance.

    python run_all.py --full          # force the online update now (NewsAPI still refused if it ran within 24 h)
    python run_all.py --full --force-news     # ... and allow a second NewsAPI run (may exceed the free 100/day)
    python run_all.py --quick         # force quick mode
    python run_all.py --dry-run       # show the plan (mode, steps, expected API calls) and exit - nothing runs
    python run_all.py --positions f.csv       # alert on another positions file (default: my_positions.csv if it exists)
    python run_all.py --only main | --from scoring | --list | --backtests | --visualization | --keep-going
    python run_all.py --sync-paper    # also refresh my_positions.csv from your Alpaca PAPER account (off by default)
    python run_all.py --trade         # after the pipeline, auto-trade the PAPER account from the fresh signals
                                      # (pulls live positions + equity; BUY only for buy-signal ('add')
                                      #  symbols, sized Weight * equity in whole shares net of shares already
                                      #  held - never above the weight, overweight trimmed with a SELL of the
                                      #  excess; hold-signal symbols are never traded; sells = entire
                                      #  position; sells sent before buys as extended-hours
                                      #  DAY limit orders at the closing price; logs to
                                      #  Reports/paper_orders_log.csv). If the trade step runs at/after
                                      #  7:00 PM CT (extended hours over), nothing is submitted - the planned
                                      #  orders are staged for the next morning's --fill-check instead.
                                      #  PAPER only, never live.
                                      # Runs once per scheduled window (Mon/Wed/Fri after 3:15 PM CT on a
                                      # trading day): a second --trade is refused until the next window
                                      # (trade_allowed guard), and overlapping --trade processes are
                                      # serialized by Reports/.trade.lock. last_trade_at is stamped only
                                      # when orders were actually submitted or staged, so a run that sent
                                      # nothing never blocks the next window. If the Mac was asleep/off at
                                      # 3:15 PM, a login agent (trade_catchup.sh, installed by
                                      # setup_stock_schedule.sh) fires the missed --trade once at the next
                                      # login that same Mon/Wed/Fri between 3:15 and 7:00 PM CT.
    python run_all.py --fill-check    # the next trading morning after a --trade (from 9:00 AM CT): check the
                                      # evening's extended-hours orders and complete any unfilled remainder
                                      # with regular-hours market orders (orders staged by a past-7PM --trade
                                      # are sent in full). Runs no notebooks and uses no quota APIs.

State: Reports/run_state.json - the runner's memory (all values are ISO timestamps unless noted):
    last_full_date / last_full_at .. last successful full online update (date + time)
    last_news_at ................... NewsAPI 24 h guard; stamped only AFTER the sentiment step's NewsAPI
                                     calls complete, so a crash can never cause a false lockout
    last_fundamentals_date ......... last Alpha Vantage rotation (date; max once/day)
    last_trade_at .................. last --trade that actually submitted or staged orders (a run that sent
                                     nothing is never marked complete, so it can't block the next window)
    last_fill_check_at ............. last morning fill check
    last_run_at / last_mode ......... last invocation (any mode)
Logs: Reports/logs/run_*.log (last 30 kept).
By default no orders are placed. Alpaca account endpoints are only called with --sync-paper (3 read-only GETs via
alpaca_paper.py), --trade (paper_trade.auto_trade(): reads positions + equity, then submits extended-hours DAY
limit orders to PAPER only) or --fill-check (paper_trade.complete_unfilled_orders(): checks the evening orders
and completes unfilled remainders with regular-hours market orders; keys ALPACA_PAPER_KEY_ID /
ALPACA_PAPER_SECRET_KEY in .env).
"""
import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
LOG_DIR = os.path.join(REPORTS, "logs")
STATE_FILE = os.path.join(REPORTS, "run_state.json")
CT, ET = ZoneInfo("America/Chicago"), ZoneInfo("America/New_York")
FULL_DAYS = {0, 2, 4}                  # Mon, Wed, Fri
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


# ----------------------------------------------------------------------------- state + mode (pure logic, unit-tested)
def load_state(path=STATE_FILE):
    """The runner's memory. First run: seeded from the evidence on disk (sentiment_history.csv is written only by online
    NewsAPI runs; fetch_run_log.csv logs every Alpha Vantage attempt) so a same-day repeat is still refused."""
    try:
        with open(path) as f:
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


def save_state(state, path=STATE_FILE):
    """Write run_state.json atomically (temp file + rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def is_trading_day(d):
    """NYSE session (full-day holidays excluded), same calendar as the strategy."""
    import backtest_engine as be
    day = datetime(d.year, d.month, d.day)
    return len(__import__("pandas").date_range(day, day, freq=be.NYSE_SESSION)) == 1


def full_window(now_ct):
    """(True, why) when the automatic full update is due by the clock: Mon/Wed/Fri trading day after 3:15 PM CT."""
    if now_ct.weekday() not in FULL_DAYS:
        return False, f"{now_ct:%a} is not a Mon/Wed/Fri full-update day"
    if (now_ct.hour, now_ct.minute) < FULL_AFTER:
        return False, f"before {FULL_AFTER[0] - 12}:{FULL_AFTER[1]:02d} PM CT"
    if not is_trading_day(now_ct.date()):
        return False, f"{now_ct:%a %b %d} is a market holiday"
    return True, f"{now_ct:%a} after {FULL_AFTER[0] - 12}:{FULL_AFTER[1]:02d} PM CT"


def choose_mode(now_ct, state, force_full=False, force_quick=False):
    """'full' or 'quick' plus a plain reason."""
    if force_quick:
        return "quick", "--quick"
    if force_full:
        return "full", "--full"
    due, why = full_window(now_ct)
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


def next_scheduled_trade(after_ct):
    """Next Mon/Wed/Fri 3:15 PM CT trading day strictly after after_ct."""
    base = after_ct.date()
    for k in range(0, 12):
        d = base + timedelta(days=k)
        cand = datetime(d.year, d.month, d.day, FULL_AFTER[0], FULL_AFTER[1], tzinfo=CT)
        if cand > after_ct and d.weekday() in FULL_DAYS and is_trading_day(d):
            return cand
    return None


def trade_allowed(now_ct, state):
    """(allowed, reason): auto-trade runs once per scheduled window - a second --trade is skipped
    until the next Mon/Wed/Fri 3:15 PM CT trading day after the last trade."""
    last = state.get("last_trade_at")
    if not last:
        return True, "ok"
    last_dt = datetime.fromisoformat(last)
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=CT)
    nxt = next_scheduled_trade(last_dt)
    if nxt is not None and now_ct < nxt:
        return False, (f"trade already ran {last_dt:%a %b %d %I:%M %p}; next scheduled trade is "
                       f"{nxt:%a %b %d %I:%M %p} - skipping to avoid duplicate orders")
    return True, "ok"


TRADE_LOCK = os.path.join(REPORTS, ".trade.lock")
TRADE_LOCK_STALE_S = 2 * 3600              # a lock older than this is taken over (crashed run)


def acquire_trade_lock(path=TRADE_LOCK):
    """(True, None) when this process now owns the --trade lock; (False, reason) when another
    live --trade run holds it. The lock serializes overlapping --trade processes so two of them
    can never submit duplicate orders. A stale lock (dead PID or older than TRADE_LOCK_STALE_S)
    is taken over with a warning instead of blocking forever."""
    import errno
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
        age = (datetime.now(CT) - datetime.fromisoformat(at)).total_seconds() if at else float("inf")
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


def release_trade_lock(path=TRADE_LOCK):
    """Release the --trade lock (best effort)."""
    try:
        os.remove(path)
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
    """(allowed, reason): the morning fill check completes the previous evening's extended-hours
    orders. It runs once the next trading day after the evening trade reaches 9:00 AM CT, and only
    while a pending fill check (Reports/paper_pending_orders.json) exists."""
    import paper_trade
    path = pending_path or paper_trade.PENDING_ORDERS_JSON
    try:
        with open(path) as f:
            pend = json.load(f)
    except OSError:
        return False, "no pending fill check (no evening orders awaiting completion)"
    except ValueError:
        # A corrupt pending file must never be silently treated as "no pending orders" - that
        # would orphan the evening's orders. Back it up loudly and stop instead.
        bad = path + f".corrupt_{datetime.now(CT):%Y%m%d_%H%M%S}"
        try:
            os.replace(path, bad)
        except OSError:
            bad = path
        return False, (f"the pending fill-check file is corrupt (moved to {bad}) - investigate it "
                       f"before re-running; the evening's orders were NOT completed")
    if not isinstance(pend, dict) or not pend.get("orders"):
        return False, "no pending fill check (evening order list is empty)"
    try:
        evening = datetime.fromisoformat(pend["evening_date"]).date()
    except (KeyError, ValueError, TypeError):
        return False, ("the pending fill-check file has no valid evening_date - investigate "
                       "Reports/paper_pending_orders.json before re-running")
    d = next_trading_day_after(evening)
    earliest = datetime(d.year, d.month, d.day, FILL_CHECK_AFTER[0], FILL_CHECK_AFTER[1], tzinfo=CT)
    if now_ct < earliest:
        return False, (f"fill check for the {evening:%a %b %d} evening run opens "
                       f"{earliest:%a %b %d} at 9:00 AM CT")
    if not is_trading_day(now_ct.date()):
        return False, f"{now_ct:%a %b %d} is a market holiday - the completion orders would not fill"
    return True, "ok"


def next_full_update(now_ct, state):
    """Plain text: when `python run_all.py` will next do the full online update by itself."""
    for k in range(0, 10):
        d = (now_ct + timedelta(days=k)).replace(hour=FULL_AFTER[0], minute=FULL_AFTER[1], second=0, microsecond=0)
        if full_window(d)[0] and d.date().isoformat() != state.get("last_full_date"):
            return ("today" if k == 0 else f"{d:%a %b} {d.day}") + " after 3:15 PM CT"
    return "the next Mon/Wed/Fri trading day after 3:15 PM CT"


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
    """Run a subprocess, stream its output into the log, return the exit code."""
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if line.strip():
            logging.info("    %s", line.rstrip())
    return proc.wait()


def run_notebook(path, env):
    """Execute a notebook in place with nbconvert."""
    return run_cmd([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    f"--ExecutePreprocessor.timeout={NB_TIMEOUT}", path], env)


def read_calls(path):
    """Sum the API request counts the steps appended to `path` (one JSON dict per line)."""
    calls = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                for k, v in json.loads(line).items():
                    calls[k] = calls.get(k, 0) + int(v)
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
    """Full mode on a decision day: the Mon/Wed/Fri decision needs today's completed daily bar (final after 4:30 PM ET)."""
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
            out.update(data_date=str(m["As_Of"].iloc[0])[:10], next=m["Next_Message"].iloc[0], rules=m["Rules"].iloc[0])
            out["decisions"] = [f"{'>> ' if getattr(r, 'Is_Latest', False) else '   '}{r.Message} [{r.Status}]"
                                for r in m.itertuples()]
    return out


def sync_paper(positions_path=None):
    """Optional (--sync-paper): refresh my_positions.csv + Reports/paper_*.csv from the Alpaca PAPER account (read-only)."""
    import alpaca_paper
    kw = {"positions_csv": positions_path} if positions_path else {}
    r = alpaca_paper.sync_paper_account(**kw)
    logging.info("  paper account: equity $%s, cash $%s, %d positions -> %s", f"{r['Equity']:,.2f}", f"{r['Cash']:,.2f}",
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
    p.add_argument("--sync-paper", action="store_true",
                   help="refresh my_positions.csv from the Alpaca PAPER account before the alert (3 read-only GET calls)")
    p.add_argument("--trade", action="store_true",
                   help="after the pipeline, auto-trade the Alpaca PAPER account from the fresh signals "
                        "(pulls live positions + equity, submits extended-hours DAY limit orders; PAPER only)")
    p.add_argument("--fill-check", action="store_true",
                   help="morning fill check: complete the previous evening's unfilled extended-hours orders "
                        "with regular-hours market orders (no notebooks run; PAPER only)")
    p.add_argument("--now", help=argparse.SUPPRESS)          # tests: pretend it is this CT time ("2026-09-28 15:40")
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

    def clock():                                            # the run's clock (= real time unless --now is given)
        """The run's clock as ISO text (= real time unless --now is given)."""
        return (now + timedelta(seconds=time.time() - t_wall)).isoformat(timespec="seconds")
    state = load_state()
    mode, why = choose_mode(now, state, a.full, a.quick)
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
    if a.sync_paper:
        logging.info("  %-13s %s", "sync_paper", "alpaca_paper.py  [Alpaca PAPER account: 3 read-only GET calls]")
    if a.trade:
        logging.info("  %-13s %s", "trade", "paper_trade.auto_trade  [Alpaca PAPER: reads positions+equity, submits extended-hours DAY limit orders]")
    if a.fill_check:
        logging.info("  %-13s %s", "fill-check", "paper_trade.complete_unfilled_orders  [Alpaca PAPER: completes unfilled evening orders]")
    if a.dry_run:
        return 0

    if a.trade:
        due, why = full_window(now)
        if not due:
            logging.info("TRADE: not a scheduled trade window (%s) - skipping", why)
            logging.info("Nothing ran - --trade runs Mon/Wed/Fri after 3:15 PM CT on a trading day.")
            return 0
        ok, reason = trade_allowed(now, state)
        if not ok:
            logging.info("TRADE GUARD: %s", reason)
            logging.info("Nothing ran - to avoid accidental double orders, --trade runs once per scheduled window.")
            return 0
        locked, lock_reason = acquire_trade_lock()
        if not locked:
            logging.info("TRADE LOCK: %s - another --trade run is already in progress; exiting.", lock_reason)
            logging.info("Nothing ran - the other --trade run owns this window.")
            return 0

    if a.fill_check:
        ok, reason = fill_check_allowed(now)
        if not ok:
            logging.info("FILL CHECK: %s - nothing to do", reason)
            return 0
        logging.info("=== fill-check (paper_trade.complete_unfilled_orders, PAPER only)")
        try:
            import paper_trade
            results = paper_trade.complete_unfilled_orders(dry_run=False)
            if results.empty:
                logging.info("  fill check: no pending orders")
            else:
                for r in results.itertuples():
                    logging.info("  %s %-6s %s shares -> %s", r.Side, r.Symbol, r.Shares, r.Status)
            state["last_fill_check_at"] = clock(); save_state(state)
        except Exception as e:
            logging.warning("  fill check failed: %s", e)
            return 1
        return 0

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
        if name == "main":
            wait_for_final_bar(mode)
        if name == "fundamentals":
            state["last_fundamentals_date"] = now.date().isoformat(); save_state(state)
        t0 = time.time()
        logging.info("=== %s (%s)", name, target or "report checks")
        try:
            if kind == "script":
                rc = run_cmd([sys.executable, target], env)
            elif kind == "notebook":
                rc = run_notebook(target, env)
            else:
                problems = validate()
                for prob in problems:
                    logging.error("  %s", prob)
                rc = 1 if problems else 0
        except Exception as e:                              # e.g. jupyter missing
            logging.exception("step %s crashed: %s", name, e)
            rc = 1
        ran.append((name, rc == 0, time.time() - t0))
        if name == "sentiment" and rc == 0:
            # Stamp the NewsAPI 24 h guard only after the calls actually completed. The notebook
            # writes its own stamp when its NewsAPI loop finishes - prefer that timestamp so the
            # runner's final save_state() can't clobber it with a stale value, and a crash can
            # never create a false lockout (no stamp on failure, so a retry is still allowed).
            state["last_news_at"] = load_state().get("last_news_at") or clock()
            save_state(state)
        logging.info("--- %s %s (%.0fs)", name, "OK" if rc == 0 else f"FAILED (exit {rc})", time.time() - t0)
        if rc != 0:
            failures.append(name)
            if not a.keep_going:
                break
    if mode == "full" and not failures and not a.only and not a.start:
        state["last_full_date"] = now.date().isoformat()
        state["last_full_at"] = clock()
    state["last_run_at"], state["last_mode"] = clock(), mode
    save_state(state)
    paper_synced = False
    if a.sync_paper:                                        # opt-in: refresh the positions file before the alert
        logging.info("=== sync_paper (alpaca_paper.py, read-only)")
        try:
            sync_paper(a.positions)
            paper_synced = True
        except Exception as e:                              # missing keys / network: the rest of the run still counts
            logging.warning("  paper account sync failed: %s", e)
            failures.append("sync_paper")

    trade_results, trade_meta = None, None
    if a.trade:
        try:
            if failures:
                logging.warning("=== trade skipped: pipeline had failures (%s) - no orders sent", ", ".join(failures))
                failures.append("trade_skipped")
            else:
                logging.info("=== trade (paper_trade.auto_trade, PAPER only)")
                try:
                    import paper_trade
                    orders, trade_meta, trade_results = paper_trade.auto_trade(target="auto", dry_run=False)
                    n_buy = int((trade_results["Side"] == "BUY").sum()) if not trade_results.empty else 0
                    n_sell = int((trade_results["Side"] == "SELL").sum()) if not trade_results.empty else 0
                    n_skip = int((trade_results["Status"].str.startswith("SKIPPED")).sum()) if not trade_results.empty else 0
                    n_fail = int((trade_results["Status"].str.startswith("FAILED")).sum()) if not trade_results.empty else 0
                    n_sent = (0 if trade_results.empty else
                              int((~trade_results["Status"].str.startswith(("SKIPPED", "FAILED"))).sum()))
                    logging.info("  trade: %d buys, %d sells, %d skipped, %d failed, %d submitted/staged (target=%s, as_of=%s) -> Reports/paper_orders_log.csv",
                                 n_buy, n_sell, n_skip, n_fail, n_sent, trade_meta["source"], trade_meta["as_of"])
                    if n_fail:
                        failures.append("trade_partial")
                    elif n_sent:
                        state["last_trade_at"] = clock(); save_state(state)
                    else:
                        logging.info("  trade: nothing was submitted or staged - this window is NOT marked complete")
                except Exception as e:
                    logging.warning("  trade failed: %s", e)
                    failures.append("trade")
        finally:
            release_trade_lock()

    # ------------------------------------------------------------------ summary
    info, calls = data_summary(), read_calls(calls_file)
    logging.info("")
    logging.info("=" * 78)
    logging.info("SUMMARY  %s mode (%s), %.0f s%s", mode.upper(), why, time.time() - t_start,
                 f"  |  FAILED at: {', '.join(failures)}" if failures else "")
    logging.info("  Ran:     %s", ", ".join(f"{n} {'ok' if ok else 'FAILED'}" for n, ok, _ in ran) or "nothing")
    if skip:
        logging.info("  Skipped: %s", "; ".join(f"{k} ({v.split(' - ')[0]})" for k, v in skip.items()))
    quota = {k: v for k, v in calls.items() if k in ("newsapi", "finnhub", "alphavantage")}
    logging.info("  API calls used: %s", ", ".join(f"{k} {v}" for k, v in quota.items()) if quota
                 else "none (Alpaca market-data bars only)")
    if paper_synced:
        logging.info("  Alpaca PAPER account: 3 read-only GET calls (my_positions.csv refreshed)")
    if trade_results is not None and not trade_results.empty:
        if trade_results["Status"].str.startswith("STAGED").any():
            logging.info("  Trade: %d orders STAGED for the morning fill check (past 7 PM CT - nothing submitted)",
                         len(trade_results))
        else:
            logging.info("  Trade: %d orders submitted to PAPER (see Reports/paper_orders_log.csv)", len(trade_results))
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
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
