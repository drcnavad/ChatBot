"""Live-rule settings (read from backtest_engine.WINNER, used for labels and captions only), the rules text,
the report file locations and their freshness windows."""
import os
import sys
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the project folder
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # project modules (backtest_engine, sector_mapping) importable from any cwd

try:
    from sector_mapping import sector_etf_for, symbol_sector
except Exception:  # deployed without the pipeline modules: relative strength vs SPY only
    symbol_sector = {}

    def sector_etf_for(symbol):
        return None

try:
    from backtest_engine import LIVE_INVESTED, MIN_BARS, WINNER
except Exception:
    WINNER, LIVE_INVESTED, MIN_BARS = {}, 0.99, 200

STRATEGY_TAG = WINNER.get("tag", "C6")
RS_LABEL = {"etf": "vs sector ETF and SPY",
            "sector_median": "vs the median of its sector peers, and sector ETF vs SPY",
            "median_all": "vs the median of its sector peers, and sector median vs the universe median",
            }.get(WINNER.get("rs_benchmark", "etf"), "vs sector ETF and SPY")
MIDWEEK = WINNER.get("midweek_swap")                                 # Mon-Thu checks (None = weekly only)
EXIT_BELOW = WINNER.get("midweek_exit_below") if MIDWEEK else None   # mid-week exit below this rank
EXIT_TO_TOP = WINNER.get("midweek_exit_to_top") if EXIT_BELOW else None  # refill an exit from the best non-held top-N
EARNINGS = WINNER.get("earnings_block_days")                         # no new buys with earnings within N days (None = off)

N_PICKS = WINNER.get("n", 10)                                        # stocks picked (10, live RS short-term from Oct 27, 2026)
W_TECH = WINNER.get("w_tech", 0.5)                                   # score = W_TECH x technical + (1 - W_TECH) x RS
LOW_VOL = WINNER.get("score") == "low_vol"                          # low volatility in uptrend (else relative strength)
MONTHLY = WINNER.get("calendar") == "monthly"                       # rebalance on the last trading day of the month
STRATEGY_NAME = ("Low volatility in uptrend (monthly)" if LOW_VOL and MONTHLY else
                 "Low volatility in uptrend" if LOW_VOL else "C6 (relative strength)")
SLEEVES = WINNER.get("sleeves")                                      # the live sleeve (sleeves.py), else None
HALVES = bool(WINNER.get("halves"))                                 # (Oct 27, 2026) live = RS short-term (weekly + Mon–Thu sells)
if SLEEVES and HALVES:
    STRATEGY_NAME = "Live: Short-term relative strength"
REGIME_OFF = f"{WINNER.get('regime_symbol', 'QQQ')} at or below its 200-day average"   # market filter off
SCALE = WINNER.get("regime_scale", 0.5) if WINNER.get("use_regime", True) else None
HALVED = "halved" if SCALE == 0.5 else f"multiplied by {SCALE:g}" if SCALE else "unchanged"
try:                                                                 # Friday replacement down to this rank
    from paper_trade import FRIDAY_SUB_MAX_RANK
except Exception:
    FRIDAY_SUB_MAX_RANK = 20

def halves_rules_text():
    """The live rules (from the first RS short-term rebalance, Fri Oct 30, 2026; defensive sleeve off Oct 29) in plain words."""
    return (f"**Strategy rules: {STRATEGY_NAME} ({STRATEGY_TAG})**\n"
            "- **Whole account** runs forward_test's \"Short-term relative strength\" exactly (Reports/sleeve_ledger.json).\n"
            "- **Score:** relative strength over 10, 21 and 63 trading days (60% stock vs its sector ETF, 40% that ETF vs SPY); "
            "the top 10, equal weights (~9.9% each), 99% invested. No defensive sleeve (removed Oct 29, 2026).\n"
            "- **When:** full rebalance every Friday 2:30 PM CT (Thursday when Friday is a holiday); Mon-Thu 2:30 PM a holding "
            "worse than rank 20 is sold and replaced by the next eligible stock ranked 20 or better (earnings within 5 days "
            "skipped). On early-close days (market closes "
            "12:00 PM CT: day after Thanksgiving, July 3 / Dec 24 when weekday) the decision runs at **11:00 AM CT** instead.\n"
            "- **Filters:** no new buy with earnings within 5 trading days (a held stock is not topped up); with "
            f"{REGIME_OFF} every weight is {HALVED}; do_not_buy never bought, do_not_sell never sold; a holding within 1 "
            "point of its weight is not traded.\n"
            "- **No idle cash:** a pick that cannot be bought (manual block, earnings) and leftover cash are spread over that "
            "day's buys in proportion (not in the backtest, which keeps that cash).\n"
            "- **Orders:** sells first, limit orders at the live quote (buy at the ask + 0.05%, sell at the bid - 0.05%); the "
            "9 AM CT check the next trading day sends any rest.\n")

def rules_text():
    """The live trading rules in plain words, in one place (numbers from backtest_engine.WINNER). The weekly C6 text is
    the non-low-volatility fallback (a switch back); the Mon-Thu rules are gone from the live code (Oct 16, 2026)."""
    if SLEEVES and HALVES:
        return halves_rules_text()
    band = WINNER.get("rebalance_band")
    return (f"**Strategy rules ({STRATEGY_TAG})**\n"
            "- **When:** decisions use the prices at about 2:30 PM CT, 30 min before the close (the pipeline starts at 2:30 PM CT "
            "and trades in market hours: sells first, then buys sized from the cash free after the sells; any rest at 9 AM CT): "
            "the Friday rebalance (the week's last trading day).\n"
            f"- **Score:** {W_TECH:g} × Technical + {1 - W_TECH:g} × Relative Strength {RS_LABEL}. Only stocks with a score "
            f"above {WINNER.get('min_score', 0):g} and at least {MIN_BARS} trading days of prices are ranked "
            "(newer stocks are listed as not traded yet).\n"
            f"- **Friday picks:** the {N_PICKS} best-ranked stocks by rank (no sector limit); fewer qualifying stocks = "
            "the rest in cash. Stocks that are not picked are sold.\n"
            f"- **Size:** equal weights, scaled to {LIVE_INVESTED:.0%} invested, "
            "each weight rounded down to 0.01%.\n"
            + (f"- **Market filter:** with {REGIME_OFF} at a rebalance, every weight is {HALVED}.\n" if SCALE else "")
            + (f"- **Rebalance:** every pick is brought back to its weight unless it is within {band * 100:g} percentage point "
               "of it; overweight holdings are trimmed so new buys get their full weight.\n" if band else "")
            + (f"- **Earnings:** a stock that is not held is not bought when its next earnings date is within {EARNINGS} "
               "calendar days; on Friday its slot goes to the next eligible stock. "
               "A held stock is not topped up before them.\n" if EARNINGS else "")
            + (f"- **Friday replacement:** a new pick the account does not hold that is blocked (earnings within {EARNINGS} days, "
               "or no buy back after an earnings-day stop sale) gets no cash slot: the next best-ranked eligible stock that is not "
               f"already a pick (score above 0, not blocked, down to rank {FRIDAY_SUB_MAX_RANK}) is bought at the blocked "
               "pick's weight, so the account still ends "
               f"with {N_PICKS} stocks; only if none is left does that weight stay in cash.\n" if EARNINGS else "")
            + "- **Earnings-day stop:** on a held stock's earnings day the price is checked every 30 seconds in the "
            "pre-market, regular and after-hours sessions; if it is ever 5% or more below the previous trading day's close, "
            "all whole shares are sold once with a limit at the bid - 0.05% (extended hours allowed; the fraction at the "
            "9 AM CT check). Earnings day = the session where the reaction happens: a before-open report is watched that "
            "day; an after-close (or unknown-time) report that day and the next day's pre-market and regular session "
            "(then measured against the earnings-day close). The cash waits for the next scheduled run, and the stock is "
            "not bought back until after its last earnings-day session.\n"
            + "- **Drop guard:** every 15 min ~8:30 AM–2:15 PM CT on market days, any held stock whose mid quote is 4.1% or more below the previous close is reviewed by the LLM with a full technical snapshot (and earnings date/time when a recent report exists; no news). Sell only on action=SELL with confidence ≥ 0.75 (Reports/drop_guard_config.json). After a fill, sale proceeds buy the top-ranked eligible replacement (not do_not_buy, ≤15% weight, above SMA200, not sold by the guard today, not itself down ≥4.1%); otherwise cash waits for the 2:30 run. Never sells do_not_sell. Kill switch + dry_run in the config (Mac mini LIVE: dry_run=false as of Oct 29 night CT; still respects do_not_sell and conf≥0.75).\n"
            + "- **Orders:** sells go first. Every order is a limit at the live quote: buy at the ask + 0.05%, sell at the "
            "bid - 0.05% (no market orders). An order whose quote is stale or wider than 0.5% waits for the next 9 AM CT "
            "check. The decision-slot run (2:30 PM CT, or 11:00 AM CT on early-close days) trades in market hours (2-decimal shares); from 5 minutes before the close, "
            "whole-share after-hours limit orders (until 7 PM CT). The 9 AM CT check the next trading day sends any rest. A missed decision is caught up at "
            "the next market session, unless the next decision is already due; a decision never runs twice.\n"
            "- **Signals** (the strategy's decision, dated): Buy = enters the portfolio · Hold = stays · Sold = leaves · "
            f"Watch = ranked but not picked · Score below {WINNER.get('min_score', 0):g} = not eligible. The plan chip = what "
            "the next Friday rebalance would do at the latest close.\n"
            "- **Changes:** Don't change the strategy until 12+ weeks of forward results (from Oct 29, 2026) compare against QQQ.\n")

# Report files (all written by run_all.py)
REPORTS = os.path.join(ROOT, "Reports")
SIGNAL_CSV = os.path.join(REPORTS, "signal_analysis.csv")
EARNINGS_CSV = os.path.join(REPORTS, "earnings_date.csv")
PICKS_CSV = os.path.join(REPORTS, "strategy_picks.csv")
CHANGES_CSV = os.path.join(REPORTS, "strategy_changes.csv")
HOLDINGS_CSV = os.path.join(REPORTS, "strategy_holdings.csv")
DECISIONS_CSV = os.path.join(REPORTS, "strategy_decisions.csv")
BENCH_CSV = os.path.join(REPORTS, "benchmark_prices.csv")
BARS_TABLE = "bars_daily"  # SQLite single source of truth for daily OHLCV
DRAWINGS_DIR = os.path.join(ROOT, "data", "drawings")   # your chart lines, one JSON per stock
NEWS_CSV = os.path.join(REPORTS, "news_cleaned_df.csv")
SHORT_HISTORY_CSV = os.path.join(REPORTS, "short_history_reference.csv")   # main_signal_analysis.ipynb: too new to trade
COMPANY_XLSX = os.path.join(REPORTS, "complete_company_analysis.xlsx")
NOTES_CSV = os.path.join(REPORTS, "my_notes.csv")   # the Stock Analysis tab's notes table (written by the dashboard)
CT = ZoneInfo("America/Chicago")

# file -> (what it is, max age in days before it is flagged stale)
FRESHNESS = {
    "signal_analysis.csv": ("prices, signals, strategy weights", 3),
    "strategy_picks.csv": ("current / provisional portfolio", 3),
    "strategy_decisions.csv": ("decision history: weekly rebalances + mid-week replacements/exits (chart markers)", 3),
    "strategy_midweek_check.csv": ("this week's decisions: Friday rebalance + Mon-Thu checks", 3),
    "benchmark_prices.csv": ("SPY / QQQ / sector ETF closes (RS lines)", 3),
    "weighted_sentiment.csv": ("news sentiment scores", 7),
    "news_cleaned_df.csv": ("news articles for AI summaries", 7),
    "earnings_date.csv": ("earnings calendar", 14),
    "complete_company_analysis.xlsx": ("fundamentals / fair value", 30),
    "balance_sheet_weights.csv": ("balance-sheet scores", 30),
    "balance_sheet.csv": ("raw quarterly fundamentals", 100),
    "forward_test_daily.csv": ("forward test, one row per trading day (forward_test.py)", 4),
}
APP_FILES = {"signal_analysis.csv", "strategy_picks.csv", "news_cleaned_df.csv", "earnings_date.csv",
             "complete_company_analysis.xlsx", "strategy_decisions.csv", "benchmark_prices.csv"}
