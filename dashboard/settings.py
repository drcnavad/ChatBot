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
    from backtest_engine import LIVE_INVESTED, MIN_BARS, WINNER, winner_max_per_sector
    SECTOR_MAX = winner_max_per_sector()
except Exception:
    WINNER, SECTOR_MAX, LIVE_INVESTED, MIN_BARS = {}, 4, 0.99, 200

STRATEGY_TAG = WINNER.get("tag", "C6")
RS_LABEL = {"etf": "vs sector ETF and SPY",
            "sector_median": "vs the median of its sector peers, and sector ETF vs SPY",
            "median_all": "vs the median of its sector peers, and sector median vs the universe median",
            }.get(WINNER.get("rs_benchmark", "etf"), "vs sector ETF and SPY")
MIDWEEK = WINNER.get("midweek_swap")                                 # Mon/Wed swap check (None = weekly only)
EXIT_BELOW = WINNER.get("midweek_exit_below") if MIDWEEK else None   # mid-week exit below this rank
EXIT_TO_TOP = WINNER.get("midweek_exit_to_top") if EXIT_BELOW else None  # replace with best top-N; else cash until Fri
MAX_PICK = WINNER.get("max_pick_rank")                               # picks only from ranks 1..MAX_PICK
CAP_SOFT = bool(WINNER.get("cap_soft"))                              # sector limit relaxed to fill 10 slots
EARNINGS = WINNER.get("earnings_block_days")                         # no new buys with earnings within N days (None = off)

N_PICKS = WINNER.get("n", 10)                                        # portfolio size (top 10)
W_TECH = WINNER.get("w_tech", 0.5)                                   # score = W_TECH x technical + (1 - W_TECH) x RS
REGIME_OFF = f"{WINNER.get('regime_symbol', 'QQQ')} at or below its 200-day average"   # market filter off
SCALE = WINNER.get("regime_scale", 0.5) if WINNER.get("use_regime", True) else None
HALVED = "halved" if SCALE == 0.5 else f"multiplied by {SCALE:g}" if SCALE else "unchanged"
MAX_WEIGHT = WINNER.get("max_weight")                                # no stock above this weight; the extra stays cash


def rules_text():
    """The live trading rules in plain words, in one place (numbers from backtest_engine.WINNER)."""
    band = WINNER.get("rebalance_band")
    days = " and ".join(MIDWEEK["days"]) if MIDWEEK else ""
    return (f"**Strategy rules ({STRATEGY_TAG})**\n"
            "- **When:** decisions use the prices at about 2:30 PM CT, 30 min before the close (the pipeline starts at 2:30 PM CT "
            "and trades in market hours: sells first, then buys sized from the cash free after the sells; any rest at 9 AM CT): "
            "the Friday rebalance (the week's last trading day)"
            + (f" and the {days} checks (the next trading day after a holiday)" if MIDWEEK else "") + ".\n"
            f"- **Score:** {W_TECH:g} × Technical + {1 - W_TECH:g} × Relative Strength {RS_LABEL}. Only stocks with a score "
            f"above {WINNER.get('min_score', 0):g} and at least {MIN_BARS} trading days of prices are ranked "
            "(newer stocks are listed as not traded yet).\n"
            f"- **Friday picks:** the {N_PICKS} best-ranked stocks"
            + (f" from ranks 1–{MAX_PICK}" if MAX_PICK else " by rank (no sector limit)")
            + (f", max {SECTOR_MAX} per sector" if SECTOR_MAX < N_PICKS else "")
            + (f"; slots the sector limit leaves empty are filled from ranks 1–{MAX_PICK or 20} anyway" if CAP_SOFT else "")
            + "; fewer qualifying stocks = the rest in cash. Stocks that are not picked are sold.\n"
            f"- **Size:** weights ∝ 1 / 63-day volatility (less volatile = larger), scaled to {LIVE_INVESTED:.0%} invested, "
            "each weight rounded down to 0.01%.\n"
            + (f"- **Market filter:** with {REGIME_OFF} at a rebalance, every weight is {HALVED}.\n" if SCALE else "")
            + (f"- **Max per stock:** No single stock gets more than {MAX_WEIGHT:.0%}; any extra stays in cash.\n" if MAX_WEIGHT else "")
            + (f"- **Rebalance:** every pick is brought back to its weight unless it is within {band * 100:g} percentage point "
               "of it; overweight holdings are trimmed so new buys get their full weight.\n" if band else "")
            + (f"- **{days} swap:** if a stock that is not held ranks in the top {MIDWEEK['enter_top']} and a held stock has "
               f"fallen below rank {MIDWEEK['exit_below']}, the worst-ranked held stock is sold and the new one bought for "
               "the same dollar amount (repeated while both are true"
               + ("; no sector limit" if SECTOR_MAX >= N_PICKS else
                  ("; the sector limit does not apply" if CAP_SOFT else f"; max {SECTOR_MAX} per sector still applies"))
               + ").\n" if MIDWEEK else "")
            + (f"- **{days} exit:** after the swaps, any holding ranked worse than {EXIT_BELOW} (or no longer ranked) is "
               + (f"swapped for the best top-{EXIT_TO_TOP} stock not held (same dollars; earnings rule); the cash waits for "
                  "Friday only if none is left.\n" if EXIT_TO_TOP else
                  "sold; the cash waits for the Friday rebalance.\n") if EXIT_BELOW else "")
            + (f"- **Earnings:** a stock that is not held is not bought when its next earnings date is within {EARNINGS} "
               "calendar days; on Friday its slot goes to the next eligible stock (else cash), mid-week it is just not bought. "
               "A held stock is not topped up before them.\n" if EARNINGS else "")
            + "- **Pre-earnings stop:** a held stock with earnings within 7 calendar days is sold if its price falls to its "
            "highest close since bought - 3 × ATR(14), from 7 days before the report through the reaction day (checked "
            "every 10 minutes in the pre-market, regular and after-hours sessions; in regular hours every share at once, "
            "outside them the whole shares and the fraction at the 9 AM CT check). One sale per report; the cash waits for "
            "the next scheduled run, and the stock is not bought back until after its reaction day.\n"
            + "- **Orders:** sells go first. Every order is a limit at the live quote: buy at the ask + 0.05%, sell at the "
            "bid - 0.05% (no market orders). An order whose quote is stale or wider than 0.5% waits for the next 9 AM CT "
            "check. The 2:30 PM CT run trades in market hours (2-decimal shares); from 5 minutes before the close, "
            "whole-share after-hours limit orders (until 7 PM CT). The 9 AM CT check the next trading day sends any rest. A missed decision is caught up at "
            "the next market session, unless the next decision is already due; a decision never runs twice.\n"
            "- **Signals** (the strategy's decision, dated): Buy = enters the portfolio · Hold = stays · Sold = leaves · "
            f"Watch = ranked but not picked · Score below {WINNER.get('min_score', 0):g} = not eligible. The plan chip = what "
            "the next Friday rebalance would do at the latest close.\n"
            "- **Changes:** Don't change the strategy until 12+ weeks of forward results (from Oct 2, 2026) compare against QQQ.\n")


# Report files (all written by run_all.py)
REPORTS = os.path.join(ROOT, "Reports")
SIGNAL_CSV = os.path.join(REPORTS, "signal_analysis.csv")
EARNINGS_CSV = os.path.join(REPORTS, "earnings_date.csv")
PICKS_CSV = os.path.join(REPORTS, "strategy_picks.csv")
CHANGES_CSV = os.path.join(REPORTS, "strategy_changes.csv")
HOLDINGS_CSV = os.path.join(REPORTS, "strategy_holdings.csv")
DECISIONS_CSV = os.path.join(REPORTS, "strategy_decisions.csv")
MIDWEEK_CSV = os.path.join(REPORTS, "strategy_midweek_check.csv")
BENCH_CSV = os.path.join(REPORTS, "benchmark_prices.csv")
NEWS_CSV = os.path.join(REPORTS, "news_cleaned_df.csv")
SHORT_HISTORY_CSV = os.path.join(REPORTS, "short_history_reference.csv")   # main_signal_analysis.ipynb: too new to trade
COMPANY_XLSX = os.path.join(REPORTS, "complete_company_analysis.xlsx")
CT = ZoneInfo("America/Chicago")


# file -> (what it is, max age in days before it is flagged stale)
FRESHNESS = {
    "signal_analysis.csv": ("prices, signals, strategy weights", 3),
    "strategy_picks.csv": ("current / provisional portfolio", 3),
    "strategy_decisions.csv": ("decision history: weekly rebalances + mid-week swaps (chart markers)", 3),
    "strategy_midweek_check.csv": ("this week's decisions: Friday rebalance + Mon/Wed swap checks", 3),
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
