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

N_PICKS = WINNER.get("n", 10)                                        # stocks per half (5 from Oct 21, 2026)
W_TECH = WINNER.get("w_tech", 0.5)                                   # score = W_TECH x technical + (1 - W_TECH) x RS
LOW_VOL = WINNER.get("score") == "low_vol"                          # low volatility in uptrend (else relative strength)
MONTHLY = WINNER.get("calendar") == "monthly"                       # rebalance on the last trading day of the month
STRATEGY_NAME = ("Low volatility in uptrend (monthly)" if LOW_VOL and MONTHLY else
                 "Low volatility in uptrend" if LOW_VOL else "C6 (relative strength)")
SLEEVES = WINNER.get("sleeves")                                      # two halves (sleeves.py), else None
if SLEEVES:
    STRATEGY_NAME = (f"{SLEEVES['lowvol']:.0%} Low volatility in uptrend (monthly) + "
                     f"{SLEEVES['sector']:.0%} Sector rotation (weekly)")
REGIME_OFF = f"{WINNER.get('regime_symbol', 'QQQ')} at or below its 200-day average"   # market filter off
SCALE = WINNER.get("regime_scale", 0.5) if WINNER.get("use_regime", True) else None
HALVED = "halved" if SCALE == 0.5 else f"multiplied by {SCALE:g}" if SCALE else "unchanged"
MAX_WEIGHT = WINNER.get("max_weight")                                # no stock above this weight; the extra stays cash
try:                                                                 # Friday replacement down to this rank
    from paper_trade import FRIDAY_SUB_MAX_RANK
except Exception:
    FRIDAY_SUB_MAX_RANK = 20

def low_vol_rules_text():
    """The monthly low-volatility rules (live from Fri Oct 16, 2026) in plain words."""
    band = WINNER.get("rebalance_band")
    per_sector = max(1, int(WINNER.get("sector_cap", 1.0) * N_PICKS))
    return (f"**Strategy rules: {STRATEGY_NAME} ({STRATEGY_TAG})**\n"
            "- **When:** a full rebalance on Fri Oct 16, 2026 (the switch), then on the last trading day of each month"
            + (" (the low-volatility half; the sector half every Friday, from Fri Oct 23; Mondays Oct 19-22)" if SLEEVES else "") + ", with "
            "the prices at about 2:30 PM CT (sells first, then buys from the cash free after the sells; any rest at 9 AM CT "
            "the next trading day)" + (", and the Friday check on the other Fridays" if WINNER.get("friday_falling_check") else "")
            + ". **Other days:** the 2:30 PM runs only refresh the data and place no orders (a month-end on Mon-Thu is still a "
            "rebalance day).\n"
            "- **Score:** 1 / 63-day volatility (calmest first), only stocks whose close is above their 200-day average; "
            f"ties by relative strength. At least {MIN_BARS} trading days of prices.\n"
            f"- **Picks:** the {N_PICKS} best, max {per_sector} per sector; do_not_buy stocks are never bought (the next "
            "stock takes the slot). No earnings rule.\n"
            f"- **Size:** equal weights, scaled to {LIVE_INVESTED:.0%} invested, each rounded down to 0.01%"
            + (f"; max {MAX_WEIGHT:.0%} per stock" if MAX_WEIGHT else "") + ".\n"
            + (f"- **Market filter:** with {REGIME_OFF} at a rebalance, every weight is {HALVED}.\n" if SCALE else "")
            + (f"- **Rebalance:** every pick is brought back to its weight unless within {band * 100:g} point of it.\n" if band else "")
            + ("- **Keep rule:** at a rebalance a holding that is no longer a pick is sold only if its price is below its "
               "Alpaca average entry price or its close is at/below its 200-day average; one in profit and in an uptrend is "
               "kept as it is (not sold, not trimmed) and the picks share the rest of the 99%.\n"
               if WINNER.get("keep_profit_at_rebalance") else "")
            + ("- **Friday check:** every Friday that is not a rebalance day (2:30 PM CT), a holding is sold in full when it "
               "is below its Alpaca average entry price, its close is at/below its 200-day average, or it is no longer one of "
               "the day's top-10 picks; each sale buys the best-ranked pick not held with the same dollars"
               + (f" (max {MAX_WEIGHT * LIVE_INVESTED:.1%} of equity)" if MAX_WEIGHT else "")
               + ", do_not_buy skipped; none left = cash. No trims or top-ups.\n" if WINNER.get("friday_falling_check") else "")
            + ("- **Earnings-day stop:** off (no orders from earnings_stop.py).\n" if not WINNER.get("earnings_day_stop", True) else "")
            + (f"- **Two halves:** the account is split {SLEEVES['lowvol']:.0%} / {SLEEVES['sector']:.0%}. The rules above run the "
               "**low-volatility half** (weights are fractions of that half). The **sector-rotation half**: the 3 sector "
               "ETFs with the best 63-day return, then the best stocks of those sectors by the C6 score (50% technical + "
               f"50% relative strength), top {N_PICKS}, weights ∝ 1 / 63-day volatility, max {max(1, int(0.4 * N_PICKS))} per sector, max 20% of the half (the extra stays cash), "
               f"{HALVED} with {REGIME_OFF}, do_not_buy never bought; a full rebalance of that half every Friday 2:30 PM CT "
               "(the week's last trading day: Thursday when Friday is a holiday; Mondays Oct 19-22, 2026). When the month's last "
               "trading day is a Friday both halves rebalance that day, without overlap. **No overlap:** the sector half never buys a stock the low-volatility half "
               "holds or targets; if the low-volatility half buys one the sector half holds, the shares move over at the "
               "current price (no trade) and the sector half rebalances that day without it. Each half is sized from its "
               "own value and keeps its own cash (Reports/sleeve_ledger.json); a half that is not rebalancing is left "
               "alone. Holdings no half owns are sold at the next trade run.\n" if SLEEVES else "")
            + "- **Orders:** limit orders at the live quote (buy at the ask + 0.05%, sell at the bid - 0.05%); the 9 AM CT "
            "check the next trading day sends any rest. A missed rebalance is caught up at the next market session.\n")

def rules_text():
    """The live trading rules in plain words, in one place (numbers from backtest_engine.WINNER). The weekly C6 text is
    the non-low-volatility fallback (a switch back); the Mon-Thu rules are gone from the live code (Oct 16, 2026)."""
    if LOW_VOL:
        return low_vol_rules_text()
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
            f"- **Size:** weights ∝ 1 / 63-day volatility (less volatile = larger), scaled to {LIVE_INVESTED:.0%} invested, "
            "each weight rounded down to 0.01%.\n"
            + (f"- **Market filter:** with {REGIME_OFF} at a rebalance, every weight is {HALVED}.\n" if SCALE else "")
            + (f"- **Max per stock:** No single stock gets more than {MAX_WEIGHT:.0%}; any extra stays in cash.\n" if MAX_WEIGHT else "")
            + (f"- **Rebalance:** every pick is brought back to its weight unless it is within {band * 100:g} percentage point "
               "of it; overweight holdings are trimmed so new buys get their full weight.\n" if band else "")
            + (f"- **Earnings:** a stock that is not held is not bought when its next earnings date is within {EARNINGS} "
               "calendar days; on Friday its slot goes to the next eligible stock. "
               "A held stock is not topped up before them.\n" if EARNINGS else "")
            + (f"- **Friday replacement:** a new pick the account does not hold that is blocked (earnings within {EARNINGS} days, "
               "or no buy back after an earnings-day stop sale) gets no cash slot: the next best-ranked eligible stock that is not "
               f"already a pick (score above 0, not blocked, down to rank {FRIDAY_SUB_MAX_RANK}) is bought at the blocked "
               f"pick's weight{f' (max {MAX_WEIGHT * LIVE_INVESTED:.1%})' if MAX_WEIGHT else ''}, so the account still ends "
               f"with {N_PICKS} stocks; only if none is left does that weight stay in cash.\n" if EARNINGS else "")
            + "- **Earnings-day stop:** on a held stock's earnings day the price is checked every 30 seconds in the "
            "pre-market, regular and after-hours sessions; if it is ever 5% or more below the previous trading day's close, "
            "all whole shares are sold once with a limit at the bid - 0.05% (extended hours allowed; the fraction at the "
            "9 AM CT check). Earnings day = the session where the reaction happens: a before-open report is watched that "
            "day; an after-close (or unknown-time) report that day and the next day's pre-market and regular session "
            "(then measured against the earnings-day close). The cash waits for the next scheduled run, and the stock is "
            "not bought back until after its last earnings-day session.\n"
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
BENCH_CSV = os.path.join(REPORTS, "benchmark_prices.csv")
NEWS_CSV = os.path.join(REPORTS, "news_cleaned_df.csv")
SHORT_HISTORY_CSV = os.path.join(REPORTS, "short_history_reference.csv")   # main_signal_analysis.ipynb: too new to trade
COMPANY_XLSX = os.path.join(REPORTS, "complete_company_analysis.xlsx")
NOTES_CSV = os.path.join(REPORTS, "my_notes.csv")   # the Home tab's notes table (written by the dashboard)
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
