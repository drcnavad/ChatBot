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
    from backtest_engine import WINNER
except Exception:
    WINNER = {}

EARNINGS = WINNER.get("earnings_block_days")                         # no new buys with earnings within N days (None = off)
N_PICKS = WINNER.get("n", 10)                                        # stocks picked
STRATEGY_NAME = "Live: Short-term relative strength, skip stretched buys"
REGIME_OFF = f"{WINNER.get('regime_symbol', 'QQQ')} at or below its 200-day average"   # market filter off
SCALE = WINNER.get("regime_scale", 0.5) if WINNER.get("use_regime", True) else None
HALVED = "halved" if SCALE == 0.5 else f"multiplied by {SCALE:g}" if SCALE else "unchanged"

def rules_text():
    """The live rules (from the first RS short-term rebalance, Fri Oct 30, 2026; defensive sleeve off Oct 29) in plain words."""
    return (f"**Strategy rules: {STRATEGY_NAME}**\n"
            "- **Whole account** runs forward_test's \"Short-term relative strength, skip stretched buys\" exactly (from Fri Nov 6, 2026) (Reports/sleeve_ledger.json).\n"
            "- **Score:** relative strength over 10, 21 and 63 trading days, the 63-day counted double from Fri Nov 6, 2026 (60% stock vs its sector ETF, 40% that ETF vs SPY); "
            "a stock more than 15% above its 20-day average not newly bought (held ones kept); the top 10 (at most 4 from one sector), equal weights (~9.9% each), 99% invested. No defensive sleeve (removed Oct 29, 2026).\n"
            "- **When:** full rebalance every Friday 2:30 PM CT (Thursday when Friday is a holiday) only; no Mon-Thu trades "
            "(Nov 4, 2026). A blocked pick is replaced by the next eligible, not stretched stock ranked 20 or better. "
            "Strategy locked to ~Jan 29, 2027. On early-close days (market closes "
            "12:00 PM CT: day after Thanksgiving, July 3 / Dec 24 when weekday) the decision runs at **11:00 AM CT** instead.\n"
            "- **Filters:** no new buy with earnings within 5 trading days (a held stock is not topped up); with "
            f"{REGIME_OFF} every weight is {HALVED}; do_not_buy never bought, do_not_sell never sold; a holding within 1 "
            "point of its weight is not traded.\n"
            "- **Cash:** each pick targets (equity × 0.99) / 10; leftover cash stays cash (no spreading, Oct 31, 2026).\n"
            "- **Earnings-day stop (ON, Nov 4, 2026):** a held stock 5% or more below the prior close on its earnings day is sold "
            "(do_not_sell exempt) and not bought back until after that session.\n"
            "- **Orders:** sells first, limit orders at the live quote (buy at the ask + 0.05%, sell at the bid - 0.05%); the "
            "9 AM CT check the next trading day sends any rest.\n")

# Report files (all written by run_all.py)
REPORTS = os.path.join(ROOT, "Reports")
SIGNAL_CSV = os.path.join(REPORTS, "signal_analysis.csv")
EARNINGS_CSV = os.path.join(REPORTS, "earnings_date.csv")
PICKS_CSV = os.path.join(REPORTS, "strategy_picks.csv")
CHANGES_CSV = os.path.join(REPORTS, "strategy_changes.csv")
HOLDINGS_CSV = os.path.join(REPORTS, "strategy_holdings.csv")
DECISIONS_CSV = os.path.join(REPORTS, "strategy_decisions.csv")
BENCH_CSV = os.path.join(REPORTS, "benchmark_prices.csv")
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
