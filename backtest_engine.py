"""
backtest_engine.py
------------------
Point-in-time signal construction + honest portfolio backtests for the stock pipeline.

Rules enforced everywhere:
  * every feature at date t uses only bars <= t (no full-series min/max, no unconfirmed swings)
  * decisions are made at the CLOSE of day t and filled at the OPEN of day t+1
  * 0.1% cost per side on traded notional, no interest on cash
  * partial intraday bars are dropped before backtesting
  * fundamentals / sentiment have no history -> treated as UNAVAILABLE in backtests
  * QQQ / SPY / sector ETFs are benchmarks and inputs, never traded by the strategies

Fidelity limits (what the backtest does NOT model):
  * fills are modeled at the next session's open; live trades evening extended-hours limit
    orders plus a morning fill-check for leftovers
  * no partial fills and no rejected orders (every modeled order fills in full at the open)
  * fractional shares (live trades whole shares)
  * 0.1%/side cost only: no spread, no borrow fees, no corporate-action handling
  * a symbol untradable on its add day (halt) is not bought later; a delisted holding is
    liquidated at its last close on the first session with no bars

Market data only: market_data_client() (historical bars). No trading endpoints.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
                                    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday)
from pandas.tseries.offsets import CustomBusinessDay

import sector_mapping

log = logging.getLogger("backtest_engine")

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "Reports"
CACHE_DIR = REPORTS_DIR / "cache"
EASTERN = ZoneInfo("America/New_York")

COST = 0.001                 # per side
MIN_BARS = 200               # a stock is eligible once it has a full ma_200 (handles late IPOs consistently)
DATA_START = "2023-06-01"    # warm-up for ma_200 and 252-bar rescaling windows (live pipeline)
LONG_CACHE, LONG_START = "bars_daily_long.pkl", "2020-06-01"   # long bar history for the backtest and the tests
RS_WINDOWS = (21, 63, 126)
RS_WEIGHTS = {"stock_vs_sector": 0.6, "sector_vs_spy": 0.4}
REGIME_SYMBOL, REGIME_MA = "SPY", 200

# The live rules (WINNER). History: C6 = weekly top-10 ranking + SOFT regime (at a rebalance where QQQ closes below its
# 200-day average, all weights are halved) won the 2022-04 -> now walk-forward tests. Tested 2026-09-24 and not adopted:
# max 2 per sector (statistically tied with 4, lagged the last 12 months; Reports/strategy_comparison_sector_caps.csv),
# max 5-8 / no cap (failed the never-seen 2022-24 period), rank buffers, score exits, minimum holds, stops, thresholds.
# The universe is hand-picked with hindsight, which inflates absolute backtest returns. Backtest: backtest.ipynb.
WINNER = {
    "name": "C6: weekly top-10 ranking, max 4 per sector + soft QQQ regime",
    "tag": "C6",
    "n": 10,                # number of names held
    "w_tech": 0.5,          # score = 0.5 * Technical_Score + 0.5 * RS_score
    "use_regime": True,     # regime is used in SOFT mode (see regime_scale)
    "regime_symbol": "QQQ", # regime = QQQ close > its 200-day SMA
    "regime_scale": 0.5,    # regime off at a rebalance -> weights x 0.5 (no hard block on new names)
    "min_score": 0.0,       # only names with score > 0
    "sector_cap": 0.4,      # max 40% of names per sector (= 4 of 10); 0.2 (cap 2, C6b) tested 2026-09-24, not adopted
    "vol_sizing": True,     # inverse 63-day volatility weights
    "buffer_rank": None,    # rank buffer did not help in the walk-forward
    "atr_stop_k": 3.0,      # ATR multiple shown as a risk reference in the app (not an automatic exit)
    "rs_benchmark": "etf",  # relative-strength benchmark: 'etf' (live) | 'sector_median' | 'median_all' (tested 2026-09-24, see relative_strength)
    # Mid-week swap (LIVE since 2026-09-24, user decision; variant D of Reports/rebalance_frequency_test.csv, +438% / Sharpe 1.46
    # vs +391% / 1.40 weekly-only, 2022-04 -> 2026-09, 0.1%/side). At the Mon and Wed closes (first session on/after that
    # weekday if it is a holiday, skipped when it is the week's last session = the Friday rebalance): if a NOT-held stock ranks in
    # the top `enter_top` AND a held stock ranks below `exit_below`, the worst-ranked held stock is sold and the best new
    # entrant bought with the same weight (sector cap respected; repeated while pairs qualify); filled at the next open.
    # REVERT: set "midweek_swap": None and rerun `python run_all.py` -> plain weekly C6-U96 (tag loses "-MW").
    "midweek_swap": {"enter_top": 3, "exit_below": 15, "days": ["Mon", "Wed"]},
    # Mid-week exit (LIVE from 2026-09-25, user decision; variant S3 of Reports/sell_rule_test.csv: +421.6% / Sharpe 1.47 / max DD
    # -32.2% / never-seen Sharpe 0.82 vs plain MW +438.1% / 1.46 / -31.8% / 0.84). At each Mon/Wed check, AFTER the swap step
    # above: every holding ranked worse than this (or no longer qualifying) is sold at the next open and the cash stays idle
    # until the Friday rebalance. Needs "midweek_swap" on. Tag gets "30" (C6-U96-MW30).
    # REVERT: set "midweek_exit_below": None and rerun `python run_all.py` -> plain C6-U96-MW.
    "midweek_exit_below": 30,
    # Selection from ranks 1-20 only (LIVE from 2026-09-25, user decision, not a tested rule). Weekly: walk ranks 1..20 with the
    # max-4-per-sector cap; if fewer than n are filled, fill the rest from the unused ranks 1..20 in rank order IGNORING the
    # sector cap; never pick worse than rank 20; fewer than n qualifying names -> the rest stays cash. Mid-week (cap_soft):
    # a non-held top-3 stock always qualifies regardless of sector and replaces the worst-ranked holding below 15.
    # Tag gets "-T20" (C6-U96-T20-MW30); "-T20H" if cap_soft is False (hard cap inside the top 20), "-SC" for cap_soft alone.
    # REVERT: "max_pick_rank": None and "cap_soft": False, then `python run_all.py` -> C6-U96-MW30.
    "max_pick_rank": 20,
    "cap_soft": True,
    # Earnings rule (LIVE from 2026-09-25, user decision, not a tested rule): a stock that is NOT held is not bought when its
    # next earnings date E is within the next N calendar days after the decision date d (d < E <= d + N), at the Friday
    # rebalance and at the Mon/Wed swap. Its slot goes to the next eligible stock within ranks 1-20 (same T20 logic; none ->
    # cash); a top-3 swap candidate with earnings is skipped. Held stocks are never sold because of earnings. Dates:
    # Reports/earnings_date.csv. Tag gets "-E5". REVERT: "earnings_block_days": None, then `python run_all.py`.
    "earnings_block_days": 5,
}


# Universe label: when the tested universe expansion is switched on (sector_mapping.EXPANDED_UNIVERSE), tracking rows and the
# app say e.g. "C6-U142" so rows from the 78-stock universe stay distinguishable. No effect while the switch is None.
# LIVE since 2026-09-24 (user decision): EXPANDED_UNIVERSE = "u96" -> tag C6-U96 (U91 + CRDO NBIS LITE CLS RBRK; ranks 1-10, max 4 per
# sector, RS vs sector ETF). Before that: "high_beta_91" -> C6-U91, None -> C6 (78 stocks).
if getattr(sector_mapping, "EXPANDED_UNIVERSE", None):
    WINNER["tag"] = f'{WINNER["tag"]}-U{len(sector_mapping.tradable_symbols)}'
    WINNER["name"] = f'{WINNER["name"]} (expanded universe, {len(sector_mapping.tradable_symbols)} stocks)'
if WINNER.get("rs_benchmark", "etf") != "etf":   # median-based relative strength (tested 2026-09-24, not live)
    WINNER["tag"] += {"sector_median": "-MED", "median_all": "-MEDALL"}[WINNER["rs_benchmark"]]
    WINNER["name"] += f' [RS vs {WINNER["rs_benchmark"].replace("_", " ")}]'
if WINNER.get("max_pick_rank") or WINNER.get("cap_soft"):   # picks from ranks 1..20, soft sector cap (live from 2026-09-25)
    WINNER["tag"] += (f'-T{WINNER["max_pick_rank"]}' + ("" if WINNER.get("cap_soft") else "H")) if WINNER.get("max_pick_rank") else "-SC"
    WINNER["name"] += ((f' [picks from ranks 1-{WINNER["max_pick_rank"]} only' if WINNER.get("max_pick_rank") else " [")
                       + ("; sector cap relaxed to fill the 10 slots; top-3 swaps ignore the cap]" if WINNER.get("cap_soft") else "]"))
if WINNER.get("midweek_swap"):                   # mid-week swap on top of the weekly rebalance (live since 2026-09-24)
    _mw = WINNER["midweek_swap"]
    WINNER["tag"] += "-MW"
    WINNER["name"] += (f' + mid-week swap ({"/".join(_mw["days"])} close: top {_mw["enter_top"]} in, '
                       f'below rank {_mw["exit_below"]} out)')
    if WINNER.get("midweek_exit_below"):         # mid-week exit to cash (live from 2026-09-25)
        WINNER["tag"] += str(WINNER["midweek_exit_below"])
        WINNER["name"] += f' + mid-week exit (sell if worse than rank {WINNER["midweek_exit_below"]}, cash until Friday)'
if WINNER.get("earnings_block_days"):             # no new buys shortly before earnings (live from 2026-09-25)
    WINNER["tag"] += f'-E{WINNER["earnings_block_days"]}'
    WINNER["name"] += f' + no new buys with earnings in the next {WINNER["earnings_block_days"]} days'


def winner_max_per_sector():
    """Max names per sector implied by WINNER (same formula as rank_targets)."""
    return max(1, int(np.floor(WINNER["sector_cap"] * WINNER["n"])))


def winner_rank_args(regime):
    """kwargs for rank_targets() implementing WINNER."""
    S = WINNER
    return dict(n=S["n"], regime=regime if S["use_regime"] else None, min_score=S["min_score"],
                sector_cap=S["sector_cap"], vol_sizing=S["vol_sizing"], buffer_rank=S["buffer_rank"],
                regime_scale=S["regime_scale"] if S["use_regime"] else None,
                max_pick_rank=S.get("max_pick_rank"), cap_soft=bool(S.get("cap_soft")))


TRADABLE = list(sector_mapping.tradable_symbols)
SECTOR_ETFS = list(sector_mapping.sector_etfs)
BENCHMARKS = list(sector_mapping.BENCHMARK_SYMBOLS)


# ----------------------------------------------------------------------------- calendar
class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Full-day NYSE holidays (early closes are ignored)."""
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


NYSE_SESSION = CustomBusinessDay(calendar=NYSEHolidayCalendar())


def next_sessions(after, n):
    """The n NYSE sessions strictly after `after`."""
    return pd.date_range(pd.Timestamp(after).normalize() + NYSE_SESSION, periods=n, freq=NYSE_SESSION)


# ----------------------------------------------------------------------------- data
def market_data_client():
    """Alpaca market-data client (historical bars only; no trading or account endpoints). Keys: ALPACA_API_KEY /
    ALPACA_SECRET_KEY in .env."""
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise EnvironmentError("Missing API keys. Make sure .env has ALPACA_API_KEY and ALPACA_SECRET_KEY")
    return StockHistoricalDataClient(key, secret)


def fetch_daily_bars(symbols, start=DATA_START, end=None, data_client=None, retries=3):
    """Split+dividend adjusted daily bars (long format). Market-data endpoint only."""
    from alpaca.data.enums import Adjustment
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    if data_client is None:
        data_client = market_data_client()
    symbols = list(dict.fromkeys(symbols))
    # Free Alpaca plans cannot query the most recent 15 minutes of SIP data -> stop 16 minutes ago.
    end_ts = pd.Timestamp(end).to_pydatetime() if end else datetime.now(EASTERN) - timedelta(minutes=16)

    def fetch_chunk(chunk):
        """Download one chunk of symbols, retrying on errors."""
        req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                               start=pd.Timestamp(start).to_pydatetime(), end=end_ts, adjustment=Adjustment.ALL)
        for attempt in range(retries):
            try:
                got = data_client.get_stock_bars(req).df
                return got.reset_index() if got is not None and len(got) else None
            except Exception as e:  # network / rate limit (HTTP 429) -> back off and retry
                if attempt == retries - 1:
                    raise RuntimeError(f"Alpaca bars failed for {chunk[:3]}... after {retries} tries: {e}") from e
                wait = 2 ** attempt * 3
                log.warning("Alpaca bars attempt %d failed (%s); retrying in %ss", attempt + 1, e, wait)
                time.sleep(wait)

    # chunks of 40 symbols, requested in parallel (3 requests for the live universe); results are kept in chunk order
    chunks = [symbols[i:i + 40] for i in range(0, len(symbols), 40)]
    with ThreadPoolExecutor(max_workers=min(4, len(chunks) or 1)) as pool:
        frames = [f for f in pool.map(fetch_chunk, chunks) if f is not None]
    if not frames:
        raise RuntimeError("Alpaca returned no bars at all - check keys / network")
    bars = pd.concat(frames, ignore_index=True).rename(columns={
        "symbol": "Symbol", "timestamp": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume"})
    bars["Date"] = pd.to_datetime(bars["Date"].dt.tz_convert(EASTERN).dt.date)
    missing = sorted(set(symbols) - set(bars["Symbol"]))
    if missing:
        log.warning("No Alpaca bars for: %s", missing)
    bars = apply_history_start(bars)
    return bars[["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]].sort_values(["Symbol", "Date"]).reset_index(drop=True)


def apply_history_start(bars):
    """Drop vendor bars before sector_mapping.HISTORY_START[symbol] (e.g. NBIS before 2024-10-21 = Yandex N.V. history incl. a flat
    zero-volume 2022-24 halt). Bar counts, MAs and the 200-bar eligibility then start at the first real session: no lookahead."""
    starts = getattr(sector_mapping, "HISTORY_START", {}) or {}
    if not starts or not len(bars):
        return bars
    first = bars["Symbol"].map(lambda s: pd.Timestamp(starts[s]) if s in starts else pd.NaT)
    return bars[first.isna() | (bars["Date"] >= first)].reset_index(drop=True)


def drop_partial_last_bar(bars, now=None, close_buffer_min=30):
    """Drop today's bar if the US session (16:00 ET + buffer) has not finished yet."""
    now = now or datetime.now(EASTERN)
    today = pd.Timestamp(now.date())
    session_done = (now.hour * 60 + now.minute) >= (16 * 60 + close_buffer_min)
    if not session_done and (bars["Date"] == today).any():
        return bars[bars["Date"] < today].reset_index(drop=True), True
    return bars, False


def load_bars(refresh=False, cache_name=LONG_CACHE, start=LONG_START):
    """All bars needed by the backtest (tradable + benchmarks + sector ETFs), cached under Reports/cache.
    The cache is refetched when asked, when it is missing, or when it starts later than `start`."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / cache_name
    stale = not path.exists() or pd.read_pickle(path)["Date"].min() > pd.Timestamp(start) + pd.Timedelta(days=7)
    if refresh or stale:
        bars = fetch_daily_bars(TRADABLE + BENCHMARKS + SECTOR_ETFS, start=start)
        bars.to_pickle(path)
    bars = apply_history_start(pd.read_pickle(path))
    bars, dropped = drop_partial_last_bar(bars)
    return bars, dropped


# ----------------------------------------------------------------------------- technical indicators
# (merged from signal_analysis_functions.py) calculate_technical_indicators (moving averages, RSI, MACD, Bollinger
# bands, ATR, OBV, Force Index) -> one signal column per indicator (+1 buy / 0 / -1 sell) -> weighted_signal combines
# them into combined_signal = Technical_Score. All rescaling is point-in-time (trailing windows only).
pd.options.display.float_format = '{:.2f}'.format    # notebook display only (2 decimals, all columns)
pd.set_option('display.max_columns', None)


def apply_by_symbol(df, fn, ticker_col='Symbol'):
    """Run fn on each symbol's rows and concatenate (keeps the Symbol column; avoids the deprecated
    DataFrameGroupBy.apply-on-grouping-columns behaviour that drops it in pandas 3)."""
    if df.empty:
        return fn(df)
    return pd.concat([fn(g) for _, g in df.groupby(ticker_col, sort=False)], ignore_index=False)

# Point-in-time settings: every indicator below uses only data available at each row's date.
SCALE_WINDOW = 252      # trailing window (bars) for rescaling Force Index / OBV (expanding until full)
SCALE_MIN_PERIODS = 20
OBV_SLOPE_DAYS = 3
OBV_THRESHOLD = 1.0     # 3-day net signed volume must exceed 1x the 20-day average daily volume
BB_WINDOW = 20          # same window as bb_middle/bb_upper/bb_lower in calculate_technical_indicators


def pit_scale(series, window=SCALE_WINDOW, min_periods=SCALE_MIN_PERIODS):
    """Point-in-time rescale to -100..100: value / trailing max(|value|) over `window` bars.

    """
    s = pd.Series(series, dtype=float)
    denom = s.abs().rolling(window, min_periods=min_periods).max()
    return (s / denom.replace(0, np.nan) * 100).clip(-100, 100)


def wilder_rsi(close, period=14):
    """RSI with Wilder smoothing; 100 when there are no losses, 50 when flat."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi


def calculate_technical_indicators(df):
    """Indicators for ONE symbol (rows sorted by date). All values are point-in-time."""
    df = df.copy()
    # --- Moving Averages ---
    for window in [10, 30, 50, 100, 200]:
        df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()

    # --- RSI (Wilder) ---
    df['rsi'] = wilder_rsi(df['Close'], 14)

    # --- ATR (Wilder, 14) for stops / volatility sizing ---
    prev_close = df['Close'].shift(1)
    true_range = pd.concat([df['High'] - df['Low'], (df['High'] - prev_close).abs(),
                            (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
    df['atr'] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # --- Bollinger Bands ---
    df['bb_middle'] = df['Close'].rolling(window=BB_WINDOW).mean()
    df['bb_std'] = df['Close'].rolling(window=BB_WINDOW).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    # --- MACD ---
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # --- Force Index ---
    # fi_raw keeps the sign/units; fi is rescaled point-in-time to -100..100 (trailing 252-bar max |fi|)
    df['fi_raw'] = (df['Close'].diff() * df['Volume']).ewm(span=13, adjust=False).mean()
    df['fi'] = pit_scale(df['fi_raw'])

    # --- OBV ---
    df['obv_std'] = (np.sign(df['Close'].diff()).fillna(0) * df['Volume']).cumsum()
    roll_min = df['obv_std'].rolling(SCALE_WINDOW, min_periods=SCALE_MIN_PERIODS).min()
    roll_max = df['obv_std'].rolling(SCALE_WINDOW, min_periods=SCALE_MIN_PERIODS).max()
    df['obv'] = ((df['obv_std'] - roll_min) / (roll_max - roll_min).replace(0, np.nan)).fillna(0.5)  # 0..1, trailing window
    avg_volume = df['Volume'].rolling(20, min_periods=5).mean()
    df['obv_slope'] = df['obv_std'].diff(OBV_SLOPE_DAYS) / avg_volume.replace(0, np.nan)  # in "average days of volume"

    # --- Adaptive Fibonacci levels from the most recent CONFIRMED swing high/low ---
    # A 3-bar fractal at bar j needs bar j+1 to exist, so it is only known from bar j+1 onward:
    # the swing value is placed on j+1 (shift(1)) and carried forward.
    high, low = df['High'], df['Low']
    is_swing_high = (high > high.shift(1)) & (high > high.shift(-1))
    is_swing_low = (low < low.shift(1)) & (low < low.shift(-1))
    last_high = high.where(is_swing_high).shift(1).ffill()
    last_low = low.where(is_swing_low).shift(1).ffill()
    last_high = last_high.fillna(high.cummax())
    last_low = last_low.fillna(low.cummin())

    diff = last_high - last_low
    df['fib_0%'] = last_high
    df['fib_23.6%'] = last_high - diff * 0.236
    df['fib_38.2%'] = last_high - diff * 0.382
    df['fib_50%'] = last_high - diff * 0.500
    df['fib_61.8%'] = last_high - diff * 0.618
    df['fib_76.4%'] = last_high - diff * 0.764
    df['fib_100%'] = last_low

    # --- Clean up --- (warm-up rows get 0; callers drop rows without a full ma_200)
    df = df.fillna(0)
    return df


def generate_strict_signals(df):
    """Moving-average signals (+1 when the close is 1% above an MA, -1 when 1% below; MA50/100/200 also need the
    shorter MAs to agree) and the OBV momentum signal."""
    # --- MA Signals ---
    for ma in ['ma_10', 'ma_30', 'ma_50', 'ma_100', 'ma_200']:
        buffer = df[ma] * 0.01  # Increased to 1% buffer for stricter BUY
        df[f'signal_{ma}'] = np.where(df['Close'] > df[ma] + buffer, 1,
                                      np.where(df['Close'] < df[ma] - buffer, -1, 0))
        
        # Additional strictness: For longer MAs, require shorter MA alignment
        if ma == 'ma_50':
            # For MA50 BUY, require MA10 > MA30 (short-term uptrend)
            df.loc[(df['signal_ma_50'] == 1) & (df['ma_10'] <= df['ma_30']), 'signal_ma_50'] = 0
            # STRICT: For MA50 SELL, require MA10 < MA30 (short-term downtrend)
            df.loc[(df['signal_ma_50'] == -1) & (df['ma_10'] >= df['ma_30']), 'signal_ma_50'] = 0
        elif ma == 'ma_100':
            # For MA100 BUY, require MA30 > MA50 (medium-term uptrend)
            df.loc[(df['signal_ma_100'] == 1) & (df['ma_30'] <= df['ma_50']), 'signal_ma_100'] = 0
            # STRICT: For MA100 SELL, require MA30 < MA50 (medium-term downtrend)
            df.loc[(df['signal_ma_100'] == -1) & (df['ma_30'] >= df['ma_50']), 'signal_ma_100'] = 0
        elif ma == 'ma_200':
            # For MA200 BUY, require MA50 > MA100 (long-term uptrend)
            df.loc[(df['signal_ma_200'] == 1) & (df['ma_50'] <= df['ma_100']), 'signal_ma_200'] = 0
            # STRICT: For MA200 SELL, require MA50 < MA100 (long-term downtrend)
            df.loc[(df['signal_ma_200'] == -1) & (df['ma_50'] >= df['ma_100']), 'signal_ma_200'] = 0

    # --- OBV momentum ---
    # obv_slope = 3-day OBV change / 20-day average volume (computed in calculate_technical_indicators).
    # Threshold is in the same units (1.0 = one average day of net buying), not on a 0-100 rescale.
    if 'obv_slope' not in df.columns:
        df['obv_slope'] = df.groupby('Symbol')['obv'].transform(lambda s: s.diff().rolling(OBV_SLOPE_DAYS).sum())
    df['signal_obv'] = np.where(df['obv_slope'] > OBV_THRESHOLD, 1,
                                np.where(df['obv_slope'] < -OBV_THRESHOLD, -1, 0))

    return df

def rsi_signals(group, lower=20, upper=85, ma_period=3, use_trend=True, oversold_threshold=30):
    """RSI signals for one symbol: +1 when RSI is oversold (< 30) and the price is at/above its short MA,
    -1 when RSI is overbought (> 70) and the price is at/below it."""
    group = group.copy()
    group['rsi_signal'] = 0
    
    # Rolling MA trend filter
    if use_trend:
        group['ma'] = group['Close'].rolling(ma_period).mean()  # no bfill (would use future bars)
    else:
        group['ma'] = group['Close'] * 0 + 1  # all True
    
    # --- BUY ---
    # STRICT: RSI oversold in uptrend - price must be above MA (not in downtrend)
    buy_condition = group['rsi'] < lower  # RSI < 20
    trend_buy = group['Close'] > group['ma']  # Price above short-term MA (uptrend)
    group.loc[buy_condition & trend_buy, 'rsi_signal'] = 1
    
    # STRICT: Moderate oversold (RSI 20-30) but ONLY if price is at least at MA level (not below)
    # Removed the loose conditions that allowed buying 20-30% below MA
    moderate_oversold = (group['rsi'] < oversold_threshold) & (group['rsi'] >= 20)
    price_at_ma = group['Close'] >= group['ma']  # Price at or above MA (strict)
    group.loc[moderate_oversold & price_at_ma, 'rsi_signal'] = 1
    
    # --- SELL ---
    # STRICT: RSI overbought in downtrend - price must be below MA (not in uptrend)
    sell_condition = group['rsi'] > upper  # RSI > 85
    trend_sell = group['Close'] < group['ma']  # Price below short-term MA (downtrend)
    group.loc[sell_condition & trend_sell, 'rsi_signal'] = -1
    
    # STRICT: Moderate overbought (RSI 70-85) but ONLY if price is at or below MA level (not above)
    # This catches overbought conditions in downtrends
    moderate_overbought = (group['rsi'] > 70) & (group['rsi'] <= upper)
    price_at_or_below_ma = group['Close'] <= group['ma']  # Price at or below MA (strict)
    group.loc[moderate_overbought & price_at_or_below_ma, 'rsi_signal'] = -1
    
    return group.drop(columns=['ma'])


def fi_signals_strict(df, lookback=3, min_fi=2):
    """
    fi_signal = 1 for buy, -1 for sell, 0 for hold.
    lookback = number of consecutive FI values needed
    min_fi = minimum absolute FI value to count toward streak
    """
    df = df.copy()
    df['fi_signal'] = 0

    # Only consider FI values above threshold
    df['fi_direction'] = df['fi'].apply(lambda x: 1 if x >= min_fi else -1 if x <= -min_fi else 0)
    
    # Compute streaks
    df['fi_streak'] = df['fi_direction'].groupby((df['fi_direction'] != df['fi_direction'].shift()).cumsum()).cumcount() + 1
    
    # STRICT: For BUY, require price to be in uptrend (Close > MA10) to avoid buying in downtrends
    # Calculate MA10 for trend confirmation if not already present
    ma_10_was_present = 'ma_10' in df.columns
    if not ma_10_was_present:
        df['ma_10'] = df['Close'].rolling(window=10).mean()
    
    # Buy after lookback consecutive positives AND price above MA10 (uptrend confirmation)
    buy_condition = (df['fi_direction'] == 1) & (df['fi_streak'] >= lookback)
    trend_confirmation = df['Close'] > df['ma_10']  # Price in uptrend
    df.loc[buy_condition & trend_confirmation, 'fi_signal'] = 1
    
    # STRICT: Sell after lookback consecutive negatives AND price below MA10 (downtrend confirmation)
    sell_condition = (df['fi_direction'] == -1) & (df['fi_streak'] >= lookback)
    downtrend_confirmation = df['Close'] < df['ma_10']  # Price in downtrend
    df.loc[sell_condition & downtrend_confirmation, 'fi_signal'] = -1

    # Clean up helper columns (drop ma_10 only if we created it)
    if not ma_10_was_present and 'ma_10' in df.columns:
        df = df.drop(columns=['ma_10'])
    df = df.drop(columns=["fi_direction", "fi_streak"], errors='ignore')
    return df

def bollinger_signal_middle(df, bb_window=BB_WINDOW, rsi_col='rsi', close_col='Close', ma_period=20, ticker_col='Symbol'):
    """
    Generates BB+RSI signals using middle band as trend filter.
    Now also detects oversold conditions when price is at/below lower Bollinger band.
    """
    def bb_middle_group(group):
        """Bollinger middle-band signals for one symbol."""
        group = group.copy()
        
        # Bollinger Bands (recalculate to ensure we have lower band)
        group['bb_middle'] = group[close_col].rolling(bb_window).mean()
        group['bb_std'] = group[close_col].rolling(bb_window).std()
        group['bb_lower'] = group['bb_middle'] - 2 * group['bb_std']
        group['bb_upper'] = group['bb_middle'] + 2 * group['bb_std']
        
        # Buy: price above middle band + RSI in healthy range (not overbought)
        # STRICT: RSI must be < 60 (not just < 70) to avoid buying in overbought conditions
        group['bb_signal'] = 0
        healthy_rsi = (group[rsi_col] < 60) & (group[rsi_col] > 30)  # RSI in healthy range
        price_above_middle = group[close_col] > group['bb_middle']
        group.loc[price_above_middle & healthy_rsi, 'bb_signal'] = 1
        
        # Additional buy signal: price at or below lower Bollinger band + RSI oversold
        # STRICT: RSI must be < 30 (not < 35) for stronger oversold confirmation
        oversold_bb = (group[close_col] <= group['bb_lower']) & (group[rsi_col] < 30)
        group.loc[oversold_bb, 'bb_signal'] = 1
        
        # STRICT: Sell: price below middle band + RSI in overbought range (not just > 30)
        # Require RSI > 50 to ensure we're selling in overbought conditions, not just neutral
        overbought_rsi = group[rsi_col] > 50  # RSI in overbought range
        price_below_middle = group[close_col] < group['bb_middle']
        group.loc[price_below_middle & overbought_rsi, 'bb_signal'] = -1
        
        # Additional sell signal: price at or above upper Bollinger band + RSI overbought
        # STRICT: RSI must be > 70 (not just > 50) for stronger overbought confirmation
        overbought_bb = (group[close_col] >= group['bb_upper']) & (group[rsi_col] > 70)
        group.loc[overbought_bb, 'bb_signal'] = -1
        
        return group.drop(columns=['bb_middle', 'bb_std', 'bb_lower', 'bb_upper'])
    
    return apply_by_symbol(df, bb_middle_group, ticker_col)

def macd_signals(group):
    """MACD signals for one symbol: +1 on a cross above the signal line with MACD rising, -1 on a cross below with MACD falling."""
    group = group.copy()
    group['macd_trade'] = 0
    
    # MACD crossover conditions
    cross_up = (group['macd'].shift(1) < group['macd_signal'].shift(1)) & (group['macd'] >= group['macd_signal'])
    cross_down = (group['macd'].shift(1) > group['macd_signal'].shift(1)) & (group['macd'] <= group['macd_signal'])
    
    # STRICT: For BUY, require MACD histogram to be positive (MACD > Signal) and increasing
    # This ensures we're buying on confirmed bullish momentum, not just a weak crossover
    macd_positive = group['macd'] > group['macd_signal']  # Histogram positive
    macd_increasing = group['macd'] > group['macd'].shift(1)  # MACD line increasing
    
    group.loc[cross_up & macd_positive & macd_increasing, 'macd_trade'] = 1
    
    # STRICT: For SELL, require MACD histogram to be negative (MACD < Signal) and decreasing
    # This ensures we're selling on confirmed bearish momentum, not just a weak crossover
    macd_negative = group['macd'] < group['macd_signal']  # Histogram negative
    macd_decreasing = group['macd'] < group['macd'].shift(1)  # MACD line decreasing
    
    group.loc[cross_down & macd_negative & macd_decreasing, 'macd_trade'] = -1
    
    return group

def fibonacci_signals(df, close_col='Close'):
    """
    Generates signals based on price position relative to Fibonacci retracement levels.
    Buy signal when price is near key support levels (fib_61.8%, fib_50%, fib_38.2%).
    Sell signal when price is near resistance levels (fib_0%, fib_23.6%).
    """
    df = df.copy()
    df['fib_signal'] = 0
    
    # Calculate distance from each Fibonacci level (as percentage)
    tolerance = 0.015  # Reduced to 1.5% tolerance for stricter matching
    
    # STRICT: Require trend confirmation - price should be above MA50 for BUY signals
    # This ensures we're buying at support in an uptrend, not in a downtrend
    ma_50_was_present = 'ma_50' in df.columns
    if not ma_50_was_present:
        df['ma_50'] = df[close_col].rolling(window=50).mean()
    
    # Buy signals: Price near support levels (fib_61.8%, fib_50%, fib_38.2%)
    for fib_level in ['fib_61.8%', 'fib_50%', 'fib_38.2%']:
        if fib_level in df.columns:
            distance = abs((df[close_col] - df[fib_level]) / df[fib_level])
            # Buy when price is near support and potentially bouncing up
            near_support = distance <= tolerance
            price_above_fib = df[close_col] >= df[fib_level] * 0.98  # Allow slight below
            # STRICT: Require price above MA50 (uptrend) to avoid buying in downtrends
            uptrend_confirmation = df[close_col] > df['ma_50']
            df.loc[near_support & price_above_fib & uptrend_confirmation, 'fib_signal'] = 1
    
    # Sell signals: Price near resistance levels (fib_0%, fib_23.6%)
    for fib_level in ['fib_0%', 'fib_23.6%']:
        if fib_level in df.columns:
            distance = abs((df[close_col] - df[fib_level]) / df[fib_level])
            # Sell when price is near resistance and potentially reversing
            near_resistance = distance <= tolerance
            price_below_fib = df[close_col] <= df[fib_level] * 1.02  # Allow slight above
            # STRICT: Require price below MA50 (downtrend) to avoid selling in uptrends
            # Note: ma_50 is still available here since we haven't dropped it yet
            downtrend_confirmation = df[close_col] < df['ma_50']
            df.loc[near_resistance & price_below_fib & downtrend_confirmation, 'fib_signal'] = -1
    
    # Clean up: drop ma_50 only if we created it (after both BUY and SELL signals are processed)
    if not ma_50_was_present and 'ma_50' in df.columns:
        df = df.drop(columns=['ma_50'])
    
    return df






def weighted_signal(df, weights=None, signal_cols=None, final_col='combined_signal'):
    """
    Combine multiple signals with given weights into a final score.
    Arguments:
        df: dataframe with signal columns
        weights: dict of {column_name: weight_in_percent}
        signal_cols: list of columns to include (optional, inferred from weights if None)
        final_col: name of output column
    Returns:
        df with new weighted score column
    """
    df = df.copy()
    
    # Default weights if not provided
    # Optimized based on strictness updates and signal reliability
    if weights is None:
        weights = {
            # Moving Averages - Trend indicators (Total: 38)
            # Higher weights for MAs with trend alignment confirmations
            'signal_ma_10': 3,      # Short-term, no alignment → Lower weight (less reliable)
            'signal_ma_30': 6,      # Medium-term, no alignment → Moderate weight
            'signal_ma_50': 10,     # Medium-term WITH MA10>MA30 alignment → High weight (very reliable)
            'signal_ma_100': 8,     # Long-term WITH MA30>MA50 alignment → High weight (very reliable)
            'signal_ma_200': 11,    # Major trend WITH MA50>MA100 alignment → Highest weight (most reliable)
            
            # Momentum Indicators (Total: 20)
            # High weights for momentum indicators with trend confirmations
            'rsi_signal': 10,       # Trend-confirmed RSI → Very high weight (most reliable momentum)
            'macd_trade': 4,        # Crossovers lagged (pointed the wrong way in 1-year check) → Low weight
            'fi_signal': 6,         # Streak + trend confirmed → Moderate weight (good reliability)
            
            # Volume & Volatility (Total: 7)
            # Moderate weights - important but may generate fewer signals due to strictness
            'signal_obv': 4,        # Threshold-based, might be too strict → Lower weight
            'bb_signal': 3,         # RSI-filtered BB (pointed the wrong way in 1-year check) → Low weight
            
            # Support/Resistance (Total: 3)
            # Lower weight - strict conditions may generate fewer signals
            'fib_signal': 3         # Trend-confirmed Fibonacci (weak in 1-year check) → Low weight
            }
    
    if signal_cols is None:
        signal_cols = list(weights.keys())
    
    # Normalize weights to sum to 100
    total_weight = sum(weights.values())
    norm_weights = {k: v/total_weight for k,v in weights.items()}
    
    # Compute weighted score
    df[final_col] = 0.0
    
    for col in signal_cols:
        if col not in df.columns:
            continue
            
        signal_value = df[col].fillna(0)
        weight = norm_weights[col] * 100
        df[final_col] += signal_value * weight
    
    return df


# ----------------------------------------------------------------------------- signals
def build_technical(bars, symbols=None):
    """Current technical rules recomputed from raw bars, point-in-time. Returns long df."""
    symbols = symbols or TRADABLE
    b = bars[bars["Symbol"].isin(symbols)]
    df = pd.concat([calculate_technical_indicators(g.reset_index(drop=True)) for _, g in b.groupby("Symbol")],
                   ignore_index=True)
    df["bar_n"] = df.groupby("Symbol").cumcount() + 1
    df = generate_strict_signals(df)
    df = apply_by_symbol(df, rsi_signals)
    df = apply_by_symbol(df, fi_signals_strict)
    df = bollinger_signal_middle(df)
    df = apply_by_symbol(df, macd_signals)
    df = fibonacci_signals(df)
    df = weighted_signal(df).reset_index(drop=True)
    df["Technical_Score"] = pd.to_numeric(df["combined_signal"], errors="coerce").fillna(0.0)
    df["eligible"] = df["bar_n"] >= MIN_BARS
    # B-6: warm-up rows lack a full ma_200 lookback, so their indicator values (NaN -> 0 in
    # calculate_technical_indicators) and the signals derived from them are fake (e.g. Close >
    # ma_200 = 0 reads as a buy). Zero them so no caller can turn a warm-up row into a rank/pick
    # even if it forgets the `eligible` filter (rank_targets and apply_midweek_swaps already filter).
    warm = ~df["eligible"].to_numpy(dtype=bool)
    if warm.any():
        sig_cols = [c for c in df.columns
                    if c.startswith("signal_") or c.endswith("_signal") or c == "macd_trade"]
        df.loc[warm, sig_cols + ["Technical_Score"]] = 0.0
    return df.sort_values(["Symbol", "Date"]).reset_index(drop=True)


def wide(df, col, index="Date", columns="Symbol"):
    """Long table -> wide Date x Symbol matrix of one column (last value per cell)."""
    return df.pivot_table(index=index, columns=columns, values=col, aggfunc="last").sort_index()


def bool_wide(df, col, index, columns):
    """Boolean dates x symbols matrix (missing -> False) without object-dtype downcasting warnings."""
    w = df.pivot_table(index="Date", columns="Symbol", values=col, aggfunc="last").astype(float)
    return w.reindex(index=index, columns=columns).fillna(0.0).astype(bool)


def _peer_median(r, symbols, sector_of, min_peers=3, leave_one_out=True):
    """Per stock: median return of its sector peers inside `symbols` (excluding the stock itself when leave_one_out),
    NaN where fewer than `min_peers` peers have a value on that date (the caller falls back to the sector ETF)."""
    out = {}
    by_sector = {}
    for s in symbols:
        by_sector.setdefault(sector_of.get(s), []).append(s)
    for sec, members in by_sector.items():
        block = r[members]
        for s in members:
            peers = block.drop(columns=[s]) if leave_one_out else block
            med = peers.median(axis=1, skipna=True)
            out[s] = med.where(peers.notna().sum(axis=1) >= min_peers)
    return pd.DataFrame(out, index=r.index)[symbols]


def relative_strength(close_w, symbols=None, vol_adjust=False, benchmark=None):
    """Point-in-time relative strength, all -100..100 cross-sectional scores.

    benchmark (default WINNER['rs_benchmark'], live = 'etf'):
      'etf'           stock vs its SPDR sector ETF; sector ETF vs SPY (original rule)
      'sector_median' stock vs the MEDIAN return of its sector peers in `symbols` (leave-one-out, >= 3 peers with data,
                      otherwise the sector ETF); the sector-vs-SPY half is unchanged (ETF vs SPY)
      'median_all'    as 'sector_median', and the second half = sector median (all members, >= 3) vs the median of the
                      whole universe instead of ETF vs SPY (falls back to ETF vs SPY for small sectors)
    Medians are cross-sectional on each date over the same 21/63/126-day returns, so there is no lookahead.

    stock_vs_sector: stock return - its sector ETF return; sector_vs_spy: sector ETF return - SPY return,
    each over 21/63/126 trading days, converted to cross-sectional percentile ranks per date and blended
    60/40. Also returns Sector_RS_63 (sector ETF 63-day excess return vs SPY, in %) for display.
    """
    symbols = symbols or [s for s in TRADABLE if s in close_w.columns]
    benchmark = benchmark or WINNER.get("rs_benchmark", "etf")
    assert benchmark in ("etf", "sector_median", "median_all"), benchmark
    etf_of = {s: sector_mapping.sector_etf_for(s) for s in symbols}
    parts_sv, parts_ss = [], []
    daily = close_w.pct_change(fill_method=None) if vol_adjust else None
    for w in RS_WINDOWS:
        r = close_w / close_w.shift(w) - 1
        stock = r[symbols]
        sector = pd.DataFrame({s: r[etf_of[s]] if etf_of[s] in r else r["SPY"] for s in symbols})
        if benchmark == "etf":
            sv = stock - sector
            ss = sector.sub(r["SPY"], axis=0)
        else:
            sec_of = sector_mapping.symbol_sector
            sv = stock - _peer_median(r, symbols, sec_of).fillna(sector)
            if benchmark == "sector_median":
                ss = sector.sub(r["SPY"], axis=0)
            else:
                sec_all = _peer_median(r, symbols, sec_of, leave_one_out=False)
                univ_med = stock.median(axis=1, skipna=True)
                ss = sec_all.sub(univ_med, axis=0).fillna(sector.sub(r["SPY"], axis=0))
        if vol_adjust:  # excess return per unit of the stock's own volatility over the same window
            vol = (daily.rolling(w, min_periods=max(10, w // 2)).std() * np.sqrt(w)).replace(0, np.nan)
            sv = sv / vol[symbols]
            sec_vol = pd.DataFrame({s: vol[etf_of[s]] if etf_of[s] in vol else vol["SPY"] for s in symbols})
            ss = ss / sec_vol
        parts_sv.append(sv.rank(axis=1, pct=True))
        parts_ss.append(ss.rank(axis=1, pct=True))
    pct = (RS_WEIGHTS["stock_vs_sector"] * sum(parts_sv) / len(parts_sv)
           + RS_WEIGHTS["sector_vs_spy"] * sum(parts_ss) / len(parts_ss))
    rs_score = (pct * 2 - 1) * 100
    r63 = close_w / close_w.shift(63) - 1
    sector_rs63 = pd.DataFrame({s: (r63[etf_of[s]] if etf_of[s] in r63 else r63["SPY"]) - r63["SPY"] for s in symbols}) * 100
    return rs_score, sector_rs63


def regime_series(close_w, symbol=REGIME_SYMBOL, ma=REGIME_MA):
    """Market filter: True while the regime symbol (QQQ) closes above its moving average."""
    c = close_w[symbol]
    return (c > c.rolling(ma).mean()).fillna(False)


# ----------------------------------------------------------------------------- earnings
def load_earnings(path=None):
    """Reports/earnings_date.csv with clean symbols and normalized dates."""
    e = pd.read_csv(path or REPORTS_DIR / "earnings_date.csv")
    e["Symbol"] = e["Symbol"].str.strip().str.upper()
    e["Earnings Date"] = pd.to_datetime(e["Earnings Date"], errors="coerce").dt.normalize()
    return e.dropna(subset=["Earnings Date"])


def earnings_days_ahead(dates, symbols, earnings, block_days):
    """Days from each decision date d to the stock's next earnings date E when d < E <= d + block_days (calendar days);
    NaN when there is none in that window or no date on file. `earnings` = load_earnings() frame."""
    d = pd.DatetimeIndex(dates).normalize().values
    out = np.full((len(d), len(symbols)), np.nan)
    by_symbol = {s: np.sort(g.dropna().unique()) for s, g in earnings.groupby("Symbol")["Earnings Date"]}
    for j, sym in enumerate(symbols):
        e = by_symbol.get(sym)
        if e is None or not len(e):
            continue
        k = np.searchsorted(e, d, side="right")                     # first earnings date strictly after d
        nxt = e[np.minimum(k, len(e) - 1)]
        days = (nxt - d) / np.timedelta64(1, "D")
        out[:, j] = np.where((k < len(e)) & (days <= block_days), days, np.nan)
    return pd.DataFrame(out, index=dates, columns=symbols)


def earnings_note(days, d):
    """'earnings in 3 days (Wed Sep 30)' for a decision on date d."""
    days = int(days)
    return f"earnings in {days} day{'' if days == 1 else 's'} ({pd.Timestamp(d) + pd.Timedelta(days=days):%a %b %d})"


# ----------------------------------------------------------------------------- simulator
def simulate(open_w, close_w, target, start, end=None, rebalance=None, cost=COST):
    """Share-based daily simulation.

    target: weights decided at the close of each date (row d executes at the open of d+1).
    Trading follows the LIVE rule: only adds and exits are ever traded, holds are never resized.
    An "add" is a symbol whose target went 0 -> positive since the previous decision: it is traded
    to exactly its target weight (bought, or trimmed if the position is already above target). An
    "exit" is a symbol whose target went to 0: its position is closed entirely. A "hold" (target
    positive on both decisions) is left untouched, so positions drift with prices between decisions.
    The book starts flat, so at `start` every positive target is an add.
    rebalance: kept for backward compatibility; trading is driven by target changes, so the flag no
    longer affects behavior.
    A holding whose bars end (delisted) is liquidated at the last available close (normal cost) on
    the first session with no bars, instead of being carried at a frozen price.
    Accounting starts flat with equity 1.0 just before the open of `start`; the order from the
    previous session's decision (close before `start`) is filled at `start`'s open. Sells execute
    before buys; buys are scaled to available cash. 0.1%/side cost on traded notional, no interest
    on cash.

    Fidelity limits (honest): fills are modeled at the next session's open (live uses evening
    extended-hours limit orders plus a morning fill-check); no partial fills, no rejected orders,
    no corporate-action handling; fractional shares (live trades whole shares); a symbol untradable
    on its add day (halt) is not bought later - its add day has passed.
    """
    dates = open_w.index
    s0 = dates.searchsorted(pd.Timestamp(start))
    s1 = len(dates) if end is None else dates.searchsorted(pd.Timestamp(end), side="right")
    cols = list(open_w.columns)
    O = open_w.to_numpy(float)
    C = close_w.reindex(columns=cols).ffill().to_numpy(float)
    W = target.reindex(index=dates, columns=cols).fillna(0.0).to_numpy(float)
    N = len(cols)
    # B-4: last session with a real open per symbol; a held symbol past that point is delisted.
    last_open = np.where(~np.isnan(O), np.arange(len(dates))[:, None], -1).max(axis=0)
    shares = np.zeros(N)
    basis = np.zeros(N)            # cost basis per share incl. buy cost
    entry_day = np.full(N, -1)
    cash = 1.0
    eq, expo, turn = [], [], []
    trades = []
    for t in range(s0, s1):
        o = O[t]
        tradable = ~np.isnan(o)
        # B-4: delisted while held -> liquidate at the last available close (normal cost), so no
        # phantom position is carried at a frozen price. Only fires when no future bars exist at
        # all (a mere halt has later bars, so t <= last_open there and the position is kept).
        for j in np.where((shares > 0) & ~tradable & (last_open >= 0) & (t > last_open))[0]:
            px = C[t, j]
            if not np.isfinite(px):
                continue
            q = shares[j]
            px_net = px * (1 - cost)
            cash += q * px_net
            entry = dates[entry_day[j]] if entry_day[j] >= 0 else dates[t]
            trades.append((cols[j], entry, dates[t], basis[j], px_net,
                           px_net / basis[j] - 1 if basis[j] > 0 else np.nan, "delist"))
            shares[j] = 0.0
            basis[j] = 0.0
            entry_day[j] = -1
        px_open = np.where(np.isnan(o), C[t - 1] if t > 0 else np.nan, o)
        V = cash + np.nansum(shares * np.nan_to_num(px_open))
        w = W[t - 1] if t > 0 else np.zeros(N)          # decision executed at today's open
        wp = W[t - 2] if t > s0 and t >= 2 else np.zeros(N)  # previous decision's target (flat at start)
        # B-1 (live rule): trade only adds and exits, never resize holds.
        desired = shares.copy()
        is_exit = tradable & (shares > 0) & (w <= 0)
        is_add = tradable & (w > 0) & (wp <= 0)
        desired[is_exit] = 0.0
        desired[is_add] = w[is_add] * V / o[is_add]     # adds trade to exactly target (buy or trim)
        delta = desired - shares
        delta[np.abs(delta * np.nan_to_num(px_open)) < 1e-10] = 0.0
        traded_notional = 0.0
        # sells first
        for j in np.where(delta < 0)[0]:
            q = -delta[j]
            exit_px = o[j] * (1 - cost)
            cash += q * exit_px
            traded_notional += q * o[j]
            entry = dates[entry_day[j]] if entry_day[j] >= 0 else dates[t]
            ret = exit_px / basis[j] - 1 if basis[j] > 0 else np.nan
            if desired[j] == 0:
                trades.append((cols[j], entry, dates[t], basis[j], exit_px, ret, "close"))
                basis[j] = 0.0
                entry_day[j] = -1
            else:
                trades.append((cols[j], entry, dates[t], basis[j], exit_px, ret, "trim"))  # B-5: partial sells
            shares[j] = desired[j]
        buys = np.where(delta > 0)[0]
        need = float(np.sum(delta[buys] * o[buys] * (1 + cost))) if len(buys) else 0.0
        scale = min(1.0, max(cash, 0.0) / need) if need > 0 else 1.0
        for j in buys:
            q = delta[j] * scale
            if q <= 0:
                continue
            cash -= q * o[j] * (1 + cost)
            traded_notional += q * o[j]
            new_sh = shares[j] + q
            basis[j] = (basis[j] * shares[j] + q * o[j] * (1 + cost)) / new_sh
            if shares[j] == 0:
                entry_day[j] = t
            shares[j] = new_sh
        pos_val = np.nansum(shares * C[t])
        e = cash + pos_val
        eq.append(e)
        expo.append(pos_val / e if e > 0 else 0.0)
        turn.append(traded_notional / V if V > 0 else 0.0)
    idx = dates[s0:s1]
    trades = pd.DataFrame(trades, columns=["Symbol", "Entry", "Exit", "EntryPx", "ExitPx", "Return", "Kind"])
    open_pos = pd.DataFrame({"Symbol": [cols[j] for j in np.where(shares > 0)[0]],
                             "Entry": [dates[entry_day[j]] for j in np.where(shares > 0)[0]]})
    return {"equity": pd.Series(eq, idx), "exposure": pd.Series(expo, idx), "turnover": pd.Series(turn, idx),
            "trades": trades, "open_positions": open_pos}


def metrics(res, name=None):
    """Summary statistics of one simulation (return, CAGR, Sharpe, drawdown, turnover, trade stats)."""
    eq, ex = res["equity"], res["exposure"]
    r = eq.pct_change()
    r.iloc[0] = eq.iloc[0] - 1
    n = len(eq)
    years = n / 252
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else np.nan
    vol = r.std()
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    dd = (eq / eq.cummax().clip(lower=1.0) - 1).min()
    closed = res["trades"]
    opened = len(closed) + len(res["open_positions"])
    out = {
        "Strategy": name,
        "Start": eq.index[0].date(), "End": eq.index[-1].date(), "Days": n,
        "Total Return %": total * 100,
        "CAGR %": cagr * 100,
        "Sharpe": r.mean() / vol * np.sqrt(252) if vol > 0 else np.nan,
        "Sortino": r.mean() / downside * np.sqrt(252) if downside > 0 else np.nan,
        "Max DD %": dd * 100,
        "Calmar": cagr / abs(dd) if dd < 0 else np.nan,
        "Exposure %": ex.mean() * 100,
        "Time in Market %": (ex > 0.001).mean() * 100,
        "Return per Invested Day (bp)": (r.mean() / ex.mean() * 1e4) if ex.mean() > 0 else np.nan,
        "Turnover x/yr": res["turnover"].sum() / years,
        "Trades": opened,
        "Closed Trades": len(closed),
        "Win Rate % (closed, net)": (closed["Return"] > 0).mean() * 100 if len(closed) else np.nan,
        "Avg Trade % (closed, net)": closed["Return"].mean() * 100 if len(closed) else np.nan,
    }
    return out


# ----------------------------------------------------------------------------- ranking and selection
def _name_positions(cols):
    """Alphabetical position of each column name (final, deterministic tie-break)."""
    pos = np.empty(len(cols), int)
    pos[np.argsort(np.array([str(c) for c in cols]))] = np.arange(len(cols))
    return pos


def ranking_order(idx, scores, tiebreak, name_pos, decimals=6):
    """Indices idx sorted best-first: score (rounded to `decimals`) desc, then tiebreak desc, then name A-Z.

    Rounding removes float noise (0.5*a + 0.5*b can differ in the last bit for mathematically equal scores),
    so exact ties are broken by a documented rule instead of by floating-point accident or sort instability.
    """
    idx = np.asarray(idx, int)
    if len(idx) == 0:
        return idx
    s = np.round(scores[idx], decimals)
    tb = np.zeros(len(idx)) if tiebreak is None else np.nan_to_num(np.round(tiebreak[idx], decimals), nan=-np.inf)
    return idx[np.lexsort((name_pos[idx], -tb, -s))]


def deterministic_rank(score_w, eligible_w=None, tiebreak_w=None):
    """Per-date rank 1..N (1 = best) over eligible symbols with a score; NaN otherwise. Unique ranks, same order
    as rank_targets (score desc at 6 decimals, then tiebreak desc, then name A-Z). Computed date by date."""
    s = score_w.where(eligible_w.reindex_like(score_w).astype("boolean").fillna(False)) if eligible_w is not None else score_w
    S = s.to_numpy(float)
    TB = tiebreak_w.reindex_like(score_w).to_numpy(float) if tiebreak_w is not None else None
    name_pos = _name_positions(list(score_w.columns))
    out = np.full(S.shape, np.nan)
    for t in range(len(S)):
        o = ranking_order(np.flatnonzero(~np.isnan(S[t])), S[t], None if TB is None else TB[t], name_pos)
        out[t, o] = np.arange(1, len(o) + 1)
    return pd.DataFrame(out, index=score_w.index, columns=score_w.columns)


def rank_targets(score_w, eligible_w, vol_w, n=10, regime=None, rebalance_days=None, min_score=0.0,
                 sector_cap=0.4, vol_sizing=True, buffer_rank=None, regime_scale=None, decision_log=None,
                 tiebreak_w=None, max_pick_rank=None, cap_soft=False, buy_block=None, held_w=None, start_holdings=None):
    """Weekly top-N by score with sector cap and inverse-volatility weights.

    On rebalance days: candidates = eligible & score > min_score, best first, at most floor(sector_cap*n)
    per sector. If the regime is off, no NEW names are bought (held names that still qualify are kept).
    Total exposure = (#selected / n); within that, weights ~ 1/vol (63-day). Between rebalances the
    target is carried forward unchanged.
    buffer_rank: keep a current holding while its rank among qualifying names is <= buffer_rank
                 (cuts turnover); free slots are then filled best-first.
    regime_scale: if given, a regime-off rebalance still selects normally but scales weights by this
                  factor (soft regime) instead of blocking new names.
    decision_log: optional list; one dict per symbol per rebalance with status and reason.
    tiebreak_w: optional frame used to order equal scores (higher first); remaining ties go alphabetically.
                Scores are compared at 6 decimals so float noise never decides the order (see ranking_order).
    max_pick_rank: only names ranked 1..max_pick_rank may be picked (T20: 20); fewer than n candidates -> the rest is cash.
    cap_soft: after the capped walk, fill any free slots from the unused candidates in rank order ignoring the sector cap.
    buy_block: optional frame of days until earnings (earnings_days_ahead; NaN = no block): such a stock is not bought unless
               already held; its slot goes to the next candidate. held_w: holdings that count as "already held" (default:
               this function's own carried holdings). start_holdings: holdings before the first date (default: none).
    """
    dates, cols = score_w.index, list(score_w.columns)
    S = score_w.to_numpy(float)
    E = eligible_w.reindex_like(score_w).astype("boolean").fillna(False).to_numpy(bool)
    Vv = vol_w.reindex_like(score_w).to_numpy(float)
    TB = tiebreak_w.reindex_like(score_w).to_numpy(float) if tiebreak_w is not None else None
    name_pos = _name_positions(cols)
    R = (regime.reindex(dates).astype("boolean").fillna(False).to_numpy(bool) if regime is not None else np.ones(len(dates), bool))
    reb = rebalance_days.reindex(dates).astype("boolean").fillna(False).to_numpy(bool)
    sectors = np.array([sector_mapping.symbol_sector.get(c, "Other") for c in cols])
    max_per_sector = max(1, int(np.floor(sector_cap * n)))
    BB = buy_block.reindex(index=dates, columns=cols).to_numpy(float) if buy_block is not None else None
    H = held_w.reindex(index=dates, columns=cols).fillna(0.0).to_numpy(float) if held_w is not None else None
    out = np.zeros((len(dates), len(cols)))
    current = np.zeros(len(cols)) if start_holdings is None else np.asarray(start_holdings, float).copy()
    for t in range(len(dates)):
        if reb[t]:
            blocked = set()
            if BB is not None:                 # earnings rule: not held and earnings within the window -> not bought
                held_now = H[t] > 0 if H is not None else current > 0
                blocked = set(np.where(~np.isnan(BB[t]) & ~held_now)[0])
            base_ok = E[t] & ~np.isnan(S[t]) & ~np.isnan(Vv[t]) & (Vv[t] > 0)
            ok = base_ok & (S[t] > min_score)
            soft = regime_scale is not None
            if not R[t] and not soft:
                ok &= current > 0
            order = ranking_order(np.where(ok)[0], S[t], None if TB is None else TB[t], name_pos)
            rank_of = {j: r + 1 for r, j in enumerate(order)}
            picked, per_sector, reason = [], {}, {}
            if buffer_rank is not None:  # keep holdings still within the buffer first
                for j in order:
                    if current[j] > 0 and rank_of[j] <= buffer_rank and len(picked) < n \
                            and per_sector.get(sectors[j], 0) < max_per_sector:
                        picked.append(j)
                        per_sector[sectors[j]] = per_sector.get(sectors[j], 0) + 1
                        reason[j] = f"kept (rank {rank_of[j]} <= buffer {buffer_rank})"
            cands = order if max_pick_rank is None else order[:max_pick_rank]
            for j in cands:
                if len(picked) >= n:
                    break
                if j in reason:
                    continue
                if j in blocked:
                    reason[j] = f"{earnings_note(BB[t, j], dates[t])}: not bought"
                    continue
                if per_sector.get(sectors[j], 0) >= max_per_sector:
                    reason[j] = f"skipped: sector cap ({sectors[j]} already {max_per_sector})"
                    continue
                picked.append(j)
                per_sector[sectors[j]] = per_sector.get(sectors[j], 0) + 1
                reason[j] = f"selected (rank {rank_of[j]})"
            if cap_soft:                   # free slots left: unused candidates in rank order, sector cap ignored
                for j in cands:
                    if len(picked) >= n:
                        break
                    if j in picked or j in blocked:
                        continue
                    picked.append(j)
                    per_sector[sectors[j]] = per_sector.get(sectors[j], 0) + 1
                    reason[j] = (f"selected (rank {rank_of[j]}; sector cap relaxed: fewer than {n} fit the cap within the top "
                                 f"{max_pick_rank or len(cands)})")
            if max_pick_rank is not None:
                for j in order[max_pick_rank:]:
                    if j not in reason and current[j] > 0:
                        reason[j] = f"rank {rank_of[j]} worse than {max_pick_rank} (picks only from ranks 1-{max_pick_rank})"
            new = np.zeros(len(cols))
            if picked:
                inv = 1 / Vv[t, picked] if vol_sizing else np.ones(len(picked))
                new[picked] = inv / inv.sum() * (len(picked) / n)
                if soft and not R[t]:
                    new *= regime_scale
            if decision_log is not None:
                for j in range(len(cols)):
                    if new[j] == 0 and current[j] == 0 and j not in reason:
                        continue
                    if new[j] > 0:
                        status = "hold" if current[j] > 0 else "add"
                        why = reason.get(j, "selected")
                    else:
                        status = "drop" if current[j] > 0 else "not selected"
                        if not base_ok[j]:
                            why = "not eligible / no data"
                        elif not (S[t, j] > min_score):
                            why = f"score {S[t, j]:.1f} <= {min_score:g}"
                        elif not R[t] and not soft and current[j] == 0:
                            why = "regime off: no new names"
                        elif j in reason:
                            why = reason[j]
                        else:
                            why = f"rank {rank_of.get(j, '-')} outside top {n}" + (f" / buffer {buffer_rank}" if buffer_rank else "")
                    decision_log.append({"Date": dates[t], "Symbol": cols[j], "Status": status, "Reason": why,
                                         "Rank": rank_of.get(j, np.nan), "Score": S[t, j], "Sector": sectors[j],
                                         "Old_Weight": current[j], "New_Weight": new[j]})
            current = new
        out[t] = current
    return pd.DataFrame(out, index=dates, columns=cols)


def weekly_rebalance_days(dates, live=False):
    """Last trading day of each ISO week (decision at that close, fill next open).

    live=True: the current (possibly unfinished) week only counts as rebalanced if its latest
    session is a Friday, so a Thursday run does not pretend the week is over.
    """
    d = pd.Series(dates, index=dates)
    wk = d.dt.isocalendar()
    key = wk["year"].astype(str) + "-" + wk["week"].astype(str)
    out = d.groupby(key.values).transform("max").eq(d)
    if live and len(d):
        last = d.iloc[-1]
        nxt = next_sessions(last, 1)[0]
        if nxt.isocalendar()[:2] == last.isocalendar()[:2]:  # another session left this week -> not yet
            out.iloc[-1] = False
    return out


WEEKDAY_CODES = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}


def midweek_check_days(dates, days=("Mon", "Wed"), rebalance_days=None):
    """Sessions with a mid-week swap check (decision at that close, fill next open).

    Each calendar Mon / Wed (``days``) maps to the first session on or after it, so a Monday holiday moves the check to
    Tuesday. Sessions that are also a weekly rebalance day are excluded (the full rebalance wins). Same mapping as
    the tested variant D (Reports/rebalance_frequency_test.csv; reproduced by tests/test_midweek_repro.py).
    """
    dates = pd.DatetimeIndex(dates)
    if rebalance_days is None:
        rebalance_days = weekly_rebalance_days(dates, live=True)
    if not len(dates):
        return pd.Series(False, index=dates)
    cal = pd.date_range(dates[0], dates[-1], freq="D")
    cal = cal[cal.dayofweek.isin([WEEKDAY_CODES[d] for d in days])]
    mapped = {dates[p] for p in dates.searchsorted(cal) if p < len(dates)}
    weekly = rebalance_days.reindex(dates).astype("boolean").fillna(False).to_numpy(bool)
    return pd.Series(dates.isin(list(mapped)) & ~weekly, index=dates)


def midweek_swap_pairs(cur, order, rank, sectors, enter_top=3, exit_below=15, cap=4, skip=None):
    """The mid-week swap rule on one check day (used by apply_midweek_swaps and by holdings_alert for real positions).

    cur: weight array (modified in place: each entrant takes the sold holding's weight); order: qualifying column indices,
    best first; rank: {index: rank}; sectors: sector per column. While a held name ranks worse than exit_below (or has no
    rank) and a NOT-held name is in order[:enter_top], take the best entrant and the worst-ranked holding whose removal
    leaves the entrant's sector below cap (ties among unranked holdings: column order). skip: top-N names that may not be
    bought (earnings rule); they are passed over, not replaced by rank N+1. Returns [(entrant, sold, weight)].
    """
    swaps = []
    while True:
        held = np.where(cur > 0)[0]
        weak = sorted([j for j in held if rank.get(j, 1e6) > exit_below], key=lambda j: -rank.get(j, 1e6))
        if not weak:
            break
        entrants = [j for j in order[:enter_top] if cur[j] == 0 and not (skip and j in skip)]
        done = False
        for e in entrants:
            for h in weak:
                in_sector = sum(1 for k in np.where(cur > 0)[0] if k != h and sectors[k] == sectors[e])
                if in_sector < cap:
                    swaps.append((e, h, cur[h]))
                    cur[e], cur[h] = cur[h], 0.0
                    done = True
                    break
            if done:
                break
        if not done:
            break
    return swaps


def midweek_exit_sells(cur, rank, exit_all_below):
    """The mid-week exit rule on one check day, applied AFTER midweek_swap_pairs (used by apply_midweek_swaps and by
    holdings_alert). Every holding ranked worse than exit_all_below (or with no rank) is sold; the cash stays idle until the
    next weekly rebalance. cur is modified in place. Returns [(sold, weight)] in column order."""
    if not exit_all_below:
        return []
    sells = [(j, cur[j]) for j in np.where(cur > 0)[0] if rank.get(j, 1e6) > exit_all_below]
    for j, _ in sells:
        cur[j] = 0.0
    return sells


def apply_midweek_swaps(base, score_w, eligible_w, vol_w, rebalance_days, check_days, enter_top=3, exit_below=15,
                        sector_cap=0.4, n=10, min_score=0.0, tiebreak_w=None, decision_log=None, check_log=None,
                        exit_all_below=None, cap_soft=False, buy_block=None, reselect=None):
    """Weekly targets (``base`` from rank_targets) + mid-week swaps on ``check_days``.

    On a check day: rank = position among qualifying names (eligible, score > min_score, valid vol), same order as
    rank_targets (score at 6 decimals, then tiebreak, then name). While some held name ranks worse than ``exit_below``
    (or no longer qualifies) and some NOT-held name ranks in the top ``enter_top``: take the best entrant and the
    worst-ranked held name whose removal leaves room in the entrant's sector (max floor(sector_cap*n)); the entrant takes
    the held name's weight. Repeated until no pair qualifies. Weekly rebalance days reset to ``base``.
    Ties: several held names that no longer qualify (no rank) are taken in column order (= sector_mapping.tradable_symbols),
    exactly as in the tested variant D.
    exit_all_below (mid-week exit, e.g. 30): after the swaps, every holding ranked worse than this (or unranked) is sold and
    its weight stays in cash until the next weekly rebalance (variant S3 of Reports/sell_rule_test.csv).
    cap_soft (T20): the sector cap is ignored at mid-week swaps - a non-held top-3 stock always replaces the worst-ranked
    holding below exit_below.
    buy_block (earnings rule): days-until-earnings frame; a top-N candidate with earnings in the window is skipped.
    reselect(t, held) -> (weights, log rows): redo the weekly selection with the REAL holdings (needed when the earnings
    rule is on, because "already held" then matters); its rows replace rank_targets' rows for that date.
    decision_log: rows (like rank_targets) for check days WITH a swap or exit (hold / add / drop).
    check_log: one dict per check day and swap (Action 'SWAP') and per exit (Action 'SELL'), or one 'NO SWAP' row with a Note.
    """
    dates, cols = base.index, list(base.columns)
    S = score_w.reindex(index=dates, columns=cols).to_numpy(float)
    E = eligible_w.reindex(index=dates, columns=cols).astype("boolean").fillna(False).to_numpy(bool)
    Vv = vol_w.reindex(index=dates, columns=cols).to_numpy(float)
    TB = tiebreak_w.reindex(index=dates, columns=cols).to_numpy(float) if tiebreak_w is not None else None
    B = base.to_numpy(float)
    reb = rebalance_days.reindex(dates).astype("boolean").fillna(False).to_numpy(bool)
    chk = check_days.reindex(dates).astype("boolean").fillna(False).to_numpy(bool)
    name_pos = _name_positions(cols)
    sectors = np.array([sector_mapping.symbol_sector.get(c, "Other") for c in cols])
    cap = 10 ** 6 if cap_soft else max(1, int(np.floor(sector_cap * n)))
    BB = buy_block.reindex(index=dates, columns=cols).to_numpy(float) if buy_block is not None else None
    out = np.zeros_like(B)
    cur = np.zeros(len(cols))
    replaced = {}                                 # date -> re-selected weekly log rows (earnings rule)
    by_date = {}
    if decision_log is not None:                  # weekly rows written by rank_targets, to re-base them on the swapped holdings
        for k, row in enumerate(decision_log):
            by_date.setdefault(row["Date"], []).append(k)
    for t in range(len(dates)):
        if reb[t] and reselect is not None:
            cur, rows = reselect(t, cur.copy())
            replaced[dates[t]] = rows
        elif reb[t]:
            if decision_log is not None and t > 0 and not np.array_equal(cur, B[t - 1]):
                _rebase_weekly_log(decision_log, dates[t], by_date.get(dates[t], []), cur, B[t], cols, sectors, S[t], E[t], Vv[t],
                                   None if TB is None else TB[t], name_pos, min_score, n)
            cur = B[t].copy()
        elif chk[t] and cur.sum() > 0:
            ok = E[t] & ~np.isnan(S[t]) & ~np.isnan(Vv[t]) & (Vv[t] > 0) & (S[t] > min_score)
            order = ranking_order(np.where(ok)[0], S[t], None if TB is None else TB[t], name_pos)
            rank = {j: r + 1 for r, j in enumerate(order)}
            before = cur.copy()
            skip = {j for j in order[:enter_top] if cur[j] == 0 and not np.isnan(BB[t, j])} if BB is not None else set()
            swaps = midweek_swap_pairs(cur, order, rank, sectors, enter_top, exit_below, cap, skip=skip)
            exits = midweek_exit_sells(cur, rank, exit_all_below)
            if check_log is not None:
                if swaps or exits:
                    for e, h, w in swaps:
                        check_log.append({"Date": dates[t], "Action": "SWAP", "Sell": cols[h], "Sell_Rank": rank.get(h, np.nan),
                                          "Sell_Score": S[t, h], "Buy": cols[e], "Buy_Rank": rank[e], "Buy_Score": S[t, e],
                                          "Weight": w, "Sell_Sector": sectors[h], "Buy_Sector": sectors[e], "Note": ""})
                    for h, w in exits:
                        check_log.append({"Date": dates[t], "Action": "SELL", "Sell": cols[h], "Sell_Rank": rank.get(h, np.nan),
                                          "Sell_Score": S[t, h], "Buy": "", "Buy_Rank": np.nan, "Buy_Score": np.nan,
                                          "Weight": w, "Sell_Sector": sectors[h], "Buy_Sector": "",
                                          "Note": f"worse than rank {exit_all_below}: sold, cash until the weekly rebalance"})
                else:
                    held = np.where(before > 0)[0]
                    worst = max((rank.get(j, 1e6) for j in held), default=np.nan)
                    entrants = [cols[j] for j in order[:enter_top] if before[j] == 0 and j not in skip]
                    if not entrants:
                        note = (f"no new top-{enter_top} stock can be bought" if skip
                                else f"all top-{enter_top} stocks are already held")
                    elif worst <= exit_below:
                        note = (f"no held stock is below rank {exit_below} (worst held rank {int(worst)}; "
                                f"new top-{enter_top}: {', '.join(entrants)})")
                    else:
                        weak_txt = ", ".join(f"{cols[j]} {'rank ' + str(rank[j]) if j in rank else 'no longer qualifies'}"
                                             for j in sorted(held, key=lambda k: rank.get(k, 1e6)) if rank.get(j, 1e6) > exit_below)
                        ent_txt = ", ".join(f"{cols[j]} rank {rank[j]} {sectors[j]}" for j in order[:enter_top] if before[j] == 0)
                        note = (f"max {cap} per sector blocks it: new top-{enter_top} {ent_txt} - that sector already has {cap} "
                                f"holdings and the weak holdings below rank {exit_below} ({weak_txt}) are in other sectors")
                    if skip:
                        note += "; " + ", ".join(f"{cols[j]} (rank {rank[j]}) not bought: {earnings_note(BB[t, j], dates[t])}"
                                                 for j in order[:enter_top] if j in skip)
                    if exit_all_below:
                        note += f"; no holding is worse than rank {exit_all_below}"
                    check_log.append({"Date": dates[t], "Action": "NO SWAP", "Sell": "", "Sell_Rank": np.nan,
                                      "Sell_Score": np.nan, "Buy": "", "Buy_Rank": np.nan, "Buy_Score": np.nan,
                                      "Weight": np.nan, "Sell_Sector": "", "Buy_Sector": "", "Note": note})
            if (swaps or exits) and decision_log is not None:
                partner = {h: e for e, h, _ in swaps}
                partner.update({e: h for e, h, _ in swaps})
                exited = {h for h, _ in exits}
                for j in np.where((before > 0) | (cur > 0))[0]:
                    r = rank.get(j, np.nan)
                    rtxt = f"rank {r}" if r == r else "no longer qualifies"
                    p = partner.get(j)
                    if cur[j] > 0 and before[j] == 0:
                        status = "add"
                        why = (f"mid-week swap in: {rtxt} is in the top {enter_top}; replaces {cols[p]} "
                               f"({'rank ' + str(rank[p]) if p in rank else 'no longer qualifies'})")
                    elif cur[j] == 0 and j in exited:
                        status = "drop"
                        why = (f"mid-week exit: {rtxt} (worse than {exit_all_below}); sold, cash until the weekly rebalance"
                               if r == r else "mid-week exit: no longer qualifies (score <= 0 or no data); sold, cash until "
                               "the weekly rebalance")
                    elif cur[j] == 0:
                        status = "drop"
                        why = f"mid-week swap out: {rtxt} (below {exit_below}); replaced by {cols[p]} (rank {rank[p]})"
                    else:
                        status, why = "hold", f"kept at mid-week check ({rtxt})"
                    decision_log.append({"Date": dates[t], "Symbol": cols[j], "Status": status, "Reason": why,
                                         "Rank": r, "Score": S[t, j], "Sector": sectors[j],
                                         "Old_Weight": before[j], "New_Weight": cur[j]})
        out[t] = cur
    if replaced and decision_log is not None:     # swap in the re-selected weekly rows, keeping the log's order
        new_log, done = [], set()                 # (rebalance days never have mid-week rows)
        for row in decision_log:
            d = row["Date"]
            if d in replaced:
                if d not in done:
                    new_log.extend(replaced[d])
                    done.add(d)
                continue
            new_log.append(row)
        new_log += [r for d, rows in replaced.items() if d not in done for r in rows]
        decision_log[:] = new_log
    return pd.DataFrame(out, index=dates, columns=cols)


def _rebase_weekly_log(decision_log, date, idx, held_before, new, cols, sectors, s_t, e_t, v_t, tb_t, name_pos, min_score, n):
    """After mid-week swaps the holdings going into a weekly rebalance differ from rank_targets' own carry-forward: fix that
    day's rows (Old_Weight, add/hold/drop) so the log compares with what is really held; add rows for held names it skipped."""
    ok = e_t & ~np.isnan(s_t) & ~np.isnan(v_t) & (v_t > 0) & (s_t > min_score)
    rank = {j: r + 1 for r, j in enumerate(ranking_order(np.where(ok)[0], s_t, tb_t, name_pos))}
    pos = {c: j for j, c in enumerate(cols)}
    seen = set()
    for k in idx:
        row = decision_log[k]
        j = pos[row["Symbol"]]
        seen.add(j)
        old_w = held_before[j]
        if row["Old_Weight"] == old_w:
            continue
        row["Old_Weight"] = old_w
        if row["New_Weight"] > 0:
            row["Status"] = "hold" if old_w > 0 else "add"
        elif old_w > 0:
            row["Status"] = "drop"
            if not str(row["Reason"]).startswith(("score", "not eligible", "skipped", "rank")):
                row["Reason"] = f"rank {rank.get(j, '-')} outside top {n}"
        else:
            row["Status"] = "not selected"
    for j in np.where(held_before > 0)[0]:
        if j in seen:
            continue
        if not (e_t[j] and not np.isnan(s_t[j])):
            why = "not eligible / no data"
        elif not s_t[j] > min_score:
            why = f"score {s_t[j]:.1f} <= {min_score:g}"
        else:
            why = f"rank {rank.get(j, '-')} outside top {n}"
        decision_log.append({"Date": date, "Symbol": cols[j], "Status": "drop",
                             "Reason": why, "Rank": rank.get(j, np.nan), "Score": s_t[j], "Sector": sectors[j],
                             "Old_Weight": held_before[j], "New_Weight": new[j]})


def winner_targets(score_w, eligible_w, vol_w, regime, rebalance_days, tiebreak_w=None, decision_log=None,
                   check_log=None, midweek=None, exit_all_below="winner", selection=None, earnings_block_days="winner",
                   earnings=None):
    """Live WINNER targets: weekly rank_targets + (if WINNER['midweek_swap']) mid-week swaps + (if
    WINNER['midweek_exit_below']) mid-week exits to cash.

    Returns (targets, check_days). check_days is all-False when the mid-week swap is off. ``midweek`` overrides
    WINNER['midweek_swap'] (pass False to force plain weekly); ``exit_all_below`` overrides WINNER['midweek_exit_below']
    (None = no mid-week exit); ``selection`` = dict(max_pick_rank=..., cap_soft=...) overrides the T20 keys;
    ``earnings_block_days`` overrides WINNER['earnings_block_days'] (None = no earnings rule); ``earnings`` = earnings dates
    (default: load_earnings(), i.e. Reports/earnings_date.csv)."""
    mw = WINNER.get("midweek_swap") if midweek is None else midweek
    if exit_all_below == "winner":
        exit_all_below = WINNER.get("midweek_exit_below")
    if earnings_block_days == "winner":
        earnings_block_days = WINNER.get("earnings_block_days")
    args = winner_rank_args(regime)
    args.update(selection or {})
    block = None
    if earnings_block_days:
        block = earnings_days_ahead(score_w.index, list(score_w.columns),
                                    load_earnings() if earnings is None else earnings, earnings_block_days)
    base = rank_targets(score_w, eligible_w, vol_w, rebalance_days=rebalance_days, decision_log=decision_log,
                        tiebreak_w=tiebreak_w, buy_block=block, **args)
    if not mw:
        return base, pd.Series(False, index=base.index)

    def reselect(t, held):
        """Friday selection with the REAL holdings (after mid-week changes): needed for the earnings rule."""
        rows = [] if decision_log is not None else None
        day = score_w.index[[t]]
        one = rank_targets(score_w.iloc[[t]], eligible_w, vol_w, rebalance_days=pd.Series(True, index=day),
                           decision_log=rows, tiebreak_w=tiebreak_w, buy_block=block.iloc[[t]], start_holdings=held, **args)
        return one.iloc[0].to_numpy(float), rows or []
    checks = midweek_check_days(base.index, mw.get("days", ("Mon", "Wed")), rebalance_days)
    tgt = apply_midweek_swaps(base, score_w, eligible_w, vol_w, rebalance_days, checks, mw["enter_top"], mw["exit_below"],
                              WINNER["sector_cap"], WINNER["n"], WINNER["min_score"], tiebreak_w=tiebreak_w,
                              decision_log=decision_log, check_log=check_log, exit_all_below=exit_all_below,
                              cap_soft=args["cap_soft"], buy_block=block,
                              reselect=reselect if block is not None else None)
    return tgt, checks


def next_decision(latest, days=None):
    """Next decision session after ``latest``: (date, 'full rebalance' | 'mid-week check', fill_date).

    Weekly rebalance = last session of an ISO week; mid-week check = first session on/after each Mon / Wed that is not
    the week's last session (same rules as weekly_rebalance_days / midweek_check_days)."""
    mw = WINNER.get("midweek_swap")
    if days is None:
        days = mw.get("days", ("Mon", "Wed")) if mw else ()
    codes = {WEEKDAY_CODES[d] for d in days}
    latest = pd.Timestamp(latest).normalize()
    sess = [latest] + list(next_sessions(latest, 12))
    for i in range(1, len(sess) - 1):
        d, prev = sess[i], sess[i - 1]
        if d.isocalendar()[:2] != sess[i + 1].isocalendar()[:2]:
            return d, "full rebalance", sess[i + 1]
        if any(c.dayofweek in codes for c in pd.date_range(prev + pd.Timedelta(days=1), d, freq="D")):
            return d, "mid-week check", sess[i + 1]
    return sess[1], "full rebalance", sess[2]


# ----------------------------------------------------------------------------- live helpers
def holding_details(weights, open_w, close_w, atr_w, vol_w, k=3.0):
    """Per current holding: decision/fill dates, entry price (next open after the decision), days held,
    P&L since entry, 63d annualized volatility, ATR and an ATR trailing-stop level."""
    last = weights.index[-1]
    rows = []
    for sym in weights.columns[weights.loc[last] > 0]:
        w = weights[sym]
        off = w[w <= 0]
        run_start = w.index[w.index > off.index[-1]][0] if len(off) else w.index[0]
        fills = open_w.index[open_w.index > run_start]
        fill = fills[0] if len(fills) else None
        entry = float(open_w.at[fill, sym]) if fill is not None and pd.notna(open_w.at[fill, sym]) else np.nan
        close = float(close_w.at[last, sym])
        since = close_w.loc[fill:, sym] if fill is not None else pd.Series(dtype=float)
        peak = since.max() if len(since) else np.nan
        atr = float(atr_w.at[last, sym]) if sym in atr_w else np.nan
        rows.append({"Symbol": sym, "Weight": float(w.loc[last]), "Decision_Date": run_start.date(),
                     "Entry_Date": fill.date() if fill is not None else None, "Entry_Price": entry,
                     "Close": close, "PnL_%": (close / entry - 1) * 100 if entry == entry else np.nan,
                     "Days_Held": int(len(since)) if len(since) else 0,
                     "Vol_63d_%": float(vol_w.at[last, sym]) * np.sqrt(252) * 100 if sym in vol_w else np.nan,
                     "ATR": atr, "ATR_Stop": peak - k * atr if peak == peak else np.nan,
                     "Dist_to_Stop_%": (close / (peak - k * atr) - 1) * 100 if peak == peak else np.nan})
    return pd.DataFrame(rows)


def forward_tracking(target, open_w, close_w, first_decision, benchmarks=("QQQ", "SPY"), rebalance=None):
    """Equity of the live target vs benchmarks, starting with the fill after `first_decision`."""
    start_candidates = open_w.index[open_w.index > pd.Timestamp(first_decision)]
    if not len(start_candidates):
        return pd.DataFrame()
    start = start_candidates[0]
    tgt = target.reindex(index=close_w.index, columns=close_w.columns).fillna(0.0)
    res = simulate(open_w, close_w, tgt, start, rebalance=rebalance)
    out = pd.DataFrame({"Strategy": res["equity"], "Strategy_Exposure": res["exposure"]})
    for b in benchmarks:
        t = pd.DataFrame(0.0, index=close_w.index, columns=close_w.columns)
        t[b] = 1.0
        out[b] = simulate(open_w, close_w, t, start)["equity"]
    return out.rename_axis("Date").reset_index()


# ----------------------------------------------------------------------------- backtest of the live rules (backtest.ipynb)
WALK_FORWARD_START = "2022-04-01"                              # first trading day of the backtest
NEVER_SEEN_END = "2024-09-16"                                  # 2022-04 -> 2024-09 was never used to choose the rules


def backtest_inputs(refresh=False):
    """Prices, scores, eligibility, volatility, market filter and decision calendar for the live-rules backtest, from
    Reports/cache/bars_daily_long.pkl (refresh=True downloads the bars again: Alpaca market data, no quota; the pinned test
    numbers in tests/ assume the cached bars). Same set-up as tests/backtest_setup.py."""
    bars, _ = load_bars(refresh=refresh, cache_name=LONG_CACHE, start=LONG_START)
    U = TRADABLE
    tech = build_technical(bars, symbols=U)
    close, opn = wide(bars, "Close"), wide(bars, "Open")
    idx = close.index
    tech_score = wide(tech, "Technical_Score").reindex(index=idx, columns=U)
    rs, _ = relative_strength(close, U)
    return {"close": close, "open": opn, "score": 0.5 * tech_score + 0.5 * rs, "tiebreak": rs,
            "eligible": bool_wide(tech, "eligible", idx, U),
            "vol": close[U].pct_change(fill_method=None).rolling(63).std(),
            "regime": regime_series(close, "QQQ"), "weekly": weekly_rebalance_days(idx, live=True)}


def run_rules(inp, start=WALK_FORWARD_START, **overrides):
    """Backtest WINNER (optionally with some keys changed, e.g. run_rules(inp, midweek_exit_below=None)) from `start`.
    Returns {"res": simulate() output, "targets", "checks": mid-week check log, "decisions": decision log}."""
    saved = dict(WINNER)
    try:
        WINNER.update(overrides)
        checks, decisions = [], []
        tgt, _ = winner_targets(inp["score"], inp["eligible"], inp["vol"], inp["regime"], inp["weekly"],
                                tiebreak_w=inp["tiebreak"], check_log=checks, decision_log=decisions)
    finally:
        WINNER.clear()
        WINNER.update(saved)
    full = tgt.reindex(index=inp["close"].index, columns=inp["close"].columns).fillna(0.0)
    res = simulate(inp["open"], inp["close"], full, start, rebalance=inp["weekly"], cost=COST)
    return {"res": res, "targets": tgt, "checks": pd.DataFrame(checks), "decisions": pd.DataFrame(decisions)}



def buy_and_hold(inp, symbol, start=WALK_FORWARD_START):
    """simulate() result of holding 100% of one symbol (e.g. QQQ) from `start`."""
    t = pd.DataFrame(0.0, index=inp["close"].index, columns=inp["close"].columns)
    t[symbol] = 1.0
    return simulate(inp["open"], inp["close"], t, start)


def curve_metrics(eq):
    """Total %, CAGR %, Sharpe and max DD % of one equity curve."""
    m = metrics({"equity": eq, "exposure": eq * 0 + 1, "turnover": eq * 0, "trades": pd.DataFrame({"Return": []}),
                 "open_positions": pd.DataFrame()})
    return {k: m[k] for k in ("Total Return %", "CAGR %", "Sharpe", "Max DD %")}


def period_rows(name, eq):
    """One row per standard period: the whole walk-forward, the never-seen 2022-04 -> 2024-09 part, and the last 2 years /
    last 1 year (close-to-close, e.g. close 2025-09-24 -> close 2026-09-24)."""
    rows, last = [], eq.index[-1]

    def add(period, seg):
        rows.append({"Strategy": name, "Period": period, "Start": seg.index[0].date(), "End": seg.index[-1].date(),
                     **curve_metrics(seg)})
    add("Walk-forward", eq)
    add("Never-seen 2022-04 → 2024-09", eq.loc[:NEVER_SEEN_END])
    for years in (2, 1):
        start_close = eq.index[eq.index <= last - pd.DateOffset(years=years)][-1]
        add(f"Last {years} year" + ("s" if years > 1 else ""), eq.loc[start_close:] / eq.loc[start_close])
    return rows


def trade_stats(res, dates):
    """Trades (incl. open), win rate, median trade and median hold of one simulation (medians, not averages)."""
    tr = res["trades"]
    hold = dates.searchsorted(tr["Exit"]) - dates.searchsorted(tr["Entry"]) if len(tr) else np.array([])
    return {"Trades": len(tr) + len(res["open_positions"]),
            "Win rate %": (tr["Return"] > 0).mean() * 100 if len(tr) else np.nan,
            "Median trade %": tr["Return"].median() * 100 if len(tr) else np.nan,
            "Median hold (sessions)": float(np.median(hold)) if len(hold) else np.nan}


def per_stock_table(run, inp, start=WALK_FORWARD_START):
    """Per stock over the backtest: closed trades, win rate, median trade %, median hold, share of sessions held, and the
    stock's own buy & hold return from the first open on/after `start` (for comparison)."""
    tr, dates = run["res"]["trades"], inp["close"].index
    held = run["targets"].loc[start:] > 0
    rows = []
    for sym in TRADABLE:
        t = tr[tr["Symbol"] == sym]
        first = inp["open"][sym].loc[start:].first_valid_index()
        last_close = inp["close"][sym].dropna()
        bh = (last_close.iloc[-1] / inp["open"].at[first, sym] - 1) * 100 if first is not None else np.nan
        stats = trade_stats({"trades": t, "open_positions": pd.DataFrame()}, dates)
        rows.append({"Symbol": sym, "Sector": sector_mapping.symbol_sector.get(sym, "Other"),
                     "Closed trades": len(t), "Win rate %": stats["Win rate %"], "Median trade %": stats["Median trade %"],
                     "Median hold (sessions)": stats["Median hold (sessions)"],
                     "Held % of sessions": held[sym].mean() * 100 if sym in held else 0.0,
                     "Buy & hold %": bh, "First bar": first.date() if first is not None else None})
    return pd.DataFrame(rows)
