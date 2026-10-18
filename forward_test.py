"""Forward test from backtest_engine.FORWARD_START (Fri Oct 2, 2026), counted only from that close on:
  1. the real Alpaca account (read-only GETs through alpaca_paper, never orders) -> Reports/forward_test_daily.csv, one
     row per trading day: equity, cash, net deposits (a deposit booked after the close, ~4:15 PM CT, counts from the next
     row), positions, closed / winning picks, $ traded and its cost vs the decision price;
  2. STRATEGIES: the paper-only strategies (the current live rules, the pinned Oct 2 rules and variants, the Oct 15 50/50
     and low-volatility lines of forward_combos; never traded), built from the saved
     Reports files (signal_analysis, factor_history, earnings_date, benchmark_prices) + Alpaca daily volume
     (forward_bars) -> Reports/forward_strategies.csv (Value, 1.0 at the start; Cash) + forward_strategies_holdings.csv.
leaderboard(): every strategy + the account + QQQ / SPY, ranked by RANK_RULE, with one column per week.
Same accounting for all: decide at the day's close (the 2:30 PM bar on decision days), trade only names whose target
changed (all back to target on the strategy's rebalance day, 1-point band, earnings rule), be.COST per side, max 20% per
stock (the live one: be.WINNER's cap, 15% from Oct 15), 99% invested, buys never exceed the cash (no margin), cash earns 0. Saved days are never redone; missed days are
caught up from the saved bars. On Oct 15, 2026 all but the live one were recomputed from Oct 2 (Mon-Thu checks, 11 new);
later on Oct 15, 19 more (forward_combos: two independent halves, swap-only halves, the low-volatility keep / Friday
falling rules) - the two live halves start at the Oct 16 close, the rest are computed from Oct 2 like the others. The
live line was renamed "Live 50/50 combined (no overlap)" (was "Live rules (C6)"; its saved Oct 2-15 days are kept).

    python forward_test.py --record   # run_all.daily_step (Mon-Fri from 3:05 PM CT): strategies + today's account row
    python forward_test.py            # strategies only (no account request) + the leaderboard
"""
import argparse
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import backtest_engine as be  # noqa: E402

REPORTS = os.path.join(ROOT, "Reports")
DAILY_CSV = os.path.join(REPORTS, "forward_test_daily.csv")
STRATEGIES_CSV = os.path.join(REPORTS, "forward_strategies.csv")
HOLDINGS_CSV = os.path.join(REPORTS, "forward_strategies_holdings.csv")
BENCH_CSV = os.path.join(REPORTS, "benchmark_prices.csv")
ORDERS_CSV = os.path.join(REPORTS, "live_orders_log.csv")
SIGNAL_CSV = os.path.join(REPORTS, "signal_analysis.csv")
FACTOR_CSV = os.path.join(REPORTS, "factor_history.csv")
CT = ZoneInfo("America/Chicago")
DAILY_COLS = ["Date", "Time_CT", "Equity", "Cash", "Net_Deposits", "Positions", "Closed_Picks", "Winning_Picks",
              "Traded_USD", "Cost_USD"]
SIG_COLS = ["Date", "Symbol", "Close", "ma_50", "ma_10", "ma_30", "ma_100", "ma_200", "RSI", "macd", "MACD Signal",
            "Technical_Score", "RS_Score", "Strategy_Score", "Regime_On", "final_trade"]
MAX_WEIGHT = 0.20            # paper strategies: no stock above 20% (pinned; LIVE uses the be.WINNER max_weight, 15% from Oct 15)
# The paper strategies' base selection = the live WINNER of Oct 15, 2026 (pinned when live moved to the monthly low-volatility
# rules on Oct 16): 1/volatility weights, no sector limit, no rank gate, the sleeve's 5% cap, earnings rule E5.
PIN = {"vol_sizing": True, "sector_cap": 1.0, "max_pick_rank": None, "cap_soft": False, "sleeve": (),
       "sleeve_cap": be.sector_mapping.defensive_cap}
EARN_DAYS = 5
BAND = be.WINNER.get("rebalance_band") or 0.0   # a holding within 1 point of its new target is not traded (live band)
WEEKS_TO_WIN = 12
LIVE = "Live 50/50 combined (no overlap)"   # always the CURRENT live rules (be.WINNER + sector_mapping, not pinned); saved days kept
FRI = "Live 50/50, sector rotation on Fridays"      # Oct 16 evening: the sector half moved from Friday to Monday
MON = "Live 50/50, sector rotation on Mondays"      # = the current live rules (marked on the leaderboard)
MARKED = MON
# Renamed lines: saved rows take the new name (migrate_names, and the dashboard reads through this map).
# A rule that changed (ATR dip buy, top 30 -> top 10) is a new name on purpose, so the old results are not relabeled.
OLD_NAMES = {
    "Live rules (C6)": LIVE,
    "50/50: Low vol + SR weekly 63d (live combo from Oct 2)": FRI,
    "Quant Score": "Quant Score top 10",
    "Now: calmer half": "Now: RS, calmer half of the list",
    "Now: RS 10/21/63 days": "Now: shorter RS (10/21/63 days)",
    "Now: RS 63/126/252 days": "Now: longer RS (63/126/252 days)",
    "Now: 30% technical": "Now: 30% technical + 70% RS",
    "Now: exit below 15": "Now: Mon-Thu sell below rank 15",
    "Restored RS + sleeve, Mon-Thu checks (Oct 15 rules)": "Oct 15 rules (RS + sleeve, Mon-Thu)",
}
# the live 50/50 (be.WINNER "sleeves", from the Oct 16, 2026 switch): forward_combos legs, low-volatility half first
LIVE_LEGS = (("lowvol", "full", "month", 0.5), ("sr63", "full", "weekstart", 0.5))   # sector half on Mondays (Oct 19 on)
FRI_LEGS = (LIVE_LEGS[0], ("sr63", "full", "week", 0.5))
LIVE_SWITCH = "2026-10-16"
C6 = "Oct 2 rules (pinned)" # the rules the forward test started with (pinned: MIDWEEK / EXIT_BELOW / EXIT_TO_TOP, 50/50 score)
# paper_trade's Mon-Thu spare-cash rule (forward_test never imports paper_trade; tests check these equal its constants)
DEPLOY_MAX_RANK, DEPLOY_MIN_PCT = 10, 0.01
LIVE_MARK = " · Current strategy (live)"
ACCOUNT = "My Alpaca account"
RANK_RULE = (f"After {WEEKS_TO_WIN} full weeks the top strategy is the winner only if its total "
             f"return beats both {LIVE} and QQQ; otherwise the live rules stay. Nothing switches on its own: the live "
             "rules change only when you ask.")

# Name color on the Strategy tab (dashboard/details/forward_test.py). One family per strategy.
GROUPS = {
    "quant": "#7c3aed",          # purple
    "rs": "#dc2626",             # red
    "technical": "#1d4ed8",      # blue
    "momentum": "#c2410c",       # orange
    "lowvol": "#0f766e",         # teal
    "sector": "#15803d",         # green
    "blend": "#4338ca",          # indigo: half technical + half relative strength (the Oct 2 / C6 score)
    "fundamental": "#92400e",    # brown
    "event": "#be185d",          # pink: earnings drift, dips, breakouts, reversal
    "combo": "#0369a1",          # sky: a 50/50 of two families
}
GROUP_LABEL = {
    "quant": "Quant", "rs": "Relative strength", "technical": "Technical", "momentum": "Momentum",
    "lowvol": "Low volatility", "sector": "Sector rotation", "blend": "Technical + relative strength",
    "fundamental": "Fundamentals", "event": "Earnings, dips, breakouts", "combo": "50/50 combo",
}

# ---------------------------------------------------------------------------------------------------- the strategies
# One dict per strategy. Ranked strategies go through the engine's own selection (be.winner_targets); their defaults = the
# Oct 2 rules (C6): score "live" (0.5 x technical + 0.5 x relative strength), relative strength tiebreak, Friday rebalance +
# Mon-Thu swaps and exits ("mwf": MIDWEEK / EXIT_BELOW / EXIT_TO_TOP below); Friday = pure top-10 by rank (no sector / no rank-20 gate), 1/volatility
# weights, half size when QQQ is below its 200-day average, no buys 5 days before earnings. "select" changes rank_targets
# keys (n, sector_cap, vol_sizing, regime, ...), "earnings" None drops the earnings skip, "calendar" is "mwf" / "weekly"
# (Friday only) / "monthly" (last session of the month). "weights" = a rule that is not a ranking (its own weights below);
# "reweight" = the same picks with other weights; "stop" = the live rules + an ATR stop around earnings (STOPS);
# "spare_dips" = the live rules + ATR dip buys with the cash they leave unused; "midweek" = other Mon-Thu swap settings;
# "replay" = the live rules replayed day by day with a changed Mon-Thu exit (atr_stop_targets). "live" = LIVE: whatever
# be.WINNER / sector_mapping say now (score = the pipeline's Strategy_Score) + the Mon-Thu spare-cash buys (spare_cash).
# The "Now" strategies (**NOW) = one change each to the live rules of Oct 15 (pinned): relative strength only, top 10 +
# the defensive sleeve (sector_mapping.defensive, 5% cap), Mon-Thu sell worse than rank 20 refilled from the top 10,
# do_not_buy never bought; "exit" = (sell below, refill from top), "regime" = another market filter; no spare-cash buys.
PLAIN = {"max_pick_rank": None, "cap_soft": False, "sector_cap": 0.4}  # walk every rank, strict max 4/sector (pinned; live is now 1.0)
# The mid-week rules every C6 "mwf" strategy started with on Oct 2, 2026 (live until 2026-10-07), pinned here so a later
# live change (t187u: no top-3 swap, exit below 20) never alters a forward-test strategy: top-3 swap below rank 15,
# exit below rank 30 refilled from the best non-held top 10. Days Mon-Thu like live (Chirag, Oct 15; Mon/Wed before).
MON_THU = ["Mon", "Tue", "Wed", "Thu"]
MIDWEEK = {"enter_top": 3, "exit_below": 15, "days": MON_THU}
EXIT_BELOW, EXIT_TO_TOP = 30, 10
NOW = dict(score="rs", midweek={"enter_top": None, "exit_below": None, "days": MON_THU}, exit=(20, 10), sleeve=True,
           never_buy=True)
EQ = {**PLAIN, "vol_sizing": False}                      # ... and an equal weight per stock
SIMPLE = dict(select=PLAIN, earnings=None)                # a plain top-10 strategy: no rank-20 limit, no earnings skip
STRATEGIES = [
    dict(name=LIVE, live=True, group="combo",
         rule="The rules the account trades now, tracked on paper. Two halves that never hold the same stock, each with its own cash: "
         "50% low volatility (the 10 calmest stocks above their 200-day average, at most 4 per sector, equal weight, rebalanced "
         "Oct 16 and then the last trading day of each month) and 50% sector rotation (the best Oct 2 scores in the 3 sectors "
         "with the best 63-day ETF return, top 10, 1/volatility weights; every Friday through Oct 16, every Monday from Oct 19). "
         "No Mon-Thu trading, no earnings block, no defensive sleeve, at most 20% of a half in one stock, do_not_buy never bought. "
         "Oct 2-15 stays as recorded (then named 'Live rules (C6)': Oct 13-15 was relative strength plus the sleeve; before that, the Oct 2 rules)."),
    dict(name=C6, group="blend",
         rule="What the bot traded Oct 2-7 (pinned): 50% technical + 50% relative strength, top 10, Fri rebalance + "
              "Mon-Thu top-3 swap (below rank 15) and exit below rank 30 (checks Mon/Wed until Oct 15)."),
    dict(name="C6, 4 per sector, no earnings skip", select=PLAIN, earnings=None, group="blend",
         rule="The Oct 2 rules (half technical, half relative strength, top 10, Friday rebalance and Mon-Thu checks) with two "
              "changes: at most 4 stocks from any one sector, and an earnings date never blocks a buy. It still ranks the whole "
              "list. Sector rotation is a different line: it first keeps only the 3 sectors whose ETF has the best 63-day return."),
    dict(name="Technical score only", score="tech", group="technical",
         rule="Oct 2 rules ranking on the technical score alone (moving averages, RSI, MACD). Top 10."),
    dict(name="Relative strength only", score="rs", group="rs",
         rule="Oct 2 rules ranking on relative strength alone: 60% the stock versus its sector ETF, 40% that ETF versus SPY, "
              "over 21, 63 and 126 trading days. Top 10."),
    dict(name="Quant Score top 10", score="quant", tiebreak=None, group="quant",
         rule="Quant Score only, 10 stocks. The score is 0-100 from the trend, the moving averages, the 20-day VWAP, RSI, MACD "
              "and Bollinger bands (a missing part counts as 50). It holds the 10 highest scores above 50, on the Oct 2 calendar "
              "(Friday rebalance and Mon-Thu checks), 1/volatility weights. The old name was Quant Score; that line was already "
              "these 10 stocks. Quant Score top 5 is the same score with 5 stocks."),
    dict(name="Quant Score top 5", score="quant", tiebreak=None, select={"n": 5}, group="quant",
         rule="The same Quant Score as Quant Score top 10 (the 0-100 trend, moving-average, VWAP, RSI, MACD and Bollinger score, "
              "above 50), holding 5 stocks instead of 10."),
    dict(name="Quant + relative strength", score="quant_rs", group="quant",
         rule="Oct 2 rules ranking on a 50/50 blend of Quant Score rank and relative strength rank. Top 10."),
    dict(name="Company fundamentals", score="fundamental", calendar="monthly", select=EQ, earnings=None, group="fundamental",
         rule="Top 10 by the company-report score (Fundamental_Weight), equal weight, monthly."),
    dict(name="12-1 month momentum", score="mom_12_1", calendar="monthly", **SIMPLE, group="momentum",
         rule="Top 10 by the return from 12 months ago to 1 month ago (252 trading days back, stopping 21 trading days ago). "
              "The last month is left out, so a jump in the last few weeks does not count. Monthly, 1/volatility weights. "
              "6-month momentum uses the plain last 126 trading days and rebalances every Friday."),
    dict(name="6-month momentum", score="mom_6", calendar="weekly", **SIMPLE, group="momentum",
         rule="Top 10 by the plain return over the last 126 trading days (about 6 months), including the last month. Every Friday, "
              "1/volatility weights. 12-1 month momentum leaves out the last month and rebalances once a month."),
    dict(name="Low volatility in uptrend", score="low_vol", calendar="monthly", select=EQ, earnings=None, group="lowvol",
         rule="The 10 calmest stocks by 63-day volatility, and only those whose close is above the 200-day average. "
              "How far the price rose is not the rank. Equal weight, monthly. Volatility-adjusted momentum is a different "
              "line: it ranks 6-month return divided by 6-month volatility, every Friday, with no 200-day filter."),
    dict(name="Buy the dip", weights="dip", group="event",
         rule="The copy folder's dip test: buy a stock 8% below its 50-day average, sell when it is back at the average or "
              "after 30 days; 10% each, at most 10, deepest first, checked daily."),
    dict(name="Top 5", select={"n": 5}, group="blend", rule="The Oct 2 rules with 5 stocks."),
    dict(name="Top 20", select={"n": 20}, group="blend",
         rule="The Oct 2 rules (half technical, half relative strength, Friday rebalance and Mon-Thu checks, 1/volatility "
              "weights) with 20 stocks. Now: top 15 is relative strength with 15 stocks plus the defensive sleeve. "
              "200-day high is the 20 stocks closest to their own 200-day high, monthly."),
    dict(name="Friday only", calendar="weekly", group="blend",
         rule="The Oct 2 rules (half technical, half relative strength, top 10) rebalanced on Friday only, with no Mon-Thu "
              "swaps or exits. Now: Friday only is different: it uses the Oct 15 relative-strength rules and keeps the five defensive stocks."),
    dict(name="Monthly", calendar="monthly", group="blend",
         rule="The Oct 2 rules, rebalanced on the last trading day of each month only."),
    dict(name="Dual momentum", score="dual", calendar="monthly", select={**EQ, "regime": None, "regime_scale": None},
         earnings=None, group="momentum",
         rule="Two checks, monthly, equal weight, top 10, at most 4 per sector. The stock's own 12-month return "
              "(252 trading days) must be positive, and the strongest of those returns are bought. While QQQ is below "
              "its 200-day average the whole book is cash. An earnings date does not block a buy."),
    dict(name="Trend following 50/200", weights="trend", group="technical",
         rule="Every Friday, each scored stock whose 50-day average is above its 200-day average gets one equal slot: "
              "1 divided by how many stocks are scored that day. A stock below that trend gets nothing, so part of the book "
              "stays in cash when few stocks qualify. Slots stay put until the next Friday. The count is however many stocks "
              "are in the uptrend, not a fixed top 10."),
    dict(name="Volatility-adjusted momentum", score="mom_risk", calendar="weekly", select=EQ, earnings=None, group="momentum",
         rule="Top 10 by the 6-month return divided by 6-month volatility, so more gain per unit of swing ranks higher. "
              "The 6-month return must be positive. Equal weight, every Friday. Low volatility in uptrend is a different "
              "line: it holds the calmest stocks above their 200-day average and does not rank the size of the gain."),
    dict(name="200-day high", score="high_200", calendar="monthly", select={**EQ, "n": 20}, earnings=None, group="momentum",
         rule="The 20 stocks closest to their highest close of the last 200 trading days (today's close divided by that "
              "high; 1 means the stock is at the high). Equal weight, monthly, at most 8 per sector (the 40% sector cap on a 20-stock book), half size while QQQ "
              "is below its 200-day average. Top 20 ranks the Oct 2 score instead. Now: top 15 is relative strength plus the defensive sleeve."),
    dict(name="Short-term reversal", score="reversal", calendar="weekly", select=EQ, earnings=None, group="event",
         rule="The 10 biggest 1-week losers among stocks above their 200-day average, equal weight, every Friday."),
    dict(name="Quality + momentum", score="quality_mom", calendar="monthly", **SIMPLE, group="fundamental",
         rule="Top 10 by half the company-report score (Fundamental_Weight, the quality rank) and half the 12-1 month momentum "
              "rank (return from 12 months ago to 1 month ago). Monthly, 1/volatility weights."),
    dict(name="Earnings drift", score="drift", calendar="weekly", **SIMPLE, group="event",
         rule="Stocks whose last earnings reaction beat SPY, biggest jump first. A morning report is judged that same day; "
              "an afternoon report is judged the next session. That one-day jump versus SPY is kept for 60 trading days. "
              "Top 10, every Friday. There is no EPS-surprise feed, so the price jump stands in for the beat."),
    dict(name="Sector rotation", score="sector_rot", calendar="weekly", **SIMPLE, group="sector",
         rule="The 3 sector ETFs with the best 63-day (about 3-month) return, then the best Oct 2 scores (half technical, "
              "half relative strength) inside only those sectors. Top 10, every Friday, no Mon-Thu checks, and an earnings date "
              "never blocks a buy. The C6 4-per-sector line is different: it ranks the whole list and can hold any sector."),
    dict(name="Breakout with volume", score="breakout", calendar="weekly", select=EQ, earnings=None, group="event",
         rule="A stock qualifies when, during the last 20 sessions, it closed within 2% of its 55-day high on at least "
              "1.3 times normal volume (the 5-day average volume versus the 50-day average), and the close is still above "
              "the 50-day average. The rank is that volume jump, not the size of the price gain. Top 10, equal weight, every Friday."),
    dict(name="ATR dip buy (top 10)", weights="atr_dip", group="event",
         rule="Same dip as ATR dip buy, sell on signal, with a shorter list and a fixed exit. Buy only a top-10 name on the "
              "Oct 2 score (half technical, half relative strength, score above 0) when the close is 3 times ATR(14) or more "
              "below the highest close of the last 20 days. 10% each, at most 10, best rank first. Sold after 10 trading days."),
    dict(name="ATR dip buy, sell on signal", weights="atr_dip_signal", group="event",
         rule="Same dip as ATR dip buy (top 10): the close is 3 times ATR(14) or more below the highest close of the last "
              "20 days. This line uses the top 30 of the Oct 2 score, and it does not sell after 10 days. It sells when "
              "signal analysis says SELL (score below 0) or the rank falls below 30. 10% each, at most 10, best rank first."),
    dict(name="Risk parity", reweight="risk_parity", group="blend",
         rule="The Oct 2 picks (half technical, half relative strength), weighted so each stock adds the same risk "
              "(63-day volatility and correlation)."),
    dict(name="Top 20 rank-weighted (70/30)", score="rs", select={"n": 20}, reweight="rank_tier", group="rs",
         rule="Relative strength top 20 weighted by rank: ranks 1-10 share 70% (8.5% down to 5.5%), ranks 11-20 share "
              "30% (5.0% down to 1.0%). Half size while QQQ is below its 200-day average."),
    # ---- the live rules of Oct 15, 2026 (pinned) with one change each (added Oct 15, computed from Oct 2)
    dict(name="Now: Friday only", **NOW, calendar="weekly", group="rs",
         rule="The Oct 15 rules (relative strength, top 10, the five defensive stocks) with the Friday rebalance only. "
              "Mon-Thu does not sell or replace anything. Friday only, without the Now: prefix, is the older Oct 2 score "
              "(half technical, half relative strength) and holds no defensive stocks."),
    dict(name="Now: no defensive sleeve", **{**NOW, "sleeve": False}, group="rs",
         rule="The Oct 15 rules without the five defensive stocks (LMT, NOC, ABBV, PFE, KO). On the other Oct 15 lines those "
              "five are always held, on top of the top 10, at 5% each. This line holds only the ranked top 10."),
    dict(name="Now: top 15", **NOW, select={"n": 15}, group="rs",
         rule="The Oct 15 rules (relative strength only, Friday rebalance, Mon-Thu sell below rank 20) with 15 ranked stocks "
              "plus the five defensive stocks. Top 20 is the Oct 2 score (half technical, half relative strength) with 20 stocks "
              "and no sleeve. 200-day high is the 20 stocks closest to their own 200-day high, monthly, equal weight."),
    dict(name="Now: Mon-Thu sell below rank 15", **{**NOW, "exit": (15, 10)}, group="rs",
         rule="The Oct 15 rules, except on Mon-Thu a holding ranked worse than 15 (rank 1 is best) is sold and replaced from "
              "the top 10. The Oct 15 line uses rank 20 for that sell. Friday is still a full rebalance of the top 10 plus the defensive stocks."),
    dict(name="Now: shorter RS (10/21/63 days)", **{**NOW, "score": "rs_short"}, group="rs",
         rule="The Oct 15 rules, with relative strength measured over 10, 21 and 63 trading days (about 2 weeks, 1 month and "
              "3 months). The Oct 15 windows are 21, 63 and 126 days (about 1, 3 and 6 months). Same blend on each window: "
              "60% the stock versus its sector ETF, 40% that ETF versus SPY."),
    dict(name="Now: longer RS (63/126/252 days)", **{**NOW, "score": "rs_long"}, group="rs",
         rule="The Oct 15 rules, with relative strength measured over 63, 126 and 252 trading days (about 3 months, 6 months "
              "and 1 year). The Oct 15 windows are 21, 63 and 126 days (about 1, 3 and 6 months). Same blend on each window: "
              "60% the stock versus its sector ETF, 40% that ETF versus SPY."),
    dict(name="Now: 30% technical + 70% RS", **{**NOW, "score": "tech30"}, group="blend",
         rule="The Oct 15 rules (top 10, the five defensive stocks, Friday rebalance, Mon-Thu sell below rank 20), with the "
              "rank changed to 30% technical score and 70% relative strength. The Oct 15 rank itself was 100% relative strength. "
              "The Oct 2 score was 50% technical and 50% relative strength."),
    dict(name="Now: RS, calmer half of the list", **{**NOW, "score": "rs_calm"}, group="rs",
         rule="The Oct 15 rules (relative strength, top 10, the five defensive stocks, Mon-Thu sell below rank 20), limited to "
              "the calmer half of the list. A stock can be bought only when its 63-day volatility is at or below the middle of "
              "the list that day. A holding that moves into the jumpier half is sold on Mon-Thu."),
    # ---- added Oct 15, 2026 (evening, Chirag): the live 50/50 halves (from the Oct 16 close) and every rule discussed
    # that day (from Oct 2), simulated by forward_combos (no_overlap: the low-volatility half has priority)
    dict(name="Live half: Low vol monthly", combo=dict(legs=LIVE_LEGS, start=LIVE_SWITCH, which=0), ranked=False, group="lowvol",
         rule="The live account's low-volatility half on its own (from the Oct 16 close, 1.0 then): the 10 calmest stocks "
              "above their 200-day average, max 4 per sector, equal weight, rebalanced on Oct 16 and then the last trading "
              "day of each month; do_not_buy never bought. Not ranked (starts Oct 16)."),
    dict(name="Live half: Sector rotation weekly (63d)", combo=dict(legs=LIVE_LEGS, start=LIVE_SWITCH, which=1), ranked=False,
         group="sector",
         rule="The live account's sector-rotation half on its own (from the Oct 16 close, 1.0 then): the best C6-score "
              "stocks in the 3 sectors whose ETF has the best 63-day return, top 10, 1/volatility weights, max 20% of the "
              "half, rebalanced every Friday; never buys a stock the low-volatility half holds. Not ranked (starts Oct 16)."),
    dict(name="Oct 15 rules (RS + sleeve, Mon-Thu)", **NOW, group="rs",
         rule="What the account traded on Oct 13-15: relative strength only, over 21, 63 and 126 trading days (about 1, 3 and "
              "6 months), top 10, plus five defensive stocks always held (LMT, NOC, ABBV, PFE, KO, 5% each). Friday is a full "
              "rebalance. On Mon-Thu a holding worse than rank 20 is sold and replaced from the top 10. do_not_buy is never "
              "bought. Each Now: line is this rule with one change. Now: Friday only drops the Mon-Thu checks. The account "
              "left these rules on Oct 16 for the 50/50."),
    dict(name=FRI, combo=dict(legs=FRI_LEGS), group="combo",
         rule="The live 50/50 computed from Oct 2: low vol half on the last trading day of the month + sector rotation "
              "(63-day ETF return, top 3 sectors) half rebalanced every Friday, no overlap (the live rules until Oct 16)."),
    dict(name=MON, combo=dict(legs=LIVE_LEGS), group="combo",
         rule="The live 50/50 computed from Oct 2: low vol half on the last trading day of the month + sector rotation "
              "half rebalanced every Monday (Tuesday after a Monday holiday), no overlap. Current live rules from Oct 19."),
    dict(name="50/50: Low vol + SR every 2 weeks 63d", combo=dict(legs=(LIVE_LEGS[0], ("sr63", "full", "2week", 0.5))),
         group="combo",
         rule="50/50, the sector-rotation half (63 days) rebalanced every other Friday (from Oct 2), no overlap."),
    dict(name="50/50: Low vol + SR Friday swap, keep top 10", combo=dict(legs=(LIVE_LEGS[0], ("sr63", "swap10", "week", 0.5))),
         group="combo",
         rule="50/50, the sector-rotation half swap-only (keep top 10) checked on Fridays only."),
    dict(name="50/50: Low vol + Vol-adjusted momentum weekly", combo=dict(legs=(LIVE_LEGS[0], ("vam", "full", "week", 0.5))),
         group="combo",
         rule="50/50, the second half = volatility-adjusted momentum (6-month return / 6-month volatility), top 10, equal "
              "weight, weekly, no overlap."),
    dict(name="50/50: Low vol + Calmer half weekly", combo=dict(legs=(LIVE_LEGS[0], ("calm", "full", "week", 0.5))),
         group="combo",
         rule="50/50, the second half = 'Now: RS, calmer half of the list' (relative strength among stocks at or below the "
              "median 63-day volatility) rebalanced weekly, no Mon-Thu checks, no overlap."),
]
for _c in STRATEGIES:                                     # the limits the plain rankings keep (said once, here)
    if (_c.get("select") is PLAIN or _c.get("select") is EQ) and not _c["rule"].startswith("The Oct 2 rules"):
        _c["rule"] += " At most 4 per sector; like live: half size while QQQ is below its 200-day average, score above 0."
    if _c.get("group") not in GROUPS:
        raise ValueError(f"{_c['name']}: missing color group")
GROUP_OF = {c["name"]: c["group"] for c in STRATEGIES}

def strategy_color(name):
    """CSS color of a strategy's family, or '' for the account and the benchmarks."""
    return GROUPS.get(GROUP_OF.get(str(name).replace(LIVE_MARK, ""), ""), "")
STOPS = {"pre": dict(k=3.5, arm_days=7, gap_open=True), "post": dict(k=3.0, after=10)}   # atr_stop_targets settings
FWD_BARS = os.path.join(REPORTS, "cache", "forward_bars.pkl")
FWD_BARS_START = "2026-05-01"      # ~100 sessions before the start: the 50-day volume average is full by Oct 2
_NO_SAVE = datetime(2000, 1, 1, tzinfo=be.EASTERN)   # be.keep_decision_bars(now=this) only reads Reports/decision_bars.csv

def _wide(df, col, index="Date"):
    return df.pivot(index=index, columns="Symbol", values=col).sort_index()

def _pct(w):
    return w.rank(axis=1, pct=True)

def _carry(w, days):
    """Weights that only change on `days` (carried in between)."""
    return w[days.to_numpy(bool)].reindex(w.index).ffill().fillna(0.0)

def forward_bars(fetch=True):
    """Daily bars with volume for the VWAP / volume rules: a fresh download from FWD_BARS_START (Alpaca free market data
    through be.fetch_daily_bars, the pipeline's own client; saved to Reports/cache/forward_bars.pkl and reused when a
    download fails) + the backtest bar cache (Reports/cache/bars_daily_long.pkl, read only) for the older days."""
    if fetch:
        try:
            syms = list(be.TRADABLE) + [s for s in be.BENCHMARKS if s not in be.TRADABLE]   # + QQQ / SPY (provisional)
            new, _ = be.drop_partial_last_bar(be.fetch_daily_bars(syms, start=FWD_BARS_START),
                                              close_buffer_min=be.AFTER_CLOSE_BAR_MIN)   # today's bar from 3:05 PM CT
            new.to_pickle(FWD_BARS + ".tmp")
            os.replace(FWD_BARS + ".tmp", FWD_BARS)
        except Exception as e:
            print(f"bars not refreshed ({type(e).__name__}: {e}); using the saved copy")
    return [be.apply_history_start(pd.read_pickle(p)) for p in (FWD_BARS, be.CACHE_DIR / be.LONG_CACHE) if os.path.exists(p)]

def bar_features(sources, dates, cols):
    """Close / 20-day VWAP (the copy's daily-bar VWAP: typical price x volume), the volume jump (5-day / 50-day average
    volume), ATR(14) / close and open / close, computed inside each download so splits and dividends stay consistent; the newest download wins where its
    window is full. Decision days use their 2:30 PM bar, like signal_analysis.csv."""
    out = {k: pd.DataFrame(index=dates, columns=cols, dtype=float) for k in ("vwap_ratio", "vol_jump", "atr_ratio", "open_ratio")}
    for b in sources:
        b = be.keep_decision_bars(b.sort_values(["Symbol", "Date"]).reset_index(drop=True), now=_NO_SAVE)
        w = {c: b.pivot_table(index="Date", columns="Symbol", values=c) for c in be.BAR_COLS}
        v = w["Volume"].replace(0, np.nan)
        vwap = ((w["High"] + w["Low"] + w["Close"]) / 3 * v).rolling(20).sum() / v.rolling(20).sum()
        pc = w["Close"].shift(1)                         # ATR: Wilder 14, as in be.calculate_technical_indicators
        tr = np.fmax(w["High"] - w["Low"], np.fmax((w["High"] - pc).abs(), (w["Low"] - pc).abs()))
        atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        new = {"vwap_ratio": w["Close"] / vwap, "vol_jump": v.rolling(5).mean() / v.rolling(50).mean(),
               "atr_ratio": atr / w["Close"], "open_ratio": w["Open"] / w["Close"]}
        for k in out:
            out[k] = out[k].combine_first(new[k].reindex(index=dates, columns=cols))
    return out

def quant_score(f, vwap_ratio):
    """The copy folder's Quant_Score (Stock Analysis Test Strategy/backtest_engine.quant_score), 0-100, from the saved
    signal_analysis.csv columns + the bars' VWAP: the average of price above its 5 moving averages, the moving-average
    stack, price vs the 20-day VWAP, RSI(14), the MACD histogram (point-in-time scaled), Bollinger %B (20 days) and the
    20-day slope of the 200-day average (a missing part counts as 50)."""
    c, ma = f["Close"], {n: f[f"ma_{n}"] for n in (10, 30, 50, 100, 200)}
    mid, sd = c.rolling(be.BB_WINDOW).mean(), c.rolling(be.BB_WINDOW).std()
    parts = [sum((c > m).astype(float) for m in ma.values()) / 5 * 100,
             sum((ma[a] > ma[b]).astype(float) for a, b in ((10, 30), (30, 50), (50, 100), (100, 200))) / 4 * 100,
             (50 + 2500 * (vwap_ratio - 1)).clip(0, 100),
             f["RSI"].clip(0, 100),
             (50 + (f["macd"] - f["MACD Signal"]).apply(be.pit_scale) / 2).clip(0, 100),
             ((c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)).clip(0, 1) * 100,
             (50 + 1250 * (ma[200] / ma[200].shift(20) - 1)).clip(0, 100)]
    return sum(p.fillna(50.0) for p in parts) / len(parts)

def dip_weights(c, ma50, eligible, depth=0.08, max_hold=30, n=10):
    """test_buy_high_sell_low.ipynb as a portfolio: buy at the close 8% or more below the 50-day average, sell at the close
    back at/above it or after max_hold sessions (not bought back that day); 1/n each, at most n at once (deepest first)."""
    C, M, E = c.to_numpy(float), ma50.to_numpy(float), eligible.to_numpy(bool)
    out, held = np.zeros(C.shape), {}
    for t in range(len(C)):
        sold = set()
        for j in list(held):
            held[j] += 1
            if not C[t, j] < M[t, j] or held[j] >= max_hold:
                del held[j]
                sold.add(j)
        gap = np.nan_to_num(1 - C[t] / M[t], nan=-1.0)
        new = [j for j in np.argsort(-gap, kind="stable") if E[t, j] and gap[j] >= depth and j not in held and j not in sold]
        for j in new[:n - len(held)]:
            held[j] = 0
        out[t, list(held)] = 1 / n
    return pd.DataFrame(out, index=c.index, columns=c.columns)

def earnings_drift(c, spy, earnings, hold=60):
    """Excess return vs SPY (%) of each stock's last earnings reaction day (the report day for AM reports, else the next
    session), kept for `hold` sessions from that close; NaN otherwise."""
    out, idx = np.full(c.shape, np.nan), c.index
    pos = {s: j for j, s in enumerate(c.columns)}
    e = earnings[earnings["Symbol"].isin(pos)].sort_values("Earnings Date")
    for sym, day, when in zip(e["Symbol"], e["Earnings Date"], e.get("Time", pd.Series("", index=e.index)).astype(str)):
        k = idx.searchsorted(day, side="left" if when.strip().upper() == "AM" else "right")
        if 1 <= k < len(idx):
            j = pos[sym]
            out[k:k + hold, j] = (c.iat[k, j] / c.iat[k - 1, j] - spy.iat[k] / spy.iat[k - 1]) * 100
    return pd.DataFrame(out, index=idx, columns=c.columns)

RANK_WEIGHTS = np.r_[np.linspace(8.5, 5.5, 10), np.linspace(5.0, 1.0, 10)] / 100   # ranks 1-10: 70%, 11-20: 30%

def rank_tier(tgt, score, weights=RANK_WEIGHTS):
    """The same picks reweighted by rank (best score first): rank k gets weights[k-1], always less than rank k-1;
    scaled to the day's total exposure (e.g. half size). Weights change only when the picks change."""
    out, prev, last = tgt.copy() * 0.0, None, None
    for d, row in tgt.iterrows():
        names = frozenset(row.index[row > 0])
        if names != prev:
            order = score.loc[d, [c for c in tgt.columns if c in names]].sort_values(ascending=False, kind="stable").index[:len(weights)] if names else []
            w = pd.Series(0.0, index=tgt.columns)
            w[list(order)] = weights[:len(order)]
            last, prev = w / w.sum() if w.sum() else w, names
        out.loc[d] = last * row.sum()
    return out

def risk_parity(tgt, close, days, window=63):
    """Same picks and total weight, re-split for equal risk contributions (63-day covariance; cyclical coordinate descent),
    recomputed when the picks change and on `days`; 1/volatility weights stay where the covariance is incomplete."""
    r, T = close.pct_change(fill_method=None), tgt.to_numpy(float)
    out, key, cur, reb = T.copy(), None, None, days.reindex(tgt.index).fillna(False).to_numpy(bool)
    for t in range(len(T)):
        picked = np.flatnonzero(T[t] > 0)
        k = (tuple(picked), round(T[t].sum(), 8))
        if k != key or reb[t]:
            key, cur = k, T[t].copy()
            cov = r.iloc[max(0, t - window + 1):t + 1, picked].cov().to_numpy()
            if len(picked) > 1 and np.isfinite(cov).all():
                x = np.full(len(picked), 1 / len(picked))
                for _ in range(100):
                    for i in range(len(x)):
                        b = cov[i] @ x - cov[i, i] * x[i]
                        x[i] = (-b + np.sqrt(b * b + 4 * cov[i, i] / len(x))) / (2 * cov[i, i])
                cur[picked] = x / x.sum() * T[t].sum()
        out[t] = cur
    return pd.DataFrame(out, index=tgt.index, columns=tgt.columns)

def earnings_events(dates, cols, earnings):
    """[(column, report day, reaction-day row)] per report in `dates`: the reaction day is the report day for AM reports,
    else the next session (after-close reports)."""
    pos, ev = {s: j for j, s in enumerate(cols)}, []
    e = earnings[earnings["Symbol"].isin(pos)]
    for sym, day, when in zip(e["Symbol"], e["Earnings Date"], e.get("Time", pd.Series("", index=e.index)).astype(str)):
        r = dates.searchsorted(day, side="left" if when.strip().upper() == "AM" else "right")
        if r < len(dates):
            ev.append((pos[sym], day, r))
    return ev

def atr_stop_targets(inp, k=3.0, after=None, arm_days=None, gap_open=False, exit_to_top="winner"):
    """The Oct 2 live rules (mid-week rules pinned: MIDWEEK / EXIT_BELOW / EXIT_TO_TOP) + an ATR trailing stop around earnings. They are replayed day by day with the engine's own
    pieces, exactly as be.winner_targets does (Friday selection with the real holdings, Mon-Thu swaps and exits). Stop
    window: after = from the reaction day through `after` sessions later; arm_days = a stock kept (HOLD) at a Friday
    rebalance with earnings within the next arm_days calendar days, from that Friday through its reaction day. Inside it
    a holding whose close is at or below its highest close since it was bought - k x ATR(14) is sold in full at that close
    (gap_open: on the reaction day an open already below the stop sells at the open); its weight goes to the best stock
    the live rules would buy (same selection as WINNER: pure top-10 when sector_cap is 1.0 / max_pick_rank None),
    no buy within 5 days of earnings, and not a stock stopped out in its window. exit_to_top: on Mon-Thu, after the top-3 swaps, a holding ranked worse than the exit
    rank (30) is swapped for the best non-held stock in the top exit_to_top (same weight, earnings rule) and only sold to
    cash when none is left. Default "winner" = the pinned EXIT_TO_TOP (10); pass None for cash-until-Friday.
    Returns (raw weights, open-sale prices)."""
    W, args = be.WINNER, {**be.winner_rank_args(inp["regime"]), **PIN, "sleeve": (), "max_weight": MAX_WEIGHT}   # pinned Oct 2 rules
    if exit_to_top == "winner":
        exit_to_top = EXIT_TO_TOP                         # pinned mid-week rules (MIDWEEK / EXIT_BELOW / EXIT_TO_TOP)
    score, elig, vol, tb = inp["scores"]["live"], inp["eligible"], inp["vol"], inp["scores"]["rs"]
    dates, cols = score.index, list(score.columns)
    arr = lambda x: x.reindex(index=dates, columns=cols).to_numpy(float)
    S, V, TB, C, A, O = arr(score), arr(vol), arr(tb), arr(inp["close"]), arr(inp["atr"]), arr(inp["open"])
    E = elig.reindex(index=dates, columns=cols).astype("boolean").fillna(False).to_numpy(bool)
    BB = arr(be.earnings_days_ahead(dates, cols, inp["earnings"], EARN_DAYS))   # not NaN = not bought
    mw, n, min_score = MIDWEEK, W["n"], W["min_score"]
    reb = inp["weekly"].reindex(dates).fillna(False).to_numpy(bool)
    chk = be.midweek_check_days(dates, mw["days"], inp["weekly"]).to_numpy(bool)
    name_pos, sectors = be._name_positions(cols), np.array([be.sector_mapping.symbol_sector.get(c, "Other") for c in cols])
    per_sector = max(1, int(np.floor(args["sector_cap"] * n)))
    win, react, reports = np.zeros((len(dates), len(cols)), bool), np.zeros((len(dates), len(cols)), bool), {}
    for j, day, r in earnings_events(dates, cols, inp["earnings"]):
        react[r, j] = True
        reports.setdefault(j, []).append((day, r))
        if after is not None:
            win[r:r + after + 1, j] = True
    out, opens = np.zeros((len(dates), len(cols))), np.full((len(dates), len(cols)), np.nan)
    cur, peak, armed = np.zeros(len(cols)), np.full(len(cols), np.nan), np.full(len(cols), -1)

    def ranked(t):
        ok = E[t] & ~np.isnan(S[t]) & ~np.isnan(V[t]) & (V[t] > 0) & (S[t] > min_score)
        return be.ranking_order(np.where(ok)[0], S[t], TB[t], name_pos)

    def stop_out(t, j):                                   # sold in full; not bought back until its window ends
        end = armed[j] + 1 if armed[j] >= t else (t + np.argmin(win[t:, j]) if not win[t:, j].all() else len(dates))
        BB[t:end, j], peak[j], armed[j], w = 0.0, np.nan, -1, cur[j]
        cur[j] = 0.0
        return w

    for t in range(len(dates)):
        freed = []
        if gap_open and t:                                # the reaction day opens below yesterday's stop: sold at the open
            inside = win[t] | (armed >= t)
            for j in np.where(react[t] & inside & (cur > 0) & (O[t] <= peak - k * A[t - 1]))[0]:
                opens[t, j] = O[t, j]
                freed.append(stop_out(t, j))
        if reb[t]:
            day, before = dates[[t]], cur.copy()
            cur = be.rank_targets(score.iloc[[t]], elig, vol, rebalance_days=pd.Series(True, index=day), tiebreak_w=tb,
                                  buy_block=pd.DataFrame(BB[[t]], index=day, columns=cols), start_holdings=cur.copy(),
                                  **args).iloc[0].to_numpy(float)
            freed = []                                    # the Friday selection already refilled
            if arm_days:                                  # kept at this rebalance, earnings within arm_days: armed
                for j in np.where((before > 0) & (cur > 0))[0]:
                    for day_, r in reports.get(j, []):
                        if dates[t] < day_ <= dates[t] + pd.Timedelta(days=arm_days):
                            armed[j] = max(armed[j], r)
        elif chk[t] and cur.sum() > 0:
            order = ranked(t)
            rank = {j: r + 1 for r, j in enumerate(order)}
            skip = {j for j in order[:mw["enter_top"]] if cur[j] == 0 and not np.isnan(BB[t, j])}
            be.midweek_swap_pairs(cur, order, rank, sectors, mw["enter_top"], mw["exit_below"],
                                  10 ** 6 if args["cap_soft"] else per_sector, skip=skip)
            skip_exit = {j for j in order[:exit_to_top] if cur[j] == 0 and not np.isnan(BB[t, j])} if exit_to_top else set()
            be.midweek_exit_replacements(cur, order, rank, EXIT_BELOW, exit_to_top, skip=skip_exit)
            be.midweek_exit_sells(cur, rank, EXIT_BELOW)
        peak, armed = np.where(cur > 0, peak, np.nan), np.where(cur > 0, armed, -1)   # sold by the live rules: reset
        inside = ~np.isnan(peak) & (cur > 0) & (win[t] | (armed >= t))       # held before today, inside its window
        for j in np.where(inside & (C[t] <= np.fmax(peak, C[t]) - k * A[t]))[0]:
            freed.append(stop_out(t, j))
        for w in freed:                                   # refill each freed slot (its weight) like a live buy
            _pool = ranked(t) if args.get("max_pick_rank") is None else ranked(t)[:args["max_pick_rank"]]
            cands = [j for j in _pool if cur[j] == 0 and np.isnan(BB[t, j])]
            fits = [j for j in cands if sum(sectors[cur > 0] == sectors[j]) < per_sector] or (cands if args["cap_soft"] else [])
            if fits:
                cur[fits[0]] = w
        peak = np.where(cur > 0, np.fmax(peak, C[t]), np.nan)
        out[t] = cur
    return pd.DataFrame(out, index=dates, columns=cols), pd.DataFrame(opens, index=dates, columns=cols)

def atr_dip_weights(signal, priority, hold=10, n=10, room=None, taken=None, sell=None):
    """ATR dip buys: buy at the close of a signal day (1/n each, best priority first, at most `room` positions that day,
    default n), sell at the close `hold` sessions later (None = no fixed hold) or on a `sell` day. taken = stocks it may
    not hold that day (sold if held); with fewer rooms than positions the oldest are sold first."""
    S, P = signal.to_numpy(bool), np.nan_to_num(priority.reindex_like(signal).to_numpy(float), nan=-np.inf)
    room = np.full(len(S), n) if room is None else room
    tk = np.zeros(S.shape, bool) if taken is None else taken
    X = np.zeros(S.shape, bool) if sell is None else sell.reindex_like(signal).fillna(False).to_numpy(bool)
    out, held = np.zeros(S.shape), {}                     # column -> sessions held
    for t in range(len(S)):
        for j in list(held):
            held[j] += 1
            if (hold and held[j] >= hold) or tk[t, j] or X[t, j]:
                del held[j]
        while len(held) > room[t]:
            del held[max(held, key=held.get)]
        new = [j for j in np.argsort(-P[t], kind="stable") if S[t, j] and j not in held and not tk[t, j] and not X[t, j]]
        for j in new[:max(0, room[t] - len(held))]:
            held[j] = 0
        out[t, list(held)] = 1 / n
    return pd.DataFrame(out, index=signal.index, columns=signal.columns)

def strategy_inputs(sig, facts=None, earnings=None, bench=None, bars=None):
    """Everything the strategies rank on, as Date x Symbol frames, from saved data only: `sig` = signal_analysis.csv rows
    (SIG_COLS), `facts` = factor_history.csv (the daily snapshot of news sentiment and the company-report score; each day
    uses the latest snapshot on or before it), `bench` = benchmark_prices.csv (SPY, QQQ, sector ETFs), `bars` =
    forward_bars() (volume). "through" = the last day the bars cover (later days wait for the bars)."""
    sig = sig[sig["Symbol"].isin(sig.loc[sig["Strategy_Score"].notna(), "Symbol"].unique())]   # scored stocks (no QQQ)
    f = {col: _wide(sig, col) for col in SIG_COLS[2:] if col not in ("Regime_On", "final_trade")}
    dates, c, rs, tech = f["Close"].index, f["Close"], f["RS_Score"], f["Technical_Score"]
    live = (0.5 * tech + 0.5 * rs).where(f["Strategy_Score"].notna())   # the Oct 2 score (C6); live is RS only since Oct 13
    cols, eligible, earnings = c.columns, f["Strategy_Score"].notna(), be.load_earnings() if earnings is None else earnings
    snap = {}
    for col in ("SentimentScore", "Fundamental_Weight"):
        snap[col] = pd.DataFrame(index=dates, columns=cols, dtype=float)
        if facts is not None and len(facts):
            fa = facts.assign(bar_date=pd.to_datetime(facts["bar_date"])).sort_values("as_of")
            w = _wide(fa.drop_duplicates(["bar_date", "Symbol"], keep="last"), col, index="bar_date").reindex(columns=cols)
            snap[col] = w.reindex(w.index.union(dates)).ffill().reindex(dates)
    b = pd.DataFrame(index=dates) if bench is None else \
        bench.assign(Date=pd.to_datetime(bench["Date"])).set_index("Date").sort_index()
    b = b.reindex(b.index.union(dates)).ffill().reindex(dates)
    etf_of = {s: be.sector_mapping.sector_etf_for(s) for s in cols}
    by_etf = lambda x: pd.DataFrame({s: x[etf_of[s]] if etf_of[s] in x else np.nan for s in cols}, index=dates)
    bx = bar_features(forward_bars() if bars is None else bars, dates, cols)
    quant, vol, up = quant_score(f, bx["vwap_ratio"]), be.volatility(c), c > f["ma_200"]
    regime = sig.groupby("Date")["Regime_On"].first().reindex(dates).fillna(0).astype(bool)
    weekly = be.weekly_rebalance_days(dates, live=True)
    ret = lambda n: c / c.shift(n) - 1
    rs_w = lambda w: (be.relative_strength(pd.concat([c, b], axis=1), list(cols), eligible_w=eligible, windows=w)
                      if "SPY" in b else rs)                            # RS over other windows (the decision-day closes)
    calm = vol.le(vol.where(eligible).median(axis=1), axis=0)
    mom_12_1, etf_3m = (c.shift(21) / c.shift(252) - 1) * 100, by_etf(b / b.shift(63) - 1)
    scores = {
        "live": live, "now": f["Strategy_Score"], "tech": tech, "rs": rs, "rs_short": rs_w((10, 21, 63)), "rs_long": rs_w((63, 126, 252)),
        "tech30": 0.3 * tech + 0.7 * rs, "rs_calm": rs.where(calm | rs.isna(), -1.0),   # -1: not picked, sleeve kept
        "quant": quant - 50,                                  # > 0 = Quant Score above 50 (the copy's min score)
        "quant_rs": 100 * (0.5 * _pct(quant) + 0.5 * _pct(rs)),
        "sentiment": (100 * (0.8 * _pct(live) + 0.2 * _pct(snap["SentimentScore"].fillna(0.0)))).where(live > 0),
        "fundamental": snap["Fundamental_Weight"],
        "mom_12_1": mom_12_1,
        "mom_6": ret(126) * 100,
        "low_vol": (1 / vol).where(up),
        "dual": (ret(252) * 100).where(regime, axis=0),
        "mom_risk": ret(126) / (c.pct_change(fill_method=None).rolling(126).std() * np.sqrt(126)),
        "high_200": c / c.rolling(200).max(),          # 1 = the close is the highest close of the last 200 sessions
        "reversal": (-ret(5) * 100).where(up),
        "quality_mom": (100 * (0.5 * _pct(snap["Fundamental_Weight"]) + 0.5 * _pct(mom_12_1))),
        "drift": earnings_drift(c, b["SPY"] if "SPY" in b else pd.Series(1.0, index=dates), earnings),
        "sector_rot": live.where(etf_3m.rank(axis=1, method="dense", ascending=False).le(3)),   # top 3 ETF returns
        "breakout": bx["vol_jump"].where((c >= 0.98 * c.rolling(55).max()) & (bx["vol_jump"] >= 1.3))
                    .rolling(20, min_periods=1).max().where(c > f["ma_50"]),     # a breakout in the last 20 sessions
    }
    # Legacy combined_signal (Stock Analysis Legacy/main_signal_analysis.ipynb): missing news / fundamentals = that day's mean
    fill = lambda x: x.T.fillna(x.mean(axis=1)).T.fillna(0.0)
    legacy = (0.60 * tech + 0.25 * fill(snap["Fundamental_Weight"]).clip(-10, 10) * 10
              + 0.10 * (by_etf(b / b.shift(4) - 1) * 300).clip(-30, 30) / 30 * 100
              + 0.05 * fill(snap["SentimentScore"]).clip(-10, 10) * 10)
    held = legacy.gt(20).astype(float).where(legacy.gt(20) | legacy.lt(-25)).ffill().fillna(0.0).where(eligible, 0.0)
    atr = bx["atr_ratio"] * c
    after_earn = np.zeros(c.shape, bool)                  # the reaction day + 10 sessions after each report
    for j, _, r in earnings_events(dates, cols, earnings):
        after_earn[r:r + 11, j] = True
    ranked = live.where(eligible & (live > 0)).rank(axis=1, ascending=False, method="first")
    top30, top10 = ranked <= 30, ranked <= 10
    dip = c <= c.rolling(20).max() - 3 * atr                            # 3x ATR(14) below the 20-day high close
    dip_a, dip_10 = dip & top30, dip & top10                            # sell-on-signal stays top 30; the named dip buy is top 10
    dip_b = dip_a & ~after_earn
    sell_now = _wide(sig, "final_trade").reindex(index=dates, columns=cols).eq("SELL") | ~top30   # SELL or rank worse than 30
    soon = be.earnings_days_ahead(dates, list(cols), earnings, EARN_DAYS).notna()
    n = eligible.sum(axis=1)
    weights = {"atr_dip": atr_dip_weights(dip_10, live), "atr_dip_ex_earn": atr_dip_weights(dip_b, live),
               "atr_dip_signal": atr_dip_weights(dip_a, live, hold=None, sell=sell_now),"legacy": held.div(n, axis=0),
               "dip": dip_weights(c, f["ma_50"], eligible),
               "trend": _carry(((f["ma_50"] > f["ma_200"]) & eligible).astype(float).div(n, axis=0), weekly)}
    have = bx["vwap_ratio"].notna().any(axis=1)
    qqq = b["QQQ"] if "QQQ" in b else pd.Series(np.nan, index=dates)
    return {"close": c.ffill(), "scores": scores, "weights": weights, "eligible": eligible, "vol": vol, "regime": regime,
            "bench": b, "up": up,
            "qqq50": (qqq > qqq.rolling(50).mean()).reindex(dates).fillna(False),
            "weekly": weekly, "live_cal": be.winner_rebalance_days(dates, live=True), "earnings": earnings, "atr": atr,
            "dip_spare": dip_b & ~soon.reindex(index=dates, columns=cols).fillna(False), "open": bx["open_ratio"] * c,
            "soon": soon.reindex(index=dates, columns=cols).fillna(False),          # earnings within 5 days (E5)
            "monthly": pd.Series([be.next_sessions(d, 1)[0].month != d.month for d in dates], index=dates),
            "through": have[have].index.max() if have.any() else pd.Timestamp(0)}

def spare_cash(tgt, chk, inp):
    """paper_trade's Mon-Thu spare-cash rule on target weights: on a check day, the weight above the invested level
    (1, or the regime scale while QQQ is weak) beyond 1% buys the unheld top-10 stocks at their Friday-rule weight, then
    the rest is split equally over the stocks held since an earlier day that close at or above their entry close (the
    close of the day they were first held; none -> all held), skipping ones with earnings within 5 days or on do_not_buy;
    a stock at its cap (be.WINNER max_weight, a sleeve stock its sleeve cap) passes its share to the others. Held until Friday."""
    W, score = be.WINNER, inp["scores"]["now"]
    ok = inp["eligible"] & inp["vol"].reindex_like(inp["eligible"]).gt(0) & score.gt(W["min_score"])
    rank = be.deterministic_rank(score, ok, inp["scores"]["rs"]).reindex(columns=tgt.columns)
    no = (inp["soon"].reindex_like(tgt).fillna(False) | tgt.columns.isin(be.sector_mapping.do_not_buy)).to_numpy(bool)
    T, R, cols = tgt.to_numpy(float).copy(), rank.to_numpy(float), list(tgt.columns)
    sleeve = set(be.sector_mapping.defensive)
    cap = np.array([min(W["max_weight"] or 1.0, be.sector_mapping.defensive_cap or 1.0) if c in sleeve else (W["max_weight"] or 1.0)
                    for c in cols])
    reg, reb, ck = (x.reindex(tgt.index).fillna(False).to_numpy(bool) for x in (inp["regime"], inp["weekly"], chk))
    extra, top = np.zeros(len(cols)), np.zeros(len(cols))      # unheld buys / top-ups of rule holdings
    C = inp["close"].reindex(index=tgt.index, columns=cols).to_numpy(float)
    prev, entry = np.zeros(len(cols)), np.full(len(cols), np.nan)   # yesterday's weights; entry close of each holding
    for t in range(len(T)):
        extra = np.where(reb[t] | (T[t] > 0), 0.0, extra)    # Friday sells them; a rule buy takes the stock over
        top = np.where(reb[t] | (T[t] == 0), 0.0, top)       # Friday resets; a rule sale drops the top-up too
        if ck[t]:
            left = (1.0 if reg[t] else W["regime_scale"]) - T[t].sum() - extra.sum() - top.sum()
            if left >= DEPLOY_MIN_PCT:
                day = tgt.index[[t]]
                fri = be.rank_targets(score.loc[day, cols], inp["eligible"], inp["vol"], rebalance_days=pd.Series(True, index=day),
                                      tiebreak_w=inp["scores"]["rs"], **be.winner_rank_args(inp["regime"])).iloc[0].to_numpy(float)
                order = [j for j in np.argsort(np.nan_to_num(R[t], nan=1e9), kind="stable") if R[t, j] == R[t, j]]
                for j in order:                                                   # 1. unheld top 10 at their weights
                    if R[t, j] > DEPLOY_MAX_RANK or left < DEPLOY_MIN_PCT:
                        break
                    if T[t, j] == 0 and extra[j] == 0 and not no[t, j] and fri[j] > 0:
                        w = min(fri[j], cap[j], left)
                        extra[j], left = w, left - w
                have = lambda j: T[t, j] + extra[j] + top[j]
                got = set(np.flatnonzero(extra + top > 0))       # bought today (a new buy must reach 1%, a top-up of it not)
                held = sorted((j for j in range(len(cols)) if have(j) > 0 and prev[j] > 0 and not no[t, j] and C[t, j] > 0),
                              key=lambda j: np.nan_to_num(R[t, j], nan=1e9))      # 2. held since before today
                room = lambda j: cap[j] - have(j) > 1e-9 and (j in got or cap[j] - have(j) >= DEPLOY_MIN_PCT)
                group = [j for j in held if C[t, j] >= entry[j] and room(j)] or held   # at/above entry, else all held
                while left > 1e-9:
                    act = [j for j in group if room(j)]
                    if act and left / len(act) < DEPLOY_MIN_PCT:
                        act = [j for j in act if j in got] or act[:int(left // DEPLOY_MIN_PCT)]
                    if not act:
                        break
                    share = left / len(act)
                    for j in act:                                # a stock at its cap passes its share to the others
                        add = min(share, cap[j] - have(j))
                        if T[t, j] > 0:
                            top[j] += add
                        else:
                            extra[j] += add
                        left -= add
                        got.add(j)
        T[t] += extra + top
        entry = np.where((T[t] > 0) & (prev <= 0), C[t], entry)  # a new holding's entry = that day's close
        prev = T[t].copy()
    return pd.DataFrame(T, index=tgt.index, columns=cols)

def strategy_targets(cfg, inp):
    """Daily live target weights of one STRATEGIES entry (Date x Symbol). Ranked ones go through the engine's own selection
    code (be.winner_targets): only overrides are passed, WINNER itself is never changed. A "stop" strategy also leaves its
    open-sale prices in inp["open_sells"][name]."""
    if cfg.get("combo"):                               # forward_combos line: the simulated end-of-day weights
        return combo_weights(cfg, inp)
    if cfg.get("live"):                                # the current live rules (be.WINNER), not pinned
        tgt, chk = be.winner_targets(inp["scores"]["now"], inp["eligible"], inp["vol"], inp["regime"], inp["live_cal"],
                                     tiebreak_w=inp["scores"]["rs"], earnings=inp["earnings"],
                                     never_buy=be.sector_mapping.do_not_buy)
        tgt = spare_cash(tgt, chk, inp)
    elif cfg.get("weights"):
        tgt = inp["weights"][cfg["weights"]]
    elif cfg.get("stop") or cfg.get("replay"):          # the live rules replayed day by day + a stop / a mid-week change
        tgt, inp.setdefault("open_sells", {})[cfg["name"]] = atr_stop_targets(
            inp, **STOPS.get(cfg.get("stop"), {"k": np.inf}), **cfg.get("replay", {}))
    else:
        cal, tb = cfg.get("calendar", "mwf"), cfg.get("tiebreak", "rs")
        below, top = cfg.get("exit", (EXIT_BELOW, EXIT_TO_TOP))
        sleeve = list(be.sector_mapping.defensive) if cfg.get("sleeve") else ()     # C6 rules: no sleeve
        tgt, _ = be.winner_targets(inp["scores"][cfg.get("score", "live")], inp["eligible"], inp["vol"],
                                   inp[cfg.get("regime", "regime")], inp["weekly" if cal == "mwf" else cal],
                                   tiebreak_w=inp["scores"].get(tb), midweek=cfg.get("midweek", MIDWEEK) if cal == "mwf" else False,
                                   selection={"max_weight": MAX_WEIGHT, **PIN, **(cfg.get("select") or {}), "sleeve": sleeve},
                                   earnings_block_days=cfg.get("earnings", EARN_DAYS), earnings=inp["earnings"],
                                   exit_all_below=below, exit_to_top=top,
                                   never_buy=be.sector_mapping.do_not_buy if cfg.get("never_buy") else ())
        if cfg.get("reweight") == "risk_parity":
            tgt = risk_parity(tgt, inp["close"][tgt.columns], inp["weekly"])
        if cfg.get("reweight") == "rank_tier":
            tgt = rank_tier(tgt, inp["scores"][cfg.get("score", "live")][tgt.columns])
        if cfg.get("spare_dips"):                         # the cash the live rules leave unused buys ATR dips (10% each)
            tgt = tgt + atr_dip_weights(inp["dip_spare"][tgt.columns], inp["scores"]["live"], taken=(tgt > 0).to_numpy(),
                                        room=np.floor((1 - tgt.sum(axis=1)).to_numpy() * 10 + 1e-9).astype(int))
    cap = (be.WINNER.get("max_weight") or 1.0) if cfg.get("live") else MAX_WEIGHT      # the others keep their 20%
    return be.live_weights(tgt.clip(upper=cap)).reindex(columns=inp["close"].columns).fillna(0.0)

def combo_weights(cfg, inp, through=None):
    """Date x Symbol end-of-day weights of a forward_combos line (0 before its start)."""
    import forward_combos as fc
    c = cfg["combo"]
    sim = fc.simulate(inp, tuple(c["legs"]), c.get("start") or be.FORWARD_START, through=through)
    _, h = fc.line_rows(cfg["name"], sim, c.get("which", "total"))
    w = pd.DataFrame(0.0, index=inp["close"].index, columns=inp["close"].columns)
    if h:
        hp = pd.DataFrame(h).pivot_table(index="Date", columns="Symbol", values="Weight", aggfunc="sum")
        w.loc[hp.index, hp.columns.intersection(w.columns)] = hp[hp.columns.intersection(w.columns)].fillna(0.0)
    return w

def live_switch():
    """The day the live line becomes the 50/50 (be.WINNER "sleeves" on), else None (the old single-strategy simulation)."""
    return pd.Timestamp(LIVE_SWITCH) if be.WINNER.get("sleeves") else None

def migrate_names(*paths):
    """Saved rows of a renamed line (OLD_NAMES) take its new name; a file is rewritten only when it has old names."""
    for p in paths:
        df = _read(p)
        if df is not None and len(df) and df["Strategy"].isin(list(OLD_NAMES)).any():
            df.assign(Strategy=df["Strategy"].replace(OLD_NAMES)).to_csv(p, index=False)

def _live_combo_rows(inp, old, old_h, rows, hold, after, through):
    """The live line from the switch day: the pre-switch holdings marked to the switch close, sold (be.COST on the
    invested part), then the 50/50 halves (forward_combos, LIVE_LEGS) on that value. Rows for the days `after`."""
    import forward_combos as fc
    sw, close = live_switch(), inp["close"]
    before = close.index[close.index < sw]
    vals = pd.concat([x[x["Strategy"] == LIVE] for x in (old, pd.DataFrame(rows, columns=["Date", "Strategy", "Value", "Cash"]))
                      if x is not None and len(x)], ignore_index=True) if (old is not None or rows) else pd.DataFrame()
    base = 1.0
    if len(before) and len(vals):
        vals = vals.assign(Date=pd.to_datetime(vals["Date"]))
        p = before[-1]
        if (vals["Date"] == p).any():
            cash = float(vals.loc[vals["Date"] == p, "Cash"].iloc[-1])
            hs = pd.concat([x[(x["Strategy"] == LIVE)] for x in (old_h, pd.DataFrame(hold, columns=["Date", "Strategy", "Symbol", "Weight", "Shares", "Price"]))
                            if x is not None and len(x)], ignore_index=True)
            hs = hs[pd.to_datetime(hs["Date"]) == p] if len(hs) else hs
            px = close.loc[sw] if sw in close.index else pd.Series(dtype=float)
            inv = float(sum(r.Shares * (px.get(r.Symbol) if pd.notna(px.get(r.Symbol)) else r.Price) for r in hs.itertuples()))
            base = cash + inv - be.COST * inv
    legs = tuple((k, m, cal, float((be.WINNER.get("sleeves") or {}).get("lowvol" if k == "lowvol" else "sector", share)))
                 for k, m, cal, share in LIVE_LEGS)
    sim = fc.simulate(inp, legs, sw, through=through)
    r, h = fc.line_rows(LIVE, sim, "total", scale=base, normalize=False)
    keep = set(after)
    return [x for x in r if x["Date"] in keep], [x for x in h if x["Date"] in keep]

def rebalance_days(cfg, inp):
    """Days on which every holding is brought back to its target (unless within 1 point of it), like the live Friday
    rebalance: the strategy's calendar (Fridays / month ends); the trend rule on Fridays; the dip and threshold rules
    never (only the names they buy or sell trade). None when the inputs carry no calendar."""
    if cfg.get("live"):
        return inp.get("live_cal", inp.get("weekly"))     # the live calendar (monthly + the Oct 16 switch from Oct 16, 2026)
    key = {"trend": "weekly"}.get(cfg["weights"]) if cfg.get("weights") else \
        {"mwf": "weekly"}.get(cfg.get("calendar", "mwf"), cfg.get("calendar"))
    return inp.get(key) if key else None

def update_strategies(sig=None, facts=None, earnings=None, bench=None, bars=None, path=STRATEGIES_CSV,
                      hold_path=HOLDINGS_CSV, start=be.FORWARD_START):
    """Add every trading day after each strategy's last saved day (the first one: the start close, value 1.0 after its
    buys), up to the last day the volume bars cover. Saved days are never redone. Returns the number of rows added."""
    sig = _read(SIGNAL_CSV, usecols=SIG_COLS, parse_dates=["Date"]) if sig is None else sig
    facts = _read(FACTOR_CSV) if facts is None else facts
    bench = _read(BENCH_CSV) if bench is None else bench
    inp = strategy_inputs(sig, facts, earnings, bench, bars)
    close = inp["close"]
    days = close.index[close.index >= pd.Timestamp(start)]
    if inp.get("through") is not None and len(days) and days[-1] > inp["through"]:
        print(f"waiting for volume bars after {inp['through']:%b %-d} (days after it are added on a later run)")
        days = days[days <= inp["through"]]
    migrate_names(path, hold_path)
    old = _read(path, parse_dates=["Date"])
    old_h = _read(hold_path, parse_dates=["Date"])
    rows, hold, live_after = [], [], None
    for cfg in STRATEGIES:
        name = cfg["name"]
        mine = old[old["Strategy"] == name] if old is not None else pd.DataFrame(columns=["Date"])
        last = mine["Date"].max() if len(mine) else None
        todo = days if last is None else days[days > last]
        if not len(todo):
            continue
        if cfg.get("combo"):                                  # forward_combos: replayed from its start, new days appended
            import forward_combos as fc
            c = cfg["combo"]
            sim = fc.simulate(inp, tuple(c["legs"]), c.get("start") or start, through=todo[-1])
            r, h_ = fc.line_rows(name, sim, c.get("which", "total"))
            keep = set(todo)
            rows += [x for x in r if x["Date"] in keep]
            hold += [x for x in h_ if x["Date"] in keep]
            continue
        if cfg.get("live") and live_switch() is not None:     # the live 50/50 from the switch day (after the loop)
            if (todo >= live_switch()).any():
                live_after = todo[todo >= live_switch()]
            todo = todo[todo < live_switch()]
            if not len(todo):
                continue
        h = pd.DataFrame(columns=["Symbol", "Weight", "Shares", "Price"])
        if last is not None and old_h is not None:
            h = old_h[(old_h["Strategy"] == name) & (old_h["Date"] == last)]
        h = h.set_index("Symbol")
        cols = close.columns.union(h.index)
        zero = pd.Series(0.0, index=cols)
        cash, shares, prev, last_px = 1.0, zero, None, h["Price"].reindex(cols).astype(float)
        if last is not None:                                  # resume from the last saved day (no rows = all cash)
            cash = float(mine.loc[mine["Date"] == last, "Cash"].iloc[0])
            shares, prev = zero.add(h["Shares"], fill_value=0.0), zero.add(h["Weight"], fill_value=0.0)
        w = strategy_targets(cfg, inp).reindex(columns=cols).fillna(0.0)
        opens = inp.get("open_sells", {}).get(name, pd.DataFrame()).reindex(index=todo, columns=cols)
        full = rebalance_days(cfg, inp)
        if last is not None and last in w.index:
            # Trade only on a NEW decision: compare today's target with the target recomputed for the last saved day, not
            # with the saved one. A later pipeline run can rewrite past scores/weights (a live rule change, a data fix);
            # that must not make a strategy trade on a day with no decision of its own - the saved holdings stay until a
            # target really changes or the next rebalance (like the live account). Rules with no calendar (dip /
            # threshold) sell a saved holding they no longer target, since no rebalance would ever clear it.
            stale = (shares > 0) & (w.loc[last] <= 0) if full is None else pd.Series(False, index=cols)
            prev = w.loc[last].where(~stale, np.nan)
        full = pd.Series(False, index=todo) if full is None else full.reindex(todo).fillna(False).astype(bool)
        soon = inp.get("soon") if not cfg.get("weights") and cfg.get("earnings", "winner") is not None else None
        soon = (pd.DataFrame(False, index=todo, columns=cols) if soon is None else
                soon.reindex(index=todo, columns=cols).fillna(False).astype(bool))   # live: held, earnings soon: not topped up
        for d in todo:
            px = close.loc[d].reindex(cols).fillna(last_px)  # a stock gone from the list keeps its last saved price
            at_open = opens.loc[d][(shares > 0) & opens.loc[d].notna()]
            cash += (shares[at_open.index] * at_open * (1 - be.COST)).sum()   # stop sales at the open (0.1% cost)
            shares = shares.where(~shares.index.isin(at_open.index), 0.0)
            value = cash + (shares * px).sum()
            wd = w.loc[d]
            changed = ~np.isclose(wd, zero if prev is None else prev, rtol=0, atol=1e-12)
            if changed.any() or full[d]:   # like live: the names whose target changed (all of them on a rebalance day)
                held_w = shares * px / value if value > 0 else zero
                in_band = (wd > 0) & (shares > 0) & ((held_w - wd).abs() <= BAND)   # held and within 1 point: left alone
                move = (changed | (full[d] & ((wd > 0) | (shares > 0)))) & ~in_band \
                    & ~(soon.loc[d] & (shares > 0) & (wd > held_w))                  # earnings soon: not bought up
                for m in (move, move | (in_band & (held_w > wd))):  # no cash for the buys: band holds above target trimmed
                    cost = 0.0
                    for _ in range(6):        # 0.1% of the dollars actually traded (the buys are sized after the cost)
                        cost = be.COST * (wd * (value - cost) - shares * px).where(m, 0.0).abs().sum()
                    new = shares.where(~m, (wd * (value - cost) / px.where(px > 0)).fillna(0.0))
                    if value - cost - (new * px).sum() >= -1e-12:
                        break
                gap = value - cost - (new * px).sum()
                if gap < -1e-12:              # still short of cash: the buys shrink to fit (like live: no margin)
                    up = (new - shares).clip(lower=0)
                    new = new - up * min(1.0, -gap / ((1 + be.COST) * float((up * px).sum())))
                    cost = be.COST * float(((new - shares) * px).abs().sum())
                value, shares = value - cost, new
                cash, prev = value - (shares * px).sum(), wd
            if last is None and d == todo[0]:                 # 1.0 at the start close, after its buys
                shares, cash, value = shares / value, cash / value, 1.0
            last_px = px
            rows.append({"Date": d, "Strategy": name, "Value": value, "Cash": cash})
            on = (wd > 0) | (shares > 0)
            hold += [{"Date": d, "Strategy": name, "Symbol": s, "Weight": wd[s], "Shares": shares[s], "Price": px[s]}
                     for s in cols[on.to_numpy()]]
    if live_after is not None:
        r, h_ = _live_combo_rows(inp, old, old_h, rows, hold, live_after, live_after[-1])
        rows, hold = rows + r, hold + h_
    if rows:
        for p, new, prior in ((path, rows, old), (hold_path, hold, old_h)):
            out = pd.concat([prior, pd.DataFrame(new)], ignore_index=True) if prior is not None else pd.DataFrame(new)
            out = out.assign(Date=pd.to_datetime(out["Date"])).sort_values("Date", kind="stable")   # by day, registry order
            out.assign(Date=out["Date"].dt.strftime("%Y-%m-%d")).to_csv(p, index=False)
    return len(rows)

# ---------------------------------------------------------------------------------------------------- the live account
def _fill_rows(fills):
    """GET /account/activities FILL dicts -> DataFrame (Time in CT, naive), oldest first."""
    f = pd.DataFrame([{"Time": x.get("transaction_time"), "Symbol": x.get("symbol"), "Side": str(x.get("side", "")).lower(),
                       "Qty": float(x.get("qty") or 0), "Price": float(x.get("price") or 0)} for x in fills],
                     columns=["Time", "Symbol", "Side", "Qty", "Price"])
    f["Time"] = pd.to_datetime(f["Time"], utc=True).dt.tz_convert(CT).dt.tz_localize(None)
    return f.sort_values("Time", kind="stable").reset_index(drop=True)

def picks(fills, start=be.FORWARD_START):
    """Closed picks: a pick opens with a buy into a flat position on/after `start` and closes when the position is back to
    0; P/L = sell proceeds - buy cost. A position already held before `start` is not counted (its round trip began
    earlier)."""
    f, start, out, book = _fill_rows(fills), pd.Timestamp(start), [], {}
    for r in f.itertuples():
        b = book.setdefault(r.Symbol, {"qty": 0.0, "counted": False, "entry": None, "buy": 0.0, "sell": 0.0})
        if b["qty"] <= 1e-9 and r.Side == "buy":
            b.update(counted=r.Time >= start, entry=r.Time, buy=0.0, sell=0.0)
        b["qty"] += r.Qty if r.Side == "buy" else -r.Qty
        b["buy" if r.Side == "buy" else "sell"] += r.Qty * r.Price
        if b["qty"] <= 1e-9 and r.Side != "buy":
            if b["counted"]:
                out.append({"Symbol": r.Symbol, "Entry": b["entry"], "Exit": r.Time, "P/L $": b["sell"] - b["buy"],
                            "Return %": (b["sell"] / b["buy"] - 1) * 100 if b["buy"] else np.nan})
            b.update(qty=0.0, counted=False)
    return pd.DataFrame(out, columns=["Symbol", "Entry", "Exit", "P/L $", "Return %"])

def trade_cost(fills, orders, closes, start=be.FORWARD_START):
    """($ traded, $ cost) of the fills on/after `start` vs the decision price: buys (fill - decision price) x qty, sells
    (decision price - fill) x qty, so + = it cost money. The decision price is the stock's close on the As_Of day of the
    latest live_orders_log row with the same symbol and side sent before the fill; fills with no such row are left out."""
    f = _fill_rows(fills)
    f = f[f["Time"] >= pd.Timestamp(start)]
    if f.empty or orders is None or orders.empty:
        return 0.0, 0.0
    o = orders.assign(Sent=pd.to_datetime(orders["Submitted_At_CT"], errors="coerce"),
                      Side=orders["Side"].astype(str).str.lower(), As_Of=pd.to_datetime(orders["As_Of"], errors="coerce"))
    traded = cost = 0.0
    for r in f.itertuples():
        m = o[(o["Symbol"] == r.Symbol) & (o["Side"] == r.Side) & (o["Sent"] <= r.Time)]
        ref = closes.get((m["As_Of"].iloc[-1], r.Symbol)) if len(m) else None
        if ref is None or not ref == ref:
            continue
        traded += r.Qty * r.Price
        cost += (r.Price - ref) * r.Qty * (1 if r.Side == "buy" else -1)
    return traded, cost

def _read(path, **kw):
    """A saved CSV (None if missing). round_trip parsing: a rewrite keeps every saved number exactly (the default parser
    can change the last digit, so re-saving would alter old rows)."""
    return pd.read_csv(path, float_precision="round_trip", **kw) if os.path.exists(path) else None

def record(account=None, now=None, path=DAILY_CSV):
    """Save today's row (only on an NYSE session; replaces a row already saved today). Returns the row, or None."""
    now = (now or datetime.now(CT)).astimezone(CT)
    if not be.is_session(now.date()) or now.date() < pd.Timestamp(be.FORWARD_START).date():
        print(f"idle: {now:%a %b %-d} is not a trading day on/after the forward-test start")
        return None
    import alpaca_paper as ap
    account = account or ap.PaperAccount()
    s = account.account_summary()
    at = pd.Timestamp.now(tz="UTC")             # deposits booked after this balance read are not in it yet
    fills = account.fills()
    flows = [f for f in account.cash_flows() if ap._flow_time(f) <= at]
    # A deposit booked today (Alpaca: about 4:15 PM CT, after the close) counts from the next session, so a late run's row
    # is the account after the close, before that deposit (the row's equity and net deposits both leave it out).
    late = sum(f["amount"] for f in flows if str(f["date"])[:10] == f"{now:%Y-%m-%d}")
    sig = _read(SIGNAL_CSV, usecols=["Date", "Symbol", "Close"], parse_dates=["Date"])
    closes = {} if sig is None else dict(zip(zip(sig["Date"], sig["Symbol"]), sig["Close"]))
    pk = picks(fills)
    traded, cost = trade_cost(fills, _read(ORDERS_CSV), closes)
    row = {"Date": f"{now:%Y-%m-%d}", "Time_CT": f"{now:%H:%M}", "Equity": round(s["Equity"] - late, 2),
           "Cash": round(s["Cash"] - late, 2), "Net_Deposits": round(sum(f["amount"] for f in flows) - late, 2),
           "Positions": len(account.position_dicts()),
           "Closed_Picks": len(pk), "Winning_Picks": int((pk["P/L $"] > 0).sum()), "Traded_USD": round(traded, 2),
           "Cost_USD": round(cost, 2)}
    old = _read(path, dtype={"Date": str})
    new = pd.DataFrame([row])
    if old is not None and (old["Date"] != row["Date"]).any():
        new = pd.concat([old[old["Date"] != row["Date"]], new], ignore_index=True)
    new[DAILY_COLS].to_csv(path, index=False)
    return row

# ---------------------------------------------------------------------------------------------------- leaderboard
def _stats(v):
    """Total return %, daily return % (last day vs the day before), max drawdown %, finished weeks and each week's
    return % (week end to week end: the last session of each week, from the start close; list, week 1 first; the last
    one is the unfinished week so far when the latest day is not a week end)."""
    v = v.dropna().sort_index()
    if len(v) < 1:
        return np.nan, np.nan, np.nan, 0, []
    ends = v[be.weekly_rebalance_days(v.index, live=True).to_numpy() | (v.index == v.index[0])]
    weekly, now = ends.pct_change().dropna(), [] if ends.index[-1] == v.index[-1] else [v.iloc[-1] / ends.iloc[-1] - 1]
    daily = (v.iloc[-1] / v.iloc[-2] - 1) * 100 if len(v) >= 2 else np.nan
    return ((v.iloc[-1] / v.iloc[0] - 1) * 100, daily, (v / v.cummax() - 1).min() * 100, len(weekly),
            [x * 100 for x in list(weekly) + now])

def leaderboard(daily=None, bench=None, values=None, start=be.FORWARD_START):
    """One row per strategy (STRATEGIES order) + the real account (time-weighted, alpaca_paper.account_twr_index:
    deposits and withdrawals are not returns) + QQQ / SPY (closes, comparison only), sorted by RANK_RULE. Rank counts
    strategies only; the account and QQQ / SPY show where they would sit."""
    daily = _read(DAILY_CSV, parse_dates=["Date"]) if daily is None else daily
    bench = _read(BENCH_CSV, parse_dates=["Date"]) if bench is None else bench
    values = _read(STRATEGIES_CSV, parse_dates=["Date"]) if values is None else values
    if values is not None and len(values):
        values = values.assign(Strategy=values["Strategy"].replace(OLD_NAMES))
    start, series = pd.Timestamp(start), []

    def since(df):
        return df.assign(Date=pd.to_datetime(df["Date"])).set_index("Date").sort_index().loc[start:]
    for cfg in STRATEGIES:
        v = since(values[values["Strategy"] == cfg["name"]])["Value"] if values is not None else pd.Series(dtype=float)
        series.append((cfg["name"] + (LIVE_MARK if cfg["name"] == MARKED else ""), cfg.get("ranked", True), v))
    if daily is not None and len(daily):
        d = since(daily)
        if len(d):
            import alpaca_paper as ap
            series.append((ACCOUNT, False, ap.account_twr_index(d.reset_index()) * float(d["Equity"].iloc[0])))
    if bench is not None:
        b = since(bench)
        series += [(f"{s} (comparison)", False, b[s]) for s in ("QQQ", "SPY") if s in b]
    cols = ["Total return %", "Daily return %", "Max drawdown %"]
    stats = [(n, r, _stats(v)) for n, r, v in series]
    done = max((x[3] for *_, x in stats), default=0)                    # finished weeks; the next one is in progress
    so_far = any(len(x[4]) > x[3] for *_, x in stats)
    weeks = [f"Week {k}" + (" (so far)" if so_far and k == done + 1 else "") + " %" for k in range(1, WEEKS_TO_WIN + 1)]
    t = pd.DataFrame([{"Strategy": n, "ranked": r, **dict(zip(cols, x)), **dict(zip(weeks, x[4]))} for n, r, x in stats],
                     columns=["Strategy", "ranked"] + cols + weeks)        # Week k % = the k-th Friday-to-Friday week
    t = t.sort_values(["Total return %", "Max drawdown %"], ascending=False, na_position="last", kind="stable")   # RANK_RULE
    ok = t["ranked"] & t["Total return %"].notna()
    t["Rank"] = ok.cumsum().where(ok).astype("Int64")
    return t[["Rank", "Strategy"] + cols[:2] + weeks + cols[2:]].reset_index(drop=True)

def verdict(board):
    """One plain line: too early / the winner / no winner (RANK_RULE)."""
    by = board.set_index(board["Strategy"].str.replace(LIVE_MARK, "", regex=False))
    weeks = int(by.loc[LIVE, [c for c in by.columns if c.startswith("Week ") and "so far" not in c]].notna().sum()) \
        if LIVE in by.index else 0                                    # finished weeks of the live row
    if weeks < WEEKS_TO_WIN:
        return f"Week {weeks} of {WEEKS_TO_WIN}: too early to name a winner."
    top = by[board["Rank"].notna().to_numpy()].index[0]
    bar = max(by.at[LIVE, "Total return %"], by["Total return %"].get("QQQ (comparison)", -np.inf))
    if top == LIVE:
        return f"After {weeks} weeks the live rules rank first: keep them."
    if by.at[top, "Total return %"] > bar:
        return f"After {weeks} weeks the winner is {top}: it ranks first and beats {LIVE} and QQQ."
    return f"After {weeks} weeks no winner: {top} ranks first but does not beat both {LIVE} and QQQ. Keep the live rules."

DECISION_DATE = "2026-12-24"   # the week-12 checkpoint (Thu: Fri Dec 25 is a holiday): stay only if the live line beats SPY
LIVE_HALVES = ("Live half: Low vol monthly", "Live half: Sector rotation weekly (63d)")
SPY_RULE = "We must beat SPY to stay in the current strategy, otherwise switch to SPY (Chirag, Oct 15, 2026)."

def provisional(values, bench=None, hold_path=HOLDINGS_CSV, bars_path=FWD_BARS):
    """Display only, never saved. A trading day is saved by the daily run's after-close step (from 3:05 PM CT) once it
    has signal rows (the 2:30 PM decision run Mon-Fri, or the step's own refresh on a non-decision day, which runs first); until
    then, or when a run failed, the board would show the previous day's numbers again. This marks each strategy's saved
    holdings (shares + cash of its last saved day, no trades) to the close of every later day the saved daily bars cover
    (complete days only: a bar is taken from 3:05 PM CT), and QQQ / SPY to
    the same closes. Returns (values, bench) with those rows added; values rows carry Provisional = True."""
    if values is None or values.empty or not os.path.exists(bars_path):
        return values, bench
    v = values.assign(Date=pd.to_datetime(values["Date"]))
    last = v["Date"].max()
    b = be.apply_history_start(pd.read_pickle(bars_path))
    b = be.keep_decision_bars(b.sort_values(["Symbol", "Date"]).reset_index(drop=True), now=_NO_SAVE)
    close = b.pivot_table(index="Date", columns="Symbol", values="Close")
    close = close[close.index > last]
    h = _read(hold_path, parse_dates=["Date"])
    if close.empty or h is None:
        return values, bench
    h, rows = h[h["Date"] == last], []
    for name, g in v[v["Date"] == last].groupby("Strategy", sort=False):
        mine, cash = h[h["Strategy"] == name].set_index("Symbol"), float(g["Cash"].iloc[0])
        for d, px in close.iterrows():
            p = px.reindex(mine.index).fillna(mine["Price"])     # no bar that day: its last saved price
            rows.append({"Date": d, "Strategy": name, "Value": cash + float((mine["Shares"] * p).sum()), "Cash": cash,
                         "Provisional": True})
    out = pd.concat([v.assign(Provisional=False), pd.DataFrame(rows)], ignore_index=True) if rows else v
    if bench is not None and len(bench):
        bb = bench.assign(Date=pd.to_datetime(bench["Date"]))
        extra = close[[s for s in ("QQQ", "SPY") if s in close]]
        extra = extra[extra.index > bb["Date"].max()]
        if len(extra.columns) and len(extra):
            bench = pd.concat([bb, extra.rename_axis("Date").reset_index()], ignore_index=True)
    return out, bench

def holdings(path=HOLDINGS_CSV):
    """Strategy -> 'NVDA 14.2%, ...' of its latest saved day (target weights, largest first)."""
    h = _read(path, parse_dates=["Date"])
    if h is None or h.empty:
        return {}
    h = h.assign(Strategy=h["Strategy"].replace(OLD_NAMES))
    h = h[h["Date"] == h.groupby("Strategy")["Date"].transform("max")]
    h = h[h["Weight"] > 0].sort_values(["Strategy", "Weight"], ascending=[True, False])
    return {s: ", ".join(f"{r.Symbol} {r.Weight:.1%}" for r in g.itertuples()) for s, g in h.groupby("Strategy")}

def consensus(board, hold, ranks=None, n_strategies=5, n_stocks=10):
    """Display only (never saved, never traded): the stocks most held by the best forward-test strategies.

    board = leaderboard() as the dashboard shows it (incl. provisional rows); hold = saved holdings rows (Date, Strategy,
    Symbol, Weight); ranks = {symbol: latest Strategy_Rank} (tie-break). The strategies are taken by the most recent
    week's return: the board's last Week column (the unfinished week so far, else the latest finished week; Chirag,
    Oct 15), total return since the start before any week column; a tie: smaller drawdown. Not RANK_RULE (the table). Twins (same stocks at the same target weights on the holdings day) count once: the
    better-placed one stays. Each strategy's holdings = its latest saved day (what the board's values come from; a
    provisional close keeps them, so nothing past the shown day is used). Stocks: held (target weight > 0) by the most of those strategies; ties -> larger summed weight, then the
    better latest rank. Returns dict(day, by, strategies=[(name, [twins])], stocks=DataFrame(Symbol, Count, Weight,
    Rank), by = the column used); None when there is nothing to show."""
    if board is None or board.empty or hold is None or hold.empty:
        return None
    names = {c["name"] for c in STRATEGIES}
    b = board.assign(Strategy=board["Strategy"].str.replace(LIVE_MARK, "", regex=False))
    b = b[b["Strategy"].isin(names)]
    wk = [c for c in b.columns if str(c).startswith("Week ") and ("so far" in c or b[c].notna().any())]
    by = ([c for c in wk if "so far" in c] or wk or ["Total return %"])[-1]
    b = b.sort_values([by, "Max drawdown %"], ascending=False, na_position="last", kind="stable")
    h = hold.assign(Date=pd.to_datetime(hold["Date"]), Strategy=hold["Strategy"].replace(OLD_NAMES))
    h = h[h["Strategy"].isin(names)]
    if h.empty:
        return None
    h = h[h["Date"] == h.groupby("Strategy")["Date"].transform("max")]
    day = h["Date"].max()
    pos = {s: g.loc[g["Weight"] > 0].set_index("Symbol")["Weight"] for s, g in h.groupby("Strategy")}
    picked, seen = [], {}
    for name in b["Strategy"]:
        w = pos.get(name, pd.Series(dtype=float))
        key = tuple(sorted((s, round(float(x), 4)) for s, x in w.items()))
        if key in seen:                                   # a twin of a better-placed strategy
            seen[key][1].append(name)
            continue
        if len(picked) == n_strategies:
            continue
        seen[key] = [name, []]
        picked.append(seen[key])
    rows = {}
    for name, _ in picked:
        for s, x in pos.get(name, pd.Series(dtype=float)).items():
            c, sw = rows.get(s, (0, 0.0))
            rows[s] = (c + 1, sw + float(x))
    ranks = ranks or {}
    st = pd.DataFrame([{"Symbol": s, "Count": c, "Weight": sw, "Rank": ranks.get(s, np.nan)} for s, (c, sw) in rows.items()],
                      columns=["Symbol", "Count", "Weight", "Rank"])
    st = st.assign(_r=st["Rank"].fillna(1e9)).sort_values(["Count", "Weight", "_r", "Symbol"], ascending=[False, False, True, True])
    return {"day": day, "by": by, "strategies": [(nm, tw) for nm, tw in picked],
            "stocks": st.drop(columns="_r").head(n_stocks).reset_index(drop=True)}

def _log_problem(message, error):
    """A failed forward-test update -> one Reports/run_log.csv row (the dashboard's run log), so it is not only in the
    launchd log. Never raises."""
    print(f"{message} ({type(error).__name__}: {error})")
    try:
        import run_all
        run_all.log_event("Forward test", "failed", "no", message, f"{type(error).__name__}: {error}")
    except Exception as e:
        print(f"run log not written ({e})")

def main(argv=None):
    """Update the paper strategies (and with --record save today's account row), then print the leaderboard. Returns the
    exit code: 1 when a part failed (each failure is one run-log row; a saved day is never redone or lost)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", action="store_true", help="also save today's account row (reads the live account, GET only)")
    a = p.parse_args(argv)
    rc = 0
    try:
        print(f"strategies: {update_strategies()} new row(s) -> {os.path.relpath(STRATEGIES_CSV, ROOT)}")
    except Exception as e:  # never stops the account row
        _log_problem("The paper strategies were not updated today; the next run adds the missing days.", e)
        rc = 1
    if a.record:
        try:
            row = record()
        except Exception as e:  # Alpaca down / keys missing: the next trading day's run still adds its own row
            _log_problem("Today's account row for the forward test was not saved (the Alpaca account read failed). "
                         "No money moved.", e)
            row, rc = None, 1
        if row:
            print("saved", {k: row[k] for k in ("Date", "Time_CT", "Positions", "Closed_Picks")}, "->",
                  os.path.relpath(DAILY_CSV, ROOT))
    values, bench = provisional(_read(STRATEGIES_CSV, parse_dates=["Date"]), _read(BENCH_CSV, parse_dates=["Date"]))
    board = leaderboard(bench=bench, values=values)
    if values is not None and "Provisional" in values and values["Provisional"].fillna(False).astype(bool).any():
        print(f"latest day {values['Date'].max():%a %b %-d}: saved holdings marked to the close (provisional, not saved)")
    print(board.round(2).to_string(index=False))
    print(verdict(board))
    return rc

if __name__ == "__main__":
    sys.exit(main())
