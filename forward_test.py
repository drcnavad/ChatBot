"""Forward test from backtest_engine.FORWARD_START (Fri Oct 2, 2026), counted only from that close on:
  1. the real Alpaca account (read-only GETs through alpaca_paper, never orders) -> Reports/forward_test_daily.csv, one
     row per trading day: equity, cash, net deposits (a deposit booked after the close, ~4:15 PM CT, counts from the next
     row), positions, closed / winning picks, $ traded and its cost vs the decision price;
  2. STRATEGIES: the paper-only strategies (the current live rules, the pinned Oct 2 rules and variants, the Oct 15 50/50
     and low-volatility lines of forward_combos; never traded), built from the saved
     Reports files (signal_analysis, factor_history, earnings_date, benchmark_prices) + Alpaca daily volume
     (forward_bars) -> Reports/forward_strategies.csv (Value, 1.0 at the start; Cash) + forward_strategies_holdings.csv.
leaderboard(): every strategy + the account + QQQ / SPY, ranked by total return, with one column per week.
Same accounting for all: decide at the day's close (the 2:30 PM bar on decision days), trade only names whose target
changed (all back to target on the strategy's rebalance day, 1-point band, earnings rule), be.COST per side, equal weight per
stock (1/n), 99% invested, buys never exceed the cash (no margin), cash earns 0. Saved days are never redone; missed days are
caught up from the saved bars. On Oct 15, 2026 all but the live one were recomputed from Oct 2 (Mon-Thu checks, 11 new);
later on Oct 15, 19 more (forward_combos: two independent halves, swap-only halves, the low-volatility keep / Friday
falling rules) - the two live halves start at the Oct 16 close, the rest are computed from Oct 2 like the others. The
old live line "Live 50/50 combined (no overlap)" (was "Live rules (C6)") was removed Oct 20, 2026 (near-duplicate of the
Monday 50/50, which is now LIVE).

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
import db  # noqa: E402  tables live in SQLite (db.py), not CSV files

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
# The paper strategies' base selection = the live WINNER of Oct 15, 2026 (pinned when live moved to the monthly low-volatility
# rules on Oct 16): no sector limit, no rank gate, earnings rule E5. Oct 21, 2026: equal weight everywhere (1/n each).
PIN = {"n": 10, "sector_cap": 1.0, "max_pick_rank": None, "cap_soft": False, "sleeve": ()}
EARN_DAYS = 5
BAND = be.WINNER.get("rebalance_band") or 0.0   # a holding within 1 point of its new target is not traded (live band)
WEEKS_TO_WIN = 12
MON5 = "Live 50/50, top 5, sector on Mondays"   # Oct 21: the top-5 rules with the sector half on Mondays (live Oct 21 morning)
MON = "Live 50/50, top 10 each half"    # the live rules Oct 19-25 (sector half on Mondays, 10 stocks per half)
TOP5 = "Old live: Low vol + Sector rotation (Oct 16-26)"   # the live rules Oct 21-26, 2026 (was "Current Strategy (Live from Oct End)")
# (Chirag, Oct 26, 2026, final) the live rules from the 2:30 PM CT run of Tue Oct 27: 50% "RS short-term (weekly + Mon–Thu sells) (10/21/63 days)" + 50% "RS standard (Fridays only)", two independent halves (sleeves.py) = the average of the two lines from Oct 2
NOW_HALVES = ("RS short-term (weekly + Mon–Thu sells)", "RS standard (Fridays only)")
LIVE_NOW = "Live: RS short-term + RS Fridays"
MARKED = LIVE_NOW
LIVE = LIVE_NOW     # the line the verdict and the dashboard compare against
# Removed Oct 20, 2026 (Chirag): "Live 50/50 combined (no overlap)" (was "Live rules (C6)"), a near-duplicate of MON;
# its saved rows are dropped (REMOVED, purge_removed).
# Removed Oct 21, 2026 (Chirag): near-duplicates (MACD cross + TP20 = MACD cross so far; breakouts cut to 2 and 4 days;
# Friday only ~ Oct 2 rules without Mon-Thu checks, RS standard (Fridays only) covers that).
REMOVED = {"Short-term reversal", "12-1 month momentum", "RSI pullback in uptrend", "Worst 3 sectors 3M, +10%/-3%",
           "Breakout with volume", "Relative strength only", "MACD cross, weekly", "MA10 & MA30 x MA50",
           "Live 50/50 combined (no overlap)", "Live rules (C6)", "MACD cross + TP20", "Friday only",
           "Live 50/50, sector rotation on Fridays", "50/50: Low vol + SR weekly 63d (live combo from Oct 2)",   # old top-10 Friday rows
           "200-day breakout, 1 day above, exit 2% below", "200-day breakout, 3 days above, exit 2% below",
           "200-day breakout, 5 days above, exit 2% below",
           "Top 20 rank-weighted (70/30)"}   # Oct 21, 2026: no rank / volatility weights any more
# Renamed lines: saved rows take the new name (migrate_names, and the dashboard reads through this map).
# A rule that changed (ATR dip buy, top 30 -> top 10) is a new name on purpose, so the old results are not relabeled.
OLD_NAMES = {
    "Worst sector 14d, +10%/-3%": "Worst 3 sectors 14d, +10%/-3%",   # Oct 21: 3 worst sectors, top 10 (rebuilt from Oct 2)
    "Worst sector 1M, +10%/-3%": "Worst 3 sectors 1M, +10%/-3%",
    "Worst 2 sectors 14d, +10%/-3%": "Worst 3 sectors 14d, +10%/-3%",
    "Worst 2 sectors 1M, +10%/-3%": "Worst 3 sectors 1M, +10%/-3%",
    "Quant Score": "Quant Score top 10",
    "Now: calmer half": "Now: RS, calmer half of the list",
    "Now: RS 10/21/63 days": "RS short-term (weekly + Mon–Thu sells)",
    "Now: RS 63/126/252 days": "Now: longer RS (63/126/252 days)",
    "Now: 30% technical": "Now: 30% technical + 70% RS",
    "Now: exit below 15": "Now: Mon-Thu sell below rank 15",
    "Restored RS + sleeve, Mon-Thu checks (Oct 15 rules)": "Oct 15 rules (RS + sleeve, Mon-Thu)",
    "Live 50/50, sector rotation on Mondays": MON,   # Oct 21: the top-10 live line, kept as the previous rules
    "Live 50/50, top 5 each half": MON5,             # Oct 21: the top-5 Monday line
    "Live 50/50, top 5, sector on Fridays": TOP5,    # Oct 21 (Chirag): the live rules = top 5, sector half on Fridays
    "Now: shorter RS (10/21/63 days)": "RS short-term (weekly + Mon–Thu sells)",   # Oct 26 (Chirag): the live halves' names
    "Now: Friday only": "RS standard (Fridays only)",
    "Current Strategy (Live: Shorter RS + Friday only)": "Live: RS short-term + RS Fridays",
    "Current Strategy (Live from Oct End)": TOP5,    # Oct 26 (Chirag): replaced by LIVE_NOW, kept under its old-rules name
    "Live half: Sector rotation weekly (63d)": "Live half: Sector rotation Mondays (63d)",   # Oct 21: it rebalances Mondays
}
# the live 50/50 (be.WINNER "sleeves", from the Oct 16, 2026 switch): forward_combos legs, low-volatility half first
# "+mb" = the live manual buy blocks too (Reports/manual_block.json, e.g. KO; Oct 21, 2026) on the live lines only
PREV_LEGS = (("lowvol+mb", "full", "month", 0.5), ("sr63+mb", "full", "weekstart", 0.5))   # top 10 per half (Oct 19-25)
LIVE_LEGS = (("lowvol@5+mb", "full", "month", 0.5), ("sr63@5+mb", "full", "week", 0.5))  # top 5 per half, sector half Fridays (Oct 21)
MON5_LEGS = (LIVE_LEGS[0], ("sr63@5+mb", "full", "weekstart", 0.5))   # the same with the sector half on Mondays
LV = ("lowvol", "full", "month", 0.5)   # the low vol half of the other 50/50 lines (no manual blocks)
LIVE_SWITCH = "2026-10-16"
C6 = "Oct 2 rules (pinned)" # the rules the forward test started with (pinned: MIDWEEK / EXIT_BELOW / EXIT_TO_TOP, 50/50 score)
# paper_trade's Mon-Thu spare-cash rule (forward_test never imports paper_trade; tests check these equal its constants)
LIVE_MARK = " · Current strategy (live)"
ACCOUNT = "My Alpaca account"

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
# Mon-Thu swaps and exits ("mwf": MIDWEEK / EXIT_BELOW / EXIT_TO_TOP below); Friday = pure top-10 by rank (no sector / no rank-20 gate), equal
# weights, half size when QQQ is below its 200-day average, no buys 5 days before earnings. "select" changes rank_targets
# keys (n, sector_cap, regime, ...), "earnings" None drops the earnings skip, "calendar" is "mwf" / "weekly"
# (Friday only) / "monthly" (last session of the month). "weights" = a rule that is not a ranking (its own weights below);
# "stop" = the live rules + an ATR stop around earnings (STOPS);
# "midweek" = other Mon-Thu swap settings;
# "replay" = the live rules replayed day by day with a changed Mon-Thu exit (atr_stop_targets).
#
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
EQ = PLAIN                                                # (equal weight is now the only sizing)
SIMPLE = dict(select=PLAIN, earnings=None)                # a plain top-10 strategy: no rank-20 limit, no earnings skip
STRATEGIES = [
    dict(name=C6, group="blend",
         rule="What the bot traded Oct 2-7 (pinned): 50% technical + 50% relative strength, top 10, Fri rebalance + "
              "Mon-Thu top-3 swap (below rank 15) and exit below rank 30 (checks Mon/Wed until Oct 15)."),
    dict(name="C6, 4 per sector, no earnings skip", select=PLAIN, earnings=None, group="blend",
         rule="The Oct 2 rules (half technical, half relative strength, top 10, Friday rebalance and Mon-Thu checks) with two "
              "changes: at most 4 stocks from any one sector, and an earnings date never blocks a buy. It still ranks the whole "
              "list. Sector rotation is a different line: it first keeps only the 3 sectors whose ETF has the best 63-day return."),
    dict(name="Technical score only", score="tech", group="technical",
         rule="Oct 2 rules ranking on the technical score alone (moving averages, RSI, MACD). Top 10."),
    dict(name="Quant Score top 10", score="quant", tiebreak=None, group="quant",
         rule="Quant Score only, 10 stocks. The score is 0-100 from the trend, the moving averages, the 20-day VWAP, RSI, MACD "
              "and Bollinger bands (a missing part counts as 50). It holds the 10 highest scores above 50, on the Oct 2 calendar "
              "(Friday rebalance and Mon-Thu checks), equal weights. Quant Score top 5 is the same score with 5 stocks."),
    dict(name="Quant Score top 5", score="quant", tiebreak=None, select={"n": 5}, group="quant",
         rule="The same Quant Score as Quant Score top 10 (the 0-100 trend, moving-average, VWAP, RSI, MACD and Bollinger score, "
              "above 50), holding 5 stocks instead of 10."),
    dict(name="Quant + relative strength", score="quant_rs", group="quant",
         rule="Oct 2 rules ranking on a 50/50 blend of Quant Score rank and relative strength rank. Top 10."),
    dict(name="Company fundamentals", score="fundamental", calendar="monthly", select=EQ, earnings=None, group="fundamental",
         rule="Top 10 by the company-report score (Fundamental_Weight), equal weight, monthly."),
    dict(name="6-month momentum", score="mom_6", calendar="weekly", **SIMPLE, group="momentum",
         rule="Top 10 by the plain return over the last 126 trading days (about 6 months), including the last month. Every Friday, "
              "equal weights. 12-1 month momentum leaves out the last month and rebalances once a month."),
    dict(name="Low volatility in uptrend", score="low_vol", calendar="monthly", select=EQ, earnings=None, group="lowvol",
         rule="The 10 calmest stocks by 63-day volatility, and only those whose close is above the 200-day average. "
              "How far the price rose is not the rank. Equal weight, monthly. Volatility-adjusted momentum is a different "
              "line: it ranks 6-month return divided by 6-month volatility, every Friday, with no 200-day filter."),
    dict(name="Buy the dip", weights="dip", group="event",
         rule="Buy a stock at the close when it is 8% or more below its 50-day average; sell when it is back at the average or "
              "after 30 days; 10% each, at most 10, deepest first, checked daily."),
    dict(name="Monthly", calendar="monthly", group="blend",
         rule="The Oct 2 rules, rebalanced on the last trading day of each month only."),
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
              "high; 1 means the stock is at the high). Equal weight, monthly."),
    dict(name="Quality + momentum", score="quality_mom", calendar="monthly", **SIMPLE, group="fundamental",
         rule="Top 10 by half the company-report score (Fundamental_Weight, the quality rank) and half the 12-1 month momentum "
              "rank (return from 12 months ago to 1 month ago). Monthly, equal weights."),
    dict(name="Earnings drift", score="drift", calendar="weekly", **SIMPLE, group="event",
         rule="Stocks whose last earnings reaction beat SPY, biggest jump first. A morning report is judged that same day; "
              "an afternoon report is judged the next session. That one-day jump versus SPY is kept for 60 trading days. "
              "Top 10, every Friday. There is no EPS-surprise feed, so the price jump stands in for the beat."),
    dict(name="Sector rotation", score="sector_rot", calendar="weekly", **SIMPLE, group="sector",
         rule="The 3 sector ETFs with the best 63-day (about 3-month) return, then the best Oct 2 scores (half technical, "
              "half relative strength) inside only those sectors. Top 10, every Friday, no Mon-Thu checks, and an earnings date "
              "never blocks a buy. The C6 4-per-sector line is different: it ranks the whole list and can hold any sector."),
    dict(name="ATR dip buy (top 10)", weights="atr_dip", group="event",
         rule="Same dip as ATR dip buy, sell on signal, with a shorter list and a fixed exit. Buy only a top-10 name on the "
              "Oct 2 score (half technical, half relative strength, score above 0) when the close is 3 times ATR(14) or more "
              "below the highest close of the last 20 days. 10% each, at most 10, best rank first. Sold after 10 trading days."),
    dict(name="ATR dip buy, sell on signal", weights="atr_dip_signal", group="event",
         rule="Same dip as ATR dip buy (top 10): the close is 3 times ATR(14) or more below the highest close of the last "
              "20 days. This line uses the top 30 of the Oct 2 score, and it does not sell after 10 days. It sells when "
              "signal analysis says SELL (score below 0) or the rank falls below 30. 10% each, at most 10, best rank first."),
    # ---- the live rules of Oct 15, 2026 (pinned) with one change each (added Oct 15, computed from Oct 2)
    dict(name="RS standard (Fridays only)", **NOW, calendar="weekly", group="rs",
         rule="The Oct 15 rules (relative strength, top 10, the five defensive stocks) with the Friday rebalance only. "
              "Mon-Thu does not sell or replace anything."),
    dict(name="Now: top 15", **NOW, select={"n": 15}, group="rs",
         rule="The Oct 15 rules (relative strength only, Friday rebalance, Mon-Thu sell below rank 20) with 15 ranked stocks "
              "plus the five defensive stocks."),
    dict(name="RS short-term (weekly + Mon–Thu sells)", **{**NOW, "score": "rs_short"}, group="rs",
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
    dict(name="Live half: Low vol monthly", combo=dict(legs=PREV_LEGS, start=LIVE_SWITCH, which=0), ranked=False, group="lowvol",
         rule="The live account's low-volatility half on its own (from the Oct 16 close, 1.0 then): the 10 calmest stocks "
              "above their 200-day average, max 4 per sector, equal weight, rebalanced on Oct 16 and then the last trading "
              "day of each month; do_not_buy never bought. Not ranked (starts Oct 16)."),
    dict(name="Live half: Sector rotation Mondays (63d)", combo=dict(legs=PREV_LEGS, start=LIVE_SWITCH, which=1), ranked=False,
         group="sector",
         rule="The live account's sector-rotation half on its own (from the Oct 16 close, 1.0 then): the best C6-score "
              "stocks in the 3 sectors whose ETF has the best 63-day return, top 10, equal weights, rebalanced every Monday (the first session of the week); never buys a stock the low-volatility half holds. Not ranked (starts Oct 16)."),
    dict(name=MON5, combo=dict(legs=MON5_LEGS), group="combo",
         rule="The old live top-5 rules (Old live: Low vol + Sector rotation (Oct 16-26)) computed from Oct 2 with the sector half rebalanced "
              "every Monday (Tuesday after a Monday holiday) instead of Friday: top 5 per half, max 2 per sector, equal "
              "weight, no overlap, do_not_buy and the live manual buy blocks such as KO never bought."),
    dict(name=MON, combo=dict(legs=PREV_LEGS), group="combo",
         rule="The live 50/50 computed from Oct 2: low vol half on the last trading day of the month + sector rotation "
              "half rebalanced every Monday (Tuesday after a Monday holiday), no overlap, 10 stocks per half. The live rules Oct 19 "
              "until the 5-per-half switch (sector half Fri Oct 23, low vol half Fri Oct 30)."),
    dict(name=TOP5, combo=dict(legs=LIVE_LEGS), group="combo",
         rule="The live rules (Chirag, Oct 21, 2026; live from the sector half's Fri Oct 23 and the low vol half's Fri Oct 30 "
              "rebalance): the 50/50 (low vol half on the last trading day of the month + sector rotation 63-day half every "
              "Friday (Thursday when Friday is a holiday), no overlap, do_not_buy and the live manual buy blocks such as KO never bought) computed from Oct 2, with each half holding its top 5 instead of "
              "10: max 2 per sector (the 40% sector cap on 5 stocks), the same 20% max per stock (the extra stays cash)."),
    dict(name=LIVE_NOW, blend=NOW_HALVES, ranked=True, group="combo",
         rule="The live rules from Tue Oct 27, 2026 (Chirag): two independent halves of 50% each, computed from Oct 2. Half 1 = "
              "\"RS short-term (weekly + Mon–Thu sells)\" (Friday rebalance + Mon-Thu sell below rank 20, refilled from the top 10); "
              "half 2 = \"RS standard (Fridays only)\" (Friday rebalance only). Each half follows its own line exactly; a stock both "
              "pick is held by both, each paying from its own half (no overlap rule). Value = the average of the two lines."),
    dict(name="50/50: Low vol + SR every 2 weeks 63d", combo=dict(legs=(LV, ("sr63", "full", "2week", 0.5))),
         group="combo",
         rule="50/50, the sector-rotation half (63 days) rebalanced every other Friday (from Oct 2), no overlap."),
    dict(name="50/50: Low vol + SR Friday swap, keep top 10", combo=dict(legs=(LV, ("sr63", "swap10", "week", 0.5))),
         group="combo",
         rule="50/50, the sector-rotation half swap-only (keep top 10) checked on Fridays only."),
    dict(name="50/50: Low vol + Vol-adjusted momentum weekly", combo=dict(legs=(LV, ("vam", "full", "week", 0.5))),
         group="combo",
         rule="50/50, the second half = volatility-adjusted momentum (6-month return / 6-month volatility), top 10, equal "
              "weight, weekly, no overlap."),
    dict(name="50/50: Low vol + Calmer half weekly", combo=dict(legs=(LV, ("calm", "full", "week", 0.5))),
         group="combo",
         rule="50/50, the second half = 'Now: RS, calmer half of the list' (relative strength among stocks at or below the "
              "median 63-day volatility) rebalanced weekly, no Mon-Thu checks, no overlap."),
    dict(name="200-day breakout, 2 days above, exit 2% below", b200=dict(tp=0.2, days=2, buffer=0.98), earnings=None, group="event",
         rule="Buy a stock on the 2nd close in a row above its 200-day average after a close at or below it (10% each, at most 10, calmest first). Sell at +20% from the entry or when the close is 2% or more below the 200-day average (no other stop). Checked at the close."),
    dict(name="200-day breakout, 4 days above, exit 2% below", b200=dict(tp=0.2, days=4, buffer=0.98), earnings=None, group="event",
         rule="Buy a stock on the 4th close in a row above its 200-day average after a close at or below it (10% each, at most 10, calmest first). Sell at +20% from the entry or when the close is 2% or more below the 200-day average (no other stop). Checked at the close."),
    dict(name="Tech score 80/60 + TP20", event=dict(kind="tech80", tp=0.2), earnings=None, group="event",
         rule="Buy when the stock's Technical_Score percentile (among the day's stocks) crosses above 80; sell when it falls below 60 or at +20% from the entry (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="Tech score 70/60/90", event=dict(kind="tech70", tp=None), earnings=None, group="event",
         rule="Buy when the Technical_Score percentile crosses above 70 (not if already 90+); sell when it falls below 60 or reaches 90 or above (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="MA10 x MA30", event=dict(kind="ma10_30", tp=None), earnings=None, group="event",
         rule="Buy on the 2nd close after the 10-day average crosses above the 30-day; sell when the 10-day is 2% or more below the 30-day (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="MA50 x MA200 (golden cross)", event=dict(kind="golden", tp=0.2), earnings=None, group="event",
         rule="Buy on the 2nd close after the 50-day average crosses above the 200-day; sell when the 10-day drops below the 50-day or at +20% from the entry (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="MACD cross", event=dict(kind="macd", tp=0.2), earnings=None, group="event",
         rule="Buy when MACD(12,26) crosses above its 9-day signal line with the close above the 200-day average; sell when MACD crosses below the signal or at +20% from the entry (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="Donchian 20-day breakout", event=dict(kind="donchian"), earnings=None, group="event",
         rule="Turtle-style: buy on a close at a new 20-day high; sell on a close at a 10-day low (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="Bollinger mean reversion", event=dict(kind="bollinger", stop=0.05), earnings=None, group="event",
         rule="Buy when the close is below the lower Bollinger band (20, 2) with the close above the 200-day average; sell when the close is at or above the middle band or 5% below the entry (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="EMA 9/21 cross", event=dict(kind="ema"), earnings=None, group="event",
         rule="Buy on the 2nd close after the 9-day EMA crosses above the 21-day EMA; sell when the 9-day crosses below the 21-day (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="EMA 9/21 + ADX>25", event=dict(kind="ema_adx"), earnings=None, group="event",
         rule="The EMA 9/21 cross (2 closes) only when ADX(14) is above 25 (ADX from closes); sell when the 9-day crosses below the 21-day (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="Minervini trend template", event=dict(kind="minervini"), earnings=None, group="event",
         rule="Buy when price > SMA50 > SMA150 > SMA200, the SMA200 is up over 1 month, within 25% of the 52-week high and over 30% above the 52-week low; sell on a close below the SMA50 (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="52-week high momentum", event=dict(kind="high52", stop=0.08), earnings=None, group="event",
         rule="Buy when the close gets within 5% of its 52-week high and is above the 200-day average; sell 8% below the entry or on a close below the SMA50 (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="Dual MA + RSI filter", event=dict(kind="ma_rsi"), earnings=None, group="event",
         rule="Buy when the SMA50 is above the SMA200 and RSI(14) is 50-70; sell when RSI(14) falls below 40 (10% each, at most 10, calmest first; checked at the close)."),
    dict(name="Worst 3 sectors 14d, +10%/-3%", worst=dict(look=10), earnings=None, group="event",
         rule="Each day, find the 3 sectors (sector_mapping.py) with the worst average stock return over the last 14 days (10 sessions); hold the top 10 stocks across them by Strategy_Score (the signal-analysis score, relative strength since Oct 13), 10% each. Sell at +10% or -3% from the entry close; freed cash buys the best unheld stock of the current 3 worst sectors (not one sold that same day) (fewer than 10 eligible: the rest stays cash). Skips do-not-buy and manual buy blocks (KO). Checked at the close."),
    dict(name="Worst 3 sectors 1M, +10%/-3%", worst=dict(look=21), earnings=None, group="event",
         rule="Each day, find the 3 sectors (sector_mapping.py) with the worst average stock return over the last month (21 sessions); hold the top 10 stocks across them by Strategy_Score (the signal-analysis score, relative strength since Oct 13), 10% each. Sell at +10% or -3% from the entry close; freed cash buys the best unheld stock of the current 3 worst sectors (not one sold that same day) (fewer than 10 eligible: the rest stays cash). Skips do-not-buy and manual buy blocks (KO). Checked at the close."),
    dict(name="Worst 3 sectors 2M, +10%/-3%", worst=dict(look=42), earnings=None, group="event",
         rule="Each day, find the 3 sectors (sector_mapping.py) with the worst average stock return over the last 2 months (42 sessions); hold the top 10 stocks across them by Strategy_Score (the signal-analysis score, relative strength since Oct 13), 10% each. Sell at +10% or -3% from the entry close; freed cash buys the best unheld stock of the current 3 worst sectors (not one sold that same day) (fewer than 10 eligible: the rest stays cash). Skips do-not-buy and manual buy blocks (KO). Checked at the close."),
    dict(name="Golden cross + MACD confirm", event=dict(kind="golden_macd"), earnings=None, group="event",
         rule="Buy when the SMA50 is above the SMA200 and MACD is above its signal; sell when MACD falls below the signal (10% each, at most 10, calmest first; checked at the close)."),
]
def _common_rules(c):
    """The limits a line inherits from strategy_targets / be.winner_targets, said once (appended to its rule)."""
    if c.get("combo"):
        return "" if c["name"].startswith("Live") or c["name"] in (LIVE, TOP5) else " do_not_buy stocks are never held."
    if c.get("blend"):
        return ""
    if c.get("event") or c.get("b200") or c.get("weights"):
        return " do_not_buy stocks are never held."
    if c.get("worst"):
        return ""
    sel = {**PIN, **(c.get("select") or {})}
    cap = int(np.floor(sel["sector_cap"] * sel["n"] + 1e-9))
    earn = c.get("earnings", EARN_DAYS)
    return (f" Common rules: equal weight; {'no sector limit' if sel['sector_cap'] >= 1 else f'at most {cap} per sector'}; "
            f"{f'no new buy within {earn} trading days before earnings' if earn else 'earnings dates never block a buy'}; "
            "only scores above 0 are picked; weights halved at a rebalance while QQQ is below its 200-day average; "
            "do_not_buy stocks are never held.")
for _c in STRATEGIES:                                     # the limits the code applies (said once, here)
    _c["rule"] += _common_rules(_c)
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
    return [be.apply_history_start(pd.read_pickle(p)) for p in (FWD_BARS, be.CACHE_DIR / be.LONG_CACHE) if db.exists(p)]

def bar_features(sources, dates, cols):
    """Close / 20-day VWAP (the copy's daily-bar VWAP: typical price x volume), the volume jump (5-day / 50-day average
    volume), ATR(14) / close and open / close, computed inside each download so splits and dividends stay consistent; the newest download wins where its
    window is full. Decision days use their 2:30 PM bar, like signal_analysis.csv."""
    out = {k: pd.DataFrame(index=dates, columns=cols, dtype=float) for k in ("vwap_ratio", "atr_ratio", "open_ratio")}
    for b in sources:
        b = be.keep_decision_bars(b.sort_values(["Symbol", "Date"]).reset_index(drop=True), now=_NO_SAVE)
        w = {c: b.pivot_table(index="Date", columns="Symbol", values=c) for c in be.BAR_COLS}
        v = w["Volume"].replace(0, np.nan)
        vwap = ((w["High"] + w["Low"] + w["Close"]) / 3 * v).rolling(20).sum() / v.rolling(20).sum()
        pc = w["Close"].shift(1)                         # ATR: Wilder 14, as in be.calculate_technical_indicators
        tr = np.fmax(w["High"] - w["Low"], np.fmax((w["High"] - pc).abs(), (w["Low"] - pc).abs()))
        atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        new = {"vwap_ratio": w["Close"] / vwap,
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

def breakout200_weights(c, ma200, eligible, vol, tp, stop=None, buffer=1.0, n=10, start=None, days=3):
    """200-day breakout (Chirag, Oct 19, 2026): a stock that closed at/below its 200-day average and then closes above it
    `days` sessions in a row is bought at that close (Oct 19: N = 1-5 variants) (1/n of equity, at most n open; more signals than free slots: lowest
    63-day volatility first; cash left unused stays cash). Daily exit at the close (the forward test has closes only, not the
    2:35 PM CT price): +tp from the entry close, the close below buffer x the 200-day average, or (stop) -stop from entry.
    An exit's cash waits for the next signal. Entries only from `start` on (flat at the start)."""
    C, M, E = c.to_numpy(float), ma200.reindex_like(c).to_numpy(float), eligible.reindex_like(c).fillna(False).to_numpy(bool)
    V = vol.reindex_like(c).to_numpy(float)
    above = np.nan_to_num(C > M, nan=False) & np.isfinite(M)
    out, held = np.zeros(C.shape), {}
    t0 = int(np.searchsorted(c.index.values, np.datetime64(pd.Timestamp(start)))) if start is not None else 0
    for t in range(len(C)):
        for j in list(held):
            px, e = C[t, j], held[j]
            if px >= e * (1 + tp) or px < buffer * M[t, j] or (stop is not None and px <= e * (1 - stop)):
                del held[j]
        if t >= max(t0, days):
            sig = ~above[t - days] & E[t]
            for k in range(days):
                sig = sig & above[t - k]
            new = [j for j in np.argsort(np.nan_to_num(V[t], nan=np.inf), kind="stable") if sig[j] and j not in held]
            for j in new[:n - len(held)]:
                held[j] = C[t, j]
        out[t, list(held)] = 1 / n
    return pd.DataFrame(out, index=c.index, columns=c.columns)

def event_slot_weights(c, entry, exit_, eligible, vol, tp=None, n=10, start=None, stop=None, max_days=None):
    """Event lines (Chirag, Oct 19, 2026), sized like the 200-day breakouts: buy at the close of a day `entry` is True
    (1/n of equity, at most n open; more signals than free slots: lowest 63-day volatility first; unused cash stays cash);
    sell at a later close when `exit_` is True or (tp) the close is +tp from the entry close. Entries only from `start`."""
    C = c.to_numpy(float)
    N = entry.reindex_like(c).fillna(False).to_numpy(bool) & eligible.reindex_like(c).fillna(False).to_numpy(bool)
    X, V = exit_.reindex_like(c).fillna(False).to_numpy(bool), vol.reindex_like(c).to_numpy(float)
    out, held = np.zeros(C.shape), {}
    t0 = int(np.searchsorted(c.index.values, np.datetime64(pd.Timestamp(start)))) if start is not None else 0
    for t in range(len(C)):
        for j in list(held):
            e, t_in = held[j]
            if X[t, j] or (tp is not None and C[t, j] >= e * (1 + tp)) or (stop is not None and C[t, j] <= e * (1 - stop)) \
                    or (max_days is not None and t - t_in >= max_days):
                del held[j]
        if t >= t0:
            new = [j for j in np.argsort(np.nan_to_num(V[t], nan=np.inf), kind="stable") if N[t, j] and j not in held]
            for j in new[:n - len(held)]:
                held[j] = (C[t, j], t)
        out[t, list(held)] = 1 / n
    return pd.DataFrame(out, index=c.index, columns=c.columns)

def event_signals(inp, kind):
    """(entry, exit) Date x Symbol frames of the 5 event lines. pct = the day's percentile (0-100) of Technical_Score
    (signal_analysis combined_signal) among the eligible stocks. Crosses are fresh: below the day before the run starts."""
    m10, m30, m50, m200, eg = inp["ma10"], inp["ma30"], inp["ma50"], inp["ma200"], inp["eligible"]
    pct = inp["scores"]["tech"].where(eg).rank(axis=1, pct=True) * 100
    prev = pct.shift(1)
    two = lambda a: a & a.shift(1, fill_value=False) & ~a.shift(2, fill_value=True)   # fresh cross, 2 closes above
    if kind == "tech80":
        return (pct > 80) & (prev <= 80), pct < 60
    if kind == "tech70":
        return (pct > 70) & (prev <= 70) & (pct < 90), (pct < 60) | (pct >= 90)
    if kind == "ma10_30":
        return two(m10 > m30), m10 <= 0.98 * m30
    if kind == "golden":
        return two(m50 > m200), m10 < m50
    c = inp["close"]
    up = c > c.rolling(200).mean()                       # SMA200 from the closes (same as ma_200)
    cross = lambda a: a & ~a.shift(1, fill_value=True)   # fresh cross on this close
    if kind == "macd":
        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        a = macd > macd.ewm(span=9, adjust=False).mean()
        return cross(a) & up, cross(~a)
    if kind == "donchian":
        return c >= c.rolling(20).max(), c <= c.rolling(10).min()
    if kind == "bollinger":
        mid, sd = c.rolling(20).mean(), c.rolling(20).std()
        return (c < mid - 2 * sd) & up, c >= mid
    if kind == "ema":
        a = c.ewm(span=9, adjust=False).mean() > c.ewm(span=21, adjust=False).mean()
        return two(a), cross(~a)
    # Oct 19 (11:47 PM) additions. A state rule ("price > SMA50 > ..."): bought on the close it turns true (fresh).
    fresh = lambda a: a & ~a.shift(1, fill_value=True)
    ema = lambda x, n: x.ewm(span=n, adjust=False).mean()
    macd_up = lambda x: (ema(x, 12) - ema(x, 26)) > ema(ema(x, 12) - ema(x, 26), 9)
    def rsi(n):
        d = c.diff()
        g, l = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        return 100 - 100 / (1 + g / l.replace(0, np.nan))
    s50, s150, s200 = c.rolling(50).mean(), c.rolling(150).mean(), c.rolling(200).mean()
    if kind == "ema_adx":                                # ADX(14) from closes only (no highs / lows saved): |close change|
        d = c.diff()
        pdm, mdm, tr = d.clip(lower=0), (-d).clip(lower=0), d.abs()
        sm_ = lambda x: x.ewm(alpha=1 / 14, adjust=False).mean()
        pdi, mdi = 100 * sm_(pdm) / sm_(tr), 100 * sm_(mdm) / sm_(tr)
        adx = sm_(100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))
        a = ema(c, 9) > ema(c, 21)
        return two(a) & (adx > 25), cross(~a)
    if kind == "minervini":
        hi, lo = c.rolling(252).max(), c.rolling(252).min()
        a = (c > s50) & (s50 > s150) & (s150 > s200) & (s200 > s200.shift(21)) & (c >= 0.75 * hi) & (c > 1.3 * lo)
        return fresh(a), c < s50
    if kind == "high52":
        return fresh((c >= 0.95 * c.rolling(252).max()) & (c > s200)), c < s50
    if kind == "ma_rsi":
        r14 = rsi(14)
        return fresh((s50 > s200) & (r14 >= 50) & (r14 <= 70)), r14 < 40
    if kind == "golden_macd":
        a = macd_up(c)
        return fresh((s50 > s200) & a), ~a
    raise ValueError(kind)

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
    W, args = be.WINNER, {**be.winner_rank_args(inp["regime"]), **PIN, "sleeve": ()}   # pinned Oct 2 rules
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
        "mom_6": ret(126) * 100,
        "low_vol": (1 / vol).where(up),
        "dual": (ret(252) * 100).where(regime, axis=0),
        "mom_risk": ret(126) / (c.pct_change(fill_method=None).rolling(126).std() * np.sqrt(126)),
        "high_200": c / c.rolling(200).max(),          # 1 = the close is the highest close of the last 200 sessions
        "quality_mom": (100 * (0.5 * _pct(snap["Fundamental_Weight"]) + 0.5 * _pct(mom_12_1))),
        "drift": earnings_drift(c, b["SPY"] if "SPY" in b else pd.Series(1.0, index=dates), earnings),
        "sector_rot": live.where(etf_3m.rank(axis=1, method="dense", ascending=False).le(3)),   # top 3 ETF returns
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
            "bench": b, "up": up, "ma200": f["ma_200"], "ma10": f["ma_10"], "ma30": f["ma_30"], "ma50": f["ma_50"],
            "qqq50": (qqq > qqq.rolling(50).mean()).reindex(dates).fillna(False),
            "weekly": weekly, "live_cal": be.winner_rebalance_days(dates, live=True), "earnings": earnings, "atr": atr,
            "open": bx["open_ratio"] * c,
            "soon": soon.reindex(index=dates, columns=cols).fillna(False),          # earnings within 5 days (E5)
            "monthly": pd.Series([be.next_sessions(d, 1)[0].month != d.month for d in dates], index=dates),
            "through": have[have].index.max() if have.any() else pd.Timestamp(0)}

def worst_sector_weights(c, score, eligible, look, tp=0.10, stop=0.03, n=10, k=3, start=None):
    """Worst sectors (Chirag, Oct 21, 2026): each close, the k sectors with the lowest mean stock return over `look` sessions;
    free slots (1/n each, at most n) buy their best `score` stocks not held. Exit at a close +tp / -stop from the entry close;
    a stock sold that day is not bought back the same day. Entries only from `start`."""
    sect = pd.Series({x: be.sector_mapping.symbol_sector.get(x, "Other") for x in c.columns})
    ret = (c / c.shift(look) - 1).T.groupby(sect).mean().T          # Date x sector
    C, S = c.to_numpy(float), score.reindex_like(c).to_numpy(float)
    E = eligible.reindex_like(c).fillna(False).to_numpy(bool)
    out, held = np.zeros(C.shape), {}
    t0 = int(np.searchsorted(c.index.values, np.datetime64(pd.Timestamp(start)))) if start is not None else 0
    for t in range(len(C)):
        sold = set()
        for j in list(held):
            if C[t, j] >= held[j] * (1 + tp) or C[t, j] <= held[j] * (1 - stop):
                del held[j]
                sold.add(j)
        r = ret.iloc[t].dropna()
        if t >= t0 and len(held) < n and len(r):
            inside = sect.isin(r.nsmallest(k).index).to_numpy()
            ok = [j for j in np.argsort(-np.nan_to_num(S[t], nan=-np.inf), kind="stable")
                  if inside[j] and E[t, j] and np.isfinite(S[t, j]) and j not in held and j not in sold]
            for j in ok[:n - len(held)]:
                held[j] = C[t, j]
        out[t, list(held)] = 1 / n
    return pd.DataFrame(out, index=c.index, columns=c.columns)

def _mblock(inp):
    """Date x Symbol True while a live manual buy block is on (manual_trade.load_blocks: from its day until the release)."""
    import manual_trade as mt
    c = inp["close"]
    m = pd.DataFrame(False, index=c.index, columns=c.columns)
    for sym, e in mt.load_blocks().items():
        if sym in m:
            on = pd.Series(c.index >= pd.Timestamp(e["added"]).normalize(), index=c.index)
            if e.get("block_until_price") is not None:
                on &= ~(on & c[sym].le(e["block_until_price"])).cummax()
            m[sym] = on.to_numpy()
    return m

def _dnb(inp):
    """Date x Symbol True for sector_mapping.do_not_buy (skipped by every paper strategy from Oct 19, 2026)."""
    c = inp["close"]
    return pd.DataFrame(np.broadcast_to(c.columns.isin(be.sector_mapping.do_not_buy), c.shape), index=c.index, columns=c.columns)

def strategy_targets(cfg, inp):
    """Daily live target weights of one STRATEGIES entry (Date x Symbol). Ranked ones go through the engine's own selection
    code (be.winner_targets): only overrides are passed, WINNER itself is never changed. A "stop" strategy also leaves its
    open-sale prices in inp["open_sells"][name]."""
    if cfg.get("blend"):                               # two independent halves: the average of their target weights
        a, b = (strategy_targets(next(c for c in STRATEGIES if c["name"] == n), inp) for n in cfg["blend"])
        cols = a.columns.union(b.columns)
        return 0.5 * a.reindex(columns=cols, fill_value=0.0) + 0.5 * b.reindex(columns=cols, fill_value=0.0)
    if cfg.get("combo"):                               # forward_combos line: the simulated end-of-day weights
        w = combo_weights(cfg, inp)
        if cfg["name"].startswith("Live") or cfg["name"] in (LIVE, TOP5):      # the live account's rules: as they are
            return w
        return w.mask(_dnb(inp).reindex(index=w.index, columns=w.columns, fill_value=False), 0.0)
    if cfg.get("event"):                             # event lines (Oct 19): only their entries / exits trade
        en, ex = event_signals(inp, cfg["event"]["kind"])
        tgt = event_slot_weights(inp["close"], en, ex, inp["eligible"] & ~_dnb(inp), inp["vol"], tp=cfg["event"].get("tp"),
                                 stop=cfg["event"].get("stop"), max_days=cfg["event"].get("days"), start=be.FORWARD_START)
    elif cfg.get("worst"):                             # worst sector lines (Oct 21): only entries / exits trade
        tgt = worst_sector_weights(inp["close"], inp["scores"]["now"], inp["eligible"] & ~_dnb(inp) & ~_mblock(inp),
                                   start=be.FORWARD_START, **cfg["worst"])
    elif cfg.get("b200"):                              # 200-day breakout family (no calendar: only entries / exits trade)
        tgt = breakout200_weights(inp["close"], inp["ma200"], inp["eligible"] & ~_dnb(inp), inp["vol"], start=be.FORWARD_START, **cfg["b200"])
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
                                   selection={**PIN, **(cfg.get("select") or {}), "sleeve": sleeve},
                                   earnings_block_days=cfg.get("earnings", EARN_DAYS), earnings=inp["earnings"],
                                   exit_all_below=below, exit_to_top=top,
                                   never_buy=be.sector_mapping.do_not_buy if cfg.get("never_buy") else ())
    tgt = tgt.mask(_dnb(inp).reindex(index=tgt.index, columns=tgt.columns, fill_value=False), 0.0)   # its weight: cash
    return be.live_weights(tgt).reindex(columns=inp["close"].columns).fillna(0.0)

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

def migrate_names(*paths):
    """Saved rows of a renamed line (OLD_NAMES) take its new name; a file is rewritten only when it has old names."""
    for p in paths:
        df = _read(p)
        if df is not None and len(df) and df["Strategy"].isin(list(OLD_NAMES) + list(REMOVED)).any():
            df = df[~df["Strategy"].isin(REMOVED)]
            df.assign(Strategy=df["Strategy"].replace(OLD_NAMES)).pipe(db.to_csv, p, index=False)

def rebalance_days(cfg, inp):
    """Days on which every holding is brought back to its target (unless within 1 point of it), like the live Friday
    rebalance: the strategy's calendar (Fridays / month ends); the trend rule on Fridays; the dip and threshold rules
    never (only the names they buy or sell trade). None when the inputs carry no calendar."""
    if cfg.get("b200") or cfg.get("event"):                # breakouts / event lines: only their entries / exits trade (10% at entry)
        return None
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
    rows, hold = [], []
    for cfg in STRATEGIES:
        name = cfg["name"]
        mine = old[old["Strategy"] == name] if old is not None else pd.DataFrame(columns=["Date"])
        last = mine["Date"].max() if len(mine) else None
        todo = days if last is None else days[days > last]
        if not len(todo):
            continue
        if cfg.get("blend"):                                  # 50/50 of two lines, each from its own 1.0 start
            parts = []
            for n in cfg["blend"]:
                v = pd.DataFrame([r for r in rows if r["Strategy"] == n], columns=["Date", "Strategy", "Value", "Cash"])
                if old is not None:
                    v = pd.concat([old[old["Strategy"] == n][["Date", "Strategy", "Value", "Cash"]], v], ignore_index=True)
                v = v.assign(Date=pd.to_datetime(v["Date"])).drop_duplicates("Date", keep="last").set_index("Date").sort_index()
                hh = pd.DataFrame([r for r in hold if r["Strategy"] == n], columns=["Date", "Strategy", "Symbol", "Weight", "Shares", "Price"])
                if old_h is not None:
                    hh = pd.concat([old_h[old_h["Strategy"] == n], hh], ignore_index=True)
                hh = hh.assign(Date=pd.to_datetime(hh["Date"])).drop_duplicates(["Date", "Symbol"], keep="last")
                parts.append((v.loc[v.index >= pd.Timestamp(start)], hh))
            base = [float(v["Value"].iloc[0]) if len(v) else np.nan for v, _ in parts]
            for d in todo:
                if not all(d in v.index for v, _ in parts):
                    continue
                rows.append({"Date": d, "Strategy": name, "Value": sum(0.5 * v.at[d, "Value"] / b_ for (v, _), b_ in zip(parts, base)),
                             "Cash": sum(0.5 * v.at[d, "Cash"] / b_ for (v, _), b_ in zip(parts, base))})
                hd = {}
                for (_, hh), b_ in zip(parts, base):
                    for r in hh[hh["Date"] == d].itertuples():
                        w0, s0, _p = hd.get(r.Symbol, (0.0, 0.0, r.Price))
                        hd[r.Symbol] = (w0 + 0.5 * r.Weight, s0 + 0.5 * r.Shares / b_, r.Price)
                hold += [{"Date": d, "Strategy": name, "Symbol": k, "Weight": w_, "Shares": s_, "Price": p_}
                         for k, (w_, s_, p_) in hd.items()]
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
            # Names whose saved holding disagrees with the recomputed target are re-decided on the next day (Oct 26,
            # 2026 fix: a rule change on Oct 21 put new names in the recomputed last-day target, so they never
            # showed as "changed" and were never bought; their cash sat idle). Applies to the rules that decide
            # names day by day (no calendar: dip / event / breakout lines; "mwf" ranked lines with Mon-Thu checks).
            # Weekly- and monthly-only lines keep their holdings until their own rebalance day, as their rule says.
            daily = full is None or (cfg.get("calendar", "mwf") == "mwf" and not cfg.get("weights"))
            missed = ((shares > 0) != (w.loc[last] > 0)) if daily else pd.Series(False, index=cols)
            prev = w.loc[last].where(~missed, np.nan)
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
    if rows:
        for p, new, prior, cols_ in ((path, rows, old, ["Date", "Strategy", "Value", "Cash"]),
                                     (hold_path, hold, old_h, ["Date", "Strategy", "Symbol", "Weight", "Shares", "Price"])):
            new = pd.DataFrame(new, columns=cols_)    # all-cash days add no holdings rows: keep the columns (no KeyError)
            out = pd.concat([prior, new], ignore_index=True) if prior is not None else new
            out = out.assign(Date=pd.to_datetime(out["Date"])).sort_values("Date", kind="stable")   # by day, registry order
            out.assign(Date=out["Date"].dt.strftime("%Y-%m-%d")).pipe(db.to_csv, p, index=False)
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
    return db.read_csv(path, float_precision="round_trip", **kw) if db.exists(path) else None

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
           "Positions": len(__import__("holdings").held_qty(account=account)),
           "Closed_Picks": len(pk), "Winning_Picks": int((pk["P/L $"] > 0).sum()), "Traded_USD": round(traded, 2),
           "Cost_USD": round(cost, 2)}
    old = _read(path, dtype={"Date": str})
    new = pd.DataFrame([row])
    if old is not None and (old["Date"] != row["Date"]).any():
        new = pd.concat([old[old["Date"] != row["Date"]], new], ignore_index=True)
    new[DAILY_COLS].pipe(db.to_csv, path, index=False)
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
    deposits and withdrawals are not returns) + QQQ / SPY (closes, comparison only), sorted by total return. Rank counts
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
    t = t.sort_values(["Total return %", "Max drawdown %"], ascending=False, na_position="last", kind="stable")
    ok = t["ranked"] & t["Total return %"].notna()
    t["Rank"] = ok.cumsum().where(ok).astype("Int64")
    return t[["Rank", "Strategy"] + cols[:2] + weeks + cols[2:]].reset_index(drop=True)

def verdict(board):
    """One plain line: too early / the winner / no winner. """
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

LIVE_HALVES = ("Live half: Low vol monthly", "Live half: Sector rotation Mondays (63d)")

def provisional(values, bench=None, hold_path=HOLDINGS_CSV, bars_path=FWD_BARS):
    """Display only, never saved. A trading day is saved by the daily run's after-close step (from 3:05 PM CT) once it
    has signal rows (the 2:30 PM decision run Mon-Fri, or the step's own refresh on a non-decision day, which runs first); until
    then, or when a run failed, the board would show the previous day's numbers again. This marks each strategy's saved
    holdings (shares + cash of its last saved day, no trades) to the close of every later day the saved daily bars cover
    (complete days only: a bar is taken from 3:05 PM CT), and QQQ / SPY to
    the same closes. Returns (values, bench) with those rows added; values rows carry Provisional = True."""
    if values is None or values.empty or not db.exists(bars_path):
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
    """Strategy -> 'NVDA, AAPL, ...' of its latest saved day (largest first; equal weights, so no % shown)."""
    h = _read(path, parse_dates=["Date"])
    if h is None or h.empty:
        return {}
    h = h.assign(Strategy=h["Strategy"].replace(OLD_NAMES))
    h = h[h["Date"] == h.groupby("Strategy")["Date"].transform("max")]
    h = h[h["Weight"] > 0].sort_values(["Strategy", "Weight"], ascending=[True, False])
    return {s: ", ".join(r.Symbol for r in g.itertuples()) for s, g in h.groupby("Strategy")}

def consensus(board, hold, ranks=None, n_strategies=5, n_stocks=10):
    """Display only (never saved, never traded): the stocks most held by the best forward-test strategies.

    board = leaderboard() as the dashboard shows it (incl. provisional rows); hold = saved holdings rows (Date, Strategy,
    Symbol, Weight); ranks = {symbol: latest Strategy_Rank} (tie-break). The strategies are taken in the leaderboard's
    own order (Chirag, Oct 21; was the latest week's return): its Rank (total return since the start, smaller drawdown
    breaks ties), unranked rows after, so the Top 5 are the table's Rank 1-5.  Twins (same stocks at the same target weights on the holdings day) count once: the
    better-placed one stays. Each strategy's holdings = its latest saved day (what the board's values come from; a
    provisional close keeps them, so nothing past the shown day is used). Stocks: held (target weight > 0) by the most of those strategies; ties -> larger summed weight, then the
    better latest rank. Returns dict(day, by, strategies=[(name, [twins])], stocks=DataFrame(Symbol, Count, Weight,
    Rank), by = the column used); None when there is nothing to show."""
    if board is None or board.empty or hold is None or hold.empty:
        return None
    names = {c["name"] for c in STRATEGIES}
    b = board.assign(Strategy=board["Strategy"].str.replace(LIVE_MARK, "", regex=False))
    b = b[b["Strategy"].isin(names)]
    by = "Total return %"
    rk = pd.to_numeric(b["Rank"], errors="coerce").astype(float) if "Rank" in b else pd.Series(np.nan, index=b.index)
    b = b.assign(_rk=rk.fillna(np.inf)).sort_values(["_rk", by, "Max drawdown %"], ascending=[True, False, False],
                                                   na_position="last", kind="stable")
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
