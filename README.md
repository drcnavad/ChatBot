# Stock Analysis

An automated stock strategy that trades a real-money Alpaca account, plus a Streamlit dashboard (http://localhost:8502).
All code lives in this folder. Everything the pipeline writes goes to `Reports/`, which is not committed.

## Strategy (live rules: `backtest_engine.WINNER`)
- **Stock list:** `sector_mapping.py` is the only stock list (84 tradable stocks + QQQ as the benchmark). Stocks with
  less than 200 trading days of history are shown but never traded.
- **Score (from Oct 16, 2026): low volatility in uptrend.** 1 / 63-day volatility, calmest first; only stocks above their
  200-day average (the others score -1). Ranked every day; ties by relative strength.
- **Friday rebalance** (the week's last session), decided on the 2:30 PM CT bar: hold the top 10 at equal weights,
  max 15% per stock, 99% invested. Plus the **defensive sleeve** (`defensive` in `sector_mapping.py`: LMT
  and NOC from Oct 16, 2026): always held next to the top 10, all 12 at an equal weight (8.33% x 99%);
  a sleeve stock that is also in the top 10 frees its slot for rank 11. The sleeve is on `do_not_sell` and the Mon-Thu
  rank-20 rule never sells it. When QQQ is below its 200-day average every weight is halved. A holding
  within 1 point of its target is left alone.
- **Monday-Thursday check** (2:30 PM CT): a holding at or below its 200-day average or ranked worse than 20 is sold, but only if it is below
  its Alpaca average entry price (at or above it: kept until Friday), and replaced, same dollars, by the best-ranked top-10 stock not held (cash until Friday only if none is left). On the live account this applies to every
  position. Then cash above 1% of equity buys top-10 stocks not held, and what is left is split equally over the held stocks at or above their
  average entry price (none in profit: all held; earnings-blocked / sold-today ones skipped; a stock at its cap passes its
  share on).
- **Earnings:** no new buy or top-up of a stock with earnings within 5 days (including a report after that day's close). On Friday a new pick blocked this way is
  replaced by the next eligible stock (down to rank 20). On a held stock's earnings day, a drop of 5% or more below the
  previous close sells it (`earnings_stop.py`), and it is not bought back until its earnings sessions are over.
- **Your overrides:** `do_not_buy` and `do_not_sell` in `sector_mapping.py` (e.g. `do_not_buy = ['TSLA']`). The Mon-Fri
  runs and the 9 AM fill check never buy or top up a do-not-buy stock (on Friday its slot goes to the next eligible stock,
  down to rank 20) and never sell or trim a do-not-sell stock (a Mon-Thu replacement out of it is dropped with its buy).
  These show as `SKIP (on your do-not-... list)` rows. On Friday a held do-not-sell stock outside the picks stays
  invested, so the picks share the rest (99% invested, SNOW kept at 10% -> picks scaled to 89%). The earnings-day stop
  never sells a do-not-sell stock either.

## Schedule (launchd, `launchd/install_schedule.sh`, US Central time)
| Job | When | What |
|---|---|---|
| `com.stockanalysis.support` | Mon 1:00, Tue 1:07, Wed 1:14, Thu 1:21, Fri 1:28 PM | `run_all.py --support --scheduled`: company reports (Alpha Vantage), news sentiment (NewsAPI + Finnhub), earnings dates (Finnhub + yfinance). 7 min later each day, so NewsAPI always runs more than 24 h after its last run; only on trading days 12:30-2:00 PM, never beside a trade run. |
| `com.stockanalysis.evening` | Mon-Fri 2:30 PM + 3:05 PM, at login, on wake, every 30 min | `pipeline_watchdog.py --trade --scheduled`: the Mon-Fri decision at 2:30 PM (signals from the latest support files + trade), then once per trading day from 3:05 PM the after-close step: signal refresh with the day's final bar on non-decision days + `forward_test.py --record`. Nothing runs 4:05-4:30 PM (Alpaca books deposits around 4:15 PM). |
| `com.stockanalysis.morning` | Mon-Fri 9:00 AM | `pipeline_watchdog.py --fill-check --scheduled`: sends the rest of the previous decision's orders. |
| `com.stockanalysis.earningsstop` | every 5 min | `earnings_stop.py --loop`: the earnings-day 5% stop. It only watches during a held stock's earnings sessions. |
| `com.stockanalysis.tradeaudit` | Mon-Fri 9:45 AM + 3:45 PM | `trade_audit.py --write --log`: order ledger, round trips, reconciliation and alerts. Read-only. |
| `com.stockanalysis.dashboard-8502` | always on | `run_dashboard_8502.sh` → `streamlit run app.py`. |

Missed decisions (Mac asleep) are caught up at the next regular session. Each decision runs only once
(`Reports/run_state.json`, lock files, client order ids).

## Data flow
```
company_report_autofetch.py → balance_sheet.csv → company_report_processing.ipynb → balance_sheet_weights.csv, complete_company_analysis.xlsx
sentiment_analysis.ipynb → weighted_sentiment.csv, news_cleaned_df.csv, sentiment_history.csv
earnings_date.ipynb      → earnings_date.csv
             ↓ (all read by)
main_signal_analysis.ipynb (Alpaca daily bars) → signal_analysis.csv  (the one main output: prices, scores, ranks, weights)
    + strategy_picks / strategy_changes / strategy_decisions / strategy_midweek_check / strategy_holdings.csv,
      benchmark_prices.csv, factor_history.csv, decision_bars.csv, short_history_reference.csv
             ↓
paper_trade.py (Mon-Fri, live orders) → live_orders_log.csv, live_pending_orders.json, run_log.csv
forward_test.py (daily)  → forward_test_daily.csv, forward_strategies.csv, forward_strategies_holdings.csv
trade_audit.py           → live_trade_ledger.csv, live_round_trips.csv
app.py (dashboard)       ← reads all of the above + the Alpaca account (read-only)
```
Quota APIs (Alpha Vantage, NewsAPI, Finnhub) run only in the `--support` job (about 1 PM CT on trading days); the 2:30 PM
run reads their latest files. Every other run uses only Alpaca market data.

## Files
| File | Role |
|---|---|
| `run_all.py` | The one pipeline command (mode, steps, trade gate, fill check, after-close step). |
| `pipeline_watchdog.py` | Retries transient failures for the launchd jobs. Diagnose only: never edits code or retries orders. |
| `paper_trade.py` | Order planning and live order sending (dry run by default). |
| `backtest_engine.py` | Rules, scores, ranking, NYSE calendar, the shared engine. |
| `alpaca_paper.py` | Read-only Alpaca account client (positions, orders, activities, daily history). |
| `earnings_stop.py`, `trade_audit.py`, `tax_lots.py` | Earnings-day stop, trade audit, tax lots (dashboard). |
| `forward_test.py` | Forward test of the account and 52 paper strategies from Fri Oct 2, 2026 (Fri rebalance, Mon-Thu checks). |
| `app.py`, `dashboard/` | Streamlit dashboard. Home tab "My notes" = an editable table (Stock, Date, Time, Note, Action, Price) saved to `Reports/my_notes.csv` on every change. |
| `sector_mapping.py` | The stock list, sectors and names. |
| `tests/run_tests.py` | Runs every regression test (`--fast` skips the app tests). |

## Setup
```
cp .env.example .env            # then fill in the keys (a moved folder keeps its .env)
bash setup_mac_mini.sh          # packages + git notebook filter + the launchd jobs, paths from this folder's location
python tests/run_tests.py
```
Moving to another Mac: see `MOVE.md`.
