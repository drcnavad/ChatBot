# Stock Analysis

An automated stock strategy that trades a real-money Alpaca account, plus a Streamlit dashboard (http://localhost:8502).
All code lives in this folder. Everything the pipeline writes goes to `Reports/`, which is not committed.

## Strategy (live rules: `backtest_engine.WINNER`)
- **Stock list:** `sector_mapping.py` is the only stock list (96 tradable stocks + QQQ as the benchmark). Stocks with
  less than 200 trading days of history are shown but never traded.
- **Score:** 50% technical score + 50% relative strength (stock vs its sector ETF, sector ETF vs SPY), ranked every day.
- **Friday rebalance** (the week's last session), decided on the 2:30 PM CT bar: hold the top 10, weighted by inverse
  volatility, max 20% per stock, 99% invested. When QQQ is below its 200-day average every weight is halved. A holding
  within 1 point of its target is left alone.
- **Monday / Wednesday check** (2:30 PM CT): a holding ranked worse than 20 is sold and replaced, same dollars, by the
  best-ranked top-10 stock not held (cash until Friday only if none is left). On the live account this applies to every
  position. Then cash above 1% of equity buys top-10 stocks not held, and what is left tops up ranks 1-3 (19.8% cap).
- **Earnings:** no new buy or top-up of a stock with earnings within 5 days. On Friday a new pick blocked this way is
  replaced by the next eligible stock (down to rank 20). On a held stock's earnings day, a drop of 5% or more below the
  previous close sells it (`earnings_stop.py`), and it is not bought back until its earnings sessions are over.

## Schedule (launchd, `launchd/install_schedule.sh`, US Central time)
| Job | When | What |
|---|---|---|
| `com.stockanalysis.evening` | Mon-Fri 2:30 PM + 3:05 PM, at login, on wake, every 30 min | `pipeline_watchdog.py --trade --scheduled`: the Mon/Wed/Fri decision at 2:30 PM (full data update + trade), then once per trading day from 3:05 PM the after-close step: signal refresh with the day's final bar on non-decision days + `forward_test.py --record`. Nothing runs 4:05-4:30 PM (Alpaca books deposits around 4:15 PM). |
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
paper_trade.py (Mon/Wed/Fri, live orders) → live_orders_log.csv, live_pending_orders.json, run_log.csv
forward_test.py (daily)  → forward_test_daily.csv, forward_strategies.csv, forward_strategies_holdings.csv
trade_audit.py           → live_trade_ledger.csv, live_round_trips.csv
app.py (dashboard)       ← reads all of the above + the Alpaca account (read-only)
```
Quota APIs (Alpha Vantage, NewsAPI, Finnhub) run only in the 2:30 PM full update on decision days. Every other run
uses only Alpaca market data.

## Files
| File | Role |
|---|---|
| `run_all.py` | The one pipeline command (mode, steps, trade gate, fill check, after-close step). |
| `pipeline_watchdog.py` | Retries transient failures for the launchd jobs. Diagnose only: never edits code or retries orders. |
| `paper_trade.py` | Order planning and live order sending (dry run by default). |
| `backtest_engine.py` | Rules, scores, ranking, NYSE calendar, the shared engine. |
| `alpaca_paper.py` | Read-only Alpaca account client (positions, orders, activities, daily history). |
| `earnings_stop.py`, `trade_audit.py`, `tax_lots.py` | Earnings-day stop, trade audit, tax lots (dashboard). |
| `forward_test.py` | Forward test of the account and 39 paper strategies from Fri Oct 2, 2026. |
| `app.py`, `dashboard/` | Streamlit dashboard. |
| `sector_mapping.py` | The stock list, sectors and names. |
| `tests/run_tests.py` | Runs every regression test (`--fast` skips the app tests). |

## Setup
```
/opt/anaconda3/bin/python -m pip install -r requirements.txt
cp .env.example .env            # then fill in the keys
bash launchd/install_schedule.sh
python tests/run_tests.py
```
