# Stock Analysis

An automated stock strategy that trades a **real-money Alpaca LIVE account**, plus a Streamlit dashboard
(http://localhost:8502, launchd `com.stockanalysis.dashboard-8502`). Runs on Chirag's Mac mini (`~/Stock Analysis`,
Python `/opt/anaconda3/bin/python`, times in US Central). Everything the pipeline writes goes to `Reports/` (not committed).
Keys live in `.env` (`ALPACA_LIVE_KEY_ID` / `ALPACA_LIVE_SECRET_KEY`, never print or commit them).

> **Never place orders when testing.** No `run_all.py --trade` / `--fill-check`, no `paper_trade.py --submit`, no
> `manual_trade.py buy|sell` without `--dry-run`. Tests mock Alpaca (`tests/run_tests.py` sets
> `STOCK_ANALYSIS_LIVE_HOLDINGS=off` and temp paths for every state file). Read-only checks are fine:
> `paper_trade.py --from-account` (dry run), `trade_audit.py`.

## Live strategy (`backtest_engine.WINNER`, from Fri Oct 16, 2026)
Two independent halves, each with its own cash, tracked in `Reports/sleeve_ledger.json` (`sleeves.py`); no stock in both:
**Top 5 per half from Oct 21, 2026** (`WINNER['n'] = 5`; first used at the sector half's Fri Oct 23 and the low-vol
half's Fri Oct 30 rebalance; 10 per half before; forward row "Current Strategy (Live from Oct End)").
- **50% Low volatility in uptrend (monthly):** the 5 calmest stocks (1 / 63-day volatility) whose close is above the
  200-day average, equal weight, max 2 per sector; rebalanced on the **last trading day of the month**
  (targets = `Provisional_Weight` in `Reports/strategy_picks.csv`). QQQ below its 200-day average: weights halved.
- **50% Sector rotation (weekly):** top 5 by C6 score (equal weight, 20% of the half each, max 2 per sector) in the 3 sectors with the best 63-day ETF return (not held by
  the low-vol half), rebalanced **every Friday** at 2:30 PM CT (the week's last session: Thursday when Friday is a
  holiday; `WINNER['sector_day'] = "friday"`, `backtest_engine.is_sector_day`). Fridays until Oct 16, 2026, Mondays
  Oct 19-22, Fridays again from Oct 23 (Chirag, Oct 21). When the month's last trading day is a Friday (e.g. Oct 30),
  both halves rebalance that day without overlap.
- Decisions use the ~2:30 PM CT bar. A session that is no decision **holds everything** (no Mon-Thu trading, no Friday
  falling check, no earnings rule). A holding within 1 point of its target is not traded. A position in no half of the
  ledger is sold at the next trade run.
- Never bought: `sector_mapping.do_not_buy`, manual sell blocks (`Reports/manual_block.json`), stocks sold by the
  earnings-day stop. Never sold: `sector_mapping.do_not_sell`. `earnings_stop.py` is loaded but places no orders
  (`WINNER['earnings_day_stop'] = False`).
- Orders: sells first, every order a LIMIT at the live quote (buy ask + 0.05%, sell bid - 0.05%; stale / >0.5% spread
  quotes skipped). Market hours: 2-decimal shares; after hours: whole-share extended-hours DAY limits until 7 PM CT. The
  rest goes to `Reports/live_pending_orders.json` and the **9 AM CT fill check** sends it. Client order ids make every
  send idempotent; a missed decision is caught up once at the next session.

**Holdings: one source** (Chirag, Oct 19, 2026): live Alpaca positions, read only through `holdings.py`
(`get_positions()`, 60 s cache; order code reads fresh through its trading client). Nothing is recorded as owned before a
fill. `Reports/sleeve_ledger.json` keeps per half only the symbols' tags (which half a held stock belongs to) and the
half's cash at `cash_at`; each run a half = the live shares of its tags + its cash moved by the real Alpaca fills since
then (an unfilled buy costs nothing). A queued buy (`Reports/live_pending_orders.json`) shows as "Pending order", never as a
holding; a tag goes once its stock is neither held nor queued. There is no `my_positions.csv` any more.

**Leftover sleeve cash** (Oct 19): after a rebalance, a half's cash of at least max($500, 2% of the half) (e.g. a target
blocked by a manual block) is queued for the 9 AM CT fill check as a buy of that half's next-in-line stock, limit never
above the ask (`manual_trade.queue_leftover_cash`, called by the trade run; never doubled).

**Removing a stock** (Oct 19, 2026): comment it out of `stock_symbols` in `sector_mapping.py`. It leaves the rankings,
the dashboard lists and the forward test at once. If the account still holds it, nothing special happens until its half
rebalances (sector = Fridays 2:30 PM CT, low volatility = last trading day of the month): it is no target there (both
halves keep only `tradable_symbols`), so it is sold and never rebought. Until then it stays in `Reports/sleeve_ledger.json`
(not an orphan) and is priced from Alpaca market data. No cleanup is needed after the sell. Removed Oct 19: PG
(sector, sells Fri Oct 23), EMR and PFE (low vol, sell Fri Oct 30), plus ECL, SHW, NUE, URI, SHOP, PEP, VMC, CRCL, ETN (not held).

## Schedule (launchd, `launchd/*.plist`, installed by `launchd/install_schedule.sh`; times CT)
| Job | When | What | Trades? |
|---|---|---|---|
| `com.stockanalysis.support` | Mon 1:00, Tue 1:07, Wed 1:14, Thu 1:21, Fri 1:28 PM | `run_all.py --support --scheduled`: company reports, news sentiment, earnings dates (quota APIs) | no |
| `com.stockanalysis.evening` | Mon-Fri 2:30 PM + 3:05 PM, at login, on wake, every 30 min | `pipeline_watchdog.py --trade --scheduled` → `run_all.py`: 2:30 PM signals + decision + orders on decision days (Fridays, Thursday on a holiday: sector half; month's last trading day: low-vol half); from 3:05 PM the after-close step (`run_all.daily_step`: signal refresh, `forward_test.py --record`, `manual_trade.py update-blocks`). Never 4:05-4:30 PM. | **yes** (2:30 PM) |
| `com.stockanalysis.morning` | Mon-Fri 9:00 AM | `pipeline_watchdog.py --fill-check --scheduled`: sends the pending rests | **yes** |
| `com.stockanalysis.tradeaudit` | Mon-Fri every 30 min 8:30 AM-3:00 PM + 3:45 PM | `refresh_dashboard_data.sh`: `trade_audit.py --write --log` (ledger, round trips, alerts) | no (GET only) |
| `com.stockanalysis.earningsstop` | every 5 min + at login | `earnings_stop.py --loop` (no-op while the stop is off) | no (off) |
| `com.stockanalysis.dashboard-8502` | always on | `run_dashboard_8502.sh` → `streamlit run app.py`. Restart: `launchctl kickstart -k gui/$(id -u)/com.stockanalysis.dashboard-8502` | no |

## Manual orders (`manual_trade.py`, separate from the pipeline)
```
python manual_trade.py sell KO [--block-until-drop 10] [--no-replace] [--dry-run]
python manual_trade.py buy XYZ [--amount 1000] [--dry-run]     # default amount: 10% of equity (one slot, 5 per half)
python manual_trade.py blocks                                   # list active blocks
```
- Places the order at once on the LIVE account (DAY limit, pipeline pricing); refuses outside market hours. Every
  action (also dry runs / refusals) is appended to `Reports/manual_orders.json`.
- **Sell:** the stock leaves the sleeve ledger. Once the sell fills, the freed cash buys the **next-in-line stock of the
  same half** (best-ranked not held / blocked / do-not-buy), limit never above the ask, added to that half
  (`--no-replace` keeps the cash).
- **Block rule:** `--block-until-drop N` adds `Reports/manual_block.json[SYM] = {reference_close, block_until_price =
  close x (1 - N%), added}`. The after-close step sets `reference_close` to that day's official close and removes the
  block once a close is at or below `block_until_price`. The pipeline and the fill check never buy a blocked stock.
- **Buy:** belongs to no half, so the next pipeline trade run sells it.
- A pending row with `"at_ask": true` (manual replacement buys) is retried by the 9 AM check at a limit <= the ask.

## Key files
| File | Role |
|---|---|
| `run_all.py` | The pipeline command (support / trade / fill check / after-close step). |
| `pipeline_watchdog.py` | Retries transient failures for launchd; never retries orders. |
| `paper_trade.py` | Order planning (`plan_orders` → `plan_sleeve_orders`) and live sending, 9 AM fill check (`complete_unfilled_orders`). |
| `sleeves.py` | The two halves: plan, ledger, sector rotation weights. |
| `backtest_engine.py` | The live engine despite its name: rules (`WINNER`), scores, ranking, NYSE calendar, and the simulator the forward test uses. Backtest-only code, tests and pins were removed Oct 19, 2026 (forward testing only). |
| `manual_trade.py` | Manual orders, blocks, replacement buys. |
| `decision_ranks.py` | Rank snapshot per decision (`Reports/decision_ranks.csv`, for the Last decision view). |
| `alpaca_paper.py` | Read-only Alpaca client (GET only). |
| `earnings_stop.py`, `trade_audit.py`, `tax_lots.py` | Earnings-day stop (off), trade audit, tax lots. |
| `forward_test.py`, `forward_combos.py` | Forward test of the account and paper strategies since Oct 2, 2026. Includes 2 paper lines "200-day breakout, N days above, exit 2% below" (N = 2, 4; N = 1, 3, 5 removed Oct 21, 2026): buy at the close of the Nth consecutive close above the 200-day SMA after a cross, 10% each (max 10, calmest first); sell at +20% or a close 2%+ below the 200-day (close < 0.98 x SMA200); recorded by the 3:05 PM CT after-close step. Also 5 event lines (same slots): Tech score 80/60 + TP20, Tech score 70/60/90 (percentile of `Technical_Score` among the day's stocks), MA10 x MA30, MA50 x MA200 (golden cross) (fresh crosses, bought on the 2nd close), plus MACD cross, Donchian 20-day breakout, Bollinger mean reversion (20, 2), EMA 9/21 cross, EMA 9/21 + ADX>25 (ADX from closes), Minervini trend template, 52-week high momentum, Dual MA + RSI filter, Golden cross + MACD confirm. "Current Strategy (Live from Oct End)" (added Oct 21, 2026 as "Live 50/50, top 5 each half", from Oct 2) = the live rules from the sector half's Fri Oct 23 and the low vol half's Fri Oct 30 rebalance: the 50/50 with each half holding its top 5 (backtest_engine.WINNER["n"] = 5, sleeves.sector_args; the previous top-10 rules stay as the forward row "Live 50/50, top 10 each half"; leftover-cash next-in-line buys only refill a half holding fewer than 5) (forward_combos leg key "<key>@5"; max 2 per sector). From Oct 19, 2026 no paper strategy buys a `do_not_buy` stock (its weight stays cash). The old "Live 50/50 combined (no overlap)" line was removed Oct 20, 2026; the 8 weakest lines (Short-term reversal, 12-1 month momentum, RSI pullback, Worst 3 sectors 3M, Breakout with volume, Relative strength only, MACD cross weekly, MA10 & MA30 x MA50) on Oct 21, 2026 (a near-duplicate of "Live 50/50, sector rotation on Mondays", the marked live line). |
| `sector_mapping.py` | The stock list (84 stocks + QQQ), sectors, `do_not_buy` / `do_not_sell`. |
| `main_signal_analysis.ipynb` (+ 3 support notebooks) | Signals → `Reports/signal_analysis.csv`, `strategy_picks.csv`, ... (Chirag edits these; leave them alone). |

State files in `Reports/`: `sleeve_ledger.json` (halves: tags + cash, no shares), `live_pending_orders.json` (9 AM rests), `manual_orders.json`,
`manual_block.json`, `run_state.json` (decision done / last daily), `earnings_stop_state.json`, `trade_audit_state.json`;
logs: `run_log.csv`, `live_orders_log.csv`, `logs/`.

Report tables: every `Reports/*.csv` name in the code is a table in SQLite `data/stock_analysis.db` (via `db.py`;
`db.read_csv` / `db.to_csv` take the old paths). The old CSV files are backed up in `backups/csv_backup_2026-10-26.zip`.

## Dashboard (`app.py`, `dashboard/`)
- **Home:** My notes (`Reports/my_notes.csv`), stock list (Keep / Buy next / Sell next / Next in line / Others, DNB tag;
  `dashboard/stock_groups.py`), stock view (signal_analysis.csv + live quote), sector performance table.
- **Strategy:** Last decision (Alpaca fills + ledger + decision_ranks.csv), Live vs benchmark (Alpaca portfolio history vs
  QQQ/SPY), Forward test (forward_*.csv), Strategy rules and paper holdings, Data freshness, What each strategy does. The next-rebalance plan compares picks with the live Alpaca positions and marks a do-not-buy or manually blocked new buy as Blocked.
- **Trading Account:** Live holdings (Alpaca, read-only), Trade audit (live_trade_ledger.csv), Tax view (tax_lots), rules.

## Tests and setup
```
python tests/run_tests.py          # every test file (~90 s); --fast skips the app tests
bash setup_mac_mini.sh             # packages, notebook git filter, launchd jobs (paths from this folder)
```
Tests that pin the Oct 15 weekly C6 rules import `tests/c6_rules.py`; the live rules are covered by
`test_low_vol_monthly.py`, `test_sleeves.py`, `test_manual_trade.py`, `test_strategy_tab.py`.
