# Stock Analysis

An automated stock strategy that trades a **real-money Alpaca LIVE account**, plus a Streamlit dashboard
(http://localhost:8502, launchd `com.stockanalysis.dashboard-8502`). Runs on Chirag's Mac mini (`~/Stock Analysis`,
Python `/opt/anaconda3/bin/python`, times in US Central). Everything the pipeline writes goes to `Reports/` (not committed).
Keys live in `.env` (`ALPACA_LIVE_KEY_ID` / `ALPACA_LIVE_SECRET_KEY`, never print or commit them).

> **Never place orders when testing.** No `run_all.py --trade` / `--fill-check`, no `paper_trade.py --submit`, no
> `manual_trade.py buy|sell` without `--dry-run`. Tests mock Alpaca (`tests/run_tests.py` sets
> `STOCK_ANALYSIS_LIVE_HOLDINGS=off` and temp paths for every state file). Read-only checks are fine:
> `paper_trade.py --from-account` (dry run), `trade_audit.py`.

## Live strategy (`backtest_engine.WINNER`, from the 2:30 PM CT run of Tue Oct 27, 2026)
"Live: RS short-term": one line with its own cash and share counts in `Reports/sleeve_ledger.json` (v4,
`sleeves.py`), run exactly as its `forward_test.py` line:
- **From Fri Nov 6, 2026 (locked ~12 weeks, to ~Jan 29, 2027): "Short-term relative strength, skip stretched buys":**
  relative strength over 10/21/63 days (63-day counted double from Fri Nov 6, 2026), max 4 per sector, top 10 equal weight (~9.9% each), 99% invested; full rebalance every Friday
  2:30 PM CT only (Thursday on a holiday), no Mon-Thu trades. A stock more than 15% above its 20-day
  average is not newly bought (held ones kept); blocked picks are replaced down to rank 20.
- No new buy with earnings within 5 days; QQQ below its 200-day average: weights halved. No idle cash: the weight of a
  blocked or unpriced pick goes pro rata to the names that trade (1% cash reserve).
- Decisions use the ~2:30 PM CT bar (Friday = full rebalance; Mon-Thu runs place no orders). A holding within 1 point of
  its target is not traded. A position the ledger does not own is sold at the next trade run.
- Never bought: `sector_mapping.do_not_buy`, manual sell blocks (`Reports/manual_block.json`), stocks sold by the
  earnings-day stop. Never sold: `sector_mapping.do_not_sell`. `earnings_stop.py` is ON: 5% earnings-day stop
  (`WINNER['earnings_day_stop'] = True`, Chirag Nov 4, 2026; do_not_sell exempt).
- Orders: sells first, every order a LIMIT at the live quote (buy ask + 0.05%, sell bid - 0.05%; stale / >0.5% spread
  quotes skipped). Market hours: 2-decimal shares; after hours: whole-share extended-hours DAY limits until 7 PM CT. The
  rest goes to `Reports/live_pending_orders.json` and the **9 AM CT fill check** sends it. Client order ids make every
  send idempotent; a missed decision is caught up once at the next session.

**Holdings: one source** (Chirag, Oct 19, 2026): live Alpaca positions, read only through `holdings.py`
(`get_positions()`, 60 s cache; order code reads fresh through its trading client). Nothing is recorded as owned before a
fill. `Reports/sleeve_ledger.json` keeps the line's share count per symbol and its cash. A queued buy (`Reports/live_pending_orders.json`) shows as "Pending order", never as a
holding. There is no `my_positions.csv` any more.

**Removing a stock** (Oct 19, 2026): comment it out of `stock_symbols` in `sector_mapping.py`. It leaves the rankings,
the dashboard lists and the forward test at once. If the account still holds it, it is no target at the next 2:30 PM CT decision (the
rankings keep only `tradable_symbols`), so it is sold and never rebought; until then it stays in
`Reports/sleeve_ledger.json` (not an orphan) and is priced from Alpaca market data. Removed Oct 19: PG, EMR, PFE, plus ECL, SHW, NUE, URI, SHOP, PEP, VMC, CRCL, ETN (not held).

## Schedule (launchd, `launchd/*.plist`, installed by `launchd/install_schedule.sh`; times CT)
| Job | When | What | Trades? |
|---|---|---|---|
| `com.stockanalysis.support` | Mon 1:00, Tue 1:07, Wed 1:14, Thu 1:21, Fri 1:28 PM | `run_all.py --support --scheduled`: company reports, news sentiment, earnings dates (quota APIs) | no |
| `com.stockanalysis.evening` | Mon-Fri 2:30 PM + 3:05 PM, at login, on wake, every 30 min | `pipeline_watchdog.py --trade --scheduled` → `run_all.py`: 2:30 PM signals + decision + orders (orders on Friday only); from 3:05 PM the after-close step (`run_all.daily_step`: signal refresh, `forward_test.py --record`, `manual_trade.py update-blocks`). Never 4:05-4:30 PM. | **yes** (2:30 PM) |
| `com.stockanalysis.morning` | Mon-Fri 9:00 AM | `pipeline_watchdog.py --fill-check --scheduled`: sends the pending rests | **yes** |
| `com.stockanalysis.tradeaudit` | Mon-Fri every 30 min 8:30 AM-3:00 PM + 3:45 PM | `refresh_dashboard_data.sh`: `trade_audit.py --write --log` (ledger, round trips, alerts) | no (GET only) |
| `com.stockanalysis.earningsstop` | every 5 min + at login | `earnings_stop.py --loop`: 5% earnings-day stop | **yes** |
| `com.stockanalysis.dashboard-8502` | always on | `run_dashboard_8502.sh` → `streamlit run app.py`. Restart: `launchctl kickstart -k gui/$(id -u)/com.stockanalysis.dashboard-8502` | no |

## Manual orders (`manual_trade.py`, separate from the pipeline)
```
python manual_trade.py sell KO [--block-until-drop 10] [--no-replace] [--dry-run]
python manual_trade.py buy XYZ [--amount 1000] [--dry-run]     # default amount: 10% of equity (one slot)
python manual_trade.py blocks                                   # list active blocks
```
- Places the order at once on the LIVE account (DAY limit, pipeline pricing); refuses outside market hours. Every
  action (also dry runs / refusals) is appended to `Reports/manual_orders.json`.
- **Sell:** the stock leaves the sleeve ledger. Once the sell fills, the freed cash buys the **next-in-line stock** (best-ranked not held / blocked / do-not-buy), limit never above the ask,
  (`--no-replace` keeps the cash).
- **Block rule:** `--block-until-drop N` adds `Reports/manual_block.json[SYM] = {reference_close, block_until_price =
  close x (1 - N%), added}`. The after-close step sets `reference_close` to that day's official close and removes the
  block once a close is at or below `block_until_price`. The pipeline and the fill check never buy a blocked stock.
- **Buy:** not in the ledger, so the next pipeline trade run sells it.
- A pending row with `"at_ask": true` (manual replacement buys) is retried by the 9 AM check at a limit <= the ask.

## Key files
| File | Role |
|---|---|
| `run_all.py` | The pipeline command (support / trade / fill check / after-close step). |
| `pipeline_watchdog.py` | Retries transient failures for launchd; never retries orders. |
| `paper_trade.py` | Order planning (`plan_orders` → `plan_sleeve_orders`) and live sending, 9 AM fill check (`complete_unfilled_orders`). |
| `sleeves.py` | The live RS short-term line: plan, ledger, no-idle-cash fill. |
| `backtest_engine.py` | The live engine despite its name: rules (`WINNER`), scores, ranking, NYSE calendar, and the simulator the forward test uses. Backtest-only code, tests and pins were removed Oct 19, 2026 (forward testing only). |
| `manual_trade.py` | Manual orders, blocks, replacement buys. |
| `decision_ranks.py` | Rank snapshot per decision (`Reports/decision_ranks.csv`, for the Last decision view). |
| `trade_records.py` | Record keeping only (no orders): Friday skip reasons, ledger snapshots, plan-vs-fill recap, ops banner, contribution vs QQQ. |
| `alpaca_paper.py` | Read-only Alpaca client (GET only). |
| `earnings_stop.py`, `trade_audit.py`, `tax_lots.py` | Earnings-day stop (on), trade audit, tax lots. |
| `forward_test.py`, `forward_combos.py` | Forward test of the account and every paper strategy (`ft.STRATEGIES`; the live one is `ft.LIVE`), recorded by the 3:05 PM CT after-close step. |
| `sector_mapping.py` | The stock list (84 stocks + QQQ), sectors, `do_not_buy` / `do_not_sell`. |
| `main_signal_analysis.ipynb` (+ 3 support notebooks) | Signals → `Reports/signal_analysis.csv`, `strategy_picks.csv`, ... (Chirag edits these; leave them alone). |

State files in `Reports/`: `sleeve_ledger.json`, `live_pending_orders.json` (9 AM rests), `manual_orders.json`,
`manual_block.json`, `run_state.json` (decision done / last daily), `earnings_stop_state.json`, `trade_audit_state.json`,
`decision_skips.csv` (why a Friday pick was not newly bought), `ledger_history/` (dated ledger copies),
`fill_recap.csv` / `fill_recap_summary.csv` (plan vs fill after the 9 AM check);
logs: `run_log.csv`, `live_orders_log.csv`, `logs/`.

Report tables: every `Reports/*.csv` name in the code is a table in SQLite `data/stock_analysis.db` (via `db.py`;
`db.read_csv` / `db.to_csv` take the old paths).

## Dashboard (`app.py`, `dashboard/`)
- **Home:** My notes (`Reports/my_notes.csv`), stock list (Keep / Buy next / Sell next / Next in line / Others, DNB tag;
  `dashboard/stock_groups.py`), stock view (signal_analysis.csv + live quote), sector performance table.
- **Strategy:** Decisions table (one row per stock: rank at the last Friday decision from the saved snapshot in
  `decision_ranks.csv`, rank at the next decision as a preview at the latest close, the previewed next action via
  `sleeves.plan(rebalance=True)` read-only; needs only Alpaca positions + equity, the ranks show even without them),
  Forward test (forward_*.csv: Top 5 consensus, current holdings, chart, leaderboard), Data freshness and settings.
- **Trading Account:** Live holdings (Alpaca, read-only), Trade audit (live_trade_ledger.csv), Tax view (tax_lots), rules.

## Tests and setup
```
python tests/run_tests.py          # every test file (~90 s); --fast skips the app tests
bash setup_mac_mini.sh             # packages, notebook git filter, launchd jobs (paths from this folder)
```
Tests that pin the Oct 15 weekly C6 rules import `tests/c6_rules.py`; the live rules are covered by
`test_live_rs_short.py`, `test_manual_trade.py`, `test_strategy_tab.py`.
