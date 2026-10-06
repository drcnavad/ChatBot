"""Run the regression tests:  python tests/run_tests.py  [--fast = skip the app / dashboard tests]
  test_midweek_exit_replace.py  live Mon/Wed exit refill (rank>30 -> best top-10 not held; cash only if none left)
  test_midweek_cash_deploy.py  live Mon/Wed spare cash -> best-ranked stocks not held (held untouched, Friday unchanged)
  test_midweek_repro.py  live engine reproduces the pinned backtests exactly
                         ((None,F,F) 470.31/1.4393, (30,F,F) 457.14/1.4451,
                          (30,T,F) 452.96/1.3172, (30,T,T) 387.57/1.2249)
                         and an independent re-implementation of the earnings rule
  test_rank_audit.py     saved ranks, scores, picks, mid-week decisions and earnings skips re-derived independently
  test_runner.py         run_all.py mode choice, trade gate + NewsAPI once-a-day guard (pure logic, no API calls)
  test_run_all_helpers.py run_all.py helpers: _notebook_error_summary
                         extracts cell/line/error from nbconvert output, run_cmd returns (rc, output)
  test_run_all_optional.py run_all.py optional steps: upstream API failures (fundamentals /
                         processing / sentiment / earnings) never stop the pipeline or
                         block trading; only main/validate failures are critical
  test_app.py            Streamlit AppTest: page, charts, displayed ranks, Details widgets, captions, removed sections
  test_paper_account.py  alpaca_paper.py + run_all --sync-live against a local MOCK server
  test_live_holdings.py  Details tab live holdings: fake Alpaca client, FIFO first-buy dates, totals, QQQ row,
                         60 s cache, failure message (no network to Alpaca)
  test_dashboard_http.py the app on test port 8599 answers 200 for / and /?symbol=NVDA (never touches 8502, the running app)
  test_paper_trade_live_safety.py  mocked (zero broker calls) regression tests for the LIVE paper_trade.py:
                         live- / live-fill- order id namespace, paper=False LIVE client from LIVE keys,
                         live ledgers, fail-closed signal status + buying-power guard, stable client order
                         ids + broker reconciliation, sequenced SELL-then-BUY submit, morning
                         abort on unreadable positions, morning crash recovery
  test_fill_check_fractional.py mocked: 2-decimal sizing (whole shares after hours, fractional rest
                         next morning, exact exits) + morning fill check: partial/full/no fill,
                         cancel confirmation, crash between cancel and replace, same-day rerun,
                         market closed, open order on broker
  test_pipeline_watchdog.py mocked tests for pipeline_watchdog.py: failure classification
                         (transient / missing-upstream / unknown), --from resume logic,
                         transient retry policy, no-retry rule for trade/fill-check phases,
                         diagnose-only contract (LLM stubbed, no network)
  test_alpaca_paper_reads.py mocked tests for alpaca_paper.py's read-only views: open_orders,
                         market_clock parsing + the read-only guardrails
                         (live endpoint allowlist, no order-placing code)
  test_catch_up.py       missed-decision catch-up: each decision runs once (2:30 PM slot or the next regular session),
                         superseded at the next slot; weekends wait for the open; holidays; send_now fill gate
  test_alpaca_api_names.py the REAL alpaca-py: every alpaca import / enum member the code uses exists and the
                         exact order + orders-list requests build (no network)
  test_smart_orders.py   smart limit orders (fake broker + fake quotes, real alpaca-py request classes): ask + 0.05% /
                         bid - 0.05%; normal / wide / stale / missing quotes; a skipped order's one retry slot at the next
                         9 AM CT check; partial fills, cancel + replace once, no duplicates; buying-power cap; after hours;
                         Friday's old pending rows; SIP / IEX feed check; order-log price columns + run-log cost row
  test_run_log.py        Reports/run_log.csv: one plain row per event (trim to 1000, never raises), trade /
                         fill-check / finished rows (which run, steps OK, money moved, next, what to do)
  test_live_weights.py   live weights: 99% target, each rounded down to a 2-decimal percent (never over 100%),
                         halving on top, build_orders still rejects >100%, the 1% cushion does not trim a 99% plan
  test_band_hold.py      band-hold rebalance math (above / below / mixed band holds, worst case, regime off, E5 skip,
                         sector cap, rounding, 99% once, sells fund buys), the trim fix (live + simulate) + fake-broker
                         runs of Fri 10/2 (incl. a held stock removed from the list) and Mon 10/5
  test_single_universe.py  sector_mapping.py is the only stock list: no hardcoded ticker list / symbol map / C6-U<n>
                         tag / "NN stocks" count in any other .py or notebook; derived lists (engine, autofetch, call
                         counts, bar cache) equal sector_mapping's
  test_decision_bars.py  decision days keep their 2:30 PM bar (saved as ratios, split-safe; never overwritten after the
                         trade; unchanged bars when nothing is saved or the file is bad)
  test_forward_test.py   forward testing from FORWARD_START (2026-10-02): per-stock panel cutoff / empty state;
                         forward_test.py picks, trade cost, leaderboard + ranking rule, daily record (fake account, no
                         requests); the paper strategies: registry runs, idempotent, missed days caught up, accounting,
                         dip / ATR dip buys / earnings drift / risk parity / VWAP / ATR stops (replay = live, stop + refill, open
                         sales) by hand, fake bars (no download), live rules =
                         live Strategy_Weight, no live file / WINNER change
  test_earnings_stop.py  live pre-earnings 3x ATR stop (mocked broker, quotes and bars): window through the reaction
                         day, sessions, stop = engine ATR, one whole-share limit SELL (extended_hours outside regular
                         hours), fraction to 9 AM, never twice, fails closed; dry run sends and writes nothing
  test_safety_guards.py  20% max weight per stock (extra stays cash); mocked: BUYs over 20% of equity refused (backstop);
                         a missed scheduled decision gets one run-log warning while it waits
  test_tax_lots.py       tax view, hand-worked cases: fresh start at TAX_START (older activity, prior years and
                         wash matching against older trades ignored, pre-start shares left out), FIFO / HIFO / LIFO lots, partial and fractional fills, fees, wash
                         sales before / after a loss sale with partial matching, basis + holding-period carryover, cascades,
                         splits, transfers, short / long-term boundary (leap day), dividends (qualified estimate),
                         netting + $3,000 limit + carryforward, 2026 brackets, NIIT, harvest exclusions, Form 8949 rows,
                         data checks; the app renders it with a fake account (GET only)
  test_trade_audit.py    trade audit, hand-worked: order source + plan price from the client id, slippage vs plan,
                         FIFO round trips, reconciliation (manual / orphan orders, pending rows, untracked holdings),
                         alerts (daily loss, clock drift, NYSE calendar vs Alpaca, rejects, slippage, API), once-only
                         run-log rows; the app panel with a fake account (GET only)
Only the files in TESTS run (never Archive/). Read-only for the project (temporary files only, deleted afterwards). No quota APIs, no Alpaca account calls."""
import os
import subprocess
import tempfile
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = ["test_midweek_repro.py", "test_rank_audit.py", "test_runner.py", "test_run_all_helpers.py", "test_run_all_optional.py",
         "test_app.py", "test_paper_account.py", "test_dashboard_http.py", "test_paper_trade_live_safety.py", "test_fill_check_fractional.py",
         "test_paper_trade_math_audit.py", "test_run_log.py", "test_rebalance_rules.py", "test_pipeline_watchdog.py", "test_alpaca_paper_reads.py",
         "test_live_rules_audit.py", "test_catch_up.py",
         "test_alpaca_api_names.py", "test_smart_orders.py", "test_live_weights.py", "test_band_hold.py", "test_single_universe.py",
         "test_decision_bars.py", "test_refresh_job.py", "test_live_holdings.py", "test_forward_test.py", "test_earnings_stop.py",
         "test_safety_guards.py", "test_tax_lots.py",
         "test_trade_audit.py",
         "test_midweek_exit_replace.py", "test_midweek_cash_deploy.py"]


def main():
    fast = "--fast" in sys.argv
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1", STOCK_ANALYSIS_LIVE_HOLDINGS="off",
               PYTHONWARNINGS="ignore",   # tests never write the real Reports/run_log.csv
               STOCK_ANALYSIS_RUN_LOG=os.path.join(tempfile.mkdtemp(), "run_log.csv"),
               STOCK_ANALYSIS_EARNINGS_STOP_STATE=os.path.join(tempfile.mkdtemp(), "earnings_stop_state.json"))   # never the real stop sales
    results = []
    for t in TESTS:
        if fast and t in ("test_app.py", "test_dashboard_http.py"):
            continue
        if not os.path.exists(os.path.join(HERE, t)):
            continue
        t0 = time.time()
        p = subprocess.run([sys.executable, os.path.join(HERE, t)], cwd=ROOT, env=env, capture_output=True, text=True)
        ok = p.returncode == 0
        results.append((t, ok, time.time() - t0))
        tail = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()][-4 if ok else -25:]
        print(f"{'PASS' if ok else 'FAIL'}  {t}  ({time.time() - t0:.0f}s)")
        for line in tail:
            print("      " + line)
    n_ok = sum(ok for _, ok, _ in results)
    print(f"\n{n_ok}/{len(results)} test files passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
