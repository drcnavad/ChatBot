"""Build (and optionally submit to the Alpaca LIVE account - REAL MONEY) the orders that move a portfolio to the strategy's target weights.

DEFAULT = DRY RUN: prints the order list and writes nothing to any broker.

    python paper_trade.py --account-size 25000                       # dry run, assumes an all-cash account
    python paper_trade.py --account-size 25000 --positions my.csv    # dry run vs current holdings (CSV: Symbol,Shares)
    python paper_trade.py --target provisional                       # use "if rebalanced at latest close" weights
    python paper_trade.py --target midweek --positions my.csv        # only the latest Mon/Wed mid-week swap(s) / exit(s)

Submitting talks to the LIVE account (real money):

    python paper_trade.py --submit [--account-size N]

In submit mode the LIVE account's equity (unless --account-size is given) and positions are read from Alpaca,
sells are sent before buys as DAY market orders (whole shares).

Auto mode (used by run_all.py --trade): auto_trade() pulls LIVE positions + equity +
cash, then plans with target="auto" (buys = Weight * equity, whole shares, capped at
Alpaca buying power - margin is disabled so buying power equals cash in practice; sells = the exact shares held, and only for stocks
actually in the portfolio), and submits as extended-hours DAY limit orders at the planned
closing price, so they can fill in the after-hours session. Before planning, the signal
CSVs are verified fresh (as of today, CT; for a missed decision caught up later, as of the last complete session and
not older than that decision) - stale data aborts the trade. A catch-up in regular hours stages its orders (send_now)
and run_all sends them at once as market orders. Submitted orders are recorded
in Reports/live_pending_orders.json; the next trading morning, complete_unfilled_orders()
(run_all.py --fill-check) checks their fills and completes any unfilled remainder with
regular-hours market orders, then reconciles live positions against the strategy targets.
Failures send a macOS notification + loud log alert. If auto_trade() runs at or after 7:00 PM CT (extended hours
are over), it submits nothing and instead stages the planned orders in
Reports/live_pending_orders.json (no broker order id); the morning fill check then sends
the full quantities as regular-hours market orders. Evening submission is idempotent: rows
already submitted/staged earlier the same evening are recorded after each submit, so a
retry in the same window skips them instead of duplicating orders. See auto_trade() docstring.

Targets come from Reports/strategy_picks.csv (written by main_signal_analysis.ipynb):
  - `current`     = Strategy_Weight (portfolio decided at the last weekly rebalance)
  - `provisional` = Provisional_Weight (what the rules would pick at the latest close)
  - `midweek`     = the decisions of the latest Mon/Wed mid-week check (Reports/strategy_midweek_check.csv, live rules):
                    swap = SELL all shares of the stock that fell below rank 15 and BUY the new top-3 stock with the same dollars
                    (without --positions the dollars = the old stock's target weight x account size); exit = SELL all shares of
                    a stock ranked worse than 30, SELL-ONLY (the cash stays idle until the Friday rebalance). Nothing else is traded.
  - `auto` (default) = provisional on a rebalance day (the decision day itself); midweek when the latest bar is a Mon/Wed
                    check that produced a swap or an exit; hold on a quiet mid-week day (no swap/exit at the
                    latest check) - every position is left unchanged, matching the backtest (no drift
                    rebalance, and cash from a mid-week exit stays idle until Friday); otherwise current.
Earnings rule (backtest_engine.WINNER["earnings_block_days"] = 5): the targets leave out stocks not held with earnings
within 5 calendar days. The live planner adds its own check for the LIVE account: a pick with earnings within 5 days is
not newly bought or topped up (a held stock is never sold because of earnings).
"""
import argparse
import json
import math
import os
import re
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PICKS_CSV = os.path.join(PROJECT_ROOT, "Reports", "strategy_picks.csv")
MIDWEEK_CSV = os.path.join(PROJECT_ROOT, "Reports", "strategy_midweek_check.csv")
SIGNAL_CSV = os.path.join(PROJECT_ROOT, "Reports", "signal_analysis.csv")
CHANGES_CSV = os.path.join(PROJECT_ROOT, "Reports", "strategy_changes.csv")
ORDER_LOG_CSV = os.path.join(PROJECT_ROOT, "Reports", "live_orders_log.csv")
PENDING_ORDERS_JSON = os.path.join(PROJECT_ROOT, "Reports", "live_pending_orders.json")
# NOTE (2026-09-28): LIVE account (real money). paper_* names are historical, kept so the
# pipeline keeps working unchanged. paper=False, ALPACA_LIVE_* keys, live- order ids,
# live_* ledgers. The paper version was retired; no paper rollback is kept.
ORDER_COLUMNS = ["Symbol", "Side", "Shares", "Price", "Est_Value", "Current_Shares", "Target_Shares",
                 "Target_Weight_%", "Target_Value"]
NO_TRADE_BAND = 0.01   # Friday rebalance: a target holding within 1 percentage point of its weight is not traded
CASH_CUSHION = 0.01    # buys are sized to free cash / (1 + 1%) so market fills a bit above the estimate still fit


# ----------------------------------------------------------------------------- notifications, terminal formatting, data freshness
NO_POPUPS_ENV = "STOCK_ANALYSIS_NO_POPUPS"   # set to 1 to print alerts without macOS pop-ups (the tests do)
NOTIFY_TITLE_MAX = 40                        # macOS cuts longer titles
NOTIFY_BODY_MAX = 200                        # ...and longer bodies; full detail goes to the log


def _clip(text, limit):
    """Collapse whitespace and cut to `limit` characters (ending in an ellipsis). Pure."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "\u2026"


def _popups_off():
    return os.environ.get(NO_POPUPS_ENV, "").strip() not in ("", "0")


def _notify(title, message, details=None):
    """Alert the user (macOS notification + loud print).

    title: short (<= 40 chars). message: plain words (<= 200 chars) saying what
    happened, whether money moved / orders went out, the symbols, and what to do.
    details: optional longer text that is only printed (it lands in Reports/logs and
    the launchd logs). Set STOCK_ANALYSIS_NO_POPUPS=1 to skip the pop-up (tests do,
    so running them never shows real alerts). Best effort: never raises.
    """
    title, message = _clip(title, NOTIFY_TITLE_MAX), _clip(message, NOTIFY_BODY_MAX)
    banner = (f"\n{'!' * 70}\n  ALERT: {title}\n  {message}\n"
              + (f"  Details: {details}\n" if details else "") + f"{'!' * 70}\n")
    print(banner, flush=True)
    if _popups_off():
        return
    try:
        import subprocess
        safe_title = title.replace('"', "'").replace("\\", "")
        safe_msg = message.replace('"', "'").replace("\\", "")
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{safe_msg}" with title "{safe_title}" sound name "Basso"'],
            timeout=5, capture_output=True)
    except Exception:
        pass  # the print above is the fallback


def _fmt_shares(q):
    """'4', '0.82', '1.05' - share counts for alerts (2 decimals max). Pure."""
    try:
        v = float(q)
    except (TypeError, ValueError):
        return "?"
    return f"{round(v, 2):g}" if math.isfinite(v) else "?"


def _list_syms(pairs, limit=3):
    """'ENPH 4, FIG 6, HIMS 15 +3 more' from (symbol, shares) pairs. Pure."""
    pairs = list(pairs)
    txt = ", ".join(f"{s} {_fmt_shares(q)}" for s, q in pairs[:limit])
    return txt + (f" +{len(pairs) - limit} more" if len(pairs) > limit else "")


def _trade_notice(results, prices=None):
    """(title, body, details) for the evening trade alert, from a results DataFrame
    (columns Symbol, Side, Shares, Status). prices: optional {symbol: price} for the
    dollar estimate. Pure: safe to unit-test."""
    prices = prices or {}
    ok = results["Status"].str.contains("submitted|STAGED", case=False, na=False)
    staged = ok & results["Status"].str.contains("STAGED", na=False)
    sent = ok & ~staged

    def _pairs(side):
        return [(r.Symbol, r.Shares) for r in results[ok & (results["Side"] == side)].itertuples()]

    def _usd(pairs):
        tot = 0.0
        for sym, q in pairs:
            try:
                v = float(q) * float(prices.get(sym) or 0)
            except (TypeError, ValueError):
                continue
            tot += v if math.isfinite(v) else 0.0
        return f" (~${tot:,.0f})" if tot > 0 else ""

    sells, buys = _pairs("SELL"), _pairs("BUY")
    if sent.any():
        title = f"Trades sent: {len(sells)} sell, {len(buys)} buy"
        tail = ("Money moves as they fill; small leftovers finish at 9 AM. Nothing to do."
                if not staged.any() else
                f"{int(staged.sum())} small order(s) go out at 9 AM. Nothing to do.")
    elif results.loc[staged, "Status"].str.contains("catch-up", na=False).all():
        title = f"Catch-up trades: {len(sells)} sell, {len(buys)} buy"
        tail = "A missed decision, traded now with market orders. Nothing to do."
    else:
        title = f"Trades queued: {len(sells)} sell, {len(buys)} buy"
        tail = "Not sent yet, no money moved. They go out at the 9 AM check. Nothing to do."
    for limit in (3, 2, 1):
        parts = ([f"Sell {_list_syms(sells, limit)}{_usd(sells)}"] if sells else []) + \
                ([f"Buy {_list_syms(buys, limit)}{_usd(buys)}"] if buys else [])
        body = f"{'; '.join(parts)}. {tail} Log: Reports/live_orders_log.csv"
        if len(body) <= NOTIFY_BODY_MAX:
            break
    details = "; ".join(f"{r.Side} {r.Symbol} {r.Shares} [{r.Status}]" for r in results[ok].itertuples())
    return title, body, details


def _print_header(title):
    """A clear section header so launchd/terminal logs are easy to scan."""
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}", flush=True)


def _print_section(title):
    print(f"\n  --- {title} ---", flush=True)


def check_signal_freshness(picks_csv=PICKS_CSV, changes_csv=CHANGES_CSV, decision=None, now=None):
    """Verify the signal CSVs are fresh before any trading.

    Without `decision`: both files must be as of today (CT). With `decision` (the decision date being traded, e.g. a
    missed Friday caught up on Monday morning): the files must be as of the latest complete session at `now` (today
    after 4:30 PM ET, else the previous session) and not older than the decision - so a catch-up accepts the
    decision's own files but never older ones.
    Fail-closed: raises ValueError when the data is stale or missing, so auto_trade never trades on old signals after
    a partial pipeline failure. Returns the As_Of date string when fresh.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = now or datetime.now(ZoneInfo("America/Chicago"))
    today = now.date().isoformat()
    if decision is not None:
        import backtest_engine as be
        today = be.last_complete_session(now).date().isoformat()   # the date the files must carry
        if today < pd.Timestamp(decision).date().isoformat():
            raise ValueError(f"STALE DATA: the {pd.Timestamp(decision):%Y-%m-%d} decision's close is not final yet")
    for path, label in [(picks_csv, "strategy_picks.csv"), (changes_csv, "strategy_changes.csv")]:
        if not os.path.exists(path):
            raise ValueError(f"STALE DATA: {label} not found - run the pipeline before trading")
    picks = pd.read_csv(picks_csv)
    if picks.empty or "As_Of" not in picks.columns:
        raise ValueError("STALE DATA: strategy_picks.csv has no usable As_Of - run the pipeline before trading")
    as_of = str(picks["As_Of"].iloc[0])[:10]
    if as_of != today:
        raise ValueError(
            f"STALE DATA: strategy_picks.csv is as of {as_of}, but it should be {today} (CT) - "
            "the pipeline did not produce fresh signals; refusing to trade")
    # Every consumed input must be fresh: the buy/hold signal statuses come from
    # strategy_changes.csv. A stale changes file would silently turn every target
    # into HOLD (fail closed but no trading) or, worse, let a manual run trade on
    # yesterday's statuses - so its decision date is verified too.
    changes = pd.read_csv(changes_csv)
    if changes.empty or "Date" not in changes.columns:
        raise ValueError("STALE DATA: strategy_changes.csv has no usable Date column - run the pipeline before trading")
    changes_date = str(changes["Date"].astype(str).str[:10].max())
    if changes_date != today:
        raise ValueError(
            f"STALE DATA: strategy_changes.csv is as of {changes_date}, but it should be {today} (CT) - "
            "the pipeline did not produce fresh signal statuses; refusing to trade")
    return as_of


# ----------------------------------------------------------------------------- target weights and prices (from the Reports CSVs)
def latest_midweek_swaps(midweek_csv=MIDWEEK_CSV, as_of=None):
    """Swap and exit rows (Action SWAP / SELL; Sell, Sell_Rank, Buy, Buy_Rank, Weight_%, Message, Event_Date,
    Applies_To_Open) of the latest mid-week check, only if that check was made at the latest bar (`as_of`). Exit rows
    (Action SELL) have no Buy. Empty DataFrame otherwise."""
    if not os.path.exists(midweek_csv):
        return pd.DataFrame()
    m = pd.read_csv(midweek_csv)
    if m.empty or "Event" not in m.columns:
        return pd.DataFrame()
    m = m[m["Event"] == "mid-week check"]
    if as_of is not None:
        m = m[m["Event_Date"].astype(str) == str(as_of)]
    return m[m["Action"].isin(["SWAP", "SELL"])].reset_index(drop=True)


def load_targets(source="auto", picks_csv=PICKS_CSV, midweek_csv=MIDWEEK_CSV, decision=None):
    """(targets DataFrame[Symbol, Weight, Price], meta dict). Weights are fractions of the account (rest = cash).

    meta['swaps'] holds the latest mid-week swap rows (source 'midweek'); targets are then the weights after the swap.
    decision (a date): trade that decision - a missed one caught up later - instead of the latest bar's: its rebalance
    weights (only while the data is as of that Friday) or the swaps/exits of its mid-week check."""
    picks = pd.read_csv(picks_csv)
    if picks.empty:
        raise ValueError(f"{picks_csv} is empty - run main_signal_analysis.ipynb first")
    as_of, last_reb = str(picks["As_Of"].iloc[0]), str(picks["Last_Rebalance"].iloc[0])
    last_dec = str(picks["Last_Decision"].iloc[0]) if "Last_Decision" in picks.columns else last_reb
    day = pd.Timestamp(decision).date().isoformat() if decision is not None else as_of
    swaps = latest_midweek_swaps(midweek_csv, day)
    if source == "auto" and day == last_reb and as_of != day:
        raise ValueError(f"the {day} rebalance weights are gone (data now as of {as_of}) - cannot trade that decision")
    if source == "auto" and day != as_of and day < last_reb:
        raise ValueError(f"the {day} decision is older than the {last_reb} rebalance - superseded, not traded")
    if source == "auto":
        # A quiet mid-week day (no swap/exit at the latest check) holds every position
        # unchanged, matching the backtest: no drift rebalance, and cash from a mid-week
        # exit stays idle until the Friday rebalance.
        source = "provisional" if day == last_reb else ("midweek" if len(swaps) else "hold")
    if source == "midweek" and not len(swaps):
        raise ValueError(f"no mid-week swap or exit at the latest check ({as_of}) - nothing to trade (use target 'current')")
    if source == "hold":
        t = pd.DataFrame(columns=["Symbol", "Weight", "Price"])
        return t, {"as_of": as_of, "last_rebalance": last_reb, "last_decision": last_dec, "source": "hold",
                   "strategy": str(picks["Strategy"].iloc[0]), "invested": None, "swaps": swaps}
    col = {"current": "Strategy_Weight", "provisional": "Provisional_Weight", "midweek": "Strategy_Weight"}[source]
    t = picks.loc[picks[col].fillna(0) > 0, ["Symbol", col, "Close"]].rename(columns={col: "Weight", "Close": "Price"})
    return t.reset_index(drop=True), {"as_of": as_of, "last_rebalance": last_reb, "last_decision": last_dec, "source": source,
                                      "strategy": str(picks["Strategy"].iloc[0]), "invested": float(t["Weight"].sum()),
                                      "swaps": swaps}


def latest_prices(symbols, signal_csv=SIGNAL_CSV):
    """Latest close per symbol from signal_analysis.csv (used to price positions that are not in the targets)."""
    if not symbols:
        return {}
    df = pd.read_csv(signal_csv, usecols=["Date", "Symbol", "Close"], parse_dates=["Date"])
    df = df[df["Symbol"].isin(symbols)].sort_values("Date").drop_duplicates("Symbol", keep="last")
    return dict(zip(df["Symbol"], df["Close"]))


def current_prices(symbols):
    """{symbol: latest trade price} from Alpaca market data (read-only, no account calls) - sizes a daytime catch-up at
    today's prices. {} when unavailable; the plan then uses the decision's closes."""
    try:
        from alpaca.data.requests import StockLatestTradeRequest

        import backtest_engine as be
        trades = be.market_data_client().get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sorted(symbols)))
        return {str(s): float(t.price) for s, t in trades.items() if t is not None and t.price}
    except Exception as e:
        print(f"  current prices unavailable ({e}) - sizing from the decision's closes")
        return {}


def latest_signal_status(changes_csv=CHANGES_CSV, as_of=None):
    """{SYMBOL: 'add'|'hold'|'drop'|...} from the latest decision rows of strategy_changes.csv.

    Used by build_orders(): 'add' and 'hold' picks are both brought to their target weight;
    any other/unknown status is left as is. Returns {} when the file is missing/unreadable -
    callers then treat every target as HOLD (fail closed: no buys without a confirmed signal)."""
    try:
        ch = pd.read_csv(changes_csv)
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
        return {}
    if ch.empty or "Symbol" not in ch.columns or "Status" not in ch.columns:
        return {}
    dates = ch["Date"].astype(str) if "Date" in ch.columns else None
    if dates is not None:
        want = str(as_of) if as_of else dates.max()
        ch = ch[dates == want]
        if ch.empty:
            return {}
    return dict(zip(ch["Symbol"].astype(str).str.upper().str.strip(),
                    ch["Status"].astype(str).str.lower().str.strip()))


def _is_buy_signal(status):
    """True only for a confirmed buy-signal status ('add'/'buy'/'bullish').

    None/unknown is NOT a buy signal (fail closed): a symbol whose signal status cannot be
    confirmed - strategy_changes.csv missing, unreadable, or simply no row for it - is
    never bought. build_orders() emits a HOLD row for it instead."""
    if status is None:
        return False
    return str(status).strip().lower() in ("add", "buy", "bullish")


def _is_hold_signal(status):
    """True for a confirmed 'hold' status (the strategy keeps the pick). Unknown is False."""
    return status is not None and str(status).strip().lower() == "hold"


def _safe_number(x, default=0.0):
    """Return float(x), or `default` when x is None/NaN/inf/non-numeric.

    NaN is truthy in Python so `x or 0` does NOT catch it - math.floor(nan) raises
    ValueError, and NaN silently poisons downstream arithmetic. Use this everywhere a
    broker- or CSV-supplied number feeds quantity math."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _floor2(x):
    """Round DOWN to 2 decimals (500 / 139.66 = 3.580... -> 3.58; 3.5799 -> 3.57).
    Float noise (3.58 stored as 3.57999...) is absorbed before flooring."""
    return math.floor(round(_safe_number(x) * 100, 6)) / 100


def _clean_positions(positions):
    """Normalize a {symbol: shares} map: drop zero/NaN/inf/negative holdings.

    A NaN share count would otherwise flow into build_orders() and produce NaN order
    quantities, which crash math.floor() at submit time. Negative quantities are
    impossible in a long-only account, so they are dropped too."""
    out = {}
    for k, v in (positions or {}).items():
        try:
            shares = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(shares) and shares > 0:
            out[str(k).upper()] = shares
    return out


def earnings_blocked(symbols, as_of, earnings_csv=None):
    """{SYMBOL: 'earnings Wed Sep 30, in 2 days'} for `symbols` whose next earnings date E is within
    the strategy's earnings rule window after the decision date (as_of < E <= as_of + N days, N =
    WINNER['earnings_block_days'], i.e. E5). Used so a pick is not newly bought, and an owned one not
    topped up, right before earnings. {} when the rule is off; an unreadable earnings file
    prints a warning and blocks nothing (same as a stock with no date on file)."""
    try:
        import backtest_engine as be
        n = be.WINNER.get("earnings_block_days")
        if not n or not symbols:
            return {}
        d = pd.Timestamp(str(as_of)).normalize()
        days = be.earnings_days_ahead([d], list(symbols), be.load_earnings(earnings_csv), n).iloc[0]
    except Exception as e:
        print(f"WARNING: earnings rule not checked for live buys ({e}) - no buy was blocked")
        return {}
    return {s: f"earnings {d + pd.Timedelta(days=int(v)):%a %b %d}, in {int(v)} day{'' if int(v) == 1 else 's'}"
            for s, v in days.items() if pd.notna(v)}


# ----------------------------------------------------------------------------- order sizing (whole shares by default)
def build_orders(targets, account_size, positions=None, prices=None, min_value=1.0, fractional=False,
               statuses=None, band=NO_TRADE_BAND, blocked=None):
    """Orders (sells first) to move `positions` {symbol: shares} toward target weights of `account_size` dollars.

    Every target pick ('add' or 'hold') is brought to Weight x account_size: an underweight
    holding is bought up, an overweight one is trimmed (whole shares, or 2-decimal shares with
    fractional=True, always rounded down - a buy never exceeds its weight).
    No-trade band: a target holding whose value is within `band` x account_size of its target
    (1 percentage point by default) is left alone.
    `blocked` {SYMBOL: note}: picks with earnings soon (earnings_blocked): not newly bought (SKIP row)
    and, if owned, not topped up (HOLD instead of BUY); a trim SELL still goes through.
    Positions not in the targets (sell signals) are sold completely (no band).
    Trades worth less than `min_value` are skipped. `statuses` maps SYMBOL -> status string from
    strategy_changes.csv; a missing/unknown status is treated as HOLD (fail closed), never traded.
    """
    if not account_size or account_size <= 0 or not math.isfinite(account_size):
        raise ValueError("account_size must be a positive number")
    positions = _clean_positions(positions)
    px = dict(zip(targets["Symbol"], targets["Price"]))
    px.update({k: v for k, v in (prices or {}).items() if k not in px})
    rows = []
    # Duplicate symbols are ambiguous (which weight wins?) - fail closed and abort
    # instead of silently letting one row overwrite the other in the dict below.
    if len(targets) != targets["Symbol"].nunique():
        raise ValueError("duplicate symbols in targets - refusing to size orders on ambiguous data")
    # The portfolio cannot allocate more than 100%: an over-100% total means the
    # signal output is corrupt - fail closed and abort. (Per-symbol bad weights are
    # still handled as SKIP rows below; only finite weights count toward the total.)
    total_weight = 0.0
    for w in targets["Weight"]:
        try:
            f = float(w)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            total_weight += f
    if total_weight > 1 + 1e-6:
        raise ValueError(f"target weights sum to {total_weight:.4f} (>100%) - refusing to trade on corrupt targets")
    wanted = dict(zip(targets["Symbol"], targets["Weight"]))
    statuses = statuses or {}
    for sym in sorted(set(wanted) | set(positions)):
        # Normalize the price: None/NaN/inf/non-numeric all mean "no usable price"
        # (fail closed -> SKIP). +inf must be rejected explicitly: `inf > 0` is True,
        # so a naive positivity check would let it through, and the submit would then
        # crash on round(inf) (OverflowError) or size nonsense orders.
        price = _safe_number(px.get(sym), default=None)
        cur = positions.get(sym, 0.0)
        # Validate the target weight: NaN/inf/negative/>100% weights are corrupt data -
        # fail closed with SKIP rather than crashing math.floor() or trading on garbage.
        # (A legitimate 0 weight means "sell all" and flows through normally.)
        raw_weight = wanted.get(sym, 0.0)
        try:
            raw_weight_f = float(raw_weight)
        except (TypeError, ValueError):
            raw_weight_f = float("nan")
        if not math.isfinite(raw_weight_f) or raw_weight_f < 0 or raw_weight_f > 1:
            rows.append({"Symbol": sym, "Side": "SKIP (bad weight)", "Shares": 0,
                         "Price": float(price) if price else price, "Est_Value": 0.0,
                         "Current_Shares": cur, "Target_Shares": None,
                         "Target_Weight_%": 0.0, "Target_Value": 0.0})
            continue
        weight = raw_weight_f
        status = statuses.get(sym)
        if sym in wanted and not (_is_buy_signal(status) or _is_hold_signal(status)):
            # Unknown status: fail closed - kept as is, never bought or trimmed.
            rows.append({"Symbol": sym, "Side": "HOLD", "Shares": 0,
                         "Price": float(price) if price else price, "Est_Value": 0.0,
                         "Current_Shares": cur, "Target_Shares": None,
                         "Target_Weight_%": round(weight * 100, 2), "Target_Value": round(weight * account_size, 2)})
            continue
        if sym in wanted and cur <= 0 and sym in (blocked or {}):
            # Earnings rule: a pick the account does not own is not newly bought before earnings.
            rows.append({"Symbol": sym, "Side": f"SKIP ({blocked[sym]})", "Shares": 0,
                         "Price": float(price) if price else price, "Est_Value": 0.0,
                         "Current_Shares": cur, "Target_Shares": None,
                         "Target_Weight_%": round(weight * 100, 2), "Target_Value": round(weight * account_size, 2)})
            continue
        if price is None or not (price > 0):
            rows.append({"Symbol": sym, "Side": "SKIP (no price)", "Shares": 0, "Price": price, "Est_Value": 0.0,
                         "Current_Shares": cur, "Target_Shares": None, "Target_Weight_%": weight * 100, "Target_Value": weight * account_size})
            continue
        target_value = weight * account_size
        if sym in wanted and abs(cur * price - target_value) <= band * account_size + 1e-9:
            # No-trade band: close enough to target (within 1 percentage point) - not traded.
            rows.append({"Symbol": sym, "Side": "HOLD", "Shares": 0, "Price": float(price), "Est_Value": 0.0,
                         "Current_Shares": cur, "Target_Shares": cur, "Target_Weight_%": round(weight * 100, 2),
                         "Target_Value": round(target_value, 2)})
            continue
        if fractional:
            # 2-decimal shares, rounded DOWN (buys and trims); a full exit (target 0)
            # sells the exact held quantity, fraction included.
            tgt = _floor2(target_value / price)
            delta = -cur if tgt == 0 else math.copysign(_floor2(abs(tgt - cur)), tgt - cur)
        else:
            tgt = math.floor(target_value / price)
            delta = round(tgt - cur, 4)
        if delta == 0 or abs(delta) * price < min_value:
            side = "HOLD"
        else:
            side = "BUY" if delta > 0 else "SELL"
        if side == "BUY" and sym in (blocked or {}):
            side = "HOLD"  # earnings soon: an owned pick is not bought more before earnings
        rows.append({"Symbol": sym, "Side": side, "Shares": abs(delta) if side != "HOLD" else 0, "Price": float(price),
                     "Est_Value": round(abs(delta) * price, 2) if side != "HOLD" else 0.0, "Current_Shares": cur,
                     "Target_Shares": tgt, "Target_Weight_%": round(weight * 100, 2), "Target_Value": round(target_value, 2)})
    orders = pd.DataFrame(rows, columns=ORDER_COLUMNS)
    order = {"SELL": 0, "BUY": 1, "HOLD": 2}
    return orders.sort_values(["Side", "Symbol"], key=lambda s: s.map(order).fillna(3) if s.name == "Side" else s).reset_index(drop=True)


def build_swap_orders(swaps, account_size, positions=None, prices=None, fractional=False):
    """Orders for mid-week swaps: SELL every share of `Sell`, BUY `Buy` with the same dollars (shares x latest close).
    Exit rows (Action SELL, no Buy): SELL every share of `Sell` only - the cash stays idle until the weekly rebalance.

    If the account holds no `Sell` shares (no --positions given) the dollars = the swap's target weight x account_size.
    Whole shares (rounded down) by default; fractional=True rounds DOWN to 2 decimals.
    Every other holding is left alone (HOLD rows)."""
    if not account_size or account_size <= 0 or not math.isfinite(account_size):
        raise ValueError("account_size must be a positive number")
    if "Weight_%" not in swaps.columns:
        raise ValueError("mid-week swaps data is missing 'Weight_%' column - rerun main_signal_analysis.ipynb")
    positions = _clean_positions(positions)
    prices = prices or {}
    rows, touched = [], set()
    for r in swaps.itertuples():
        out_sym = str(r.Sell)
        in_sym = str(r.Buy) if isinstance(r.Buy, str) and r.Buy.strip() else None       # None = mid-week exit (sell only)
        # Normalize prices: None/NaN/inf/non-numeric all mean "no usable price".
        # +inf must be rejected explicitly (`inf > 0` is True) - it would otherwise
        # crash the submit on round(inf) (OverflowError).
        p_out = _safe_number(prices.get(out_sym), default=None)
        p_in = _safe_number(prices.get(in_sym), default=None)
        have = positions.get(out_sym, 0.0)
        try:
            w = float(swaps.loc[r.Index, "Weight_%"]) / 100
        except (TypeError, ValueError) as e:
            raise ValueError(f"mid-week swap row {r.Index} has invalid Weight_%: {swaps.loc[r.Index, 'Weight_%']!r}") from e
        dollars = have * p_out if have and p_out else w * account_size
        rows.append({"Symbol": out_sym, "Side": "SELL" if have else "SELL (none held)", "Shares": have, "Price": p_out,
                     "Est_Value": round(have * p_out, 2) if have and p_out else 0.0, "Current_Shares": have, "Target_Shares": 0,
                     "Target_Weight_%": 0.0, "Target_Value": 0.0})
        if in_sym is None:
            touched.add(out_sym)
            continue
        if p_in and p_in > 0:
            # A buy never exceeds its weight limit, net of shares already held: the budget is
            # capped at Weight_% x account_size, the current holding's value is subtracted, and
            # if the holding is already above the limit the excess is SOLD instead of buying.
            budget = min(dollars, w * account_size)
            cur_in = positions.get(in_sym, 0.0)
            net_dollars = budget - cur_in * p_in
            if net_dollars >= 0:
                q = _floor2(net_dollars / p_in) if fractional else math.floor(net_dollars / p_in)
                if (q > 0) if fractional else (q >= 1):
                    rows.append({"Symbol": in_sym, "Side": "BUY", "Shares": q, "Price": float(p_in),
                                 "Est_Value": round(q * p_in, 2), "Current_Shares": cur_in,
                                 "Target_Shares": cur_in + q, "Target_Weight_%": round(w * 100, 2),
                                 "Target_Value": round(budget, 2)})
                else:
                    rows.append({"Symbol": in_sym, "Side": "HOLD", "Shares": 0, "Price": float(p_in),
                                 "Est_Value": 0.0, "Current_Shares": cur_in, "Target_Shares": cur_in,
                                 "Target_Weight_%": round(w * 100, 2), "Target_Value": round(budget, 2)})
            else:
                if fractional:
                    sq = min(_floor2(-net_dollars / p_in), cur_in)
                else:
                    sq = min(math.floor(-net_dollars / p_in), int(math.floor(cur_in)))
                if (sq > 0) if fractional else (sq >= 1):
                    rows.append({"Symbol": in_sym, "Side": "SELL", "Shares": sq, "Price": float(p_in),
                                 "Est_Value": round(sq * p_in, 2), "Current_Shares": cur_in,
                                 "Target_Shares": cur_in - sq, "Target_Weight_%": round(w * 100, 2),
                                 "Target_Value": round(budget, 2)})
                else:
                    rows.append({"Symbol": in_sym, "Side": "HOLD", "Shares": 0, "Price": float(p_in),
                                 "Est_Value": 0.0, "Current_Shares": cur_in, "Target_Shares": cur_in,
                                 "Target_Weight_%": round(w * 100, 2), "Target_Value": round(budget, 2)})
        else:
            rows.append({"Symbol": in_sym, "Side": "SKIP (no price)", "Shares": 0, "Price": p_in, "Est_Value": 0.0,
                         "Current_Shares": positions.get(in_sym, 0.0), "Target_Shares": None,
                         "Target_Weight_%": round(w * 100, 2), "Target_Value": round(dollars, 2)})
        touched |= {out_sym, in_sym}
    for sym in sorted(set(positions) - touched):
        px = _safe_number(prices.get(sym), default=None)
        rows.append({"Symbol": sym, "Side": "HOLD", "Shares": 0, "Price": px, "Est_Value": 0.0, "Current_Shares": positions[sym],
                     "Target_Shares": positions[sym], "Target_Weight_%": None,
                     "Target_Value": round(positions[sym] * px, 2) if px else None})
    orders = pd.DataFrame(rows, columns=ORDER_COLUMNS)
    order = {"SELL": 0, "SELL (none held)": 0, "BUY": 1, "HOLD": 2}
    return orders.sort_values(["Side", "Symbol"], key=lambda s: s.map(order).fillna(3) if s.name == "Side" else s).reset_index(drop=True)


def build_hold_orders(positions, prices=None):
    """HOLD rows for every current position: a quiet mid-week check trades nothing.

    The backtest holds positions unchanged on mid-week days with no swap/exit (no drift
    rebalance), and cash from a mid-week exit stays idle until the Friday rebalance -
    live matches it by emitting only HOLD rows, so nothing is ever submitted.
    """
    positions = _clean_positions(positions)
    prices = prices or {}
    rows = []
    for sym in sorted(positions):
        px = _safe_number(prices.get(sym), default=None)
        rows.append({"Symbol": sym, "Side": "HOLD", "Shares": 0, "Price": px, "Est_Value": 0.0,
                     "Current_Shares": positions[sym], "Target_Shares": positions[sym],
                     "Target_Weight_%": None,
                     "Target_Value": round(positions[sym] * px, 2) if px else None})
    return pd.DataFrame(rows, columns=ORDER_COLUMNS)


def plan_orders(source, account_size, positions=None, picks_csv=PICKS_CSV, signal_csv=SIGNAL_CSV, midweek_csv=MIDWEEK_CSV,
                min_value=1.0, fractional=False, decision=None, live_prices=None):
    """(orders, meta, targets) for any target source; used by the CLI and the app's order preview. No broker calls.
    decision: see load_targets. live_prices {symbol: price}: size with these (a daytime catch-up) instead of the closes."""
    targets, meta = load_targets(source, picks_csv, midweek_csv, decision=decision)
    positions = positions or {}
    live_prices = live_prices or {}
    targets = targets.assign(Price=targets["Symbol"].map(live_prices).fillna(targets["Price"]))
    if meta["source"] == "hold":
        # Quiet mid-week check: the backtest holds every position unchanged (no drift
        # rebalance), so live emits HOLD rows for all current positions - nothing is submitted.
        px = {**latest_prices(sorted(positions), signal_csv), **live_prices}
        return build_hold_orders(positions, px), meta, targets
    if meta["source"] == "midweek":
        syms = sorted(set(meta["swaps"]["Sell"].astype(str)) | set(meta["swaps"]["Buy"].dropna().astype(str)) | set(positions))
        px = latest_prices(syms, signal_csv)
        px.update(dict(zip(targets["Symbol"], targets["Price"])))
        px.update(live_prices)
        return build_swap_orders(meta["swaps"], account_size, positions, px, fractional=fractional), meta, targets
    prices = {**latest_prices([s for s in positions if s not in set(targets["Symbol"])], signal_csv), **live_prices}
    # 'add' and 'hold' statuses come from the latest decision in strategy_changes.csv: both are
    # brought to target (1-point no-trade band); unknown statuses are left as they are.
    statuses = latest_signal_status(as_of=meta["as_of"])
    blocked = earnings_blocked(list(targets["Symbol"].astype(str)), meta["as_of"])
    return build_orders(targets, account_size, positions, prices, min_value=min_value, fractional=fractional,
                        statuses=statuses, blocked=blocked), meta, targets


def apply_buying_power_guard(orders, buying_power, fractional=False, cushion=CASH_CUSHION):
    """Cap total BUY spending at the account's BUYING POWER.

    Weights are fractions of the whole portfolio, so targets are still computed on equity;
    this guard only ensures the plan never tries to spend more than Alpaca will let the
    account spend (an over-sized buy would otherwise be rejected by the broker). Buying
    power is read from the Alpaca account itself. Margin is disabled on this account, so
    buying power equals cash in practice. Buys are scaled down proportionally when their
    total exceeds buying power; a buy scaled to zero shares becomes a
    SKIP (no buying power) row and is never submitted. Dollar values stay at cent precision.
    fractional=True scales to 2-decimal shares (rounded down) instead of whole shares.
    A `cushion` (1%) is kept back: buys may spend at most buying_power / (1 + cushion).
    An unknown/invalid buying-power value fails closed: BUY rows become
    SKIP (no buying power) and are never submitted, instead of going out uncapped.
    Returns the adjusted orders DataFrame.
    """
    orders = orders.copy()
    buys = orders["Side"] == "BUY"
    if not buys.any():
        return orders
    try:
        available = float(buying_power) / (1 + cushion)
    except (TypeError, ValueError):
        available = float("nan")
    if not math.isfinite(available) or available < 0:
        # Fail closed: without a usable buying-power number the buys cannot be capped,
        # so they are skipped instead of submitted uncapped (an uncapped buy could be
        # rejected by the broker).
        for i in orders.index[buys]:
            orders.at[i, "Side"] = "SKIP (no buying power)"
            orders.at[i, "Shares"] = 0
            orders.at[i, "Est_Value"] = 0.0
        return orders
    buy_cost = 0.0
    # Validate every BUY row and recompute the total cost from shares x price - never
    # trust Est_Value for the early return below (it may be stale, understated, or NaN,
    # which would let an invalid or over-budget row slip through uncapped). A row without
    # finite positive shares and a finite positive price becomes SKIP (no buying power)
    # here, before any spending comparison.
    for i in orders.index[buys]:
        price = _safe_number(orders.at[i, "Price"])
        shares = _safe_number(orders.at[i, "Shares"])
        if not (shares > 0) or not (price > 0):
            orders.at[i, "Side"] = "SKIP (no buying power)"
            orders.at[i, "Shares"] = 0
            orders.at[i, "Est_Value"] = 0.0
        else:
            buy_cost += shares * price
    buys = orders["Side"] == "BUY"
    if not buys.any() or buy_cost <= available:
        return orders
    scale = available / buy_cost
    for i in orders.index[buys]:
        price = _safe_number(orders.at[i, "Price"])
        shares = _safe_number(orders.at[i, "Shares"])
        q = _floor2(shares * scale) if fractional else int(math.floor(shares * scale))
        if (q <= 0 if fractional else q < 1) or price <= 0:
            orders.at[i, "Side"] = "SKIP (no buying power)"
            orders.at[i, "Shares"] = 0
            orders.at[i, "Est_Value"] = 0.0
        else:
            orders.at[i, "Shares"] = q
            orders.at[i, "Est_Value"] = round(q * price, 2)
            orders.at[i, "Target_Shares"] = _safe_number(orders.at[i, "Current_Shares"]) + q
    return orders


# ----------------------------------------------------------------------------- positions file, LIVE submission, command line
def read_positions_csv(path):
    """Positions CSV (Symbol,Shares or Symbol,Qty) -> {symbol: shares}."""
    pos = pd.read_csv(path)
    pos.columns = [c.strip().lower() for c in pos.columns]
    if "shares" not in pos.columns and "qty" not in pos.columns:
        raise SystemExit(f"{path}: needs a Shares column for order sizing (Symbol,Shares); a Weight-only file works for the alert only")
    qty = "shares" if "shares" in pos.columns else "qty"
    return dict(zip(pos["symbol"].astype(str).str.upper().str.strip(), pd.to_numeric(pos[qty], errors="coerce").fillna(0)))


def paper_trading_client():
    """Alpaca TradingClient for the LIVE account (paper=False). Historical name, kept for compatibility."""
    from alpaca.trading.client import TradingClient

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    except ImportError:
        pass  # dotenv missing: env vars still work
    key, secret = os.getenv("ALPACA_LIVE_KEY_ID"), os.getenv("ALPACA_LIVE_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("No Alpaca LIVE keys in .env (ALPACA_LIVE_KEY_ID / ALPACA_LIVE_SECRET_KEY).")
    return TradingClient(key, secret, paper=False)


def _plan_qty(symbol, side, shares, positions):
    """(qty, is_exit, skip_reason) for one BUY/SELL row: 2-decimal shares, rounded DOWN.

    A SELL may only go out for shares actually held: a symbol not in the portfolio is
    SKIPPED (never submitted), and the quantity is clamped to the holding. A SELL of
    everything held (a full exit) keeps the EXACT held quantity, fraction included, so
    no dust is left. `positions` None disables the holdings check.
    After-hours callers then floor this to whole shares (limit orders).
    NaN/inf/negative quantities and holdings count as zero (fail closed)."""
    qty = _safe_number(shares)
    if side == "SELL" and positions is not None:
        have = _clean_positions(positions).get(str(symbol).upper(), 0.0)
        if have <= 0:
            return 0, False, "SKIPPED: not in portfolio"
        if qty >= have - 1e-9:
            return have, True, None
    qty = _floor2(qty)
    return (qty, False, None) if qty > 0 else (0, False, "SKIPPED: zero quantity")


def _limit_price(price):
    """Planned (closing) price rounded to cents, or 0 when unusable."""
    p = round(_safe_number(price), 2)
    return p if p > 0 else 0


def submit_paper(orders, positions=None, client=None, order_date=None):
    """Send the BUY/SELL rows as DAY market orders to the Alpaca LIVE account (sells first).

    Regular-hours market orders use fractional shares: quantities are rounded DOWN to 2
    decimals and rows with <=0 shares are skipped (reported as SKIPPED, not sent). A full
    exit sells the exact held quantity (e.g. 10.504 shares), so no dust is left.
    SELL rows are checked against `positions` ({SYMBOL: shares} from the live account when
    given): a symbol not held is SKIPPED, never submitted, and the sell quantity is
    clamped to the shares actually held (see _plan_qty).
    Every submitted order carries a deterministic `client_order_id` (see
    submit_paper_extended); `client`/`order_date` behave the same as there.
    """
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if client is None:
        client = paper_trading_client()
    date_str = order_date or _today_ct().date().strftime("%Y%m%d")
    results = []
    for r in orders[orders["Side"].isin(["SELL", "BUY"])].itertuples():
        qty, _, reason = _plan_qty(r.Symbol, r.Side, r.Shares, positions)
        if reason:
            results.append((r.Symbol, r.Side, r.Shares, reason))
            continue
        req = MarketOrderRequest(symbol=r.Symbol, qty=qty, time_in_force=TimeInForce.DAY,
                                 side=OrderSide.SELL if r.Side == "SELL" else OrderSide.BUY,
                                 client_order_id=_client_order_id(r.Symbol, r.Side, qty, 0, date_str))
        try:
            o = client.submit_order(req)
            results.append((r.Symbol, r.Side, qty, f"submitted ({getattr(o.status, 'value', o.status)})"))
        except Exception as e:  # keep going; report every failure
            results.append((r.Symbol, r.Side, qty, f"FAILED: {e}"))
    return pd.DataFrame(results, columns=["Symbol", "Side", "Shares", "Status"])


def submit_paper_extended(orders, positions=None, record=None, client=None, order_date=None):
    """Send the BUY/SELL rows as DAY limit orders at the planned (closing) price, eligible for
    extended-hours (after-hours) execution on the Alpaca LIVE account (sells first).

    After hours = WHOLE shares only: the planned 2-decimal quantity is floored to whole
    shares for the limit order. The fractional rest (e.g. 0.58 of a planned 3.58, or the
    0.054 of a 1.054-share exit) stays in the pending entry ("qty" = planned, "order_qty" =
    whole shares sent) and the next morning's regular-hours fill check buys/sells it with a
    fractional market order. A row with <1 whole share is STAGED for the morning (when
    `record` is given) instead of submitted; without `record` it is SKIPPED.
    SELL rows are checked against `positions` ({SYMBOL: shares} from the live account when
    given): a symbol not held is SKIPPED, never submitted, and the sell quantity is
    clamped to the shares actually held.
    `record`, when given, is called after each successful submission with the pending-order
    entry dict(symbol, side, qty, limit_price, order_id) so the caller can persist it
    incrementally (a retry then skips already-submitted rows instead of duplicating them).
    Every submitted order carries a deterministic `client_order_id`
    (pa-YYYYMMDD-SIDE-SYMBOL-qty-pricecents, via _client_order_id): a crash between the
    broker submit and the local record cannot duplicate the order on retry - the retry
    reconciles with the broker (see _evening_submitted_on_broker).
    `client`, when given, is the Alpaca client to use (otherwise one is created);
    `order_date` ("YYYYMMDD") stamps the client order ids (defaults to today, CT).
    Returns a DataFrame [Symbol, Side, Shares, Order_ID, Status]; Order_ID is None when not submitted.
    """
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest

    if client is None:
        client = paper_trading_client()
    date_str = order_date or _today_ct().date().strftime("%Y%m%d")
    results = []
    for r in orders[orders["Side"].isin(["SELL", "BUY"])].itertuples():
        planned, is_exit, reason = _plan_qty(r.Symbol, r.Side, r.Shares, positions)
        if reason:
            results.append((r.Symbol, r.Side, r.Shares, None, reason))
            continue
        price = _limit_price(r.Price)
        if not price:
            results.append((r.Symbol, r.Side, planned, None, "SKIPPED: no price"))
            continue
        # A BUY sized down to the cash free right now keeps its FULL plan in the pending entry
        # (Plan_Shares): the 9 AM check buys the rest that fits then.
        plan_total = max(planned, _floor2(getattr(r, "Plan_Shares", planned))) if r.Side == "BUY" else planned
        entry = {"symbol": r.Symbol, "side": r.Side, "qty": plan_total, "limit_price": price,
                 "order_id": None, "exit": is_exit}
        qty = int(math.floor(planned))  # after hours: whole shares only
        if qty < 1:
            if record is None:
                results.append((r.Symbol, r.Side, planned, None, "SKIPPED: <1 whole share"))
            else:
                record(entry)  # the morning fill check sends it as a fractional market order
                results.append((r.Symbol, r.Side, planned, None,
                                "STAGED for morning market (<1 whole share after hours)"))
            continue
        req = LimitOrderRequest(symbol=r.Symbol, qty=qty, limit_price=price,
                                time_in_force=TimeInForce.DAY, extended_hours=True,
                                side=OrderSide.SELL if r.Side == "SELL" else OrderSide.BUY,
                                client_order_id=_client_order_id(r.Symbol, r.Side, qty, price, date_str))
        try:
            o = client.submit_order(req)
            rest = round(plan_total - qty, 9)
            note = f" + {rest:g} fractional rest next morning" if rest > 0 else ""
            results.append((r.Symbol, r.Side, qty, str(o.id), f"submitted ({getattr(o.status, 'value', o.status)}){note}"))
            if record is not None:
                record({**entry, "order_qty": qty, "order_id": str(o.id)})
        except Exception as e:  # keep going; report every failure
            results.append((r.Symbol, r.Side, qty, None, f"FAILED: {e}"))
    return pd.DataFrame(results, columns=["Symbol", "Side", "Shares", "Order_ID", "Status"])


# ----------------------------------------------------------------------------- crash-safe order ids + broker reconciliation
def _client_order_id(symbol, side, qty, price, date_str, kind=""):
    """Deterministic client order id, stable across retries of the same plan.

    Format: live-[fill-]YYYYMMDD-SIDE-SYMBOL-qty-pricecents (<=48 chars, Alpaca-safe).
    `kind="fill"` marks morning fill-check completion orders. Because the id is
    deterministic, a crash between the broker submit and the local pending-file record
    cannot duplicate the order: the retry finds it on the broker by id.
    """
    sym = re.sub(r"[^A-Z0-9]", "", str(symbol).upper())
    try:
        cents = int(round(float(price) * 100))
    except (TypeError, ValueError, OverflowError):
        cents = 0
    # NOTE: qty is intentionally truncated to whole shares in the id (not rounded to
    # cents): morning completion quantities are 2-decimal, and a retry that recomputes
    # a slightly smaller remainder must still match the crashed attempt's order on the
    # broker for the no-duplicate recovery in _morning_completion_plan.
    tag = f"live-{kind + '-' if kind else ''}{date_str}-{side}-{sym}-{int(_safe_number(qty))}-{cents}"
    return tag[:48]


def _broker_orders_by_client_id(client):
    """{client_order_id: order} for the broker's recent orders (open + closed).

    Returns None when the broker cannot be read: the evening treats that as "nothing
    found" (the local pending file stays the primary record), the morning fill check
    aborts instead (it must never place an order it cannot de-duplicate).
    """
    try:
        from alpaca.trading.enums import QueryOrderStatus, SortDirection
        from alpaca.trading.requests import GetOrdersRequest
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500,
                               direction=SortDirection.DESCENDING)
        out = {}
        for o in client.get_orders(req):
            cid = getattr(o, "client_order_id", "") or ""
            if cid:
                out[cid] = o
        return out
    except Exception as e:
        print(f"WARNING: broker order reconciliation read failed ({e})")
        return None


def _evening_submitted_on_broker(client, date_str):
    """{(SYMBOL, SIDE)} already submitted to the broker this evening (any status).

    Matches our deterministic client_order_id prefix live-YYYYMMDD-SIDE-SYMBOL- (morning
    fill-check orders, live-fill-..., are excluded). A row found here was submitted before
    a crash, so its local record never got written; skipping it on retry is always safe
    because the morning fill check completes any unfilled remainder.
    """
    out = set()
    for cid in _broker_orders_by_client_id(client) or {}:
        parts = cid.split("-")
        if len(parts) < 6 or parts[0] != "live" or parts[1] == "fill":
            continue
        if parts[1] != date_str:
            continue  # another evening's orders
        out.add((parts[3], parts[2]))  # (SYMBOL, SIDE)
    return out


SELL_SETTLE_WAIT_SECS = 120  # bounded wait for evening SELL fills before sizing/submitting BUYs
SELL_SETTLE_POLL_SECS = 10


def _wait_for_terminal_all(client, order_ids, timeout_secs=SELL_SETTLE_WAIT_SECS,
                           poll_secs=SELL_SETTLE_POLL_SECS):
    """{order_id: (status, filled_qty)} after a bounded wait for terminal states.

    Best effort: ids still open when the timeout expires keep their last seen status.
    Used after the evening SELLs so the BUYs are sized on fresh buying power.
    """
    import time

    terminal = {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day"}
    state, remaining = {}, list(dict.fromkeys(order_ids))
    deadline = time.time() + timeout_secs
    while remaining:
        still = []
        for oid in remaining:
            try:
                o = client.get_order(oid)
                raw = o.status
                st = raw.value.lower() if hasattr(raw, "value") else str(raw).lower()
                try:
                    fq = _safe_number(o.filled_qty)
                except (TypeError, ValueError):
                    fq = 0.0
                state[oid] = (st, fq)
                if st not in terminal:
                    still.append(oid)
            except Exception:
                state[oid] = ("gone", 0.0)  # gone from the broker - treat as terminal
        remaining = still
        if remaining:
            if time.time() >= deadline:
                break
            time.sleep(poll_secs)
    return state


def _read_buying_power(client):
    """Fresh buying power from the broker, or None when unreadable.

    Buying power is what Alpaca will actually let the account spend. Margin is
    disabled on this account, so buying power equals cash in practice. The evening
    and morning BUY guards are capped at this number.
    """
    try:
        bp = float(client.get_account().buying_power)
    except Exception:
        return None
    return bp if math.isfinite(bp) and bp >= 0 else None


def _stage_orders_for_morning(orders_df, positions, record, reason):
    """Stage order rows for the morning fill-check instead of submitting them.

    Used when BUYs cannot be safely sized in the evening (SELLs did not settle in
    time, or buying power is unreadable): the full-size rows are recorded in the
    pending file with no broker order id, and the morning fill-check submits them
    as regular-hours market orders after the SELLs complete, with its own
    affordability check. This keeps a transient evening condition from permanently
    shrinking the BUY plan (Bug 1) or submitting oversized BUYs (Bug 2).
    Returns a results DataFrame with STAGED statuses.
    """
    cols = ["Symbol", "Side", "Shares", "Order_ID", "Status"]
    staged, skipped = [], []
    for r in orders_df[orders_df["Side"].isin(["SELL", "BUY"])].itertuples():
        # Morning = regular-hours market orders: 2-decimal shares, exact qty for full exits.
        qty, is_exit, qty_reason = _plan_qty(r.Symbol, r.Side, r.Shares, positions)
        if qty_reason:
            skipped.append((r.Symbol, r.Side, r.Shares, None, qty_reason))
            continue
        limit_price = _limit_price(r.Price)
        if not limit_price:
            skipped.append((r.Symbol, r.Side, r.Shares, None, "SKIPPED: no price"))
            continue
        entry = {"symbol": r.Symbol, "side": r.Side, "qty": qty,
                 "limit_price": limit_price, "order_id": None, "exit": is_exit}
        staged.append(entry)
        if record is not None:
            record(entry)  # incremental: a crash still leaves these staged
    return pd.DataFrame(
        [(s["symbol"], s["side"], s["qty"], None, reason) for s in staged] + skipped,
        columns=cols)


def submit_paper_extended_sequenced(orders, positions=None, record=None, client=None, order_date=None):
    """Evening extended-hours submission: SELLs first, then BUYs sized from the cash actually free.

    1. Submits the SELL rows as extended-hours limit orders (each recorded).
    2. Waits (bounded, SELL_SETTLE_WAIT_SECS) for the sells to fill - a sell's cash is only
       spendable once it fills.
    3. Reads BUYING POWER fresh from the broker and sizes the BUYs to it, keeping the 1% cushion
       (apply_buying_power_guard). Each BUY gets the part that fits (whole shares after hours);
       its pending entry keeps the FULL plan, so the 9 AM check buys the rest from the cash
       free then (e.g. once the sells filled). A BUY with no room at all is STAGED for 9 AM.
    4. Buying power unreadable: all BUYs are STAGED for 9 AM (never sent unsized).
    A crash between steps is safe: every submitted row carries a deterministic
    client_order_id and is recorded incrementally; a retry reconciles with the broker
    and the morning fill check completes any remainder.
    Returns the concatenated results DataFrame.
    """
    if client is None:
        client = paper_trading_client()
    sells = orders[orders["Side"] == "SELL"]
    buys = orders[orders["Side"] == "BUY"]
    sell_results = submit_paper_extended(sells, positions=positions, record=record,
                                         client=client, order_date=order_date)
    sell_ids = [oid for oid in sell_results["Order_ID"] if oid]
    if sell_ids:
        _wait_for_terminal_all(client, sell_ids)
    fresh_bp = _read_buying_power(client)
    if fresh_bp is None:
        buy_results = _stage_orders_for_morning(
            buys, positions, record, "STAGED for morning (buying power unreadable - the 9 AM check sizes it from real cash)")
        return pd.concat([sell_results, buy_results], ignore_index=True)
    fit = apply_buying_power_guard(buys, fresh_bp, fractional=True)
    fit["Plan_Shares"] = buys["Shares"]
    buy_results = submit_paper_extended(fit[fit["Side"] == "BUY"], positions=positions, record=record,
                                        client=client, order_date=order_date)
    later = buys.loc[fit.index[fit["Side"] != "BUY"]]
    if len(later):
        buy_results = pd.concat([buy_results, _stage_orders_for_morning(
            later, positions, record, "STAGED for morning (no free cash yet - the 9 AM check buys what fits)")],
            ignore_index=True)
    return pd.concat([sell_results, buy_results], ignore_index=True)


def _morning_completion_plan(broker_by_cid, sym, side, qty, price, date_str):
    """(client_order_id, qty_to_order, prior_order): crash-safe morning completion plan.

    A crash between the morning submit and the local record leaves the completion order
    on the broker. If a prior attempt's order already covers the quantity, returns
    (None, 0, prior_order) so the caller marks it completed instead of duplicating.
    Otherwise the already-filled part is subtracted and a fresh deterministic id
    (suffixed -rN when needed) is returned for the rest.
    """
    base = _client_order_id(sym, side, qty, price, date_str, kind="fill")
    cid, n, covered = base, 2, 0.0
    while cid in broker_by_cid:
        prior = broker_by_cid[cid]
        pf = _safe_number(prior.filled_qty)
        raw = getattr(prior, "status", "")
        st = (raw.value if hasattr(raw, "value") else str(raw)).lower()
        if st not in TERMINAL_STATUSES or pf >= qty - covered:
            return None, 0, prior  # still working, or already covers the rest: never duplicate
        covered += pf
        cid, n = (base + f"-r{n}")[:48], n + 1
    # The caller's qty is already 2-decimal (or the exact held qty of a full exit).
    return cid, (qty if covered == 0 else _floor2(qty - covered)), None


TERMINAL_STATUSES = {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day"}


def _symbol_state(client, sym):
    """(shares held, [open orders]) for one symbol, read fresh from the broker right before
    a morning order; (None, None) when either read fails (the caller fails closed)."""
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        held = sum(_safe_number(p.qty) for p in client.get_all_positions()
                   if str(p.symbol).upper() == sym)
        open_orders = []
        for o in client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[sym])):
            raw = getattr(o, "status", "")
            st = (raw.value if hasattr(raw, "value") else str(raw)).lower()
            if str(getattr(o, "symbol", "")).upper() == sym and st not in TERMINAL_STATUSES:
                open_orders.append(o)
    except Exception:
        return None, None
    return held, open_orders


def write_pending_orders(pending_orders, meta, pending_path=PENDING_ORDERS_JSON):
    """Record successfully submitted evening orders for the morning fill check (atomic write).

    If a previous pending file still holds unprocessed orders, it is backed up to
    <pending_path>.bak_<timestamp> (never silently overwritten) and a warning is printed.
    Returns the backup path, or None when there was nothing to back up.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now_ct = datetime.now(ZoneInfo("America/Chicago"))
    backup = None
    if os.path.exists(pending_path):
        try:
            existing = json.load(open(pending_path)).get("orders", [])
        except Exception:
            existing = []
        if existing:
            backup = f"{pending_path}.bak_{now_ct.strftime('%Y%m%d_%H%M%S')}"
            os.replace(pending_path, backup)
            print(f"WARNING: {pending_path} still held {len(existing)} unprocessed order(s) - "
                  f"backed up to {backup}; review it manually, it was NOT merged.")
    payload = {"evening_date": now_ct.date().isoformat(),
               "submitted_at_ct": now_ct.strftime("%Y-%m-%d %H:%M:%S"),
               "target_source": meta.get("source"), "as_of": meta.get("as_of"),
               "orders": pending_orders}
    os.makedirs(os.path.dirname(pending_path), exist_ok=True)
    tmp = pending_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, pending_path)
    return backup


def _today_ct():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/Chicago"))


def _todays_recorded_orders(pending_path=PENDING_ORDERS_JSON):
    """{(symbol, side)} already submitted or staged by an earlier --trade run this evening.

    Makes an evening retry idempotent: rows recorded in today's pending file are skipped
    instead of being sent to the broker a second time. FAILED rows are never recorded,
    so they are still retried.
    """
    try:
        with open(pending_path) as f:
            pend = json.load(f)
    except (OSError, ValueError):
        return set()
    today = _today_ct().date().isoformat()
    if pend.get("evening_date") != today:
        return set()
    # Leftovers kept from an earlier evening carry their own evening_date - not "this evening".
    return {(str(o.get("symbol", "")).upper(), str(o.get("side", "")).upper())
            for o in pend.get("orders", []) if (o.get("evening_date") or today) == today}


def record_pending_order(entry, meta, pending_path=PENDING_ORDERS_JSON):
    """Append one submitted/staged evening order to the pending file (atomic, de-duplicated).

    Called after each successful evening submission so a crash mid-batch still leaves the
    submitted orders recorded - a retry then skips them instead of duplicating them.
    Entries are keyed by (symbol, side): re-recording replaces the old entry. Unfinished
    leftovers from an earlier evening (the Mac slept through the 9 AM check) are KEPT for the
    next 9 AM retry, each stamped with its own evening_date (its completion order ids use it);
    a leftover for a symbol this evening trades is dropped - the new plan was sized from the
    current holdings and supersedes it.
    """
    now_ct = _today_ct()
    today = now_ct.date().isoformat()
    entry = {**entry, "recorded_at": now_ct.isoformat(timespec="seconds"), "decision": meta.get("decision")}
    try:
        with open(pending_path) as f:
            pend = json.load(f)
    except (OSError, ValueError):
        pend = {}
    existing = pend.get("orders", []) if isinstance(pend, dict) else []
    if existing and pend.get("evening_date") != today:
        old_date = pend.get("evening_date")
        existing = [{**o, "evening_date": o.get("evening_date") or old_date} for o in existing if isinstance(o, dict)]
        print(f"NOTE: keeping {len(existing)} unfinished order(s) from {old_date} for the next 9 AM check.")
        pend = {}
    key = (str(entry["symbol"]).upper(), str(entry["side"]).upper())
    orders_list = [o for o in existing
                   if (str(o.get("symbol", "")).upper(), str(o.get("side", "")).upper()) != key
                   and not ((o.get("evening_date") or today) != today
                            and str(o.get("symbol", "")).upper() == key[0])]
    orders_list.append(entry)
    payload = {"evening_date": today,
               "submitted_at_ct": now_ct.strftime("%Y-%m-%d %H:%M:%S"),
               "target_source": (pend.get("target_source") if isinstance(pend, dict) else None) or meta.get("source"),
               "as_of": (pend.get("as_of") if isinstance(pend, dict) else None) or meta.get("as_of"),
               "orders": orders_list}
    if meta.get("session") or (isinstance(pend, dict) and pend.get("send_now")):
        payload["send_now"] = True        # a daytime catch-up: the fill check sends these at once (not next morning)
    os.makedirs(os.path.dirname(pending_path), exist_ok=True)
    tmp = pending_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, pending_path)


def _order_expiry(row, pend):
    """When a pending row is superseded: the first decision slot (3:15 PM CT) after it was recorded (rows without
    recorded_at: after 3:15 PM CT on their evening date). None when it cannot be told."""
    from datetime import datetime

    import backtest_engine as be
    try:
        t = datetime.fromisoformat(str(row["recorded_at"]))
    except (KeyError, TypeError, ValueError):
        try:
            t = be.decision_slot(str(row.get("evening_date") or pend.get("evening_date")))
        except (TypeError, ValueError):
            return None
    return be.next_decision_slot(t if t.tzinfo else t.replace(tzinfo=be.CENTRAL))


def superseded_orders(now, pending_path=None):
    """Pending rows whose decision was replaced by a newer one (the next decision slot has passed). [] when none."""
    pending_path = pending_path or PENDING_ORDERS_JSON
    try:
        with open(pending_path) as f:
            pend = json.load(f)
    except (OSError, ValueError):
        return []                      # missing, or corrupt (the fill check reports that)
    rows = pend.get("orders") if isinstance(pend, dict) else None
    out = []
    for o in rows or []:
        exp = _order_expiry(o, pend) if isinstance(o, dict) else None
        if exp is not None and exp <= now:
            out.append(o)
    return out


def drop_superseded_orders(now=None, pending_path=None):
    """Remove the superseded rows from the pending file and alert: they are never sent - the newer decision re-plans
    the account. Unfinished orders of a decision can therefore only be completed until the next decision slot.
    Returns the dropped rows."""
    now, pending_path = now or _today_ct(), pending_path or PENDING_ORDERS_JSON
    old = superseded_orders(now, pending_path)
    if not old:
        return []
    with open(pending_path) as f:
        pend = json.load(f)
    pend["orders"] = [o for o in pend["orders"] if o not in old]
    if pend["orders"]:
        tmp = pending_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(pend, f, indent=1)
        os.replace(tmp, pending_path)
    else:
        os.remove(pending_path)
    _notify("Old orders dropped",
            f"{len(old)} unsent order(s) ({_list_syms([(o.get('symbol'), o.get('qty')) for o in old])}) belonged to an "
            f"earlier decision; a newer decision replaced it, so they were not sent. No money moved.",
            details="; ".join(f"{o.get('side')} {o.get('symbol')} {o.get('qty')} recorded {o.get('recorded_at') or o.get('evening_date')}"
                              for o in old))
    return old


def _wait_terminal(client, order_id, tries=10, pause=0.5):
    """True once the broker reports the order in a terminal state (best effort, ~5 s max).

    Used after canceling a stale evening order: confirms the cancel took effect before
    the morning fill check submits the remainder, so a fill in between cannot double-fill.
    """
    import time

    terminal = {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day"}
    for _ in range(tries):
        try:
            cur = client.get_order(order_id)
        except Exception:
            return False  # unreadable: cannot confirm the cancel - fail closed (retry later)
        raw = cur.status
        st = raw.value.lower() if hasattr(raw, "value") else str(raw).lower()
        if st in terminal:
            return True
        time.sleep(pause)
    return False


def complete_unfilled_orders(pending_path=PENDING_ORDERS_JSON, log_csv=ORDER_LOG_CSV, dry_run=False):
    """Morning fill check for the previous evening's extended-hours orders (LIVE).

    Reads Reports/live_pending_orders.json (written by auto_trade), checks each order's fill
    status on Alpaca, and for anything not fully filled submits a regular-hours DAY market order for the rest in
    fractional shares (2 decimals, rounded down; the exact held quantity for a full exit),
    including the fractional part the whole-share evening order left out. A still-open
    evening order is canceled first and the cancel must be confirmed; the final fill count
    is re-read after the cancel. Right before each order the holding and open orders for
    the symbol are re-read; an open order for the symbol means nothing is sent (kept for
    retry). Nothing is sent while the market is closed. A BUY rest worth under $1 is not
    ordered (Alpaca minimum). Completion marks are saved after every order.
    Partial fills are handled: only the unfilled remainder is ordered.
    Cash: SELLs go first; each BUY is sized to the cash free right then (limit x 1.01 per share).
    If only part fits, that part is bought; if under $1 fits, it is logged NO FILL (not enough
    cash) and dropped - the next Friday rebalance re-plans it. Rows left over from an earlier
    evening are completed too, until the next decision slot (then drop_superseded_orders removes them).
    Orders staged by a past-7PM evening run (no broker order id) were never submitted, so the
    full quantity is sent as a regular-hours market order.
    A rejected evening order is never auto-retried - it is logged as REJECTED for review
    and dropped from the pending list.

    Fail closed: when the live positions cannot be read, the run aborts with RuntimeError
    (the pending file is kept for a retry) instead of completing SELL remainders unclamped.
    A dry run never aborts on this (nothing is submitted).

    Crash-safe morning orders: each completion order carries a deterministic client_order_id
    (live-fill-YYYYMMDD-SIDE-SYMBOL-qty-...). A previous attempt that crashed between its
    submit and its local record is found on the broker by id and marked ALREADY COMPLETED
    instead of duplicated.

    The pending file is removed once every order reached a terminal state without a FAILED
    completion; if any completion failed, only the failed rows are kept (with per-row
    completion marks) so a retry never re-orders what already completed.
    A dry run changes nothing: it neither submits, nor logs, nor removes the pending file.
    Appends the morning orders to Reports/live_orders_log.csv (same columns as the evening log).
    Returns a DataFrame [Symbol, Side, Shares, Order_ID, Status].
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    cols = ["Symbol", "Side", "Shares", "Order_ID", "Status"]

    def _remainder(qty, filled, have, is_exit):
        """Unfilled remainder: 2 decimals rounded DOWN, clamped to `have` (shares held,
        SELLs only). A full exit sells the exact held quantity, fraction included."""
        if is_exit and have is not None:
            rem = min(qty - filled, have)
            return have if have - rem < 1e-6 else max(rem, 0.0)
        rem = _floor2(qty - filled)
        return min(rem, _floor2(have)) if have is not None else rem

    def _save_pending():
        """Persist per-row completion marks right away, so a crash mid-run never re-orders a row."""
        try:
            tmp = pending_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(pend, f, indent=1)
            os.replace(tmp, pending_path)
        except Exception as e:
            print(f"WARNING: could not save completion marks ({e}) - broker order ids still prevent duplicates")

    _print_header("MORNING FILL CHECK - Alpaca LIVE (REAL MONEY)")
    if not os.path.exists(pending_path):
        print("  No pending extended-hours orders - nothing to do.")
        return pd.DataFrame(columns=cols)
    with open(pending_path) as f:
        pend = json.load(f)
    evening_orders = pend.get("orders", [])
    if not evening_orders:
        os.remove(pending_path)
        return pd.DataFrame(columns=cols)
    # SELLs complete before BUYs (explicit ordering, not relying on file order):
    # SELL proceeds must be in settled cash before any BUY's affordability is judged.
    # Python's sort is stable, so same-side rows keep their file order.
    evening_orders = sorted(evening_orders,
                            key=lambda o: 0 if isinstance(o, dict) and
                            str(o.get("side")).upper() == "SELL" else 1)
    client = paper_trading_client()
    if not dry_run:
        # Market orders only go out while the regular session is open: a market order sent
        # after the close (e.g. the Mac woke late) would queue for a later open at an
        # unknown price. Nothing is sent and the pending file is kept.
        try:
            market_open = bool(client.get_clock().is_open)
        except Exception:
            market_open = False
        if not market_open:
            msg = "market closed (or clock unreadable) - nothing sent; pending orders kept"
            waiting = [(o.get("symbol", "?"), o.get("qty")) for o in evening_orders if isinstance(o, dict)]
            _notify("9 AM check waiting: market closed",
                    f"Market closed, so nothing was sent and no money moved. Still to finish: "
                    f"{_list_syms(waiting)}. Nothing to do: it retries at the next 9 AM check. "
                    f"List: Reports/live_pending_orders.json", details=msg)
            return pd.DataFrame([(o.get("symbol", "?"), o.get("side", "?"), o.get("qty"), o.get("order_id"),
                                  "WAITING: " + msg) for o in evening_orders if isinstance(o, dict)], columns=cols)
    try:  # live holdings, so a SELL remainder is clamped to the shares actually held
        live_positions = {str(p.symbol).upper(): float(p.qty) for p in client.get_all_positions()}
    except Exception as e:
        if dry_run:
            live_positions = None  # preview only: nothing is submitted, no clamping needed
        else:
            # Fail closed: without live holdings a SELL remainder cannot be clamped to the
            # shares actually held. Abort (the pending file is kept for a retry) instead
            # of completing unclamped.
            raise RuntimeError(
                f"ABORTED: cannot read live positions for the fill check ({e}) - kept for retry")
    # Crash-safety for the morning orders themselves (a submit followed by a crash before
    # the local record): match prior attempts on the broker by deterministic client id.
    broker_by_cid = {} if dry_run else _broker_orders_by_client_id(client)
    if broker_by_cid is None:
        raise RuntimeError("ABORTED: cannot read broker orders to rule out duplicates - kept for retry")
    # Completion ids use the row's EVENING date, so a retry on any later day still finds them.
    fill_date = str(pend.get("evening_date") or _today_ct().date().isoformat()).replace("-", "")
    results = []
    to_retry = []  # rows whose completion FAILED - only these are kept for a retry
    morning_sell_ids = []  # market SELL completions submitted by this run (waited on before BUYs)
    sells_settled = False  # ... have been waited on before the first BUY affordability check
    reserved_buy_spend = 0.0  # estimated cost of BUY completions already submitted this run
    bp_base = None  # buying power read ONCE after this run's SELLs settled (Alpaca already
    #                 lowers its figure for submitted buys, so re-reading AND subtracting
    #                 reserved_buy_spend would count every earlier BUY twice)
    for o in evening_orders:
        # Defensive parse: a malformed row can never be completed. Drop it loudly
        # (FAILED, not kept for retry - a retry could never parse it either) instead
        # of crashing the whole fill check. An unknown side is dropped too: without
        # this it would fall through to the BUY branch below.
        if not isinstance(o, dict):
            results.append(("?", "?", "?", None,
                            "FAILED: malformed pending row (not an object) - dropped, needs review"))
            continue
        qty = _safe_number(o.get("qty"), default=None)  # planned qty (2 decimals, exact for exits)
        order_qty = _safe_number(o.get("order_qty"), default=qty)  # whole shares sent after hours
        is_exit = bool(o.get("exit"))
        sym = o.get("symbol")
        sym = str(sym).upper().strip() if isinstance(sym, str) and sym.strip() else ""
        side = o.get("side")
        side = str(side).upper().strip() if isinstance(side, str) and side.strip() else ""
        oid = o.get("order_id")
        if not sym or side not in ("BUY", "SELL") or qty is None or qty < 0:
            results.append((sym or "?", side or "?", o.get("qty"), oid,
                            "FAILED: malformed pending row (dropped, needs review)"))
            continue
        if o.get("completed_order_id"):
            # A previous check already completed this row - never order it twice.
            results.append((sym, side, qty, o["completed_order_id"],
                            f"ALREADY COMPLETED (morning order {o['completed_order_id']})"))
            continue
        if not oid:
            # Staged by a past-7PM evening run (never submitted): the full quantity is unfilled,
            # so it goes straight to a regular-hours market order below.
            status, filled = "staged", 0.0
        else:
            try:
                alp = client.get_order(oid)
            except Exception as e:
                results.append((sym, side, qty, oid, f"FAILED: cannot read order: {e}"))
                to_retry.append(o)
                continue
            raw = alp.status
            status = raw.value.lower() if hasattr(raw, "value") else str(raw).lower()
            try:
                filled = _safe_number(alp.filled_qty)
            except (TypeError, ValueError):
                filled = 0.0
            if status == "filled":  # the whole-share evening order filled completely
                filled = max(filled, order_qty or 0.0)
            if filled >= qty - 1e-9:
                results.append((sym, side, qty, oid, f"FILLED ({filled:g}/{qty})"))
                continue
            if status == "rejected":
                # A rejected evening order needs a human look (e.g. a buying-power or
                # corporate-action reject) - it is never auto-retried as a market order.
                results.append((sym, side, qty, oid, f"REJECTED - not retried, needs review (filled {filled:g}/{qty})"))
                continue
        # Anything else not fully filled is completed with a regular-hours market order
        # for the unfilled remainder: still-open orders, and terminal-but-unfilled ones
        # (expired / done_for_day / canceled - the normal overnight state of a DAY
        # extended-hours order that did not fill).
        # Market orders allow fractional shares: round the remainder DOWN to 2 decimals
        # (never up - we never order more than the unfilled remainder). This also covers
        # the fractional rest of an evening whole-share order (e.g. 3.58 planned, 3 filled).
        remaining = _remainder(qty, filled, None, False)
        if remaining <= 0 and not is_exit:
            results.append((sym, side, qty, oid, f"NO FILL (remainder <=0, filled {filled:g}/{qty})"))
            continue
        if side == "SELL" and live_positions is not None:
            # Never sell what the portfolio does not hold: clamp the remainder to the
            # shares actually held (the evening order may have filled partially, or the
            # position may have changed overnight). 2-decimal precision for market orders.
            have = _safe_number(live_positions.get(str(sym).upper(), 0.0))
            if have <= 0:
                results.append((sym, side, qty, oid, "SKIPPED: not in portfolio (0 shares held)"))
                continue
            remaining = _remainder(qty, filled, have, is_exit)
            if remaining <= 0:
                results.append((sym, side, qty, oid,
                                f"NO FILL (remainder <=0 held, filled {filled:g}/{qty})"))
                continue
        if side == "BUY":
            # Never assume the morning market BUY will fit in buying power.
            # Estimate cost at the evening limit price + 2% buffer (market orders can
            # fill above the estimate); SKIP the completion if buying power cannot cover it.
            # This is re-checked with fresh buying power right before submit (below).
            pass  # buying-power check happens after the cancel/re-read, with fresh numbers
        if dry_run:
            results.append((sym, side, remaining, oid, f"WOULD COMPLETE ({status}, filled {filled:g}/{qty})"))
            continue
        try:
            if oid:
                try:  # the evening order should already be expired; cancel just in case it is still open
                    client.cancel_order(oid)
                except Exception:
                    pass
                if not _wait_terminal(client, oid):
                    # Fail closed: the evening order may still fill - retry later instead
                    # of risking a duplicate market order now.
                    results.append((sym, side, remaining, oid,
                                    "FAILED: evening order still open after cancel - kept for retry"))
                    to_retry.append(o)
                    continue
                try:  # re-read: it may have filled while the cancel was processed
                    chk = client.get_order(oid)
                    filled = _safe_number(chk.filled_qty)
                    raw = chk.status
                    status = raw.value.lower() if hasattr(raw, "value") else str(raw).lower()
                except Exception as e:
                    # Fail closed: without the final fill count the remainder is unknown.
                    results.append((sym, side, remaining, oid, f"FAILED: cannot re-read order after cancel ({e}) - kept for retry"))
                    to_retry.append(o)
                    continue
                if status == "filled":
                    filled = max(filled, order_qty or 0.0)
                remaining = _remainder(qty, filled, None, False) if not is_exit else qty - filled
                if remaining <= 0:
                    results.append((sym, side, qty, oid,
                                    f"NO FILL (filled while canceling, {filled:g}/{qty})"))
                    continue
            partial = ""
            # Crash-safety: a previous attempt may have submitted the morning order already
            # (crash before recording it). Match by deterministic client order id instead
            # of submitting a duplicate.
            cid, to_order, prior = _morning_completion_plan(
                broker_by_cid, sym, side, remaining, o.get("limit_price") or 0,
                str(o.get("evening_date") or "").replace("-", "") or fill_date)
            if prior is not None:
                o["completed_order_id"] = str(prior.id)  # retry-safety: this row is done
                _save_pending()
                results.append((sym, side, remaining, str(prior.id),
                                f"ALREADY COMPLETED (morning order {prior.id} covers the remainder)"))
                continue
            # Fresh broker check right before sending: current holding + any open order
            # for this symbol (fail closed - never risk a second working order).
            held_now, open_now = _symbol_state(client, sym)
            if held_now is None:
                results.append((sym, side, to_order, oid,
                                "FAILED: cannot re-check positions/open orders - kept for retry"))
                to_retry.append(o)
                continue
            if open_now:
                ids = ", ".join(str(getattr(x, "id", "?")) for x in open_now)
                results.append((sym, side, to_order, oid,
                                f"SKIP (open {sym} order already on Alpaca: {ids}) - kept for retry"))
                _notify(f"9 AM check: {sym} order already open",
                        f"An open {sym} order is already on Alpaca, so nothing new was sent for {sym}. "
                        f"Let it fill, or cancel it in Alpaca. Log: Reports/logs",
                        details=f"open order id(s): {ids}")
                to_retry.append(o)
                continue
            if side == "SELL":
                if held_now <= 0:
                    results.append((sym, side, to_order, oid, "SKIPPED: not in portfolio (0 shares held)"))
                    continue
                to_order = _remainder(to_order, 0.0, held_now, is_exit)
            if to_order <= 0:
                results.append((sym, side, remaining, oid,
                                "NO FILL (remainder <=0 after prior attempt)"))
                continue
            if side == "BUY":
                # Buying-power check for morning BUY completions. SELL completions were
                # processed first (sorted above): wait for this run's SELLs to settle,
                # then re-read fresh buying power and verify it covers the estimated market
                # cost. The estimate uses the evening limit price + 1% cushion; a BUY that
                # does not fully fit is cut to the part that fits (the rest is not bought).
                # Fail closed: an unreadable buying-power figure keeps the row for retry;
                # a missing or invalid price can never be verified, so that row is dropped
                # loudly instead of being submitted blind or retried forever.
                if not sells_settled:
                    if morning_sell_ids:
                        _wait_for_terminal_all(client, morning_sell_ids,
                                               timeout_secs=60, poll_secs=5)
                    sells_settled = True
                if bp_base is None:
                    try:
                        bp_base = _read_buying_power(client)
                    except Exception:
                        bp_base = None
                bp_now = bp_base
                est_price = _safe_number(o.get("limit_price"))
                if est_price > 0 and to_order * est_price < 1.0:
                    # Alpaca's fractional minimum is $1: a smaller BUY rest would be rejected
                    # every morning, so it is dropped (not retried).
                    results.append((sym, side, to_order, oid,
                                    "NO FILL (rest worth under $1 - below Alpaca's minimum, not ordered)"))
                    continue
                if not (est_price > 0):
                    msg = (f"FAILED: no usable price for {sym} BUY {to_order:g} shares - "
                           f"affordability cannot be verified (dropped, needs review)")
                    results.append((sym, side, to_order, oid, msg))
                    _notify(f"9 AM check: {sym} buy not sent",
                            f"{sym} buy of {_fmt_shares(to_order)} shares had no saved price, so it was NOT "
                            f"sent (no money moved). Buy it by hand in Alpaca if you still want it. "
                            f"Log: Reports/logs", details=msg)
                    continue
                if bp_now is None or not math.isfinite(bp_now) or bp_now < 0:
                    msg = (f"SKIP (morning buying power unreadable) - {sym} BUY {to_order:g} shares "
                           f"not completed")
                    results.append((sym, side, to_order, oid, msg))
                    _notify(f"9 AM check: {sym} buy on hold",
                            f"Couldn't read your buying power, so the {sym} buy of {_fmt_shares(to_order)} "
                            f"shares was NOT sent (no money moved). It retries at the next 9 AM check. Log: Reports/logs",
                            details=msg)
                    to_retry.append(o)
                    continue
                unit = est_price * (1 + CASH_CUSHION)  # 1% cushion for market slippage/gaps
                est_cost = to_order * unit
                spendable = bp_now - reserved_buy_spend  # less BUYs already submitted this run
                if est_cost > spendable:
                    # Buy the part that fits (2 decimals, rounded down); the rest is not bought.
                    fit = _floor2(max(spendable, 0.0) / unit)
                    if fit * est_price < 1.0:
                        msg = (f"NO FILL (not enough cash: need ~${est_cost:,.2f}, free ${spendable:,.2f} "
                               f"after ${reserved_buy_spend:,.2f} reserved) - {sym} BUY {to_order:g} shares "
                               f"not bought, not retried")
                        results.append((sym, side, to_order, oid, msg))
                        _notify(f"9 AM check: not enough cash for {sym}",
                                f"{sym} buy of {_fmt_shares(to_order)} shares needs ~${est_cost:,.0f}; only "
                                f"${max(spendable, 0):,.0f} is free. NOT sent, no money moved. Buy less by hand "
                                f"in Alpaca if you want it.", details=msg)
                        continue
                    partial = f"; only {fit:g} of {to_order:g} fit in free cash"
                    to_order, est_cost = fit, fit * unit
            req = MarketOrderRequest(symbol=sym, qty=to_order, time_in_force=TimeInForce.DAY,
                                     side=OrderSide.SELL if side == "SELL" else OrderSide.BUY,
                                     client_order_id=cid)
            m = client.submit_order(req)
            o["completed_order_id"] = str(m.id)  # retry-safety: this row is done
            _save_pending()
            if side == "SELL":
                morning_sell_ids.append(str(m.id))  # waited on before the first BUY
            else:
                reserved_buy_spend += est_cost  # later BUYs are judged on what is left
            results.append((sym, side, to_order, str(m.id),
                            f"COMPLETED via market ({status}, filled {filled:g}/{qty}){partial}"))
        except Exception as e:
            results.append((sym, side, remaining, oid, f"FAILED: {e}"))
            to_retry.append(o)
    results = pd.DataFrame(results, columns=cols)
    if not dry_run and log_csv and not results.empty:
        log = results.drop(columns=["Order_ID"], errors="ignore").copy()
        log["Submitted_At_CT"] = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")
        log["Target_Source"] = f"fill-check ({pend.get('target_source')})"
        log["As_Of"] = pend.get("as_of")
        log["Equity"] = None
        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        log.to_csv(log_csv, mode="a", header=not os.path.exists(log_csv), index=False)
    if not dry_run:
        if to_retry:
            # Keep only the failed rows (atomic rewrite) so a retry is exact-once.
            pend["orders"] = to_retry
            tmp = pending_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(pend, f, indent=1)
            os.replace(tmp, pending_path)
        elif not results.empty:
            os.remove(pending_path)

    # --- readable summary, failure alerts, position reconciliation ---
    _print_section("Fill-check results")
    if not results.empty:
        print(results.drop(columns=["Order_ID"], errors="ignore").to_string(index=False))
    n_failed = int(results["Status"].str.startswith("FAILED").sum())
    n_completed = int(results["Status"].str.contains("COMPLETED|FILLED", na=False).sum())
    print(f"\n  Completed/Filled: {n_completed}  |  Failed: {n_failed}  |  Kept for retry: {len(to_retry)}")
    if n_failed and not dry_run:
        failed = results[results["Status"].str.startswith("FAILED")]
        _notify(f"9 AM check: {n_failed} order(s) failed",
                f"Not completed: {_list_syms(zip(failed['Symbol'], failed['Shares']))}. The missing shares were "
                f"not traded. Check Alpaca and place any you still want by hand. "
                f"Log: Reports/live_orders_log.csv",
                details="; ".join(f"{r.Symbol}: {r.Status}" for r in failed.itertuples()))
    if not dry_run and not results.empty:
        def _pairs(mask):
            return [(f"{r.Side.lower()} {r.Symbol}", r.Shares) for r in results[mask].itertuples()]
        done = _pairs(results["Status"].str.contains("COMPLETED|FILLED", na=False))
        if done:
            left = ("Nothing to do." if not (n_failed or to_retry)
                    else f"{max(len(to_retry), n_failed)} not done - see the other alert.")
            _notify("9 AM check: orders filled",
                    f"Filled (money moved): {_list_syms(done)}. {left} "
                    f"Log: Reports/live_orders_log.csv",
                    details="; ".join(f"{r.Side} {r.Symbol} {r.Shares}: {r.Status}"
                                      for r in results.itertuples()))

    if not dry_run and not to_retry:
        # All completions done (or nothing needed doing): verify the portfolio
        # actually matches the strategy targets.
        _print_section("Position reconciliation (live vs strategy targets)")
        # Only the symbols this batch traded are checked: other holdings (e.g. stocks
        # bought by hand, or picks the strategy holds without topping up) are not
        # expected to sit at target weight and must not raise a false alarm.
        traded = {str(o.get("symbol", "")).upper() for o in evening_orders if isinstance(o, dict)}
        report, ok = reconcile_positions(pend.get("target_source") or "auto", symbols=traded)
        if not report.empty:
            print(report.to_string(index=False))
        if ok is None:
            _notify("Couldn't double-check holdings",
                    "The 9 AM check finished, but targets or Alpaca couldn't be read to compare holdings. "
                    "No orders were sent by this step. Glance at Alpaca once. Log: Reports/logs")
            print("\n  !! Could not compare holdings with targets (data unavailable).")
        elif not ok:
            drift = report[report["Status"] != "OK"]
            first = drift.iloc[0]
            more = f" (+{len(drift) - 1} more)" if len(drift) > 1 else ""
            _notify(f"{first['Symbol']} off target after trades",
                    f"After the trades, {first['Symbol']} is {first['Actual_Weight_%']:.1f}% of your account "
                    f"vs {first['Target_Weight_%']:.1f}% planned{more}. An order may not have filled. "
                    f"Check it in Alpaca. Log: Reports/logs",
                    details="; ".join(f"{r['Symbol']}: {r['Actual_Weight_%']:.1f}% held vs "
                                      f"{r['Target_Weight_%']:.1f}% planned ({r['Status']})"
                                      for _, r in drift.iterrows()))
            print("\n  !! A traded symbol is off target - notification sent, check Alpaca.")
        else:
            print("\n  Traded symbols match strategy targets (within 1 percentage point).")
    _print_header("MORNING LIVE FILL CHECK COMPLETE")
    return results


def get_live_positions_and_equity():
    """(positions dict, equity float, cash float, buying power float) from the Alpaca LIVE account.

    Positions are {SYMBOL: shares} with fractional shares preserved here. Buying power is what
    apply_buying_power_guard() caps planned BUY spending at (margin is disabled on
    the account, so buying power equals cash in practice). Cash is returned for
    display only.
    """
    client = paper_trading_client()
    acct = client.get_account()
    positions = {}
    for p in client.get_all_positions():
        qty = float(p.qty)
        if qty != 0:
            positions[str(p.symbol).upper()] = qty
    return positions, float(acct.equity), float(acct.cash), float(acct.buying_power)


EVENING_SUBMIT_CUTOFF_CT = "19:00"  # extended hours end 7:00 PM CT - later runs stage orders for the morning
EARLY_CLOSE_SUBMIT_CUTOFF_CT = "16:00"  # early-close days (1 PM ET close): extended hours end 4:00 PM CT


def reconcile_positions(target_source="auto", tolerance_pct=1.0, symbols=None):
    """Compare live Alpaca positions against the strategy's target weights.

    Called after the morning fill check: catches silent drift between what the
    strategy wants and what the broker actually holds (e.g. a rejected order,
    a partial fill that never completed, or a manual trade in the account).

    symbols: optional set of symbols to check (the fill check passes the ones it just
    traded); None checks every target and every held symbol.

    Returns (report_df, ok): report_df has one row per symbol with target vs
    actual weight; ok is True when all are within tolerance_pct percentage points,
    False when any is not, and None when targets/positions could not be read (so
    "couldn't check" is never reported as drift). Never raises.
    """
    cols = ["Symbol", "Target_Weight_%", "Actual_Weight_%", "Diff_pp", "Shares_Held", "Status"]
    try:
        targets, meta = load_targets(target_source)
        positions, equity, cash, _ = get_live_positions_and_equity()
    except Exception:
        return pd.DataFrame(columns=cols), None
    if meta.get("source") == "hold" or targets.empty:
        # A hold day orders nothing: there are no targets to reconcile against.
        # (Without this, every held position would false-alarm as DRIFT against a
        # 0% target.)
        return pd.DataFrame(columns=cols), True
    if equity <= 0:
        return pd.DataFrame(columns=cols), None
    syms = sorted(set(targets["Symbol"].astype(str)) | {str(s) for s in positions})
    if symbols is not None:
        syms = sorted({str(s).upper() for s in symbols if str(s).strip()})
    px = latest_prices(syms)
    try:
        px.update(dict(zip(targets["Symbol"].astype(str), pd.to_numeric(targets["Price"], errors="coerce"))))
    except Exception:
        pass
    target_w = {str(s): float(w) for s, w in zip(targets["Symbol"].astype(str), targets["Weight"])}
    rows = []
    for sym in syms:
        tw = target_w.get(sym, 0.0) * 100
        shares = float(positions.get(sym, 0.0) or 0.0)
        price = px.get(sym) or 0.0
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = 0.0
        aw = (shares * price / equity * 100) if price > 0 else 0.0
        diff = aw - tw
        status = "OK" if abs(diff) <= tolerance_pct else ("MISSING" if tw > 0 and shares <= 0 else "DRIFT")
        rows.append({"Symbol": sym, "Target_Weight_%": round(tw, 2), "Actual_Weight_%": round(aw, 2),
                     "Diff_pp": round(diff, 2), "Shares_Held": shares, "Status": status})
    report = pd.DataFrame(rows, columns=cols)
    ok = bool((report["Status"] == "OK").all()) if not report.empty else True
    return report, ok


def _past_evening_cutoff(now=None):
    """True when it is at or past the extended-hours submission cutoff (America/Chicago)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import backtest_engine as be

    now = now or datetime.now(ZoneInfo("America/Chicago"))
    cutoff = EARLY_CLOSE_SUBMIT_CUTOFF_CT if be.is_early_close(now.date()) else EVENING_SUBMIT_CUTOFF_CT
    return now.strftime("%H:%M") >= cutoff


def auto_trade(target="auto", min_value=1.0, log_csv=ORDER_LOG_CSV, dry_run=False, extended=True, decision=None,
               session=False):
    """Full auto flow for the pipeline: pull LIVE positions + equity, plan, optionally submit.

    - Pulls current stock positions, equity, cash and buying power from the Alpaca LIVE account.
    - Plans orders with source=target (default "auto"). On the Friday rebalance every target pick
      ('add' or 'hold') is brought to Weight * equity (2-decimal shares, rounded down): underweight
      holdings are bought up, overweight ones trimmed, and a holding within 1 percentage point of
      its target is not traded. A pick the account does not own with earnings within 5 days is not
      bought (earnings rule). Non-target stocks are sold entirely (the exact shares held).
      Mon/Wed (auto -> 'midweek' or 'hold') trade only the strategy's swaps/exits.
    - Caps total BUY spending at Alpaca's BUYING POWER less a 1% cushion via
      apply_buying_power_guard(): buys are scaled down to the part that fits instead of being
      rejected by the broker. Margin is disabled on this account, so buying power equals cash.
    - SELL rows are verified against the live portfolio before anything is sent: a symbol not
      held is SKIPPED (never submitted), and the sell quantity is clamped to the shares
      actually held. After-hours limit orders go out in WHOLE shares; the fractional rest
      is completed by the next morning's regular-hours fill check (market order, 2 decimals).
    - If dry_run=True: returns (orders, meta, results_df) without submitting; results_df is empty.
    - If extended=True (default): submits via submit_paper_extended_sequenced() - the SELLs go
      first as DAY limit orders at the planned closing price with extended_hours=True; after a
      bounded wait for the sells to fill, buying power is re-read from the broker and each BUY is
      sent for the part that fits it now; the 9 AM check buys the rest from the cash free then.
      Each successfully submitted order is recorded incrementally in
      Reports/live_pending_orders.json for the morning fill check (complete_unfilled_orders).
      Every order carries a deterministic client_order_id, so a crash between the broker submit
      and the local record cannot duplicate it on retry (the retry reconciles with the broker).
    - If extended=False: submits via submit_paper() - regular-hours DAY market orders.
    - If it is at or past 7:00 PM CT (extended hours are over), nothing is submitted regardless
      of `extended`: the planned orders are staged in Reports/live_pending_orders.json with no
      broker order id, and the morning fill check (complete_unfilled_orders) sends the full
      quantities as regular-hours market orders. Staged rows are reported as STAGED, never FAILED.
    - Idempotent retries: rows already submitted/staged earlier the same evening are recorded
      after each submit, so re-running --trade in the same window skips them instead of
      duplicating orders. Rows found on the broker from this evening but never recorded
      locally (a crash between submit and record) are likewise skipped - the morning fill
      check completes any unfilled remainder. FAILED rows are never recorded, so they are retried.
    - decision (a date): the decision being traded - run_all passes it, so a missed decision caught up later trades
      that decision's picks (its point-in-time ranks) and the freshness check accepts its files.
    - session=True (run_all: a missed decision caught up during regular hours): sized at current prices
      (current_prices), nothing is sent here - every order is STAGED with send_now, and run_all sends them at once
      as regular-hours market orders in 2-decimal shares (complete_unfilled_orders).
    - Pending rows of an earlier decision whose catch-up window closed are dropped first (drop_superseded_orders).
    - Appends a row per submitted order to Reports/live_orders_log.csv and returns
      (orders, meta, results_df).

    Raises SystemExit if keys are missing; ValueError if the
    signal CSVs are missing or empty (run main_signal_analysis.ipynb first).
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    _print_header("EVENING TRADE - Alpaca LIVE (REAL MONEY)")
    try:
        # Fail closed on stale signals: never trade on yesterday's data after a
        # partial pipeline failure.
        as_of_date = check_signal_freshness(decision=decision)
        print(f"  Signals fresh: as of {as_of_date}" + (f" (decision {pd.Timestamp(decision):%Y-%m-%d})" if decision is not None else " (today, CT)"))
    except ValueError as e:
        _notify("Trade skipped: signals not fresh",
                "Today's signal files weren't ready, so no orders were sent and no money moved. "
                "Nothing to do if the next run works. Log: Reports/logs", details=str(e))
        raise

    try:
        positions, equity, cash, buying_power = get_live_positions_and_equity()
    except Exception as e:
        _notify("Trade skipped: can't reach Alpaca",
                "Couldn't read your Alpaca account, so no orders were sent and no money moved. "
                "Check your internet or Alpaca's status page. Log: Reports/logs", details=str(e))
        raise
    print(f"  Equity ${equity:,.2f}  |  Cash ${cash:,.2f}  |  Positions: {len(positions)} symbols")

    _print_section("Planning orders")
    live = {}
    if session:
        live = current_prices(set(positions) | set(pd.read_csv(PICKS_CSV)["Symbol"].astype(str)))
        print(f"  Catch-up in regular hours: sized at current prices ({len(live)} quotes)")
    orders, meta, targets = plan_orders(target, equity, positions, min_value=min_value, fractional=True,
                                        decision=decision, live_prices=live)
    meta.update(decision=pd.Timestamp(decision).date().isoformat() if decision is not None else None, session=session)
    n_buy = int((orders["Side"] == "BUY").sum())
    n_sell = int((orders["Side"] == "SELL").sum())
    print(f"  Plan: {n_buy} BUY, {n_sell} SELL  (source: {meta['source']}, as of {meta['as_of']})")
    # Cap BUY spending at buying power (margin is disabled on the account). SELLs go first, so
    # their proceeds count here; the sequenced submit / morning fill check re-cap BUYs at the
    # actual post-sell buying power.
    planned_sells = orders.loc[orders["Side"] == "SELL", "Est_Value"].sum()
    orders = apply_buying_power_guard(orders, buying_power + planned_sells, fractional=True)
    skipped_bp = orders[orders["Side"] == "SKIP (no buying power)"]
    if not skipped_bp.empty:
        print(f"  Buying-power guard: {len(skipped_bp)} BUY row(s) SKIPPED - insufficient buying power")
    if dry_run:
        return orders, meta, pd.DataFrame(columns=["Symbol", "Side", "Shares", "Order_ID", "Status"])
    cols = ["Symbol", "Side", "Shares", "Order_ID", "Status"]
    drop_superseded_orders(pending_path=PENDING_ORDERS_JSON)
    defer = session or _past_evening_cutoff()
    # One broker client for reconciliation + submission (not needed when staging past 7 PM).
    client, date_str, broker_submitted = None, None, set()
    if not defer:
        client = paper_trading_client()
        date_str = _today_ct().date().strftime("%Y%m%d")
        # Crash-safety: rows the broker already has from this evening were submitted before
        # a crash, so their local record never got written. Skipping them is always safe:
        # the morning fill check completes any unfilled remainder.
        broker_submitted = _evening_submitted_on_broker(client, date_str)
    # Idempotent retry: skip BUY/SELL rows already submitted/staged earlier this evening.
    already = _todays_recorded_orders(PENDING_ORDERS_JSON)
    dup_rows, keep_idx = [], []
    for r in orders.itertuples():
        key = (str(r.Symbol).upper(), r.Side)
        if r.Side in ("BUY", "SELL") and key in already:
            dup_rows.append((r.Symbol, r.Side, r.Shares, None,
                             "SKIPPED: already submitted/staged this evening"))
        elif r.Side in ("BUY", "SELL") and key in broker_submitted:
            dup_rows.append((r.Symbol, r.Side, r.Shares, None,
                             "SKIPPED: already on the broker this evening (recovered after a crash - "
                             "the morning fill check completes any remainder)"))
        else:
            keep_idx.append(r.Index)
    orders_to_send = orders.loc[keep_idx]
    if defer:
        # Extended hours are over: submit nothing now. Stage the planned orders so the
        # morning fill check sends them as regular-hours fractional market orders.
        results = pd.concat([_stage_orders_for_morning(
            orders_to_send, positions, lambda e: record_pending_order(e, meta, PENDING_ORDERS_JSON),
            "STAGED for regular-hours market orders now (catch-up)" if session
            else "STAGED for morning market (past 7 PM CT - not submitted)"),
            pd.DataFrame(dup_rows, columns=cols)], ignore_index=True)
    else:
        def _recorder(entry):
            record_pending_order(entry, meta, PENDING_ORDERS_JSON)  # incremental: a crash still leaves these recorded
        if extended:
            results = submit_paper_extended_sequenced(orders_to_send, positions=positions,
                                                      record=_recorder, client=client,
                                                      order_date=date_str)
        else:
            results = submit_paper(orders_to_send, positions=positions, client=client,
                                   order_date=date_str)
        if dup_rows:
            results = pd.concat([results, pd.DataFrame(dup_rows, columns=cols)], ignore_index=True)
    if log_csv and not results.empty:
        log = results.drop(columns=["Order_ID"], errors="ignore").copy()
        log["Submitted_At_CT"] = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")
        log["Target_Source"] = meta["source"]
        log["As_Of"] = meta["as_of"]
        log["Equity"] = round(equity, 2)
        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        log.to_csv(log_csv, mode="a", header=not os.path.exists(log_csv), index=False)

    # --- readable summary + failure alerts ---
    _print_section("Results")
    if not results.empty:
        print(results.drop(columns=["Order_ID"], errors="ignore").to_string(index=False))
    n_submitted = int(((results["Status"].str.contains("submitted", case=False, na=False)) |
                       (results["Status"].str.contains("STAGED", na=False))).sum())
    n_failed = int(results["Status"].str.startswith("FAILED").sum())
    n_skipped = int(results["Status"].str.startswith("SKIP").sum())
    print(f"\n  Submitted/Staged: {n_submitted}  |  Failed: {n_failed}  |  Skipped: {n_skipped}")
    if n_failed:
        # Not retried automatically: a decision runs once (run_state last_decision), and failed rows are
        # not in the 9 AM list. Say so plainly.
        failed = results[results["Status"].str.startswith("FAILED")]
        _notify(f"Trade: {n_failed} order(s) not sent",
                f"Alpaca returned an error for: {_list_syms(zip(failed['Symbol'], failed['Shares']))}. "
                f"Other orders are fine. Check Alpaca; place any you still want by hand. "
                f"Log: Reports/live_orders_log.csv",
                details="; ".join(f"{r.Side} {r.Symbol} {r.Shares}: {r.Status}" for r in failed.itertuples()))
    if n_submitted:
        prices = dict(zip(orders["Symbol"].astype(str), pd.to_numeric(orders["Price"], errors="coerce")))
        _notify(*_trade_notice(results, prices))
    _print_header("EVENING LIVE TRADE COMPLETE")
    return orders, meta, results


def earnings_rule_note():
    """One line on the earnings rule for the printout ('' when it is off or the engine is not importable)."""
    try:
        from backtest_engine import WINNER
    except Exception:
        return ""
    n = WINNER.get("earnings_block_days")
    return (f"Earnings rule: stocks with earnings within {n} days are not bought or topped up "
            "(see Reports/strategy_changes.csv)." if n else "")


def main(argv=None):
    """Command line: print the order list (dry run) or submit to the Alpaca LIVE account (REAL MONEY)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--account-size", type=float, help="dollars to allocate (dry run default: 100000)")
    p.add_argument("--positions", help="CSV with Symbol,Shares of current holdings (dry run)")
    p.add_argument("--target", choices=["auto", "current", "provisional", "midweek", "hold"], default="auto",
                   help="'hold' keeps every position unchanged (quiet mid-week check: nothing is submitted)")
    p.add_argument("--fractional", action="store_true", help="fractional share quantities (dry run preview only)")
    p.add_argument("--min-value", type=float, default=1.0, help="skip trades smaller than this many dollars")
    p.add_argument("--out", help="also write the order list to this CSV")
    p.add_argument("--submit", action="store_true", help="send the orders to the Alpaca LIVE account (REAL MONEY)")
    p.add_argument("--paper", action="store_true", help="RETIRED: this script now trades LIVE; --paper is refused")
    p.add_argument("--fill-check", action="store_true",
                   help="morning fill check: complete unfilled extended-hours orders from the last auto_trade (LIVE)")
    p.add_argument("--live", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    if a.paper:
        sys.exit("Refusing: --paper is retired - this script now submits to the Alpaca LIVE "
                 "account. Use --submit (without --paper) to proceed.")
    if a.submit and a.fractional:
        sys.exit("Refusing: submit mode uses whole shares only.")
    if a.fill_check:
        if not a.live:
            sys.exit("Refusing: --fill-check needs --live as well (LIVE trading only).")
        if a.submit or a.fractional or a.account_size or a.positions or a.out or a.target != "auto":
            sys.exit("Refusing: --fill-check takes no other order options.")
        results = complete_unfilled_orders(dry_run=False)
        if results.empty:
            print("No pending extended-hours orders - nothing to do.")
        else:
            print(results.drop(columns=["Order_ID"]).to_string(index=False))
        return results

    positions = read_positions_csv(a.positions) if a.positions else {}
    account_size, buying_power = a.account_size, None
    if a.submit:
        client = paper_trading_client()
        positions = {pos.symbol: float(pos.qty) for pos in client.get_all_positions()}
        acct = client.get_account()
        buying_power = float(acct.buying_power)
        if account_size is None:
            account_size = float(acct.equity)
    account_size = account_size or 100_000.0

    orders, meta, targets = plan_orders(a.target, account_size, positions, min_value=a.min_value, fractional=a.fractional)
    if a.submit:   # same cash cap as the pipeline: buys fit in buying power + the sells' proceeds, less the 1% cushion
        sells_value = orders.loc[orders["Side"] == "SELL", "Est_Value"].sum()
        orders = apply_buying_power_guard(orders, buying_power + sells_value)
    inv = meta["invested"]
    print(f"Strategy: {meta['strategy']} | targets = {meta['source']} weights | as of {meta['as_of']} "
          f"(last weekly rebalance {meta['last_rebalance']}, last decision {meta['last_decision']}) | "
          f"invested {f'{inv:.0%}' if inv is not None else 'unchanged (hold)'})")
    if meta["source"] == "midweek":
        for act, m in zip(meta["swaps"]["Action"], meta["swaps"]["Message"]):
            print(("MID-WEEK EXIT: " if act == "SELL" else "MID-WEEK SWAP: ") + m)
        if not positions and (meta["swaps"]["Action"] == "SWAP").any():
            print("(no --positions given: the buy is sized at the sold stock's target weight x account size)")
    print(earnings_rule_note())
    print(f"Account size ${account_size:,.2f}; prices = latest close (actual fills will differ)\n")
    print(orders.to_string(index=False))
    buys, sells = orders.loc[orders.Side == "BUY", "Est_Value"].sum(), orders.loc[orders.Side == "SELL", "Est_Value"].sum()
    if meta["source"] == "midweek":
        print(f"\nBuys ${buys:,.2f} · Sells ${sells:,.2f} (swaps: same dollars, whole shares; exits: sell only - "
              "the cash stays idle until the Friday rebalance)")
    else:
        held_after = (orders["Target_Shares"].fillna(0) * orders["Price"].fillna(0)).sum()
        print(f"\nBuys ${buys:,.2f} · Sells ${sells:,.2f} · invested after ≈ ${held_after:,.2f} · cash after ≈ ${account_size - held_after:,.2f}")
    if a.out:
        orders.to_csv(a.out, index=False)
    if not a.submit:
        print("\nDRY RUN - nothing was sent. Use --submit to send these orders to the Alpaca LIVE account (REAL MONEY).")
        return orders
    print("\nSubmitting to Alpaca LIVE (REAL MONEY) ...")
    print(submit_paper(orders, positions=positions).to_string(index=False))
    return orders


if __name__ == "__main__":
    main()
