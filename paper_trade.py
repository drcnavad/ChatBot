"""Build (and optionally submit to Alpaca PAPER) the orders that move a portfolio to the strategy's target weights.

DEFAULT = DRY RUN: prints the order list and writes nothing to any broker.

    python paper_trade.py --account-size 25000                       # dry run, assumes an all-cash account
    python paper_trade.py --account-size 25000 --positions my.csv    # dry run vs current holdings (CSV: Symbol,Shares)
    python paper_trade.py --target provisional                       # use "if rebalanced at latest close" weights
    python paper_trade.py --target midweek --positions my.csv        # only the latest Mon/Wed mid-week swap(s) / exit(s)

Submitting requires BOTH flags and only ever talks to the PAPER environment:

    python paper_trade.py --submit --paper [--account-size N]

In submit mode the paper account's equity (unless --account-size is given) and positions are read from Alpaca,
sells are sent before buys as DAY market orders (whole shares). Live trading is not supported by this script.

Auto mode (used by run_all.py --trade): auto_trade() pulls live PAPER positions + equity +
cash, then plans with target="auto" (buys = Weight * equity, whole shares, capped at
STRICT CASH - never buying power/margin; sells = the exact shares held, and only for stocks
actually in the portfolio), and submits as extended-hours DAY limit orders at the planned
closing price, so they can fill in the after-hours session. Before planning, the signal
CSVs are verified fresh (as of today, CT) - stale data aborts the trade. Submitted orders are recorded
in Reports/paper_pending_orders.json; the next trading morning, complete_unfilled_orders()
(run_all.py --fill-check) checks their fills and completes any unfilled remainder with
regular-hours market orders, then reconciles live positions against the strategy targets.
Failures send a macOS notification + loud log alert. If auto_trade() runs at or after 7:00 PM CT (extended hours
are over), it submits nothing and instead stages the planned orders in
Reports/paper_pending_orders.json (no broker order id); the morning fill check then sends
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
Earnings rule (backtest_engine.WINNER["earnings_block_days"] = 5): the targets already leave out stocks that were not held and
have earnings within 5 calendar days of the decision, so this script needs no earnings check of its own.
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
ORDER_LOG_CSV = os.path.join(PROJECT_ROOT, "Reports", "paper_orders_log.csv")
PENDING_ORDERS_JSON = os.path.join(PROJECT_ROOT, "Reports", "paper_pending_orders.json")
ORDER_COLUMNS = ["Symbol", "Side", "Shares", "Price", "Est_Value", "Current_Shares", "Target_Shares",
                 "Target_Weight_%", "Target_Value"]


# ----------------------------------------------------------------------------- notifications, terminal formatting, data freshness
def _notify(title, message):
    """Alert the user that automated trading needs attention (macOS notification + loud print).

    Best effort: the print is the primary channel (it lands in the launchd logs);
    the macOS notification is attempted so a failure is visible even without
    checking logs. Never raises.
    """
    banner = f"\n{'!' * 70}\n  ALERT: {title}\n  {message}\n{'!' * 70}\n"
    print(banner, flush=True)
    try:
        import subprocess
        safe_title = str(title).replace('"', "'").replace("\\", "")[:100]
        safe_msg = str(message).replace('"', "'").replace("\\", "")[:300]
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{safe_msg}" with title "{safe_title}" sound name "Basso"'],
            timeout=5, capture_output=True)
    except Exception:
        pass  # the print above is the fallback


def _trade_summary(results):
    """One-line 'Bought: NVDA x10 | Sold: INTC x8' for the submitted/staged rows of a
    results DataFrame (columns Symbol, Side, Shares, Status). Pure: safe to unit-test."""
    def _fmt(side):
        rows = results[(results["Side"] == side) &
                       (results["Status"].str.contains("submitted|STAGED", case=False, na=False))]
        return ", ".join(f"{r.Symbol} x{r.Shares:g}" for r in rows.itertuples())
    buys, sells = _fmt("BUY"), _fmt("SELL")
    return f"Bought: {buys or 'none'} | Sold: {sells or 'none'}"


def _print_header(title):
    """A clear section header so launchd/terminal logs are easy to scan."""
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}", flush=True)


def _print_section(title):
    print(f"\n  --- {title} ---", flush=True)


def check_signal_freshness(picks_csv=PICKS_CSV, changes_csv=CHANGES_CSV):
    """Verify the signal CSVs are from today (CT) before any trading.

    Fail-closed: raises ValueError when the data is stale or missing, so
    auto_trade never trades on yesterday's signals after a partial pipeline
    failure. Returns the As_Of date string when fresh.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    today = datetime.now(ZoneInfo("America/Chicago")).date().isoformat()
    for path, label in [(picks_csv, "strategy_picks.csv"), (changes_csv, "strategy_changes.csv")]:
        if not os.path.exists(path):
            raise ValueError(f"STALE DATA: {label} not found - run the pipeline before trading")
    picks = pd.read_csv(picks_csv)
    if picks.empty or "As_Of" not in picks.columns:
        raise ValueError("STALE DATA: strategy_picks.csv has no usable As_Of - run the pipeline before trading")
    as_of = str(picks["As_Of"].iloc[0])[:10]
    if as_of != today:
        raise ValueError(
            f"STALE DATA: strategy_picks.csv is as of {as_of}, but today is {today} (CT) - "
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
            f"STALE DATA: strategy_changes.csv is as of {changes_date}, but today is {today} (CT) - "
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


def load_targets(source="auto", picks_csv=PICKS_CSV, midweek_csv=MIDWEEK_CSV):
    """(targets DataFrame[Symbol, Weight, Price], meta dict). Weights are fractions of the account (rest = cash).

    meta['swaps'] holds the latest mid-week swap rows (source 'midweek'); targets are then the weights after the swap."""
    picks = pd.read_csv(picks_csv)
    if picks.empty:
        raise ValueError(f"{picks_csv} is empty - run main_signal_analysis.ipynb first")
    as_of, last_reb = str(picks["As_Of"].iloc[0]), str(picks["Last_Rebalance"].iloc[0])
    last_dec = str(picks["Last_Decision"].iloc[0]) if "Last_Decision" in picks.columns else last_reb
    swaps = latest_midweek_swaps(midweek_csv, as_of)
    if source == "auto":
        # A quiet mid-week day (no swap/exit at the latest check) holds every position
        # unchanged, matching the backtest: no drift rebalance, and cash from a mid-week
        # exit stays idle until the Friday rebalance.
        source = "provisional" if as_of == last_reb else ("midweek" if len(swaps) else "hold")
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


def latest_signal_status(changes_csv=CHANGES_CSV, as_of=None):
    """{SYMBOL: 'add'|'hold'|'drop'|...} from the latest decision rows of strategy_changes.csv.

    Used to tell buy-signal (add) names apart from hold names: BUY orders go out only for
    buy-signal symbols, never for holds. Returns {} when the file is missing/unreadable -
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


def _clean_positions(positions):
    """Normalize a {symbol: shares} map: drop zero/NaN/inf/negative holdings.

    A NaN share count would otherwise flow into build_orders() and produce NaN order
    quantities, which crash math.floor() at submit time. Negative quantities are
    impossible in a long-only paper account, so they are dropped too."""
    out = {}
    for k, v in (positions or {}).items():
        try:
            shares = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(shares) and shares > 0:
            out[str(k).upper()] = shares
    return out


# ----------------------------------------------------------------------------- order sizing (whole shares by default)
def build_orders(targets, account_size, positions=None, prices=None, min_value=1.0, fractional=False,
               statuses=None):
    """Orders (sells first) to move `positions` {symbol: shares} toward target weights of `account_size` dollars.

    BUY orders go out only for buy-signal ('add') symbols: the buy is sized so the holding reaches
    Weight x account_size (whole shares, rounded down, dollar math at cent precision) - never more.
    If the holding is already above its weight, the excess is SOLD instead of buying. Symbols whose
    signal status is 'hold' get HOLD rows and are never traded on a rebalance, even when
    under/overweight. Positions not in the targets (sell signals) are sold completely.
    Trades worth less than `min_value` are skipped. `statuses` maps SYMBOL -> status string from
    strategy_changes.csv; a missing/unknown status is treated as HOLD (fail closed), never as a
    buy signal.
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
        if sym in wanted and not _is_buy_signal(statuses.get(sym)):
            # Hold-signal name: never traded on a rebalance - no BUY top-up, no trim SELL.
            rows.append({"Symbol": sym, "Side": "HOLD", "Shares": 0,
                         "Price": float(price) if price else price, "Est_Value": 0.0,
                         "Current_Shares": cur, "Target_Shares": None,
                         "Target_Weight_%": round(weight * 100, 2), "Target_Value": round(weight * account_size, 2)})
            continue
        if price is None or not (price > 0):
            rows.append({"Symbol": sym, "Side": "SKIP (no price)", "Shares": 0, "Price": price, "Est_Value": 0.0,
                         "Current_Shares": cur, "Target_Shares": None, "Target_Weight_%": weight * 100, "Target_Value": weight * account_size})
            continue
        target_value = weight * account_size
        tgt = round(target_value / price, 4) if fractional else math.floor(target_value / price)
        delta = round(tgt - cur, 4)
        if delta == 0 or abs(delta) * price < min_value:
            side = "HOLD"
        else:
            side = "BUY" if delta > 0 else "SELL"
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
    Whole shares (rounded down) by default. Every other holding is left alone (HOLD rows)."""
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
                q = round(net_dollars / p_in, 4) if fractional else math.floor(net_dollars / p_in)
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
                sq = min(math.floor(-net_dollars / p_in), int(math.floor(cur_in)))
                if sq >= 1:
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
                min_value=1.0, fractional=False):
    """(orders, meta, targets) for any target source; used by the CLI and the app's order preview. No broker calls."""
    targets, meta = load_targets(source, picks_csv, midweek_csv)
    positions = positions or {}
    if meta["source"] == "hold":
        # Quiet mid-week check: the backtest holds every position unchanged (no drift
        # rebalance), so live emits HOLD rows for all current positions - nothing is submitted.
        px = latest_prices(sorted(positions), signal_csv)
        return build_hold_orders(positions, px), meta, targets
    if meta["source"] == "midweek":
        syms = sorted(set(meta["swaps"]["Sell"].astype(str)) | set(meta["swaps"]["Buy"].dropna().astype(str)) | set(positions))
        px = latest_prices(syms, signal_csv)
        px.update(dict(zip(targets["Symbol"], targets["Price"])))
        return build_swap_orders(meta["swaps"], account_size, positions, px, fractional=fractional), meta, targets
    prices = latest_prices([s for s in positions if s not in set(targets["Symbol"])], signal_csv)
    # Buy-signal ('add') vs hold names come from the latest decision in strategy_changes.csv:
    # BUY orders go out only for buy-signal symbols, never for holds.
    statuses = latest_signal_status(as_of=meta["as_of"])
    return build_orders(targets, account_size, positions, prices, min_value=min_value, fractional=fractional,
                        statuses=statuses), meta, targets


def apply_buying_power_guard(orders, buying_power):
    """Cap total BUY spending at the account's BUYING POWER.

    Weights are fractions of the whole portfolio, so targets are still computed on equity;
    this guard only ensures the plan never tries to spend more than Alpaca will let the
    account spend (an over-sized buy would otherwise be rejected by the broker). Buying
    power is read from the Alpaca account itself. Margin is disabled on this account, so
    buying power equals cash in practice. Buys are scaled down proportionally when their
    total exceeds buying power; a buy scaled to zero shares becomes a
    SKIP (no buying power) row and is never submitted. Dollar values stay at cent precision.
    An unknown/invalid buying-power value fails closed: BUY rows become
    SKIP (no buying power) and are never submitted, instead of going out uncapped.
    Returns the adjusted orders DataFrame.
    """
    orders = orders.copy()
    buys = orders["Side"] == "BUY"
    if not buys.any():
        return orders
    try:
        available = float(buying_power)
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
        q = int(math.floor(shares * scale))
        if q < 1 or price <= 0:
            orders.at[i, "Side"] = "SKIP (no buying power)"
            orders.at[i, "Shares"] = 0
            orders.at[i, "Est_Value"] = 0.0
        else:
            orders.at[i, "Shares"] = q
            orders.at[i, "Est_Value"] = round(q * price, 2)
            orders.at[i, "Target_Shares"] = _safe_number(orders.at[i, "Current_Shares"]) + q
    return orders


# ----------------------------------------------------------------------------- positions file, PAPER submission (explicit --submit --paper only), command line
def read_positions_csv(path):
    """Positions CSV (Symbol,Shares or Symbol,Qty) -> {symbol: shares}."""
    pos = pd.read_csv(path)
    pos.columns = [c.strip().lower() for c in pos.columns]
    if "shares" not in pos.columns and "qty" not in pos.columns:
        raise SystemExit(f"{path}: needs a Shares column for order sizing (Symbol,Shares); a Weight-only file works for the alert only")
    qty = "shares" if "shares" in pos.columns else "qty"
    return dict(zip(pos["symbol"].astype(str).str.upper().str.strip(), pd.to_numeric(pos[qty], errors="coerce").fillna(0)))


def paper_trading_client():
    """Alpaca TradingClient for the PAPER environment only (keys: see alpaca_paper.paper_keys; never a live account)."""
    from alpaca.trading.client import TradingClient

    from alpaca_paper import paper_keys
    key, secret, _ = paper_keys()
    if not key or not secret:
        raise SystemExit("No Alpaca PAPER keys in .env (ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY).")
    return TradingClient(key, secret, paper=True)


def _sell_qty_check(symbol, qty, positions, fractional=False):
    """(ok_qty, skip_reason): a SELL may only go out for shares actually held.

    Returns (0, reason) when the symbol is not in the portfolio (or <1 share is
    held) - the row is then SKIPPED, never submitted, because the broker would reject it.
    Otherwise the quantity is clamped to the shares held.
    `positions` None disables the check (caller did not supply holdings).
    `fractional`: False (default) for LIMIT orders - always whole shares (Alpaca
    requires integer quantities on limit orders). True for MARKET orders - up to
    2 decimals allowed, so fractional holdings (e.g. 10.5 shares) can be sold
    completely instead of leaving a dust remainder.
    NaN/inf/negative holdings are treated as not held (fail closed)."""
    qty = _safe_number(qty)
    if qty <= 0:
        return 0, "SKIPPED: zero quantity"
    if positions is None:
        return qty, None
    norm = _clean_positions(positions)
    have = norm.get(str(symbol).upper(), 0.0)
    if have <= 0:
        return 0, "SKIPPED: not in portfolio"
    if fractional:
        # Market order: up to 2 decimals, clamped to what is held.
        have_qty = math.floor(have * 100) / 100
        ok_qty = min(math.floor(qty * 100) / 100, have_qty)
        if ok_qty <= 0:
            return 0, "SKIPPED: zero quantity"
        return ok_qty, None
    # Limit order: whole shares only.
    have_qty = int(math.floor(have))
    if have_qty < 1:
        return 0, "SKIPPED: not in portfolio"
    ok_qty = int(math.floor(qty))
    if ok_qty > have_qty:
        return have_qty, None
    if ok_qty < 1:
        return 0, "SKIPPED: <1 whole share"
    return ok_qty, None


def submit_paper(orders, positions=None, client=None, order_date=None):
    """Send the BUY/SELL rows as DAY market orders to the Alpaca PAPER account (sells first).

    Market orders allow up to 2 decimals: quantities are rounded DOWN to 2 decimals and
    rows with <=0 shares are skipped (reported as SKIPPED, not sent). This lets a SELL
    clear a fractional holding completely (e.g. 10.5 shares) instead of leaving dust.
    SELL rows are checked against `positions` ({SYMBOL: shares} from the live account when
    given): a symbol not held is SKIPPED, never submitted, and the sell quantity is
    clamped to the shares actually held (2-decimal precision for market orders).
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
        qty = math.floor(_safe_number(r.Shares) * 100) / 100
        if qty <= 0:
            results.append((r.Symbol, r.Side, r.Shares, "SKIPPED: zero quantity"))
            continue
        if r.Side == "SELL":
            qty, reason = _sell_qty_check(r.Symbol, qty, positions, fractional=True)
            if reason:
                results.append((r.Symbol, r.Side, r.Shares, reason))
                continue
        req = MarketOrderRequest(symbol=r.Symbol, qty=qty, time_in_force=TimeInForce.DAY,
                                 side=OrderSide.SELL if r.Side == "SELL" else OrderSide.BUY,
                                 client_order_id=_client_order_id(r.Symbol, r.Side, qty, 0, date_str))
        try:
            o = client.submit_order(req)
            results.append((r.Symbol, r.Side, qty, str(o.status)))
        except Exception as e:  # keep going; report every failure
            results.append((r.Symbol, r.Side, qty, f"FAILED: {e}"))
    return pd.DataFrame(results, columns=["Symbol", "Side", "Shares", "Status"])


def submit_paper_extended(orders, positions=None, record=None, client=None, order_date=None):
    """Send the BUY/SELL rows as DAY limit orders at the planned (closing) price, eligible for
    extended-hours (after-hours) execution on the Alpaca PAPER account (sells first).

    Whole shares only: fractional quantities are floored to whole shares and
    rows with <1 whole share are skipped (reported as SKIPPED, not sent).
    SELL rows are checked against `positions` ({SYMBOL: shares} from the live account when
    given): a symbol not held is SKIPPED, never submitted, and the sell quantity is
    clamped to the whole shares actually held.
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
        # Limit orders REQUIRE integer shares on Alpaca - always whole shares here.
        qty = int(math.floor(_safe_number(r.Shares)))
        if qty < 1:
            results.append((r.Symbol, r.Side, r.Shares, None, "SKIPPED: <1 whole share"))
            continue
        if r.Side == "SELL":
            qty, reason = _sell_qty_check(r.Symbol, qty, positions, fractional=False)
            if reason:
                results.append((r.Symbol, r.Side, r.Shares, None, reason))
                continue
            if qty < 1:
                results.append((r.Symbol, r.Side, r.Shares, None, "SKIPPED: <1 whole share held"))
                continue
        try:
            price = round(float(r.Price), 2)
        except (TypeError, ValueError):
            price = 0
        if not price > 0:
            results.append((r.Symbol, r.Side, qty, None, "SKIPPED: no price"))
            continue
        req = LimitOrderRequest(symbol=r.Symbol, qty=qty, limit_price=price,
                                time_in_force=TimeInForce.DAY, extended_hours=True,
                                side=OrderSide.SELL if r.Side == "SELL" else OrderSide.BUY,
                                client_order_id=_client_order_id(r.Symbol, r.Side, qty, price, date_str))
        try:
            o = client.submit_order(req)
            results.append((r.Symbol, r.Side, qty, str(o.id), str(o.status)))
            if record is not None:
                record({"symbol": r.Symbol, "side": r.Side, "qty": qty,
                        "limit_price": price, "order_id": str(o.id)})
        except Exception as e:  # keep going; report every failure
            results.append((r.Symbol, r.Side, qty, None, f"FAILED: {e}"))
    return pd.DataFrame(results, columns=["Symbol", "Side", "Shares", "Order_ID", "Status"])


# ----------------------------------------------------------------------------- crash-safe order ids + broker reconciliation
def _client_order_id(symbol, side, qty, price, date_str, kind=""):
    """Deterministic client order id, stable across retries of the same plan.

    Format: pa-[fill-]YYYYMMDD-SIDE-SYMBOL-qty-pricecents (<=48 chars, Alpaca-safe).
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
    tag = f"pa-{kind + '-' if kind else ''}{date_str}-{side}-{sym}-{int(_safe_number(qty))}-{cents}"
    return tag[:48]


def _broker_orders_by_client_id(client):
    """{client_order_id: order} for the broker's recent orders (open + closed).

    Best effort: returns {} when the broker cannot be read - the local pending file
    remains the primary record; this only closes the crash gap between a submit and
    its local record.
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
        print(f"WARNING: broker order reconciliation read failed ({e}) - relying on local record")
        return {}


def _evening_submitted_on_broker(client, date_str):
    """{(SYMBOL, SIDE)} already submitted to the broker this evening (any status).

    Matches our deterministic client_order_id prefix pa-YYYYMMDD-SIDE-SYMBOL- (morning
    fill-check orders, pa-fill-..., are excluded). A row found here was submitted before
    a crash, so its local record never got written; skipping it on retry is always safe
    because the morning fill check completes any unfilled remainder.
    """
    out = set()
    for cid in _broker_orders_by_client_id(client):
        parts = cid.split("-")
        if len(parts) < 6 or parts[0] != "pa" or parts[1] == "fill":
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


def submit_paper_extended_sequenced(orders, positions=None, record=None, client=None, order_date=None):
    """Evening extended-hours submission with SELLs settled before BUYs are sized.

    1. Submits the SELL rows as extended-hours limit orders (each recorded).
    2. Waits (bounded, SELL_SETTLE_WAIT_SECS) for the sells to reach terminal states.
    3. Refreshes BUYING POWER from the broker and re-applies the buying-power guard
       to the BUY rows with the fresh number (falls back to the already-guarded plan
       when the refresh fails). Sells that filled have released buying power.
    4. Submits the BUY rows as extended-hours limit orders (each recorded).
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
    if fresh_bp is not None:
        buys = apply_buying_power_guard(buys, fresh_bp)
    buy_results = submit_paper_extended(buys, positions=positions, record=record,
                                        client=client, order_date=order_date)
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
        if pf >= qty - covered:
            return None, 0, prior
        covered += pf
        cid, n = (base + f"-r{n}")[:48], n + 1
    # Market orders allow up to 2 decimals: round the remainder DOWN to 2 decimals.
    return cid, math.floor(_safe_number(qty - covered) * 100) / 100, None


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
    if pend.get("evening_date") != _today_ct().date().isoformat():
        return set()
    return {(str(o.get("symbol", "")).upper(), str(o.get("side", "")).upper())
            for o in pend.get("orders", [])}


def record_pending_order(entry, meta, pending_path=PENDING_ORDERS_JSON):
    """Append one submitted/staged evening order to the pending file (atomic, de-duplicated).

    Called after each successful evening submission so a crash mid-batch still leaves the
    submitted orders recorded - a retry then skips them instead of duplicating them.
    Entries are keyed by (symbol, side): re-recording replaces the old entry. A pending
    file from a previous evening (never picked up by a morning fill check) is backed up
    with a warning, never silently dropped.
    """
    now_ct = _today_ct()
    today = now_ct.date().isoformat()
    try:
        with open(pending_path) as f:
            pend = json.load(f)
    except (OSError, ValueError):
        pend = {}
    existing = pend.get("orders", []) if isinstance(pend, dict) else []
    if existing and pend.get("evening_date") != today:
        backup = f"{pending_path}.bak_{now_ct.strftime('%Y%m%d_%H%M%S')}"
        os.replace(pending_path, backup)
        print(f"WARNING: {pending_path} still held {len(existing)} unprocessed order(s) from "
              f"{pend.get('evening_date')} - backed up to {backup}; review it manually, it was NOT merged.")
        existing, pend = [], {}
    key = (str(entry["symbol"]).upper(), str(entry["side"]).upper())
    orders_list = [o for o in existing
                   if (str(o.get("symbol", "")).upper(), str(o.get("side", "")).upper()) != key]
    orders_list.append(entry)
    payload = {"evening_date": today,
               "submitted_at_ct": now_ct.strftime("%Y-%m-%d %H:%M:%S"),
               "target_source": (pend.get("target_source") if isinstance(pend, dict) else None) or meta.get("source"),
               "as_of": (pend.get("as_of") if isinstance(pend, dict) else None) or meta.get("as_of"),
               "orders": orders_list}
    os.makedirs(os.path.dirname(pending_path), exist_ok=True)
    tmp = pending_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, pending_path)


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
            return True  # gone from the broker - treat as terminal
        raw = cur.status
        st = raw.value.lower() if hasattr(raw, "value") else str(raw).lower()
        if st in terminal:
            return True
        time.sleep(pause)
    return False


def complete_unfilled_orders(pending_path=PENDING_ORDERS_JSON, log_csv=ORDER_LOG_CSV, dry_run=False):
    """Morning fill check for the previous evening's extended-hours orders (PAPER only).

    Reads Reports/paper_pending_orders.json (written by auto_trade), checks each order's fill
    status on Alpaca, and for anything not fully filled submits a regular-hours DAY market order
    for the remaining whole shares (the stale evening order is canceled first, best effort).
    Partial fills are handled: only the unfilled remainder is ordered.
    Orders staged by a past-7PM evening run (no broker order id) were never submitted, so the
    full quantity is sent as a regular-hours market order.
    A rejected evening order is never auto-retried - it is logged as REJECTED for review
    and dropped from the pending list.

    Fail closed: when the live positions cannot be read, the run aborts with RuntimeError
    (the pending file is kept for a retry) instead of completing SELL remainders unclamped.
    A dry run never aborts on this (nothing is submitted).

    Crash-safe morning orders: each completion order carries a deterministic client_order_id
    (pa-fill-YYYYMMDD-SIDE-SYMBOL-qty-...). A previous attempt that crashed between its
    submit and its local record is found on the broker by id and marked ALREADY COMPLETED
    instead of duplicated.

    The pending file is removed once every order reached a terminal state without a FAILED
    completion; if any completion failed, only the failed rows are kept (with per-row
    completion marks) so a retry never re-orders what already completed.
    A dry run changes nothing: it neither submits, nor logs, nor removes the pending file.
    Appends the morning orders to Reports/paper_orders_log.csv (same columns as the evening log).
    Returns a DataFrame [Symbol, Side, Shares, Order_ID, Status].
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    cols = ["Symbol", "Side", "Shares", "Order_ID", "Status"]
    _print_header("MORNING FILL CHECK - Alpaca PAPER")
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
    fill_date = _today_ct().date().strftime("%Y%m%d")
    results = []
    to_retry = []  # rows whose completion FAILED - only these are kept for a retry
    morning_sell_ids = []  # market SELL completions submitted by this run (waited on before BUYs)
    sells_settled = False  # ... have been waited on before the first BUY affordability check
    reserved_buy_spend = 0.0  # estimated cost of BUY completions already submitted this run
    for o in evening_orders:
        # Defensive parse: a malformed row can never be completed. Drop it loudly
        # (FAILED, not kept for retry - a retry could never parse it either) instead
        # of crashing the whole fill check. An unknown side is dropped too: without
        # this it would fall through to the BUY branch below.
        if not isinstance(o, dict):
            results.append(("?", "?", "?", None,
                            "FAILED: malformed pending row (not an object) - dropped, needs review"))
            continue
        try:
            qty = int(o["qty"])
        except (KeyError, TypeError, ValueError):
            qty = None
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
            if status == "filled" or filled >= qty:
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
        # Market orders allow up to 2 decimals: round the remainder DOWN to 2 decimals
        # (never up - we never order more than the unfilled remainder).
        remaining = math.floor(_safe_number(qty - filled) * 100) / 100
        if remaining <= 0:
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
            remaining = min(remaining, math.floor(have * 100) / 100)
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
                except Exception:
                    pass
                remaining = math.floor(_safe_number(qty - filled) * 100) / 100
                if side == "SELL" and live_positions is not None:
                    # Re-apply the clamp: the re-read recomputed the remainder from the
                    # evening order, discarding the clamp applied above.
                    # 2-decimal precision for market orders.
                    remaining = min(remaining,
                                    math.floor(_safe_number(
                                        live_positions.get(str(sym).upper(), 0.0)) * 100) / 100)
                if remaining <= 0:
                    results.append((sym, side, qty, oid,
                                    f"NO FILL (filled while canceling, {filled:g}/{qty})"))
                    continue
            # Crash-safety: a previous attempt may have submitted the morning order already
            # (crash before recording it). Match by deterministic client order id instead
            # of submitting a duplicate.
            cid, to_order, prior = _morning_completion_plan(
                broker_by_cid, sym, side, remaining, o.get("limit_price") or 0, fill_date)
            if prior is not None:
                o["completed_order_id"] = str(prior.id)  # retry-safety: this row is done
                results.append((sym, side, remaining, str(prior.id),
                                f"ALREADY COMPLETED (morning order {prior.id} covers the remainder)"))
                continue
            if to_order <= 0:
                results.append((sym, side, remaining, oid,
                                "NO FILL (remainder <=0 after prior attempt)"))
                continue
            if side == "BUY":
                # Buying-power check for morning BUY completions. SELL completions were
                # processed first (sorted above): wait for this run's SELLs to settle,
                # then re-read fresh buying power and verify it covers the estimated market
                # cost. The estimate uses the evening limit price + 5% buffer - market
                # orders can fill above the estimate and overnight gaps can exceed 2%;
                # the broker independently rejects anything buying power cannot cover, so
                # this guard is planning-layer strictness. Fail closed: an unreadable
                # buying-power figure or insufficient buying power keeps the row for
                # retry; a missing or invalid price can never be verified, so that row
                # is dropped loudly instead of being submitted blind or retried forever.
                if not sells_settled:
                    if morning_sell_ids:
                        _wait_for_terminal_all(client, morning_sell_ids,
                                               timeout_secs=60, poll_secs=5)
                    sells_settled = True
                try:
                    bp_now = _read_buying_power(client)
                except Exception:
                    bp_now = None
                est_price = _safe_number(o.get("limit_price"))
                if not (est_price > 0):
                    msg = (f"FAILED: no usable price for {sym} BUY {to_order:g} shares - "
                           f"affordability cannot be verified (dropped, needs review)")
                    results.append((sym, side, to_order, oid, msg))
                    _notify("Fill-check: BUY dropped (no price)",
                            f"{sym}: morning BUY {to_order:g} shares has no usable limit price, "
                            f"so its cost cannot be verified against buying power. Dropped for manual review.")
                    continue
                if bp_now is None or not math.isfinite(bp_now) or bp_now < 0:
                    msg = (f"SKIP (morning buying power unreadable) - {sym} BUY {to_order:g} shares "
                           f"not completed")
                    results.append((sym, side, to_order, oid, msg))
                    _notify("Fill-check: buying power unreadable",
                            f"{sym}: cannot verify buying power for morning BUY completion "
                            f"({to_order:g} shares). Kept for retry.")
                    to_retry.append(o)
                    continue
                est_cost = to_order * est_price * 1.05  # 5% buffer for market slippage/gaps
                spendable = bp_now - reserved_buy_spend  # less BUYs already submitted this run
                if est_cost > spendable:
                    msg = (f"SKIP (insufficient morning buying power: need ~${est_cost:,.2f}, "
                           f"have ${spendable:,.2f} after ${reserved_buy_spend:,.2f} reserved) - "
                           f"{sym} BUY {to_order:g} shares not completed")
                    results.append((sym, side, to_order, oid, msg))
                    _notify("Fill-check: insufficient buying power",
                            f"{sym}: morning BUY {to_order:g} shares needs ~${est_cost:,.2f} "
                            f"but only ${spendable:,.2f} buying power is available. Kept for retry.")
                    to_retry.append(o)
                    continue
            req = MarketOrderRequest(symbol=sym, qty=to_order, time_in_force=TimeInForce.DAY,
                                     side=OrderSide.SELL if side == "SELL" else OrderSide.BUY,
                                     client_order_id=cid)
            m = client.submit_order(req)
            o["completed_order_id"] = str(m.id)  # retry-safety: this row is done
            if side == "SELL":
                morning_sell_ids.append(str(m.id))  # waited on before the first BUY
            else:
                reserved_buy_spend += est_cost  # later BUYs are judged on what is left
            results.append((sym, side, to_order, str(m.id),
                            f"COMPLETED via market ({status}, filled {filled:g}/{qty})"))
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
        failed_syms = ", ".join(results.loc[results["Status"].str.startswith("FAILED"), "Symbol"].astype(str))
        _notify("Fill-check had FAILED orders",
                f"{n_failed} completion(s) failed: {failed_syms}. "
                "They are kept in Reports/paper_pending_orders.json for retry.")
    if not dry_run and not results.empty:
        def _syms(mask):
            return ", ".join(f"{r.Symbol} x{r.Shares:g}" for r in results[mask].itertuples())
        completed = _syms(results["Status"].str.contains("COMPLETED", na=False))
        filled = _syms(results["Status"].str.startswith("FILLED"))
        _notify("Morning fill-check done",
                f"Completed: {completed or 'none'} | Evening fills: {filled or 'none'} | "
                f"Failed: {n_failed} | Kept for retry: {len(to_retry)}")

    if not dry_run and not to_retry:
        # All completions done (or nothing needed doing): verify the portfolio
        # actually matches the strategy targets.
        _print_section("Position reconciliation (live vs strategy targets)")
        report, ok = reconcile_positions(pend.get("target_source") or "auto")
        if not report.empty:
            print(report.to_string(index=False))
        if not ok:
            drift = report[report["Status"] != "OK"]
            syms = ", ".join(drift["Symbol"].astype(str))
            _notify("Position DRIFT detected",
                    f"These differ from strategy targets by >1pp: {syms}. "
                    "Review manually - a fill may have failed silently.")
            print("\n  !! DRIFT detected - notification sent, review manually.")
        else:
            print("\n  All positions match strategy targets (within 1pp).")
    _print_header("MORNING FILL CHECK COMPLETE")
    return results


def get_live_positions_and_equity():
    """(positions dict, equity float, cash float, buying power float) from the Alpaca PAPER account.

    Positions are {SYMBOL: shares} with fractional shares preserved here
    (submit_paper floors them to whole shares when sending). Buying power is what
    apply_buying_power_guard() caps planned BUY spending at (margin is disabled on
    the account, so buying power equals cash in practice). Cash is returned for
    display only.
    """
    from alpaca_paper import PaperAccount
    acct = PaperAccount()
    summary = acct.account_summary()
    pos_df = acct.positions()
    if pos_df.empty:
        positions = {}
    else:
        positions = dict(zip(pos_df["Symbol"].astype(str).str.upper().str.strip(),
                             pd.to_numeric(pos_df["Qty"], errors="coerce").fillna(0)))
        positions = {k: float(v) for k, v in positions.items() if float(v) != 0}
    return positions, float(summary["Equity"]), float(summary["Cash"]), float(summary["Buying power"])


EVENING_SUBMIT_CUTOFF_CT = "19:00"  # extended hours end 7:00 PM CT - later runs stage orders for the morning


def reconcile_positions(target_source="auto", tolerance_pct=1.0):
    """Compare live Alpaca positions against the strategy's target weights.

    Called after the morning fill check: catches silent drift between what the
    strategy wants and what the broker actually holds (e.g. a rejected order,
    a partial fill that never completed, or a manual trade in the account).

    Returns (report_df, ok): report_df has one row per symbol with target vs
    actual weight; ok is False when any symbol differs by more than tolerance_pct.
    Never raises - returns (empty_df, False) with a note when data is unavailable.
    """
    cols = ["Symbol", "Target_Weight_%", "Actual_Weight_%", "Diff_pp", "Shares_Held", "Status"]
    try:
        targets, meta = load_targets(target_source)
        positions, equity, cash, _ = get_live_positions_and_equity()
    except Exception as e:
        return pd.DataFrame(columns=cols), False
    if meta.get("source") == "hold" or targets.empty:
        # A hold day orders nothing: there are no targets to reconcile against.
        # (Without this, every held position would false-alarm as DRIFT against a
        # 0% target.)
        return pd.DataFrame(columns=cols), True
    if equity <= 0:
        return pd.DataFrame(columns=cols), False
    syms = sorted(set(targets["Symbol"].astype(str)) | {str(s) for s in positions})
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
        status = "OK" if abs(diff) <= tolerance_pct else ("MISSING" if tw > 0 and shares < 1 else "DRIFT")
        rows.append({"Symbol": sym, "Target_Weight_%": round(tw, 2), "Actual_Weight_%": round(aw, 2),
                     "Diff_pp": round(diff, 2), "Shares_Held": shares, "Status": status})
    report = pd.DataFrame(rows, columns=cols)
    ok = bool((report["Status"] == "OK").all()) if not report.empty else True
    return report, ok


def _past_evening_cutoff(now=None):
    """True when it is at or past the extended-hours submission cutoff (America/Chicago)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = now or datetime.now(ZoneInfo("America/Chicago"))
    return now.strftime("%H:%M") >= EVENING_SUBMIT_CUTOFF_CT


def auto_trade(target="auto", min_value=1.0, log_csv=ORDER_LOG_CSV, dry_run=False, extended=True):
    """Full auto flow for the pipeline: pull live PAPER positions + equity, plan, optionally submit.

    - Pulls current stock positions, equity, cash and buying power from Alpaca PAPER.
    - Plans orders with source=target (default "auto"): BUY orders go out only for buy-signal
      ('add') symbols, sized as Weight * equity in whole shares (rounded down) and net of shares
      already held - a buy never exceeds its weight limit, and a holding already above its weight
      is trimmed with a SELL of the excess instead of buying. Hold-signal symbols are never
      traded on a rebalance. Sells sell-signal stocks entirely (the exact shares held).
    - Caps total BUY spending at Alpaca's BUYING POWER via apply_buying_power_guard():
      a buy that would exceed available buying power is scaled down proportionally
      instead of being rejected by the broker. Margin is disabled on this account,
      so buying power equals cash in practice.
    - SELL rows are verified against the live portfolio before anything is sent: a symbol not
      held is SKIPPED (never submitted), and the sell quantity is clamped to the whole shares
      actually held.
    - If dry_run=True: returns (orders, meta, results_df) without submitting; results_df is empty.
    - If extended=True (default): submits via submit_paper_extended_sequenced() - the SELLs go
      first as DAY limit orders at the planned closing price with extended_hours=True; after a
      bounded wait for the sells to settle, buying power is re-read from the broker and the
      BUYs are re-checked against it (apply_buying_power_guard) before they are submitted the same way.
      Each successfully submitted order is recorded incrementally in
      Reports/paper_pending_orders.json for the morning fill check (complete_unfilled_orders).
      Every order carries a deterministic client_order_id, so a crash between the broker submit
      and the local record cannot duplicate it on retry (the retry reconciles with the broker).
    - If extended=False: submits via submit_paper() - regular-hours DAY market orders.
    - If it is at or past 7:00 PM CT (extended hours are over), nothing is submitted regardless
      of `extended`: the planned orders are staged in Reports/paper_pending_orders.json with no
      broker order id, and the morning fill check (complete_unfilled_orders) sends the full
      quantities as regular-hours market orders. Staged rows are reported as STAGED, never FAILED.
    - Idempotent retries: rows already submitted/staged earlier the same evening are recorded
      after each submit, so re-running --trade in the same window skips them instead of
      duplicating orders. Rows found on the broker from this evening but never recorded
      locally (a crash between submit and record) are likewise skipped - the morning fill
      check completes any unfilled remainder. FAILED rows are never recorded, so they are retried.
    - Appends a row per submitted order to Reports/paper_orders_log.csv and returns
      (orders, meta, results_df).

    Raises PaperAccountError / SystemExit if keys are missing; ValueError if the
    signal CSVs are missing or empty (run main_signal_analysis.ipynb first).
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    _print_header("EVENING TRADE - Alpaca PAPER")
    try:
        # Fail closed on stale signals: never trade on yesterday's data after a
        # partial pipeline failure.
        as_of_date = check_signal_freshness()
        print(f"  Signals fresh: as of {as_of_date} (today, CT)")
    except ValueError as e:
        _notify("Trade ABORTED - stale signals", str(e))
        raise

    try:
        positions, equity, cash, buying_power = get_live_positions_and_equity()
    except Exception as e:
        _notify("Trade ABORTED - cannot read account", f"Could not read Alpaca PAPER account: {e}")
        raise
    print(f"  Equity ${equity:,.2f}  |  Cash ${cash:,.2f}  |  Positions: {len(positions)} symbols")

    _print_section("Planning orders")
    orders, meta, targets = plan_orders(target, equity, positions, min_value=min_value, fractional=False)
    n_buy = int((orders["Side"] == "BUY").sum())
    n_sell = int((orders["Side"] == "SELL").sum())
    print(f"  Plan: {n_buy} BUY, {n_sell} SELL  (source: {meta['source']}, as of {meta['as_of']})")
    # Cap BUY spending at buying power (margin is disabled on the account).
    orders = apply_buying_power_guard(orders, buying_power)
    skipped_bp = orders[orders["Side"] == "SKIP (no buying power)"]
    if not skipped_bp.empty:
        print(f"  Buying-power guard: {len(skipped_bp)} BUY row(s) SKIPPED - insufficient buying power")
    if dry_run:
        return orders, meta, pd.DataFrame(columns=["Symbol", "Side", "Shares", "Order_ID", "Status"])
    cols = ["Symbol", "Side", "Shares", "Order_ID", "Status"]
    defer = _past_evening_cutoff()
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
    already = _todays_recorded_orders()
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
        # Extended hours are over: submit nothing now. Stage the planned orders (whole shares)
        # so the morning fill check sends the full quantities as regular-hours market orders.
        staged, skipped = [], []
        for r in orders_to_send[orders_to_send["Side"].isin(["SELL", "BUY"])].itertuples():
            try:
                qty = int(math.floor(float(r.Shares)))
            except (TypeError, ValueError):
                qty = 0
            if qty < 1:
                skipped.append((r.Symbol, r.Side, r.Shares, None, "SKIPPED: <1 whole share"))
                continue
            if r.Side == "SELL":
                qty, reason = _sell_qty_check(r.Symbol, qty, positions)
                if reason:
                    skipped.append((r.Symbol, r.Side, r.Shares, None, reason))
                    continue
                if qty < 1:
                    skipped.append((r.Symbol, r.Side, r.Shares, None, "SKIPPED: <1 whole share held"))
                    continue
            try:
                limit_price = round(float(r.Price), 2)
            except (TypeError, ValueError):
                limit_price = 0
            if not limit_price > 0:
                skipped.append((r.Symbol, r.Side, r.Shares, None, "SKIPPED: no price"))
                continue
            entry = {"symbol": r.Symbol, "side": r.Side, "qty": qty,
                     "limit_price": limit_price, "order_id": None}
            staged.append(entry)
            record_pending_order(entry, meta)  # incremental: a crash still leaves these staged
        results = pd.DataFrame(
            [(s["symbol"], s["side"], s["qty"], None, "STAGED for morning market (past 7 PM CT - not submitted)")
             for s in staged] + skipped + dup_rows, columns=cols)
    else:
        def _recorder(entry):
            record_pending_order(entry, meta)  # incremental: a crash still leaves these recorded
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
    status_counts = results["Status"].str.split(":").str[0].str.strip().value_counts()
    n_submitted = int(((results["Status"].str.contains("submitted", case=False, na=False)) |
                       (results["Status"].str.contains("STAGED", na=False))).sum())
    n_failed = int(results["Status"].str.startswith("FAILED").sum())
    n_skipped = int(results["Status"].str.startswith("SKIP").sum())
    print(f"\n  Submitted/Staged: {n_submitted}  |  Failed: {n_failed}  |  Skipped: {n_skipped}")
    if n_failed:
        failed_syms = ", ".join(results.loc[results["Status"].str.startswith("FAILED"), "Symbol"].astype(str))
        _notify("Trade had FAILED orders",
                f"{n_failed} order(s) failed this evening: {failed_syms}. "
                "Check Reports/paper_orders_log.csv - failed rows will be retried.")
    if n_submitted:
        staged = bool(results["Status"].str.contains("STAGED", na=False).any())
        _notify("Evening PAPER trades " + ("staged for the morning" if staged else "submitted"),
                _trade_summary(results))
    _print_header("EVENING TRADE COMPLETE")
    return orders, meta, results


def earnings_rule_note():
    """One line on the earnings rule for the printout ('' when it is off or the engine is not importable)."""
    try:
        from backtest_engine import WINNER
    except Exception:
        return ""
    n = WINNER.get("earnings_block_days")
    return (f"Earnings rule: stocks not held with earnings within {n} days are not bought (already applied to the targets; "
            "see Reports/strategy_changes.csv).") if n else ""


def main(argv=None):
    """Command line: print the order list (dry run) or submit to the Alpaca PAPER account."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--account-size", type=float, help="dollars to allocate (dry run default: 100000)")
    p.add_argument("--positions", help="CSV with Symbol,Shares of current holdings (dry run)")
    p.add_argument("--target", choices=["auto", "current", "provisional", "midweek", "hold"], default="auto",
                   help="'hold' keeps every position unchanged (quiet mid-week check: nothing is submitted)")
    p.add_argument("--fractional", action="store_true", help="fractional share quantities (dry run preview only)")
    p.add_argument("--min-value", type=float, default=1.0, help="skip trades smaller than this many dollars")
    p.add_argument("--out", help="also write the order list to this CSV")
    p.add_argument("--submit", action="store_true", help="send the orders (requires --paper)")
    p.add_argument("--paper", action="store_true", help="confirm the PAPER environment for --submit")
    p.add_argument("--fill-check", action="store_true",
                   help="morning fill check: complete unfilled extended-hours orders from the last auto_trade (PAPER only)")
    p.add_argument("--live", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    if a.live:
        sys.exit("Refusing: this script never trades a live account.")
    if a.submit and not a.paper:
        sys.exit("Refusing: --submit needs --paper as well (paper trading only).")
    if a.submit and a.fractional:
        sys.exit("Refusing: submit mode uses whole shares only.")
    if a.fill_check:
        if a.submit or a.fractional or a.account_size or a.positions or a.out or a.target != "auto":
            sys.exit("Refusing: --fill-check takes no other order options.")
        results = complete_unfilled_orders(dry_run=False)
        if results.empty:
            print("No pending extended-hours orders - nothing to do.")
        else:
            print(results.drop(columns=["Order_ID"]).to_string(index=False))
        return results

    positions = read_positions_csv(a.positions) if a.positions else {}
    account_size = a.account_size
    if a.submit:
        client = paper_trading_client()
        positions = {pos.symbol: float(pos.qty) for pos in client.get_all_positions()}
        if account_size is None:
            account_size = float(client.get_account().equity)
    account_size = account_size or 100_000.0

    orders, meta, targets = plan_orders(a.target, account_size, positions, min_value=a.min_value, fractional=a.fractional)
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
        print("\nDRY RUN - nothing was sent. Use --submit --paper to send these orders to the Alpaca PAPER account.")
        return orders
    print("\nSubmitting to Alpaca PAPER ...")
    print(submit_paper(orders, positions=positions).to_string(index=False))
    return orders


if __name__ == "__main__":
    main()
