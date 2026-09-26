"""Holdings alert for the live rules C6-U96-T20-MW30-E5 (presentation only - the strategy logic lives in backtest_engine).

Reads the pipeline outputs (Reports/signal_analysis.csv, Reports/strategy_midweek_check.csv) and, if present, the user's real
holdings in my_positions.csv (project root; same format as `paper_trade.py --positions`: Symbol,Shares - Shares optional,
an optional Weight column in % or as a fraction). Returns ONE plain line per action, e.g.
  "Swap at the Tue Sep 29 open: sell X and buy Y (rank 2), same dollar amount."      (Mon/Wed check day)
  "Sell TRGP (rank 45) at the Tue Sep 29 open, hold cash until Friday."              (Mon/Wed check day, mid-week exit)
  "With today's ranks the swap rule would sell ANET and buy RBRK (rank 1). Next check: ..."   (other days)
  "No swap with today's ranks. Next check: Mon Sep 28."
  "Full rebalance at the Mon Sep 28 open: sell ...; buy ..."                         (Friday / week's last session)
Tickers link to the app (http://localhost:8501/?symbol=XXX). The swap rule is backtest_engine.midweek_swap_pairs (the exact
function the strategy uses: top 3 in, below rank 15 out, weight inheritance; max 4 per sector unless WINNER['cap_soft'] (T20,
live: a top-3 stock always qualifies)), followed by the mid-week exit
backtest_engine.midweek_exit_sells (WINNER["midweek_exit_below"] = 30: any holding worse than rank 30 is sold, cash until the
Friday rebalance; None = off). Earnings rule (WINNER["earnings_block_days"] = 5): a stock that is not held is not bought
when its next earnings date (Reports/earnings_date.csv) is within 5 calendar days after the decision date; the alert names
such skipped stocks ("MU rank 2 not bought: earnings in 5 days (Wed Sep 30)"). Held stocks are never sold for earnings.
Stocks outside the 96-stock universe count as not ranked (below rank 15 and worse than 30). No network calls.

    python holdings_alert.py                 # print the alert (uses my_positions.csv if it exists)
    python holdings_alert.py --positions f.csv | --strategy
"""
import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import backtest_engine as be
import sector_mapping as sm

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
POSITIONS_FILE = os.path.join(ROOT, "my_positions.csv")
SIGNAL_CSV = os.path.join(REPORTS, "signal_analysis.csv")
MIDWEEK_CSV = os.path.join(REPORTS, "strategy_midweek_check.csv")
ET = ZoneInfo("America/New_York")
LEVEL_ICON = {"red": "🔴", "green": "🟢", "blue": "🔵"}
APP_URL = "http://localhost:8501/?symbol="   # the app's ticker view (symbol query parameter)


# ----------------------------------------------------------------------------- small helpers: dates, live rule settings, positions file
def _day(d):
    """Date as 'Mon Sep 28'."""
    d = pd.Timestamp(d)
    return f"{d:%a %b} {d.day}"


def rule_params():
    """(enter_top, exit_below, sector cap for swaps, mid-week on). T20 (WINNER['cap_soft']): swaps ignore the sector cap."""
    mw = be.WINNER.get("midweek_swap") or {}
    cap = 10 ** 6 if be.WINNER.get("cap_soft") else be.winner_max_per_sector()
    return mw.get("enter_top", 3), mw.get("exit_below", 15), cap, bool(mw)


def exit_below():
    """Mid-week exit threshold (WINNER['midweek_exit_below'], e.g. 30) or None (off; also off when the mid-week swap is off)."""
    return be.WINNER.get("midweek_exit_below") if be.WINNER.get("midweek_swap") else None


def cash_until(d):
    """Weekday of the next full rebalance after session d ('Friday' normally, 'Thursday' before a Friday holiday)."""
    return f"{be.next_decision(d, days=())[0]:%A}"


def load_earnings_dates():
    """Earnings dates for the earnings rule, or None when the rule is off or Reports/earnings_date.csv is missing."""
    if not be.WINNER.get("earnings_block_days"):
        return None
    try:
        return be.load_earnings()
    except (OSError, ValueError, KeyError):
        return None


def read_positions(path):
    """DataFrame[Symbol, Shares, Weight] from a positions CSV (Symbol required; Shares/Qty and Weight optional)."""
    pos = pd.read_csv(path)
    pos.columns = [str(c).strip().lower() for c in pos.columns]
    if "symbol" not in pos.columns:
        raise ValueError(f"{os.path.basename(path)} needs a 'Symbol' column")
    out = pd.DataFrame({"Symbol": pos["symbol"].astype(str).str.upper().str.strip()})
    qty = "shares" if "shares" in pos.columns else ("qty" if "qty" in pos.columns else None)
    out["Shares"] = pd.to_numeric(pos[qty], errors="coerce") if qty else np.nan
    w = pd.to_numeric(pos["weight"], errors="coerce") if "weight" in pos.columns else pd.Series(np.nan, index=pos.index)
    out["Weight"] = w / 100 if w.dropna().gt(1).any() else w          # 12.5 (percent) or 0.125 (fraction)
    out = out[(out["Symbol"] != "") & (out["Symbol"] != "NAN")]
    out = out[~(out["Shares"].fillna(1) == 0)]                         # zero shares = not held
    return out.drop_duplicates("Symbol", keep="last").reset_index(drop=True)


def last_completed_session(now=None):
    """Latest NYSE session whose daily bar is final (after 4:30 PM ET, same buffer as the pipeline)."""
    now = pd.Timestamp(datetime.now(tz=ET)) if now is None else pd.Timestamp(now)
    now = now.tz_localize(ET) if now.tzinfo is None else now.tz_convert(ET)
    today = now.tz_localize(None).normalize()
    sess = pd.date_range(today - pd.Timedelta(days=12), today, freq=be.NYSE_SESSION)
    if len(sess) and sess[-1] == today and now.hour * 60 + now.minute < 16 * 60 + 30:
        sess = sess[:-1]
    return sess[-1]


# ----------------------------------------------------------------------------- one day's ranks (same symbol order as the engine)
class _Day:
    """Ranks / scores of one session: qualifying order (score > 0, ranked) as the strategy walks it."""

    def __init__(self, day_df, symbols, earnings=None):
        """Ranks and scores of one day, indexed like the engine's symbol columns. earnings = load_earnings_dates()."""
        d = day_df.drop_duplicates("Symbol").set_index("Symbol")
        self.date = pd.Timestamp(day_df["Date"].iloc[0])
        self.symbols = list(symbols)
        self.col = {s: i for i, s in enumerate(self.symbols)}
        rk = d["Strategy_Rank"].reindex(self.symbols)
        sc = d["Strategy_Score"].reindex(self.symbols)
        q = [(rk[s], s) for s in self.symbols if pd.notna(rk[s]) and pd.notna(sc[s]) and sc[s] > 0]
        self.order = [self.col[s] for _, s in sorted(q)]
        self.rank = {j: r + 1 for r, j in enumerate(self.order)}
        self.score = sc
        self.close = d["Close"].reindex(self.symbols) if "Close" in d.columns else pd.Series(np.nan, index=self.symbols)
        self.weight = d["Strategy_Weight"].reindex(self.symbols).fillna(0.0)
        self.sectors = np.array([sm.symbol_sector.get(s, "Other") for s in self.symbols])
        self.earn_days = {}                                            # column -> days to earnings (earnings rule)
        if earnings is not None:
            ahead = be.earnings_days_ahead([self.date], self.symbols, earnings, be.WINNER["earnings_block_days"]).iloc[0]
            self.earn_days = {self.col[s]: v for s, v in ahead.items() if pd.notna(v)}

    def earnings_txt(self, sym):
        """'MU rank 2 not bought: earnings in 5 days (Wed Sep 30)'."""
        return f"{sym} {self.rank_txt(sym)} not bought: {be.earnings_note(self.earn_days[self.col[sym]], self.date)}"

    def rank_of(self, sym):
        """Rank of a symbol on this day (None = not ranked)."""
        return self.rank.get(self.col.get(sym), None)

    def why_unranked(self, sym):
        """Plain reason a symbol has no rank."""
        if sym not in sm.tradable_symbols:
            return f"not ranked: not in the {len(sm.tradable_symbols)}-stock universe"
        s = self.score.get(sym)
        return "not ranked: no score yet" if pd.isna(s) else f"not ranked: score {s:.1f} is not above 0"

    def rank_txt(self, sym):
        """'rank N' or the reason it is not ranked."""
        r = self.rank_of(sym)
        return f"rank {r}" if r else self.why_unranked(sym)


# ----------------------------------------------------------------------------- the rules applied to a set of holdings
def evaluate(day, held):
    """Run the strategy's mid-week rule on `held` {symbol: weight} with this day's ranks: the swap step, then (if
    WINNER['midweek_exit_below']) the exit step. Returns (swaps, weak, entrants, exits); exits = [symbol] sold to cash,
    worst rank first. Top-3 names blocked by the earnings rule are not entrants (see blocked_entrants)."""
    top, below, cap, _ = rule_params()
    cur = np.zeros(len(day.symbols))
    for s, w in held.items():
        cur[day.col[s]] = w if w > 0 else 1e-9
    before = cur.copy()
    pairs = be.midweek_swap_pairs(cur, day.order, day.rank, day.sectors, top, below, cap, skip=set(day.earn_days))
    swaps = [(day.symbols[h], day.symbols[e], before[h]) for e, h, _ in pairs]
    exits = [day.symbols[j] for j, _ in be.midweek_exit_sells(cur, day.rank, exit_below())]
    exits = sorted(exits, key=lambda s: -(day.rank_of(s) or 1e9))
    weak = [day.symbols[j] for j in np.where(before > 0)[0] if day.rank.get(j, 1e9) > below]
    weak = sorted(weak, key=lambda s: -(day.rank_of(s) or 1e9))
    entrants = [day.symbols[j] for j in day.order[:top] if before[j] == 0 and j not in day.earn_days]
    return swaps, weak, entrants, exits


def blocked_entrants(day, held):
    """Top-3 names not held that the earnings rule stops from being bought, as ['MU rank 2 not bought: earnings in ...']."""
    top = rule_params()[0]
    return [day.earnings_txt(day.symbols[j]) for j in day.order[:top]
            if j in day.earn_days and held.get(day.symbols[j], 0) <= 0]


def blocked_picks(day, new, old):
    """Friday: names ranked above the last pick (within ranks 1..max_pick_rank) that were passed over for earnings."""
    worst = max([day.rank_of(s) or 0 for s in new], default=0)
    limit = be.WINNER.get("max_pick_rank") or worst
    if len(new) < be.WINNER.get("n", 10):
        worst = limit
    return [day.earnings_txt(day.symbols[j]) for j in day.order
            if j in day.earn_days and day.symbols[j] not in old and day.symbols[j] not in new
            and day.rank[j] <= min(worst, limit)]


def earnings_suffix(items):
    """' (MU rank 2 not bought: earnings in 5 days (Wed Sep 30))' or ''."""
    return f" ({'; '.join(items)})" if items else ""


def T(sym):
    """Ticker segment (rendered as a link to the app's ticker view)."""
    return ("ticker", sym)


# ----------------------------------------------------------------------------- build the alert (list of plain-text segments; tickers become links in the app)
def build_alert(sig=None, positions_path=None, use_positions=True, now=None, midweek_csv=MIDWEEK_CSV):
    """One plain line per action (list of segments: str or ("ticker", SYM)) + level / source / data date.

    level: red = a swap to do (check day, or a missed check per the positions file), blue = full rebalance due,
    green = nothing to do (includes 'with today's ranks the rule would ...' on non-check days, which is informational)."""
    top, below, cap, mw_on = rule_params()
    if sig is None:
        cols = ["Date", "Symbol", "Close", "Strategy_Score", "Strategy_Rank", "Strategy_Weight", "Rebalance_Day"]
        sig = pd.read_csv(SIGNAL_CSV, usecols=cols + (["Midweek_Check"] if _has_col(SIGNAL_CSV, "Midweek_Check") else []),
                          parse_dates=["Date"])
    sig = sig.copy()
    sig["Date"] = pd.to_datetime(sig["Date"])
    if "Midweek_Check" not in sig.columns:
        sig["Midweek_Check"] = 0
    sessions = sorted(sig["Date"].unique())
    D = pd.Timestamp(sessions[-1])
    P = pd.Timestamp(sessions[-2]) if len(sessions) > 1 else D
    positions = None
    path = positions_path or POSITIONS_FILE
    if use_positions and path and os.path.exists(path):
        positions = read_positions(path)
    universe = list(sm.tradable_symbols)
    extra = [s for s in (positions["Symbol"] if positions is not None else []) if s not in universe]
    symbols = universe + extra
    earnings = load_earnings_dates()
    dday = _Day(sig[sig["Date"] == D], symbols, earnings)
    first = sig[sig["Date"] == D].iloc[0]
    is_reb, is_chk = int(first["Rebalance_Day"]) == 1, int(first["Midweek_Check"]) == 1 and mw_on
    fill = be.next_sessions(D, 1)[0]
    nxt_d, nxt_kind, _nxt_fill = be.next_decision(D)
    next_txt = f"Next check: {_day(nxt_d)}" + (" (full rebalance)." if nxt_kind == "full rebalance" else ".")

    def holdings_at(d):
        """{symbol: weight} the strategy held on day d."""
        w = sig[(sig["Date"] == d) & (sig["Strategy_Weight"] > 0)].drop_duplicates("Symbol")
        return dict(zip(w["Symbol"], w["Strategy_Weight"]))

    if positions is not None:
        source = "your positions file"
        pw = positions.set_index("Symbol")["Weight"]
        held_now = {s: (float(pw[s]) if pd.notna(pw[s]) else 1.0 / len(positions)) for s in positions["Symbol"]}
    else:
        source = "strategy holdings"
        held_now = holdings_at(D)
    stale = None
    last_done = last_completed_session(now)
    if D < last_done:
        stale = f"Data is from {_day(D)}; run python run_all.py."
    out = {"level": "green", "source": source, "data_date": D, "lines": [], "stale": stale,
           "uses_positions": positions is not None}

    def pair_line(prefix, pairs, day, suffix=""):
        """Alert segment for swap pairs: 'sell X and buy Y (rank r)'."""
        seg = [prefix]
        for i, (sell, buy, _w) in enumerate(pairs):
            seg += (["; " if i else "", "sell ", T(sell), " and buy ", T(buy), f" (rank {day.rank_of(buy)})"])
        return seg + [suffix]

    def sells_seg(syms, day):
        """Alert segment listing stocks to sell with their ranks."""
        seg = []
        for i, sym in enumerate(syms):
            r = day.rank_of(sym)
            seg += ([", "] if i else []) + [T(sym), f" (rank {r})" if r else " (not ranked)"]
        return seg

    ex_rule = exit_below()
    none_txt = "No swap or exit" if ex_rule else "No swap"

    if is_reb:                                              # week's last session: full rebalance at the next open
        new = holdings_at(D)
        old = holdings_at(P) if positions is None else held_now
        sells = sorted((s for s in old if s not in new), key=lambda s: -(dday.rank_of(s) or 999))
        buys = sorted((s for s in new if s not in old), key=lambda s: dday.rank_of(s) or 999)
        seg = [f"Full rebalance at the {_day(fill)} open: sell "]
        seg += _join([T(s) for s in sells]) if sells else ["nothing"]
        seg += ["; buy "] + (_join([T(s) for s in buys]) if buys else ["nothing"]) + ["."]
        skipped = blocked_picks(dday, new, old)
        if skipped:
            seg += [f" Not bought (earnings within {be.WINNER['earnings_block_days']} days): " + "; ".join(skipped) + "."]
        out.update(level="blue", lines=[seg])
        return out
    if not mw_on:
        out["lines"] = [["No mid-week swaps (weekly rules). " + next_txt]]
        return out
    if is_chk:                                              # Mon/Wed check: the actual decision at this close
        held_chk = holdings_at(P) if positions is None else held_now
        swaps, _, _, exits = evaluate(dday, held_chk)
        lines = []
        if swaps:
            lines.append(pair_line(f"Swap at the {_day(fill)} open: ", swaps, dday, ", same dollar amount."))
        if exits:
            lines.append(["Sell "] + sells_seg(exits, dday) + [f" at the {_day(fill)} open, hold cash until {cash_until(D)}."])
        if lines:
            out.update(level="red", lines=lines)
        else:
            out["lines"] = [[f"{none_txt} at the {_day(D)} check{earnings_suffix(blocked_entrants(dday, held_chk))}. " + next_txt]]
        return out
    if positions is not None:                               # missed swap at this week's check?
        last_reb = max([pd.Timestamp(d) for d in sig.loc[sig["Rebalance_Day"] == 1, "Date"].unique()], default=None)
        checks = [pd.Timestamp(d) for d in sig.loc[sig["Midweek_Check"] == 1, "Date"].unique()
                  if last_reb is None or pd.Timestamp(d) > last_reb]
        if checks:
            C = max(checks)
            cday = _Day(sig[sig["Date"] == C], symbols, earnings)
            missed, _, _, missed_exits = evaluate(cday, held_now)
            lines = []
            if missed:
                lines.append(pair_line(f"The {_day(C)} check called for: ", missed, cday,
                                       f" (not done per your positions file; next open {_day(fill)})."))
            if missed_exits:
                lines.append([f"The {_day(C)} check called for selling "] + sells_seg(missed_exits, cday)
                             + [f" and holding cash until {cash_until(C)} (still held per your positions file; next open "
                                f"{_day(fill)})."])
            if lines:
                out.update(level="red", lines=lines)
                return out
    swaps, _, _, exits = evaluate(dday, held_now)
    lines = []
    if swaps:
        lines.append(pair_line("With today's ranks the swap rule would ", swaps, dday, "." + ("" if exits else " " + next_txt)))
    if exits:
        lines.append([f"With today's ranks the rank-{ex_rule} exit would sell "] + sells_seg(exits, dday)
                     + [" (cash until the rebalance). " + next_txt])
    out["lines"] = lines or [[f"{none_txt} with today's ranks{earnings_suffix(blocked_entrants(dday, held_now))}. " + next_txt]]
    return out


# ----------------------------------------------------------------------------- text output
def _join(items):
    """List of segments separated by ', '."""
    seg = []
    for i, x in enumerate(items):
        seg += ([", "] if i else []) + [x]
    return seg


def _has_col(path, col):
    """True when the CSV has this column (reads the header only)."""
    return col in pd.read_csv(path, nrows=0).columns


def line_text(seg, urls=True):
    """Plain text of one line; tickers followed by their app link (pipeline printout)."""
    return "".join(x if isinstance(x, str) else (f"{x[1]} ({APP_URL}{x[1]})" if urls else x[1]) for x in seg)


def alert_text(a, urls=True):
    """Pipeline printout: the same one line per action (+ a stale-data note if any)."""
    lines = [f"{LEVEL_ICON[a['level']]} {line_text(seg, urls)}" for seg in a["lines"]]
    if a["stale"]:
        lines.append("⚠️ " + a["stale"])
    return lines


# ----------------------------------------------------------------------------- command line
def main(argv=None):
    """Command line: print the alert."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positions", help="positions CSV (default: my_positions.csv in the project root, if it exists)")
    p.add_argument("--strategy", action="store_true", help="ignore the positions file; use the strategy's holdings")
    a = p.parse_args(argv)
    for line in alert_text(build_alert(positions_path=a.positions, use_positions=not a.strategy)):
        print(line)


if __name__ == "__main__":
    main()
