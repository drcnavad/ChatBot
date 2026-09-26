"""Independent audit of the saved pipeline output (Reports/signal_analysis.csv + strategy_decisions.csv): RS_Score,
Strategy_Score, Strategy_Rank, the weekly top-10 selection, the Mon/Wed mid-week checks (swap + rank-exit sells) and the
earnings rule (no new buy when an earnings date is within WINNER['earnings_block_days'] calendar days) are re-derived from
raw bars and Reports/earnings_date.csv WITHOUT the engine's code (backtest_engine is only read for its settings).
Run: python tests/run_tests.py  (or PYTHONPATH=. python tests/test_rank_audit.py)"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import pandas as pd

import backtest_engine as be
import sector_mapping as sm

FAIL = []


def expect(ok, what):
    if not ok:
        FAIL.append(what)
        print("   !! FAIL:", what)
CAP = be.winner_max_per_sector()
MAX_RANK = be.WINNER.get("max_pick_rank")          # T20: picks only from ranks 1..20
SOFT = bool(be.WINNER.get("cap_soft"))             # T20: fill free slots ignoring the cap; mid-week swaps ignore the cap
MODE = be.WINNER.get("rs_benchmark", "etf")
EARN_DAYS = be.WINNER.get("earnings_block_days")   # E5: no new buy with earnings in the next N calendar days
_ed = pd.read_csv("Reports/earnings_date.csv")
EARN = {}
for _s, _d in zip(_ed["Symbol"].astype(str).str.strip().str.upper(), pd.to_datetime(_ed["Earnings Date"], errors="coerce")):
    if pd.notna(_d):
        EARN.setdefault(_s, []).append(_d.normalize())


def blocked(sym, D):
    """True when sym has an earnings date E with D < E <= D + EARN_DAYS (earnings rule on)."""
    D = pd.Timestamp(D).normalize()
    return bool(EARN_DAYS) and any(D < e <= D + pd.Timedelta(days=EARN_DAYS) for e in EARN.get(sym, []))


def held_before(D):
    """Strategy holdings at the session before D (what counts as 'already held' for the earnings rule)."""
    sess = sorted(sa.Date.unique())
    i = sess.index(D)
    return set() if i == 0 else set(sa[(sa.Date == sess[i - 1]) & (sa.Strategy_Weight > 0)].Symbol)
print("RS benchmark:", MODE)

sa = pd.read_csv("Reports/signal_analysis.csv", parse_dates=["Date"])
bars = pd.read_pickle("Reports/cache/bars_daily_long.pkl")
close = bars.pivot(index="Date", columns="Symbol", values="Close").sort_index()
tradable = list(sm.tradable_symbols)
etf_of = {s: dict((n, e) for e, n in sm.sector_etfs.items()).get(sm.symbol_sector.get(s)) for s in tradable}

# raw closes agree with the pipeline's closes?
m = sa.merge(bars[["Symbol", "Date", "Close"]], on=["Symbol", "Date"], suffixes=("", "_raw"))
print(f"Close check: {len(m)} rows, max |diff| {np.nanmax(np.abs(m.Close - m.Close_raw)):.4f}")
expect(np.nanmax(np.abs(m.Close - m.Close_raw)) < 0.01, "closes differ from the bar cache")

CACHE_MAX = close.index.max().date()
uncovered = sorted({pd.Timestamp(d).date() for d in sa.Date.unique()} - set(close.index.date))
if uncovered:
    print(f"Close check: {len(uncovered)} signal dates have no cached bars {uncovered} - rows on those dates cannot be audited from the cache")

SKIPPED = []


def auditable(D, label):
    """False (with a loud SKIP) when D is newer than the bar cache.

    The pipeline computes each row from live bars fetched at run time; this audit
    recomputes from the cached bars. A row newer than the cache is unauditable -
    it is skipped, not failed. Refresh the cache to audit it (Alpaca market data,
    no quota): python -c "import backtest_engine as be; be.backtest_inputs(refresh=True)"
    """
    if pd.Timestamp(D).date() > CACHE_MAX:
        msg = f"SKIP {label} {pd.Timestamp(D).date()}: bar cache ends {CACHE_MAX} (pipeline used live bars)"
        print("   !! " + msg)
        SKIPPED.append(msg)
        return False
    return True

main_start = pd.Timestamp("2023-09-20")  # main notebook fetch window (~1100 days before the run) for bar counting


def rs_at(D):
    """RS_Score at date D from data up to and including D only (no look-ahead by construction)."""
    c = close.loc[:D]
    parts_sv, parts_ss = [], []
    for w in (21, 63, 126):
        r = c.iloc[-1] / c.iloc[-1 - w] - 1
        stock = r[tradable]
        sector = pd.Series({s: r[etf_of[s]] for s in tradable})
        bench = sector.copy()
        if MODE in ("sector_median", "median_all"):   # independent re-implementation of the median benchmark
            for s in tradable:
                peers = stock[[x for x in tradable if x != s and sm.symbol_sector.get(x) == sm.symbol_sector.get(s)]].dropna()
                if len(peers) >= 3:
                    bench[s] = peers.median()
        sec_part = sector - r["SPY"]
        if MODE == "median_all":
            for s in tradable:
                members = stock[[x for x in tradable if sm.symbol_sector.get(x) == sm.symbol_sector.get(s)]].dropna()
                if len(members) >= 3:
                    sec_part[s] = members.median() - stock.median()
        parts_sv.append((stock - bench).rank(pct=True))
        parts_ss.append(sec_part.rank(pct=True))
    pct = 0.6 * sum(parts_sv) / 3 + 0.4 * sum(parts_ss) / 3
    return (pct * 2 - 1) * 100


def eligible_at(D):
    c = close.loc[main_start:D, tradable]
    return c.notna().sum() >= 200


def check_scores(D, title=""):
    """(b) RS, (a) score, (c) rank checks at D; returns (day frame, tradable rows)."""
    day = sa[sa.Date == D].set_index("Symbol")
    rs = rs_at(D)
    elig = eligible_at(D)
    t = day.loc[day.index.intersection(tradable)]
    mw = int(day.Midweek_Check.iloc[0]) if "Midweek_Check" in day.columns else 0
    print(f"\n=== {D.date()} {title}(Rebalance_Day={int(day.Rebalance_Day.iloc[0])}, Midweek_Check={mw}, Regime_On={int(day.Regime_On.iloc[0])}) ===")
    # (b) RS
    d_rs = (t.RS_Score - rs.reindex(t.index)).abs()
    print(f"(b) RS_Score recomputed: {d_rs.notna().sum()} symbols, max |diff| {d_rs.max():.4f}")
    expect(d_rs.max() < 0.01, f"RS_Score mismatch on {D.date()}")
    # (a) Strategy score
    exp = 0.5 * t.Technical_Score + 0.5 * t.RS_Score
    d_ss = (t.Strategy_Score - exp).abs()
    print(f"(a) Strategy_Score = 0.5*Tech + 0.5*RS: max |diff| {d_ss.max():.4f}; NaN scores (ineligible): {sorted(t.index[t.Strategy_Score.isna()])}")
    expect(d_ss.max() < 0.01, f"Strategy_Score mismatch on {D.date()}")
    print(f"    eligible (>=200 bars): {int(elig.sum())} of {len(tradable)}; Strategy_Score non-null: {int(t.Strategy_Score.notna().sum())}; "
          f"mismatch: {sorted(set(elig.index[elig]) ^ set(t.index[t.Strategy_Score.notna()]))}")
    # (c) rank
    o = t.dropna(subset=["Strategy_Score"]).assign(k1=lambda x: -x.Strategy_Score.round(4), k2=lambda x: -x.RS_Score, k3=lambda x: x.index)
    o = o.sort_values(["k1", "k2", "k3"])
    my_rank = pd.Series(range(1, len(o) + 1), index=o.index).reindex(t.index)
    d_rk = (t.Strategy_Rank - my_rank).abs()
    ties = t.Strategy_Score.dropna().duplicated().sum()
    expect(d_rk.max() == 0, f"Strategy_Rank mismatch on {D.date()}")
    print(f"(c) Strategy_Rank: max |diff| {d_rk.max():.0f}; ranks {int(t.Strategy_Rank.min())}..{int(t.Strategy_Rank.max())} over "
          f"{int(t.Strategy_Rank.notna().sum())} ranked; tied scores {ties}; QQQ rank: {day.loc['QQQ','Strategy_Rank'] if 'QQQ' in day.index else 'n/a'}")
    print("    top 5:", [(s, int(r), round(sc, 1)) for s, r, sc in t.sort_values('Strategy_Rank')[['Strategy_Rank', 'Strategy_Score']].head(5).itertuples()])
    return day, t


def audit(D, compare_col):
    day, t = check_scores(D)
    # (e) selection
    vol = close.loc[:D, tradable].pct_change(fill_method=None).iloc[-63:].std()
    cands = t[(t.Strategy_Score > 0) & vol.reindex(t.index).notna()].sort_values("Strategy_Rank")
    if MAX_RANK:
        cands = cands.iloc[:MAX_RANK]
    prior = held_before(D)
    earn_skip = [s for s in cands.index if blocked(s, D) and s not in prior]
    cands = cands.drop(earn_skip)
    picked, per, skipped = [], {}, []
    for s in cands.index:
        if len(picked) == 10:
            break
        sec = sm.symbol_sector.get(s, "Other")
        if per.get(sec, 0) >= CAP:
            skipped.append(s); continue
        picked.append(s); per[sec] = per.get(sec, 0) + 1
    relaxed = []
    if SOFT:                                      # T20: free slots from the unused top-20 names, sector cap ignored
        for s in cands.index:
            if len(picked) < 10 and s not in picked:
                picked.append(s); relaxed.append(s)
    inv = 1 / vol[picked]
    w = inv / inv.sum() * len(picked) / 10
    if int(day.Regime_On.iloc[0]) == 0:
        w *= 0.5
    actual = t[compare_col].fillna(0)
    held = sorted(actual.index[actual > 0])
    print(f"(e) walk-down picks: {picked}  (skipped for sector cap before slot 10: {skipped})"
          + (f"; picked with the cap relaxed (T20): {relaxed}" if SOFT else "")
          + (f"; not bought, earnings within {EARN_DAYS} days: {earn_skip}" if EARN_DAYS else ""))
    print(f"    {compare_col} > 0: {held} -> same set: {sorted(picked) == held}; max |weight diff| {(w.reindex(held).fillna(0) - actual[held]).abs().max() if held else 0:.4f}; "
          f"total invested {actual.sum():.3f} vs {w.sum():.3f}")
    expect(sorted(picked) == held and (w.reindex(held).fillna(0) - actual[held]).abs().max() < 1e-3, f"selection mismatch on {D.date()}")
    return picked


def audit_midweek(D, enter_top=3, exit_below=15, exit_all=None):
    """Independent re-derivation of a Mon/Wed mid-week check at D from the saved scores and the holdings of the previous session."""
    day, t = check_scores(D, "MID-WEEK CHECK ")
    sess = sorted(sa.Date.unique())
    P = sess[sess.index(D) - 1]
    before = sa[(sa.Date == P) & (sa.Strategy_Weight > 0)].set_index("Symbol").Strategy_Weight.to_dict()
    vol = close.loc[:D, tradable].pct_change(fill_method=None).iloc[-63:].std()
    q = t[(t.Strategy_Score > 0) & (vol.reindex(t.index) > 0)]
    q = q.assign(k1=-q.Strategy_Score.round(4), k2=-q.RS_Score, k3=q.index).sort_values(["k1", "k2", "k3"])
    rank = {s: i + 1 for i, s in enumerate(q.index)}
    held, swaps = dict(before), []
    sector = lambda s: sm.symbol_sector.get(s, "Other")
    while True:                                   # best entrant first; for it, the worst-ranked holding whose removal fits the cap
        # worst rank first; several holdings that no longer qualify (no rank) -> order of sector_mapping.tradable_symbols
        weak = sorted([s for s in held if rank.get(s, 10 ** 9) > exit_below], key=lambda s: (-rank.get(s, 10 ** 9), tradable.index(s)))
        ent = [s for s in list(q.index[:enter_top]) if s not in held and not blocked(s, D)]
        pair = next(((e, h) for e in ent for h in weak
                     if SOFT or sum(sector(k) == sector(e) for k in held if k != h) < CAP), None)
        if pair is None:
            break
        e, h = pair
        held[e] = held.pop(h)
        swaps.append((h, rank.get(h), e, rank[e], round(held[e], 4)))
    exits = []
    if exit_all:                                   # rank-exit: every remaining holding worse than exit_all (or unranked) -> cash
        for s in [s for s in held if rank.get(s, 10 ** 9) > exit_all]:
            exits.append((s, rank.get(s)))
            held.pop(s)
    actual = sa[(sa.Date == D) & (sa.Strategy_Weight > 0)].set_index("Symbol").Strategy_Weight.to_dict()
    same = set(held) == set(actual) and all(abs(held[s] - actual[s]) < 1e-3 for s in held)
    dec = pd.read_csv("Reports/strategy_decisions.csv", parse_dates=["Date"])
    dd = dec[dec.Date == D]
    logged = (sorted(dd.loc[dd.Status == "drop", "Symbol"]), sorted(dd.loc[dd.Status == "add", "Symbol"]))
    mine = (sorted([h for h, *_ in swaps] + [x for x, _ in exits]), sorted(e for _, _, e, _, _ in swaps))
    print(f"(m) held before (at {pd.Timestamp(P).date()} close): {sorted(before)}")
    print(f"    independent swaps (sell, sell rank, buy, buy rank, weight): {swaps if swaps else 'none'}"
          + (f"; rank-{exit_all} exits (sell, rank): {exits if exits else 'none'}" if exit_all else ""))
    print(f"    engine Strategy_Weight at {D.date()} matches: {same}; decision log drop/add {logged} vs mine {mine}: {logged == mine}")
    expect(same and logged == mine, f"mid-week check mismatch on {D.date()}")
    return swaps, exits, same and logged == mine


reb = sa.loc[sa.Rebalance_Day == 1, "Date"].drop_duplicates().sort_values()
off = sa.loc[(sa.Rebalance_Day == 1) & (sa.Regime_On == 0), "Date"].drop_duplicates()
dates = [(reb.iloc[-1], "Strategy_Weight"), (pd.Timestamp("2026-07-10"), "Strategy_Weight"), (pd.Timestamp("2026-06-18"), "Strategy_Weight"), (off.iloc[-1], "Strategy_Weight")]
latest = sa.Date.max()
if MAX_RANK or SOFT:                              # also audit the latest weekly decisions where the T20 rule changed the picks
    dec0 = pd.read_csv("Reports/strategy_decisions.csv", parse_dates=["Date"])
    t20_days = sorted(dec0.loc[dec0.Reason.astype(str).str.contains("sector cap relaxed|worse than 20 \\(picks", regex=True), "Date"].unique())
    print(f"T20: {len(t20_days)} weekly decisions in the window used the relaxed cap or dropped a holding ranked worse than {MAX_RANK}; "
          f"last: {[str(pd.Timestamp(d).date()) for d in t20_days[-3:]]}")
    dates += [(pd.Timestamp(d), "Strategy_Weight") for d in t20_days[-2:] if pd.Timestamp(d) not in {x for x, _ in dates}]
if EARN_DAYS:                                     # also audit the latest weekly decisions where the earnings rule skipped a stock
    dec_e = pd.read_csv("Reports/strategy_decisions.csv", parse_dates=["Date"])
    e_days = sorted(dec_e.loc[dec_e.Reason.astype(str).str.startswith("earnings in") & dec_e.Date.isin(set(reb)), "Date"].unique())
    print(f"Earnings rule: {len(e_days)} weekly decisions in the window skipped a stock; last: "
          f"{[str(pd.Timestamp(d).date()) for d in e_days[-3:]]}")
    expect(len(e_days) >= 1, "no weekly decision with an earnings skip to audit")
    dates += [(pd.Timestamp(d), "Strategy_Weight") for d in e_days[-2:] if pd.Timestamp(d) not in {x for x, _ in dates}]
n_reb_audited = 0
for D, col in dates:
    if auditable(D, "rebalance"):
        audit(D, col)
        n_reb_audited += 1
if auditable(latest, "provisional"):
    audit(latest, "Provisional_Weight")
if n_reb_audited == 0:
    FAIL.append("no rebalance row could be audited: the bar cache ends "
                f"{CACHE_MAX}, older than every checked rebalance date - refresh it: "
                "python -c \"import backtest_engine as be; be.backtest_inputs(refresh=True)\"")

# --- Mid-week checks: the last 3 historical swap days, the last 3 historical rank-exit days (if the exit rule is on) and this
#     week's Mon/Wed checks ---
if be.WINNER.get("midweek_swap"):
    mw = be.WINNER["midweek_swap"]
    exit_all = be.WINNER.get("midweek_exit_below")
    dec = pd.read_csv("Reports/strategy_decisions.csv", parse_dates=["Date"])
    swap_days = sorted(dec.loc[dec.Reason.astype(str).str.startswith("mid-week swap"), "Date"].unique())
    exit_days = sorted(dec.loc[dec.Reason.astype(str).str.startswith("mid-week exit"), "Date"].unique())
    checks = sorted(sa.loc[sa.Midweek_Check == 1, "Date"].unique())
    print(f"\nMid-week: {len(checks)} check sessions in the window, {len(swap_days)} with a swap, {len(exit_days)} with a "
          f"rank-{exit_all} exit; last swap days {[str(pd.Timestamp(d).date()) for d in swap_days[-3:]]}, last exit days "
          f"{[str(pd.Timestamp(d).date()) for d in exit_days[-3:]]}")
    if exit_all:
        expect(len(exit_days) >= 2, "fewer than 2 historical rank-exit days to audit")
    days = sorted({pd.Timestamp(d) for d in swap_days[-3:] + exit_days[-3:] + checks[-2:]})
    results, n_exit, mw_skipped = [], 0, 0
    for D in days:
        if not auditable(D, "mid-week"):
            mw_skipped += 1
            continue
        _, ex, ok = audit_midweek(D, mw["enter_top"], mw["exit_below"], exit_all)
        results.append(ok); n_exit += len(ex)
    print(f"\nMID-WEEK AUDIT: {sum(results)}/{len(results)} check days reproduced independently"
          + (f" (incl. {n_exit} rank-{exit_all} sells)" if exit_all else "")
          + (f"; {mw_skipped} skipped (newer than bar cache)" if mw_skipped else ""))
    if exit_all and not mw_skipped:
        expect(n_exit >= 2, "fewer than 2 rank-exit sells audited")

if SKIPPED:
    print("\nNOTE: some audit dates were skipped - they are newer than the bar cache "
          f"(cache ends {CACHE_MAX}); the pipeline computed those rows from live bars.")
    print("To audit them, refresh the cache (Alpaca market data, no quota):")
    print("  python -c \"import backtest_engine as be; be.backtest_inputs(refresh=True)\"")
print("\nRANK AUDIT OK" if not FAIL else f"\nRANK AUDIT FAILURES: {FAIL}")
sys.exit(1 if FAIL else 0)
