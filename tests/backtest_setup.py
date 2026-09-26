"""Shared, read-only set-up for the regression tests: the walk-forward data and ranking inputs of the 2026-09-24 mid-week
test (rebalance frequency, variant D) plus an INDEPENDENT re-implementation of the mid-week rules (not using the engine's
mid-week code) and of the earnings rule (E5). Reads Reports/cache/bars_daily_long.pkl and Reports/earnings_date.csv only
(no network, writes nothing)."""
import os
import warnings

import numpy as np
import pandas as pd

import backtest_engine as be
import sector_mapping as sm

warnings.filterwarnings("ignore")

U = list(sm.tradable_symbols)
WF, NEVER_END = "2022-04-01", "2024-09-16"          # walk-forward start; never-seen segment ends 2024-09-16
bars, _ = be.load_bars(refresh=False, cache_name="bars_daily_long.pkl")
tech = be.build_technical(bars, symbols=U)
C, O = be.wide(bars, "Close"), be.wide(bars, "Open")
idx = C.index
st = be.wide(tech, "Technical_Score").reindex(index=idx, columns=U)
el = be.bool_wide(tech, "eligible", idx, U)
rs, _ = be.relative_strength(C, U)
sc = 0.5 * st + 0.5 * rs
vol = C[U].pct_change(fill_method=None).rolling(63).std()
reg = be.regime_series(C, "QQQ")
RA = be.winner_rank_args(reg)
last = idx[-1]

# decision calendars: Friday = last session of each ISO week; Mon/Wed checks = first session on/after each Mon/Wed, not a Friday
weekly = be.weekly_rebalance_days(idx, live=True)
cal = pd.date_range(idx[0], last, freq="D")
mw_days = set(idx[p] for p in idx.searchsorted(cal[cal.dayofweek.isin([0, 2])]) if p < len(idx))
midweek = pd.Series(idx.isin(list(mw_days)) & ~weekly.to_numpy(bool), index=idx)
midweek.iloc[-1] = midweek.iloc[-1] and last.dayofweek in (0, 2, 4)

S_ = sc.to_numpy(float)
E_ = el.reindex_like(sc).astype("boolean").fillna(False).to_numpy(bool)
V_ = vol.reindex_like(sc).to_numpy(float)
T_ = rs.reindex_like(sc).to_numpy(float)
SECT = np.array([sm.symbol_sector.get(s, "Other") for s in U])
CAP = be.winner_max_per_sector()


def order_at(t):
    """Qualifying names at session t, best first (score at 6 decimals, then RS, then A-Z) - independent of the engine."""
    ok = E_[t] & ~np.isnan(S_[t]) & ~np.isnan(V_[t]) & (V_[t] > 0) & (S_[t] > 0)
    s, tb = np.round(S_[t], 6), np.round(T_[t], 6)
    return sorted(np.where(ok)[0], key=lambda k: (-s[k], -tb[k] if tb[k] == tb[k] else np.inf, U[k]))


PLAIN = dict(max_pick_rank=None, cap_soft=False)    # the tested C6 selection (walk all ranks, hard max-4 cap)
REG_ = reg.reindex(idx).astype("boolean").fillna(False).to_numpy(bool)


def targets(reb):
    """Engine weekly targets with the tested C6 selection (T20 off)."""
    return be.rank_targets(sc, el, vol, rebalance_days=reb, tiebreak_w=rs, **{**RA, **PLAIN})


def targets_t20(reb, max_rank=20, n=10):
    """INDEPENDENT weekly selection of the T20 rule (no engine selection code): walk ranks 1..max_rank with max CAP per sector,
    then fill free slots from the unused ranks 1..max_rank in rank order ignoring the cap; weights ~ 1/vol x (#picked / n),
    x 0.5 when QQQ is at/below its 200-day average on the rebalance day. Returns (targets, log of cap-relaxed picks)."""
    W = reb.reindex(idx).astype("boolean").fillna(False).to_numpy(bool)
    out = np.zeros((len(idx), len(U))); cur = np.zeros(len(U)); relaxed = []
    for t in range(len(idx)):
        if W[t]:
            cands = order_at(t)[:max_rank]
            picked, per = [], {}
            for j in cands:
                if len(picked) < n and per.get(SECT[j], 0) < CAP:
                    picked.append(j); per[SECT[j]] = per.get(SECT[j], 0) + 1
            for j in cands:
                if len(picked) < n and j not in picked:
                    picked.append(j); relaxed.append({"Date": idx[t], "Symbol": U[j], "Rank": cands.index(j) + 1})
            cur = np.zeros(len(U))
            if picked:
                inv = 1 / V_[t, picked]
                cur[picked] = inv / inv.sum() * (len(picked) / n)
                if not REG_[t]:
                    cur *= 0.5
        out[t] = cur
    return pd.DataFrame(out, index=idx, columns=U), pd.DataFrame(relaxed)


def block_matrix(days=5, path=None):
    """INDEPENDENT earnings block (no engine earnings code): True at (session t, stock j) when an earnings date E of j in
    Reports/earnings_date.csv satisfies t < E <= t + days (calendar days)."""
    e = pd.read_csv(path or os.path.join(be.REPORTS_DIR, "earnings_date.csv"))
    dates = {}
    for sym, d in zip(e["Symbol"].astype(str).str.strip().str.upper(), pd.to_datetime(e["Earnings Date"], errors="coerce")):
        if pd.notna(d):
            dates.setdefault(sym, set()).add(d.normalize())
    days_idx = idx.normalize()
    B = np.zeros((len(idx), len(U)), bool)
    for j, sym in enumerate(U):
        for d in dates.get(sym, ()):
            B[:, j] |= (days_idx < d) & (days_idx >= d - pd.Timedelta(days=days))
    return B


def select_t20(t, held, block, max_rank=20, n=10):
    """INDEPENDENT T20 selection at session t with the earnings rule: names not held (held[j] == 0) and blocked at t are
    left out of ranks 1..max_rank before the capped walk and the relaxed fill. Returns the weight vector."""
    cands = [j for j in order_at(t)[:max_rank] if not (block[t, j] and held[j] <= 0)]
    picked, per = [], {}
    for j in cands:
        if len(picked) < n and per.get(SECT[j], 0) < CAP:
            picked.append(j); per[SECT[j]] = per.get(SECT[j], 0) + 1
    for j in cands:
        if len(picked) < n and j not in picked:
            picked.append(j)
    cur = np.zeros(len(U))
    if picked:
        inv = 1 / V_[t, picked]
        cur[picked] = inv / inv.sum() * (len(picked) / n)
        if not REG_[t]:
            cur *= 0.5
    return cur


def buffered_midweek(N=3, M=15, exit_all=None, t20=False, block=None):
    """Friday full selection; at Mon/Wed checks: while a non-held name is in the top N and a held name ranks worse than M
    (or no longer qualifies), swap the worst-ranked held name for the best entrant (max CAP per sector); the entrant takes
    that weight. Then (exit_all) every remaining holding ranked worse than exit_all (or unranked) is sold -> cash until
    Friday. t20: Friday selection = targets_t20 and the mid-week swap ignores the sector cap. block (with t20; from
    block_matrix): earnings rule - the Friday selection uses select_t20 with the real holdings and blocked names are not
    mid-week entrants. Returns (targets, swap log, sell log)."""
    base = (targets_t20(weekly)[0] if t20 else targets(weekly)).to_numpy(float)
    blk = np.zeros((len(idx), len(U)), bool) if block is None else block
    cap = 10 ** 6 if t20 else CAP
    W, MW = weekly.to_numpy(bool), midweek.to_numpy(bool)
    out = np.zeros_like(base); cur = np.zeros(len(U)); log, sells = [], []
    for t in range(len(idx)):
        if W[t]:
            cur = base[t].copy() if block is None else select_t20(t, cur, block)
        elif MW[t] and cur.sum() > 0:
            order = order_at(t); rank = {j: r + 1 for r, j in enumerate(order)}
            while True:
                held = np.where(cur > 0)[0]
                weak = sorted([j for j in held if rank.get(j, 10 ** 6) > M], key=lambda j: -rank.get(j, 10 ** 6))
                entrants = [j for j in order[:N] if cur[j] == 0 and not blk[t, j]]
                pair = next(((e, h) for e in entrants for h in weak
                             if sum(SECT[k] == SECT[e] for k in held if k != h) < cap), None)
                if not weak or pair is None:
                    break
                e, h = pair
                cur[e], cur[h] = cur[h], 0.0
                log.append({"Date": idx[t], "In": U[e], "In_rank": rank[e], "Out": U[h], "Out_rank": rank.get(h, np.nan)})
            if exit_all is not None:
                for j in np.where(cur > 0)[0]:
                    if rank.get(j, 10 ** 6) > exit_all:
                        sells.append({"Date": idx[t], "Sell": U[j], "Rank": rank.get(j, np.nan), "Weight": cur[j]})
                        cur[j] = 0.0
        out[t] = cur
    return pd.DataFrame(out, index=idx, columns=U), pd.DataFrame(log), pd.DataFrame(sells)


def full(t):
    return t.reindex(index=idx, columns=C.columns).fillna(0.0)


def em(eq):
    """Metrics of a plain equity curve."""
    return be.metrics({"equity": eq, "exposure": eq * 0 + 1, "turnover": eq * 0, "trades": pd.DataFrame({"Return": []}),
                       "open_positions": pd.DataFrame()})
