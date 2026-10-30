"""Paper-only forward-test lines the weight-based simulator in forward_test.update_strategies cannot express (added Oct 15,
2026, Chirag): two independent halves (50/50, each with its own cash, the second half never buys what the first holds),
and swap-only halves. Ported from the Oct 15
backtests (/tmp/combo bt.py + swap.py, /tmp/lvbt bt4.py) so the forward lines match what was backtested.

Same conventions as forward_test: decide and trade at the day's close, be.COST (0.1%) per side, a holding within 1 point
of its target is left alone, 99% invested, equal weight (1/n of its half), no margin, do_not_buy never bought, every
score/eligibility/regime from forward_test.strategy_inputs. Deterministic: a line is replayed from its start close on
every run from the same saved data; forward_test appends only the days after its last saved one (saved days are never
redone). Display only: none of these lines is ever traded.

A line = legs [(key, mode, calendar, share)]:
  key      ("<key>+mb" = that leg + the live manual buy blocks; "<key>@5" = that leg with its top 5 instead of 10; the sector cap scales: floor(0.4 x 5) = 2 per sector)
           "lowvol" (low volatility in uptrend, equal weight) / "sr<n>" (sector rotation: the C6 score in the 3 sectors
           whose ETF has the best n-day return, equal weights) / "vam" (volatility-adjusted momentum, equal weight)
           / "calm" (the Oct 15 rules' calmer half, "Relative strength among calm stocks": relative strength, 63-day volatility at or below the median, no buys
           5 days before earnings)
  mode     "full" = the leg's top 10 on each calendar day (a full rebalance) / "swap<k>" = buy the top 10 once, then on
           each calendar day sell a holding ranked worse than k (or held by the first leg) for the best stock not held
  calendar "month" (last session of the month) / "week" (last session of the week) / "weekstart" (first session of the week) / "2week" (every other week end from
           the start) / "day"; the start close always buys.
With no_overlap the first leg has priority: the second never buys a name the first holds, and when the first buys a name
the second holds, the second rebalances (full) or swaps it out (swap) that same day."""
import numpy as np
import pandas as pd

import backtest_engine as be
import sector_mapping as sm

_CACHE = {}

def _cfg(name):
    import forward_test as ft
    return next(c for c in ft.STRATEGIES if c["name"] == name)

class Ctx:
    """The frames every leg ranks on (one per strategy_inputs result)."""
    def __init__(self, inp):
        import forward_test as ft
        self.inp, self.ft = inp, ft
        self.c = inp["close"]
        self.idx, self.cols = self.c.index, list(self.c.columns)
        self.elig, self.vol, self.rs = inp["eligible"], inp["vol"], inp["scores"]["rs"]
        self.cal = {"month": inp["monthly"].reindex(self.idx).fillna(False).astype(bool),
                    "week": pd.Series([be.next_sessions(d, 1)[0].isocalendar()[:2] != d.isocalendar()[:2]
                                       for d in self.idx], index=self.idx),
                    "weekstart": pd.Series([i == 0 or self.idx[i - 1].isocalendar()[:2] != d.isocalendar()[:2]
                                            for i, d in enumerate(self.idx)], index=self.idx),
                    "day": pd.Series(True, index=self.idx)}
        self.never = [s for s in sm.do_not_buy if s in self.cols]
        self.e5 = be.earnings_days_ahead(self.idx, self.cols, inp["earnings"], ft.EARN_DAYS).reindex(
            index=self.idx, columns=self.cols)
        self.sect = {s: sm.symbol_sector.get(s, "Other") for s in self.cols}
        self._legs, self._rank = {}, {}
        self.mblock = self.manual_block()

    def calendar(self, name, start):
        if name == "2week":
            w = self.cal["week"]
            f = w[w & (w.index >= start)].index[::2]
            return pd.Series(self.idx.isin(f), index=self.idx)
        return self.cal[name]

    def leg(self, key):
        """(score, tiebreak, rank_targets args, buy block) of one leg key."""
        if key in self._legs:
            return self._legs[key]
        ft, inp = self.ft, self.inp
        if key.endswith("+mb"):                            # + the live manual buy blocks (manual_trade, Reports/manual_block.json)
            score, tb, a, block = self.leg(key[:-3])
            self._legs[key] = (score, tb, a, block.where(~self.mblock, 999.0))
            return self._legs[key]
        base, _, n_top = key.partition("@")                # "<key>@5" = the same leg holding its top 5 (default 10)
        if base != key:
            score, tb, a, block = self.leg(base)
            self._legs[key] = (score, tb, {**a, "n": int(n_top)}, block)
            return self._legs[key]
        if key == "lowvol":
            cfg, score = _cfg("Low volatility in uptrend"), inp["scores"]["low_vol"]
        elif key == "vam":
            cfg, score = _cfg("Momentum per unit of risk"), inp["scores"]["mom_risk"]
        elif key == "calm":
            cfg, score = _cfg("Relative strength among calm stocks"), inp["scores"]["rs_calm"]
        elif key.startswith("sr"):
            cfg, n = _cfg("Best stocks in hottest sectors"), int(key[2:])
            b = inp["bench"]
            etf_of = {s: sm.sector_etf_for(s) for s in self.cols}
            r = b / b.shift(n) - 1
            e = pd.DataFrame({s: r[etf_of[s]] if etf_of[s] in r else np.nan for s in self.cols}, index=self.idx)
            score = inp["scores"]["live"].where(e.rank(axis=1, method="dense", ascending=False).le(3))
        else:
            raise KeyError(key)
        a = be.winner_rank_args(inp["regime"])
        a.update({**ft.PIN, **(cfg.get("select") or {}), "sleeve": ()})
        a["n"] = 10
        block = (self.e5.copy() if cfg.get("earnings", ft.EARN_DAYS) else
                 pd.DataFrame(np.nan, index=self.idx, columns=self.cols))
        block[self.never] = 999.0
        self._legs[key] = (score, inp["scores"].get(cfg.get("tiebreak", "rs")), a, block)
        return self._legs[key]

    def manual_block(self):
        """Date x Symbol True while a live manual buy block (manual_trade.load_blocks) is active: from the session it was
        added (the live manual sale that day) until a close at or below block_until_price."""
        import manual_trade as mt
        m = pd.DataFrame(False, index=self.idx, columns=self.cols)
        for sym, e in mt.load_blocks().items():
            if sym not in m:
                continue
            first = self.idx[self.idx >= pd.Timestamp(e["added"])]
            if not len(first):
                continue
            on = pd.Series(self.idx >= first[0], index=self.idx)
            if e.get("block_until_price") is not None:
                on &= ~(on & self.c[sym].le(e["block_until_price"])).cummax()
            m[sym] = on.to_numpy()
        return m

    def pick(self, key, t, held_w=None, exclude=()):
        score, tb, args, block = self.leg(key)
        day = self.idx[[t]]
        el = self.elig
        if exclude:
            el = el.copy()
            el.loc[day, [x for x in exclude if x in el.columns]] = False
        w = be.rank_targets(score.iloc[[t]], el, self.vol, rebalance_days=pd.Series(True, index=day), tiebreak_w=tb,
                            buy_block=block.iloc[[t]], start_holdings=held_w, **args).iloc[0]
        return be.live_weights(w).reindex(self.cols).fillna(0.0)

    def rank(self, key):
        """Date x Symbol rank of the leg's candidates (1 = best; NaN = not a candidate)."""
        if key not in self._rank:
            score, tb, args, _ = self.leg(key)
            ok = self.elig & score.gt(args.get("min_score", 0.0))
            self._rank[key] = be.deterministic_rank(score, ok, tb).reindex(columns=self.cols).to_numpy(float)
        return self._rank[key]

class Leg:
    def __init__(self, x, key, mode, cal, cash):
        self.x, self.key, self.mode, self.cal, self.cash = x, key, mode, cal, cash
        self.sh = pd.Series(0.0, index=x.cols)

    def value(self, px):
        return self.cash + float((self.sh * px.fillna(0)).sum())

    def held(self):
        return set(self.sh[self.sh > 0].index)

    def trade(self, sym, dq, p):
        if abs(dq) * p < 1e-12:
            return
        v = dq * p
        self.sh[sym] += dq
        self.cash -= v + abs(v) * be.COST
        if self.sh[sym] < 1e-12:
            self.sh[sym] = 0.0

    # ---- full rebalance (bt.py Sleeve.rebalance)
    def rebalance(self, t, px, exclude=()):
        val = self.value(px)
        held_w = (self.sh * px.fillna(0) / val).to_numpy(float) if val > 0 else None
        w = self.x.pick(self.key, t, held_w, exclude)
        for s in sorted(self.held()):
            if not w[s] > 0:
                self.trade(s, -self.sh[s], px[s])
        hw = self.sh * px.fillna(0) / val
        move = [s for s in w[w > 0].index if not (self.sh[s] > 0 and abs(hw[s] - w[s]) <= self.x.ft.BAND)]
        for s in move:
            q = w[s] * val / px[s] - self.sh[s]
            if q < 0:
                self.trade(s, q, px[s])
        buys = [(s, w[s] * val / px[s] - self.sh[s]) for s in move if w[s] * val / px[s] - self.sh[s] > 0]
        need = sum(q * px[s] * (1 + be.COST) for s, q in buys)
        f = min(1.0, max(self.cash, 0) / need) if need > 0 else 1.0
        for s, q in buys:
            self.trade(s, q * f, px[s])

    # ---- swap-only (swap.py Swap)
    def _best(self, t, exclude):
        R, NB = self.x.rank(self.key), self.x.leg(self.key)[3].to_numpy()
        cap = int(np.floor(self.x.leg(self.key)[2]["sector_cap"] * 10 + 1e-9))
        cnt = {}
        for s in self.held():
            cnt[self.x.sect[s]] = cnt.get(self.x.sect[s], 0) + 1
        for j in np.argsort(np.nan_to_num(R[t], nan=1e9), kind="stable"):
            if np.isnan(R[t, j]):
                return None
            s = self.x.cols[j]
            if self.sh[s] > 0 or not np.isnan(NB[t, j]) or s in exclude or cnt.get(self.x.sect[s], 0) >= cap:
                continue
            return s
        return None

    def swap_start(self, t, px, exclude):
        dollars = self.cash * 0.99 / 10
        for _ in range(10):
            s = self._best(t, exclude)
            if s is None:
                break
            self.trade(s, dollars / (1 + be.COST) / px[s], px[s])

    def swap_check(self, t, px, exclude):
        R, keep = self.x.rank(self.key), int(self.mode[4:])
        j_of = {s: self.x.cols.index(s) for s in self.held()}
        out = [s for s in self.held() if not (R[t, j_of[s]] <= keep) or s in exclude]
        for s in sorted(out, key=lambda s: (-np.nan_to_num(R[t, j_of[s]], nan=1e9), s)):
            d = self.sh[s] * px[s]
            self.trade(s, -self.sh[s], px[s])
            b = self._best(t, exclude)
            if b is not None:
                self.trade(b, d * (1 - be.COST) / (1 + be.COST) / px[b], px[b])
        if self.cash > 0.02 * self.value(px) and len(self.held()) < 10:
            b = self._best(t, exclude)
            if b is not None:
                self.trade(b, min(self.cash / (1 + be.COST), self.value(px) / 10) / px[b], px[b])

    def step(self, i, t, px, exclude=()):
        """Today's decision of this leg (i = 0: the start close). True if it traded on a calendar/start day."""
        if self.mode.startswith("swap"):
            if i == 0:
                self.swap_start(t, px, exclude)
            elif bool(self.cal.iloc[t]):
                self.swap_check(t, px, exclude)
            return
        if i == 0 or bool(self.cal.iloc[t]):
            self.rebalance(t, px, exclude)

def simulate(inp, legs, start, no_overlap=True, through=None):
    """Replay a line from the `start` close: {"days", "legs": [[(value, cash, {sym: (shares, price)}) per day] per leg]};
    each leg starts with its share of 1.0 in cash (before its buys)."""
    key = (id(inp), tuple(legs), str(start), no_overlap, str(through))
    if key in _CACHE:
        return _CACHE[key]
    x = inp.setdefault("_combo_ctx", None) or Ctx(inp)
    inp["_combo_ctx"] = x
    start = pd.Timestamp(start)
    days = x.idx[(x.idx >= start) & (x.idx <= pd.Timestamp(through or x.idx[-1]))]
    out = {"days": days, "legs": [[] for _ in legs]}
    if not len(days):
        return out
    t0 = x.idx.get_loc(days[0])
    L = [Leg(x, k, m, x.calendar(cal, days[0]), share) for k, m, cal, share in legs]
    for i, d in enumerate(days):
        t, px = t0 + i, x.c.iloc[t0 + i]
        for j, leg in enumerate(L):
            ex = L[0].held() if (no_overlap and j > 0) else set()
            leg.step(i, t, px, ex)
        if no_overlap and len(L) > 1 and i > 0 and (L[0].held() & L[1].held()):   # the first leg bought a name the second holds
            if L[1].mode.startswith("swap"):
                L[1].swap_check(t, px, L[0].held())
            else:
                L[1].rebalance(t, px, L[0].held())
        for j, leg in enumerate(L):                # "+mb": a held blocked stock is sold that day (the live manual sale) and
            if leg.key.endswith("+mb") and i > 0:  # its slot refilled by a rebalance of that half (the next in line)
                bad = {s for s in leg.held() if x.mblock.iloc[t][s]}
                if bad:
                    leg.rebalance(t, px, (L[0].held() if (no_overlap and j > 0) else set()) | bad)
        for j, leg in enumerate(L):
            out["legs"][j].append((leg.value(px), leg.cash, {s: (leg.sh[s], float(px[s])) for s in leg.held()}))
    _CACHE[key] = out
    return out

def line_rows(name, sim, which="total", scale=None, normalize=True):
    """(value rows, holdings rows) of a simulated line: which = "total" / leg index. Value 1.0 at the start close after its
    buys (normalize, forward_test's convention), else `scale` x the raw value (1.0 = cash before the buys)."""
    rows, hold = [], []
    legs = sim["legs"] if which == "total" else [sim["legs"][which]]
    if not len(sim["days"]):
        return rows, hold
    base = sum(lg[0][0] for lg in legs)
    k = (1.0 / base) if normalize else float(scale if scale is not None else 1.0)
    for i, d in enumerate(sim["days"]):
        val = sum(lg[i][0] for lg in legs) * k
        cash = sum(lg[i][1] for lg in legs) * k
        pos = {}
        for lg in legs:
            for s, (q, p) in lg[i][2].items():
                pos[s] = (pos.get(s, (0.0, p))[0] + q * k, p)
        rows.append({"Date": d, "Strategy": name, "Value": val, "Cash": cash})
        hold += [{"Date": d, "Strategy": name, "Symbol": s, "Weight": q * p / val if val > 0 else 0.0, "Shares": q,
                  "Price": p} for s, (q, p) in sorted(pos.items())]
    return rows, hold
