"""Two halves in one account (Chirag, Oct 15, 2026; live from Fri Oct 16): 50% monthly low volatility + 50% weekly sector
rotation, no overlap (sleeves.py, paper_trade.plan_sleeve_orders). Mocked: no broker calls; ledger in a temp file.

Covers: the config; the calendar (Oct 16 both halves, other Fridays the sector half, month-ends the low-volatility half,
Mon-Thu nothing); the sector half's weights = forward_test's Sector rotation selection (the backtest's); the split
(each half 50% at the switch, sized from its own value); no overlap (the sector half never buys a low-volatility name, a
low-volatility buy of a sector holding moves the shares over); a half that does not rebalance is left alone; holdings no
half owns are sold; the ledger reconciliation (an unfilled buy gives its dollars back to its own half; the rest pro rata)."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["STOCK_ANALYSIS_SLEEVE_LEDGER"] = os.path.join(tempfile.mkdtemp(), "sleeve_ledger.json")   # never the real one
import pandas as pd

import backtest_engine as be
import forward_test as ft
import sleeves

FAIL = []
def check(ok, what, detail=""):
    print(("PASS " if ok else "FAIL ") + what + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(what)

check(be.WINNER.get("sleeves") == {"lowvol": 0.5, "sector": 0.5}, "WINNER['sleeves'] = 50/50")
check("+SR-W" in be.WINNER["tag"] and "Sector rotation" in be.WINNER["name"], "tag / name show both halves", be.WINNER["tag"])
check(sleeves.LEDGER_JSON.startswith(tempfile.gettempdir()) or "tmp" in sleeves.LEDGER_JSON, "test ledger is a temp file")

# calendar
kinds = {d.date().isoformat(): be.decision_kind(d) for d in pd.date_range("2026-10-15", "2026-12-04") if be.decision_kind(d)}
check(kinds == {"2026-10-16": "full rebalance", "2026-10-19": be.SECTOR_KIND, "2026-10-26": be.SECTOR_KIND,
                "2026-10-30": "full rebalance", "2026-11-02": be.SECTOR_KIND, "2026-11-09": be.SECTOR_KIND,
                "2026-11-16": be.SECTOR_KIND, "2026-11-23": be.SECTOR_KIND, "2026-11-30": "full rebalance"},
      "decisions: Oct 16 / Oct 30 / Nov 30 (a Monday: both halves) full, every other Monday the sector half, no Fridays", kinds)
check(be.is_sector_day("2027-01-19") and not be.is_sector_day("2027-01-18") and be.is_sector_day("2026-11-30"),
      "Monday holiday (MLK, Jan 18, 2027) -> the sector half on Tuesday; Nov 30 (month end, Monday) is a sector day too")
cal = dict(sleeves.calendar("2026-10-15", 40))
check(cal[pd.Timestamp("2026-10-16")] == ["lowvol"] and cal[pd.Timestamp("2026-10-19")] == ["sector"]
      and cal[pd.Timestamp("2026-11-30")] == ["lowvol", "sector"] and pd.Timestamp("2026-10-23") not in cal,
      "halves: Oct 16 low vol (the switch bought both: ledger init), Mon Oct 19 sector, Nov 30 (Mon, month end) both, Fri Oct 23 none")

# the sector half's weights = the forward test's Sector rotation (the backtest's own selection)
inp = sleeves.sector_inputs()
idx = inp["close"].index
d = idx[-1]
cfg = next(c for c in ft.STRATEGIES if c["name"] == "Sector rotation")
fri = pd.Series([be.next_sessions(x, 1)[0].isocalendar()[:2] != x.isocalendar()[:2] for x in idx], index=idx)
args, _ = sleeves.sector_args(inp)
tgt, _ = be.winner_targets(inp["scores"]["sector_rot"], inp["eligible"], inp["vol"], inp["regime"], fri,
                           tiebreak_w=inp["scores"]["rs"], midweek=False, selection={k: v for k, v in args.items() if k != "regime"},
                           earnings_block_days=None, earnings=inp["earnings"], exit_all_below=None, exit_to_top=None,
                           never_buy=be.sector_mapping.do_not_buy)
last_fri = fri[fri].index[-1]
fw = be.live_weights(tgt.loc[[last_fri]].clip(upper=ft.MAX_WEIGHT)).iloc[0]
prev = tgt.iloc[idx.get_loc(last_fri) - 1]
sw = sleeves.sector_weights(last_fri, held={s: w for s, w in prev.items() if w > 0}, inp=inp)
check(set(fw[fw > 0].index) == set(sw[sw > 0].index) and float((fw.reindex(sw.index).fillna(0) - sw).abs().max()) < 1e-12,
      "sector half weights = forward_test Sector rotation (weekly, 10 stocks) on the last Friday")
check(cfg["score"] == "sector_rot" and args["n"] == be.WINNER["n"] == 5 and args["sector_cap"] == 0.4 and args["vol_sizing"]
      and args["regime_scale"] == 0.5, "sector half: score sector_rot, 5 stocks (Oct 21), max 2/sector, 1/vol weights, QQQ halving")
ex = set(sw[sw > 0].index[:3])
sw2 = sleeves.sector_weights(last_fri, exclude=ex, inp=inp)
check(not (set(sw2[sw2 > 0].index) & ex) and (sw2 > 0).sum() == (sw > 0).sum(), "excluded names are never picked; the next stock takes the slot")
check(not (set(sw[sw > 0].index) & set(be.sector_mapping.do_not_buy)), "do_not_buy never picked")

# a synthetic account: plan the switch, then a sector-only Friday, then a month-end
day = d
P = {s: float(inp["close"].at[day, s]) for s in inp["close"].columns if inp["close"].at[day, s] == inp["close"].at[day, s]}
lvw = pd.Series(0.099, index=[s for s in list(P)[:10]])
lv_t = pd.DataFrame({"Symbol": lvw.index, "Weight": lvw.values, "Price": [P[s] for s in lvw.index]})
sec_first = sleeves.sector_weights(day, exclude=set(lvw.index), inp=inp)
sec_name = sec_first[sec_first > 0].index[0]
old = "ZZZOLD"
P[old] = 50.0
positions = {lvw.index[0]: 10.0, sec_name: 5.0, old: 100.0}
equity = 100_000.0 + sum(q * P[s] for s, q in positions.items())
orig_lv, orig_wk = sleeves.lowvol_day, sleeves.sector_day
sleeves.lowvol_day, sleeves.sector_day = (lambda x: True), (lambda x: True)
o, info = sleeves.plan(day, positions, equity, P, lv_targets=lv_t, lv_statuses={s: "add" for s in lvw.index}, ledger=None, inp=inp)
L = info["ledger"]["halves"]
check(info["init"] and abs(info["values"]["lowvol"] - equity / 2) < 0.01 and abs(info["values"]["sector"] - equity / 2) < 0.01,
      "switch: each half = 50% of equity")
def planned(inf, h):
    """{sym: (target shares, price)} of half h after the plan (orders.Target_Shares - the ledger holds no shares)."""
    o_ = inf["orders"]
    return {r.Symbol: (float(r.Target_Shares), float(r.Price)) for r in o_[(o_.Half == h)].itertuples() if float(r.Target_Shares or 0) > 0}
def fills_for(o_, skip=()):
    return [{"symbol": r.Symbol, "side": r.Side, "qty": float(r.Shares), "price": float(r.Price)}
            for r in o_[o_.Side.isin(["BUY", "SELL"])].itertuples() if r.Symbol not in skip]
PL = {h: planned(info, h) for h in sleeves.HALVES}
check(lvw.index[0] in PL["lowvol"] and sec_name in PL["sector"]
      and not o[(o.Symbol == lvw.index[0]) & (o.Side == "SELL") & (o.Shares >= 10)].shape[0],
      "switch: a held stock one half targets is taken over by that half (not sold and bought back)")
check(((o.Symbol == old) & (o.Side == "SELL") & (o.Shares == 100)).any(), "switch: a holding no half targets is sold in full")
lvn, secn = set(PL["lowvol"]), set(PL["sector"])
check(not (lvn & secn) and len(lvn) == 10 and len(secn) == (sec_first > 0).sum(), "no overlap; 10 + 10 names", (lvn & secn))
check(all("shares" not in L[h] and set(L[h]["tags"]) >= set(PL[h]) for h in sleeves.HALVES),
      "ledger: tags only (held + this run's buys), never shares (live positions are the record)", L)
inv = {h: sum(q * p for q, p in PL[h].values()) / info["values"][h] for h in sleeves.HALVES}
check(0.97 <= inv["lowvol"] <= 0.99 + 1e-9 and 0.75 <= inv["sector"] <= 0.99 + 1e-9,
      "each half invested of its OWN value: low vol ~99%, sector <= 99% (5 stocks: the 20% cap leaves cash)", inv)
w_lv = {s: q * p / info["values"]["lowvol"] for s, (q, p) in PL["lowvol"].items()}
check(all(abs(w - 0.099) < 0.0101 for w in w_lv.values()), "low-volatility weights = 9.9% of the half (1-point band)", w_lv)

# everything fills as planned: the next run's halves come from the live positions + the real fills
pos2 = {s: q for h in sleeves.HALVES for s, (q, _p) in PL[h].items()}
F = fills_for(o)
cash_h = {h: L[h]["cash"] + sum(f_["qty"] * f_["price"] * (-1 if f_["side"] == "BUY" else 1) for f_ in F
                                 if sleeves.owners(info["ledger"]).get(f_["symbol"]) == h) for h in sleeves.HALVES}
cash2 = sum(cash_h.values())                    # ZZZOLD was in the switch equity already
P2 = {s: p * (1.05 if s in lvn else 0.97) for s, p in P.items()}
eq2 = cash2 + sum(q * P2[s] for s, q in pos2.items())
st2, _u2, r2 = sleeves.reconcile(info["ledger"], pos2, P2, eq2, fills=F)
check(all(st2[h]["shares"] == {s: q for s, (q, _p) in PL[h].items()} for h in sleeves.HALVES),
      "reconcile: each half's shares = the LIVE shares of its tagged symbols")
sleeves.lowvol_day = lambda x: False
o2, info2 = sleeves.plan(day, pos2, eq2, P2, ledger=info["ledger"], inp=inp, fills=F)
t2 = o2[o2.Side.isin(["BUY", "SELL"])]
check(not info2["lowvol"] and info2["sector"] and not (set(t2.Symbol) & lvn), "sector-only Friday: no low-volatility name traded")
lv_own = cash_h["lowvol"] + sum(q * P2[s] for s, (q, _p) in PL["lowvol"].items())
check(abs(info2["values"]["lowvol"] - lv_own) / lv_own < 0.01, "each half keeps its own gains (low-volatility half +5%, sector half -3%)",
      (info2["values"]["lowvol"], lv_own))

# a buy that never filled: no fill -> it costs its half nothing, and its tag goes (not held, nothing queued)
miss = sorted(secn)[0]
pos3 = {s: q for s, q in pos2.items() if s != miss}
F3 = fills_for(o, skip={miss})
c3 = {h: L[h]["cash"] + sum(f_["qty"] * f_["price"] * (-1 if f_["side"] == "BUY" else 1) for f_ in F3
                             if sleeves.owners(info["ledger"]).get(f_["symbol"]) == h) for h in sleeves.HALVES}
eq3 = sum(c3.values()) + sum(q * P[s] for s, q in pos3.items())
st, unowned, rinfo = sleeves.reconcile(info["ledger"], pos3, P, eq3, fills=F3)
check(st["sector"]["cash"] - c3["sector"] > -0.01 and miss in rinfo["dropped"] and miss not in st["sector"]["shares"],
      "unfilled buy: no fill, so its dollars stay in the sector half; its tag is dropped", rinfo)
st_p, _u, r_p = sleeves.reconcile(info["ledger"], pos3, P, eq3, fills=F3, pending={miss})
check(miss not in r_p["dropped"] and miss not in st_p["sector"]["shares"], "a queued buy keeps its tag but is not a holding")
pos4 = {**pos2, "HANDBUY": 1.0}
P4 = {**P, "HANDBUY": 100.0}
st4, un4, r4 = sleeves.reconcile(info["ledger"], pos4, P4, eq2 + 100.0, fills=F)
check(un4 == {"HANDBUY": 1.0}, "a holding no half owns is reported as unowned (sold at the next trade run)")

# month-end (low volatility only): it buys a name the sector half holds -> shares move over, sector half rebalances
sleeves.lowvol_day, sleeves.sector_day = (lambda x: True), (lambda x: False)
take = sorted(secn)[0]
lvw5 = pd.Series(0.099, index=list(lvw.index[:9]) + [take])
lv_t5 = pd.DataFrame({"Symbol": lvw5.index, "Weight": lvw5.values, "Price": [P[s] for s in lvw5.index]})
o5, info5 = sleeves.plan(day, pos2, eq2, P, lv_targets=lv_t5, lv_statuses={s: "add" for s in lvw5.index},
                         ledger=info["ledger"], inp=inp, fills=F)
P5 = {h: planned(info5, h) for h in sleeves.HALVES}
check(any(t[0] == take and t[1] == "sector" and t[2] == "lowvol" for t in info5["transfers"]) and info5["sector"]
      and take in P5["lowvol"] and take not in P5["sector"] and take in info5["ledger"]["halves"]["lowvol"]["tags"]
      and take not in info5["ledger"]["halves"]["sector"]["tags"]
      and not ((o5.Symbol == take) & (o5.Half == "sector") & o5.Side.isin(["BUY", "SELL"])).any()
      and abs(P5["lowvol"][take][0] * P[take] / info5["values"]["lowvol"] - 0.099) < 0.0101,
      "month-end clash: the shares move to the low-volatility half (the sector half does not sell them), trimmed to its 9.9%; "
      "the sector half rebalances without it")
sleeves.lowvol_day, sleeves.sector_day = orig_lv, orig_wk

# no ledger on a sector-only day: fail closed
try:
    sleeves.plan(pd.Timestamp("2026-10-23"), pos2, 1e5, P, ledger=None, inp=inp)
    check(False, "no ledger on a sector-only Friday refuses")
except ValueError:
    check(True, "no ledger on a sector-only Friday refuses (fail closed)")

# save / load
path = sleeves.save_ledger(info["ledger"])
check(sleeves.load_ledger(path)["halves"]["sector"]["tags"] == info["ledger"]["halves"]["sector"]["tags"], "ledger save / load")

print("\nSLEEVES OK" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
