"""Friday replacement (Chirag, t191u, 2026-10-07): a NEW pick (the account holds none) blocked by earnings within the
window or by the no-buy-back after an earnings-day stop sale gives its weight to the next best-ranked eligible stock not
already picked (score above 0, not blocked, down to rank 20), so the account still ends with 10 stocks; cash only if
none is left. Held picks with earnings keep the old behaviour (kept, not topped up). Mid-week is unchanged.
Hand-worked cases with fake data; no network, no orders.
Run: python tests/run_tests.py  (or python tests/test_friday_substitute.py)"""
import json
import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import paper_trade as pt

FAIL = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
    if not ok:
        FAIL.append(name)


EQ = 100_000.0
SYMS = [f"S{i:02d}" for i in range(1, 26)]                            # S01 = rank 1 ... S25 = rank 25, all $100
R = pd.DataFrame({"Symbol": SYMS, "Rank": range(1, 26), "Score": [80 - i for i in range(1, 26)], "Close": 100.0,
                  "Provisional_Weight": [0.099] * 10 + [0.0] * 15, "Regime_On": 1, "Midweek_Check": 0})
T = pd.DataFrame({"Symbol": SYMS[:10], "Weight": [0.087] * 9 + [0.12], "Price": 100.0})   # the 10 Friday picks
HELD8 = {s: 99 for s in SYMS[:8]}                                     # S09, S10 are new picks


def kept(t):
    return sorted(t.loc[pd.to_numeric(t["Weight"]) > 0, "Symbol"])


def plan(t, st, bl, pos=HELD8):
    return pt.build_orders(t, EQ, pos, {s: 100.0 for s in SYMS}, fractional=True, statuses=st, blocked=bl)


check("floor: replacements only down to rank 20", pt.FRIDAY_SUB_MAX_RANK == 20)
ST = {s: "hold" if s in HELD8 else "add" for s in SYMS[:10]}

# 1) a new pick with earnings soon -> rank 11 takes its weight; still 10 stocks
t, st, bl, info = pt.substitute_blocked_picks(T, HELD8, R, {"S09": "earnings Tue Oct 13, in 4 days"}, ST)
o = plan(t, st, bl)
b = o[o["Side"] == "BUY"].set_index("Symbol")
check("S09 (new, earnings) replaced by S11 (rank 11) at S09's 8.7%; exactly 10 targets",
      info["replaced"] == [("S09", 9, "earnings Tue Oct 13, in 4 days", "S11", 11, 0.087)] and len(kept(t)) == 10
      and "S09" not in kept(t) and "S11" in kept(t), (info, kept(t)))
check("orders: BUY S11 ~$8,700 and BUY S10; S09 is a SKIP row that names the replacement",
      abs(b.loc["S11", "Est_Value"] - 8_700) < 1 and "S10" in b.index
      and o.loc[o.Symbol == "S09", "Side"].iloc[0].startswith("SKIP (earnings") and "replaced by S11" in o.loc[o.Symbol == "S09", "Side"].iloc[0],
      o[["Symbol", "Side", "Est_Value"]].values.tolist())
check("the replacement is a buy signal (status add) - an unknown status would never be bought", st["S11"] == "add", st.get("S11"))

# 2) a HELD pick with earnings soon: unchanged (kept, not topped up), no replacement
t, st, bl, info = pt.substitute_blocked_picks(T, HELD8, R, {"S01": "earnings Mon Oct 12, in 3 days"}, ST)
check("held S01 with earnings: no replacement (old behaviour)", info["replaced"] == [] and kept(t) == SYMS[:10], info)

# 3) blocked candidates are skipped: rank 11 earnings, rank 12 stop sale -> rank 13; a stop-blocked new pick is replaced too
blk = {"S10": "sold by the earnings-day stop, no buy back until after Wed Oct 14",
       "S11": "earnings Tue Oct 13, in 4 days", "S12": "sold by the earnings-day stop, no buy back until after Thu Oct 15"}
t, st, bl, info = pt.substitute_blocked_picks(T, HELD8, R, blk, ST)
check("stop-blocked new pick S10 (12%) -> S13 (S11 earnings, S12 stop sale skipped)",
      [(a, r) for a, _, _, r, *_ in info["replaced"]] == [("S10", "S13")] and info["replaced"][0][5] == 0.12
      and [s for s, _ in info["skipped"]] == ["S11", "S12"], info)

# 4) two blocked new picks: the better-ranked one gets the better replacement
t, st, bl, info = pt.substitute_blocked_picks(T, HELD8, R, {"S10": "earnings x", "S09": "earnings y"}, ST)
check("S09 -> S11, S10 -> S12 (rank order), 10 targets",
      [(a, r) for a, _, _, r, *_ in info["replaced"]] == [("S09", "S11"), ("S10", "S12")] and len(kept(t)) == 10, info)

# 5) floor: ranks 11-20 all blocked -> no replacement (rank 21 is never used); that weight stays cash
blk = {"S09": "earnings z", **{s: "earnings soon" for s in SYMS[10:20]}}
t, st, bl, info = pt.substitute_blocked_picks(T, HELD8, R, blk, ST)
o = plan(t, st, bl)
check("ranks 11-20 blocked: S09 unfilled (cash), S21 not bought; 9 stocks",
      info["unfilled"] == [("S09", 9, "earnings z")] and "S21" not in set(o["Symbol"])
      and sorted(o.loc[~o.Side.str.startswith("SKIP") & (pd.to_numeric(o.Target_Shares) > 0), "Symbol"]) == SYMS[:8] + ["S10"]
      and "no eligible stock down to rank 20" in o.loc[o.Symbol == "S09", "Side"].iloc[0], (info, o.values.tolist()))

# 6) cap: a 25% weight is capped at 19.8% for the replacement
T25 = T.assign(Weight=[0.08] * 9 + [0.25])
t, st, bl, info = pt.substitute_blocked_picks(T25, HELD8, R, {"S10": "earnings x"}, ST)
check("replacement weight capped at 19.8%", info["replaced"][0][5] == 0.198, info)

# 7) a replacement the account already holds (an extra ranked 11) is kept and brought to the weight, not sold
pos = {**HELD8, "S11": 30}
t, st, bl, info = pt.substitute_blocked_picks(T, pos, R, {"S09": "earnings x"}, ST)
o = plan(t, st, bl, pos)
r11 = o[o["Symbol"] == "S11"]
check("held S11 (rank 11, $3,000) replaces S09: status hold, bought up to 8.7% (not sold as a non-pick)",
      st["S11"] == "hold" and r11["Side"].iloc[0] == "BUY" and abs(r11["Target_Value"].iloc[0] - 8_700) < 1, r11.values.tolist())

# 8) nothing blocked -> targets untouched
t, st, bl, info = pt.substitute_blocked_picks(T, HELD8, R, {}, ST)
check("nothing blocked: targets unchanged", t.equals(T.reset_index(drop=True)) and info["replaced"] == [], info)

# --- end to end through plan_orders (fake picks / signals / earnings / stop state) ---
tmp = tempfile.mkdtemp()
picks, sig, mid, stop = (os.path.join(tmp, n) for n in ("picks.csv", "sig.csv", "mid.csv", "stop.json"))
pd.DataFrame([{"As_Of": "2026-10-09", "Last_Rebalance": "2026-10-09", "Last_Decision": "2026-10-09", "Strategy": "C6",
               "Symbol": s, "Strategy_Weight": 0.099, "Provisional_Weight": 0.099, "Close": 100.0} for s in SYMS[:10]]
             ).to_csv(picks, index=False)
rows = []
for i, s in enumerate(SYMS + ["NEG"], start=1):
    rk = 11 if s == "NEG" else (i if i <= 10 else i + 1)              # NEG: rank 11 but score below 0 (not eligible)
    rows.append({"Date": "2026-10-09", "Symbol": s, "Close": 100.0, "Strategy_Score": -2 if s == "NEG" else 70 - rk,
                 "Strategy_Rank": rk, "Strategy_Weight": 0.099 if i <= 10 else 0, "Provisional_Weight": 0.099 if i <= 10 else 0,
                 "Regime_On": 1, "Midweek_Check": 0})
pd.DataFrame(rows).to_csv(sig, index=False)
pd.DataFrame(columns=["As_Of", "Event", "Event_Date", "Action", "Sell", "Buy", "Message", "Weight_%"]).to_csv(mid, index=False)
json.dump({"sold": {"S12|2026-10-13": {"react": "2026-10-14"}}}, open(stop, "w"))       # S12: stop sale, no buy back
real = (pt.earnings_blocked, pt.latest_signal_status, pt.EARNINGS_STOP_STATE)
pt.earnings_blocked = lambda syms, as_of, earnings_csv=None: {s: "earnings Tue Oct 13, in 4 days" for s in ("S09", "S11") if s in syms}
pt.latest_signal_status = lambda changes_csv=None, as_of=None: {s: "hold" if s in HELD8 else "add" for s in SYMS[:10]}
pt.EARNINGS_STOP_STATE = stop
try:
    o, meta, tg = pt.plan_orders("auto", EQ, HELD8, picks_csv=picks, signal_csv=sig, midweek_csv=mid, fractional=True,
                                 cash=EQ - 8 * 9_900)
    sub = meta["substitutions"]
    b = set(o.loc[o.Side == "BUY", "Symbol"])
    check("Friday plan: S09 (earnings) -> S13 (S11 earnings, S12 stop sale skipped; NEG score below 0 never a candidate)",
          [(a, r) for a, _, _, r, *_ in sub["replaced"]] == [("S09", "S13")]
          and [s for s, _ in sub["skipped"]] == ["S11", "S12"], sub)
    check("source provisional+subs; buys S10 and S13; the account ends with exactly 10 stocks",
          meta["source"] == "provisional+subs" and b == {"S10", "S13"}
          and len(set(HELD8) | b) == 10 and abs(meta["invested"] - 0.99) < 1e-9, (meta["source"], b, meta["invested"]))
    rep, ok = pt.reconcile_positions("provisional+subs")
    check("reconcile_positions('provisional+subs') -> empty, ok (holdings differ from Provisional_Weight on purpose)",
          rep.empty and ok is True, (rep, ok))
    # mid-week: no Friday replacement step
    pd.DataFrame([{**r_, "As_Of": "2026-10-09", "Last_Rebalance": "2026-10-02"} for r_ in pd.read_csv(picks).to_dict("records")]
                 ).to_csv(picks, index=False)
    o, meta, _ = pt.plan_orders("auto", EQ, HELD8, picks_csv=picks, signal_csv=sig, midweek_csv=mid, fractional=True)
    check("not a rebalance day (hold): no Friday replacement step", "substitutions" not in meta and meta["source"] == "hold",
          meta["source"])
finally:
    pt.earnings_blocked, pt.latest_signal_status, pt.EARNINGS_STOP_STATE = real

print(f"\n{len(FAIL)} failed" if FAIL else "\nFRIDAY SUBSTITUTE OK")
sys.exit(1 if FAIL else 0)
