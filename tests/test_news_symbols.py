"""run_all.news_symbols (Nov 5, 2026): holdings + QQQ + top 45 + bottom 45 buyable by live RS score, deduped, capped
at 99 (bottom end dropped first; holdings and QQQ always kept); no ranks -> fixed cap. Fake scores, no network."""
import json, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import run_all as r, sector_mapping as sm

buy = [s for s in sm.tradable_symbols if s not in sm.do_not_buy]
scores = {s: len(buy) / 2 - i for i, s in enumerate(buy)}          # first half > 0 (ranked), rest <= 0
dnb = sm.do_not_buy[0]
out = r.news_symbols(held=[dnb, buy[60]], scores=scores)
assert len(out) == len(set(out)) <= 99, len(out)
assert out[:3] == [dnb, buy[60], "QQQ"], out[:3]
assert set(buy[:45]) <= set(out), "top 45 kept"
bottom = buy[45:][-45:]
assert set(bottom) <= set(out) or len(out) == 99
assert not (set(out) - set(buy) - {dnb, "QQQ"}), "only buyable + held + QQQ"
print(f"PASS ranked: {len(out)} symbols, holdings + QQQ first, top/bottom 45 in")

r.NEWS_TOP = r.NEWS_BOTTOM = 60                                   # force the cap: bottom end drops first
out = r.news_symbols(held=[dnb], scores=scores)
assert len(out) == 99 and dnb in out and "QQQ" in out and set(buy[:60]) <= set(out)
assert buy[-1] in out, "worst stock kept; best-of-the-bottom dropped first"
r.NEWS_TOP = r.NEWS_BOTTOM = 45
print("PASS cap 99: holdings/QQQ/top kept, bottom end trimmed")

def boom():
    raise RuntimeError("no signal table")
import sleeves; real = sleeves.inputs; sleeves.inputs = boom
out = r.news_symbols(held=[])
sleeves.inputs = real
assert len(out) == min(99, len(buy) + 1) and out[0] == "QQQ" and out[1:] == buy[:len(out) - 1]
print("PASS no ranks: deterministic list-order cap of 99")

d = tempfile.mkdtemp(); p = os.path.join(d, "l.json")
json.dump({"halves": {"rs_short": {"shares": {"aaa": 2, "BBB": 0}}}}, open(p, "w"))
assert r.saved_holdings(p) == ["AAA"] and r.saved_holdings(os.path.join(d, "none.json")) == []
print("PASS saved_holdings from the sleeve ledger (missing file = none)")
