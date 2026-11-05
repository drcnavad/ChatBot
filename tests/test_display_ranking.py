"""sleeves.display_ranking (Nov 5, 2026, dashboard views only): every buyable stock gets a display rank (score <= 0 continue the numbering),
while trading's sleeves.ranking (score > 0) is untouched: display order starts with exactly the trading order."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import sleeves, sector_mapping as sm

inp = sleeves.inputs()
buy = {s for s in sm.tradable_symbols if s not in sm.do_not_buy}
for d in inp["close"].index[-5:]:
    for h in sleeves.HALVES:
        r, dr = sleeves.ranking(h, d, inp), sleeves.display_ranking(h, d, inp)
        assert (r > 0).all(), "trading ranking still score > 0 only"
        assert list(dr.index[:len(r)]) == list(r.index), "display starts with the trading order"
        assert set(dr.index) == buy and len(dr) == len(set(dr.index)), "every buyable stock once"
        tail = dr.iloc[len(r):].dropna()
        assert tail.is_monotonic_decreasing or len(tail) < 2
lr = sleeves.live_ranks(inp=inp, all_stocks=True)
r0 = sleeves.ranking(sleeves.HALVES[0], inp["close"].index[-1], inp)
assert list(sleeves.live_ranks(inp=inp)) == list(r0.index), "default live_ranks (chatbot) unchanged"
assert sorted(v[0] for v in lr.values()) == list(range(1, len(buy) + 1))
print(f"PASS display_ranking: {len(buy)} buyable ranked 1..{len(buy)}, trading ranking (score > 0) unchanged as prefix")
