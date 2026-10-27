"""The live sleeve's ranking at the latest close (manual_trade.py uses _rankings; the old stock-list groups were removed).
Display only: nothing here feeds the strategy or orders."""


def _rankings():
    """{half: (targets, ranking)} at the latest close (live: RS short-term, Oct 27, 2026): targets = sleeves.half_weights
    > 0, ranking = sleeves.ranking (best first)."""
    import sleeves
    import sector_mapping as sm
    inp = sleeves.inputs()
    day = inp["close"].index[-1]
    keep = set(sm.tradable_symbols)
    f = lambda xs: [x for x in xs if str(x).upper() in keep]
    t = {h: list(sleeves.half_weights(h, day, inp).pipe(lambda w: w[w > 0]).sort_values(ascending=False, kind="stable").index)
         for h in sleeves.HALVES}
    r = {h: list(sleeves.ranking(h, day, inp).index) for h in sleeves.HALVES}
    return {h: (f(t[h]), f(r[h])) for h in sleeves.HALVES}
