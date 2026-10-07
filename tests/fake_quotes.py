"""Fake quotes for the mocked tests: replaces paper_trade._latest_quote (no market-data call, no network).

    fq = fake_quotes.install(pt, quotes={"ANET": (139.60, 139.70)})   # (bid, ask) or (bid, ask, age in seconds)
A list gives one quote per call (the last one repeats); an Exception in it is raised (quote unreadable).
prices = {symbol: price} (read at each call): a tight quote with the ask at that price. Every other symbol gets `default`. Waits and retry pauses are set to 0 so the tests run instantly."""
from datetime import datetime, timedelta, timezone

class FakeQuotes:
    def __init__(self, quotes=None, default=(99.99, 100.01), feed="sip", prices=None):
        self.quotes, self.default, self.feed, self.calls = dict(quotes or {}), default, feed, []
        self.prices = prices if prices is not None else {}

    def __call__(self, sym):
        self.calls.append(sym)
        v = self.quotes.get(sym) or ((self.prices[sym] * 0.9998, self.prices[sym]) if sym in self.prices else self.default)
        if isinstance(v, list):
            v = v.pop(0) if len(v) > 1 else v[0]
        if isinstance(v, Exception):
            raise v
        bid, ask, age = (tuple(v) + (0,))[:3]
        return bid, ask, datetime.now(timezone.utc) - timedelta(seconds=age), self.feed

def install(pt, **kw):
    fq = FakeQuotes(**kw)
    pt._latest_quote = fq
    pt.QUOTE_RETRY_SECS = pt.LIMIT_WAIT_SECS = pt.LIMIT_POLL_SECS = 0
    return fq
