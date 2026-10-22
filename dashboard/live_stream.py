"""Display-only live prices for the dashboard: ONE Alpaca StockDataStream (SIP websocket: trades + 1-minute bars) per
dashboard server process, in a background thread (st.cache_resource singleton). Keeps the latest trade and today's
1-minute candles in memory for the symbols the dashboard watches (the open stock). Never trades, never touches the
bot. Alpaca allows one market-data websocket per account: if another process holds it (connection limit), this waits
60 s and retries, and the dashboard polls REST meanwhile (status() says which). Turn off: STOCK_ANALYSIS_STREAM=off."""
import asyncio
import logging
import os
import threading
import time

import pandas as pd
import streamlit as st

NY = "America/New_York"
log = logging.getLogger("sa.live_stream")

class LiveStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.last = {}            # sym -> (price, pd.Timestamp UTC, received epoch)
        self.mins = {}            # sym -> {minute NY naive: [o, h, l, c, v]}
        self.symbols = set()
        self.connected, self.error, self.stream, self.thread = False, None, None, None

    # ---- the websocket thread
    def _start(self):
        import backtest_engine as be  # noqa: F401  (loads .env)
        from alpaca.data.enums import DataFeed
        from alpaca.data.live import StockDataStream
        key = os.getenv("ALPACA_LIVE_KEY_ID") or os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_LIVE_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            self.error = "no Alpaca keys"
            return
        owner = self

        class _Stream(StockDataStream):
            async def _start_ws(self):
                try:
                    await super()._start_ws()
                    owner.connected, owner.error = True, None
                except Exception as e:           # e.g. "connection limit exceeded": back off, REST meanwhile
                    owner.connected, owner.error = False, str(e)[:120]
                    await self.close()
                    await asyncio.sleep(60)
                    raise

            async def _consume(self):
                try:
                    await super()._consume()
                except Exception:
                    owner.connected = False
                    raise

        self.stream = _Stream(key, secret, feed=DataFeed.SIP)
        self.thread = threading.Thread(target=self._run, name="sa-live-stream", daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            try:
                self.stream.run()
            except Exception as e:
                self.connected, self.error = False, str(e)[:120]
            time.sleep(30)

    # trade conditions that do not set the last price (out-of-sequence, average-price, derivative, prior-reference,
    # extended-hours / sold-late prints...): these made 1-minute spikes, so they are skipped
    SKIP = set("BCGHMNPQTUVWZ479")

    async def _on_trade(self, t):
        if any(c in self.SKIP for c in (t.conditions or [])):
            return
        ts = pd.Timestamp(t.timestamp)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        px, sz = float(t.price), float(t.size or 0)
        m = ts.tz_convert(NY).tz_localize(None).floor("1min")
        with self.lock:
            self.last[t.symbol] = (px, ts, time.time())
            c = self.mins.setdefault(t.symbol, {}).get(m)
            if c is None:
                self.mins[t.symbol][m] = [px, px, px, px, sz]
            else:
                c[1], c[2], c[3], c[4] = max(c[1], px), min(c[2], px), px, c[4] + sz

    async def _on_bar(self, b):
        ts = pd.Timestamp(b.timestamp)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        m = ts.tz_convert(NY).tz_localize(None).floor("1min")
        with self.lock:                                   # the official minute bar replaces the trade-built one
            self.mins.setdefault(b.symbol, {})[m] = [float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume)]

    # ---- used by the dashboard
    def watch(self, symbols):
        new = [s for s in symbols if s and s not in self.symbols]
        if not new:
            return
        self.symbols.update(new)
        try:
            if self.stream is None:
                self._start()
            if self.stream is not None:
                self.stream.subscribe_trades(self._on_trade, *new)
                self.stream.subscribe_bars(self._on_bar, *new)
        except Exception as e:
            self.error = str(e)[:120]

    def latest(self, sym, max_age=60):
        """(price, trade time UTC) when a streamed trade arrived in the last max_age seconds, else None."""
        with self.lock:
            x = self.last.get(sym)
        return (x[0], x[1]) if x and time.time() - x[2] <= max_age else None

    def minutes(self, sym):
        """Today's streamed 1-minute candles (NY time, no zone) as an OHLCV frame."""
        today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
        with self.lock:
            rows = {k: list(v) for k, v in self.mins.get(sym, {}).items() if k >= today}
        return pd.DataFrame.from_dict(rows, orient="index", columns=["Open", "High", "Low", "Close", "Volume"]).sort_index()

    def status(self):
        return "SIP stream" if self.connected else f"REST polling ({self.error})" if self.error else "REST polling"

@st.cache_resource(show_spinner=False)
def live_stream():
    """The one LiveStream of this server process, or None when turned off (tests)."""
    if os.getenv("STOCK_ANALYSIS_STREAM", "on") == "off" or os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off":
        return None
    return LiveStream()

def merge_into(bars, minutes, freq="5min"):
    """REST candles (`freq`, NY time) + streamed 1-minute candles: the streamed minutes are rolled up to `freq` and
    replace / extend the REST candles from the last REST candle on (that one may still be forming)."""
    if minutes is None or minutes.empty or bars.empty:
        return bars
    hm = minutes.index.hour * 60 + minutes.index.minute
    minutes = minutes[(hm >= 570) & (hm < 960) & (minutes.index.normalize() == bars.index[-1].normalize())]
    if minutes.empty:
        return bars
    roll = minutes.groupby(minutes.index.floor(freq)).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    roll = roll[roll.index >= bars.index[-1]]
    out = bars.copy()
    for t, r in roll.iterrows():
        if t in out.index and t == bars.index[-1]:     # the forming candle: keep its REST open, widen with the stream
            o = out.loc[t]
            out.loc[t] = [o["Open"], max(o["High"], r["High"]), min(o["Low"], r["Low"]), r["Close"], max(o["Volume"], r["Volume"])]
        else:
            out.loc[t] = r.values
    return out.sort_index()
