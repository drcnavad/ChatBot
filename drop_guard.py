"""Intraday drop guard for the LIVE Alpaca account (Chirag, Oct 29–30 2026).

Every 15 minutes on market days ~8:30 AM–2:15 PM CT, checks every held live position.
Trigger (configurable): mid quote down >= DROP_PCT (default 4.1%) vs the previous regular close.
When triggered, an LLM reviews earnings facts (when a recent report exists) plus a full
technical snapshot computed in code (SMAs/EMAs, Fib, RSI, MACD, Bollinger, ATR, S/R,
intraday volume vs average, gap, RS vs QQQ/sector). No news headlines.
Sell only when the model returns action=SELL with confidence >= MIN_CONFIDENCE.
After a sell fills, immediately buy the top-ranked eligible replacement with that sale's
proceeds only (do_not_buy, 15% weight cap, above SMA200, not sold by the guard today,
not itself down >= the same drop_pct today). If nothing qualifies, cash waits for the 2:30 run.

Hard guardrails: never sell do_not_sell; sell path never buys a new name except the
replacement; max one sell per symbol per day; bid/ask limit orders (same as paper_trade);
kill switch + dry_run in config; idempotent lock; shares the trade lock with the 2:30 job.

Schedule: launchd com.stockanalysis.drop_guard every 15 min.
  python drop_guard.py --dry-run [--as-of "2026-10-30 11:00"]
  python drop_guard.py --scheduled          # one LIVE check (launchd)
  python drop_guard.py --backfill-outcomes  # fill close / next-day prices on past log rows
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import db  # noqa: E402  tables live in SQLite
import backtest_engine as be
import sector_mapping as sm

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "Reports")
CONFIG_JSON = os.path.join(REPORTS, "drop_guard_config.json")
STATE_JSON = os.path.join(REPORTS, "drop_guard_state.json")
LOG_CSV = os.path.join(REPORTS, "drop_guard_log.csv")
LOCK_PATH = os.path.join(REPORTS, ".drop_guard.lock")
EARNINGS_CSV = os.path.join(REPORTS, "earnings_date.csv")
SIGNAL_CSV = os.path.join(REPORTS, "signal_analysis.csv")
RUN = "Drop guard"

DEFAULT_CONFIG = {
    "enabled": True,                 # kill switch: False = evaluate+log only, never order
    "dry_run": True,                 # True = never submit orders (safe default; live Reports JSON overrides)
    "drop_pct": 0.041,                # trigger: mid <= prev_close * (1 - drop_pct)
    "min_confidence": 0.75,          # sell only when AI confidence >= this
    "window_start_ct": [8, 30],      # CT inclusive
    "window_end_ct": [14, 15],       # CT exclusive (stop before 2:30 job)
    "max_weight": 0.15,              # replacement: do not push a name above this of equity
    "earnings_lookback_days": 5,     # include earnings facts when a report falls in this window
    "llm_required_for_live": True,   # refuse live sells when no LLM key is available
    "vol_avg_days": 20,              # days for average volume / typical cumulative volume
}

LOG_COLS = [
    "Checked_At_CT", "Symbol", "Shares", "Prev_Close", "Price", "Vs_Prev_%",
    "Triggered", "AI_Action", "AI_Confidence", "AI_Reasons", "Action",
    "Sell_Order_Id", "Sell_Limit", "Sell_Filled_Qty", "Sell_Proceeds",
    "Replace_Symbol", "Replace_Order_Id", "Replace_Limit", "Replace_Qty",
    "Replace_Note", "Close_Price", "Close_Vs_Sell_%", "Next_Close", "Next_Vs_Sell_%",
    "Dry_Run", "Note",
]

# LLM provider chain (same order as pipeline_watchdog.py).
# Gemini: flash-latest can take 30s+ / empty with small max tokens; flash-lite is fast.
# Groq: Cloudflare blocks Python-urllib default UA (error 1010); llama-3.3-70b retired for this key.
LLM_PROVIDERS = (
    ("GEMINI_API_KEY", "gemini-flash-lite-latest", "gemini"),
    ("GROQ_API_KEY", "openai/gpt-oss-20b", "groq"),
    ("ANTHROPIC_API_KEY", "claude-haiku-4-5", "anthropic"),
    ("OPENAI_API_KEY", "gpt-4o-mini", "openai"),
    ("XAI_API_KEY", "grok-2-latest", "xai"),
)
GEMINI_FALLBACK_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
)
GROQ_FALLBACK_MODELS = (
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
)
LLM_TIMEOUT_S = 45
LLM_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ----------------------------------------------------------------------------- config / state / lock
def load_config(path=None):
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(path or CONFIG_JSON) as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULT_CONFIG or k.startswith("_")})
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg, path=None):
    path = path or CONFIG_JSON
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)


def _load_state(path=None):
    try:
        with open(path or STATE_JSON) as f:
            s = json.load(f)
    except (OSError, ValueError):
        s = {}
    s.setdefault("sold_today", {})       # {SYMBOL: {at, order_id, ...}} keyed per calendar day via prune
    s.setdefault("day", "")
    return s


def _save_state(s, path=None):
    path = path or STATE_JSON
    today = str(datetime.now(be.CENTRAL).date())
    if s.get("day") != today:
        s["sold_today"] = {}
        s["day"] = today
    s.pop("notes", None)  # unused (earnings_stop has its own state)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, path)


def _ensure_day(state):
    today = str(datetime.now(be.CENTRAL).date())
    if state.get("day") != today:
        state["sold_today"] = {}
        state["day"] = today
    return today


def _acquire_lock(path=LOCK_PATH, stale_s=14 * 60):
    """One guard process at a time. Stale lock (dead PID or older than stale_s) is taken over.
    Uses O_EXCL so two concurrent ticks cannot both think they own the lock."""
    now = time.time()
    payload = json.dumps({"pid": os.getpid(), "at": now})
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        return True, ""
    except FileExistsError:
        pass
    try:
        with open(path) as f:
            meta = json.load(f)
        pid, at = int(meta.get("pid", 0)), float(meta.get("at", 0))
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive and (now - at) < stale_s:
            return False, f"held by pid {pid}"
    except (OSError, ValueError, TypeError):
        pass
    try:
        os.remove(path)
    except OSError:
        pass
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        return True, ""
    except FileExistsError:
        return False, "lock race"


def _release_lock(path=LOCK_PATH):
    try:
        with open(path) as f:
            meta = json.load(f)
        if int(meta.get("pid", 0)) == os.getpid():
            os.remove(path)
    except (OSError, ValueError, TypeError):
        pass


def _lock_busy(path):
    try:
        with open(path) as f:
            pid = int(json.load(f).get("pid", 0))
        os.kill(pid, 0)
        return pid > 0
    except (OSError, ValueError, TypeError):
        return False


# ----------------------------------------------------------------------------- schedule window
def in_window(now_ct, cfg):
    """True when now_ct is a regular (or early-close) session day inside the CT guard window."""
    d = now_ct.date()
    if not be.is_session(d):
        return False, "market closed (weekend/holiday)"
    sh, sm = cfg["window_start_ct"]
    eh, em = cfg["window_end_ct"]
    start = now_ct.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now_ct.replace(hour=eh, minute=em, second=0, microsecond=0)
    # Early-close days: stop 30 min before the early close (1:00 PM ET = 12:00 CT)
    if be.is_early_close(d):
        end = min(end, now_ct.replace(hour=11, minute=45, second=0, microsecond=0))
    if now_ct < start:
        return False, f"before window ({sh:02d}:{sm:02d} CT)"
    if now_ct >= end:
        return False, f"after window (ends {eh:02d}:{em:02d} CT; leaves the 2:30 run alone)"
    # Regular hours only for quotes/OR (pre-market not required for this guard)
    et = now_ct.astimezone(be.EASTERN)
    close_h = 13 if be.is_early_close(d) else 16
    m = et.hour * 60 + et.minute
    if m < 9 * 60 + 30:
        return False, "before the regular open (9:30 ET)"
    if m >= close_h * 60:
        return False, "after the regular close"
    return True, "in window"


# ----------------------------------------------------------------------------- inputs (tests replace)
def account():
    import alpaca_paper as ap
    return ap.PaperAccount()


def trading_client():
    import paper_trade as pt
    return pt.paper_trading_client()


def latest_quote(sym):
    import paper_trade as pt
    return pt._latest_quote(sym)


def log_event(status, money, message, details=""):
    import run_all
    run_all.log_event(RUN, status, money, message, details)


def daily_bars(symbols, now, lookback=260):
    """Daily OHLCV ending before today until the bar is final; includes today after ~3:05 CT."""
    import run_all
    et = now.astimezone(be.EASTERN)
    start = (min(et, datetime.now(be.EASTERN)) - timedelta(days=lookback + 30)).date()
    bars = be.fetch_daily_bars(list(symbols), start=start)
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])
    if (et.hour, et.minute) < run_all.BAR_FINAL_ET:
        bars = bars[bars["Date"] < pd.Timestamp(et.date())]
    return bars


def minute_bars(symbols, day, data_client=None):
    """1-minute bars for `day` (ET calendar date) for volume / intraday context. Empty on failure."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    client = data_client or be.market_data_client()
    start = datetime(day.year, day.month, day.day, 9, 30, tzinfo=be.EASTERN)
    end = datetime(day.year, day.month, day.day, 16, 0, tzinfo=be.EASTERN)
    now = datetime.now(be.EASTERN)
    if end > now:
        end = now
    out = []
    syms = list(symbols)
    for i in range(0, len(syms), 50):
        chunk = syms[i:i + 50]
        req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Minute,
                               start=start, end=end, feed=DataFeed.IEX)
        try:
            df = client.get_stock_bars(req).df
        except Exception:
            continue
        if df is None or df.empty:
            continue
        d = df.reset_index()
        if "symbol" in d.columns:
            d = d.rename(columns={"symbol": "Symbol", "timestamp": "ts",
                                  "open": "Open", "high": "High", "low": "Low",
                                  "close": "Close", "volume": "Volume"})
        out.append(d)
    if not out:
        return pd.DataFrame(columns=["Symbol", "ts", "Open", "High", "Low", "Close", "Volume"])
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- technical snapshot
def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _sma(s, n):
    return s.rolling(n).mean()


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close):
    macd = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd, 9)
    return macd, signal, macd - signal


def _bollinger(close, n=20, k=2.0):
    mid = _sma(close, n)
    sd = close.rolling(n).std()
    return mid + k * sd, mid, mid - k * sd


def _atr(h, l, c, n=14):
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _fib_levels(high, low):
    """Retracements from recent swing high -> low (and extensions)."""
    if not (math.isfinite(high) and math.isfinite(low)) or high <= low:
        return {}
    diff = high - low
    return {f"fib_{p}": round(high - diff * p, 4) for p in (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)}


def _support_resistance(h, l, c, look=40):
    """Simple recent swing high/low + prior closes as S/R."""
    hh = float(h.tail(look).max()) if len(h) else float("nan")
    ll = float(l.tail(look).min()) if len(l) else float("nan")
    pivots = sorted({round(float(x), 2) for x in c.tail(look).tolist() if math.isfinite(x)})
    return {"swing_high": hh, "swing_low": ll,
            "near_supports": pivots[:3], "near_resistances": pivots[-3:]}


def technical_snapshot(daily, minute, price, prev_close, cfg, qqq_ret=None, sector_ret=None):
    """Dict of indicators for one symbol. `daily` = Date-indexed OHLCV; `minute` optional."""
    c = daily["Close"].astype(float)
    h = daily["High"].astype(float) if "High" in daily else c
    l = daily["Low"].astype(float) if "Low" in daily else c
    v = daily["Volume"].astype(float) if "Volume" in daily else pd.Series(dtype=float)
    last = float(c.iloc[-1]) if len(c) else float("nan")
    px = float(price) if price and math.isfinite(price) else last
    smas = {f"sma_{n}": float(_sma(c, n).iloc[-1]) for n in (20, 50, 100, 200) if len(c) >= n}
    emas = {f"ema_{n}": float(_ema(c, n).iloc[-1]) for n in (20, 50, 100, 200) if len(c) >= n}
    pos = {}
    for name, series in {**smas, **emas}.items():
        if math.isfinite(series) and series:
            pos[f"vs_{name}_pct"] = round((px / series - 1) * 100, 3)
    macd, sig, hist = _macd(c)
    upper, mid, lower = _bollinger(c)
    atr = _atr(h, l, c)
    swing_high = float(h.tail(60).max()) if len(h) else float("nan")
    swing_low = float(l.tail(60).min()) if len(l) else float("nan")
    fib = _fib_levels(swing_high, swing_low)
    sr = _support_resistance(h, l, c)
    gap_pct = float("nan")
    if math.isfinite(prev_close) and prev_close > 0 and "Open" in daily.columns and len(daily):
        # today's open if present in minute bars, else last daily open
        today_open = float("nan")
        if minute is not None and not minute.empty:
            today_open = float(minute["Open"].iloc[0])
        elif len(daily) and math.isfinite(float(daily["Open"].iloc[-1])):
            today_open = float(daily["Open"].iloc[-1])
        if math.isfinite(today_open):
            gap_pct = (today_open / prev_close - 1) * 100
    vol_info = {}
    n = int(cfg.get("vol_avg_days", 20))
    if len(v) >= n:
        vol_info["avg_daily_vol"] = float(v.tail(n).mean())
        vol_info["last_daily_vol"] = float(v.iloc[-1])
    if minute is not None and not minute.empty and "Volume" in minute.columns:
        cum = float(minute["Volume"].sum())
        vol_info["intraday_cum_vol"] = cum
        # rough time-of-day: fraction of session elapsed * avg daily
        elapsed = max(len(minute), 1)
        session_mins = 390
        frac = min(1.0, elapsed / session_mins)
        if vol_info.get("avg_daily_vol"):
            expected = vol_info["avg_daily_vol"] * frac
            vol_info["vol_vs_expected"] = round(cum / expected, 3) if expected else None
            vol_info["intraday_vs_avg_day"] = round(cum / vol_info["avg_daily_vol"], 3)
    rs = {}
    day_ret = (px / prev_close - 1) * 100 if prev_close else float("nan")
    rs["day_return_pct"] = round(day_ret, 3) if math.isfinite(day_ret) else None
    if qqq_ret is not None and math.isfinite(day_ret) and math.isfinite(qqq_ret):
        rs["vs_qqq_pct"] = round(day_ret - qqq_ret * 100, 3)
    if sector_ret is not None and math.isfinite(day_ret) and math.isfinite(sector_ret):
        rs["vs_sector_pct"] = round(day_ret - sector_ret * 100, 3)
    return {
        "price": round(px, 4),
        "prev_close": round(float(prev_close), 4) if prev_close else None,
        "day_return_pct": rs.get("day_return_pct"),
        "gap_pct": round(gap_pct, 3) if math.isfinite(gap_pct) else None,
        "smas": {k: round(v, 4) for k, v in smas.items() if math.isfinite(v)},
        "emas": {k: round(v, 4) for k, v in emas.items() if math.isfinite(v)},
        "position_vs_mas_pct": pos,
        "rsi_14": round(float(_rsi(c).iloc[-1]), 2) if len(c) >= 15 else None,
        "macd": round(float(macd.iloc[-1]), 4) if len(macd) else None,
        "macd_signal": round(float(sig.iloc[-1]), 4) if len(sig) else None,
        "macd_hist": round(float(hist.iloc[-1]), 4) if len(hist) else None,
        "bollinger_upper": round(float(upper.iloc[-1]), 4) if len(upper) else None,
        "bollinger_mid": round(float(mid.iloc[-1]), 4) if len(mid) else None,
        "bollinger_lower": round(float(lower.iloc[-1]), 4) if len(lower) else None,
        "atr_14": round(float(atr.iloc[-1]), 4) if len(atr) else None,
        "fibonacci": fib,
        "support_resistance": sr,
        "volume": vol_info,
        "relative_strength": rs,
        "above_sma200": bool(smas.get("sma_200") and px >= smas["sma_200"]),
    }


def recent_earnings(sym, day, earnings, lookback_days):
    """Facts about the nearest earnings report within lookback (past or today)."""
    rows = earnings[earnings["Symbol"] == sym].copy()
    if rows.empty:
        return None
    rows["Earnings Date"] = pd.to_datetime(rows["Earnings Date"], errors="coerce")
    day = pd.Timestamp(day).normalize()
    rows = rows.dropna(subset=["Earnings Date"])
    rows["delta"] = (rows["Earnings Date"] - day).dt.days
    near = rows[(rows["delta"] >= -int(lookback_days)) & (rows["delta"] <= 1)]
    if near.empty:
        return None
    r = near.sort_values("delta", key=lambda s: s.abs()).iloc[0]
    return {
        "earnings_date": f"{pd.Timestamp(r['Earnings Date']):%Y-%m-%d}",
        "report_time": str(r["Time"]).strip().upper() if "Time" in r and pd.notna(r["Time"]) else "?",
        "days_from_today": int(r["delta"]),
        "note": "EPS/revenue vs estimates are not stored in this repo; date/time only.",
    }


# ----------------------------------------------------------------------------- LLM
def _load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:
        pass


def llm_provider():
    _load_dotenv()
    override = os.getenv("DROP_GUARD_LLM_MODEL") or os.getenv("PIPELINE_LLM_MODEL") or os.getenv("LLM_PROVIDERS")
    for env, model, name in LLM_PROVIDERS:
        key = os.getenv(env)
        if key:
            if override and name in ("gemini", "groq", "anthropic", "openai", "xai"):
                # keep provider from first key found; model override only when it matches provider family
                pass
            return name, (override if (override and name == "gemini" and "gemini" in override.lower()) else model), key
    return None, None, None


def _llm_complete(provider, model, key, prompt):
    headers = {"content-type": "application/json", "User-Agent": LLM_USER_AGENT}
    if provider == "gemini":
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
               f":generateContent?key={key}")
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 1024,
                    "temperature": 0.1,
                    "responseMimeType": "application/json",
                }}
    elif provider in ("groq", "openai", "xai"):
        base = {"groq": "https://api.groq.com/openai/v1/chat/completions",
                "openai": "https://api.openai.com/v1/chat/completions",
                "xai": "https://api.x.ai/v1/chat/completions"}[provider]
        url = base
        headers["Authorization"] = f"Bearer {key}"
        body = {"model": model, "max_tokens": 800, "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]}
    else:  # anthropic
        url = "https://api.anthropic.com/v1/messages"
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        body = {"model": model, "max_tokens": 800,
                "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if provider == "gemini":
        parts = payload["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    if provider in ("groq", "openai", "xai"):
        return payload["choices"][0]["message"]["content"]
    return "".join(b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text")


def _parse_ai_json(text):
    if not text:
        return None
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    action = str(obj.get("action", "")).strip().upper()
    if action not in ("SELL", "HOLD"):
        return None
    try:
        conf = float(obj.get("confidence", 0))
    except (TypeError, ValueError):
        return None
    conf = max(0.0, min(1.0, conf))
    reasons = obj.get("reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {"action": action, "confidence": conf, "reasons": [str(r)[:200] for r in reasons[:6]]}


def ask_ai(sym, tech, earnings, cfg):
    """(parsed|None, raw_text|err). Tries every configured provider that has a key."""
    _load_dotenv()
    prompt = (
        "You are a risk guard for a live stock account. A held stock is down sharply today. "
        "Decide whether to SELL now or HOLD for the scheduled 2:30 PM CT rebalance.\n"
        "Reply with JSON ONLY, no markdown fences:\n"
        '{"action":"SELL"|"HOLD","confidence":0.0-1.0,"reasons":["short bullet",...]}\n'
        "Sell only if technical damage looks persistent (breakdown vs MAs, weak RSI/MACD, "
        "failed support, heavy volume, bad relative strength) and/or earnings look clearly bad. "
        "Prefer HOLD when the drop looks like noise or the stock still holds major MAs/support.\n\n"
        f"Symbol: {sym}\n"
        f"Technical snapshot: {json.dumps(tech, default=str)[:3500]}\n"
        f"Earnings (if any): {json.dumps(earnings, default=str) if earnings else 'none in lookback'}\n"
    )
    last_err = "no LLM key (need GEMINI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY / XAI_API_KEY)"
    any_key = False
    for env, model, provider in LLM_PROVIDERS:
        key = os.getenv(env)
        if not key:
            continue
        any_key = True
        if provider == "gemini":
            models = [model] + [m for m in GEMINI_FALLBACK_MODELS if m != model]
        elif provider == "groq":
            models = [model] + [m for m in GROQ_FALLBACK_MODELS if m != model]
        else:
            models = [model]
        for m in models:
            try:
                text = _llm_complete(provider, m, key, prompt)
                parsed = _parse_ai_json(text)
                if parsed:
                    parsed["provider"] = provider
                    parsed["model"] = m
                    return parsed, text
                last_err = f"unparseable reply from {provider}/{m}"
            except Exception as e:
                code = getattr(e, "code", None)
                last_err = f"{provider}/{m}: {type(e).__name__}: {str(e)[:120]}"
                # 404 / model_not_found: try next fallback model; else next provider
                if code in (404, 400) and m != models[-1]:
                    continue
                break  # try next provider
    if not any_key:
        return None, last_err
    return None, last_err


# ----------------------------------------------------------------------------- trigger + candidates
def prev_close_map(bars, day):
    """{symbol: previous regular close before `day`}."""
    day = pd.Timestamp(day).normalize()
    out = {}
    if bars is None or bars.empty:
        return out
    for sym, g in bars.groupby("Symbol"):
        g = g[g["Date"] < day].sort_values("Date")
        if not g.empty and math.isfinite(float(g["Close"].iloc[-1])) and float(g["Close"].iloc[-1]) > 0:
            out[sym] = float(g["Close"].iloc[-1])
    return out


def quote_mid(q, now):
    """(mid, bid, ask, skip_reason)."""
    import earnings_stop as es
    bid, ask, ts, _ = q
    mid, skip = es.quote_check(q, "regular", now)
    return mid, bid, ask, skip


def held_candidates(now, cfg, acct=None, earnings=None):
    """Held names (not do_not_sell) with prev close. Returns (rows, held_dict)."""
    import holdings
    acct = acct or account()
    held = holdings.held_qty(account=acct)
    keep = {str(x).upper() for x in sm.do_not_sell}
    syms = sorted(s for s, q in held.items() if q > 0 and s not in keep)
    if not syms:
        return [], held
    day = pd.Timestamp(now.astimezone(be.EASTERN).date())
    bars = daily_bars(syms + ["QQQ"], now)
    prev = prev_close_map(bars, day)
    earnings = be.load_earnings(EARNINGS_CSV) if earnings is None else earnings
    rows = []
    for s in syms:
        if s not in prev:
            continue
        rows.append({
            "symbol": s, "shares": float(held[s]), "prev": prev[s],
            "trigger": prev[s] * (1 - float(cfg["drop_pct"])),
            "earnings": recent_earnings(s, day, earnings, cfg.get("earnings_lookback_days", 5)),
            "bars": bars[bars["Symbol"] == s].sort_values("Date").set_index("Date")
                    if not bars.empty else pd.DataFrame(),
        })
    # benchmark day returns (vs their own prev closes) for RS
    bench = {}
    for b in ("QQQ",):
        if b in prev:
            bench[b] = prev[b]
    return rows, held, bars, prev, bench


# ----------------------------------------------------------------------------- replacement
def latest_ranking():
    """DataFrame Symbol, Strategy_Rank, Strategy_Score, Close from latest signal_analysis day."""
    sig = db.read_csv(SIGNAL_CSV)
    if sig is None or sig.empty:
        return pd.DataFrame()
    sig = sig.copy()
    sig["Date"] = pd.to_datetime(sig["Date"], errors="coerce")
    day = sig["Date"].max()
    sig = sig[sig["Date"] == day].copy()
    for c in ("Strategy_Rank", "Strategy_Score", "Close", "ma_200"):
        if c in sig.columns:
            sig[c] = pd.to_numeric(sig[c], errors="coerce")
    return sig.sort_values("Strategy_Rank")


def pick_replacement(sold_sym, proceeds, equity, held, sold_today, cfg, now, price_map):
    """Top-ranked eligible symbol to buy with `proceeds`, or (None, reason)."""
    rank = latest_ranking()
    if rank.empty:
        return None, "no signal_analysis ranking"
    do_not = {str(x).upper() for x in sm.do_not_buy}
    try:
        import sold_block
        do_not |= set(sold_block.active_blocks())
    except Exception:
        pass
    max_w = float(cfg.get("max_weight", 0.15))
    drop = float(cfg["drop_pct"])
    day = pd.Timestamp(now.astimezone(be.EASTERN).date())
    # need SMA200 + prev closes for candidates — use ranking Close + ma_200 when present, else bars
    for _, r in rank.iterrows():
        sym = str(r["Symbol"]).upper()
        if sym == sold_sym or sym in held or sym in sold_today or sym in do_not:
            continue
        if sym in ("SPY", "QQQ", "IWM") or sym.endswith("USD"):
            continue
        score = r.get("Strategy_Score", np.nan)
        if pd.notna(score) and float(score) <= 0:
            continue
        px = price_map.get(sym)
        try:
            px = float(px) if px is not None else None
        except (TypeError, ValueError):
            px = None
        if not px or not math.isfinite(px) or px <= 0:
            # fall back to signal close
            px = float(r["Close"]) if pd.notna(r.get("Close")) else None
        if not px or px <= 0:
            continue
        # under 200-day MA?
        ma200 = float(r["ma_200"]) if "ma_200" in r and pd.notna(r.get("ma_200")) else None
        if ma200 and px < ma200:
            continue
        # already held weight + new buy must stay <= max_w of equity
        cur_val = float(held.get(sym, 0)) * px
        room = max_w * equity - cur_val
        if room <= 1:
            continue
        budget = min(proceeds, room)
        qty = math.floor((budget / px) * 100) / 100  # 2-decimal shares like live
        if qty * px < 1:
            continue
        return {"symbol": sym, "qty": qty, "price": px, "rank": int(r["Strategy_Rank"])
                if pd.notna(r.get("Strategy_Rank")) else None,
                "ma200": ma200, "budget": budget}, "ok"
    return None, "no eligible replacement"


def _cid(kind, day, sym, attempt=1):
    d = pd.Timestamp(day).strftime("%Y%m%d")
    base = f"live-dguard-{kind}-{d}-{sym}"
    return base if attempt <= 1 else f"{base}-r{attempt}"


# ----------------------------------------------------------------------------- orders
def _submit_limit(client, sym, qty, side, limit, cid, extended=False):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest
    req = LimitOrderRequest(
        symbol=sym, qty=qty,
        side=OrderSide.SELL if side == "SELL" else OrderSide.BUY,
        time_in_force=TimeInForce.DAY, limit_price=limit,
        extended_hours=extended, client_order_id=cid,
    )
    return client.submit_order(req)


def _wait_fill(client, order_id, timeout_s=90, poll_s=3):
    """Poll until filled/partial enough or timeout. Returns (status, filled_qty, filled_avg_price)."""
    import paper_trade as pt
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            o = client.get_order_by_id(order_id)
        except Exception:
            time.sleep(poll_s)
            continue
        last = o
        st = getattr(o.status, "value", str(o.status)).lower()
        filled = pt._safe_number(getattr(o, "filled_qty", 0))
        avg = pt._safe_number(getattr(o, "filled_avg_price", 0))
        if st in ("filled", "done_for_day") or (filled > 0 and st in ("partially_filled", "canceled", "expired")):
            return st, filled, avg
        if st in ("canceled", "expired", "rejected"):
            return st, filled, avg
        time.sleep(poll_s)
    if last is None:
        return "timeout", 0.0, 0.0
    return (getattr(last.status, "value", str(last.status)).lower(),
            pt._safe_number(getattr(last, "filled_qty", 0)),
            pt._safe_number(getattr(last, "filled_avg_price", 0)))


def execute_sell_and_replace(c, mid, bid, ask, tech, ai, cfg, state, now, dry_run, equity, held, bars_all):
    """Sell the candidate; on fill, try a replacement buy. Returns log-field dict."""
    import paper_trade as pt
    import run_all
    s = c["symbol"]
    day = now.astimezone(be.CENTRAL).date()
    out = {
        "Sell_Order_Id": "", "Sell_Limit": None, "Sell_Filled_Qty": 0.0, "Sell_Proceeds": 0.0,
        "Replace_Symbol": "", "Replace_Order_Id": "", "Replace_Limit": None, "Replace_Qty": 0.0,
        "Replace_Note": "", "Action": "HOLD",
    }
    if dry_run or not cfg.get("enabled", True):
        whole = math.floor(c["shares"] + 1e-9)
        limit = pt._tick(bid * (1 - pt.LIMIT_OFFSET), False) if bid else None
        out["Action"] = "WOULD_SELL"
        out["Sell_Limit"] = limit
        out["Note_extra"] = f"dry-run/disabled would sell {whole:g} @ {limit}"
        # still propose replacement for the log
        proceeds = (limit or mid) * whole
        price_map = {}
        pick, why = pick_replacement(s, proceeds, equity, held, state.get("sold_today", {}), cfg, now, price_map)
        if pick:
            out["Replace_Symbol"] = pick["symbol"]
            out["Replace_Qty"] = pick["qty"]
            out["Replace_Limit"] = pt._tick(pick["price"] * (1 + pt.LIMIT_OFFSET), True)
            out["Replace_Note"] = f"would buy rank {pick['rank']} (dry-run)"
        else:
            out["Replace_Note"] = f"no replacement: {why}"
        return out

    # Live path: share the fill-check lock pattern used by earnings_stop (avoid stacking with 2:30)
    if _lock_busy(run_all.TRADE_LOCK):
        out["Action"] = "SKIP"
        out["Replace_Note"] = "trade job running — skipped"
        return out
    ok, why = run_all.acquire_trade_lock(run_all.FILL_LOCK)
    if not ok:
        out["Action"] = "SKIP"
        out["Replace_Note"] = f"fill-check lock busy ({why})"
        return out
    try:
        client = trading_client()
        broker = pt._broker_orders_by_client_id(client)
        if broker is None:
            out["Action"] = "SKIP"
            out["Replace_Note"] = "broker orders unreadable"
            return out
        attempt = 1
        while (o := broker.get(_cid("sell", day, s, attempt))) is not None:
            st = getattr(o.status, "value", str(o.status)).lower()
            if st != "rejected" or pt._safe_number(getattr(o, "filled_qty", 0)) > 0:
                out["Action"] = "ALREADY"
                out["Sell_Order_Id"] = str(o.id)
                out["Sell_Filled_Qty"] = pt._safe_number(getattr(o, "filled_qty", 0))
                return out
            attempt += 1
            if attempt > 3:
                out["Action"] = "SKIP"
                out["Replace_Note"] = "sell rejected 3 times"
                return out
        # Free any target_sells limit so we can sell at the bid
        try:
            import target_sells as ts
            oid, note = ts.cancel_for_symbol(client, s, reason="drop_guard sell")
            if oid:
                out["Replace_Note"] = (out.get("Replace_Note") or "") + f"canceled target {oid}; "
                time.sleep(0.5)  # let Alpaca free qty_available before we read / submit
        except Exception as e:
            out["Replace_Note"] = (out.get("Replace_Note") or "") + f"target cancel err: {e}; "
        held_now, open_now = pt._symbol_state(client, s)
        if held_now is None:
            out["Action"] = "SKIP"
            out["Replace_Note"] = "position unreadable"
            return out
        # Ignore remaining target cids (should be gone); other open orders still block
        try:
            import target_sells as ts
            open_now = [o for o in (open_now or [])
                        if not ts.is_target_cid(getattr(o, "client_order_id", "") or "")]
        except Exception:
            pass
        if open_now:
            out["Action"] = "SKIP"
            out["Replace_Note"] = "open order already working"
            return out
        if held_now <= 0:
            out["Action"] = "SKIP"
            out["Replace_Note"] = "no longer held"
            return out
        whole = math.floor(held_now + 1e-9)
        qty = held_now if whole < 1 else whole
        if qty <= 0:
            out["Action"] = "SKIP"
            out["Replace_Note"] = "qty 0"
            return out
        limit = pt._tick(bid * (1 - pt.LIMIT_OFFSET), False)
        cid = _cid("sell", day, s, attempt)
        try:
            order = _submit_limit(client, s, qty, "SELL", limit, cid, extended=False)
        except Exception as e:
            out["Action"] = "SKIP"
            out["Replace_Note"] = f"sell submit failed: {str(e)[:100]}"
            return out
        out["Sell_Order_Id"] = str(order.id)
        out["Sell_Limit"] = limit
        out["Action"] = "SELL_SENT"
        st, filled, avg = _wait_fill(client, order.id, timeout_s=90)
        out["Sell_Filled_Qty"] = filled
        proceeds = filled * (avg if avg else limit)
        out["Sell_Proceeds"] = round(proceeds, 2)
        # Max one sell attempt per symbol per day (even if still working / rejected).
        state.setdefault("sold_today", {})[s] = {
            "at": now.astimezone(be.CENTRAL).isoformat(timespec="seconds"),
            "order_id": str(order.id), "qty": filled, "limit": limit, "status": st,
        }
        _save_state(state)
        # append pending for morning fill check (fractional rest / unfilled working order)
        try:
            frac = round(held_now - qty, 6)
            if frac > 0 or filled + 1e-9 < qty:
                _append_pending({
                    "symbol": s, "side": "SELL", "qty": held_now, "limit_price": limit,
                    "order_id": str(order.id), "order_qty": qty, "exit": True,
                    "recorded_at": now.astimezone(be.CENTRAL).isoformat(timespec="seconds"),
                    "decision": None, "evening_date": str(day), "source": "drop-guard",
                }, pt.PENDING_ORDERS_JSON)
        except Exception:
            pass
        working = st in ("new", "accepted", "pending_new", "partially_filled", "held")
        # sold_block only when shares actually left (filled) or a working sell is still live.
        # Rejected/canceled/expired with 0 fill must NOT block rebuy or drop a target limit.
        if filled > 0 or working:
            try:
                import sold_block
                sold_block.record_sell(s, sold_on=day, source="drop_guard",
                                       note=f"filled {filled:g} @ {limit} ({st})")
            except Exception:
                pass
        # Remove target config only once the position has actually been sold.
        # If the bid sell is still working / failed, leave the target entry so the
        # next target_sells tick can re-place the high limit (or remove when flat).
        if filled > 0:
            try:
                import target_sells as ts
                ts.remove_target(s, reason="drop_guard sold")
            except Exception:
                pass
            try:
                _mirror_stop_block(s, c.get("earnings"), now)
            except Exception:
                pass
        if filled <= 0:
            out["Replace_Note"] = (out.get("Replace_Note") or "") + f"sell not filled ({st}) — no replacement"
            log_event("ok", "unknown",
                      f"Drop guard: sold signal on {s} but order not filled ({st}). Limit ${limit:,.2f}.",
                      f"order {order.id}")
            return out
        out["Action"] = "SOLD"
        log_event("ok", "yes",
                  f"Drop guard: sold {filled:g} {s} at ~${avg or limit:,.2f} "
                  f"({((avg or limit) / c['prev'] - 1):.1%} vs prev close ${c['prev']:,.2f}). "
                  f"AI confidence {ai['confidence']:.2f}.",
                  f"order {order.id}; reasons={ai.get('reasons')}")
        # Refresh holdings after sell
        held2 = dict(held)
        held2[s] = max(0.0, held_now - filled)
        # Build price map with live quotes for top candidates (first 30 ranks)
        rank = latest_ranking()
        price_map = {}
        candidates = []
        if not rank.empty:
            for _, r in rank.head(40).iterrows():
                candidates.append(str(r["Symbol"]).upper())
        # filter quickly then quote
        for sym in candidates[:25]:
            if sym == s or sym in held2 or sym in state["sold_today"] or sym in sm.do_not_buy:
                continue
            try:
                q = latest_quote(sym)
                m, b, a, skip = quote_mid(q, now)
                if m is None:
                    continue
                # prev close from bars_all
                prev = None
                if bars_all is not None and not bars_all.empty:
                    g = bars_all[bars_all["Symbol"] == sym]
                    g = g[g["Date"] < pd.Timestamp(day)].sort_values("Date")
                    if not g.empty:
                        prev = float(g["Close"].iloc[-1])
                if prev and m <= prev * (1 - float(cfg["drop_pct"])):
                    continue  # candidate itself down >= drop
                price_map[sym] = m
            except Exception:
                continue
        pick, why = pick_replacement(s, proceeds, equity, held2, state["sold_today"], cfg, now, price_map)
        if not pick:
            out["Replace_Note"] = f"no replacement: {why} — cash for 2:30"
            return out
        # re-check live quote for limit
        try:
            q = latest_quote(pick["symbol"])
            m, b, a, skip = quote_mid(q, now)
        except Exception as e:
            out["Replace_Note"] = f"replacement quote failed: {e}"
            return out
        if m is None or a is None:
            out["Replace_Note"] = f"replacement quote bad ({skip}) — cash for 2:30"
            return out
        buy_limit = pt._tick(a * (1 + pt.LIMIT_OFFSET), True)
        buy_qty = math.floor((min(proceeds, pick["budget"]) / buy_limit) * 100) / 100
        if buy_qty * buy_limit < 1:
            out["Replace_Note"] = "replacement qty too small — cash for 2:30"
            return out
        # final down-5% check
        prev_c = None
        g = bars_all[bars_all["Symbol"] == pick["symbol"]] if bars_all is not None else pd.DataFrame()
        g = g[g["Date"] < pd.Timestamp(day)].sort_values("Date") if not g.empty else g
        if not g.empty:
            prev_c = float(g["Close"].iloc[-1])
            if m <= prev_c * (1 - float(cfg["drop_pct"])):
                out["Replace_Note"] = f"{pick['symbol']} itself down >= drop — cash for 2:30"
                out["Replace_Symbol"] = pick["symbol"]
                return out
        bcid = _cid("buy", day, pick["symbol"])
        if bcid in (broker or {}):
            out["Replace_Note"] = "replacement already submitted"
            out["Replace_Symbol"] = pick["symbol"]
            return out
        try:
            border = _submit_limit(client, pick["symbol"], buy_qty, "BUY", buy_limit, bcid, extended=False)
        except Exception as e:
            out["Replace_Note"] = f"buy submit failed: {str(e)[:100]} — cash for 2:30"
            out["Replace_Symbol"] = pick["symbol"]
            return out
        out["Replace_Symbol"] = pick["symbol"]
        out["Replace_Order_Id"] = str(border.id)
        out["Replace_Limit"] = buy_limit
        out["Replace_Qty"] = buy_qty
        out["Replace_Note"] = f"bought rank {pick['rank']} with sale proceeds"
        out["Action"] = "SOLD_REPLACED"
        log_event("ok", "yes",
                  f"Drop guard: replaced {s} → {pick['symbol']} ({buy_qty:g} @ limit ${buy_limit:,.2f}) "
                  f"using sale proceeds ${proceeds:,.2f}.",
                  f"buy order {border.id}")
        try:
            _append_pending({
                "symbol": pick["symbol"], "side": "BUY", "qty": buy_qty, "limit_price": buy_limit,
                "order_id": str(border.id), "order_qty": buy_qty, "exit": False,
                "recorded_at": now.astimezone(be.CENTRAL).isoformat(timespec="seconds"),
                "decision": None, "evening_date": str(day), "source": "drop-guard-replace",
            }, pt.PENDING_ORDERS_JSON)
        except Exception:
            pass
        return out
    finally:
        run_all.release_trade_lock(run_all.FILL_LOCK)


def _append_pending(row, path):
    try:
        with open(path) as f:
            pend = json.load(f)
    except FileNotFoundError:
        pend = None
    if not isinstance(pend, dict) or not pend.get("orders"):
        t = datetime.fromisoformat(row["recorded_at"])
        day = pd.Timestamp(t.date()) - be.NYSE_SESSION if t.hour < 9 else pd.Timestamp(t.date())
        pend = {"evening_date": f"{day:%Y-%m-%d}", "submitted_at_ct": row["recorded_at"],
                "target_source": "auto", "as_of": None, "orders": []}
    pend["orders"] = [o for o in pend["orders"]
                      if not (o.get("symbol") == row["symbol"] and o.get("source") == row.get("source"))] + [row]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(pend, f, indent=2)
    os.replace(tmp, path)


def _mirror_stop_block(sym, earnings_facts, now):
    """When the sale was on/near earnings, record in earnings_stop_state so live won't buy it back today."""
    if not earnings_facts:
        return
    path = os.environ.get("STOCK_ANALYSIS_EARNINGS_STOP_STATE") or os.path.join(REPORTS, "earnings_stop_state.json")
    try:
        with open(path) as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {"sold": {}, "notes": {}, "armed": {}}
    st.setdefault("sold", {})
    e = earnings_facts.get("earnings_date")
    if not e:
        return
    key = f"{sym}|{e}"
    if key in st["sold"]:
        return
    # react = last watch day approx today
    st["sold"][key] = {
        "at": now.astimezone(be.CENTRAL).isoformat(timespec="seconds"),
        "order_id": None, "react": str(now.astimezone(be.CENTRAL).date()),
        "note": "drop-guard sale",
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- logging
def append_log(row):
    full = {c: row.get(c, "") for c in LOG_COLS}
    db.append_rows(LOG_CSV, [[full[c] for c in LOG_COLS]], LOG_COLS)


def backfill_outcomes(now=None):
    """Fill Close_Price / Next_Close for past SELL rows. Returns number updated."""
    now = now or datetime.now(be.CENTRAL)
    if not db.exists(LOG_CSV):
        print("no drop_guard_log yet")
        return 0
    log = db.read_csv(LOG_CSV)
    if log is None or log.empty:
        return 0
    need = log["Action"].astype(str).str.contains("SOLD|WOULD_SELL", na=False)
    need &= (log["Close_Price"].isna() | (log["Close_Price"].astype(str) == "") |
             log["Next_Close"].isna() | (log["Next_Close"].astype(str) == ""))
    if not need.any():
        print("nothing to backfill")
        return 0
    syms = sorted(log.loc[need, "Symbol"].astype(str).unique())
    bars = be.fetch_daily_bars(syms, start=(now - timedelta(days=14)).date())
    updated = 0
    for i in log.index[need]:
        sym = str(log.at[i, "Symbol"])
        try:
            checked = pd.Timestamp(str(log.at[i, "Checked_At_CT"])[:10])
        except Exception:
            continue
        g = bars[bars["Symbol"] == sym].sort_values("Date") if bars is not None else pd.DataFrame()
        if g.empty:
            continue
        sell_px = pd.to_numeric(log.at[i, "Sell_Limit"], errors="coerce")
        day_bar = g[g["Date"] == checked]
        if not day_bar.empty and (pd.isna(log.at[i, "Close_Price"]) or str(log.at[i, "Close_Price"]) == ""):
            close = float(day_bar["Close"].iloc[0])
            log.at[i, "Close_Price"] = close
            if sell_px and sell_px == sell_px and sell_px > 0:
                log.at[i, "Close_Vs_Sell_%"] = round((close / float(sell_px) - 1) * 100, 3)
            updated += 1
        nxt = g[g["Date"] > checked]
        if not nxt.empty and (pd.isna(log.at[i, "Next_Close"]) or str(log.at[i, "Next_Close"]) == ""):
            # only if next day already complete
            n = nxt.iloc[0]
            if pd.Timestamp(n["Date"]).date() < now.date() or (now.hour * 60 + now.minute) >= 15 * 60 + 5:
                close = float(n["Close"])
                log.at[i, "Next_Close"] = close
                if sell_px and sell_px == sell_px and sell_px > 0:
                    log.at[i, "Next_Vs_Sell_%"] = round((close / float(sell_px) - 1) * 100, 3)
                updated += 1
    if updated:
        db.to_csv(log[LOG_COLS] if set(LOG_COLS) <= set(log.columns) else log, LOG_CSV, index=False)
    print(f"backfilled {updated} fields")
    return updated


# ----------------------------------------------------------------------------- main check
def run_check(now=None, dry_run=None, cfg=None):
    """One pass over held positions. Returns list of log rows."""
    cfg = cfg or load_config()
    now = now or datetime.now(be.CENTRAL)
    if now.tzinfo is None:
        now = now.replace(tzinfo=be.CENTRAL)
    now_ct = now.astimezone(be.CENTRAL)
    if dry_run is None:
        dry_run = bool(cfg.get("dry_run", False))
    ok, why = in_window(now_ct, cfg)
    # Always honor the CT window (including dry_run / --scheduled). --as-of still
    # goes through in_window against that clock, so only in-window as-of times evaluate.
    # Prevents overnight dry_run launchd ticks from calling the LLM on a coincidental trigger.
    if not ok:
        print(f"{now_ct:%Y-%m-%d %H:%M:%S} CT: idle — {why}")
        return []
    if not cfg.get("enabled", True) and not dry_run:
        # still evaluate when explicitly scheduled? kill switch: evaluate+log, never order
        dry_run = True
        print(f"{now_ct:%Y-%m-%d %H:%M:%S} CT: kill switch off — dry evaluation only")

    got_lock, lock_why = _acquire_lock()
    if not got_lock:
        print(f"{now_ct:%Y-%m-%d %H:%M:%S} CT: idle — lock {lock_why}")
        return []
    rows_out = []
    try:
        state = _load_state()
        today = _ensure_day(state)
        try:
            pack = held_candidates(now_ct, cfg)
        except Exception as e:
            log_event("failed", "no", f"Drop guard could not read holdings/bars ({str(e)[:150]}).")
            return []
        cands, held, bars_all, prev_map, bench = pack
        # equity for weight cap
        try:
            snap = account().account_summary()
            equity = float(snap.get("Equity") or snap.get("equity") or 0) or 1.0
        except Exception:
            equity = 1.0
        # QQQ day return so far (for RS) — use quote if possible
        qqq_ret = None
        try:
            if "QQQ" in prev_map:
                qm, *_ = quote_mid(latest_quote("QQQ"), now_ct)
                if qm:
                    qqq_ret = qm / prev_map["QQQ"] - 1
        except Exception:
            pass

        # minute bars for triggered names only (fetched lazily)
        minute_cache = {}
        day = now_ct.astimezone(be.EASTERN).date()
        stamp = now_ct.strftime("%Y-%m-%d %H:%M:%S")
        provider, model, key = llm_provider()
        if not key and cfg.get("llm_required_for_live") and not dry_run:
            print("No LLM key — refusing live sells (set dry_run or add GEMINI_API_KEY).")
            dry_run = True

        for c in cands:
            s = c["symbol"]
            mid = bid = ask = None
            skip = None
            try:
                q = latest_quote(s)
                mid, bid, ask, skip = quote_mid(q, now_ct)
            except Exception as e:
                skip = f"no quote ({str(e)[:60]})"
            vs = ((mid / c["prev"] - 1) * 100) if mid and c["prev"] else float("nan")
            triggered = bool(mid is not None and mid <= c["trigger"])
            base = {
                "Checked_At_CT": stamp, "Symbol": s, "Shares": c["shares"],
                "Prev_Close": round(c["prev"], 4), "Price": round(mid, 4) if mid else "",
                "Vs_Prev_%": round(vs, 3) if math.isfinite(vs) else "",
                "Triggered": "yes" if triggered else "no",
                "AI_Action": "", "AI_Confidence": "", "AI_Reasons": "",
                "Action": "WATCH", "Sell_Order_Id": "", "Sell_Limit": "",
                "Sell_Filled_Qty": "", "Sell_Proceeds": "",
                "Replace_Symbol": "", "Replace_Order_Id": "", "Replace_Limit": "",
                "Replace_Qty": "", "Replace_Note": "",
                "Close_Price": "", "Close_Vs_Sell_%": "", "Next_Close": "", "Next_Vs_Sell_%": "",
                "Dry_Run": "yes" if dry_run else "no",
                "Note": skip or "",
            }
            if s in state.get("sold_today", {}):
                base["Action"] = "ALREADY_SOLD_TODAY"
                base["Note"] = f"sold earlier {state['sold_today'][s].get('at','')}"
                append_log(base)
                rows_out.append(base)
                continue
            if not triggered:
                append_log(base)
                rows_out.append(base)
                continue
            if skip and mid is None:
                base["Action"] = "SKIP_QUOTE"
                append_log(base)
                rows_out.append(base)
                continue

            # sector RS
            sector_ret = None
            try:
                etf = sm.sector_etf_for(s)
                if etf and etf in prev_map:
                    em, *_ = quote_mid(latest_quote(etf), now_ct)
                    if em:
                        sector_ret = em / prev_map[etf] - 1
                elif etf:
                    # fetch prev for etf
                    eb = daily_bars([etf], now_ct, lookback=40)
                    pm = prev_close_map(eb, pd.Timestamp(day))
                    if etf in pm:
                        em, *_ = quote_mid(latest_quote(etf), now_ct)
                        if em:
                            sector_ret = em / pm[etf] - 1
            except Exception:
                pass

            if s not in minute_cache:
                try:
                    minute_cache[s] = minute_bars([s], day)
                except Exception:
                    minute_cache[s] = pd.DataFrame()
            mins = minute_cache[s]
            mins_s = mins[mins["Symbol"] == s] if not mins.empty and "Symbol" in mins.columns else mins
            tech = technical_snapshot(c["bars"], mins_s, mid, c["prev"], cfg, qqq_ret, sector_ret)
            ai, raw = ask_ai(s, tech, c.get("earnings"), cfg)
            if not ai:
                base["Action"] = "SKIP_AI"
                base["Note"] = (base["Note"] + "; " if base["Note"] else "") + f"AI unavailable: {raw}"
                append_log(base)
                rows_out.append(base)
                continue
            base["AI_Action"] = ai["action"]
            base["AI_Confidence"] = ai["confidence"]
            base["AI_Reasons"] = " | ".join(ai.get("reasons") or [])[:500]
            sell_ok = ai["action"] == "SELL" and ai["confidence"] >= float(cfg["min_confidence"])
            if not sell_ok:
                base["Action"] = "HOLD_AI"
                append_log(base)
                rows_out.append(base)
                continue

            result = execute_sell_and_replace(
                c, mid, bid, ask, tech, ai, cfg, state, now_ct, dry_run, equity, held, bars_all)
            base["Action"] = result.get("Action", "HOLD")
            for k in ("Sell_Order_Id", "Sell_Limit", "Sell_Filled_Qty", "Sell_Proceeds",
                      "Replace_Symbol", "Replace_Order_Id", "Replace_Limit", "Replace_Qty", "Replace_Note"):
                if result.get(k) not in (None, ""):
                    base[k] = result[k]
            if result.get("Note_extra"):
                base["Note"] = (base["Note"] + "; " if base["Note"] else "") + result["Note_extra"]
            append_log(base)
            rows_out.append(base)
            _save_state(state)

        _save_state(state)
        n_trig = sum(1 for r in rows_out if r.get("Triggered") == "yes")
        n_sell = sum(1 for r in rows_out if str(r.get("Action", "")).startswith("SOLD") or r.get("Action") == "WOULD_SELL")
        print(f"{stamp} CT: checked {len(rows_out)} held; triggered {n_trig}; sell actions {n_sell}; "
              f"dry_run={dry_run}; enabled={cfg.get('enabled')}")
        return rows_out
    finally:
        _release_lock()


def ensure_config():
    if not os.path.exists(CONFIG_JSON):
        save_config(dict(DEFAULT_CONFIG))
        print(f"wrote {CONFIG_JSON}")
    return load_config()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Intraday drop guard (LIVE)")
    ap.add_argument("--dry-run", action="store_true", help="evaluate + log; never submit orders")
    ap.add_argument("--scheduled", action="store_true", help="one live check (launchd)")
    ap.add_argument("--as-of", help='dry-run as of CT/ET time, e.g. "2026-10-30 11:00"')
    ap.add_argument("--backfill-outcomes", action="store_true")
    ap.add_argument("--write-config", action="store_true", help="write default config if missing")
    args = ap.parse_args(argv)
    cfg = ensure_config()
    if args.write_config:
        return 0
    if args.backfill_outcomes:
        backfill_outcomes()
        return 0
    now = None
    if args.as_of:
        raw = pd.Timestamp(args.as_of)
        now = raw.tz_localize(be.CENTRAL) if raw.tzinfo is None else raw
        args.dry_run = True
    dry = True if args.dry_run else None
    if args.scheduled and not args.dry_run:
        dry = None  # honor config
    run_check(now=now, dry_run=dry, cfg=cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
