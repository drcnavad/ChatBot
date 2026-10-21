"""The news band's read-aloud voice: Voice Genie's own voice (Kokoro, af_heart, speed 1.0, the model files in
~/Voice Genie/models; same settings via VOICE_GENIE_VOICE / VOICE_GENIE_SPEED), rendered here on the Mac mini, so
every browser hears the same voice. Each headline is rendered once in the background (one thread), saved as AAC
(macOS afconvert) in dashboard/news_voice/audio/<hash>.m4a and served by the news_voice component's file handler;
a headline not rendered yet is read by the browser's own voice instead. Display only: nothing here touches trading."""
import hashlib
import re
import os
import subprocess
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

HERE = Path(__file__).with_name("news_voice")
AUDIO = HERE / "audio"
GENIE = Path(os.getenv("VOICE_GENIE_DIR", Path.home() / "Voice Genie"))
VOICE, SPEED = os.getenv("VOICE_GENIE_VOICE", "af_heart"), float(os.getenv("VOICE_GENIE_SPEED", "1.0"))
SOURCES = {"CNBC": "CNBC", "YFinance": "Yahoo Finance", "Google News": "Google News", "Nasdaq": "Nasdaq", "MarketWatch": "MarketWatch", "News": "the newswires"}
KEEP_DAYS = 2
_component = components.declare_component("news_voice", path=str(HERE))

MONTHS = dict(Jan="January", Feb="February", Mar="March", Apr="April", Jun="June", Jul="July", Aug="August",
              Sep="September", Sept="September", Oct="October", Nov="November", Dec="December")
QUARTER = {"1": "first", "2": "second", "3": "third", "4": "fourth"}   # Q3 -> third-quarter
TERMS = [(r"\bEPS\b", "earnings per share"), (r"\bYoY\b", "year over year"), (r"\bQoQ\b", "quarter over quarter"),
         (r"\bvs\.?(?=\s)", "versus"), (r"\bCEO\b", "C.E.O."), (r"\bCFO\b", "C.F.O."), (r"\bAI\b", "A.I."),
         (r"\bIPO\b", "I.P.O."), (r"\bFDA\b", "F.D.A."), (r"\bFSD\b", "full self-driving"), (r"\bY/Y\b", "year over year"),
         (r"\bFY\s?(\d{2,4})\b", r"fiscal \1"), (r"\bH([12])\b", lambda m: ("first" if m[1] == "1" else "second") + " half")]
ACRONYMS = {"CNBC", "NASA", "NATO", "OPEC", "NYSE", "SEC", "GDP", "CPI", "PPI", "ETF", "USA", "FOMC", "AMD", "IBM", "NBA", "NFL"}

@lru_cache(maxsize=1)
def _names():
    """{ticker: company name as said} from sector_mapping.py (no Inc. / Corp.), e.g. TSLA -> Tesla."""
    try:
        import sector_mapping as sm
        clean = lambda n: re.sub(r"^The |,? (Inc\.?|Corporation|Corp\.?|Company|Co\.?|Incorporated|Ltd\.?|plc|N\.V\.)\b.*$|"
                                 r" (Holdings?|Group|Platforms|Technologies|Systems|Networks)$", "", n).strip()
        out = {s: clean(sm.symbol_name.get(s, s)) for s in sm.stock_symbols}
        out.update(GOOGL="Alphabet", NVDA="Nvidia", META="Meta", AMZN="Amazon")
        return out
    except Exception:
        return {}

def script(tag, title):
    """The headline as a reporter would say it: a lead-in ("From CNBC:" for market news, the company for a stock's,
    left out when the headline already names it), tickers as company names, no " - Source" suffix, months, quarters
    and abbreviations spelled out, ALL-CAPS words in normal case. Rule based, local; money and percents are spelled
    out later by Voice Genie's speakable()."""
    names, t = _names(), (title or "").strip()
    t = re.sub(r"\s+[-|–—]\s+[A-Z][\w.&' ]{1,40}$", "", t)                       # "... - CNBC"
    t = re.sub(r"\(\s*(?:NASDAQ|NYSE)?:?\s*\$?([A-Z]{1,5})\s*\)", lambda m: "" if m[1] in names else m[0], t)
    t = re.sub(r"\$([A-Z]{1,5})\b", lambda m: names.get(m[1], m[1]), t)
    t = re.sub(r"\b([A-Z]{2,5})\b", lambda m: names.get(m[1], m[1]) if len(m[1]) > 2 else m[1], t)
    t = re.sub(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?(?=\s+\d)", lambda m: MONTHS[m[1]], t)
    t = re.sub(r"\b[Qq]([1-4])\b", lambda m: f"{QUARTER[m[1]]}-quarter", t)
    t = re.sub(r",? (Inc|Corp|Co|Ltd)\.?(?=[\s,:;]|$)", "", t)                       # "Meta Platforms, Inc."
    t = re.sub(r"\s*\((?:Live|Video|Update[sd]?|Podcast|Live Coverage|Live Updates)[^)]*\)\s*$", "", t, flags=re.I)
    for rx, rep in TERMS:
        t = re.sub(rx, rep, t)
    t = re.sub(r"\b([A-Z][A-Z']{3,})\b", lambda m: m[1] if m[1] in ACRONYMS else m[1].capitalize(), t)
    t = re.sub(r"\b(\w+)( \1\b)+", r"\1", t, flags=re.I)                       # "Tesla Tesla" -> "Tesla"
    t = re.sub(r"\s{2,}", " ", t).strip(" :-")
    if t and t[-1] not in ".!?":
        t += "."
    if tag in SOURCES:
        return f"From {SOURCES[tag]}: {t}"
    name = names.get(tag, tag)
    if not tag or name.split()[0].lower() in t.lower():                            # already named: no lead-in
        return t
    return f"{name}: {t}"

def key(text):
    return hashlib.sha1(f"{VOICE}|{SPEED}|{text}".encode()).hexdigest()[:16]

def _speakable(text):
    """Voice Genie's text clean-up (money, percents, tickers as words) when it is there, else the text as is."""
    try:
        if str(GENIE) not in sys.path:
            sys.path.append(str(GENIE))
        from speech_text import speakable
        return speakable(text)
    except Exception:
        return text

@lru_cache(maxsize=1)
def _kokoro():
    from kokoro_onnx import Kokoro
    return Kokoro(str(GENIE / "models" / "kokoro-v1.0.onnx"), str(GENIE / "models" / "voices-v1.0.bin"))

def _render(text, out):
    if out.exists():
        return
    import numpy as np
    samples, rate = _kokoro().create(_speakable(text), voice=VOICE, speed=SPEED, lang="en-us")
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
        tmp = out.with_suffix(".part.m4a")
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "64000", wav, str(tmp)], check=True, capture_output=True)
        tmp.replace(out)                      # appears only when complete
    finally:
        os.remove(wav)

@st.cache_resource(show_spinner=False)
def _worker():
    return ThreadPoolExecutor(max_workers=1), set()

def _safe(text, out, pending):
    try:
        _render(text, out)
    except Exception:
        pass                                  # that headline falls back to the browser voice
    finally:
        pending.discard(out.name)

def prepare(texts):
    """Queue every headline not rendered yet; prune audio older than KEEP_DAYS. Returns {text: key} of the headlines
    whose audio is ready (the others are read by the browser voice until a later rerun)."""
    if not (GENIE / "models" / "kokoro-v1.0.onnx").exists():
        return {}
    AUDIO.mkdir(parents=True, exist_ok=True)
    pool, pending = _worker()
    for f in AUDIO.glob("*.m4a"):
        if time.time() - f.stat().st_mtime > KEEP_DAYS * 86400:
            f.unlink(missing_ok=True)
    keys = {}
    for t in texts:
        k = key(t)
        out = AUDIO / f"{k}.m4a"
        if out.exists():
            keys[t] = k
        elif out.name not in pending:
            pending.add(out.name)
            pool.submit(_safe, t, out, pending)
    return keys

def render_voice():
    """The speaker button's controller (index.html): plays the rendered audio, else the browser voice."""
    _component(key="news_voice", default=None)
