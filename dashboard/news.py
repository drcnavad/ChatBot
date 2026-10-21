"""Breaking-news banner under the ticker strip: one headline at a time, fading to the next every 6 s (CSS only),
paused on hover, click opens the article. Display only and separate from the bot: free public RSS feeds (no API keys,
nothing from the trading accounts), read with the standard library, cached 5 minutes, silent when a feed is down."""
import email.utils
import html
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import streamlit as st

from dashboard.style import show_html

FEEDS = [("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
         ("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^IXIC,^DJI&region=US&lang=en-US"),
         ("Google News", "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB"
                         "?hl=en-US&gl=US&ceid=US:en")]
STOCK_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"   # free, keyless
MAX_ITEMS, PER_STOCK, SECONDS = 30, 2, 6
NY = ZoneInfo("America/New_York")

def _one_line(p):
    return re.sub(r"\n\s*", "", p)

# Only headlines that report an event: a headline matching DROP (clickbait, questions, comparisons, lists, buy/sell
# advice, predictions, promo) is left out; the rest needs an EVENT word, or a wire / newsroom source (TRUSTED).
DROP = re.compile(_one_line(r"""\?\s*$|\bvs\.?\b|\bversus\b|\bcompare[sd]?\b|how (low|high|far) can|should you|\bis it (time|too late)\b
    |\bis (it|this|now) a\b|\b(a|the) (buy|sell)\b|\bbuy (now|today|before|and hold)\b|\b(top|best|worst) \d+\b|^\d+ (stocks?|reasons?|things?|ways?)
    |\bstocks? to (buy|watch|own|sell)\b|prediction|predict|could (soar|double|triple|make you|turn)|\bmillionaire\b|here'?s (why|what|how)
    |\bwhy (i|you|we)\b|motley fool|zacks|\bmy (top|favorite)\b|\bopinion\b|\bwatchlist\b|\bsponsored\b|\bpromo\b|\bunder the radar\b
    |\bno-brainer\b|\bsupercharge|\bforever\b|\bretire\b|\bpassive income\b|\bdividend (king|aristocrat)s?\b|\bdon'?t miss\b|\bbargain\b
    |\bsnapshot\b|^(how|why|forget|here are|what)\b|\bto watch\b|\bcalendar\b|stock reports?\b|\bfavorite\b|warning signs?|\bproblem with\b
    |\bcagr\b|market report|\bthese \d+\b|\b\d+ (\w+ )?stocks\b|deep dive|what it means|\bhow it\b|\bbullish plays\b|\bwhat to (expect|know)\b|\bthings to know\b|\bpreview\b|\bconference call\b|\bto participate in\b|\binvestor (day|conference)\b"""), re.I)
EVENT = re.compile(_one_line(r"""earnings|results|revenue|profit|loss|quarter|\bq[1-4]\b|guidance|outlook|forecast|beats?\b|miss(es|ed)?\b|tops?\b
    |acquir|acquisition|merger|merge|deal\b|buyout|takeover|stake|spin.?off|ipo\b|listing|upgrade|downgrade|price target|initiat|\brating\b
    |fda|approv|recall|lawsuit|sues?\b|settle|fine[sd]?\b|penalt|probe|investigat|antitrust|regulat|ban(s|ned)?\b|tariff|sanction
    |launch|unveil|introduc|announce|ceo|cfo|chief|resign|steps? down|appoint|names? .* (ceo|cfo|chair)|layoff|job cuts|cuts? \d|plant|strike
    |\bfed\b|federal reserve|powell|rate (cut|hike)|interest rate|inflation|cpi|ppi|jobs report|payrolls|unemployment|gdp|retail sales|treasury|yields?
    |dow|s&p|nasdaq|stocks? (rise|fall|jump|drop|slide|surge|rally|sink|climb|tumble)|record|plunge|soar|surge|tumble|slump|rall(y|ies)|selloff
    |\bto buy\b|\$?\d[\d.,]*\s?(billion|bn|million)|debut|certif|partnership|expands?\b|contract|\bfaa\b|clears?\b|wins?\b|agrees?\b|files? for
    |split|buyback|dividend|bankrupt|default|downgrad|outage|hack|breach|halt"""), re.I)
TRUSTED = {"CNBC", "Reuters", "AP", "Associated Press", "BBC", "Bloomberg", "The Wall Street Journal", "Financial Times", "CNN", "WSJ", "Axios", "MarketWatch"}

def is_news(title, source=""):
    """True for a headline that reports an event (see DROP / EVENT / TRUSTED)."""
    t = (title or "").strip()
    if not t or DROP.search(t):
        return False
    return bool(EVENT.search(t)) or source in TRUSTED

def _off():
    return os.getenv("STOCK_ANALYSIS_NEWS", "on") == "off" or os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off"

def _read(source, url, ticker=""):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (StockAnalysis dashboard)"})
    with urllib.request.urlopen(req, timeout=6) as r:
        root = ET.fromstring(r.read())
    out = []
    for it in root.iter("item"):
        title, link, when = ((it.findtext(k) or "").strip() for k in ("title", "link", "pubDate"))
        try:
            t = email.utils.parsedate_to_datetime(when).astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        src = source
        if source == "Google News" and " - " in title:          # "Headline - Publisher"
            title, src = title.rsplit(" - ", 1)
        if title and link:
            out.append({"title": html.unescape(title), "link": link, "source": src, "time": t, "ticker": ticker})
    return out

def _safe(args):
    try:
        return _read(*args)
    except Exception:
        return []                                              # a feed that is down is skipped silently

@st.cache_data(ttl=600, show_spinner=False)
def headlines():
    """The market feeds plus Yahoo Finance headlines for every stock in sector_mapping.py (tagged with the ticker,
    newest PER_STOCK each), read in parallel. Newest first, a headline once; today's (New York) when there are any,
    else the last 24 h."""
    from concurrent.futures import ThreadPoolExecutor
    try:
        import sector_mapping as sm
        syms = list(dict.fromkeys(sm.stock_symbols))
    except Exception:
        syms = []
    jobs = [(src, url) for src, url in FEEDS] + [("Yahoo Finance", STOCK_FEED.format(sym=s), s) for s in syms]
    items = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for job, got in zip(jobs, pool.map(_safe, jobs)):
            items += [i for i in got if is_news(i["title"], i["source"])][:PER_STOCK] if len(job) == 3 else got
    return _recent(items, 24)[0][:MAX_ITEMS]

def _recent(items, fallback_hours):
    """(news headlines newest first, each once; today's in New York, else the last fallback_hours, label)."""
    seen, uniq = set(), []
    for it in sorted(items, key=lambda x: x["time"], reverse=True):
        key = "".join(c for c in it["title"].lower() if c.isalnum())[:80]
        if key not in seen and is_news(it["title"], it["source"]):
            seen.add(key)
            uniq.append(it)
    now = datetime.now(timezone.utc)
    today = [i for i in uniq if i["time"].astimezone(NY).date() == now.astimezone(NY).date()]
    if today:
        return today, "today"
    return [i for i in uniq if now - i["time"] <= timedelta(hours=fallback_hours)], f"last {fallback_hours} hours"

def _ago(t):
    m = int((datetime.now(timezone.utc) - t).total_seconds() // 60)
    return "just now" if m < 1 else f"{m}m ago" if m < 60 else f"{m // 60}h ago"

def render_news_banner(box):
    if _off():
        return
    try:
        items = headlines()
    except Exception:
        return
    if not items:
        return
    n, total = len(items), len(items) * SECONDS
    show = 100 / n                                             # each headline's share of the cycle, in %
    fade = min(1.5, show / 4)
    rows = "".join(
        f'<a class="sa-news-item" href="{html.escape(i["link"], quote=True)}" target="_blank" rel="noopener" '
        f'style="animation-delay:{k * SECONDS}s"><span class="sa-news-src">{html.escape(i["source"])}</span>'
        + (f'<span class="sa-news-sym">{html.escape(i["ticker"])}</span>' if i["ticker"] else "")
        + f'<span class="sa-news-t">{html.escape(i["title"])}</span><span class="sa-news-ago">{_ago(i["time"])}</span></a>'
        for k, i in enumerate(items))
    css = (f'<style>.sa-news .sa-news-item{{animation:sa-news {total}s linear infinite}}'
           f'@keyframes sa-news{{0%{{opacity:0;visibility:visible}}{fade:.2f}%{{opacity:1}}{show - fade:.2f}%{{opacity:1}}'
           f'{show:.2f}%{{opacity:0;visibility:hidden}}100%{{opacity:0;visibility:hidden}}}}</style>')
    with box:
        show_html(css + f'<div class="sa-news"><span class="sa-news-tag">LIVE NEWS</span><div class="sa-news-box">{rows}</div></div>')

@st.cache_data(ttl=600, show_spinner=False)
def stock_news(sym, limit=5):
    """(items, label) for one stock from its Yahoo Finance feed: today's (New York) headlines, else the last 48 h."""
    items, label = _recent(_safe(("Yahoo Finance", STOCK_FEED.format(sym=sym), sym)), 48)
    return items[:limit], label

def render_stock_news(sym):
    """A compact list of the stock's headlines (links open in a new tab); nothing when there are none."""
    if _off():
        return
    try:
        items, label = stock_news(sym)
    except Exception:
        return
    if not items:
        return
    rows = "".join(f'<a href="{html.escape(i["link"], quote=True)}" target="_blank" rel="noopener">'
                   f'<span class="t">{html.escape(i["title"])}</span><span class="s">{html.escape(i["source"])}</span>'
                   f'<span class="a">{_ago(i["time"])}</span></a>' for i in items)
    show_html(f'<div class="sa-snews"><div class="sa-snews-h">{html.escape(sym)} news <span>· {label}</span></div>{rows}</div>')
