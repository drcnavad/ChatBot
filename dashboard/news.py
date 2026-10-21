"""Breaking-news banner under the ticker strip: one headline at a time, fading to the next every 6 s (CSS only),
paused on hover, click opens the article. Display only and separate from the bot: free public RSS feeds (no API keys,
nothing from the trading accounts), read with the standard library, cached 5 minutes, silent when a feed is down."""
import email.utils
import html
import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import streamlit as st

from dashboard.style import show_html

FEEDS = [("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
         ("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^IXIC,^DJI&region=US&lang=en-US"),
         ("Google News", "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB"
                         "?hl=en-US&gl=US&ceid=US:en")]
STOCK_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"   # free, keyless
MAX_ITEMS, PER_STOCK, SECONDS = 30, 2, 6

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
            items += got[:PER_STOCK] if len(job) == 3 else got
    seen, uniq = set(), []
    for it in sorted(items, key=lambda x: x["time"], reverse=True):
        key = "".join(c for c in it["title"].lower() if c.isalnum())[:80]
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    now = datetime.now(timezone.utc)
    today = [i for i in uniq if i["time"].astimezone(timezone(timedelta(hours=-4))).date()
             == now.astimezone(timezone(timedelta(hours=-4))).date()]
    return (today or [i for i in uniq if now - i["time"] <= timedelta(hours=24)])[:MAX_ITEMS]

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
        show_html(css + f'<div class="sa-news"><span class="sa-news-tag">NEWS</span><div class="sa-news-box">{rows}</div></div>')
