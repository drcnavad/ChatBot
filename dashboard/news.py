"""Breaking-news banner under the ticker strip: three headlines at a time (one per row), fading to the next three every 15 s (CSS only),
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

from dashboard.news_voice import prepare, render_voice, script
from dashboard.style import show_html

FEEDS = [("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
         ("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^IXIC,^DJI&region=US&lang=en-US"),
         ("Google News", "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB"
                         "?hl=en-US&gl=US&ceid=US:en")]
STOCK_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"   # free, keyless
FEEDS += [("Nasdaq", "https://www.nasdaq.com/feed/rssoutbound?category=Markets"),                  # free, keyless
          ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
          ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_marketpulse")]
NASDAQ_STOCK_FEED = "https://www.nasdaq.com/feed/rssoutbound?symbol={sym}"   # mixes in market stories: tagged by title
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129 Safari/537.36"
MAX_ITEMS, PER_STOCK, PER_PAGE, SECONDS = 30, 2, 3, 15   # banner: 3 headline rows at a time, 15 s each
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

TRANSCRIPT = re.compile(r"\btranscripts?\b|\bearnings call\b|\bconference call\b", re.I)
FILLER = re.compile(r"^\s*(live updates?|live|live blog|live coverage|watch live|breaking)\s*[:\-–—|]\s*"
                    r"|\s*[:\-–—|(]\s*(live updates?|live blog|live coverage)\s*\)?\s*$", re.I)

QUESTION = re.compile(r"\?|^(why|how|what|is|are|should|can|will|could|does|do|which)\b", re.I)   # questions are not news

def clean_title(title):
    """The headline without filler such as "Live updates:" / "LIVE:" / "(Live blog)"."""
    t = (title or "").strip()
    for _ in range(2):
        t = FILLER.sub("", t).strip()
    return t

def is_news(title, source=""):
    """True for a headline that reports an event (see DROP / EVENT / TRUSTED)."""
    t = clean_title(title)
    q = re.sub(r"\s+[-|–—]\s+[A-Z][\w.&' ]{1,40}$", "", t)          # "... - Reuters" before the question check
    if not t or DROP.search(t) or TRANSCRIPT.search(t) or QUESTION.search(q):   # transcripts, questions: not news
        return False
    return bool(EVENT.search(t)) or source in TRUSTED

SHORT = {"Yahoo Finance": "YFinance"}                          # banner tag of a market headline with no stock in it
GENERIC = {"Advanced", "Applied", "Texas", "Live", "Royal", "Interactive", "Steel", "Martin", "Super", "Bloom", "Monster",
           "Delta", "Arm", "Lam", "Palo", "Cadence", "Quanta", "Credo", "Nu", "Upstart", "Johnson", "Williams", "Parker",
           "Devon", "Celestica", "Morgan", "Invesco", "GE", "APA"}
ALIAS = {"GOOGL": ["Google", "GOOG"], "NVDA": ["Nvidia"], "GS": ["Goldman"], "C": ["Citi"], "JPM": ["JPMorgan"], "DIS": ["Disney"],
         "KO": ["Coca-Cola"], "UNH": ["UnitedHealth"], "AMD": ["AMD"], "META": ["Facebook"], "MS": ["Morgan Stanley"]}
SUFFIX = re.compile(r"^The |,? (Inc\.?|Corporation|Corp\.?|Company|Co\.?|Incorporated|Ltd\.?|plc|N\.V\.|Holdings?|Group)\b.*$")

@st.cache_resource(show_spinner=False)
def _matcher():
    """(regex, word -> ticker) that finds a sector_mapping.py stock in a headline: its name (no Inc./Corp.), the
    name's first word unless generic, a few common aliases, or the symbol itself ($X, (X), or a bare 4+ letter one)."""
    import sector_mapping as sm
    words = {}
    for sym in dict.fromkeys(sm.stock_symbols):
        if sym in sm.BENCHMARK_SYMBOLS:
            continue
        name = SUFFIX.sub("", sm.symbol_name.get(sym, "")).strip()
        first = name.split()[0].rstrip(",") if name else ""
        for w in [name, first if first not in GENERIC and len(first) > 2 else "", *ALIAS.get(sym, [])]:
            if w:
                words.setdefault(w.lower(), sym)
        for w in [f"${sym}", f"({sym})"] + ([sym] if len(sym) > 3 else []):   # 2-3 letters are often words (NOW, ARM)
            words.setdefault(w.lower(), sym)
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![\w$])({alts})(?![\w-])", re.I), words

def _named(title):
    """The sector_mapping.py stock a headline names, else None."""
    try:
        rx, words = _matcher()
        m = rx.search(title)
        if m and (m.group(1)[0] in "$(" or m.group(1)[0].isupper()):   # a name, not a lowercase word
            return words[m.group(1).lower()]
    except Exception:
        pass
    return None

def tag(item):
    """The amber box before a headline: its stock (feed ticker, else one named in the title), else the feed's short name."""
    if item.get("ticker"):
        return "GOOGL" if item["ticker"] == "GOOG" else item["ticker"]        # one Alphabet tag
    named = _named(item["title"])
    if named:
        return named
    feed = item.get("feed") or item.get("source", "")
    return SHORT.get(feed, feed) or "News"

def _off():
    return os.getenv("STOCK_ANALYSIS_NEWS", "on") == "off" or os.getenv("STOCK_ANALYSIS_LIVE_HOLDINGS", "on") == "off"

def _read(source, url, ticker=""):
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA if "nasdaq.com" in url else "Mozilla/5.0 (StockAnalysis dashboard)"})
    with urllib.request.urlopen(req, timeout=4) as r:
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
        if source == "Nasdaq" and ticker and _named(html.unescape(title)) != ticker:
            ticker_ = ""                                       # a market story in the stock's feed
        else:
            ticker_ = ticker
        if title and link:
            out.append({"title": clean_title(html.unescape(title)), "link": link, "source": src, "feed": source, "time": t, "ticker": ticker_})
    return out

def _safe(args):
    try:
        return _read(*args)
    except Exception:
        return []                                              # a feed that is down is skipped silently

_HEADLINES_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "headlines_cache.json")
_HEADLINES_TTL = 300  # seconds; matches st.cache_data ttl — disk survives a Streamlit restart


def _headlines_from_disk():
    try:
        import json, time as _t
        with open(_HEADLINES_CACHE) as f:
            blob = json.load(f)
        if _t.time() - float(blob.get("saved_at", 0)) > _HEADLINES_TTL:
            return None
        items = blob.get("items") or []
        for i in items:
            i["time"] = datetime.fromisoformat(i["time"])
        return items
    except Exception:
        return None


def _headlines_to_disk(items):
    try:
        import json, time as _t
        os.makedirs(os.path.dirname(_HEADLINES_CACHE), exist_ok=True)
        payload = []
        for i in items:
            d = dict(i)
            d["time"] = i["time"].isoformat()
            payload.append(d)
        tmp = _HEADLINES_CACHE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"saved_at": _t.time(), "items": payload}, f)
        os.replace(tmp, _HEADLINES_CACHE)
    except Exception:
        pass


@st.cache_data(ttl=300, show_spinner=False)
def headlines():
    """Market RSS feeds only (CNBC, Yahoo, Google News, Nasdaq, MarketWatch), read in parallel. Stock names in a
    title are tagged via _named (sector_mapping). Per-stock Yahoo/Nasdaq feeds live in stock_news() for the Stock
    Analysis tab — fetching them here for every symbol was ~200 HTTP calls and ~11 s on a cold page. Newest first,
    a headline once; today's (New York) when there are any, else the last 24 h. Disk-cached 5 min so a Streamlit
    restart does not wait on the feeds again."""
    cached = _headlines_from_disk()
    if cached is not None:
        return cached
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(src, url) for src, url in FEEDS]
    with ThreadPoolExecutor(max_workers=8) as pool:
        items = [i for got in pool.map(_safe, jobs) for i in got]
    out = _recent(items, 24)[0][:MAX_ITEMS]
    _headlines_to_disk(out)
    return out

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

def _news_link(i, said):
    """One headline link (banner and Latest News): ticker tag, title, source · age; data-say / data-text = what
    Voice Genie reads (the summarized script and its pre-rendered audio)."""
    text = script(tag(i), i["title"])
    return (f'<a class="sa-news-item" href="{html.escape(i["link"], quote=True)}" target="_blank" rel="noopener" '
            f'data-say="{said.get(text, "")}" data-text="{html.escape(text, quote=True)}">'
            f'<span class="sa-news-sym">{html.escape(tag(i))}</span><span class="sa-news-t">{html.escape(i["title"])}</span>'
            f'<span class="sa-news-ago">{html.escape(i["source"])} · {_ago(i["time"])}</span></a>')

def render_news_banner(box):
    if _off():
        return
    try:
        items = headlines()
    except Exception:
        return
    if not items:
        return
    pages = [items[k:k + PER_PAGE] for k in range(0, len(items), PER_PAGE)]
    n, total = len(pages), len(pages) * SECONDS
    show = 100 / n                                             # each page's share of the cycle, in %
    fade = min(1.5, show / 4)
    said = prepare([script(tag(i), i["title"]) for i in items])          # the read-aloud audio, rendered in the background
    item = lambda i: _news_link(i, said)
    rows = "".join(f'<div class="sa-news-page" style="animation-delay:{k * SECONDS}s">{"".join(map(item, pg))}</div>'
                   for k, pg in enumerate(pages))
    css = (f'<style>.sa-news .sa-news-page{{animation:sa-news {total}s linear infinite}}'
           f'@keyframes sa-news{{0%{{opacity:0;visibility:visible}}{fade:.2f}%{{opacity:1}}{show - fade:.2f}%{{opacity:1}}'
           f'{show:.2f}%{{opacity:0;visibility:hidden}}100%{{opacity:0;visibility:hidden}}}}</style>')
    with box:
        show_html(css + f'<div class="sa-news"><span class="sa-news-tag">LIVE NEWS</span><div class="sa-news-box">{rows}</div></div>')
        render_voice()
        render_latest_news()

# ---- "Latest News": a static list of the 10 most trading-relevant headlines (same feed / filters / format as the banner)
LATEST_N = 10
MAJOR = re.compile(_one_line(r"""earnings|results|guidance|outlook|forecast|beats?\b|miss(es|ed)?\b|acquir|acquisition|merger|buyout
    |takeover|upgrade|downgrade|price target|fda|approv|recall|lawsuit|probe|investigat|antitrust|tariff|sanction|resign|steps? down
    |layoff|job cuts|\bfed\b|federal reserve|powell|rate (cut|hike)|interest rate|inflation|cpi|ppi|jobs report|payrolls|gdp
    |bankrupt|default|halt|buyback|split|plunge|soar|surge|tumble|selloff"""), re.I)

def importance(item):
    """0-3: +2 when it names one of your stocks (feed ticker or a sector_mapping.py name in the title), +1 for a
    market-moving event (earnings / guidance, M&A, analyst rating, FDA / legal, Fed / macro data, big moves)."""
    return (2 if item.get("ticker") or _named(item["title"]) else 0) + (1 if MAJOR.search(item["title"]) else 0)

def _key(i):
    return i["link"] or "".join(c for c in i["title"].lower() if c.isalnum())[:80]

@st.fragment(run_every=300)      # checks the feeds every 5 min; only this list reruns (the banner keeps scrolling)
def render_latest_news():
    """The 10 most important headlines of the banner's feed (importance, then newest). Kept stable in the session: a new
    headline that makes the top 10 goes on top and the oldest drops; known ones keep their place."""
    if _off():
        return
    try:
        items = headlines()
    except Exception:
        items = []
    top = sorted(items, key=lambda i: (importance(i), i["time"]), reverse=True)[:LATEST_N]
    keys = {_key(i) for i in top}
    prev = st.session_state.get("_latest_news", [])
    kept = [i for i in prev if _key(i) in keys]
    old = {_key(i) for i in kept}
    new = sorted((i for i in top if _key(i) not in old), key=lambda i: i["time"], reverse=True)
    shown = (new + kept)[:LATEST_N] if prev else sorted(top, key=lambda i: i["time"], reverse=True)
    st.session_state["_latest_news"] = shown
    if not shown:
        return
    said = prepare([script(tag(i), i["title"]) for i in shown])       # the same spoken text and Voice Genie audio as the band
    row = lambda i: _news_link(i, said)
    show_html('<div class="sa-news sa-latest"><span class="sa-news-tag">LATEST NEWS</span>'
              f'<div class="sa-latest-box">{"".join(map(row, shown))}</div></div>'
              '<style>.sa-latest .sa-latest-box{display:flex;flex-direction:column;gap:4px;flex:1;min-width:0}.sa-latest .sa-news-tag:before{animation:none}'
              '.sa-latest{align-items:flex-start;margin-top:6px}.sa-latest .sa-news-item{position:static;opacity:1;visibility:visible}</style>')


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
