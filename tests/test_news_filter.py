"""The news banner / stock news keep only headlines that report an event (dashboard.news.is_news): earnings, guidance,
deals, analyst actions, regulatory / legal, launches, executive changes, layoffs, macro data, big market moves; and
drop clickbait, questions, comparisons, lists, buy / sell advice, predictions and promo. No network.
Run: python tests/run_tests.py  (or PYTHONPATH=. python tests/test_news_filter.py)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard.news import is_news, tag  # noqa: E402

KEEP = [("Meta posts Q3 revenue beat and strong guidance: Live updates", "CNBC"),
        ("Tesla beats on earnings and sales even as auto revenue declines", "CNBC"),
        ("Texas Instruments Tops Q3 Targets, Guides Above Q4 Views", "Yahoo Finance"),
        ("Morgan Stanley upgrades Nvidia to overweight, raises price target to $250", "Yahoo Finance"),
        ("FDA approves Eli Lilly's new obesity drug", "Yahoo Finance"),
        ("Pfizer to acquire Metsera in $7.3 billion deal", "Yahoo Finance"),
        ("Intel CEO steps down effective immediately", "Yahoo Finance"),
        ("Amazon announces 14,000 layoffs in corporate workforce", "Yahoo Finance"),
        ("Fed cuts interest rates by a quarter point", "Yahoo Finance"),
        ("CPI rises 0.3% in September as inflation stays sticky", "Yahoo Finance"),
        ("Dow jumps 500 points as bank stocks rally", "Yahoo Finance"),
        ("Meta ordered to pay $25 billion penalty in New Mexico case", "CNBC"),
        ("Apple unveils new iPhone lineup", "Yahoo Finance"),
        ("Salesforce to Buy Customer-Engagement Platform for Nearly $5.7 Billion", "WSJ"),
        ("Boeing's largest 737 MAX jet wins FAA certification, clearing way for deliveries", "Yahoo Finance")]
DROP = [("How low can Super Micro stock go?", "Yahoo Finance"),
        ("AAPL vs SMCI: How They Compare", "Yahoo Finance"),
        ("Is Nvidia a buy right now?", "Yahoo Finance"),
        ("Should you buy Tesla stock before earnings", "Yahoo Finance"),
        ("3 Top AI Stocks to Buy in October", "Yahoo Finance"),
        ("Top 10 dividend stocks for passive income", "Yahoo Finance"),
        ("Prediction: This stock will be worth more than Apple by 2030", "Yahoo Finance"),
        ("Here's why Amazon stock could soar", "Yahoo Finance"),
        ("1 Growth Stock Down 40% to Buy Now", "Yahoo Finance"),
        ("Motley Fool: the no-brainer stock to own forever", "Yahoo Finance"),
        ("Meta Platforms: Q3 Earnings Snapshot", "Yahoo Finance"),
        ("Martin Marietta Announces Third-Quarter 2026 Earnings Conference Call", "Yahoo Finance"),
        ("ManpowerGroup to Participate in J.P. Morgan Ultimate Services Investor Conference", "Yahoo Finance"),
        ("Bank of America (BAC): Is Its Wealth Business More Valuable Than the Headline Bank Suggests?", "Yahoo Finance"),
        ("Savvy Investors Love This Often Overlooked Clue", "Yahoo Finance"),
        ("Earnings To Watch: Omnicom Group (OMC) Reports Q3 Results Tomorrow", "Yahoo Finance"),
        ("Forget Flashy Stocks: These 3 Boring Businesses Have Decades of Dividend Payments", "Yahoo Finance"),
        ("Top Stock Reports for Procter & Gamble, Sandisk & Analog Devices", "Yahoo Finance"),
        ("Analyst Rating Boosts May Signal More Upside for These 3 Stocks", "Yahoo Finance"),
        ("HAL Q3 2026 Deep Dive: International Growth and Margin Expansion Shape Outlook", "Yahoo Finance"),
        ("How WMT, TGT & BJ Are Betting Big on AI-Powered Shopping", "Yahoo Finance")]

bad = [(t, "should keep") for t, s in KEEP if not is_news(t, s)] + [(t, "should drop") for t, s in DROP if is_news(t, s)]
for t, why in bad:
    print("FAIL", why, ":", t)
print(f"{len(KEEP) + len(DROP) - len(bad)}/{len(KEEP) + len(DROP)} headlines classified right")
assert not bad

# Banner tag: the feed's ticker, else a stock named in the title, else the feed's short name
TAGS = [({"title": "Fed holds rates", "feed": "CNBC", "ticker": "AAPL"}, "AAPL"),
        ({"title": "Nvidia shares jump on chip deal", "feed": "CNBC", "ticker": ""}, "NVDA"),
        ({"title": "Walt Disney names new CEO", "feed": "Google News", "ticker": ""}, "DIS"),
        ({"title": "Coca-Cola earnings top estimates", "feed": "Yahoo Finance", "ticker": ""}, "KO"),
        ({"title": "Fed holds rates steady", "feed": "Yahoo Finance", "ticker": ""}, "YFinance"),
        ({"title": "Stocks slide as yields climb, now at a record", "feed": "CNBC", "ticker": ""}, "CNBC"),
        ({"title": "Treasury yields rise", "feed": "Google News", "ticker": ""}, "Google News")]
bad = [(i["title"], tag(i), want) for i, want in TAGS if tag(i) != want]
for b in bad:
    print("FAIL tag:", b)
assert not bad
from dashboard.news_voice import key, script  # noqa: E402
SAID = [("CNBC", "Fed holds rates - Reuters", "From CNBC: Fed holds rates."),
        ("YFinance", "Stock Market Today, Oct. 21: SpaceX Surges on Q3 Earnings Beat",
         "From Yahoo Finance: Stock Market Today, October 21: SpaceX Surges on third-quarter Earnings Beat."),
        ("TSLA", "TESLA (TSLA) EPS beats by $0.05, revenue up 12% YoY - Reuters",
         "Tesla earnings per share beats by $0.05, revenue up 12% year over year."),
        ("TSLA", "Tesla Earnings Mixed, FSD Revenue Strong (Live Coverage)", "Tesla Earnings Mixed, full self-driving Revenue Strong."),
        ("NVDA", "Shares jump after $NVDA chip deal", "Shares jump after Nvidia chip deal."),
        ("AAPL", "Shares slip in late trade", "Apple: Shares slip in late trade.")]
bad = [(t, script(g, t), want) for g, t, want in SAID if script(g, t) != want]
for b_ in bad:
    print("FAIL spoken:", b_)
assert not bad and key("a") == key("a") != key("b")
print("NEWS FILTER OK")
