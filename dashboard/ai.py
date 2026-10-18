"""Optional analysis (only runs when the button is pressed).

News bullets use a small summarizer that runs on this machine (sshleifer/distilbart-cnn-12-6).
The stock paragraph is written from the saved scores. Neither calls the Hugging Face inference API,
which now returns payment required for the old Llama chat model."""
import re

import streamlit as st

from dashboard.data import load_news
from dashboard.style import MA_COLS

NEWS_MODEL = "sshleifer/distilbart-cnn-12-6"   # local weights; the first summary downloads them, later runs stay on disk

@st.cache_resource(show_spinner=False)
def news_summarizer():
    """DistilBART loaded once per dashboard session."""
    from transformers import pipeline
    return pipeline("summarization", model=NEWS_MODEL, framework="pt")

def trend_deltas_text(ticker_df, windows=(14, 50, 200)):
    """Compact multi-window trend text for the stock paragraph."""
    recent = ticker_df.sort_values("Date")
    latest = recent.iloc[-1]
    text = "Trend Deltas:\n"
    for w in windows:
        if len(recent) < w:
            continue
        past, tail = recent.iloc[-w], recent.tail(w)
        text += (f"Last {w} days: Price {(latest['Close'] / past['Close'] - 1) * 100:.2f}%, "
                 f"RSI change {latest['RSI'] - past['RSI']:.2f}, MACD change {latest['macd'] - past['macd']:.2f}, "
                 f"Price vs MA30 {(latest['Close'] / latest['ma_30'] - 1) * 100:.2f}%, "
                 f"Price vs MA200 {(latest['Close'] / latest['ma_200'] - 1) * 100:.2f}%, "
                 f"Above MA200 {(tail['Close'] > tail['ma_200']).mean() * 100:.2f}% of days\n")
    return text

def _tone(value, high, mid):
    return "strong" if value > high else "mixed" if value >= mid else "weak"

def ai_stock_summary(ticker, ticker_df, signal, why):
    """A short read of one ticker from the saved scores. No model call."""
    latest = ticker_df.nlargest(1, "Date").iloc[0]
    price = float(latest["Close"])
    above = sum(price > float(latest[ma]) for ma in MA_COLS)
    trend = trend_deltas_text(ticker_df).splitlines()
    recent = trend[1] if len(trend) > 1 else ""
    why_bit = f" ({why})" if why else ""
    rec = {"Buy": "BULLISH", "Hold": "HOLD", "Sold": "BEARISH"}.get(signal, "HOLD")
    return (f"{ticker} is {signal}{why_bit} on the weekly strategy. "
            f"The close is above {above} of the 5 moving averages. "
            f"Technicals look {_tone(float(latest['Technical_Score']), 40, 0)}, "
            f"the balance sheet looks {_tone(float(latest['Fundamental_Weight']), 5, 2)}, "
            f"and the saved news tone looks {_tone(float(latest['SentimentScore']), 0, 0)}. "
            f"{recent}\n\n"
            f"Recommendation: {rec}. This follows the strategy signal already in force, read with the trend and those scores.")

def news_bullets(text):
    """2-4 bullets from a summarizer paragraph."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return "\n".join(f"- {s}" for s in parts[:4])

def ai_news_summary(news, sentiment_type, symbol, max_articles=20):
    """Bullet summary of the positive or negative news, from the local DistilBART model."""
    news = news.sort_values("date", ascending=False).head(max_articles)
    text = f"{symbol} {sentiment_type} news. " + " ".join(f"{r.headline}. {r.summary}" for r in news.itertuples())
    if len(text.split()) < 40:
        return "\n".join(f"- {r.headline}" for r in news.head(4).itertuples())
    try:
        out = news_summarizer()(text[:6000], max_length=120, min_length=30, truncation=True, do_sample=False)
    except Exception:
        return "\n".join(f"- {r.headline}" for r in news.head(4).itertuples())
    return news_bullets(out[0]["summary_text"])

@st.dialog("AI Analysis", width="large")
def ai_analysis_dialog(ticker, ticker_df, signal, why):
    """Pop-up: AI technical summary plus positive / negative news summaries."""
    with st.spinner(f"Generating AI summary for {ticker}..."):
        summary = ai_stock_summary(ticker, ticker_df, signal, why)
    st.markdown(f"### {ticker}")
    st.markdown(summary)
    st.divider()
    news = load_news()
    symbol_news = news[news['symbol'] == ticker]
    if symbol_news.empty:
        st.info(f"No news articles found for {ticker}")
        return
    for col, label in zip(st.columns(2), ("positive", "negative")):
        subset = symbol_news[symbol_news['sentiment_label'] == label]
        with col:
            st.markdown(f"**{label.capitalize()} news**")
            if subset.empty:
                st.caption("None found")
                continue
            with st.spinner(f"Summarizing {label} headlines..."):
                st.markdown(ai_news_summary(subset, label, ticker))
