"""Optional AI analysis (Hugging Face; only runs when the button is pressed)."""
import os
import re

import streamlit as st

from dashboard.data import load_news
from dashboard.style import MA_COLS


LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@st.cache_resource(show_spinner=False)
def hf_token():
    """HF token from Streamlit secrets, falling back to .env (cached: st.secrets lookups are slow)."""
    try:
        return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN", "")
    except Exception:
        return os.getenv("HF_TOKEN", "")


def llm_chat(messages, max_tokens):
    """Run a Llama chat completion and strip trailing prompt artifacts."""
    try:
        from huggingface_hub import InferenceClient   # imported on first use
        response = InferenceClient(token=hf_token()).chat_completion(
            model=LLM_MODEL, messages=messages, max_tokens=max_tokens, temperature=0.2)
        return re.split(r'\[/?USER\]|Can you|Could you', response.choices[0].message.content.strip())[0].strip()
    except Exception as e:
        return f"Error generating summary: {e}"


def trend_deltas_text(ticker_df, windows=(14, 50, 200)):
    """Compact multi-window trend text for the LLM prompt."""
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


def ai_stock_summary(ticker, ticker_df, signal, why):
    """AI summary + recommendation for one ticker."""
    latest = ticker_df.nlargest(1, 'Date').iloc[0]
    price = latest['Close']
    ma_lines = "\n".join(f"Price - {ma.upper().replace('_', '')}: ${price - latest[ma]:.2f} ({(price / latest[ma] - 1) * 100:.2f}%)"
                         for ma in MA_COLS)
    context = (f"Stock: {ticker}\nDate: {latest['Date']:%Y-%m-%d}\nCurrent Price: ${price:.2f}\n"
               f"Weekly strategy signal: {signal} ({why})\nStrategy Rank: {latest.get('Strategy_Rank')}\n"
               f"Technical Score: {latest['Technical_Score']:.2f}\n"
               f"Relative Strength Score: {latest.get('RS_Score', float('nan')):.2f}\n"
               f"Strategy Score (0.5 technical + 0.5 relative strength): {latest['combined_signal']:.2f}\n"
               f"RSI: {latest['RSI']:.2f}\nMACD: {latest['macd']:.2f}\n\n"
               f"Price vs Moving Averages (Difference):\n{ma_lines}\n\n"
               f"Balance Sheet Score: {latest['Fundamental_Weight']:.2f}\nSentiment Score: {latest['SentimentScore']:.2f}\n\n"
               f"{trend_deltas_text(ticker_df)}")
    system = ("You are a financial advisor. Provide ONLY a concise summary (4-5 sentences) followed by a clear AI recommendation. "
              "DO NOT list individual metrics, scores, or numbers in your response. "
              "DO NOT mention specific values like 'Balance Sheet Score: X', 'News Sentiment Score: Y', or 'RSI: Z'. "
              "Instead, synthesize all the data into a brief, readable summary that considers all factors holistically. "
              "Keep numbers and units intact when absolutely necessary. Ensure text is clean and readable (no LaTeX/special fonts). "
              "Your output should be brief, precise, and easy to read - focus on the overall picture, not individual data points.")
    user = ("Analyze the following stock data comprehensively. Consider ALL factors: "
            "- Price trends and moving average positions (positive % = above MA/bullish, negative % = below MA/bearish) "
            "- Balance Sheet Score (above 11=excellent, above 5=good, above 2=average, below 2=bad, below -5=very bad) "
            "- News Sentiment Score (above 7=excellent, above 4=good, above 0=neutral, below -1=bad, below -4=very bad) "
            "- Technical indicators (MA, RSI, MACD) and trend deltas \n\n"
            "Provide ONLY: 1. A concise 4-5 sentence summary synthesizing the key factors (DO NOT list individual metrics or scores) "
            "2. A clear AI recommendation: BULLISH, BEARISH, or HOLD with brief 1-2 sentences reasoning \n\n"
            "Remember: Do NOT mention specific score values or metrics in your response. Synthesize everything into a holistic view. "
            f"\n\n{context}")
    return llm_chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=400)


def ai_news_summary(news, sentiment_type, symbol, max_articles=20):
    """AI bullet summary of the positive or negative news for a symbol."""
    news = news.sort_values('date', ascending=False).head(max_articles)
    articles = "".join(f"Article {i}:\nDate: {r['date']}\nSource: {r['source']}\nHeadline: {r['headline']}\nSummary: {r['summary']}\n\n"
                       for i, (_, r) in enumerate(news.iterrows(), 1))
    system = ("You are a financial news analyst. Provide a concise summary of the news articles provided. "
              "Focus on key themes, trends, and important information that would be relevant for stock analysis. "
              "Respond in 2-4 bullet points, each on a new line. Keep the summary factual and objective. Do not repeat the same information.")
    user = (f"Analyze the following {sentiment_type} news articles for {symbol} and provide a summary:\n\n"
            f"Total articles: {len(news)}\n\n{articles}\n\n"
            f"Provide a concise summary highlighting the main themes and key information from these {sentiment_type} news articles. "
            "Ensure that the text is clean and readable. Do not use LaTeX formatting or special fonts for numbers (e.g. use '100' not '$100$'). "
            "Make sure words are not broken up and sentences are complete.")
    return llm_chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=500)


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
