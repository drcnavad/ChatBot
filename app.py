import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="Stock Signal Lookup",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hugging Face Configuration - get token safely
def get_hf_token():
    """Get Hugging Face token from secrets or environment variable"""
    # Try environment variable first (works in all contexts)
    token = os.getenv("HF_TOKEN", "")
    
    # Only try Streamlit secrets if we're in a proper Streamlit context
    try:
        # Check if we're in a Streamlit runtime context
        if hasattr(st, 'secrets'):
            secrets = getattr(st, 'secrets', None)
            if secrets is not None:
                # Use .get() to safely access secrets
                # Note: This may trigger parsing of secrets.toml, so we catch all exceptions
                streamlit_token = secrets.get("HF_TOKEN", "")
                if streamlit_token:
                    return streamlit_token
    except Exception:
        # Catch ALL exceptions including StreamlitSecretNotFoundError, TOML parsing errors, etc.
        # This ensures the app never crashes due to secrets file issues
        # Fallback to environment variable
        pass
    
    return token

# Get token (safe for Streamlit Cloud)
HF_TOKEN = get_hf_token()

def build_trend_deltas(df, ticker, windows=(3, 5, 14, 50, 100)):
    """
    Build compact trend-difference rows for LLM context.
    Returns a list of dicts, one per window.
    Optimized to filter once and reuse the filtered dataframe.
    """
    # Filter once and sort - more efficient than filtering multiple times
    recent = df[df["Symbol"] == ticker].sort_values("Date")
    
    if len(recent) == 0:
        return []
    
    latest = recent.iloc[-1]
    rows = []

    for w in windows:
        if len(recent) < w:
            continue

        past = recent.iloc[-w]
        # Pre-calculate tail window data to avoid multiple tail() calls
        tail_data = recent.tail(w)

        row = {
            "window_days": w,
            "price_change_pct": round(
                (latest["Close"] / past["Close"] - 1) * 100, 2
            ),
            "rsi_change": round(
                latest["RSI Options Rate"] - past["RSI Options Rate"], 2
            ),
            "macd_change": round(
                latest["macd"] - past["macd"], 2
            ),
            "ma30_diff_pct": round(
                (latest["Close"] - latest["ma_30"]) / latest["ma_30"] * 100, 2
            ),
            "ma200_diff_pct": round(
                (latest["Close"] - latest["ma_200"]) / latest["ma_200"] * 100, 2
            ),
            "above_ma200_days_pct": round(
                (tail_data["Close"] > tail_data["ma_200"]).mean() * 100, 2
            )
        }

        rows.append(row)

    return rows


def format_trend_rows(trend_rows):
    text = "Trend Deltas:\n"
    for r in trend_rows:
        text += (
            f"Last {r['window_days']} days: "
            f"Price {r['price_change_pct']:.2f}%, "
            f"RSI change {r['rsi_change']:.2f}, "
            f"MACD change {r['macd_change']:.2f}, "
            f"Price vs MA30 {r['ma30_diff_pct']:.2f}%, "
            f"Price vs MA200 {r['ma200_diff_pct']:.2f}%, "
            f"Above MA200 {r['above_ma200_days_pct']:.2f}% of days\n"
        )
    return text


def generate_ai_summary(ticker, stock_data, df):
    """Generate AI summary using Hugging Face Llama-3 model"""
    try:
        client = InferenceClient(token=HF_TOKEN)

        # Calculate differences between current price and moving averages
        current_price = stock_data['Close']
        ma_differences = {}
        ma_percentages = {}
        
        for ma_name in ['ma_10', 'ma_30', 'ma_50', 'ma_100', 'ma_200']:
            ma_value = stock_data.get(ma_name)
            if pd.notna(ma_value) and ma_value != 0:
                diff = current_price - ma_value
                diff_pct = (diff / ma_value) * 100
                ma_differences[ma_name] = round(diff, 2)
                ma_percentages[ma_name] = round(diff_pct, 2)
            else:
                ma_differences[ma_name] = 'N/A'
                ma_percentages[ma_name] = 'N/A'

        # Format MA differences for display (values already rounded to 2 decimals)
        def format_ma_diff(diff, pct):
            if diff == 'N/A' or pct == 'N/A':
                return 'N/A'
            return f"${diff:.2f} ({pct:.2f}%)"
        
        # Format numbers to 2 decimals, handle N/A
        def format_num(value, default='N/A'):
            if pd.isna(value) or value == 'N/A':
                return default
            return f"{float(value):.2f}"
        
        rsi_value = stock_data.get('RSI Options Rate')
        macd_value = stock_data.get('macd')
        fundamental_weight = stock_data.get('Fundamental_Weight')
        sentiment_score = stock_data.get('SentimentScore')
        
        context = f"""Stock: {ticker}
                    Date: {stock_data['Date'].strftime('%Y-%m-%d')}
                    Current Price: ${current_price:.2f}
                    Technical Score: {format_num(stock_data.get('combined_signal'))}
                    RSI: {format_num(rsi_value)}
                    MACD: {format_num(macd_value)}
                    
                    Price vs Moving Averages (Difference):
                    Price - MA10: {format_ma_diff(ma_differences['ma_10'], ma_percentages['ma_10'])}
                    Price - MA30: {format_ma_diff(ma_differences['ma_30'], ma_percentages['ma_30'])}
                    Price - MA50: {format_ma_diff(ma_differences['ma_50'], ma_percentages['ma_50'])}
                    Price - MA100: {format_ma_diff(ma_differences['ma_100'], ma_percentages['ma_100'])}
                    Price - MA200: {format_ma_diff(ma_differences['ma_200'], ma_percentages['ma_200'])}
                    
                    Balance Sheet Score: {format_num(fundamental_weight)}
                    Sentiment Score: {format_num(sentiment_score)}
                    """
        # Add trend analysis to context (only filter relevant data for this ticker)
        ticker_normalized = str(ticker).strip().upper()
        ticker_df = df[df['Symbol'] == ticker_normalized].copy()
        trend_rows = build_trend_deltas(ticker_df, ticker_normalized, windows=(14, 50, 200))
        trend_text = format_trend_rows(trend_rows)

        context += f"\n{trend_text}"


        messages = [
        {
            "role": "system",
            "content": (
                "You are a financial advisor. Provide ONLY a concise summary (4-5 sentences) followed by a clear AI recommendation. "
                "DO NOT list individual metrics, scores, or numbers in your response. "
                "DO NOT mention specific values like 'Balance Sheet Score: X', 'News Sentiment Score: Y', or 'RSI: Z'. "
                "Instead, synthesize all the data into a brief, readable summary that considers all factors holistically. "
                "Keep numbers and units intact when absolutely necessary. Make sure words are not broken up. "
                "Your output should be brief, precise, and easy to read - focus on the overall picture, not individual data points."
            )
        },
        {
            "role": "user",
            "content": (
                "Analyze the following stock data comprehensively. Consider ALL factors: "
                "- Price trends and moving average positions (positive % = above MA/bullish, negative % = below MA/bearish) "
                "- Balance Sheet Score (above 11=excellent, above 5=good, above 2=average, below 2=bad, below -5=very bad) "
                "- News Sentiment Score (above 7=excellent, above 4=good, above 0=neutral, below -1=bad, below -4=very bad) "
                "- Technical indicators (MA, RSI, MACD) and trend deltas "
                "\n\n"
                "Provide ONLY: "
                "1. A concise 4-5 sentence summary synthesizing the key factors (DO NOT list individual metrics or scores) "
                "2. A clear AI recommendation: BULLISH, BEARISH, or HOLD with brief 1-2 sentences reasoning "
                "\n\n"
                "Remember: Do NOT mention specific score values or metrics in your response. Synthesize everything into a holistic view. "
                f"\n\n{context}"
            )
        }
    ]


        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            max_tokens=400,
            temperature=0.2
        )

        summary_raw = response.choices[0].message.content.strip()

        # Clean artifacts just in case
        summary = re.split(r'\[/?USER\]|Can you|Could you', summary_raw)[0].strip()

        return summary

    except Exception as e:
        return f"Error generating summary: {str(e)}"


# Load data with optimized caching and dtype optimization
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load CSV data with optimized memory usage"""
    csv_path = os.path.join("Reports", "signal_analysis.csv")
    
    if not os.path.exists(csv_path):
        st.error(f"Data file not found. Expected {csv_path}")
        return None
    
    # Optimize CSV reading with low_memory=False for better performance
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Ensure proper data types
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
    df['final_trade'] = df['final_trade'].astype(str).str.strip()
    
    # Optimize numeric columns - downcast float64 to float32 to reduce memory
    numeric_cols = df.select_dtypes(include=['float64']).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
    
    return df

# Cache latest data processing (expensive operation)
@st.cache_data(ttl=3600)
def get_latest_data(df):
    """Get latest data for each symbol - cached to avoid recomputation"""
    return df.sort_values('Date', ascending=False).drop_duplicates(subset='Symbol', keep='first')

# Cache top stocks calculation
@st.cache_data(ttl=3600)
def get_top_stocks(latest_data, n=100):
    """Get top N stocks by signal - cached"""
    top_stocks = latest_data.nlargest(n, 'combined_signal')[['Symbol', 'combined_signal', 'final_trade', 'Close']].copy()
    return top_stocks.sort_values('combined_signal', ascending=False).reset_index(drop=True)

# Cache min/max values for gauges (expensive computation)
@st.cache_data(ttl=3600)
def get_gauge_ranges(df):
    """Get min/max values for gauge charts - cached"""
    ranges = {}
    if 'combined_signal' in df.columns:
        ranges['combined_signal'] = {
            'min': float(df['combined_signal'].min()),
            'max': float(df['combined_signal'].max())
        }
    if 'Fundamental_Weight' in df.columns:
        ranges['Fundamental_Weight'] = {
            'min': float(df['Fundamental_Weight'].min()),
            'max': float(df['Fundamental_Weight'].max())
        }
    if 'SentimentScore' in df.columns:
        ranges['SentimentScore'] = {
            'min': float(df['SentimentScore'].min()),
            'max': float(df['SentimentScore'].max())
        }
    return ranges

# Cache available symbols (expensive operation)
@st.cache_data(ttl=3600)
def get_available_symbols(df):
    """Get available symbols from latest date - cached"""
    latest_date = df['Date'].max()
    latest_date_df = df[df['Date'] == latest_date]
    return sorted(latest_date_df['Symbol'].unique().tolist())

# News sentiment summary functions
@st.cache_data(ttl=3600)
def load_news_data():
    """Load news data from CSV - cached"""
    news_csv = os.path.join("Reports", "news_cleaned_df.csv")
    if not os.path.exists(news_csv):
        return None
    return pd.read_csv(news_csv)

def get_news_by_symbol(news_df, symbol):
    """Filter news data for a specific stock symbol"""
    if news_df is None:
        return pd.DataFrame()
    return news_df[news_df['symbol'] == symbol.upper()].copy()

def group_news_by_sentiment(df):
    """Group news into positive and negative based on sentiment_label"""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    positive_news = df[df['sentiment_label'] == 'positive'].copy()
    negative_news = df[df['sentiment_label'] == 'negative'].copy()
    return positive_news, negative_news

def format_news_for_llm(df, max_articles=20):
    """Format news headlines and summaries for LLM input"""
    if df.empty:
        return "No news articles available."
    
    # Limit to most recent articles
    df_sorted = df.sort_values('date', ascending=False).head(max_articles)
    
    formatted_text = f"Total articles: {len(df_sorted)}\n\n"
    
    for article_num, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        headline = str(row.get('headline', 'N/A'))
        summary = str(row.get('summary', 'N/A'))
        date = str(row.get('date', 'N/A'))
        source = str(row.get('source', 'N/A'))
        
        formatted_text += f"Article {article_num}:\n"
        formatted_text += f"Date: {date}\n"
        formatted_text += f"Source: {source}\n"
        formatted_text += f"Headline: {headline}\n"
        formatted_text += f"Summary: {summary}\n\n"
    
    return formatted_text

def generate_news_summary(news_text, sentiment_type, symbol):
    """Generate AI summary for news articles using Hugging Face Llama-3 model"""
    if not HF_TOKEN:
        return "Error: HF_TOKEN not configured"
    
    if news_text == "No news articles available.":
        return "No news articles available for analysis."
    
    try:
        client = InferenceClient(token=HF_TOKEN)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a financial news analyst. Provide a concise summary of the news articles provided. "
                    "Focus on key themes, trends, and important information that would be relevant for stock analysis. "
                    "Respond in 2-4 bullet points, each on a new line. Keep the summary factual and objective. Do not repeat the same information."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Analyze the following {sentiment_type} news articles for {symbol} and provide a summary:\n\n"
                    f"{news_text}\n\n"
                    f"Provide a concise summary highlighting the main themes and key information from these {sentiment_type} news articles."
                    f"Make sure that words and numbersare not broken up. Make sure that the summary is not too long. "
                )
            }
        ]
        
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            max_tokens=500,
            temperature=0.2
        )
        
        summary_raw = response.choices[0].message.content.strip()
        
        # Clean artifacts
        summary = re.split(r'\[/?USER\]|Can you|Could you', summary_raw)[0].strip()
        
        return summary
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Main app – header
st.markdown(
    """
    <div style="margin-bottom: 1rem;">
        <h1 style="margin-bottom: 0.25rem;">📈 Stock Signal Lookup</h1>
        <p style="color: #555; font-size: 0.95rem; margin-top: 0;">
            Analyze technical signals, sentiment, and fundamentals for any ticker in your watchlist.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data (cached) - show loading indicator only on first load
if 'data_loaded' not in st.session_state:
    with st.spinner("Loading data... This may take a moment on first load."):
        df = load_data()
        st.session_state.data_loaded = True
else:
    df = load_data()

# Sidebar with top stocks (lazy loaded)
with st.sidebar:
    st.markdown(
        """
        <h2 style="margin-bottom: 0.25rem;">🏆 Top Stocks by Signal</h2>
        <p style="color: #777; font-size: 0.85rem; margin-top: 0;">
            Click a ticker to load its full signal and chart details.
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    if df is not None:
        # Use cached functions for expensive operations
        latest_data = get_latest_data(df)
        top_stocks = get_top_stocks(latest_data, n=100)
        
        # Filter by signal
        filter_option = st.selectbox(
            "Filter by signal",
            ["All", "BUY", "SELL", "HOLD"],
            key="signal_filter"
        )
        
        # Apply filter
        if filter_option != "All":
            top_stocks = top_stocks[top_stocks['final_trade'] == filter_option].reset_index(drop=True)
        
        # Initialize session state for showing more stocks
        if 'show_stocks_count' not in st.session_state:
            st.session_state.show_stocks_count = 10
        
        # Reset count and select first stock when filter changes
        if 'last_filter' not in st.session_state:
            st.session_state.last_filter = filter_option
        
        if st.session_state.last_filter != filter_option:
            st.session_state.show_stocks_count = 10
            st.session_state.last_filter = filter_option
            # Auto-select first stock from filtered list
            if len(top_stocks) > 0:
                st.session_state.selected_ticker = top_stocks.iloc[0]['Symbol']
        
        # Check if current selected ticker is in filtered list, if not, select first one
        current_ticker = st.session_state.get('selected_ticker', '')
        if current_ticker and len(top_stocks) > 0:
            if current_ticker not in top_stocks['Symbol'].values:
                st.session_state.selected_ticker = top_stocks.iloc[0]['Symbol']
        
        # Display stocks
        for idx in range(min(st.session_state.show_stocks_count, len(top_stocks))):
            stock = top_stocks.iloc[idx]
            signal_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(stock['final_trade'], '⚪')
            
            # Make it clickable
            if st.button(
                f"{signal_emoji} {stock['Symbol']}  ·  {stock['final_trade']}  ·  {stock['combined_signal']:.1f}",
                key=f"stock_{idx}",
                use_container_width=True
            ):
                st.session_state.selected_ticker = stock['Symbol']
        
        # Show more button
        if st.session_state.show_stocks_count < len(top_stocks):
            if st.button("See More", use_container_width=True):
                st.session_state.show_stocks_count += 10
    else:
        st.info("Loading data...")

if df is not None:
    # Get tickers from the most recent date only (cached)
    available_symbols = get_available_symbols(df)
    
    # Cache latest_data once to avoid multiple calls
    latest_data = get_latest_data(df)
    
    # Use selected ticker from sidebar if clicked (before creating columns)
    if 'selected_ticker' in st.session_state and st.session_state.selected_ticker:
        selected_ticker = st.session_state.selected_ticker
        st.session_state.selected_ticker = None  # Reset after use
        # Update the selectbox key to force refresh
        if 'ticker_select' in st.session_state:
            st.session_state.ticker_select = selected_ticker
    else:
        selected_ticker = None
    
    # Main content area
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        # Determine the index for the selectbox
        default_index = 0
        ticker_select_value = st.session_state.get('ticker_select', '') or (selected_ticker if selected_ticker else '')
        if ticker_select_value and ticker_select_value in available_symbols:
            default_index = available_symbols.index(ticker_select_value)
        
        ticker = st.selectbox(
            "Select Ticker Symbol",
            options=available_symbols,
            index=default_index,
            key="ticker_select"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🤖 Generate AI Summary", type="primary"):
            st.session_state.generate_summary = True
    
    with col3:
        # Show ticker symbol and date if ticker is entered
        display_ticker = ticker or (selected_ticker if selected_ticker else '')
        if display_ticker:
            # Normalize ticker
            display_ticker = str(display_ticker).strip().upper()
            # Use cached latest_data instead of filtering again
            ticker_latest = latest_data[latest_data['Symbol'] == display_ticker]
            if not ticker_latest.empty:
                latest_temp = ticker_latest.iloc[0]
                date_str = latest_temp['Date'].strftime("%m/%d/%Y")
                st.markdown(f"""
                <div style="font-size: 1.25rem; padding-top: 0.5rem; padding-bottom: 0.5rem;">
                    <strong>{display_ticker}</strong> | {date_str}
                </div>
                """, unsafe_allow_html=True)
    
    # Display AI Summary if requested
    if st.session_state.get('generate_summary', False) and ticker:
        ticker_normalized = str(ticker).strip().upper() if ticker else None
        ticker_data_summary = df[df['Symbol'] == ticker_normalized].copy() if ticker_normalized else pd.DataFrame()
        if not ticker_data_summary.empty:
            latest = ticker_data_summary.nlargest(1, 'Date').iloc[0]
            
            # Generate main AI summary
            with st.spinner(f"Generating AI summary for {ticker}..."):
                summary = generate_ai_summary(ticker, latest, df)
            
            with st.expander(f"🤖 AI Summary for {ticker}", expanded=True):
                st.markdown(summary)
            
            # Generate positive and negative news summaries (only if news file exists)
            news_df = load_news_data()
            if news_df is not None:
                symbol_news = get_news_by_symbol(news_df, ticker)
                if not symbol_news.empty:
                    positive_news, negative_news = group_news_by_sentiment(symbol_news)
                    
                    # Create two columns for positive and negative summaries
                    col_pos, col_neg = st.columns(2)
                    
                    with col_pos:
                        if not positive_news.empty:
                            with st.spinner(f"Analyzing {len(positive_news)} positive news articles..."):
                                positive_news_text = format_news_for_llm(positive_news, max_articles=20)
                                positive_summary = generate_news_summary(positive_news_text, "positive", ticker)
                            
                            with st.expander(f"✅ Positive News Summary ({len(positive_news)} articles)", expanded=True):
                                st.markdown(positive_summary)
                        else:
                            st.info("No positive news articles found")
                    
                    with col_neg:
                        if not negative_news.empty:
                            with st.spinner(f"Analyzing {len(negative_news)} negative news articles..."):
                                negative_news_text = format_news_for_llm(negative_news, max_articles=20)
                                negative_summary = generate_news_summary(negative_news_text, "negative", ticker)
                            
                            with st.expander(f"❌ Negative News Summary ({len(negative_news)} articles)", expanded=True):
                                st.markdown(negative_summary)
                        else:
                            st.info("No negative news articles found")
                else:
                    st.info(f"No news articles found for {ticker}")
            # If news file doesn't exist, silently skip (don't show warning to avoid cluttering UI)
            
            st.session_state.generate_summary = False  # Reset flag
    
    if ticker:
        # Normalize ticker (uppercase, strip whitespace)
        ticker = str(ticker).strip().upper()
        # Filter data for the ticker (only once)
        ticker_data = df[df['Symbol'] == ticker].copy()
        
        if len(ticker_data) == 0:
            st.warning(f"❌ Ticker '{ticker}' not found in database.")
            st.info(f"Available tickers: {', '.join(available_symbols[:20])}..." if len(available_symbols) > 20 else f"Available tickers: {', '.join(available_symbols)}")
        else:
            # Get latest data (most recent date) - optimized to avoid re-sorting
            latest = ticker_data.nlargest(1, 'Date').iloc[0]
            
            # Display signal
            signal = latest['final_trade']
            signal_color = {
                'BUY': '🟢',
                'SELL': '🔴',
                'HOLD': '🟡'
            }
            signal_bg = {
                'BUY': 'background-color: #d4edda; color: #155724;',
                'SELL': 'background-color: #f8d7da; color: #721c24;',
                'HOLD': 'background-color: #fff3cd; color: #856404;'
            }
            
            st.markdown("---")
            
            # Details - Main metrics (with signal in first column)
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1.2, 1.2, 1, 1, 0.1])
            
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border-radius: 5px; {signal_bg.get(signal, '')}">
                    <strong>{signal_color.get(signal, '')} {signal}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Current Price", f"${latest['Close']:.2f}")
            
            with col3:
                st.metric("Combined Signal", f"{latest['combined_signal']:.2f}")
            
            with col4:
                buy_streak = latest.get('Buy Streak', 0)
                st.metric("Buy Streak", f"{int(buy_streak)} days")
            
            with col5:
                sell_streak = latest.get('Sell Streak', 0)
                st.metric("Sell Streak", f"{int(sell_streak)} days")
            
            # Gauge charts for key metrics
            gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
            
            # Get min/max values from cached function (expensive operation)
            gauge_ranges = get_gauge_ranges(df)
            
            combined_signal_min = gauge_ranges.get('combined_signal', {}).get('min', 0)
            combined_signal_max = gauge_ranges.get('combined_signal', {}).get('max', 100)
            combined_signal_val = float(latest['combined_signal']) if 'combined_signal' in latest.index and pd.notna(latest['combined_signal']) else 0
            combined_signal_mean = df['combined_signal'].mean() #####

            fundamental_weight_min = gauge_ranges.get('Fundamental_Weight', {}).get('min', 0)
            fundamental_weight_max = gauge_ranges.get('Fundamental_Weight', {}).get('max', 100)
            fundamental_weight_val = float(latest['Fundamental_Weight']) if 'Fundamental_Weight' in latest.index and pd.notna(latest['Fundamental_Weight']) else 0
            fundamental_weight_mean = df['Fundamental_Weight'].mean() #####

            sentiment_score_min = gauge_ranges.get('SentimentScore', {}).get('min', -10)
            sentiment_score_max = gauge_ranges.get('SentimentScore', {}).get('max', 10)
            sentiment_score_val = float(latest['SentimentScore']) if 'SentimentScore' in latest.index and pd.notna(latest['SentimentScore']) else 0
            sentiment_score_mean = df['SentimentScore'].mean() #####
        

            # Gauge 1: Combined Signal
            with gauge_col1:
                # combined_signal_mid = (combined_signal_min + combined_signal_max) / 2
                combined_signal_mid = round(combined_signal_mean, 2)
                fig1 = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = combined_signal_val,
                    domain = {'x': [0, 1], 'y': [0, 0.5]},
                    title = {'text': "Combined Signal", 'font': {'color': 'black'}},
                    number = {'font': {'color': 'black'}},
                    gauge = {
                        'axis': {
                            'range': [combined_signal_min, combined_signal_max],
                            'tickmode': 'array',
                            'tickvals': [combined_signal_min, combined_signal_mid, combined_signal_max],
                            'ticktext': [f'Min: {combined_signal_min:.1f}', f'Mid: {combined_signal_mid:.1f}', f'Max: {combined_signal_max:.1f}'],
                            'tickcolor': 'black',
                            'tickfont': {'color': 'black'}
                        },
                        'bar': {'color': "darkorange"},
                        'steps': [
                            {'range': [combined_signal_min, combined_signal_val], 'color': "orange"},
                            {'range': [combined_signal_val, combined_signal_max], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': combined_signal_val
                        },
                        'shape': 'angular'
                    }
                ))
                fig1.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=40, b=60),
                    paper_bgcolor="white",
                    annotations=[dict(
                        x=0.5, y=-0.15,
                        text=f'Current: {combined_signal_val:.2f}',
                        showarrow=False,
                        font=dict(size=12, color="black")
                    )]
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            # Gauge 2: Balance Sheet
            with gauge_col2:
                # fundamental_weight_mid = (fundamental_weight_min + fundamental_weight_max) / 2
                fundamental_weight_mid = round(fundamental_weight_mean, 2)
                fig2 = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = fundamental_weight_val,
                    domain = {'x': [0, 1], 'y': [0, 0.5]},
                    title = {'text': "Balance Sheet", 'font': {'color': 'black'}},
                    number = {'font': {'color': 'black'}},
                    gauge = {
                        'axis': {
                            'range': [fundamental_weight_min, fundamental_weight_max],
                            'tickmode': 'array',
                            'tickvals': [fundamental_weight_min, fundamental_weight_mid, fundamental_weight_max],
                            'ticktext': [f'Min: {fundamental_weight_min:.1f}', f'Mid: {fundamental_weight_mid:.1f}', f'Max: {fundamental_weight_max:.1f}'],
                            'tickcolor': 'black',
                            'tickfont': {'color': 'black'}
                        },
                        'bar': {'color': "darkorange"},
                        'steps': [
                            {'range': [fundamental_weight_min, fundamental_weight_val], 'color': "orange"},
                            {'range': [fundamental_weight_val, fundamental_weight_max], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': fundamental_weight_val
                        },
                        'shape': 'angular'
                    }
                ))
                fig2.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=40, b=60),
                    paper_bgcolor="white",
                    annotations=[dict(
                        x=0.5, y=-0.15,
                        text=f'Current: {fundamental_weight_val:.2f}',
                        showarrow=False,
                        font=dict(size=12, color="black")
                    )]
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Gauge 3: News Sentiment
            with gauge_col3:
                # sentiment_score_mid = (sentiment_score_min + sentiment_score_max) / 2
                sentiment_score_mid = round(sentiment_score_mean, 2)
                fig3 = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = sentiment_score_val,
                    domain = {'x': [0, 1], 'y': [0, 0.5]},
                    title = {'text': "News Sentiment", 'font': {'color': 'black'}},
                    number = {'font': {'color': 'black'}},
                    gauge = {
                        'axis': {
                            'range': [sentiment_score_min, sentiment_score_max],
                            'tickmode': 'array',
                            'tickvals': [sentiment_score_min, sentiment_score_mid, sentiment_score_max],
                            'ticktext': [f'Min: {sentiment_score_min:.1f}', f'Mid: {sentiment_score_mid:.1f}', f'Max: {sentiment_score_max:.1f}'],
                            'tickcolor': 'black',
                            'tickfont': {'color': 'black'}
                        },
                        'bar': {'color': "darkorange"},
                        'steps': [
                            {'range': [sentiment_score_min, sentiment_score_val], 'color': "orange"},
                            {'range': [sentiment_score_val, sentiment_score_max], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': sentiment_score_val
                        },
                        'shape': 'angular'
                    }
                ))
                fig3.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=40, b=60),
                    paper_bgcolor="white",
                    annotations=[dict(
                        x=0.5, y=-0.15,
                        text=f'Current: {sentiment_score_val:.2f}',
                        showarrow=False,
                        font=dict(size=12, color="black")
                    )]
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            # Filter to last 1 year of data (optimized - avoid unnecessary copies)
            latest_date = ticker_data['Date'].max()
            one_year_ago = latest_date - pd.Timedelta(days=365)
            ticker_data_sorted = ticker_data[ticker_data['Date'] >= one_year_ago].sort_values('Date', ascending=True)
            
            # Only render charts if we have data
            if len(ticker_data_sorted) > 0:
                # Charts section - stacked subplots
                st.markdown("### 📈 Technical Analysis Charts (Last 1 Year)")
                
                # Secondary metrics - professional display
                st.markdown("""
                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
                    <div style='display: flex; justify-content: space-around; flex-wrap: wrap; gap: 15px;'>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
            
                with col1:
                    ma_10 = latest.get('ma_10', 0)
                    if pd.notna(ma_10):
                        st.metric("MA 10", f"${ma_10:.2f}", delta=None)
                
                with col2:
                    ma_30 = latest.get('ma_30', 0)
                    if pd.notna(ma_30):
                        st.metric("MA 30", f"${ma_30:.2f}", delta=None)
                
                with col3:
                    ma_50 = latest.get('ma_50', 0)
                    if pd.notna(ma_50):
                        st.metric("MA 50", f"${ma_50:.2f}", delta=None)
                
                with col4:
                    ma_100 = latest.get('ma_100', 0)
                    if pd.notna(ma_100):
                        st.metric("MA 100", f"${ma_100:.2f}", delta=None)
                
                with col5:
                    ma_200 = latest.get('ma_200', 0)
                    if pd.notna(ma_200):
                        st.metric("MA 200", f"${ma_200:.2f}", delta=None)
                
                with col6:
                    rsi = latest.get('RSI Options Rate', 0)
                    if pd.notna(rsi):
                        st.metric("RSI", f"{rsi:.2f}", delta=None)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Prepare data
                price_data = ticker_data_sorted[['Date', 'Close', 'ma_10', 'ma_30', 'ma_50', 'ma_100', 'ma_200', 
                                                 'combined_signal', 'Buy Streak', 'Sell Streak']].copy()
                price_data = price_data.set_index('Date')
                price_data = price_data.dropna(subset=['Close'])
                # Fill NaN values for signal and streaks with 0
                price_data['combined_signal'] = price_data['combined_signal'].fillna(0)
                price_data['Buy Streak'] = price_data['Buy Streak'].fillna(0)
                price_data['Sell Streak'] = price_data['Sell Streak'].fillna(0)
                
                # Create subplots: 3 rows, 1 column
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.65, 0.175, 0.175],
                    subplot_titles=(
                        '<b>Price and Moving Averages</b>',
                        '<b>RSI (Relative Strength Index)</b>',
                        '<b>MACD (Moving Average Convergence Divergence)</b>'
                    )
                )
                
                # Update subplot titles styling
                fig.update_annotations(
                    font=dict(size=13, color='#1f1f1f', family='Arial, sans-serif'),
                    yshift=5
                )
                
                # Row 1: Price and Moving Averages
                # Create custom hover text with combined signal and streaks
                hover_text_close = []
                for idx in price_data.index:
                    signal_val = price_data.loc[idx, 'combined_signal']
                    buy_streak = price_data.loc[idx, 'Buy Streak']
                    sell_streak = price_data.loc[idx, 'Sell Streak']
                    hover_info = f"Combined Signal: {signal_val:.2f}"
                    if buy_streak > 0:
                        hover_info += f"<br>Buy Streak: {int(buy_streak)} days"
                    if sell_streak > 0:
                        hover_info += f"<br>Sell Streak: {int(sell_streak)} days"
                    hover_text_close.append(hover_info)
                
                fig.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['Close'],
                    name='Close Price',
                    line=dict(color='#0d47a1', width=3),  # Darker blue
                    mode='lines',
                    customdata=hover_text_close,
                    hovertemplate='<b>Close</b><br>$%{y:.2f}<br>%{customdata}<extra></extra>'
                ), row=1, col=1)
                
                if 'ma_10' in price_data.columns:
                    fig.add_trace(go.Scatter(
                        x=price_data.index,
                        y=price_data['ma_10'],
                        name='MA 10',
                        line=dict(color='rgba(255, 127, 14, 0.5)', width=1.2),  # Lighter orange
                        mode='lines',
                        showlegend=True,
                        hovertemplate='<b>MA 10</b><br>$%{y:.2f}<extra></extra>'
                    ), row=1, col=1)
                
                if 'ma_30' in price_data.columns:
                    fig.add_trace(go.Scatter(
                        x=price_data.index,
                        y=price_data['ma_30'],
                        name='MA 30',
                        line=dict(color='rgba(214, 39, 40, 0.5)', width=1.2),  # Lighter red
                        mode='lines',
                        showlegend=True,
                        hovertemplate='<b>MA 30</b><br>$%{y:.2f}<extra></extra>'
                    ), row=1, col=1)
                
                if 'ma_50' in price_data.columns:
                    fig.add_trace(go.Scatter(
                        x=price_data.index,
                        y=price_data['ma_50'],
                        name='MA 50',
                        line=dict(color='rgba(148, 103, 189, 0.5)', width=1.2),  # Lighter purple
                        mode='lines',
                        showlegend=True,
                        hovertemplate='<b>MA 50</b><br>$%{y:.2f}<extra></extra>'
                    ), row=1, col=1)
                
                if 'ma_100' in price_data.columns:
                    fig.add_trace(go.Scatter(
                        x=price_data.index,
                        y=price_data['ma_100'],
                        name='MA 100',
                        line=dict(color='rgba(44, 160, 44, 0.5)', width=1.2),  # Lighter green
                        mode='lines',
                        showlegend=True,
                        hovertemplate='<b>MA 100</b><br>$%{y:.2f}<extra></extra>'
                    ), row=1, col=1)
                
                if 'ma_200' in price_data.columns:
                    fig.add_trace(go.Scatter(
                        x=price_data.index,
                        y=price_data['ma_200'],
                        name='MA 200',
                        line=dict(color='rgba(140, 86, 75, 0.5)', width=1.2),  # Lighter brown
                        mode='lines',
                        showlegend=True,
                        hovertemplate='<b>MA 200</b><br>$%{y:.2f}<extra></extra>'
                    ), row=1, col=1)
                
                # Add Buy Streak and Sell Streak indicators at top of price chart
                if 'Buy Streak' in ticker_data_sorted.columns and 'Sell Streak' in ticker_data_sorted.columns:
                    # Get max price for placing streak lines at top
                    max_price = price_data['Close'].max()
                    min_price = price_data['Close'].min()
                    price_range = max_price - min_price
                    top_y_price = max_price + (price_range * 0.05)  # Place lines 5% above max price
                    
                    streak_data = ticker_data_sorted[['Date', 'Buy Streak', 'Sell Streak']].copy()
                    streak_data = streak_data.set_index('Date')
                    streak_data = streak_data.reindex(price_data.index).fillna(0)
                    
                    # Identify continuous Buy Streak periods
                    buy_active = (streak_data['Buy Streak'] > 0).astype(int)
                    buy_periods = []
                    start_idx = None
                    
                    for idx, (date, is_active) in enumerate(buy_active.items()):
                        if is_active == 1 and start_idx is None:
                            start_idx = idx
                        elif is_active == 0 and start_idx is not None:
                            buy_periods.append((streak_data.index[start_idx], streak_data.index[idx-1]))
                            start_idx = None
                    if start_idx is not None:
                        buy_periods.append((streak_data.index[start_idx], streak_data.index[-1]))
                    
                    # Identify continuous Sell Streak periods
                    sell_active = (streak_data['Sell Streak'] > 0).astype(int)
                    sell_periods = []
                    start_idx = None
                    
                    for idx, (date, is_active) in enumerate(sell_active.items()):
                        if is_active == 1 and start_idx is None:
                            start_idx = idx
                        elif is_active == 0 and start_idx is not None:
                            sell_periods.append((streak_data.index[start_idx], streak_data.index[idx-1]))
                            start_idx = None
                    if start_idx is not None:
                        sell_periods.append((streak_data.index[start_idx], streak_data.index[-1]))
                    
                    # Identify Hold periods (when neither Buy nor Sell streak is active)
                    hold_active = ((buy_active == 0) & (sell_active == 0)).astype(int)
                    hold_periods = []
                    start_idx = None
                    
                    for idx, (date, is_active) in enumerate(hold_active.items()):
                        if is_active == 1 and start_idx is None:
                            start_idx = idx
                        elif is_active == 0 and start_idx is not None:
                            hold_periods.append((streak_data.index[start_idx], streak_data.index[idx-1]))
                            start_idx = None
                    if start_idx is not None:
                        hold_periods.append((streak_data.index[start_idx], streak_data.index[-1]))
                    
                    # Add horizontal lines for Buy Streak periods (green)
                    for start_date, end_date in buy_periods:
                        fig.add_shape(
                            type="line",
                            x0=start_date,
                            x1=end_date,
                            y0=top_y_price,
                            y1=top_y_price,
                            line=dict(color="#2ca02c", width=3.5),
                            row=1, col=1
                        )
                    
                    # Add horizontal lines for Sell Streak periods (red)
                    for start_date, end_date in sell_periods:
                        fig.add_shape(
                            type="line",
                            x0=start_date,
                            x1=end_date,
                            y0=top_y_price,
                            y1=top_y_price,
                            line=dict(color="#d62728", width=3.5),
                            row=1, col=1
                        )
                    
                    # Add horizontal lines for Hold periods (yellow-golden, thicker)
                    for start_date, end_date in hold_periods:
                        fig.add_shape(
                            type="line",
                            x0=start_date,
                            x1=end_date,
                            y0=top_y_price,
                            y1=top_y_price,
                            line=dict(color="#FFD700", width=4.5),
                            row=1, col=1
                        )
                
                # Row 2: RSI
                if 'RSI Options Rate' in ticker_data_sorted.columns:
                    rsi_data = ticker_data_sorted[['Date', 'RSI Options Rate']].copy()
                    rsi_data = rsi_data.set_index('Date')
                    rsi_data = rsi_data.dropna()
                    
                    fig.add_trace(go.Scatter(
                        x=rsi_data.index,
                        y=rsi_data['RSI Options Rate'],
                        name='RSI',
                        line=dict(color='#ff7f0e', width=2),
                        mode='lines',
                        showlegend=False,
                        hovertemplate='<b>RSI</b><br>%{y:.2f}<extra></extra>',
                        fill='tozeroy',
                        fillcolor='rgba(255, 127, 14, 0.1)'
                    ), row=2, col=1)
                    
                    # Add RSI overbought/oversold reference lines
                    fig.add_hline(y=70, line_dash="dash", line_color="rgba(200, 0, 0, 0.3)", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0, 200, 0, 0.3)", row=2, col=1)
                
                # Row 3: MACD
                if 'macd' in ticker_data_sorted.columns and 'MACD Signal' in ticker_data_sorted.columns:
                    macd_data = ticker_data_sorted[['Date', 'macd', 'MACD Signal']].copy()
                    macd_data = macd_data.set_index('Date')
                    macd_data = macd_data.dropna()
                    
                    fig.add_trace(go.Scatter(
                        x=macd_data.index,
                        y=macd_data['macd'],
                        name='MACD',
                        line=dict(color='#d62728', width=2.5),
                        mode='lines',
                        showlegend=False,
                        hovertemplate='<b>MACD</b><br>%{y:.4f}<extra></extra>'
                    ), row=3, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=macd_data.index,
                        y=macd_data['MACD Signal'],
                        name='MACD Signal',
                        line=dict(color='#1f77b4', width=2.5, dash='dash'),
                        mode='lines',
                        showlegend=False,
                        hovertemplate='<b>MACD Signal</b><br>%{y:.4f}<extra></extra>'
                    ), row=3, col=1)
                    
                    # Add zero line
                    fig.add_hline(y=0, line_dash="dot", line_color="rgba(128, 128, 128, 0.4)", line_width=1, row=3, col=1)
                
                # Update layout with professional styling
                fig.update_layout(
                    height=850,
                    hovermode='x unified',
                    margin=dict(l=50, r=50, t=120, b=50),  # Increased top margin for legend
                    plot_bgcolor='rgba(250, 250, 250, 1)',
                    paper_bgcolor='white',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.08,  # Position legend higher above the chart
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10, color='#333'),
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='rgba(200, 200, 200, 0.5)',
                        borderwidth=1,
                        itemwidth=30
                    ),
                    font=dict(family="Arial, sans-serif", size=11, color='#333'),
                    dragmode=False,
                    hoverlabel=dict(
                        bgcolor="white",
                        bordercolor="#333",
                        font_size=11,
                        font_family="Arial, sans-serif"
                    )
                )
                
                # Update x-axes (all charts show same dates - months only) with crosshair
                fig.update_xaxes(
                    tickformat='%b %Y',
                    showspikes=True,
                    spikecolor="#888",
                    spikesnap="cursor",
                    spikemode="across",
                    spikethickness=1,
                    spikedash="solid",
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    gridwidth=1,
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(200, 200, 200, 0.5)',
                    linewidth=1,
                    tickfont=dict(size=10, color='#666'),
                    title_font=dict(size=12, color='#333')
                )
                
                # Update y-axis labels with professional styling
                fig.update_yaxes(
                    title_text="Price ($)",
                    title_font=dict(size=11, color='#333'),
                    showspikes=True,
                    spikecolor="#888",
                    spikesnap="cursor",
                    spikemode="toaxis",
                    spikethickness=1,
                    spikedash="solid",
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    gridwidth=1,
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(200, 200, 200, 0.5)',
                    linewidth=1,
                    tickfont=dict(size=10, color='#666'),
                    tickformat='$,.0f',
                    row=1, col=1
                )
                fig.update_yaxes(
                    title_text="RSI",
                    title_font=dict(size=11, color='#333'),
                    showspikes=True,
                    spikecolor="#888",
                    spikesnap="cursor",
                    spikemode="toaxis",
                    spikethickness=1,
                    spikedash="solid",
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    gridwidth=1,
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(200, 200, 200, 0.5)',
                    linewidth=1,
                    tickfont=dict(size=10, color='#666'),
                    range=[0, 100],
                    row=2, col=1
                )
                fig.update_yaxes(
                    title_text="MACD",
                    title_font=dict(size=11, color='#333'),
                    showspikes=True,
                    spikecolor="#888",
                    spikesnap="cursor",
                    spikemode="toaxis",
                    spikethickness=1,
                    spikedash="solid",
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    gridwidth=1,
                    zeroline=True,
                    zerolinecolor='rgba(128, 128, 128, 0.5)',
                    zerolinewidth=1,
                    showline=True,
                    linecolor='rgba(200, 200, 200, 0.5)',
                    linewidth=1,
                    tickfont=dict(size=10, color='#666'),
                    row=3, col=1
                )
                
                st.plotly_chart(
                    fig, 
                    use_container_width=True,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d'],
                        'scrollZoom': False,
                        'doubleClick': 'reset'
                    }
                )
            else:
                st.warning("No data available for charting.")

