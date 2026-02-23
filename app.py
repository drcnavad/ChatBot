"""
Stock Analysis Report - Streamlit app for technical/fundamental analysis with AI-powered signals.
"""
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import re
from urllib.parse import quote

# --- App configuration ---
load_dotenv()
st.set_page_config(
    page_title="Stock Analysis Report",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS styling (cream / greyish-white theme) ---
st.markdown("""
<style>
    .stApp, .main { background-color: #f8f6f0; }
    .main .block-container { padding-top: 0 !important; padding-bottom: 2rem; margin-top: 0 !important; background-color: #f8f6f0; }
    .stApp > header {
        padding-top: 0 !important;
        margin-top: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        display: none !important;
    }
    header[data-testid="stHeader"] { display: none !important; height: 0 !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important; }
    .main .block-container > div:first-child {
        margin-top: 0 !important; padding-top: 0 !important;
    }
    .stApp {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
        height: 0 !important;
    }
    #root > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    h1 {
        color: #1f2937;
        font-weight: 600;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    h2 {
        color: #374151;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    h3 {
        color: #4b5563;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 12px;
    }
    [data-testid="stMetricValue"] {
        color: #1f2937;
        font-weight: 600;
        font-size: 1.02rem;
    }
    [data-testid="stMetricLabel"] {
        color: #6b7280;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stButton > button {
        background: #238636;
        color: #ffffff;
        border: 1px solid #2ea043;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #2ea043;
        border-color: #3fb950;
        box-shadow: 0 0 0 1px #3fb950;
    }
    .stInfo {
        background-color: #faf8f5;
        border-left: 4px solid #58a6ff;
        border-radius: 6px;
    }
    .stWarning {
        background-color: #faf8f5;
        border-left: 4px solid #d29922;
        border-radius: 6px;
    }
    .stError {
        background-color: #faf8f5;
        border-left: 4px solid #f85149;
        border-radius: 6px;
    }
    .stSuccess {
        background-color: #faf8f5;
        border-left: 4px solid #3fb950;
        border-radius: 6px;
    }
    [data-testid="stExpander"] {
        background-color: #faf8f5;
        border: 1px solid #e8e4dc;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    [data-testid="stExpander"] [data-testid="stExpanderHeader"] {
        background-color: #f5f3ed;
        border-radius: 8px 8px 0 0;
        padding: 12px;
        font-weight: 600;
        color: #374151;
    }
    [data-baseweb="tab-list"] {
        background-color: #faf8f5;
        border-radius: 8px;
        padding: 4px;
    }
    [data-baseweb="tab"] {
        border-radius: 6px;
        font-weight: 500;
    }
    hr {
        border: none;
        height: 1px;
        background: #e8e4dc;
        margin: 30px 0;
    }
    code {
        background-color: #f0ede6;
        color: #4a4a4a;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e8e4dc;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .metric-card {
        background: #faf8f5;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e8e4dc;
        margin-bottom: 15px;
    }
    .js-plotly-plot {
        border-radius: 8px;
        background-color: #faf8f5;
        padding: 10px;
        border: 1px solid #e8e4dc;
    }
    .symbol-link { color: #1a73e8; text-decoration: none; font-weight: 500; }
    .symbol-link:hover { color: #1557b0; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# --- Hugging Face API ---
def get_hf_token():
    """Get Hugging Face token from env or Streamlit secrets."""
    token = os.getenv("HF_TOKEN", "")
    try:
        if hasattr(st, 'secrets') and st.secrets is not None:
            t = st.secrets.get("HF_TOKEN", "")
            if t:
                return t
    except Exception:
        pass
    return token

HF_TOKEN = get_hf_token()

# --- AI summary helpers (trend deltas for LLM context) ---
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


# --- AI stock summary (Hugging Face Llama) ---
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
                "Keep numbers and units intact when absolutely necessary. Ensure text is clean and readable (no LaTeX/special fonts). "
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


# --- Data loading and caching ---
@st.cache_data(ttl=3600)
def load_data():
    """Load signal analysis CSV, parse dates, normalize symbols, downcast numerics."""
    csv_path = os.path.join("Reports", "signal_analysis.csv")
    
    if not os.path.exists(csv_path):
        st.error(f"Data file not found. Expected {csv_path}")
        return None
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Ensure proper data types
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
    df['final_trade'] = df['final_trade'].astype(str).str.strip()
    numeric_cols = df.select_dtypes(include=['float64']).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
    
    if 'is_earnings_date' in df.columns:
        df['is_earnings_date'] = pd.to_numeric(df['is_earnings_date'], errors='coerce').fillna(0).astype(int)
    
    return df


@st.cache_data(ttl=3600)
def get_latest_data(df):
    """Get most recent row per symbol."""
    return df.sort_values('Date', ascending=False).drop_duplicates(subset='Symbol', keep='first')


@st.cache_data(ttl=3600)
def get_top_stocks(latest_data, n=100):
    """Top N stocks by combined_signal, sorted descending."""
    top_stocks = latest_data.nlargest(n, 'combined_signal')[['Symbol', 'combined_signal', 'final_trade', 'Close']].copy()
    return top_stocks.sort_values('combined_signal', ascending=False).reset_index(drop=True)


@st.cache_data(ttl=3600)
def get_gauge_ranges(df):
    """Min/max for each gauge metric (combined_signal, Fundamental_Weight, SentimentScore)."""
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


@st.cache_data(ttl=3600)
def get_available_symbols(df):
    """Sorted list of unique symbols from latest data."""
    latest_data = get_latest_data(df)
    return sorted([str(s).strip().upper() for s in latest_data['Symbol'].unique().tolist()])


@st.cache_data(ttl=3600)
def load_company_analysis():
    """Load complete company analysis Excel (latest quarter sheet)."""
    xlsx_path = os.path.join("Reports", "complete_company_analysis.xlsx")
    if not os.path.exists(xlsx_path):
        return None
    return pd.read_excel(xlsx_path, sheet_name="2_Latest_Quarter_Complete", engine="openpyxl")


def get_company_metrics(ticker, company_df):
    """Get Fair value, PE, PB, Revenue Growth YoY for ticker from company analysis."""
    if company_df is None or company_df.empty:
        return {}
    sym_col = 'Symbol' if 'Symbol' in company_df.columns else 'symbol'
    if sym_col not in company_df.columns:
        return {}
    row = company_df[company_df[sym_col].astype(str).str.strip().str.upper() == str(ticker).strip().upper()]
    if row.empty:
        return {}
    r = row.iloc[0]
    metrics = {}
    for col, key in [
        ('FairValue_Composite', 'fair_value'),
        ('PE_Ratio', 'pe_ratio'),
        ('PB_Ratio', 'pb_ratio'),
        ('RevenueGrowth_YoY', 'revenue_growth_yoy')
    ]:
        if col in r.index and pd.notna(r[col]):
            val = r[col]
            metrics[key] = float(val) if isinstance(val, (int, float)) else val
    return metrics


# --- News sentiment helpers ---
@st.cache_data(ttl=3600)
def load_news_data():
    """Load news CSV from Reports folder."""
    news_csv = os.path.join("Reports", "news_cleaned_df.csv")
    if not os.path.exists(news_csv):
        return None
    return pd.read_csv(news_csv)

def get_news_by_symbol(news_df, symbol):
    """Filter news to rows matching symbol."""
    if news_df is None:
        return pd.DataFrame()
    return news_df[news_df['symbol'] == symbol.upper()].copy()

def group_news_by_sentiment(df):
    """Split news into positive and negative DataFrames by sentiment_label."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    positive_news = df[df['sentiment_label'] == 'positive'].copy()
    negative_news = df[df['sentiment_label'] == 'negative'].copy()
    return positive_news, negative_news

def format_news_for_llm(df, max_articles=20):
    """Format news headlines/summaries as text for LLM input."""
    if df.empty:
        return "No news articles available."
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
    """Generate AI summary of news articles via Hugging Face Llama."""
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
                    f"Provide a concise summary highlighting the main themes and key information from these {sentiment_type} news articles. "
                    f"Ensure that the text is clean and readable. Do not use LaTeX formatting or special fonts for numbers (e.g. use '100' not '$100$'). "
                    f"Make sure words are not broken up and sentences are complete."
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


# --- Main UI ---
st.markdown(
    """
    <div style="background: rgba(220, 218, 215, 0.9); padding: 24px 32px; border-radius: 8px; margin-bottom: 1.5rem; border: 1px solid rgba(180, 178, 175, 0.8);">
        <h1 style="color: #1f2937; margin: 0; font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em;">
            Stock Analysis Report
        </h1>
        <p style="color: #6b7280; font-size: 0.95rem; margin: 0.5rem 0 0 0; font-weight: 400;">
            Technical and fundamental indicators with AI-powered signals
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Load data (cached, spinner on first load) ---
if 'data_loaded' not in st.session_state:
    with st.spinner("Loading data... This may take a moment on first load."):
        df = load_data()
        st.session_state.data_loaded = True
else:
    df = load_data()

if df is not None:
    # --- Initialize session state ---
    available_symbols = get_available_symbols(df)
    latest_data = get_latest_data(df)
    if 'ticker_select' not in st.session_state:
        st.session_state.ticker_select = available_symbols[0] if available_symbols else ''
    else:
        current_ticker = str(st.session_state.ticker_select).strip().upper()
        if current_ticker not in available_symbols:
            st.session_state.ticker_select = available_symbols[0] if available_symbols else ''

    # Handle click from symbol link (?symbol=AAPL in URL)
    q_symbol = str(st.query_params.get("symbol", "") or "").strip().upper()
    if q_symbol and q_symbol in available_symbols:
        st.session_state.ticker_select = q_symbol
        try:
            del st.query_params["symbol"]
        except Exception:
            pass

    top_stocks_main = get_top_stocks(latest_data, n=500)

    # Group by signal (BUY/HOLD/SELL), each sorted by combined_signal descending
    buy_symbols = top_stocks_main[top_stocks_main['final_trade'] == 'BUY'].sort_values('combined_signal', ascending=False)['Symbol'].str.strip().str.upper().tolist()
    hold_symbols = top_stocks_main[top_stocks_main['final_trade'] == 'HOLD'].sort_values('combined_signal', ascending=False)['Symbol'].str.strip().str.upper().tolist()
    sell_symbols = top_stocks_main[top_stocks_main['final_trade'] == 'SELL'].sort_values('combined_signal', ascending=False)['Symbol'].str.strip().str.upper().tolist()

    def make_clickable_list(symbols):
        return ", ".join(f'<a href="?symbol={quote(s)}" class="symbol-link" target="_self">{s}</a>' for s in symbols)

    # --- Symbol lists by signal (clickable, all symbols, sorted by score descending) ---
    st.markdown("---")
    last_updated_str = df['Date'].max().strftime("%m/%d/%Y") if not df.empty and pd.notna(df['Date'].max()) else ""
    with st.expander(f"Stock tickers (click to select). Last updated: {last_updated_str}", expanded=True):
        if buy_symbols:
            st.markdown(f"**🟢 BULLISH:** {make_clickable_list(buy_symbols)}", unsafe_allow_html=True)
        if hold_symbols:
            st.markdown(f"**🟡 HOLD:** {make_clickable_list(hold_symbols)}", unsafe_allow_html=True)
        if sell_symbols:
            st.markdown(f"**🔴 BEARISH:** {make_clickable_list(sell_symbols)}", unsafe_allow_html=True)

        symbol_data = latest_data.set_index('Symbol')
        symbol_signal = symbol_data['combined_signal'].to_dict()
        symbol_streak_label = {}
        for s in available_symbols:
            try:
                buy_streak = int(float(symbol_data.loc[s, 'Buy Streak']) or 0)
                sell_streak = int(float(symbol_data.loc[s, 'Sell Streak']) or 0)
                if buy_streak > 0:
                    symbol_streak_label[s] = f"Buy Streak of {buy_streak} days"
                elif sell_streak > 0:
                    symbol_streak_label[s] = f"Sell Streak of {sell_streak} days"
                else:
                    symbol_streak_label[s] = "Streak = None"
            except (KeyError, TypeError, ValueError):
                symbol_streak_label[s] = "Streak = None"
        dropdown_options = sorted(available_symbols, key=lambda s: symbol_signal.get(s, 0) or 0, reverse=True)
        current_ticker = str(st.session_state.get('ticker_select', available_symbols[0] if available_symbols else '')).strip().upper()
        default_idx = dropdown_options.index(current_ticker) if current_ticker in dropdown_options else 0
        ticker_col, _ = st.columns([2, 4])
        with ticker_col:
            selected = st.selectbox(
                "Enter stock ticker:",
                options=dropdown_options,
                format_func=lambda s: f"{s} : {symbol_streak_label.get(s, 'Streak = None')} (Score = {float(symbol_signal.get(s, 0) or 0):.0f})",
                index=default_idx,
                key="ticker_dropdown"
            )
        if selected:
            st.session_state.ticker_select = selected

    ticker = str(st.session_state.get('ticker_select', available_symbols[0] if available_symbols else '')).strip().upper()

    # --- AI Technical Analysis section ---
    st.markdown(f"""
    <div style="background: rgba(220, 218, 215, 0.9); padding: 16px 24px; border-radius: 8px; margin: 16px 0 16px 0; border: 1px solid rgba(180, 178, 175, 0.8);">
        <h3 style="color: #374151; margin: 0; font-size: 1.25rem; font-weight: 600;">AI Technical and News Analysis for {ticker}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate AI summary and news analysis when button clicked
    if st.button(f"🤖 Generate AI Analysis for {ticker}", type="primary", key="generate_ai_btn"):
        st.session_state.generate_summary = True
    
    if ticker and st.session_state.get('generate_summary', False):
        ticker_data_ai = df[df['Symbol'] == ticker]
        if not ticker_data_ai.empty:
            latest_ai = ticker_data_ai.nlargest(1, 'Date').iloc[0]
            with st.spinner(f"Generating AI summary for {ticker}..."):
                summary = generate_ai_summary(ticker, latest_ai, df)
            with st.expander(f"🤖 AI Summary for {ticker}", expanded=True):
                st.markdown(summary)
            news_df = load_news_data()
            if news_df is not None:
                symbol_news = get_news_by_symbol(news_df, ticker)
                if not symbol_news.empty:
                    positive_news, negative_news = group_news_by_sentiment(symbol_news)
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
        st.session_state.generate_summary = False

    # --- Human Technical Analysis section (when ticker selected) ---
    if ticker:
        ticker_data = df[df['Symbol'] == ticker].copy()
        
        if len(ticker_data) == 0:
            st.warning(f"❌ Ticker '{ticker}' not found in database.")
            st.info(f"Available tickers: {', '.join(available_symbols[:20])}..." if len(available_symbols) > 20 else f"Available tickers: {', '.join(available_symbols)}")
        else:
            latest = ticker_data.nlargest(1, 'Date').iloc[0]
            st.markdown(f"""
            <div style="background: rgba(220, 218, 215, 0.9); padding: 16px 24px; border-radius: 8px; margin: 24px 0 16px 0; border: 1px solid rgba(180, 178, 175, 0.8);">
                <h3 style="color: #374151; margin: 0; font-size: 1.25rem; font-weight: 600;">Human Technical Analysis for {ticker}</h3>
            </div>
            """, unsafe_allow_html=True)

            # Signal metrics row: Technical Signal, Current Price, Combined Signal, Buy/Sell Streak
            signal_raw = latest['final_trade']
            signal_mapping = {
                'BUY': 'BULLISH',
                'SELL': 'BEARISH',
                'HOLD': 'HOLD'
            }
            signal = signal_mapping.get(signal_raw, signal_raw)
            
            signal_color = {
                'BULLISH': '🟢',
                'BEARISH': '🔴',
                'HOLD': '🟡'
            }
            signal_bg = {
                'BULLISH': 'background-color: rgba(35, 134, 54, 0.15); color: #1a7f37; border: 1px solid #238636;',
                'BEARISH': 'background-color: rgba(248, 81, 73, 0.15); color: #c41e3a; border: 1px solid #da3633;',
                'HOLD': 'background-color: rgba(210, 153, 34, 0.15); color: #9e6a03; border: 1px solid #9e6a03;'
            }
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown("""
                <div style="font-size: 0.875rem; color: #6b7280; font-weight: 500; margin-bottom: 4px;">
                    Technical Signal
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="text-align: center; padding: 8px 12px; border-radius: 6px; {signal_bg.get(signal, '')} font-size: 0.95rem; line-height: 1.5;">
                    <strong>{signal_color.get(signal, '')} {signal}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                buy_streak = latest.get('Buy Streak', 0)
                st.metric("Buy Streak", f"{int(buy_streak)} days")
            
            with col3:
                sell_streak = latest.get('Sell Streak', 0)
                st.metric("Sell Streak", f"{int(sell_streak)} days")

            # Gauge charts: Combined Signal, Balance Sheet, News Sentiment
            gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
            gauge_ranges = get_gauge_ranges(df)
            
            combined_signal_min = gauge_ranges.get('combined_signal', {}).get('min', 0)
            combined_signal_max = gauge_ranges.get('combined_signal', {}).get('max', 100)
            combined_signal_val = float(latest['combined_signal']) if 'combined_signal' in latest.index and pd.notna(latest['combined_signal']) else 0
            combined_signal_mean = df['combined_signal'].mean()
            
            fundamental_weight_min = gauge_ranges.get('Fundamental_Weight', {}).get('min', 0)
            fundamental_weight_max = gauge_ranges.get('Fundamental_Weight', {}).get('max', 100)
            fundamental_weight_val = float(latest['Fundamental_Weight']) if 'Fundamental_Weight' in latest.index and pd.notna(latest['Fundamental_Weight']) else 0
            fundamental_weight_mean = df['Fundamental_Weight'].mean()
            
            sentiment_score_min = gauge_ranges.get('SentimentScore', {}).get('min', -10)
            sentiment_score_max = gauge_ranges.get('SentimentScore', {}).get('max', 10)
            sentiment_score_val = float(latest['SentimentScore']) if 'SentimentScore' in latest.index and pd.notna(latest['SentimentScore']) else 0
            sentiment_score_mean = df['SentimentScore'].mean()
            with gauge_col1:
                combined_signal_mid = round(combined_signal_mean, 2)
                fig1 = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = combined_signal_val,
                    domain = {'x': [0, 1], 'y': [0, 0.5]},
                    title = {'text': "Combined Signal", 'font': {'color': '#374151'}},
                    number = {'font': {'color': '#1f2937'}},
                    gauge = {
                        'axis': {
                            'range': [combined_signal_min, combined_signal_max],
                            'tickmode': 'array',
                            'tickvals': [combined_signal_min, combined_signal_mid, combined_signal_max],
                            'ticktext': [f'Min: {combined_signal_min:.1f}', f'Mid: {combined_signal_mid:.1f}', f'Max: {combined_signal_max:.1f}'],
                            'tickcolor': '#6b7280',
                            'tickfont': {'color': '#6b7280'}
                        },
                        'bar': {'color': "#58a6ff"},
                        'steps': [
                            {'range': [combined_signal_min, combined_signal_val], 'color': "#388bfd"},
                            {'range': [combined_signal_val, combined_signal_max], 'color': "#e8e4dc"}
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
                    paper_bgcolor="#faf8f5",
                    plot_bgcolor="#faf8f5",
                    annotations=[dict(
                        x=0.5, y=-0.15,
                        text=f'Current: {combined_signal_val:.2f}',
                        showarrow=False,
                        font=dict(size=12, color="#6b7280")
                    )]
                )
                st.plotly_chart(fig1, use_container_width=True)
            with gauge_col2:
                company_df = load_company_analysis()
                comp = get_company_metrics(ticker, company_df)
                fv = comp.get('fair_value', 'N/A')
                pe = comp.get('pe_ratio', 'N/A')
                pb = comp.get('pb_ratio', 'N/A')
                rev = comp.get('revenue_growth_yoy', 'N/A')
                fv_str = f"${fv:.2f}" if isinstance(fv, (int, float)) else str(fv)
                pe_str = f"{pe:.1f}" if isinstance(pe, (int, float)) else str(pe)
                pb_str = f"{pb:.2f}" if isinstance(pb, (int, float)) else str(pb)
                rev_str = f"{rev:.1f}%" if isinstance(rev, (int, float)) else str(rev)
                metrics_line1 = f"Fair value: {fv_str}, PE ratio: {pe_str}"
                metrics_line2 = f"PB ratio: {pb_str}, Revenue Growth YoY: {rev_str}"
                fundamental_weight_mid = round(fundamental_weight_mean, 2)
                fig2 = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = fundamental_weight_val,
                    domain = {'x': [0, 1], 'y': [0, 0.5]},
                    title = {'text': "Balance Sheet", 'font': {'color': '#374151'}},
                    number = {'font': {'color': '#1f2937'}},
                    gauge = {
                        'axis': {
                            'range': [fundamental_weight_min, fundamental_weight_max],
                            'tickmode': 'array',
                            'tickvals': [fundamental_weight_min, fundamental_weight_mid, fundamental_weight_max],
                            'ticktext': [f'Min: {fundamental_weight_min:.1f}', f'Mid: {fundamental_weight_mid:.1f}', f'Max: {fundamental_weight_max:.1f}'],
                            'tickcolor': '#6b7280',
                            'tickfont': {'color': '#6b7280'}
                        },
                        'bar': {'color': "#58a6ff"},
                        'steps': [
                            {'range': [fundamental_weight_min, fundamental_weight_val], 'color': "#388bfd"},
                            {'range': [fundamental_weight_val, fundamental_weight_max], 'color': "#e8e4dc"}
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
                    height=240,
                    margin=dict(l=20, r=20, t=40, b=85),
                    paper_bgcolor="#faf8f5",
                    plot_bgcolor="#faf8f5",
                    annotations=[
                        dict(x=0.5, y=-0.12, text=f'Current: {fundamental_weight_val:.2f}', showarrow=False, font=dict(size=12, color="#6b7280")),
                        dict(x=0.5, y=-0.28, text=metrics_line1, showarrow=False, font=dict(size=11, color="#6b7280")),
                        dict(x=0.5, y=-0.40, text=metrics_line2, showarrow=False, font=dict(size=11, color="#6b7280"))
                    ]
                )
                st.plotly_chart(fig2, use_container_width=True)
            with gauge_col3:
                news_df = load_news_data()
                symbol_news = get_news_by_symbol(news_df, ticker) if news_df is not None else pd.DataFrame()
                pos_news, neg_news = group_news_by_sentiment(symbol_news) if not symbol_news.empty else (pd.DataFrame(), pd.DataFrame())
                pos_count = len(pos_news)
                neg_count = len(neg_news)
                sentiment_score_mid = round(sentiment_score_mean, 2)
                fig3 = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = sentiment_score_val,
                    domain = {'x': [0, 1], 'y': [0, 0.5]},
                    title = {'text': "News Sentiment", 'font': {'color': '#374151'}},
                    number = {'font': {'color': '#1f2937'}},
                    gauge = {
                        'axis': {
                            'range': [sentiment_score_min, sentiment_score_max],
                            'tickmode': 'array',
                            'tickvals': [sentiment_score_min, sentiment_score_mid, sentiment_score_max],
                            'ticktext': [f'Min: {sentiment_score_min:.1f}', f'Mid: {sentiment_score_mid:.1f}', f'Max: {sentiment_score_max:.1f}'],
                            'tickcolor': '#6b7280',
                            'tickfont': {'color': '#6b7280'}
                        },
                        'bar': {'color': "#58a6ff"},
                        'steps': [
                            {'range': [sentiment_score_min, sentiment_score_val], 'color': "#388bfd"},
                            {'range': [sentiment_score_val, sentiment_score_max], 'color': "#e8e4dc"}
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
                    height=220,
                    margin=dict(l=20, r=20, t=40, b=70),
                    paper_bgcolor="#faf8f5",
                    plot_bgcolor="#faf8f5",
                    annotations=[
                        dict(x=0.5, y=-0.15, text=f'Current: {sentiment_score_val:.2f}', showarrow=False, font=dict(size=12, color="#6b7280")),
                        dict(x=0.5, y=-0.32, text=f'Positive: {pos_count}, Negative: {neg_count}', showarrow=False, font=dict(size=13, color="#6b7280"))
                    ]
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Filter to last 1 year and build price chart
            latest_date = ticker_data['Date'].max()
            one_year_ago = latest_date - pd.Timedelta(days=365)
            ticker_data_sorted = ticker_data[ticker_data['Date'] >= one_year_ago].sort_values('Date', ascending=True)
            if len(ticker_data_sorted) > 0:
                # Price/MA/RSI metrics row
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                close = latest.get('Close', 0)
                if pd.notna(close):
                    with col1:
                        st.metric("Price", f"${close:.2f}", delta=None)
                with col2:
                    ma_10 = latest.get('ma_10', 0)
                    if pd.notna(ma_10):
                        st.metric("MA 10", f"${ma_10:.2f}", delta=None)
                with col3:
                    ma_30 = latest.get('ma_30', 0)
                    if pd.notna(ma_30):
                        st.metric("MA 30", f"${ma_30:.2f}", delta=None)
                with col4:
                    ma_50 = latest.get('ma_50', 0)
                    if pd.notna(ma_50):
                        st.metric("MA 50", f"${ma_50:.2f}", delta=None)
                with col5:
                    ma_100 = latest.get('ma_100', 0)
                    if pd.notna(ma_100):
                        st.metric("MA 100", f"${ma_100:.2f}", delta=None)
                with col6:
                    ma_200 = latest.get('ma_200', 0)
                    if pd.notna(ma_200):
                        st.metric("MA 200", f"${ma_200:.2f}", delta=None)
                with col7:
                    rsi = latest.get('RSI Options Rate', 0)
                    if pd.notna(rsi):
                        st.metric("RSI", f"{rsi:.2f}", delta=None)
                # Prepare chart data: price, MAs, signal, streaks, earnings
                cols_to_select = ['Date', 'Close', 'ma_10', 'ma_30', 'ma_50', 'ma_100', 'ma_200',
                                 'combined_signal', 'Buy Streak', 'Sell Streak']
                if 'is_earnings_date' in ticker_data_sorted.columns:
                    cols_to_select.append('is_earnings_date')
                price_data = ticker_data_sorted[cols_to_select].copy()
                price_data = price_data.set_index('Date')
                price_data = price_data.dropna(subset=['Close'])
                price_data['combined_signal'] = price_data['combined_signal'].fillna(0)
                price_data['Buy Streak'] = price_data['Buy Streak'].fillna(0)
                price_data['Sell Streak'] = price_data['Sell Streak'].fillna(0)
                if 'is_earnings_date' in price_data.columns:
                    price_data['is_earnings_date'] = pd.to_numeric(price_data['is_earnings_date'], errors='coerce').fillna(0).astype(int)
                has_rsi = 'RSI Options Rate' in ticker_data_sorted.columns and not ticker_data_sorted[['Date', 'RSI Options Rate']].dropna().empty
                has_macd = ('macd' in ticker_data_sorted.columns and 'MACD Signal' in ticker_data_sorted.columns and
                           not ticker_data_sorted[['Date', 'macd', 'MACD Signal']].dropna().empty)

                # Build subplot layout (Price + optional RSI + optional MACD)
                num_rows = 1
                row_heights = [1.0]
                subplot_titles = ['<b>Price and Moving Averages</b>']
                
                if has_rsi:
                    num_rows += 1
                    row_heights = [0.65, 0.35]
                    subplot_titles.append('')
                
                if has_macd:
                    num_rows += 1
                    if has_rsi:
                        row_heights = [0.65, 0.175, 0.175]
                    else:
                        row_heights = [0.65, 0.35]
                    subplot_titles.append('')
                fig = make_subplots(
                    rows=num_rows, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=row_heights,
                    subplot_titles=subplot_titles
                )
                fig.update_annotations(
                    font=dict(size=13, color='#374151', family='Arial, sans-serif'),
                    yshift=5
                )

                # Row 1: Close price line with hover (signal, streaks)
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
                    line=dict(color='#27ae60', width=2),
                    mode='lines',
                    customdata=hover_text_close,
                    hovertemplate='<b>Close</b><br>$%{y:.2f}<br>%{customdata}<extra></extra>'
                ), row=1, col=1)

                # Add earnings date markers on price chart
                if 'is_earnings_date' not in price_data.columns and 'is_earnings_date' in ticker_data_sorted.columns:
                    earnings_col = ticker_data_sorted.set_index('Date')['is_earnings_date'].reindex(price_data.index).fillna(0)
                    price_data['is_earnings_date'] = pd.to_numeric(earnings_col, errors='coerce').fillna(0).astype(int)
                if 'is_earnings_date' in price_data.columns:
                    price_data['is_earnings_date'] = pd.to_numeric(price_data['is_earnings_date'], errors='coerce').fillna(0).astype(int)
                    earnings_dates = price_data[price_data['is_earnings_date'] == 1]
                    if len(earnings_dates) > 0:
                        fig.add_trace(go.Scatter(
                            x=earnings_dates.index,
                            y=earnings_dates['Close'],
                            name='Earnings Date',
                            mode='markers',
                            marker=dict(symbol='circle', size=10, color='#ff6b35', line=dict(color='#ff6b35', width=0), opacity=1.0),
                            hovertemplate='<b>Earnings Date</b><br>$%{y:.2f}<br>%{x|%b %d, %Y}<extra></extra>',
                            showlegend=True,
                            legendgroup='earnings'
                        ), row=1, col=1)

                # Add moving average lines (MA 10, 30, 50, 100, 200)
                for ma_name, ma_col, color in [
                    ('ma_10', 'ma_10', '#ffd700'),
                    ('ma_30', 'ma_30', '#e74c3c'),
                    ('ma_50', 'ma_50', '#3498db'),
                    ('ma_100', 'ma_100', '#8b4513'),
                    ('ma_200', 'ma_200', '#808080')
                ]:
                    if ma_col in price_data.columns:
                        fig.add_trace(go.Scatter(
                            x=price_data.index,
                            y=price_data[ma_col],
                            name=ma_name.upper().replace('_', ' '),
                            line=dict(color=color, width=1),
                            mode='lines',
                            showlegend=True,
                            hovertemplate=f'<b>{ma_name.upper().replace("_", " ")}</b><br>$%{{y:.2f}}<extra></extra>'
                        ), row=1, col=1)

                # Buy/Sell/Hold streak horizontal lines above price chart
                if 'Buy Streak' in ticker_data_sorted.columns and 'Sell Streak' in ticker_data_sorted.columns:
                    max_price = price_data['Close'].max()
                    min_price = price_data['Close'].min()
                    price_range = max_price - min_price
                    top_y_price = max_price + (price_range * 0.05)
                    streak_data = ticker_data_sorted[['Date', 'Buy Streak', 'Sell Streak']].copy()
                    streak_data = streak_data.set_index('Date')
                    streak_data = streak_data.reindex(price_data.index).fillna(0)
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
                current_row = 2
                if has_rsi:
                    rsi_data = ticker_data_sorted[['Date', 'RSI Options Rate']].copy()
                    rsi_data = rsi_data.set_index('Date')
                    rsi_data = rsi_data.dropna()
                    
                    if not rsi_data.empty:
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
                        ), row=current_row, col=1)
                        
                        fig.add_hline(y=70, line_dash="dash", line_color="rgba(200, 0, 0, 0.3)", row=current_row, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0, 200, 0, 0.3)", row=current_row, col=1)
                    current_row += 1
                elif has_macd:
                    current_row = 2

                # Row 3: MACD chart (if data available)
                if has_macd:
                    macd_data = ticker_data_sorted[['Date', 'macd', 'MACD Signal']].copy()
                    macd_data = macd_data.set_index('Date')
                    macd_data = macd_data.dropna()
                    
                    if not macd_data.empty:
                        fig.add_trace(go.Scatter(
                            x=macd_data.index,
                            y=macd_data['macd'],
                            name='MACD',
                            line=dict(color='#d62728', width=1.8),
                            mode='lines',
                            showlegend=False,
                            hovertemplate='<b>MACD</b><br>%{y:.4f}<extra></extra>'
                        ), row=current_row, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=macd_data.index,
                            y=macd_data['MACD Signal'],
                            name='MACD Signal',
                            line=dict(color='#1f77b4', width=1.8, dash='dash'),
                            mode='lines',
                            showlegend=False,
                            hovertemplate='<b>MACD Signal</b><br>%{y:.4f}<extra></extra>'
                        ), row=current_row, col=1)
                        fig.add_hline(y=0, line_dash="dot", line_color="rgba(128, 128, 128, 0.4)", line_width=1, row=current_row, col=1)

                # Chart layout, axes, and display
                fig.update_layout(
                    height=850,
                    hovermode='x unified',
                    margin=dict(l=50, r=50, t=120, b=50),
                    plot_bgcolor='#faf8f5',
                    paper_bgcolor='#f8f6f0',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.08,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10, color='#374151'),
                        bgcolor='rgba(250, 248, 245, 0.95)',
                        bordercolor='#e8e4dc',
                        borderwidth=1,
                        itemwidth=30
                    ),
                    font=dict(family="Arial, sans-serif", size=11, color='#374151'),
                    dragmode=False,
                    hoverlabel=dict(
                        bgcolor="#faf8f5",
                        bordercolor="#e8e4dc",
                        font_size=11,
                        font_family="Arial, sans-serif"
                    )
                )
                fig.update_xaxes(
                    tickformat='%b %Y',
                    showspikes=True,
                    spikecolor="#6b7280",
                    spikesnap="cursor",
                    spikemode="across",
                    spikethickness=1,
                    spikedash="solid",
                    showgrid=True,
                    gridcolor='rgba(200, 198, 195, 0.35)',
                    gridwidth=1,
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(200, 198, 195, 0.4)',
                    linewidth=1,
                    tickfont=dict(size=10, color='#6b7280'),
                    title_font=dict(size=12, color='#374151')
                )
                fig.update_yaxes(
                    title_text="Price ($)",
                    title_font=dict(size=11, color='#374151'),
                    showspikes=True,
                    spikecolor="#6b7280",
                    spikesnap="cursor",
                    spikemode="toaxis",
                    spikethickness=1,
                    spikedash="solid",
                    showgrid=True,
                    gridcolor='rgba(200, 198, 195, 0.35)',
                    gridwidth=1,
                    zeroline=False,
                    showline=True,
                    linecolor='rgba(200, 198, 195, 0.4)',
                    linewidth=1,
                    tickfont=dict(size=10, color='#6b7280'),
                    tickformat='$,.0f',
                    row=1, col=1
                )
                if has_rsi:
                    fig.update_yaxes(
                        title_text="RSI",
                        title_font=dict(size=11, color='#374151'),
                        showspikes=True, spikecolor="#6b7280", spikesnap="cursor", spikemode="toaxis",
                        spikethickness=1, spikedash="solid",
                        showgrid=True, gridcolor='rgba(200, 198, 195, 0.35)', gridwidth=1,
                        zeroline=False, showline=True, linecolor='rgba(200, 198, 195, 0.4)', linewidth=1,
                        tickfont=dict(size=10, color='#6b7280'), range=[0, 100],
                        row=2, col=1
                    )
                if has_macd:
                    macd_row = 3 if has_rsi else 2
                    fig.update_yaxes(
                        title_text="MACD",
                        title_font=dict(size=11, color='#374151'),
                        showspikes=True, spikecolor="#6b7280", spikesnap="cursor", spikemode="toaxis",
                        spikethickness=1, spikedash="solid",
                        showgrid=True, gridcolor='rgba(200, 198, 195, 0.35)', gridwidth=1,
                        zeroline=True, zerolinecolor='rgba(200, 198, 195, 0.4)', zerolinewidth=1,
                        showline=True, linecolor='rgba(200, 198, 195, 0.4)', linewidth=1,
                        tickfont=dict(size=10, color='#6b7280'),
                        row=macd_row, col=1
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

