import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import re
# Load environment variables from .env file
load_dotenv()

# Hugging Face Configuration
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    HF_TOKEN = os.getenv("HF_TOKEN", "")

# Page config
st.set_page_config(
    page_title="Stock Signal Lookup",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def build_trend_deltas(df, ticker, windows=(14, 50, 200)):
    """
    Build compact trend-difference rows for LLM context.
    Returns a list of dicts, one per window.
    """
    recent = df[df["Symbol"] == ticker].sort_values("Date")
    latest = recent.iloc[-1]

    rows = []

    for w in windows:
        if len(recent) < w:
            continue

        past = recent.iloc[-w]

        row = {
            "window_days": w,
            "price_change_pct": round(
                (latest["Close"] / past["Close"] - 1) * 100, 2
            ),
            "rsi_change": round(
                latest["RSI Options Rate"] - past["RSI Options Rate"], 2
            ),
            "macd_change": round(
                latest["macd"] - past["macd"], 4
            ),
            "ma30_diff_pct": round(
                (latest["Close"] - latest["ma_30"]) / latest["ma_30"] * 100, 2
            ),
            "ma200_diff_pct": round(
                (latest["Close"] - latest["ma_200"]) / latest["ma_200"] * 100, 2
            ),
            "above_ma200_days_pct": round(
                (recent.tail(w)["Close"] > recent.tail(w)["ma_200"]).mean() * 100, 1
            )
        }

        rows.append(row)

    return rows


def format_trend_rows(trend_rows):
    text = "Trend Deltas:\n"
    for r in trend_rows:
        text += (
            f"Last {r['window_days']} days: "
            f"Price {r['price_change_pct']}%, "
            f"RSI change {r['rsi_change']}, "
            f"MACD change {r['macd_change']}, "
            f"Price vs MA30 {r['ma30_diff_pct']}%, "
            f"Price vs MA200 {r['ma200_diff_pct']}%, "
            f"Above MA200 {r['above_ma200_days_pct']}% of days\n"
        )
    return text

# st.text("Using model: meta-llama/Meta-Llama-3-8B-Instruct")

def generate_ai_summary(ticker, stock_data, df):
    """Generate AI summary using Hugging Face Llama-3 model"""
    try:
        client = InferenceClient(token=HF_TOKEN)

        context = f"""Stock: {ticker}
                    Date: {stock_data['Date'].strftime('%Y-%m-%d')}
                    Price: ${stock_data['Close']:.2f}
                    Signal: {stock_data['final_trade']}
                    Combined Score: {stock_data['combined_signal']:.2f}
                    RSI: {stock_data.get('RSI Options Rate', 'N/A')}
                    MACD: {stock_data.get('macd', 'N/A')}
                    MA 30: ${stock_data.get('ma_30', 'N/A')}
                    MA 50: ${stock_data.get('ma_50', 'N/A')}
                    MA 100: ${stock_data.get('ma_100', 'N/A')}
                    MA 200: ${stock_data.get('ma_200', 'N/A')}
                    Balance Sheet Score: {stock_data.get('Fundamental_Weight', 'N/A')}
                    """
        # Add trend analysis to context
        trend_rows = build_trend_deltas(df, ticker, windows=(14, 50, 200))
        trend_text = format_trend_rows(trend_rows)

        context += f"\n{trend_text}"


        messages = [
            {"role": "system",
            "content": ("You are a financial advisor. Respond in 4 bullet points and each bullet point must be on a new line. Keep numbers and units intact." 
                    "Each bullet point must be on a new line. Do not split words. Do not use dashes." 
                    "Always double check below or above context for the stock price. example: The Moving Averages analysis shows that MA 10, MA 30, MA 100, and MA 200 are below the current price, and MA 50 is above the current price. This suggests a short-term bullish trend and a long-term bearish trend."
                    "Make sure words are not broken up."
                    "In a new paragraph, ending statement must include a bullish or bearish recommendation with reasoning starting with: AI Opinion: ")},
            {"role": "user",
            "content": (
                    "Analyze stock data. Focus on Signal, Moving Averages, Balance Sheet Score, and Trend Analysis."
                    "Balance Sheet Score ranges from -15 (worst) to 15 (best)."
                    "Complete the analysis with a bullish or bearish recommendation with reasoning."
                    f"{context}")}
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


# Load data
@st.cache_data(ttl=3600)  # Cache for 1 hour, or clear cache in Streamlit Cloud settings
def load_data():
    # Try CSV first (for > 100 symbols), then Excel (for <= 100 symbols)
    csv_path = os.path.join("Reports", "signal_analysis.csv")
    excel_path = os.path.join("Reports", "signal_analysis.xlsx")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        print("df loaded from csv")
        return df
    elif os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        df['Date'] = pd.to_datetime(df['Date'])
        print("df loaded from excel")
        return df
    else:
        st.error(f"Data file not found. Expected {csv_path} or {excel_path}")
        return None

# Main app
st.title("📈 Stock Signal Lookup")
st.markdown("Enter a ticker symbol to see the latest trading signal")

df = load_data()

# Sidebar with top stocks (always visible)
with st.sidebar:
    st.header("🏆 Top Stocks by Signal")
    
    if df is not None:
        # Get latest data for each symbol (most recent date)
        latest_data = df.sort_values('Date', ascending=False).drop_duplicates(subset='Symbol', keep='first')
        top_stocks = latest_data.nlargest(100, 'combined_signal')[['Symbol', 'combined_signal', 'final_trade', 'Close']].copy()
        top_stocks = top_stocks.sort_values('combined_signal', ascending=False).reset_index(drop=True)
        
        # Filter by signal
        filter_option = st.selectbox(
            "Filter by Signal",
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
                f"{signal_emoji} {stock['Symbol']} | Score: {stock['combined_signal']:.1f}",
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
    # Get tickers from the most recent date only
    latest_date = df['Date'].max()
    latest_date_df = df[df['Date'] == latest_date]
    available_symbols = sorted(latest_date_df['Symbol'].unique().tolist())
    
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
            ticker_data_temp = df[df['Symbol'] == display_ticker].copy()
            if not ticker_data_temp.empty:
                latest_temp = ticker_data_temp.sort_values('Date', ascending=False).iloc[0]
                date_str = latest_temp['Date'].strftime("%m/%d/%Y")
                st.markdown(f"""
                <div style="font-size: 1.25rem; padding-top: 0.5rem; padding-bottom: 0.5rem;">
                    <strong>{display_ticker}</strong> | {date_str}
                </div>
                """, unsafe_allow_html=True)
    
    # Display AI Summary if requested
    if st.session_state.get('generate_summary', False) and ticker:
        ticker_data = df[df['Symbol'] == ticker].copy()
        if not ticker_data.empty:
            latest = ticker_data.sort_values('Date', ascending=False).iloc[0]
            with st.spinner(f"Generating AI summary for {ticker}..."):
                summary = generate_ai_summary(ticker, latest, df)
            
            with st.expander(f"🤖 AI Summary for {ticker}", expanded=True):
                st.markdown(summary)
            
            st.session_state.generate_summary = False  # Reset flag
    
    if ticker:
        # Filter data for the ticker
        ticker_data = df[df['Symbol'] == ticker].copy()
        
        if len(ticker_data) == 0:
            st.warning(f"❌ Ticker '{ticker}' not found in database.")
            st.info(f"Available tickers: {', '.join(available_symbols[:20])}..." if len(available_symbols) > 20 else f"Available tickers: {', '.join(available_symbols)}")
        else:
            # Get latest data (most recent date)
            latest = ticker_data.sort_values('Date', ascending=False).iloc[0]
            
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
            
            # Get min/max values from all data for proper gauge scaling
            combined_signal_min = df['combined_signal'].min() if 'combined_signal' in df.columns else 0
            combined_signal_max = df['combined_signal'].max() if 'combined_signal' in df.columns else 100
            combined_signal_val = float(latest['combined_signal']) if 'combined_signal' in latest.index and pd.notna(latest['combined_signal']) else 0
            
            fundamental_weight_min = float(df['Fundamental_Weight'].min()) if 'Fundamental_Weight' in df.columns else 0
            fundamental_weight_max = float(df['Fundamental_Weight'].max()) if 'Fundamental_Weight' in df.columns else 100
            # Access the value - latest is a Series from iloc[0]
            fundamental_weight_val = float(latest['Fundamental_Weight']) if 'Fundamental_Weight' in latest.index and pd.notna(latest['Fundamental_Weight']) else 0
            
            sentiment_score_min = float(df['SentimentScore'].min()) if 'SentimentScore' in df.columns else -10
            sentiment_score_max = float(df['SentimentScore'].max()) if 'SentimentScore' in df.columns else 10
            # Access the value - latest is a Series from iloc[0]
            sentiment_score_val = float(latest['SentimentScore']) if 'SentimentScore' in latest.index and pd.notna(latest['SentimentScore']) else 0
            
            # Gauge 1: Combined Signal
            with gauge_col1:
                combined_signal_mid = (combined_signal_min + combined_signal_max) / 2
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
                fundamental_weight_mid = (fundamental_weight_min + fundamental_weight_max) / 2
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
                sentiment_score_mid = (sentiment_score_min + sentiment_score_max) / 2
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
            
            # Filter to last 1 year of data
            latest_date = ticker_data['Date'].max()
            one_year_ago = latest_date - pd.Timedelta(days=365)
            ticker_data_filtered = ticker_data[ticker_data['Date'] >= one_year_ago].copy()
            ticker_data_sorted = ticker_data_filtered.sort_values('Date', ascending=True).copy()
            
            # Charts section - stacked subplots
            st.markdown("### 📈 Technical Analysis Charts (Last 1 Year)")
            
            # Secondary metrics - subheading size (below chart title)
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            
            with col1:
                ma_10 = latest.get('ma_10', 0)
                if pd.notna(ma_10):
                    st.markdown(f'MA 10: ${ma_10:.2f}')
            
            with col2:
                ma_30 = latest.get('ma_30', 0)
                if pd.notna(ma_30):
                    st.markdown(f'MA 30: ${ma_30:.2f}')
            
            with col3:
                ma_50 = latest.get('ma_50', 0)
                if pd.notna(ma_50):
                    st.markdown(f'MA 50: ${ma_50:.2f}')
            
            with col4:
                ma_100 = latest.get('ma_100', 0)
                if pd.notna(ma_100):
                    st.markdown(f'MA 100: ${ma_100:.2f}')
            
            with col5:
                ma_200 = latest.get('ma_200', 0)
                if pd.notna(ma_200):
                    st.markdown(f'MA 200: ${ma_200:.2f}')
            
            with col6:
                rsi = latest.get('RSI Options Rate', 0)
                if pd.notna(rsi):
                    st.markdown(f'RSI: {rsi:.2f}')
            
            # Prepare data
            price_data = ticker_data_sorted[['Date', 'Close', 'ma_10', 'ma_30', 'ma_50', 'ma_100', 'ma_200']].copy()
            price_data = price_data.set_index('Date')
            price_data = price_data.dropna(subset=['Close'])
            
            # Create subplots: 3 rows, 1 column
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.65, 0.175, 0.175],
                subplot_titles=('Price and Moving Averages', 'RSI', 'MACD')
            )
            
            # Row 1: Price and Moving Averages
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Close'],
                name='Close',
                line=dict(color='green', width=3),
                mode='lines'
            ), row=1, col=1)
            
            if 'ma_10' in price_data.columns:
                fig.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['ma_10'],
                    name='MA 10',
                    line=dict(color='rgba(255, 140, 0, 0.6)', width=0.8),
                    mode='lines',
                    showlegend=True
                ), row=1, col=1)
            
            if 'ma_30' in price_data.columns:
                fig.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['ma_30'],
                    name='MA 30',
                    line=dict(color='rgba(255, 0, 0, 0.6)', width=0.8),
                    mode='lines',
                    showlegend=True
                ), row=1, col=1)
            
            if 'ma_50' in price_data.columns:
                fig.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['ma_50'],
                    name='MA 50',
                    line=dict(color='rgba(0, 0, 255, 0.6)', width=0.8),
                    mode='lines',
                    showlegend=True
                ), row=1, col=1)
            
            if 'ma_100' in price_data.columns:
                fig.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['ma_100'],
                    name='MA 100',
                    line=dict(color='rgba(0, 128, 0, 0.6)', width=0.8),
                    mode='lines',
                    showlegend=True
                ), row=1, col=1)
            
            if 'ma_200' in price_data.columns:
                fig.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['ma_200'],
                    name='MA 200',
                    line=dict(color='rgba(128, 128, 128, 0.6)', width=0.8),
                    mode='lines',
                    showlegend=True
                ), row=1, col=1)
            
            # Row 2: RSI
            if 'RSI Options Rate' in ticker_data_sorted.columns:
                rsi_data = ticker_data_sorted[['Date', 'RSI Options Rate']].copy()
                rsi_data = rsi_data.set_index('Date')
                rsi_data = rsi_data.dropna()
                
                fig.add_trace(go.Scatter(
                    x=rsi_data.index,
                    y=rsi_data['RSI Options Rate'],
                    name='RSI',
                    line=dict(color='orange', width=1.5),
                    mode='lines',
                    showlegend=False
                ), row=2, col=1)
            
            # Row 3: MACD
            if 'macd' in ticker_data_sorted.columns and 'MACD Signal' in ticker_data_sorted.columns:
                macd_data = ticker_data_sorted[['Date', 'macd', 'MACD Signal']].copy()
                macd_data = macd_data.set_index('Date')
                macd_data = macd_data.dropna()
                
                fig.add_trace(go.Scatter(
                    x=macd_data.index,
                    y=macd_data['macd'],
                    name='MACD',
                    line=dict(color='red', width=1.5),
                    mode='lines',
                    showlegend=False
                ), row=3, col=1)
                
                fig.add_trace(go.Scatter(
                    x=macd_data.index,
                    y=macd_data['MACD Signal'],
                    name='MACD Signal',
                    line=dict(color='skyblue', width=1.5),
                    mode='lines',
                    showlegend=False
                ), row=3, col=1)
            
            # Update layout
            fig.update_layout(
                height=800,
                hovermode='x unified',
                margin=dict(l=0, r=0, t=30, b=10),  # Minimal margins - only bottom for date labels
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                dragmode=False  # Disable drag/zoom on mobile
            )
            
            # Update x-axes (all charts show same dates - months only)
            fig.update_xaxes(
                tickformat='%b %Y'  # Show months only (e.g., "Jan 2025")
            )
            
            # Update y-axis labels - removed to increase chart size on mobile
            fig.update_yaxes(title_text="", row=1, col=1)
            fig.update_yaxes(title_text="", row=2, col=1)
            fig.update_yaxes(title_text="", row=3, col=1)
            
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

