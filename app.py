"""
Stock Analysis Report - Streamlit app for technical/fundamental analysis with AI-powered signals.
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
from datetime import datetime
from zoneinfo import ZoneInfo
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

# --- Custom CSS (slate / teal — light financial workspace) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

    .stApp, .main { background: #f1f5f9; }
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
        background: transparent;
    }
    .stApp > header, header[data-testid="stHeader"], [data-testid="stDecoration"] {
        display: none !important; height: 0 !important; min-height: 0 !important;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }

    h1, h2, h3 { color: #0f172a; font-weight: 600; letter-spacing: -0.02em; border: none; padding: 0; }

    [data-testid="stMetricValue"] {
        color: #0f172a; font-weight: 700; font-size: 1.05rem;
        font-family: 'JetBrains Mono', monospace;
    }
    [data-testid="stMetricLabel"] { color: #64748b; font-weight: 500; font-size: 0.8rem; }

    .stButton > button {
        background: #0f766e; color: #fff; border: none; border-radius: 10px;
        font-weight: 600; padding: 0.5rem 1.1rem; transition: background 0.15s ease;
    }
    .stButton > button:hover { background: #0d9488; color: #fff; border: none; }

    [data-testid="stExpander"] {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    [data-baseweb="tab-list"] {
        background: #e2e8f0; border-radius: 12px; padding: 4px; gap: 4px;
    }
    [data-baseweb="tab"] { border-radius: 10px; font-weight: 600; color: #64748b; }

    div[data-testid="stDialog"] > div { border-radius: 16px; border: 1px solid #e2e8f0; }

    #MainMenu, footer, header { visibility: hidden; }

    .js-plotly-plot {
        border-radius: 12px; background: #fff;
        border: 1px solid #e2e8f0; padding: 4px;
    }

    .symbol-link {
        color: #0f766e; text-decoration: none; font-weight: 600;
        font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;
    }
    .symbol-link:hover { color: #0d9488; text-decoration: underline; }

    .sa-topbar {
        display: flex; align-items: flex-end; justify-content: space-between;
        gap: 1rem; margin-bottom: 0.75rem; flex-wrap: wrap;
    }
    .sa-topbar h1 {
        margin: 0; font-size: 1.45rem; font-weight: 700; color: #0f172a;
        border: none; padding: 0;
    }
    .sa-topbar p { margin: 0.2rem 0 0; color: #64748b; font-size: 0.9rem; }
    .sa-chip {
        font-size: 0.75rem; font-weight: 600; color: #0f766e;
        background: #ccfbf1; border: 1px solid #99f6e4;
        padding: 0.35rem 0.7rem; border-radius: 999px; white-space: nowrap;
    }
    .sa-hero {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 1rem 1.25rem; margin: 0.5rem 0 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    .sa-hero-row {
        display: flex; align-items: center; justify-content: space-between;
        gap: 1rem; flex-wrap: wrap;
    }
    .sa-sym {
        font-size: 1.75rem; font-weight: 700; color: #0f172a;
        font-family: 'JetBrains Mono', monospace; letter-spacing: -0.03em;
    }
    .sa-ident { display: flex; align-items: center; gap: 0.85rem; }
    .sa-price { font-size: 1.25rem; font-weight: 700; color: #334155; font-family: 'JetBrains Mono', monospace; }
    .sa-badge {
        display: inline-block; padding: 0.35rem 0.8rem; border-radius: 999px;
        font-weight: 700; font-size: 0.8rem;
    }
    .sa-badge-bull { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .sa-badge-bear { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .sa-badge-hold { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
    .sa-stats { display: flex; flex-wrap: wrap; gap: 0.5rem 1.35rem; justify-content: flex-end; }
    .sa-stat { min-width: 3.5rem; }
    .sa-stat-label { font-size: 0.66rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }
    .sa-stat-val { font-size: 0.95rem; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

# --- Hugging Face API ---
def get_hf_token():
    """HF token from Streamlit secrets, falling back to the environment."""
    try:
        return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN", "")
    except Exception:
        return os.getenv("HF_TOKEN", "")


HF_TOKEN = get_hf_token()
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MA_COLS = ['ma_10', 'ma_30', 'ma_50', 'ma_100', 'ma_200']


def _chat(messages, max_tokens):
    """Run a Llama chat completion and strip trailing prompt artifacts."""
    try:
        response = InferenceClient(token=HF_TOKEN).chat_completion(
            model=LLM_MODEL, messages=messages, max_tokens=max_tokens, temperature=0.2
        )
        return re.split(r'\[/?USER\]|Can you|Could you', response.choices[0].message.content.strip())[0].strip()
    except Exception as e:
        return f"Error generating summary: {e}"


# --- AI stock summary ---
def format_trend_deltas(ticker_df, windows=(14, 50, 200)):
    """Compact multi-window trend text for LLM context."""
    recent = ticker_df.sort_values("Date")
    latest = recent.iloc[-1]
    text = "Trend Deltas:\n"
    for w in windows:
        if len(recent) < w:
            continue
        past = recent.iloc[-w]
        tail = recent.tail(w)
        text += (
            f"Last {w} days: "
            f"Price {(latest['Close'] / past['Close'] - 1) * 100:.2f}%, "
            f"RSI change {latest['RSI'] - past['RSI']:.2f}, "
            f"MACD change {latest['macd'] - past['macd']:.2f}, "
            f"Price vs MA30 {(latest['Close'] / latest['ma_30'] - 1) * 100:.2f}%, "
            f"Price vs MA200 {(latest['Close'] / latest['ma_200'] - 1) * 100:.2f}%, "
            f"Above MA200 {(tail['Close'] > tail['ma_200']).mean() * 100:.2f}% of days\n"
        )
    return text


def generate_ai_summary(ticker, ticker_df):
    """AI summary + recommendation for one ticker."""
    latest = ticker_df.nlargest(1, 'Date').iloc[0]
    price = latest['Close']
    ma_lines = "\n".join(
        f"Price - {ma.upper().replace('_', '')}: ${price - latest[ma]:.2f} ({(price / latest[ma] - 1) * 100:.2f}%)"
        for ma in MA_COLS
    )
    context = (
        f"Stock: {ticker}\n"
        f"Date: {latest['Date']:%Y-%m-%d}\n"
        f"Current Price: ${price:.2f}\n"
        f"Model Signal: {latest['final_trade']}\n"
        f"Technical Score: {latest['Technical_Score']:.2f}\n"
        f"Combined Score: {latest['combined_signal']:.2f}\n"
        f"RSI: {latest['RSI']:.2f}\n"
        f"MACD: {latest['macd']:.2f}\n\n"
        f"Price vs Moving Averages (Difference):\n{ma_lines}\n\n"
        f"Balance Sheet Score: {latest['Fundamental_Weight']:.2f}\n"
        f"Sentiment Score: {latest['SentimentScore']:.2f}\n\n"
        f"{format_trend_deltas(ticker_df)}"
    )
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
    return _chat(messages, max_tokens=400)


# --- Data loading ---
_REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reports")
_SIGNAL_CSV = os.path.join(_REPORTS_DIR, "signal_analysis.csv")
_EARNINGS_CSV = os.path.join(_REPORTS_DIR, "earnings_date.csv")
_RANK_CSV = os.path.join(_REPORTS_DIR, "daily_rank.csv")
_RANK_COLS = ["Date", "Symbol", "Rank", "combined_signal"]


@st.cache_data(ttl=3600)
def load_data(mtime: float):
    """Load signal CSV; `mtime` busts the cache when the file is rewritten."""
    return pd.read_csv(_SIGNAL_CSV, parse_dates=['Date'])


def load_signal_df():
    return load_data(os.path.getmtime(_SIGNAL_CSV))


@st.cache_data(ttl=3600)
def get_latest_data(df):
    """Most recent row per symbol."""
    return df.sort_values('Date', ascending=False).drop_duplicates(subset='Symbol', keep='first')


# --- Daily rank snapshot ---
def _compute_day_ranks(df, day):
    """Overall rank by combined_signal for one date. Rank 1 = highest score."""
    sub = (
        df.loc[df["Date"] == day, ["Symbol", "combined_signal"]]
        .sort_values("combined_signal", ascending=False)
        .drop_duplicates(subset=["Symbol"])
    )
    sub["Date"] = day
    sub["Rank"] = range(1, len(sub) + 1)
    return sub[_RANK_COLS]


def sync_daily_ranks(df):
    """Keep Reports/daily_rank.csv: past days stay frozen; today and the last saved day are recomputed.

    The last saved day may have been snapshotted from a partial intraday run, so it is
    refreshed once more with complete data. Keeps the last 30 trading days.
    """
    work = df[["Symbol", "Date", "combined_signal"]].dropna().copy()
    work["Date"] = work["Date"].dt.normalize()
    recent_days = sorted(work["Date"].unique())[-30:]
    today = recent_days[-1]

    snap = pd.read_csv(_RANK_CSV) if os.path.exists(_RANK_CSV) else pd.DataFrame(columns=_RANK_COLS)
    snap["Date"] = pd.to_datetime(snap["Date"]).dt.normalize()
    frozen = snap[(snap["Date"] < today) & (snap["Date"] < snap["Date"].max()) & snap["Date"].isin(recent_days)]
    frozen_days = set(frozen["Date"])

    out = pd.concat(
        [frozen]
        + [_compute_day_ranks(work, d) for d in recent_days if d < today and d not in frozen_days]
        + [_compute_day_ranks(work, today)],
        ignore_index=True,
    ).sort_values(["Date", "Rank"]).reset_index(drop=True)
    out.to_csv(_RANK_CSV, index=False)
    return out


def day_rank_change(df):
    """Symbol -> (yesterday_rank - today_rank, today_rank). Positive = moved up."""
    snap = sync_daily_ranks(df)
    days = sorted(snap["Date"].unique())
    if len(days) < 2:
        return {}
    t = snap[snap["Date"] == days[-1]].set_index("Symbol")["Rank"]
    y = snap[snap["Date"] == days[-2]].set_index("Symbol")["Rank"]
    return {s: (int(y[s]) - int(t[s]), int(t[s])) for s in t.index.intersection(y.index)}


def build_signal_rank_table(df, n_days=20):
    """Symbol x last-n-dates table of combined_signal ranks (newest first) with Trend and earnings dates."""
    work = df[['Symbol', 'Date', 'combined_signal']].dropna()
    dates = sorted(work['Date'].unique())[-n_days:]
    work = (
        work[work['Date'].isin(dates)]
        .sort_values('combined_signal', ascending=False)
        .drop_duplicates(subset=['Symbol', 'Date'])
    )
    work['rank'] = work.groupby('Date')['combined_signal'].rank(ascending=False, method='min').astype(int)
    pivot = work.pivot(index='Symbol', columns='Date', values='rank')
    pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)

    def _rank_trend(row):
        """BULL = 4+ consecutive rank improvements ending today, BEAR = 3+ declines."""
        vals = [int(v) for v in row.iloc[::-1] if pd.notna(v)]
        if len(vals) < 4:
            return "HOLD"
        improve = worsen = 0
        for i in range(len(vals) - 1, 0, -1):
            if vals[i] < vals[i - 1] and not worsen:
                improve += 1
            elif vals[i] > vals[i - 1] and not improve:
                worsen += 1
            else:
                break
        return "BULL" if improve >= 4 else ("BEAR" if worsen >= 3 else "HOLD")

    trend = pivot.apply(_rank_trend, axis=1)
    pivot.columns = [pd.Timestamp(c).strftime("%m/%d") for c in pivot.columns]
    rank_cols = list(pivot.columns)
    pivot["Trend"] = trend
    pivot = pivot.sort_index().rename_axis("Symbol").reset_index()
    pivot = pivot.merge(get_last_next_earnings(pivot["Symbol"].tolist()), on="Symbol", how="left")
    return pivot[["Symbol", "Last ED", "Next ED", "Trend"] + rank_cols]


# --- Earnings / fundamentals / news ---
@st.cache_data(ttl=3600)
def _load_earnings_dates(mtime: float):
    ed = pd.read_csv(_EARNINGS_CSV)
    ed['Symbol'] = ed['Symbol'].astype(str).str.strip().str.upper()
    ed['Earnings Date'] = pd.to_datetime(ed['Earnings Date'], errors='coerce')
    return ed.dropna(subset=['Earnings Date'])


def load_earnings_dates():
    return _load_earnings_dates(os.path.getmtime(_EARNINGS_CSV))


def get_last_next_earnings(symbols):
    """Per symbol: most recent past and nearest upcoming earnings date."""
    ed = load_earnings_dates()
    today = pd.Timestamp.now().normalize()
    rows = []
    for sym in symbols:
        dates = ed.loc[ed['Symbol'] == sym, 'Earnings Date']
        last, nxt = dates[dates <= today].max(), dates[dates >= today].min()
        rows.append({
            'Symbol': sym,
            'Last ED': last.strftime('%Y-%m-%d') if pd.notna(last) else '',
            'Next ED': nxt.strftime('%Y-%m-%d') if pd.notna(nxt) else '',
        })
    return pd.DataFrame(rows, columns=['Symbol', 'Last ED', 'Next ED'])


def get_upcoming_earnings(symbols, days=7):
    """Earnings within the next `days` days, soonest first (one row per symbol)."""
    ed = load_earnings_dates()
    today = pd.Timestamp.now().normalize()
    soon = ed[ed['Symbol'].isin(symbols) & ed['Earnings Date'].between(today, today + pd.Timedelta(days=days))]
    return soon.sort_values(['Earnings Date', 'Symbol']).drop_duplicates('Symbol').fillna({'Time': '—'})


@st.cache_data(ttl=3600)
def load_company_analysis():
    return pd.read_excel(
        os.path.join(_REPORTS_DIR, "complete_company_analysis.xlsx"),
        sheet_name="2_Latest_Quarter_Complete", engine="openpyxl",
    )


def get_company_metrics(ticker, company_df):
    """Fair value and key ratios for ticker (empty if the ticker has no fundamentals)."""
    row = company_df[company_df['Symbol'].astype(str).str.strip().str.upper() == ticker]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        key: float(r[col])
        for col, key in [
            ('FairValue_Composite', 'fair_value'),
            ('PE_Ratio', 'pe_ratio'),
            ('PB_Ratio', 'pb_ratio'),
            ('RevenueGrowth_YoY', 'revenue_growth_yoy'),
            ('TTM_ROE', 'roe'),
            ('TTM_NetProfitMargin', 'net_margin'),
            ('Debt_to_Equity', 'debt_to_equity'),
        ]
        if pd.notna(r[col])
    }


@st.cache_data(ttl=3600)
def load_news_data():
    return pd.read_csv(os.path.join(_REPORTS_DIR, "news_cleaned_df.csv"))


def format_news_for_llm(news, max_articles=20):
    news = news.sort_values('date', ascending=False).head(max_articles)
    articles = "".join(
        f"Article {i}:\nDate: {r['date']}\nSource: {r['source']}\nHeadline: {r['headline']}\nSummary: {r['summary']}\n\n"
        for i, (_, r) in enumerate(news.iterrows(), 1)
    )
    return f"Total articles: {len(news)}\n\n{articles}"


def generate_news_summary(news_text, sentiment_type, symbol):
    """AI bullet summary of positive or negative news for a symbol."""
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
    return _chat(messages, max_tokens=500)


@st.dialog("AI Analysis", width="large")
def ai_analysis_dialog(ticker, ticker_df):
    """Modal: technical AI summary plus positive/negative news summaries."""
    with st.spinner(f"Generating AI summary for {ticker}..."):
        summary = generate_ai_summary(ticker, ticker_df)
    st.markdown(f"### {ticker}")
    st.markdown(summary)
    st.divider()
    news = load_news_data()
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
                st.markdown(generate_news_summary(format_news_for_llm(subset), label, ticker))


def _periods(mask):
    """(start, end) pairs for each consecutive True run in a date-indexed boolean Series.

    `end` is the next trading day, so runs touch without gaps and one-day runs stay visible.
    """
    runs = (mask != mask.shift()).cumsum()
    next_day = pd.Series(mask.index, index=mask.index).shift(-1).fillna(mask.index[-1] + pd.Timedelta(days=1))
    return [(g.index[0], next_day[g.index[-1]]) for _, g in mask[mask].groupby(runs[mask])]


def direction_flips(frame):
    """Rows where final_trade flips between BUY and SELL; HOLD/EARNING days in between are ignored."""
    direction = frame[frame['final_trade'].isin(['BUY', 'SELL'])].sort_values(['Symbol', 'Date'])
    return direction[direction['final_trade'].ne(direction.groupby('Symbol')['final_trade'].shift())]


# --- Main UI ---
with st.spinner("Loading data..."):
    df = load_signal_df()

latest_data = get_latest_data(df)
available_symbols = sorted(latest_data['Symbol'].unique())
if st.session_state.get('ticker_select') not in available_symbols:
    st.session_state.ticker_select = available_symbols[0]

jumped_from_link = False
q_symbol = str(st.query_params.get("symbol", "")).strip().upper()
if q_symbol in available_symbols:
    st.session_state.ticker_select = q_symbol
    jumped_from_link = True
    del st.query_params["symbol"]

by_score = latest_data.sort_values('combined_signal', ascending=False)
buy_symbols, hold_symbols, sell_symbols, earning_symbols = (
    by_score.loc[by_score['final_trade'] == sig, 'Symbol'].tolist() for sig in ('BUY', 'HOLD', 'SELL', 'EARNING')
)
rank_info = day_rank_change(df)


def make_clickable_list(symbols):
    parts = []
    for s in symbols:
        link = f'<a href="?symbol={quote(s)}" class="symbol-link" target="_self">{s}</a>'
        if s not in rank_info:
            parts.append(link)
            continue
        d, today_rank = rank_info[s]
        color = "#1a7f37" if d > 0 else ("#cf222e" if d < 0 else "#6b7280")
        parts.append(
            f'{link}<span style="color:{color};font-size:0.85em;margin-left:2px;">#{today_rank} ({d:+d})</span>'
        )
    return ", ".join(parts)


last_updated_str = (
    datetime.fromtimestamp(os.path.getmtime(_SIGNAL_CSV), tz=ZoneInfo("America/Chicago"))
    .strftime("%m/%d/%Y %I:%M %p CT")
)
st.markdown(
    f"""
    <div class="sa-topbar">
        <div>
            <h1>Stock Analysis</h1>
            <p>Technical · fundamental · news signals</p>
        </div>
        <div class="sa-chip">Updated {last_updated_str}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

symbol_data = latest_data.set_index('Symbol')
STREAK_COLS = ['Buy Streak', 'Sell Streak', 'Hold Streak']
streaks = symbol_data[STREAK_COLS].fillna(0).astype(int)


def _ticker_label(s):
    streak = next((f"{col} of {n} days" for col, n in streaks.loc[s].items() if n), "No streak")
    return f"{s}  ·  {streak}  ·  score {symbol_data.loc[s, 'combined_signal']:.0f}"


dropdown_options = by_score['Symbol'].tolist()
pick_col, ai_col = st.columns([3, 1])
with pick_col:
    ticker = st.selectbox(
        "Ticker",
        options=dropdown_options,
        format_func=_ticker_label,
        index=dropdown_options.index(st.session_state.ticker_select),
        key="ticker_dropdown",
        label_visibility="collapsed",
    )
st.session_state.ticker_select = ticker
ticker_data = df[df['Symbol'] == ticker]

with ai_col:
    if st.button("Generate AI Analysis", type="primary", width="stretch", key="generate_ai_btn"):
        ai_analysis_dialog(ticker, ticker_data)

with st.expander(f"Universe · {len(available_symbols)} tickers (click a symbol)", expanded=False):
    for label, symbols in (("BULLISH", buy_symbols), ("HOLD", hold_symbols), ("BEARISH", sell_symbols),
                           ("EARNINGS ≤2 DAYS", earning_symbols)):
        if symbols:
            st.markdown(f"**{label}** · {make_clickable_list(symbols)}", unsafe_allow_html=True)

st.markdown('<div id="ticker-focus"></div>', unsafe_allow_html=True)
if jumped_from_link:
    components.html(
        """
        <script>
        const el = window.parent.document.getElementById('ticker-focus');
        if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
        </script>
        """,
        height=0,
    )

# --- Selected ticker header ---
latest = ticker_data.nlargest(1, 'Date').iloc[0]


def _num(v):
    return float(v) if pd.notna(v) else None


def _fmt(v, spec, prefix="", suffix=""):
    return f"{prefix}{v:{spec}}{suffix}" if v is not None else "—"


def _sign_color(v):
    return "#0f172a" if v is None else ("#15803d" if v >= 0 else "#b91c1c")


def _stat(label, value, color="#0f172a"):
    return (
        f'<div class="sa-stat"><div class="sa-stat-label">{label}</div>'
        f'<div class="sa-stat-val" style="color:{color};">{value}</div></div>'
    )


signal = {'BUY': 'BULLISH', 'SELL': 'BEARISH', 'EARNING': 'EARNINGS'}.get(latest['final_trade'], 'HOLD')
badge_cls = {'BULLISH': 'sa-badge-bull', 'BEARISH': 'sa-badge-bear'}.get(signal, 'sa-badge-hold')
close = _num(latest['Close'])
sentiment = _num(latest['SentimentScore'])
comp = get_company_metrics(ticker, load_company_analysis())
fv = comp.get('fair_value')
upside = (fv / close - 1) * 100 if fv and close else None

stats = "".join([
    _stat("Score", _fmt(_num(latest['combined_signal']), ".1f")),
    _stat("Balance sheet", _fmt(_num(latest['Fundamental_Weight']), ".2f")),
    _stat("Sentiment", _fmt(sentiment, ".2f"), _sign_color(sentiment)),
    _stat("Fair value", _fmt(fv, ",.2f", "$")),
    _stat("Upside", _fmt(upside, "+.1f", suffix="%"), _sign_color(upside)),
    _stat("P/E", _fmt(comp.get('pe_ratio'), ".1f")),
    _stat("P/B", _fmt(comp.get('pb_ratio'), ".2f")),
    _stat("Rev YoY", _fmt(comp.get('revenue_growth_yoy'), ".1f", suffix="%")),
    _stat("ROE", _fmt(comp.get('roe'), ".1f", suffix="%")),
    _stat("Net margin", _fmt(comp.get('net_margin'), ".1f", suffix="%")),
    _stat("Debt/Eq", _fmt(comp.get('debt_to_equity'), ".2f")),
])
st.markdown(
    f"""
    <div class="sa-hero">
      <div class="sa-hero-row">
        <div class="sa-ident">
          <div class="sa-sym">{ticker}</div>
          <div class="sa-price">{_fmt(close, ",.2f", "$")}</div>
          <span class="sa-badge {badge_cls}">{signal}</span>
        </div>
        <div class="sa-stats">{stats}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_charts, tab_rank = st.tabs(["Charts", "Rank"])

with tab_charts:
    for col, ma in zip(st.columns(len(MA_COLS)), MA_COLS):
        col.metric(ma.upper().replace('_', ' '), _fmt(_num(latest[ma]), ",.2f", "$"))

    one_year_ago = ticker_data['Date'].max() - pd.Timedelta(days=365)
    chart = ticker_data[ticker_data['Date'] >= one_year_ago].sort_values('Date').set_index('Date')

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.175, 0.175],
        subplot_titles=['<b>Price and Moving Averages</b>', '', ''],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]],
    )
    fig.update_annotations(font=dict(size=13, color='#374151', family='Arial, sans-serif'), yshift=5)

    # Row 1: price, earnings markers, moving averages, streak bars
    hover_close = [
        f"Combined Signal: {sig:.2f}" + "".join(f"<br>{col}: {int(n)} days" for col, n in zip(STREAK_COLS, counts) if n > 0)
        for sig, *counts in zip(chart['combined_signal'], *(chart[c] for c in STREAK_COLS))
    ]
    fig.add_trace(go.Scatter(
        x=chart.index, y=chart['Close'], name='Close Price',
        line=dict(color='#27ae60', width=2), mode='lines', customdata=hover_close,
        hovertemplate='<b>Close</b><br>$%{y:.2f}<br>%{customdata}<extra></extra>'
    ), row=1, col=1)

    earnings = chart[chart['is_earnings_date'] == 1]
    fig.add_trace(go.Scatter(
        x=earnings.index, y=earnings['Close'], name='Earnings Date', mode='markers',
        marker=dict(symbol='circle', size=10, color='#ff6b35'),
        hovertemplate='<b>Earnings Date</b><br>$%{y:.2f}<br>%{x|%b %d, %Y}<extra></extra>'
    ), row=1, col=1)

    for ma, color in zip(MA_COLS, ['#ffd700', '#e74c3c', '#3498db', '#8b4513', '#808080']):
        name = ma.upper().replace('_', ' ')
        fig.add_trace(go.Scatter(
            x=chart.index, y=chart[ma], name=name, line=dict(color=color, width=1), mode='lines',
            hovertemplate=f'<b>{name}</b><br>$%{{y:.2f}}<extra></extra>'
        ), row=1, col=1)

    top_y = chart['Close'].max() + (chart['Close'].max() - chart['Close'].min()) * 0.05
    buy_active = chart['Buy Streak'] > 0
    sell_active = chart['Sell Streak'] > 0
    for mask, color, width in (
        (buy_active, "#2ca02c", 3.5),
        (sell_active, "#d62728", 3.5),
        (~buy_active & ~sell_active, "#FFD700", 4.5),
    ):
        for start, end in _periods(mask):
            fig.add_shape(type="line", x0=start, x1=end, y0=top_y, y1=top_y,
                          line=dict(color=color, width=width), row=1, col=1)
    flips = direction_flips(ticker_data)
    for day, trade in flips.loc[flips['Date'] >= chart.index[0], ['Date', 'final_trade']].itertuples(index=False):
        fig.add_vline(x=day, line=dict(color={'BUY': "#2ca02c", 'SELL': "#d62728"}[trade], width=1, dash="dot"),
                      opacity=0.6, row="all", col=1)

    # Row 2: RSI with dotted combined score on the right axis
    fig.add_trace(go.Scatter(
        x=chart.index, y=chart['RSI'], name='RSI',
        line=dict(color='#ff7f0e', width=2), mode='lines',
        hovertemplate='<b>RSI</b><br>%{y:.2f}<extra></extra>',
        fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.1)'
    ), row=2, col=1, secondary_y=False)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(200, 0, 0, 0.3)", row=2, col=1, secondary_y=False)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0, 200, 0, 0.3)", row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=chart.index, y=chart['combined_signal'], name='My Combined Score',
        line=dict(color='#7c3aed', width=1.8, dash='dot'), mode='lines',
        hovertemplate='<b>My Combined Score</b><br>%{y:.2f}<extra></extra>',
    ), row=2, col=1, secondary_y=True)

    # Row 3: MACD
    fig.add_trace(go.Scatter(
        x=chart.index, y=chart['macd'], name='MACD', line=dict(color='#d62728', width=1.8),
        mode='lines', showlegend=False, hovertemplate='<b>MACD</b><br>%{y:.4f}<extra></extra>'
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=chart.index, y=chart['MACD Signal'], name='MACD Signal',
        line=dict(color='#1f77b4', width=1.8, dash='dash'), mode='lines', showlegend=False,
        hovertemplate='<b>MACD Signal</b><br>%{y:.4f}<extra></extra>'
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(128, 128, 128, 0.4)", line_width=1, row=3, col=1)

    grid = dict(
        showspikes=True, spikecolor="#6b7280", spikesnap="cursor", spikethickness=1, spikedash="solid",
        showgrid=True, gridcolor='rgba(200, 198, 195, 0.35)', gridwidth=1,
        showline=True, linecolor='rgba(200, 198, 195, 0.4)', linewidth=1,
        tickfont=dict(size=10, color='#6b7280'),
    )
    axis_title = dict(size=11, color='#374151')
    fig.update_layout(
        height=850,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=120, b=50),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        legend=dict(
            orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5,
            font=dict(size=10, color='#374151'), bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#e2e8f0', borderwidth=1, itemwidth=30
        ),
        font=dict(family="Arial, sans-serif", size=11, color='#374151'),
        dragmode=False,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", font_size=11, font_family="Arial, sans-serif"),
    )
    fig.update_xaxes(tickformat='%b %Y', spikemode="across", zeroline=False, **(grid | dict(showgrid=False)))
    fig.update_xaxes(showticklabels=True, tickformat='%b %d', dtick=7 * 86400000, tick0="2024-01-01",
                     tickangle=-90, tickfont=dict(size=9, color='#6b7280'), row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", title_font=axis_title, spikemode="toaxis", zeroline=False,
                     tickformat='$,.0f', row=1, col=1, **grid)
    fig.update_yaxes(title_text="RSI", title_font=axis_title, spikemode="toaxis", zeroline=False,
                     range=[0, 100], row=2, col=1, secondary_y=False, **grid)
    fig.update_yaxes(
        title_text="Combined Score", title_font=dict(size=11, color='#7c3aed'), showgrid=False,
        zeroline=True, zerolinecolor='rgba(124, 58, 237, 0.25)', zerolinewidth=1,
        showline=True, linecolor='rgba(124, 58, 237, 0.35)', linewidth=1,
        tickfont=dict(size=10, color='#7c3aed'), row=2, col=1, secondary_y=True
    )
    fig.update_yaxes(title_text="MACD", title_font=axis_title, spikemode="toaxis",
                     zeroline=True, zerolinecolor='rgba(200, 198, 195, 0.4)', zerolinewidth=1,
                     row=3, col=1, **grid)
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d'],
            'scrollZoom': False,
            'doubleClick': 'reset'
        }
    )

with tab_rank:
    st.markdown("**Combined-signal rank · last 20 sessions**")
    st.caption("Rank 1 = highest score. Green/red cells mark qualifying BULL/BEAR streak days. ED colors: last ≤10d red, next ≤7d green.")

    rank_table = build_signal_rank_table(df, n_days=20)
    date_cols = [c for c in rank_table.columns if c not in ('Symbol', 'Last ED', 'Next ED', 'Trend')]
    today = pd.Timestamp.now().normalize()
    GREEN, RED, AMBER, TEXT = "#1a7f37", "#c41e3a", "#9e6a03", "#374151"

    def _streak_flags(row):
        """Mark rank cells inside a 4+ step improving run (BULL) or 3+ step worsening run (BEAR)."""
        cols = [c for c in reversed(date_cols) if pd.notna(row[c])]
        ranks = [int(row[c]) for c in cols]
        flags = {}
        n, i = len(ranks), 0
        while i < n - 1:
            j = i
            while j + 1 < n and ranks[j + 1] < ranks[j]:
                j += 1
            if j - i >= 4:
                flags.update({cols[k]: "BULL" for k in range(i, j + 1)})
                i = j + 1
                continue
            j = i
            while j + 1 < n and ranks[j + 1] > ranks[j]:
                j += 1
            if j - i >= 3:
                for k in range(i, j + 1):
                    flags.setdefault(cols[k], "BEAR")
                i = j + 1
                continue
            i += 1
        return flags

    def _td(text, color=TEXT, weight="400", extra=""):
        return f'<td style="{extra}color:{color};font-weight:{weight};text-align:center;padding:6px 8px;white-space:nowrap;">{text}</td>'

    def _row_html(row):
        sel_bg = "background-color:#e8f0fe;" if row['Symbol'] == ticker else ""
        cells = [
            f'<td style="font-weight:600;text-align:left;padding:6px 8px;position:sticky;left:0;z-index:1;'
            f'background:{"#e8f0fe" if sel_bg else "#ffffff"};">{row["Symbol"]}</td>'
        ]
        for col, lo, hi in (("Last ED", -10, 0), ("Next ED", 0, 7)):
            d = pd.to_datetime(row[col], errors="coerce")
            hit = pd.notna(d) and today + pd.Timedelta(days=lo) <= d <= today + pd.Timedelta(days=hi)
            cells.append(_td(row[col] or "", (RED if col == "Last ED" else GREEN) if hit else TEXT, "700" if hit else "400", sel_bg))
        trend_color = {"BULL": GREEN, "BEAR": RED}.get(row['Trend'], AMBER)
        cells.append(_td(row['Trend'], trend_color, "700" if row['Trend'] != "HOLD" else "600", sel_bg))
        flags = _streak_flags(row)
        for c in date_cols:
            flag = flags.get(c)
            color = {"BULL": GREEN, "BEAR": RED}.get(flag, TEXT)
            cells.append(_td("" if pd.isna(row[c]) else int(row[c]), color, "700" if flag else "400", sel_bg))
        return f"<tr>{''.join(cells)}</tr>"

    thead = "".join(
        f'<th style="position:sticky;top:0;background:#f1f5f9;padding:8px;text-align:center;'
        f'font-size:0.8rem;color:#374151;border-bottom:1px solid #e2e8f0;white-space:nowrap;">{c}</th>'
        for c in rank_table.columns
    )
    body = "".join(_row_html(row) for _, row in rank_table.iterrows())
    st.markdown(
        f"""
        <div style="max-height:520px;overflow:auto;border:1px solid #e2e8f0;border-radius:12px;background:#ffffff;">
          <table style="border-collapse:collapse;width:100%;font-size:0.85rem;font-family:Arial,sans-serif;">
            <thead><tr>{thead}</tr></thead>
            <tbody>{body}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Upcoming earnings (next 7 days) ---
st.markdown("**Earnings · next 7 days**")
upcoming = get_upcoming_earnings(available_symbols)
if upcoming.empty:
    st.caption("No earnings in the next 7 days.")
else:
    cell = 'padding:6px 12px;text-align:center;white-space:nowrap;border-right:1px solid #e2e8f0;'
    symbol_row = "".join(
        f'<td style="{cell}font-weight:700;"><a href="?symbol={quote(s)}" class="symbol-link" target="_self">{s}</a></td>'
        for s in upcoming['Symbol']
    )
    date_row = "".join(
        f'<td style="{cell}color:#374151;">{d:%a %b %d} · {t}</td>'
        for d, t in zip(upcoming['Earnings Date'], upcoming['Time'])
    )
    st.markdown(
        f"""
        <div style="overflow-x:auto;border:1px solid #e2e8f0;border-radius:12px;background:#ffffff;">
          <table style="border-collapse:collapse;font-size:0.85rem;font-family:Arial,sans-serif;">
            <tr style="background:#f1f5f9;">{symbol_row}</tr>
            <tr>{date_row}</tr>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("AM = before market open, PM = after market close. Not-yet-announced times are predicted from the stock's past reports.")

# --- Day-1 BUY / SELL streaks ---
st.markdown("**New signals · day 1**")
flips_today = direction_flips(df).merge(latest_data[['Symbol', 'Date']], on=['Symbol', 'Date'])
day1 = by_score[by_score['Symbol'].isin(flips_today['Symbol'])]
day1_rows = "".join(
    f'<tr><td style="padding:8px 12px;font-weight:700;color:{color};white-space:nowrap;border-right:1px solid #e2e8f0;">{trade} day 1</td>'
    f'<td style="padding:8px 12px;">{make_clickable_list(day1.loc[day1["final_trade"] == trade, "Symbol"]) or "—"}</td></tr>'
    for trade, color in (("BUY", "#1a7f37"), ("SELL", "#cf222e"))
)
st.markdown(
    f"""
    <div style="border:1px solid #e2e8f0;border-radius:12px;background:#ffffff;">
      <table style="border-collapse:collapse;width:100%;font-size:0.85rem;font-family:Arial,sans-serif;">{day1_rows}</table>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Stocks whose green or red line starts on the latest day (first BUY after a SELL, or first SELL after a BUY; HOLD days in between are ignored), ordered by combined score.")
