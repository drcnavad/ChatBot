# alpaca_setup.py
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import os
import nest_asyncio
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env
load_dotenv()

# Apply async patch (for Jupyter)
nest_asyncio.apply()


# 🕒 Timezone
EASTERN = ZoneInfo("America/New_York")

# ✅ Import Alpaca classes
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.trading.stream import TradingStream
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

from alpaca.data.requests import (
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
    StockLatestQuoteRequest, 
    StockLatestTradeRequest
)

from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest
)

from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
    QueryOrderStatus
)

# 🧠 Initialize Clients (LIVE ONLY)
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise EnvironmentError("⚠️ Missing API keys. Make sure .env has ALPACA_API_KEY and ALPACA_SECRET_KEY")

# 🚀 Live Trading Client (no paper flag)
trade_client = TradingClient(API_KEY, SECRET_KEY, paper=False)

# You can still use the same data client — it automatically uses live data
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

#trade steam clinet
trade_stream_client = TradingStream(API_KEY, SECRET_KEY)


# --- 1️⃣ Get account summary ---
acct = trade_client.get_account()
buying_power = float(acct.non_marginable_buying_power)
cash = float(acct.cash)
margin_buying_power = float(acct.buying_power)
equity = float(acct.equity)
total_fees = float(acct.accrued_fees)

