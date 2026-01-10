#!/usr/bin/env python3
"""
Trading Bot Execution Script
This script can be run from GitHub Actions to execute trading functions
"""

import sys
import os
from datetime import datetime, timedelta
import pytz

# Add current directory to path (for GitHub Actions)
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
sys.path.insert(0, script_dir)
sys.path.insert(0, cwd)  # Also add current working directory

def setup_environment():
    """Setup environment and import required modules"""
    try:
        # Import required modules
        import pandas as pd
        import numpy as np
        from alpaca_setup import (
            trade_client, 
            data_client,
            GetOrdersRequest,
            QueryOrderStatus,
            LimitOrderRequest,
            OrderSide,
            OrderType,
            TimeInForce,
            StockLatestTradeRequest
        )
        import signal_analysis_functions
        
        print("✅ Environment setup complete")
        return {
            'pd': pd,
            'np': np,
            'trade_client': trade_client,
            'data_client': data_client,
            'GetOrdersRequest': GetOrdersRequest,
            'QueryOrderStatus': QueryOrderStatus,
            'LimitOrderRequest': LimitOrderRequest,
            'OrderSide': OrderSide,
            'OrderType': OrderType,
            'TimeInForce': TimeInForce,
            'StockLatestTradeRequest': StockLatestTradeRequest,
            'datetime': datetime,
            'timedelta': timedelta,
            'pytz': pytz
        }
    except Exception as e:
        print(f"❌ Error setting up environment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def load_signal_data():
    """Load signal analysis data"""
    env = setup_environment()
    pd = env['pd']
    
    REPORTS_DIR = os.path.join(os.getcwd(), "Reports")
    signal_file = os.path.join(REPORTS_DIR, "signal_analysis.csv")
    
    if not os.path.exists(signal_file):
        print(f"❌ Signal file not found: {signal_file}")
        return None, None
    
    signal_base_df = pd.read_csv(signal_file)
    signal_base_df["Date"] = pd.to_datetime(signal_base_df["Date"])
    signal_base_df = signal_base_df.sort_values(by=["Symbol", "Date"], ascending=[True, False])
    signal_base_df = signal_base_df[signal_base_df["Date"] == pd.to_datetime(datetime.now().date())]
    
    sell_signal_df = signal_base_df[(signal_base_df["combined_signal"] < 20)]
    
    buy_signal_df = signal_base_df[(signal_base_df["Buy Streak"] > 3) & (signal_base_df["Buy Streak"] < 100)]
    buy_signal_df = buy_signal_df.drop_duplicates(subset=["Symbol"]).sort_values(by=["combined_signal"], ascending=False).reset_index(drop=True)
    buy_signal_df = buy_signal_df[buy_signal_df["Fundamental_Weight"] > 0]
    
    if len(buy_signal_df) > 0:
        buy_signal_df["Weights"] = buy_signal_df["Fundamental_Weight"] / buy_signal_df["Fundamental_Weight"].sum()
        buy_signal_df = buy_signal_df.sort_values(by="Weights", ascending=False).reset_index(drop=True)
    
    # Exclude certain stocks
    stocks_to_exclude = ["EW", "EXAS", "FERG", "HPQ", "LIN", "MA", "MELI", "SERV", "V", "WM"]
    buy_signal_df = buy_signal_df[~buy_signal_df["Symbol"].isin(stocks_to_exclude)]
    
    return buy_signal_df, sell_signal_df

def cancel_all_open_orders():
    """Cancel all currently open/pending orders"""
    env = setup_environment()
    trade_client = env['trade_client']
    GetOrdersRequest = env['GetOrdersRequest']
    QueryOrderStatus = env['QueryOrderStatus']
    
    print("\n" + "="*70)
    print("❌ Cancelling All Open Orders...")
    print("="*70)
    
    try:
        filter_params = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = trade_client.get_orders(filter=filter_params)
        
        if not open_orders:
            print("✅ No open or pending orders to cancel.")
            return
        
        print(f"Found {len(open_orders)} active orders:")
        for order in open_orders:
            price_info = f"@ ${order.limit_price}" if order.limit_price else ""
            print(f"  - {order.symbol} ({order.side.upper()}) | {order.qty} shares {price_info}")
        
        print("\nCancelling all orders...")
        
        cancelled_count = 0
        failed_count = 0
        
        for order in open_orders:
            try:
                trade_client.cancel_order_by_id(order.id)
                print(f"✅ Cancelled: {order.symbol} ({order.side.upper()}) - OrderID: {order.id}")
                cancelled_count += 1
            except Exception as e:
                print(f"❌ Failed to cancel {order.symbol} (OrderID: {order.id}): {e}")
                failed_count += 1
        
        print(f"\n✅ Completed: {cancelled_count} cancelled, {failed_count} failed")
        
    except Exception as e:
        print(f"❌ Error cancelling orders: {e}")
        import traceback
        traceback.print_exc()

def get_stocks_transacted_today():
    """Get symbols of stocks that were transacted today"""
    env = setup_environment()
    trade_client = env['trade_client']
    GetOrdersRequest = env['GetOrdersRequest']
    QueryOrderStatus = env['QueryOrderStatus']
    pytz = env['pytz']
    datetime = env['datetime']
    timedelta = env['timedelta']
    
    try:
        central = pytz.timezone("America/Chicago")
        now_central = datetime.now(central)
        start_of_day_central = central.localize(datetime(now_central.year, now_central.month, now_central.day))
        end_of_day_central = start_of_day_central + timedelta(days=1, seconds=-1)
        
        # Convert to UTC for API filter
        start_of_day_utc = start_of_day_central.astimezone(pytz.UTC).isoformat()
        end_of_day_utc = end_of_day_central.astimezone(pytz.UTC).isoformat()
        
        filter_params = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=start_of_day_utc,
            until=end_of_day_utc
        )
        todays_orders = trade_client.get_orders(filter=filter_params)
        symbols_transaction_today = {o.symbol for o in todays_orders}
        return symbols_transaction_today
    except Exception as e:
        print(f"⚠️ Warning: Could not get today's transactions: {e}")
        return set()

def execute_buy_orders():
    """Execute buy orders based on buy_signal_df"""
    env = setup_environment()
    trade_client = env['trade_client']
    data_client = env['data_client']
    StockLatestTradeRequest = env['StockLatestTradeRequest']
    LimitOrderRequest = env['LimitOrderRequest']
    OrderSide = env['OrderSide']
    OrderType = env['OrderType']
    TimeInForce = env['TimeInForce']
    datetime = env['datetime']
    
    print("\n" + "="*70)
    print("🟢 Executing Buy Orders...")
    print("="*70)
    
    buy_signal_df, _ = load_signal_data()
    
    # Exclude stocks transacted today
    symbols_transaction_today = get_stocks_transacted_today()
    if symbols_transaction_today:
        print(f"📋 Excluding stocks transacted today: {symbols_transaction_today}")
        buy_signal_df = buy_signal_df[~buy_signal_df["Symbol"].isin(symbols_transaction_today)]
    
    if buy_signal_df is None or len(buy_signal_df) == 0:
        print("⚠️ No buy signals found or signal file not available")
        return
    
    # Refresh account data
    acct = trade_client.get_account()
    available_buying_power = float(acct.non_marginable_buying_power)
    print(f"💵 Available buying power: ${available_buying_power:,.2f}")
    
    # Get current positions
    positions = trade_client.get_all_positions()
    position_dict = {pos.symbol: pos for pos in positions}
    
    MAX_BUYING_POWER_PER_STOCK = 3500
    
    # Compute raw allocation
    buy_signal_df["raw_allocation_usd"] = buy_signal_df["Weights"] * available_buying_power
    
    for idx, row in buy_signal_df.iterrows():
        symbol = row["Symbol"]
        raw_allocation_usd = row["raw_allocation_usd"]
        
        # Get current price
        try:
            latest_trade = data_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol)
            )
            latest_price = round(float(latest_trade[symbol].price), 2)
        except Exception as e:
            print(f"⚠️ Error getting price for {symbol}: {e}")
            continue
        
        # Determine max usable amount
        capped_allocation = min(raw_allocation_usd, MAX_BUYING_POWER_PER_STOCK)
        
        if symbol in position_dict:
            current_position = position_dict[symbol]
            current_position_value = float(current_position.market_value)
            
            if current_position_value >= MAX_BUYING_POWER_PER_STOCK:
                print(f"⚪ Skipping {symbol}: Position value ${current_position_value:.2f} already at cap")
                continue
            
            remaining_capacity = MAX_BUYING_POWER_PER_STOCK - current_position_value
            amount_to_use = min(capped_allocation, remaining_capacity, available_buying_power)
        else:
            amount_to_use = min(capped_allocation, available_buying_power)
        
        # Calculate quantity
        buy_qty = int(amount_to_use / latest_price)
        
        if buy_qty <= 0:
            print(f"⚠️ Amount too small to buy any shares of {symbol}. Skipping...")
            continue
        
        order_value = buy_qty * latest_price
        
        if order_value > available_buying_power:
            buy_qty = int(available_buying_power / latest_price)
            if buy_qty <= 0:
                print(f"⚠️ Insufficient buying power for {symbol}. Skipping...")
                continue
            order_value = buy_qty * latest_price
        
        print(f"🟢 {symbol}: Buying {buy_qty} shares @ ${latest_price:.2f} = ${order_value:.2f}")
        
        # Submit buy order
        buy_req = LimitOrderRequest(
            symbol=symbol,
            qty=buy_qty,
            limit_price=latest_price,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            extended_hours=True
        )
        
        try:
            order = trade_client.submit_order(buy_req)
            print(f"✅ Submitted {symbol}: {buy_qty} shares @ ${latest_price:.2f} (OrderID: {order.id})")
            available_buying_power -= order_value
        except Exception as e:
            print(f"❌ Failed to submit order for {symbol}: {e}")
    
    print(f"🏁 Buy orders completed. Remaining buying power: ${available_buying_power:,.2f}")

def execute_sell_orders():
    """Execute sell orders based on sell_signal_df"""
    env = setup_environment()
    trade_client = env['trade_client']
    data_client = env['data_client']
    StockLatestTradeRequest = env['StockLatestTradeRequest']
    LimitOrderRequest = env['LimitOrderRequest']
    OrderSide = env['OrderSide']
    OrderType = env['OrderType']
    TimeInForce = env['TimeInForce']
    datetime = env['datetime']
    
    print("\n" + "="*70)
    print("🔴 Executing Sell Orders...")
    print("="*70)
    
    _, sell_signal_df = load_signal_data()
    
    if sell_signal_df is None or len(sell_signal_df) == 0:
        print("⚠️ No sell signals found or signal file not available")
        return
    
    # Get current positions
    positions = trade_client.get_all_positions()
    if not positions:
        print("📭 No open positions to sell.")
        return
    
    position_dict = {pos.symbol: pos for pos in positions}
    
    # Get symbols from sell_signal_df that we have positions in
    sell_symbols = sell_signal_df[sell_signal_df["Symbol"].isin(position_dict.keys())]["Symbol"].unique()
    
    if len(sell_symbols) == 0:
        print("📭 No positions match sell signals.")
        return
    
    print(f"📉 Processing {len(sell_symbols)} sell orders from sell_signal_df...")
    
    for symbol in sell_symbols:
        position = position_dict[symbol]
        sell_quantity = int(float(position.qty))
        
        if sell_quantity <= 0:
            continue
        
        # Get current price
        try:
            latest_trade = data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
            current_price = float(latest_trade[symbol].price)
        except Exception as e:
            print(f"⚠️ Error getting price for {symbol}: {e}")
            continue
        
        print(f"🟢 {symbol}: Selling {sell_quantity} shares @ ${current_price:.2f}")
        
        # Submit sell order
        sell_req = LimitOrderRequest(
            symbol=symbol,
            qty=sell_quantity,
            limit_price=current_price,
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            extended_hours=True
        )
        
        try:
            order = trade_client.submit_order(sell_req)
            print(f"✅ Sell order submitted for {symbol}: {sell_quantity} shares @ ${current_price:.2f} (OrderID: {order.id})")
        except Exception as e:
            print(f"❌ Failed to submit sell order for {symbol}: {e}")
    
    print(f"🏁 Sell orders completed for {len(sell_symbols)} stocks")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Execute trading bot functions')
    parser.add_argument('action', choices=['buy', 'sell', 'cancel'], 
                       help='Action to execute: buy, sell, or cancel')
    
    args = parser.parse_args()
    
    if args.action == 'buy':
        execute_buy_orders()
    elif args.action == 'sell':
        execute_sell_orders()
    elif args.action == 'cancel':
        cancel_all_open_orders()
