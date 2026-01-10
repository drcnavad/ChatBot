# Trading Bot Automation Summary

## Files Created

### 1. `run_trading_bot.py`
Python script that executes trading functions:
- `execute_buy_orders()` - Executes buy orders based on buy_signal_df
- `execute_sell_orders()` - Executes sell orders based on sell_signal_df  
- `cancel_all_open_orders()` - Cancels all open/pending orders
- Includes logic to exclude stocks transacted today
- Can be run from command line: `python run_trading_bot.py [buy|sell|cancel]`

### 2. `run_signal_analysis.py`
Python script that executes `main_signal_analysis.ipynb`:
- Uses papermill (preferred) or nbconvert to run the notebook
- Handles timeouts and errors gracefully
- Can be run from command line: `python run_signal_analysis.py`

### 3. `.github/workflows/trading_bot.yml`
GitHub Actions workflow that:
- Runs on scheduled times (see schedule below)
- Executes signal analysis before sell orders
- Executes buy orders at specific times with git push
- Can be manually triggered via workflow_dispatch

### 4. `GITHUB_ACTIONS_SETUP.md`
Setup guide with instructions for configuring GitHub secrets and understanding the workflow

## Schedule (Central Time, Monday-Friday)

### Buy Orders (with git push):
- **9:30 AM CT** (14:30 UTC)
- **1:00 PM CT** (18:00 UTC)

### Sell Orders (hourly during market hours):
- **9:30 AM CT** (14:30 UTC)
- **10:30 AM CT** (15:30 UTC)
- **11:30 AM CT** (16:30 UTC)
- **12:30 PM CT** (17:30 UTC)
- **1:00 PM CT** (18:00 UTC)
- **1:30 PM CT** (18:30 UTC)
- **2:00 PM CT** (19:00 UTC)
- **2:30 PM CT** (19:30 UTC)
- **3:00 PM CT** (20:00 UTC)

## Execution Flow

### At Buy Order Times (9:30 AM & 1:00 PM CT):
1. ✅ Run `main_signal_analysis.ipynb`
2. ✅ Execute sell orders
3. ✅ Cancel all open orders
4. ✅ Execute buy orders
5. ✅ Commit and push changes to git

### At Other Hours:
1. ✅ Run `main_signal_analysis.ipynb`
2. ✅ Execute sell orders

## Setup Required

1. **Add GitHub Secrets:**
   - Go to Repository Settings → Secrets and variables → Actions
   - Add `ALPACA_API_KEY`
   - Add `ALPACA_SECRET_KEY`

2. **Verify Files:**
   - `.github/workflows/trading_bot.yml` exists
   - `run_trading_bot.py` is executable
   - `run_signal_analysis.py` is executable
   - `requirements.txt` includes all dependencies

3. **Test Locally (optional):**
   ```bash
   python run_trading_bot.py cancel
   python run_signal_analysis.py
   python run_trading_bot.py sell
   ```

## Features

- ✅ Automatic execution on schedule
- ✅ Manual triggering via GitHub Actions UI
- ✅ Signal analysis runs before sell orders
- ✅ Buy orders exclude stocks transacted today
- ✅ Git push after buy orders
- ✅ Error handling and logging
- ✅ Market hours only (weekdays)

## Notes

- All times are in Central Time (CT)
- Workflows only run Monday-Friday
- Git commits include timestamp
- Failed workflows can be retried manually
