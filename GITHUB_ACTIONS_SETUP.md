# GitHub Actions Setup Guide

This guide explains how to set up GitHub Actions to automatically run your trading bot.

## Prerequisites

1. Your repository must be on GitHub
2. You need admin access to the repository to add secrets

## Setup Steps

### 1. Add GitHub Secrets

Go to your repository on GitHub:
- Navigate to **Settings** → **Secrets and variables** → **Actions**
- Click **New repository secret**

Add the following secrets:

#### Required Secrets:
- **`ALPACA_API_KEY`**: Your Alpaca API key
- **`ALPACA_SECRET_KEY`**: Your Alpaca secret key

### 2. Verify Workflow File

The workflow file is located at:
```
.github/workflows/trading_bot.yml
```

### 3. Schedule Overview

The workflow runs on the following schedule (Central Time, Monday-Friday):

#### Buy Orders (with git push):
- **9:30 AM CT** (14:30 UTC)
- **1:00 PM CT** (18:00 UTC)

#### Sell Orders (hourly during market hours):
- **9:30 AM CT** (14:30 UTC)
- **10:30 AM CT** (15:30 UTC)
- **11:30 AM CT** (16:30 UTC)
- **12:30 PM CT** (17:30 UTC)
- **1:00 PM CT** (18:00 UTC)
- **1:30 PM CT** (18:30 UTC)
- **2:00 PM CT** (19:00 UTC)
- **2:30 PM CT** (19:30 UTC)
- **3:00 PM CT** (20:00 UTC)

### 4. Workflow Execution Flow

#### For Buy Order Times (9:30 AM & 1:00 PM CT):
1. Run `main_signal_analysis.ipynb`
2. Execute sell orders
3. Cancel all open orders
4. Execute buy orders
5. Commit and push changes to git

#### For Other Hours:
1. Run `main_signal_analysis.ipynb`
2. Execute sell orders

### 5. Manual Execution

You can manually trigger the workflow:
- Go to **Actions** tab in your repository
- Select **Trading Bot Automation** workflow
- Click **Run workflow**

### 6. Monitoring

- Check the **Actions** tab to see workflow runs
- View logs for each step to debug issues
- Failed runs will show error messages

## Troubleshooting

### Workflow Not Running
- Check that secrets are properly set
- Verify cron schedule syntax
- Ensure repository has Actions enabled

### Script Errors
- Check that all dependencies are in `requirements.txt`
- Verify that `.env` file variables are set as GitHub secrets
- Check logs in the Actions tab for specific error messages

### Git Push Fails
- Ensure `GITHUB_TOKEN` has write permissions
- Check that workflow has permission to push (should be automatic)

## Files Created

- `run_trading_bot.py` - Executes trading functions (buy/sell/cancel)
- `run_signal_analysis.py` - Runs main_signal_analysis.ipynb
- `.github/workflows/trading_bot.yml` - GitHub Actions workflow

## Notes

- All times are in Central Time (CT)
- Workflows only run on weekdays (Monday-Friday)
- Market hours: 9:30 AM - 3:00 PM CT
- Git commits are only made after buy orders execute
