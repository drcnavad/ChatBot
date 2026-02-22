# Company Report Processing - Financial Analysis Documentation

## Overview

Processes balance sheet and income statement data to calculate financial ratios, TTM metrics, and fair value estimations. Input is typically **5 years** of data (from base notebook); processing filters `>= 2014-01-01`.

## Input

- **`Reports/balance_sheet.csv`**: Merged quarterly income + balance sheet
- **Required columns**: Symbol, FiscalDateEnding, TotalRevenue, GrossProfit, NetIncome, OperatingIncome, OperatingMargin, TotalAssets, TotalLiabilities, CommonStockSharesOutstanding, TotalShareholderEquity, BVPS, Debt_to_Equity, Date
- **Filter**: `FiscalDateEnding >= 2014-01-01`

## Processing Steps

### 1. Financial Ratios
- Profitability: ROE, ROA, NetProfitMargin, GrossMargin, OperatingMargin
- Per Share: EPS, RevenuePerShare, AssetsPerShare, BVPS
- Efficiency: AssetTurnover, EquityTurnover
- Health: DebtToAssets, EquityRatio
- Growth: Revenue/NetIncome/EPS (QoQ, YoY)

### 2. TTM Metrics
- Income: Sum of last 4 quarters (Revenue, NetIncome, OperatingIncome, GrossProfit)
- Balance: Avg of beginning/ending (Equity, Assets)
- Ratios: TTM_ROE, TTM_ROA, TTM_NetProfitMargin, TTM_OperatingMargin, TTM_GrossMargin, TTM_EPS, TTM_AssetTurnover, TTM_EquityTurnover

### 3. Fair Value
- P/E, P/B, P/S, Graham Formula, Growth-adjusted P/E
- Profitable: P/E, Growth P/E, P/B, Graham
- Unprofitable: P/B, P/S
- Output: FairValue, Price_vs_FairValue, Valuation_Signal

### 4. Advanced
- DuPont, ROIC, leverage ratios, working capital

## Output

**`Reports/complete_company_analysis.xlsx`**
- Sheet `1_Historical_All_Quarters`: All quarters, ~49+ columns
- Sheet `2_Latest_Quarter_Complete`: Latest quarter per symbol (input for scoring)

## Dependencies

pandas, numpy, openpyxl, datetime
