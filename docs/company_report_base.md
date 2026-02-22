# Company Report Base - Data Fetching Documentation

## Overview

Fetches fundamental financial data (income statements and balance sheets) from **Alpha Vantage API**. Backup when yFinance is down. Processes **one symbol at a time** via `SYMBOL_TO_PROCESS`. Saves only **last 5 years** of data with **IQR-based outlier capping** per symbol.

## Key Configuration

```python
YEARS_TO_KEEP = 5      # Only last 5 years saved
IQR_MULT = 1.5         # IQR capping: Q1 - 1.5*IQR to Q3 + 1.5*IQR
SYMBOL_TO_PROCESS = 'AAPL'  # Manual symbol input - change per run
```

## Data Flow

1. **Income**: Fetch from API → save via `save_symbol_data` (or load existing if empty)
2. **Balance**: Fetch from API → merge with income
3. **Final**: Filter to 5 years → save to `balance_sheet.csv`

## save_symbol_data()

- **5-year filter**: `FiscalDateEnding >= today - 5 years`
- **Numeric conversion**: `pd.to_numeric(errors='coerce')` for known columns
- **IQR capping**: Per symbol, per column. Caps at Q1 - 1.5*IQR and Q3 + 1.5*IQR
- **Columns capped**: TotalRevenue, GrossProfit, NetIncome, OperatingIncome, OperatingMargin, TotalAssets, TotalLiabilities, TotalShareholderEquity, CommonStockSharesOutstanding, BVPS, Debt_to_Equity
- **Empty guard**: If `new_df` empty or no Symbol column → returns existing CSV or empty DataFrame
- **Income save guard**: Only calls save when `income_df` has data; else loads from existing CSV

## Output Files

| File | Contents |
|------|----------|
| `Reports/income_statement.csv` | Symbol, FiscalDateEnding, TotalRevenue, GrossProfit, NetIncome, OperatingIncome, DateAdded |
| `Reports/balance_sheet.csv` | Merged income + balance. Symbol, FiscalDateEnding, TotalRevenue, GrossProfit, NetIncome, OperatingIncome, OperatingMargin, TotalAssets, TotalLiabilities, CommonStockSharesOutstanding, TotalShareholderEquity, BVPS, Debt_to_Equity, DateAdded |

## API

- **Alpha Vantage**: INCOME_STATEMENT, BALANCE_SHEET
- **Rate limit**: 12 sec between calls
- **Manual symbol**: Set `SYMBOL_TO_PROCESS` in notebook

## Dependencies

pandas, numpy, requests, os, datetime
