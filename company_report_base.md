# Company Report Base - Data Fetching Documentation

## Overview

This notebook fetches fundamental financial data (income statements and balance sheets) from the **Alpha Vantage API** and stores it in CSV files. It serves as a backup when yFinance is down and implements rate limiting to respect API constraints.

## API Details

### Current API: Alpha Vantage

- **API Endpoint**: `https://www.alphavantage.co/query`
- **API Key**: Configured via `ALPHA_VANTAGE_API` variable
- **Rate Limits**: 
  - Free tier: **25 API calls per day**
  - Each symbol requires **2 API calls** (income statement + balance sheet)
  - Maximum **12 symbols per day** (12 symbols × 2 calls = 24 calls, leaving 1 spare)

### API Functions Used

1. **Income Statement**: `function=INCOME_STATEMENT`
   - Endpoint: `https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={SYMBOL}&apikey={API_KEY}`
   - Returns quarterly income statement data

2. **Balance Sheet**: `function=BALANCE_SHEET`
   - Endpoint: `https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={SYMBOL}&apikey={API_KEY}`
   - Returns quarterly balance sheet data

## Data Fetching Logic

### Symbol Selection

- **Source**: Symbols are imported from `sector_mapping.py` (`stock_symbols`)
- **Processing Limit**: Maximum 12 symbols per day (due to API rate limits)
- **Freshness Check**: Skips symbols with data added within the last **7 days**
- **Deduplication**: Only processes symbols if **BOTH** income and balance sheet data are older than 7 days

### Function: `get_symbols_to_process_today()`

This function determines which symbols need to be processed:

1. Checks `Reports/income_statement.csv` for recent data (within 7 days)
2. Checks `Reports/balance_sheet.csv` for recent data (within 7 days)
3. Skips symbols only if **BOTH** datasets have recent data
4. Limits to `MAX_SYMBOLS_PER_DAY` (12 symbols)
5. Returns list of symbols to process

### Rate Limiting

- **Wait Time**: 12 seconds between API calls
- **Error Handling**: Detects API limit messages ("Note" or "Thank you for using Alpha Vantage")
- **Graceful Degradation**: Stops processing if API limit is reached

## Data Fetched

### Income Statement Data

Fetches quarterly income statement reports with the following fields:

- `Symbol`: Stock ticker symbol
- `FiscalDateEnding`: End date of fiscal quarter
- `TotalRevenue`: Total revenue for the quarter
- `GrossProfit`: Gross profit for the quarter
- `NetIncome`: Net income for the quarter
- `OperatingIncome`: Operating income for the quarter
- `DateAdded`: Date when data was fetched (for freshness tracking)

**Calculated Fields**:
- `OperatingMargin`: OperatingIncome / TotalRevenue × 100

### Balance Sheet Data

Fetches quarterly balance sheet reports with the following fields:

- `Symbol`: Stock ticker symbol
- `FiscalDateEnding`: End date of fiscal quarter
- `TotalAssets`: Total assets
- `TotalLiabilities`: Total liabilities
- `CommonStockSharesOutstanding`: Number of shares outstanding
- `TotalShareholderEquity`: Total shareholder equity
- `DateAdded`: Date when data was fetched (for freshness tracking)

**Calculated Fields**:
- `BVPS`: Book Value Per Share (TotalShareholderEquity / CommonStockSharesOutstanding)
- `Debt_to_Equity`: Debt-to-Equity ratio (TotalLiabilities / TotalShareholderEquity)

## Data Storage

### Output Files

1. **`Reports/income_statement.csv`**
   - Contains all quarterly income statement data
   - Appends new data (does not overwrite)
   - Prevents duplicates based on `Symbol` + `FiscalDateEnding` combination
   - Keeps most recent `DateAdded` if duplicates exist
   
   **Exact Column Structure**:
   ```
   Symbol, FiscalDateEnding, TotalRevenue, GrossProfit, NetIncome, OperatingIncome, DateAdded
   ```
   - `Symbol`: String (uppercase ticker symbol)
   - `FiscalDateEnding`: Date (YYYY-MM-DD format)
   - `TotalRevenue`: Float (numeric, can be very large)
   - `GrossProfit`: Float (numeric, can be very large)
   - `NetIncome`: Float (numeric, can be negative)
   - `OperatingIncome`: Float (numeric, can be negative)
   - `DateAdded`: Date (YYYY-MM-DD format, when data was fetched)

2. **`Reports/balance_sheet.csv`**
   - Contains **merged** quarterly balance sheet AND income statement data
   - This file is created by merging income and balance data in the notebook
   - Appends new data (does not overwrite)
   - Prevents duplicates based on `Symbol` + `FiscalDateEnding` combination
   - Keeps most recent `DateAdded` if duplicates exist
   
   **Exact Column Structure**:
   ```
   Symbol, FiscalDateEnding, TotalRevenue, GrossProfit, NetIncome, OperatingIncome, 
   OperatingMargin, TotalAssets, TotalLiabilities, CommonStockSharesOutstanding, 
   TotalShareholderEquity, BVPS, Debt_to_Equity, Date
   ```
   - `Symbol`: String (uppercase ticker symbol)
   - `FiscalDateEnding`: Date (YYYY-MM-DD format)
   - `TotalRevenue`: Float (numeric)
   - `GrossProfit`: Float (numeric)
   - `NetIncome`: Float (numeric, can be negative)
   - `OperatingIncome`: Float (numeric, can be negative)
   - `OperatingMargin`: Float (percentage, calculated: OperatingIncome / TotalRevenue × 100)
   - `TotalAssets`: Float (numeric, can be very large)
   - `TotalLiabilities`: Float (numeric, can be very large)
   - `CommonStockSharesOutstanding`: Float (numeric, shares in millions typically)
   - `TotalShareholderEquity`: Float (numeric, can be negative)
   - `BVPS`: Float (Book Value Per Share, calculated: TotalShareholderEquity / CommonStockSharesOutstanding, rounded to 2 decimals)
   - `Debt_to_Equity`: Float (ratio, calculated: TotalLiabilities / TotalShareholderEquity, rounded to 2 decimals)
   - `Date`: Date (YYYY-MM-DD format, appears to be same as DateAdded)

### Data Management

- **Deduplication**: Removes existing rows that match new data before appending
- **Sorting**: Data is sorted by `Symbol`, `FiscalDateEnding` (descending), and `DateAdded` (descending)
- **Data Types**: 
  - Numeric columns are converted using `pd.to_numeric(errors='coerce')` - missing/invalid values become NaN
  - Dates are converted using `pd.to_datetime(errors='coerce')` - invalid dates become NaT
  - Rows with null `FiscalDateEnding` are filtered out before saving
- **Merge Logic**: Income and balance data are merged on `Symbol` and `FiscalDateEnding` using outer join
- **Numeric Precision**: 
  - `BVPS` and `Debt_to_Equity` are rounded to 2 decimal places
  - `OperatingMargin` is calculated as percentage (× 100) but not rounded in base notebook

## Configuration Parameters

```python
ALPHA_VANTAGE_API = 'API_KEY_HERE'  # Alpha Vantage API key
MAX_SYMBOLS_PER_DAY = 12            # Maximum symbols to process per day
DATA_FRESHNESS_DAYS = 7              # Skip symbols with data newer than this
```

## Workflow

1. **Import symbols** from `sector_mapping.py`
2. **Determine symbols to process** using `get_symbols_to_process_today()`
3. **Fetch income statement data**:
   - Loop through symbols
   - Make API call for each symbol
   - Wait 12 seconds between calls
   - Parse quarterly reports
   - Append to CSV
4. **Fetch balance sheet data**:
   - Use same symbols as income statement
   - Make API call for each symbol
   - Wait 12 seconds between calls
   - Parse quarterly reports
   - Append to CSV

## Error Handling

- **API Limit Detection**: Checks for "Note" or "Thank you for using Alpha Vantage" in response
- **Missing Data**: Handles cases where `quarterlyReports` is missing
- **HTTP Errors**: Reports status codes for failed requests
- **Empty Results**: Handles empty DataFrames gracefully

## Notes for Future API Migration

When migrating to a different API (SimFin, FNP, etc.), you'll need to:

### 1. API Endpoint Updates
   - Replace Alpha Vantage URLs with new API endpoints
   - Update request parameters to match new API format
   - Handle authentication (API keys, tokens, etc.)

### 2. Response Parsing Updates
   - Modify JSON parsing logic to match new API response structure
   - **Critical Field Mappings** (Alpha Vantage → Your API):
     - `quarterlyReports` → [Your API's array key for quarterly data]
     - `fiscalDateEnding` → [Your API's date field name]
     - `totalRevenue` → [Your API's revenue field]
     - `grossProfit` → [Your API's gross profit field]
     - `netIncome` → [Your API's net income field]
     - `operatingIncome` → [Your API's operating income field]
     - `totalAssets` → [Your API's total assets field]
     - `totalLiabilities` → [Your API's total liabilities field]
     - `commonStockSharesOutstanding` → [Your API's shares outstanding field]
     - `totalShareholderEquity` → [Your API's shareholder equity field]

### 3. Rate Limiting Updates
   - Adjust `MAX_SYMBOLS_PER_DAY` based on new API limits
   - Modify wait times between calls if needed
   - Update API limit detection logic (currently checks for "Note" or "Thank you for using Alpha Vantage")

### 4. Data Structure Requirements (CRITICAL)

   **Income Statement CSV (`income_statement.csv`) - Must have exactly these columns:**
   ```
   Symbol, FiscalDateEnding, TotalRevenue, GrossProfit, NetIncome, OperatingIncome, DateAdded
   ```
   - Column names must match exactly (case-sensitive)
   - All columns must be present
   - `DateAdded` must be added during fetch (use `datetime.now().date()`)

   **Balance Sheet CSV (`balance_sheet.csv`) - Must have exactly these columns:**
   ```
   Symbol, FiscalDateEnding, TotalRevenue, GrossProfit, NetIncome, OperatingIncome, 
   OperatingMargin, TotalAssets, TotalLiabilities, CommonStockSharesOutstanding, 
   TotalShareholderEquity, BVPS, Debt_to_Equity, Date
   ```
   - This is a **merged file** containing both income and balance data
   - `OperatingMargin` must be calculated: `OperatingIncome / TotalRevenue × 100`
   - `BVPS` must be calculated: `TotalShareholderEquity / CommonStockSharesOutstanding` (rounded to 2 decimals)
   - `Debt_to_Equity` must be calculated: `TotalLiabilities / TotalShareholderEquity` (rounded to 2 decimals)
   - Column order should match (or ensure downstream notebooks can handle different order)

### 5. Data Type Requirements
   - **Numeric fields**: Must be convertible to float (use `pd.to_numeric(errors='coerce')`)
   - **Date fields**: Must be in YYYY-MM-DD format (use `pd.to_datetime(errors='coerce')`)
   - **Symbol**: Must be uppercase string
   - Handle missing values: Use NaN for missing numeric data, NaT for missing dates

### 6. Data Quality Checks
   - Filter out rows where `FiscalDateEnding` is null before saving
   - Ensure `FiscalDateEnding` represents end of fiscal quarter (not start)
   - Handle negative values appropriately (NetIncome, OperatingIncome can be negative)
   - Handle very large numbers (revenue/assets can be in billions)

### 7. Deduplication Logic
   - Must deduplicate on `Symbol` + `FiscalDateEnding` combination
   - Keep most recent `DateAdded` when duplicates exist
   - Sort by `Symbol`, `FiscalDateEnding` (descending), `DateAdded` (descending) before deduplication

### 8. Testing Checklist
   - [ ] Verify income_statement.csv has exact 7 columns listed above
   - [ ] Verify balance_sheet.csv has exact 14 columns listed above (including calculated fields)
   - [ ] Verify all numeric columns are float type (not string)
   - [ ] Verify dates are in YYYY-MM-DD format
   - [ ] Verify deduplication works correctly
   - [ ] Verify data can be loaded by `company_report_processing.ipynb` without errors

## Dependencies

- `pandas`: Data manipulation and CSV handling
- `requests`: HTTP requests to Alpha Vantage API
- `datetime`: Date handling and freshness checks
- `time`: Rate limiting delays
- `os`: File path operations
- `sector_mapping`: Symbol list import

## Output File Specifications Summary

### File 1: `Reports/income_statement.csv`
**Purpose**: Store quarterly income statement data

**Columns (7 total, exact order)**:
1. `Symbol` (string, uppercase)
2. `FiscalDateEnding` (date, YYYY-MM-DD)
3. `TotalRevenue` (float)
4. `GrossProfit` (float)
5. `NetIncome` (float, can be negative)
6. `OperatingIncome` (float, can be negative)
7. `DateAdded` (date, YYYY-MM-DD)

**Data Requirements**:
- One row per symbol per quarter
- Sorted by Symbol, FiscalDateEnding (descending)
- No duplicates on Symbol + FiscalDateEnding
- Missing FiscalDateEnding rows filtered out

### File 2: `Reports/balance_sheet.csv`
**Purpose**: Store merged quarterly balance sheet AND income statement data

**Columns (14 total, exact order)**:
1. `Symbol` (string, uppercase)
2. `FiscalDateEnding` (date, YYYY-MM-DD)
3. `TotalRevenue` (float) - from income statement
4. `GrossProfit` (float) - from income statement
5. `NetIncome` (float, can be negative) - from income statement
6. `OperatingIncome` (float, can be negative) - from income statement
7. `OperatingMargin` (float, percentage) - CALCULATED: OperatingIncome / TotalRevenue × 100
8. `TotalAssets` (float) - from balance sheet
9. `TotalLiabilities` (float) - from balance sheet
10. `CommonStockSharesOutstanding` (float) - from balance sheet
11. `TotalShareholderEquity` (float, can be negative) - from balance sheet
12. `BVPS` (float, rounded to 2 decimals) - CALCULATED: TotalShareholderEquity / CommonStockSharesOutstanding
13. `Debt_to_Equity` (float, rounded to 2 decimals) - CALCULATED: TotalLiabilities / TotalShareholderEquity
14. `Date` (date, YYYY-MM-DD)

**Data Requirements**:
- One row per symbol per quarter
- Contains BOTH income statement and balance sheet data (merged)
- Sorted by Symbol, FiscalDateEnding (descending)
- No duplicates on Symbol + FiscalDateEnding
- Missing FiscalDateEnding rows filtered out
- Data filtered to dates after 2014-01-01

**CRITICAL**: This file is used as input by `company_report_processing.ipynb` - all columns must be present!
