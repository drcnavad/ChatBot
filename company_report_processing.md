# Company Report Processing - Financial Analysis Documentation

## Overview

This notebook processes balance sheet and income statement data to calculate comprehensive financial ratios, TTM (Trailing Twelve Months) metrics, and fair value estimations. It transforms raw financial data into actionable analysis metrics.

## Input Data

### Source Files

- **`Reports/balance_sheet.csv`**: Contains **merged** quarterly balance sheet AND income statement data
  - **CRITICAL**: This file must contain BOTH income statement and balance sheet columns
  - **Exact Required Columns** (14 columns):
    ```
    Symbol, FiscalDateEnding, TotalRevenue, GrossProfit, NetIncome, OperatingIncome, 
    OperatingMargin, TotalAssets, TotalLiabilities, CommonStockSharesOutstanding, 
    TotalShareholderEquity, BVPS, Debt_to_Equity, Date
    ```
  - All columns must be present for the notebook to work correctly
  - Data should be sorted by `Symbol` and `FiscalDateEnding` (ascending)

### Data Requirements

- **Data Format**: Quarterly data (not annual)
- **Date Format**: `FiscalDateEnding` must be in YYYY-MM-DD format (pandas datetime)
- **Numeric Types**: All financial metrics must be numeric (float), not strings
- **Missing Data**: NaN values are handled, but too many missing values may cause calculation errors
- **Data Range**: Notebook filters data after 2014-01-01 (see code: `final_df1[final_df1["FiscalDateEnding"] > '2014-01-01']`)

## Processing Steps

### 1. Financial Ratios & Metrics Calculation

Calculates key financial ratios and metrics for stock valuation:

#### Profitability Ratios
- **ROE (Return on Equity)**: `NetIncome / TotalShareholderEquity × 100`
- **ROA (Return on Assets)**: `NetIncome / TotalAssets × 100`
- **Net Profit Margin**: `NetIncome / TotalRevenue × 100`
- **Gross Margin**: `GrossProfit / TotalRevenue × 100`
- **Operating Margin**: `OperatingIncome / TotalRevenue × 100` (already calculated in base data)

#### Per Share Metrics
- **EPS (Earnings Per Share)**: `NetIncome / CommonStockSharesOutstanding`
- **Revenue Per Share**: `TotalRevenue / CommonStockSharesOutstanding`
- **Assets Per Share**: `TotalAssets / CommonStockSharesOutstanding`
- **Operating Income Per Share**: `OperatingIncome / CommonStockSharesOutstanding`
- **BVPS (Book Value Per Share)**: `TotalShareholderEquity / CommonStockSharesOutstanding`

#### Efficiency Ratios
- **Asset Turnover**: `TotalRevenue / TotalAssets`
- **Equity Turnover**: `TotalRevenue / TotalShareholderEquity`
- **Revenue to Equity Ratio**: `TotalRevenue / TotalShareholderEquity`
- **Net Income to Assets**: `NetIncome / TotalAssets × 100`

#### Financial Health Ratios
- **Debt to Assets Ratio**: `TotalLiabilities / TotalAssets × 100`
- **Equity Ratio**: `TotalShareholderEquity / TotalAssets × 100`
- **Debt to Equity Ratio**: `TotalLiabilities / TotalShareholderEquity` (already calculated in base data)

#### Growth Metrics
- **Revenue Growth (QoQ)**: Quarter-over-quarter percentage change
- **Revenue Growth (YoY)**: Year-over-year percentage change (4 quarters)
- **Net Income Growth (QoQ)**: Quarter-over-quarter percentage change
- **Net Income Growth (YoY)**: Year-over-year percentage change (4 quarters)
- **EPS Growth (QoQ)**: Quarter-over-quarter percentage change
- **EPS Growth (YoY)**: Year-over-year percentage change (4 quarters)

### 2. TTM (Trailing Twelve Months) Metrics

Calculates TTM metrics that match online financial sources (Yahoo Finance, Investing.com):

#### TTM Calculation Method
- **Income Statement Items**: Sum of last 4 quarters
- **Balance Sheet Items**: Average of beginning and ending values

#### TTM Metrics Calculated
- `TTM_Revenue`: Sum of last 4 quarters' revenue
- `TTM_NetIncome`: Sum of last 4 quarters' net income
- `TTM_OperatingIncome`: Sum of last 4 quarters' operating income
- `TTM_GrossProfit`: Sum of last 4 quarters' gross profit
- `Avg_ShareholderEquity`: Average of beginning and ending equity
- `Avg_TotalAssets`: Average of beginning and ending assets

#### TTM Ratios
- `TTM_ROE`: `TTM_NetIncome / Avg_ShareholderEquity × 100`
- `TTM_ROA`: `TTM_NetIncome / Avg_TotalAssets × 100`
- `TTM_NetProfitMargin`: `TTM_NetIncome / TTM_Revenue × 100`
- `TTM_OperatingMargin`: `TTM_OperatingIncome / TTM_Revenue × 100`
- `TTM_GrossMargin`: `TTM_GrossProfit / TTM_Revenue × 100`
- `TTM_EPS`: `TTM_NetIncome / CommonStockSharesOutstanding`
- `TTM_RevenuePerShare`: `TTM_Revenue / CommonStockSharesOutstanding`
- `TTM_AssetTurnover`: `TTM_Revenue / Avg_TotalAssets`
- `TTM_EquityTurnover`: `TTM_Revenue / Avg_ShareholderEquity`

### 3. Fair Value Estimation

Estimates fair value using multiple valuation methods:

#### Valuation Methods

1. **P/E Ratio Analysis**: Based on industry average P/E
2. **P/B Ratio**: Price-to-Book valuation
3. **P/S Ratio**: Price-to-Sales (for unprofitable companies)
4. **PEG Ratio**: P/E adjusted for growth
5. **Graham Formula**: Intrinsic value calculation
6. **Growth-Adjusted P/E**: Fair P/E based on growth rate
7. **Composite Fair Value**: Weighted average of multiple methods

#### Valuation Logic

- **Profitable Companies**: Use P/E, Growth P/E, P/B, and Graham methods
- **Unprofitable Companies**: Use P/B and P/S methods only

#### Output Fields
- `FairValue`: Estimated fair value per share
- `Price_vs_FairValue`: Percentage difference from current price to fair value
- `Is_Profitable`: Boolean flag indicating profitability status
- `Valuation_Signal`: Buy/Hold/Sell signal based on fair value

### 4. Advanced Metrics

Calculates additional advanced financial metrics:

- **DuPont Analysis**: Breakdown of ROE components
- **ROIC (Return on Invested Capital)**: Profitability relative to invested capital
- **Leverage Ratios**: Various debt and leverage metrics
- **Working Capital Metrics**: Current assets vs. liabilities analysis

## Output Data

### Excel Export: `Reports/complete_company_analysis.xlsx`

Exports data to Excel with multiple sheets:

1. **Sheet 1: `1_Historical_All_Quarters`**
   - Complete historical data with all calculated metrics
   - All quarters for all symbols
   - ~49+ columns including all ratios, TTM metrics, and growth rates

2. **Sheet 2: `2_Latest_Quarter_Complete`**
   - Latest quarter data for each symbol
   - Same columns as historical sheet
   - Used as input for scoring notebook

### Data Structure

Each row contains:
- **Identifier**: `Symbol`, `FiscalDateEnding`
- **Raw Financial Data**: `TotalRevenue`, `GrossProfit`, `NetIncome`, `OperatingIncome`, `TotalAssets`, `TotalLiabilities`, `TotalShareholderEquity`, `CommonStockSharesOutstanding`
- **Calculated Ratios**: All profitability, efficiency, and health ratios
- **TTM Metrics**: All trailing twelve months calculations
- **Growth Metrics**: QoQ and YoY growth rates
- **Fair Value**: Valuation estimates and signals

## Key Calculations Summary

### Total Columns Generated
- **Base Columns**: ~14 (from input CSV)
- **Calculated Ratios**: ~20 additional metrics
- **TTM Metrics**: ~15 additional TTM calculations
- **Fair Value Metrics**: ~5-10 valuation fields
- **Total**: ~49-54 columns in final output

### Data Processing Features

- **Grouped Calculations**: All growth metrics calculated per symbol using `groupby()`
- **Time Series Handling**: Proper sorting by `FiscalDateEnding` for accurate TTM calculations
- **Error Handling**: Replaces infinite values and handles division by zero
- **Data Validation**: Ensures proper data types and handles missing values

## Dependencies

- `pandas`: Data manipulation and Excel export
- `numpy`: Numerical calculations and array operations
- `openpyxl`: Excel file writing
- `datetime`: Date handling for TTM calculations

## Output File Specifications Summary

### File: `Reports/complete_company_analysis.xlsx`

**Purpose**: Store comprehensive financial analysis with all calculated metrics

### Sheet 1: `1_Historical_All_Quarters`
**Purpose**: Complete historical data with all calculated metrics

**Key Columns** (~49-54 total):
- Identifiers: `Symbol`, `FiscalDateEnding`
- Raw Data: `TotalRevenue`, `GrossProfit`, `NetIncome`, `OperatingIncome`, `OperatingMargin`, `TotalAssets`, `TotalLiabilities`, `CommonStockSharesOutstanding`, `TotalShareholderEquity`, `BVPS`, `Debt_to_Equity`
- Profitability Ratios: `ROE`, `ROA`, `NetProfitMargin`, `GrossMargin`
- Per Share Metrics: `EPS`, `RevenuePerShare`, `AssetsPerShare`, `OperatingIncomePerShare`
- Efficiency Ratios: `AssetTurnover`, `EquityTurnover`, `RevenueToEquity`, `NetIncomeToAssets`
- Health Ratios: `DebtToAssets`, `EquityRatio`
- Growth Metrics: `RevenueGrowth_QoQ`, `RevenueGrowth_YoY`, `NetIncomeGrowth_QoQ`, `NetIncomeGrowth_YoY`, `EPSGrowth_QoQ`, `EPSGrowth_YoY`
- TTM Metrics: `TTM_Revenue`, `TTM_NetIncome`, `TTM_OperatingIncome`, `TTM_GrossProfit`, `Avg_ShareholderEquity`, `Avg_TotalAssets`, `TTM_ROE`, `TTM_ROA`, `TTM_NetProfitMargin`, `TTM_OperatingMargin`, `TTM_GrossMargin`, `TTM_EPS`, `TTM_RevenuePerShare`, `TTM_AssetTurnover`, `TTM_EquityTurnover`
- Fair Value: `FairValue`, `Price_vs_FairValue`, `Is_Profitable`, `Valuation_Signal`

**Data Requirements**:
- All quarters for all symbols
- Sorted by Symbol, FiscalDateEnding (ascending)
- Data after 2014-01-01 only

### Sheet 2: `2_Latest_Quarter_Complete` (CRITICAL - Used by scoring notebook)
**Purpose**: Latest quarter data for each symbol (input for scoring)

**Required Columns** (must match exactly for scoring notebook):
- `Symbol` - Stock ticker
- `Price_vs_FairValue` - Percentage difference from fair value
- `TTM_ROE` - Trailing Twelve Months ROE
- `TTM_ROA` - Trailing Twelve Months ROA
- `TTM_NetProfitMargin` - Trailing Twelve Months Net Profit Margin
- `RevenueGrowth_YoY` - Year-over-year revenue growth
- `Debt_to_Equity` - Debt to equity ratio
- `DebtToAssets` - Debt to assets ratio
- `TTM_AssetTurnover` - Trailing Twelve Months Asset Turnover
- `TTM_EquityTurnover` - Trailing Twelve Months Equity Turnover
- `EPSGrowth_YoY` - Year-over-year EPS growth
- `NetIncomeGrowth_YoY` - Year-over-year net income growth
- `TTM_GrossMargin` - Trailing Twelve Months Gross Margin
- `TTM_OperatingMargin` - Trailing Twelve Months Operating Margin
- Plus all other calculated columns from historical sheet

**Data Requirements**:
- One row per symbol (latest quarter only)
- All columns from historical sheet
- Must have all 14 columns listed above for scoring to work
- Sorted by Symbol

## Notes for Future API Migration

When adapting this notebook for a different data source (SimFin, FNP, etc.):

### 1. Input Data Structure (CRITICAL)
   **The `balance_sheet.csv` file MUST contain these exact columns:**
   ```
   Symbol, FiscalDateEnding, TotalRevenue, GrossProfit, NetIncome, OperatingIncome, 
   OperatingMargin, TotalAssets, TotalLiabilities, CommonStockSharesOutstanding, 
   TotalShareholderEquity, BVPS, Debt_to_Equity, Date
   ```
   - All 14 columns must be present
   - Column names must match exactly (case-sensitive)
   - Missing columns will cause the notebook to fail

### 2. Data Format Requirements
   - **Quarterly data**: Must be quarterly (not annual) - TTM calculations depend on this
   - **Date format**: `FiscalDateEnding` must be pandas datetime (YYYY-MM-DD)
   - **Numeric types**: All financial metrics must be float64 (not strings)
   - **Data sorting**: Data should be sorted by `Symbol`, `FiscalDateEnding` (ascending) for TTM calculations
   - **Minimum data**: Each symbol should have at least 4 quarters for accurate TTM calculations

### 3. TTM Calculation Requirements
   - **Chronological order**: Data must be sorted by `FiscalDateEnding` (ascending) per symbol
   - **Quarter spacing**: Assumes quarters are approximately 3 months apart
   - **Data completeness**: TTM requires last 4 quarters - missing quarters will result in NaN
   - **Grouping**: Calculations are done per `Symbol` using `groupby()`

### 4. Fair Value Calculations
   - **Stock price**: Requires current stock price (may need to fetch separately if not in input)
   - **Industry data**: May need industry/sector classifications for P/E comparisons
   - **Market benchmarks**: May need market average ratios for comparison
   - **Growth rates**: Uses `RevenueGrowth_YoY` and `EPSGrowth_YoY` for growth-adjusted valuations

### 5. Output Format Requirements (CRITICAL)
   **Excel file `complete_company_analysis.xlsx` must have these sheets:**
   
   **Sheet 1: `1_Historical_All_Quarters`**
   - All historical quarters with all calculated metrics
   - ~49-54 columns total
   
   **Sheet 2: `2_Latest_Quarter_Complete`** (REQUIRED for scoring notebook)
   - Latest quarter data for each symbol
   - Must contain these columns for scoring:
     - `Symbol` (identifier)
     - `Price_vs_FairValue` (for Valuation Score)
     - `TTM_ROE`, `TTM_ROA`, `TTM_NetProfitMargin` (for Profitability Score)
     - `RevenueGrowth_YoY` (for Growth Score)
     - `Debt_to_Equity`, `DebtToAssets` (for Debt Score)
     - `TTM_AssetTurnover`, `TTM_EquityTurnover` (for Efficiency Score)
     - `EPSGrowth_YoY`, `NetIncomeGrowth_YoY` (for Earnings Quality Score)
     - `TTM_GrossMargin`, `TTM_OperatingMargin` (for Margin Quality Score)
   - Column names must match exactly for `company_report_scoring.ipynb` to work

### 6. Calculation Dependencies
   - **ROE/ROA**: Require `NetIncome`, `TotalShareholderEquity`, `TotalAssets`
   - **Margins**: Require `GrossProfit`, `OperatingIncome`, `TotalRevenue`
   - **Per Share Metrics**: Require `CommonStockSharesOutstanding`
   - **Growth Metrics**: Require historical data (at least 4 quarters for YoY)
   - **TTM Metrics**: Require at least 4 consecutive quarters per symbol

### 7. Error Handling
   - Division by zero: Handled with `.replace([np.inf, -np.inf], np.nan)`
   - Missing values: NaN values propagate through calculations
   - Date parsing: Uses `pd.to_datetime(errors='coerce')` - invalid dates become NaT
   - Data filtering: Filters out data before 2014-01-01

### 8. Testing Checklist
   - [ ] Verify `balance_sheet.csv` has all 14 required columns
   - [ ] Verify data loads without errors
   - [ ] Verify TTM calculations produce valid results (not all NaN)
   - [ ] Verify Excel export creates both required sheets
   - [ ] Verify Sheet 2 has all columns required by scoring notebook
   - [ ] Verify calculations handle missing data gracefully
