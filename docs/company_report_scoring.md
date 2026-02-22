# Company Report Scoring - Fundamental Weight Analysis

## Overview

Multi-factor fundamental scoring. Output: `Fundamental_Weight` scaled **-10 to +10**.

## Input

- **`Reports/complete_company_analysis.xlsx`** (Sheet: `2_Latest_Quarter_Complete`)
- Required: Symbol, Price_vs_FairValue, TTM_ROE, TTM_ROA, TTM_NetProfitMargin, RevenueGrowth_YoY, Debt_to_Equity, DebtToAssets, TTM_AssetTurnover, TTM_EquityTurnover, EPSGrowth_YoY, NetIncomeGrowth_YoY, TTM_GrossMargin, TTM_OperatingMargin

## Component Weights (8, total 100%)

| Component | Weight | Metrics |
|-----------|--------|---------|
| Valuation | 25% | Price_vs_FairValue |
| Profitability | 20% | TTM_ROE, TTM_ROA, TTM_NetProfitMargin |
| Growth | 15% | RevenueGrowth_YoY |
| Debt | 15% | Debt_to_Equity, DebtToAssets |
| Earnings Quality | 10% | EPSGrowth_YoY, NetIncomeGrowth_YoY |
| Efficiency | 5% | TTM_AssetTurnover, TTM_EquityTurnover |
| Margin Quality | 5% | TTM_GrossMargin, TTM_OperatingMargin |
| Return Quality | 5% | TTM_ROE, TTM_ROA |

## Scoring Logic (summary)

- **Valuation**: < -20% → 10, < -10% → 7, < 10% → 3, >= 10% → -5
- **Profitability/Growth/Debt/Earnings/Efficiency/Margins/Return**: Each component has thresholds; scores capped -10 to +10
- **Final**: Weighted sum → MinMaxScaler → -10 to +10

## Output

**`Reports/balance_sheet_weights.csv`**
- Symbol, 8 component scores, Fundamental_Weight, underlying metrics
- Used by `app.py` (Streamlit) – `Fundamental_Weight` column must stay

## Dependencies

pandas, numpy, sklearn.preprocessing.MinMaxScaler, openpyxl
