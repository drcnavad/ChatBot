# Company Report Visualization

## Overview

Plotly line charts for a chosen symbol. Set `SYMBOL` in the charts cell, run all cells. Uses winsorization (2nd/98th percentile) before plotting.

## Data Source

- **Primary**: `Reports/complete_company_analysis.xlsx` (Sheet: `1_Historical_All_Quarters`)
- **Fallback**: `Reports/balance_sheet.csv`

## Structure

1. **Cell 1**: Load df, define `prepare_data(symbol)` and `build_chart()`
2. **Cell 2**: `SYMBOL = 'TEAM'` + 4 charts (change symbol here)

## Charts (4)

1. Revenue & Net Income (dual axis)
2. Margins (%)
3. Balance Sheet ($M)
4. Growth YoY (%)

## prepare_data(symbol)

- Filters df by symbol, sorts by FiscalDateEnding
- Fills ROE, ROA, NetProfitMargin from raw components when missing
- Winsorizes chart columns at 2nd/98th percentile
- Replaces inf/-inf with NaN

## build_chart(d, trace_configs, title, ...)

- trace_configs: list of (col, name, color, yaxis)
- yaxis='y2' for secondary axis
- hline=True adds zero reference line

## Dependencies

numpy, pandas, plotly, IPython.display
