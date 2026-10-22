"""The Stock Analysis dashboard (Streamlit), one module per tab / major panel; app.py is the entry point.

  settings.py      live-rule settings, rules text, report file locations
  style.py         page config + CSS, colors, HTML / text helpers shared by every panel
  data.py          cached report loading, live quote
  ai.py            optional AI analysis
  signals.py       plain-language signals, rebalance plan
  history.py       one stock's live-strategy history
  page.py          page state (build_page), title bar
  stock_view.py    Stock Analysis tab: stock list, picker, header, detail card
  stock_groups.py  stock rankings helper (manual_trade.py uses _rankings)
  heatmap.py       Stock Analysis tab: the clickable tile grid (signal order, day % colors, hover intraday lines)
  live_stream.py   one Alpaca SIP websocket per server process for the live 1D chart
  stock_chart.py   Stock Analysis tab: price chart
  short_stock.py   Stock Analysis tab: stocks too new to trade
  sector_table.py  Home tab: sector performance (14 days to 1 year)
  notes.py         Stock Analysis tab: your notes table (add / edit / delete, saved to Reports/my_notes.csv)
  details/         Strategy and Trading Account tabs: one module per expander (details/__init__.py lays them out)
"""
