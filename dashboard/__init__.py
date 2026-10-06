"""The Stock Analysis dashboard (Streamlit), one module per tab / major panel; app.py is the entry point.

  settings.py      live-rule settings, rules text, report file locations
  style.py         page config + CSS, colors, HTML / text helpers shared by every panel
  data.py          cached report loading, live quote
  ai.py            optional AI analysis
  signals.py       plain-language signals, rebalance plan
  history.py       one stock's live-strategy history
  page.py          page state (build_page), title bar
  stock_view.py    Home tab: stock list, picker, header, detail card
  stock_chart.py   Home tab: price chart
  short_stock.py   Home tab: stocks too new to trade
  details/         Strategy and Trading Account tabs: one module per expander (details/__init__.py lays them out)
"""
