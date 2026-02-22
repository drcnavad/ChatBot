# Stock Price Prediction Model
## Machine Learning Tool for Technical Analysis

---

## 📋 Overview

This is a reusable machine learning model for predicting stock prices based on technical analysis. It uses an ensemble of three models:
- **Linear Regression** (20% weight) - Captures linear trends
- **Random Forest** (40% weight) - Handles non-linear patterns  
- **Gradient Boosting** (40% weight) - Sequential error correction

## 🚀 Quick Start

### Basic Usage

```bash
python stock_predictor.py --ticker BKR --current-price 56.85 --days 30
python stock_predictor.py --ticker AAPL --current-price 185.50 --days 60
python stock_predictor.py --ticker TSLA --current-price 195.30 --days 14
```

### Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--ticker` | Yes | - | Stock ticker symbol (e.g., BKR, AAPL, MSFT) |
| `--current-price` | Yes* | - | Current stock price |
| `--days` | No | 30 | Number of days to predict ahead |
| `--output-dir` | No | /mnt/user-data/outputs | Directory for output files |

*Note: Required when using synthetic data mode

## 📦 Installation

### Requirements

```bash
pip install numpy scikit-learn matplotlib --break-system-packages
```

Or if you're in a standard Python environment:

```bash
pip install numpy scikit-learn matplotlib
```

## 📊 What the Model Does

### Technical Features Analyzed

The model creates the following technical indicators:

1. **Moving Averages** (5, 10, 20, 30-day)
2. **Distance from Moving Averages**
3. **Momentum Indicators** (5, 10, 20-day momentum)
4. **Volatility Measures** (10 and 20-day standard deviation)
5. **Price Position** (relative position in recent trading range)

### Training Process

1. Historical price data is split into training (80%) and validation (20%)
2. Technical features are computed for each time period
3. Three models are trained independently
4. Validation metrics (MAE, RMSE, MAPE) are calculated
5. Models are combined into weighted ensemble

### Prediction Process

1. Uses the most recent 60 days of price data
2. Generates features for the current state
3. Each model makes an independent prediction
4. Predictions are combined using weighted averaging
5. Process repeats for each day in the prediction horizon

## 📈 Output Files

When you run the model, it creates three files:

### 1. `{TICKER}_prediction.json`
```json
{
  "ticker": "BKR",
  "current_price": 56.85,
  "prediction_days": 30,
  "ensemble_prediction": 50.65,
  "confidence_range": {
    "lower": 47.12,
    "upper": 54.19
  },
  "model_predictions": {
    "Linear Regression": 56.60,
    "Random Forest": 49.75,
    "Gradient Boosting": 48.59
  }
}
```

### 2. `{TICKER}_prediction.png`
- Historical price chart
- Predicted price trajectory
- Confidence intervals
- Model comparison bar chart

### 3. `{TICKER}_model.json`
- Model parameters
- Validation scores
- Feature scaling information

## 🔧 Using as a Python Library

You can also import and use the model programmatically:

```python
from stock_predictor import StockPricePredictor
import numpy as np

# Create predictor
predictor = StockPricePredictor(ticker='AAPL')

# Your historical price data (last 180 days recommended)
prices = np.array([150.5, 151.2, 152.0, ...])  # Your actual data

# Train the model
predictor.train(prices)

# Make prediction
results = predictor.predict_future(prices, days_ahead=30)

# Print report
predictor.print_prediction_report(results)

# Create visualization
predictor.create_visualization(prices, results, 'output.png')

# Save model
predictor.save_model('my_model.json')
```

## 📊 Example Output

```
======================================================================
BKR STOCK PRICE PREDICTION
======================================================================

Current Price:           $56.85
Predicted Price (30 days): $50.65
Expected Change:         $-6.20 (-10.90%)

Confidence Range:        $47.12 - $54.19
Model Agreement (σ):     $3.53

======================================================================
INDIVIDUAL MODEL PREDICTIONS
======================================================================
Linear Regression        : $ 56.60  ( -0.25 /  -0.44%)
Random Forest            : $ 49.75  ( -7.10 / -12.50%)
Gradient Boosting        : $ 48.59  ( -8.26 / -14.53%)
```

## 🎯 Use Cases

### ✅ Good For:
- **Technical analysis** and pattern recognition
- **Short-term predictions** (1-30 days)
- **Educational purposes** and learning ML
- **Comparing multiple models** on same stock
- **Backtesting** technical strategies

### ❌ Not Good For:
- Long-term predictions (>30 days)
- Fundamental analysis
- Accounting for earnings or news events
- Making actual investment decisions without further research

## ⚠️ Important Limitations

### What the Model DOES NOT Include:

1. **Fundamental Analysis**: No earnings, revenue, P/E ratios
2. **News Events**: Company announcements, mergers, scandals
3. **Market Conditions**: Fed policy, interest rates, inflation
4. **Sector Trends**: Industry-specific developments
5. **Macroeconomic Factors**: GDP, unemployment, consumer sentiment
6. **Geopolitical Events**: Wars, elections, policy changes
7. **Market Sentiment**: Social media trends, analyst ratings changes

### Key Points to Remember:

- ⚠️ Stock prices are inherently **unpredictable**
- ⚠️ Model accuracy **decreases** with longer time horizons
- ⚠️ Past performance does **NOT** guarantee future results
- ⚠️ This is **NOT** financial advice
- ⚠️ Always do your own research
- ⚠️ Consult a qualified financial advisor before investing

## 🔄 Adapting for Production Use

Currently, the model uses **synthetic data** for demonstration. To use with real stock data:

### Option 1: Using yfinance (Recommended)

Replace the `generate_synthetic_data` function with:

```python
import yfinance as yf

def get_real_stock_data(ticker, period='1y'):
    """Fetch real stock data from Yahoo Finance"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist['Close'].values
```

Then modify `main()`:
```python
# Replace this line:
prices = generate_synthetic_data(args.ticker, args.current_price)

# With this:
prices = get_real_stock_data(args.ticker)
```

### Option 2: Using Alpha Vantage API

```python
import requests

def get_stock_data_alpha_vantage(ticker, api_key):
    """Fetch data from Alpha Vantage"""
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': ticker,
        'apikey': api_key,
        'outputsize': 'full'
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extract close prices
    time_series = data['Time Series (Daily)']
    prices = [float(v['4. close']) for v in time_series.values()]
    return np.array(prices[::-1])  # Reverse to chronological order
```

### Option 3: CSV File

```python
import pandas as pd

def load_stock_data_from_csv(filepath):
    """Load stock data from CSV file"""
    df = pd.read_csv(filepath)
    # Assuming CSV has 'Close' column
    return df['Close'].values
```

## 📚 Model Performance Metrics

The model reports three key metrics during validation:

1. **MAE (Mean Absolute Error)**: Average prediction error in dollars
2. **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily
3. **MAPE (Mean Absolute Percentage Error)**: Error as percentage of actual price

**Typical Performance:**
- MAE: $2-5 for stable stocks
- MAPE: 3-10% for short-term predictions
- Accuracy decreases significantly beyond 1-week horizon

## 🤝 Contributing & Customization

### Adding New Models

To add a new model to the ensemble:

```python
from sklearn.svm import SVR

# In __init__ method:
self.models['Support Vector Regression'] = SVR(kernel='rbf', C=1.0)

# Update ensemble weights in predict_future:
ensemble_pred = (
    0.15 * lr_pred +
    0.30 * rf_pred +
    0.30 * gb_pred +
    0.25 * svr_pred
)
```

### Adding New Features

Add to the `create_features` method:

```python
# Example: RSI indicator
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

# Then add to features:
feat.append(calculate_rsi(prices[:i]))
```

## 📞 Support

For questions or issues:
1. Review this README thoroughly
2. Check that all dependencies are installed
3. Verify your data format matches expectations
4. Try with different stocks to see if issue persists

## 📄 License

This model is provided for educational purposes only. Use at your own risk.

## ⚡ Quick Reference Card

```bash
# Predict BKR for next month
python stock_predictor.py --ticker BKR --current-price 56.85 --days 30

# Predict AAPL for next 2 weeks
python stock_predictor.py --ticker AAPL --current-price 185.50 --days 14

# Predict TSLA for next quarter (90 days)
python stock_predictor.py --ticker TSLA --current-price 195.30 --days 90

# Custom output directory
python stock_predictor.py --ticker MSFT --current-price 410.00 --days 30 --output-dir ./my_predictions
```

---

## 📁 Documentation Index

| Doc | Description |
|-----|-------------|
| [company_report_base.md](company_report_base.md) | Data fetching (Alpha Vantage, 5-year, IQR capping) |
| [company_report_processing.md](company_report_processing.md) | Financial ratios, TTM, fair value |
| [company_report_scoring.md](company_report_scoring.md) | Fundamental weight scoring (-10 to +10) |
| [company_report_visualization.md](company_report_visualization.md) | Plotly charts for symbol analysis |

---

**Remember**: This is a statistical model for educational purposes. Always do thorough research and consult financial professionals before making investment decisions. Past performance does not guarantee future results.
