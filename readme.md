# 📈 Stock Price Forecasting with Technical Indicators & News Sentiment

## 🔹 Project Overview
This project predicts the **next-day stock closing price** by combining **historical price behavior** with **financial news sentiment**.  
It demonstrates an end-to-end forecasting pipeline that integrates **technical indicators**, **sentiment analysis**, and **machine learning**, deployed as an interactive **Streamlit application**.


## 🔹 Data Sources
- **Stock Prices:** Daily OHLCV data fetched dynamically from Polygon.io API  
- **News Data:** Financial headlines and summaries from Polygon.io News API  
- **Sentiment:** News scored using **FinBERT** and aggregated daily  

All data is fetched **live based on user input** (ticker and date range).


## 🔹 Feature Engineering
- Technical indicators: SMA, RSI, MACD, ATR, Bollinger Band Width  
- Lag features: 1-day, 2-day, 3-day close prices  
- News sentiment: Daily aggregated FinBERT score  
- Target variable: **Next-day closing price**


## 🔹 Models Evaluated
- **LSTM (baseline)** – captures time-series patterns  
- **XGBoost** – strong non-linear tabular learner  
- **Hybrid LSTM + XGBoost (Final Model)** – LSTM learns sequences, XGBoost corrects errors  

**Best performance achieved with the hybrid model.**


## 🔹 Results (R² Approx.)
- **AAPL:** ~0.80  
- **TSLA:** ~0.68  
- **CRM:** ~0.70  
- **GOOGL:** ~0.60  
- **MSFT / NVDA:** lower due to higher volatility  

Performance varies by stock volatility and news sensitivity.


## 🔹 Deployment
- Interactive **Streamlit dashboard**
- User selects ticker and date range
- Model retrains automatically
- Displays metrics, backtest plots, and future forecasts (1–15 days)


## 🔹 Key Takeaways
- Sentiment improves predictions for volatile stocks  
- Technical indicators stabilize forecasts for trending stocks  
- Hybrid modeling outperforms single-model approaches  
- Designed as a **forecasting prototype**, not a trading bot  


## 🔹 Tools & Tech
Python, Pandas, NumPy, Streamlit, TensorFlow/Keras, XGBoost, HuggingFace (FinBERT), Polygon.io


## 🔹 Results (Screenshots)

### Apple (AAPL)
![Apple Results](/images/apple_results.jpeg)

### Tesla (TSLA)
![Tesla Results](/images/tesla_results.jpeg)

### Google (GOOG)
![Google Results](/images/google_results.jpeg)

### Salesforce (CRM)
![CRM Results](/images/CRM_results.jpeg)

### Price Prediction
![Price Prediction](/images/price_prediction.jpeg)
