# 📈 Stock Price Forecasting with Technical Indicators and News Sentiment

## 🔹 Project Goal
The goal of this project is to **predict next-day stock closing prices** by combining:
- **Numerical features** from historical price data and technical indicators
- **Textual features** from financial news sentiment (FinBERT)

This hybrid approach allows us to capture both **market trends** and **investor sentiment** for more reliable short-term stock forecasting.

---

## 🔹 Key Features
- ✅ Interactive **Streamlit dashboard** to test multiple tickers (AAPL, TSLA, GOOGL, CRM, etc.)
- ✅ **Hybrid deep learning model** (LSTM for sequential data + XGBoost for residual learning)
- ✅ Integration of **news sentiment** via FinBERT (ProsusAI/finbert)
- ✅ **Auto-tuning of hyperparameters** (sequence length, epochs) for each ticker
- ✅ **Future price forecasting** for 1–15 days
- ✅ **Smoothing option** (moving average) to stabilize volatile forecasts
- ✅ Evaluation with standard metrics:
  - RMSE (Root Mean Squared Error)  
  - MAE (Mean Absolute Error)  
  - MAPE (Mean Absolute Percentage Error)  
  - R² (Coefficient of Determination)

---

## 🔹 How It Works
### 1. Data Collection
- Historical daily stock prices from **Polygon.io API**  
- Financial news headlines + summaries from **Polygon.io News API**

### 2. Feature Engineering
- Technical indicators:
  - SMA (Simple Moving Average)  
  - RSI (Relative Strength Index)  
  - MACD (Moving Average Convergence Divergence)  
  - ATR (Average True Range)  
  - Bollinger Band Width  
  - Price lags (1-day, 2-day, 3-day)  
- News sentiment analysis:
  - Headlines + summaries scored with **FinBERT**
  - Daily sentiment aggregated and merged with price data

### 3. Modeling
- **LSTM (Long Short-Term Memory)** to capture time series dependencies  
- **XGBoost regressor** stacked on top of LSTM outputs  
- Auto-tuned hyperparameters chosen by lowest RMSE on validation split

### 4. Forecasting
- Next-day close predictions compared against true values  
- Multi-step **future forecasting** (up to 15 days)  
- **Optional smoothing** using moving averages  

---

## 🔹 Tools & Technologies
- **Python** (NumPy, Pandas, Matplotlib, Plotly)  
- **Streamlit** for interactive dashboard  
- **Scikit-learn** for metrics and scaling  
- **TensorFlow / Keras** for LSTM model  
- **XGBoost** for hybrid boosting  
- **Transformers (HuggingFace)** for FinBERT sentiment  
- **TA-lib / Technical Analysis Library** for indicators  
- **Polygon.io API** for stock + news data  

---

## 🔹 Results

### 📊 Performance Highlights
- **AAPL (Apple)**: R² ~ 0.80, very strong predictive power  
- **TSLA (Tesla)**: R² ~ 0.68, stable despite volatility  
- **GOOGL (Alphabet)**: R² ~ 0.60, balanced results  
- **CRM (Salesforce)**: R² ~ 0.70, consistent predictions  
- **MSFT (Microsoft) & NVDA (Nvidia)**: weaker performance due to higher volatility & noisier sentiment signal  

### 📷 Visual Results
- Interactive charts showing **True vs Predicted prices**  
- Future forecast tables with raw + smoothed predictions  
- Sentiment analysis tables for daily news impact  

## 📊 Results (Screenshots)

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


---

## 🔹 Key Learnings
- Sentiment improves performance for volatile stocks (TSLA, CRM).  
- Technical indicators stabilize predictions for trending stocks (AAPL, GOOGL).  
- Some tickers (MSFT, NVDA) show lower predictability, highlighting the challenge of **volatility patterns**.  
- Auto-tuning per ticker improves consistency across assets.  

