# ==== app.py (auto-tuned + volatility features + future forecast + smoothing + sentiment tables) ====
import os, time, math, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta
from dateutil import parser as dateparser
import streamlit as st
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from transformers import pipeline
import xgboost as xgb
import warnings; warnings.filterwarnings("ignore")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or "YOUR_FALLBACK_KEY"
BASE = "https://api.polygon.io"

def from_unix_ms(ms): return datetime.utcfromtimestamp(ms / 1000.0)

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0),
        "R2": float(r2_score(y_true, y_pred)),
    }

def fetch_prices(ticker, sd, ed):
    url=f"{BASE}/v2/aggs/ticker/{ticker}/range/1/day/{sd}/{ed}"
    params={"adjusted":"true","sort":"asc","limit":50000,"apiKey":POLYGON_API_KEY}
    r=requests.get(url,params=params,timeout=30); r.raise_for_status()
    res=r.json().get("results",[])
    if not res: return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    df=pd.DataFrame(res)
    df["date"]=pd.to_datetime(df["t"].apply(from_unix_ms))
    df=df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["date","open","high","low","close","volume"]].sort_values("date").reset_index(drop=True)

def fetch_news(ticker, start_date, end_date, max_articles=100):
    url = f"{BASE}/v2/reference/news"
    params = {"ticker":ticker,"order":"asc","sort":"published_utc","limit":50,
              "published_utc.gte":start_date,"published_utc.lte":end_date,"apiKey":POLYGON_API_KEY}
    got=[]; tries=0
    while len(got)<max_articles and url:
        r=requests.get(url,params=params,timeout=30)
        if r.status_code==429:
            time.sleep(min(30, 2**min(tries,4))); tries+=1; continue
        r.raise_for_status()
        js=r.json(); results=js.get("results",[])
        if not results: break
        got.extend(results)
        url=js.get("next_url"); params=None
        if url and "apiKey=" not in url: url += f"&apiKey={POLYGON_API_KEY}"
        time.sleep(0.1)
    rows=[]
    for it in got[:max_articles]:
        pu=it.get("published_utc")
        if not pu: continue
        try: d=dateparser.parse(pu).date()
        except: continue
        rows.append({"date":pd.Timestamp(d),"headline":it.get("title") or "","summary":it.get("description") or ""})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame(columns=["date","headline","summary"])

def finbert_daily_sentiment(df_news):
    if df_news.empty: return pd.DataFrame({"date":[],"sentiment":[]})
    clf=pipeline("text-classification",model="ProsusAI/finbert")
    texts=(df_news["headline"].fillna("")+". "+df_news["summary"].fillna("")).tolist()
    labels=[]
    for i in range(0,len(texts),16):
        preds=clf(texts[i:i+16],truncation=True)
        for p in preds:
            lab=(p.get("label") or "NEUTRAL").upper()
            labels.append(1.0 if "POS" in lab else -1.0 if "NEG" in lab else 0.0)
    df=df_news.copy(); df["score"]=labels[:len(df)]
    return df.groupby("date",as_index=False)["score"].mean().rename(columns={"score":"sentiment"})

def make_features(df_px, df_sent):
    df=df_px.copy()
    df["sma_10"]=SMAIndicator(df["close"],10,fillna=True).sma_indicator()
    df["rsi_14"]=RSIIndicator(df["close"],14,fillna=True).rsi()
    macd=MACD(df["close"],26,12,9,fillna=True)
    df["macd"],df["macd_signal"],df["macd_hist"]=macd.macd(),macd.macd_signal(),macd.macd_diff()

    atr=AverageTrueRange(high=df["high"],low=df["low"],close=df["close"],window=14,fillna=True)
    df["atr_14"]=atr.average_true_range()
    ret=df["close"].pct_change().fillna(0.0)
    df["vol_20"]=ret.rolling(20,min_periods=5).std().fillna(method="bfill").fillna(0.0)
    bb=BollingerBands(close=df["close"],window=20,window_dev=2,fillna=True)
    df["bb_width"]=(bb.bollinger_hband()-bb.bollinger_lband())/df["close"].replace(0,np.nan)
    df["bb_width"]=df["bb_width"].replace([np.inf,-np.inf],np.nan).fillna(method="bfill").fillna(0.0)

    for k in (1,2,3): df[f"lag_{k}"]=df["close"].shift(k)

    df["date_only"]=df["date"].dt.floor("D")
    if not df_sent.empty:
        ds=df_sent.copy(); ds["date_only"]=pd.to_datetime(ds["date"]).dt.floor("D")
        df=df.merge(ds[["date_only","sentiment"]],on="date_only",how="left")
    else:
        df["sentiment"]=0.0
    df["sentiment"]=df["sentiment"].fillna(0.0)

    df=df.dropna().reset_index(drop=True)
    df["target"]=df["close"].shift(-1)
    df=df.dropna().reset_index(drop=True)

    feat_cols=["open","high","low","close","volume",
               "sma_10","rsi_14","macd","macd_signal","macd_hist",
               "atr_14","vol_20","bb_width",
               "lag_1","lag_2","lag_3","sentiment"]
    X=df[feat_cols].values.astype(float)
    y=df["target"].values.astype(float)
    return df,X,y,feat_cols

def build_sequences(X,y,seq_len):
    X_seq,y_seq=[],[]
    for i in range(seq_len-1,len(X)):
        X_seq.append(X[i-seq_len+1:i+1,:]); y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def build_lstm(input_dim,seq_len,units=64,lr=1e-3):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM,Dense,Dropout
    from tensorflow.keras.optimizers import Adam
    m=Sequential([LSTM(units,return_sequences=False,input_shape=(seq_len,input_dim)),
                  Dropout(0.2), Dense(1)])
    m.compile(optimizer=Adam(lr),loss="mse"); return m

def smooth_series(arr, window=3):
    if window<=1: return np.array(arr, dtype=float)
    s=pd.Series(arr, dtype=float)
    return s.rolling(window, min_periods=1).mean().values

# -------- train/eval --------
def train_and_eval(X,y,seq_len,epochs,return_models=False):
    n=len(X); n_tr=int(n*0.6); n_va=int(n*0.2)
    Xtr,ytr = X[:n_tr],y[:n_tr]
    Xva,yva = X[n_tr:n_tr+n_va],y[n_tr:n_tr+n_va]
    Xte,yte = X[n_tr+n_va:],y[n_tr+n_va:]

    xs,ys=StandardScaler(),MinMaxScaler()
    Xtr_s,Xva_s,Xte_s=xs.fit_transform(Xtr),xs.transform(Xva),xs.transform(Xte)
    ytr_s,yva_s=ys.fit_transform(ytr.reshape(-1,1)).ravel(),ys.transform(yva.reshape(-1,1)).ravel()

    Xtr_seq,ytr_s_seq=build_sequences(Xtr_s,ytr_s,seq_len)
    Xva_seq,yva_s_seq=build_sequences(Xva_s,yva_s,seq_len)
    yte_s_full=ys.transform(yte.reshape(-1,1)).ravel()
    Xte_seq,yte_s_seq=build_sequences(Xte_s,yte_s_full,seq_len)
    yte_usd=yte[(seq_len-1):]

    lstm=build_lstm(Xtr_s.shape[1],seq_len,units=64)
    lstm.fit(Xtr_seq,ytr_s_seq,validation_data=(Xva_seq,yva_s_seq),epochs=epochs,batch_size=32,verbose=0)

    p_tr=lstm.predict(Xtr_seq,verbose=0).ravel(); p_va=lstm.predict(Xva_seq,verbose=0).ravel()
    Xtr_stack=np.hstack([Xtr_seq[:,-1,:], p_tr.reshape(-1,1)])
    Xva_stack=np.hstack([Xva_seq[:,-1,:], p_va.reshape(-1,1)])
    xgbm=xgb.XGBRegressor(n_estimators=350,max_depth=5,learning_rate=0.05,
                          subsample=0.9,colsample_bytree=0.9,random_state=42,n_jobs=4)
    xgbm.fit(Xtr_stack,ytr_s_seq,eval_set=[(Xva_stack,yva_s_seq)],verbose=False)

    p_te=lstm.predict(Xte_seq,verbose=0).ravel()
    Xte_stack=np.hstack([Xte_seq[:,-1,:], p_te.reshape(-1,1)])
    ypred_h=ys.inverse_transform(xgbm.predict(Xte_stack).reshape(-1,1)).ravel()

    metrics=compute_metrics(yte_usd, ypred_h)

    if not return_models:
        return metrics, yte_usd, ypred_h
    else:
        return metrics, yte_usd, ypred_h, lstm, xgbm, xs, ys, Xte_seq, X, y

def auto_tune(X,y):
    configs=[(5,30),(5,50),(10,30),(10,50)]
    best=None; best_m=float("inf"); best_hist=None
    for seq,ep in configs:
        try:
            m, y_true, y_pred = train_and_eval(X,y,seq,ep,return_models=False)
            if m["RMSE"]<best_m: best,best_m,best_hist=(seq,ep),m["RMSE"],(m,y_true,y_pred)
        except Exception:
            continue
    return best,best_hist

# -------- future forecast (recursive, fixed shapes + proper scaling for 'close') --------
def forecast_future(lstm, xgbm, last_seq, last_feature_row_scaled, scaler_y,
                    feat_cols, xs_mean, xs_scale, horizon=7, close_name="close"):
    """
    last_seq:          shape (1, seq_len, F) in feature-scaled space
    last_feature_row_scaled: shape (F,)    in feature-scaled space
    """
    preds=[]
    seq = last_seq.copy()                             # (1, L, F)
    close_idx = feat_cols.index(close_name)

    for _ in range(horizon):
        # LSTM -> scaled y
        lstm_pred_s = lstm.predict(seq, verbose=0).ravel()[-1]
        # Hybrid input must be 2D; fix shape with [[...]]
        xgb_in = np.hstack([seq[:, -1, :], np.array([[lstm_pred_s]])])
        xgb_pred_s = xgbm.predict(xgb_in).ravel()[0]
        pred_usd = scaler_y.inverse_transform([[xgb_pred_s]])[0,0]
        preds.append(pred_usd)

        # Map predicted USD 'close' into feature-scaled space using feature scaler stats
        scaled_close = (pred_usd - xs_mean[close_idx]) / (xs_scale[close_idx] if xs_scale[close_idx] != 0 else 1.0)

        new_features = last_feature_row_scaled.copy()
        new_features[close_idx] = scaled_close

        # advance sequence window
        new_step = new_features.reshape(1,1,-1)
        seq = np.concatenate([seq[:,1:,:], new_step], axis=1)
        last_feature_row_scaled = new_features

    return np.array(preds, dtype=float)

# -------------- STREAMLIT UI --------------
st.set_page_config(page_title="Hybrid Stock Forecast (Auto-Tuned)", layout="wide")
st.title("📈 Stock Price Forecasting with Technical Indicators and News Sentiment")

with st.sidebar:
    ticker=st.text_input("Ticker",value="AAPL").upper().strip()
    default_end=datetime.utcnow().date(); default_start=default_end - timedelta(days=730)
    start_date=st.date_input("Start date",value=default_start)
    end_date=st.date_input("End date",value=default_end)
    future_days=st.slider("Forecast horizon (days)",1,15,7)
    smooth_win=st.slider("Smooth forecast (MA window)",1,7,3)
    run_btn=st.button("Run Auto-Tuned Model")

if run_btn:
    sd,ed=start_date.strftime("%Y-%m-%d"),end_date.strftime("%Y-%m-%d")

    with st.spinner("Fetching price data..."):
        df_px=fetch_prices(ticker,sd,ed)
    if df_px.empty:
        st.error("No data returned. Check ticker/dates or API limits.")
        st.stop()

    with st.spinner("Fetching news (capped at 100)…"):
        df_news=fetch_news(ticker,sd,ed,100)

    with st.spinner("Scoring sentiment (FinBERT)…"):
        df_sent=finbert_daily_sentiment(df_news)

    # ---- Sentiment tables ----
    with st.expander("📰 Recent headlines (sample)"):
        if not df_news.empty: st.dataframe(df_news.tail(15), use_container_width=True)
        else: st.info("No news fetched.")
    with st.expander("😊 Daily sentiment (aggregated)"):
        if not df_sent.empty: st.dataframe(df_sent.tail(20), use_container_width=True)
        else: st.info("No sentiment available.")

    with st.spinner("Engineering features…"):
        df,X,y,feat_cols=make_features(df_px,df_sent)

    st.caption(f"Samples available: **{len(df)}** | Features: `{feat_cols}`")
    if len(df)<200: st.warning("Small dataset; metrics may be unstable.")

    with st.spinner("Auto-tuning hyperparameters…"):
        best,results=auto_tune(X,y)
    if not results:
        st.error("Model failed during tuning.")
        st.stop()

    seq_len, epochs = best
    with st.spinner(f"Retraining with best config (seq_len={seq_len}, epochs={epochs})…"):
        m2, y_true2, y_pred2, lstm, xgbm, xs, ys, Xte_seq, X_all, y_all = train_and_eval(
            X, y, seq_len, epochs, return_models=True
        )

    # ---- Metrics (backtest) ----
    st.subheader("📊 Best Config")
    st.write(f"Seq_len={seq_len}, Epochs={epochs}")

    st.subheader("📈 Test Metrics (Hybrid)")
    st.json({k:round(v,4) for k,v in m2.items()})

    # backtest chart
    dates=df.iloc[-len(y_true2):]["date"].values
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=y_true2,mode="lines",name="True"))
    fig.add_trace(go.Scatter(x=dates,y=y_pred2,mode="lines",name="Pred (Hybrid)"))
    fig.update_layout(title="Test: True vs Predicted (Next-Day Close)", yaxis_title="USD")
    st.plotly_chart(fig,use_container_width=True)

    # ---- Future Forecast (fixed shapes + correct scaling) ----
    st.subheader("🔮 Future Forecast")
    last_seq = Xte_seq[-1:].copy()                       # (1, L, F) in scaled space
    # scale the last raw feature row to match sequence space
    last_row_scaled = xs.transform(X_all[-1].reshape(1,-1))[0]
    future_preds = forecast_future(
        lstm, xgbm, last_seq, last_row_scaled, ys,
        feat_cols, xs.mean_, xs.scale_, horizon=future_days, close_name="close"
    )
    smoothed = smooth_series(future_preds, window=smooth_win)

    df_future = pd.DataFrame({
        "Day": np.arange(1, future_days+1),
        "Predicted_Close": np.round(future_preds, 2),
        "Pred_Close_Smoothed": np.round(smoothed, 2)
    })
    st.dataframe(df_future, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_future["Day"], y=df_future["Predicted_Close"],
                              mode="lines+markers", name="Forecast"))
    if smooth_win > 1:
        fig2.add_trace(go.Scatter(x=df_future["Day"], y=df_future["Pred_Close_Smoothed"],
                                  mode="lines+markers", name=f"Smoothed (MA {smooth_win})"))
    fig2.update_layout(title=f"{ticker} — {future_days}-Day Forecast (Hybrid)",
                       xaxis_title="Day", yaxis_title="USD")
    st.plotly_chart(fig2, use_container_width=True)

    st.success("Done.")
