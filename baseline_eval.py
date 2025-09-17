# ==== baseline_eval.py (shallow 1-step LSTM + XGB, early stopping) ====
import os, time, math, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from transformers import pipeline
import warnings; warnings.filterwarnings("ignore")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or "YOUR_FALLBACK_KEY"
BASE = "https://api.polygon.io"

# ---------------- utils ----------------
def from_unix_ms(ms): return datetime.utcfromtimestamp(ms / 1000.0)
def compute_metrics(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse); mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== Test Metrics — {name} ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R^2 : {r2:.4f}")

# -------------- data fetch --------------
def fetch_prices(ticker, sd, ed):
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/1/day/{sd}/{ed}"
    params = {"adjusted":"true","sort":"asc","limit":50000,"apiKey":POLYGON_API_KEY}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    res = r.json().get("results", [])
    if not res: return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    df = pd.DataFrame(res)
    df["date"] = pd.to_datetime(df["t"].apply(from_unix_ms))
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["date","open","high","low","close","volume"]].sort_values("date").reset_index(drop=True)

def fetch_news(ticker, start_date, end_date, max_articles=200):
    url = f"{BASE}/v2/reference/news"
    params = {
        "ticker": ticker, "order": "asc", "sort": "published_utc",
        "limit": 50, "published_utc.gte": start_date,
        "published_utc.lte": end_date, "apiKey": POLYGON_API_KEY
    }
    got, tries = [], 0
    while len(got) < max_articles and url:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            wait = min(30, 2 ** min(tries, 4)); time.sleep(wait); tries += 1; continue
        r.raise_for_status()
        js = r.json(); results = js.get("results", [])
        if not results: break
        got.extend(results)
        next_url = js.get("next_url")
        if next_url and len(got) < max_articles:
            if "apiKey=" not in next_url:
                sep = "&" if "?" in next_url else "?"
                next_url = f"{next_url}{sep}apiKey={POLYGON_API_KEY}"
            url, params = next_url, None
            time.sleep(0.1)
        else:
            url = None
    if not got: return pd.DataFrame(columns=["date","headline","summary"])
    rows=[]
    for it in got[:max_articles]:
        pu = it.get("published_utc"); 
        if not pu: continue
        try: d = dateparser.parse(pu).date()
        except Exception: continue
        rows.append({"date": pd.Timestamp(d), "headline": it.get("title") or "", "summary": it.get("description") or ""})
    df = pd.DataFrame(rows); df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

# -------------- sentiment ---------------
def finbert_daily_sentiment(df_news):
    if df_news.empty: return pd.DataFrame({"date": [], "sentiment": []})
    clf = pipeline("text-classification", model="ProsusAI/finbert")
    texts = (df_news["headline"].fillna("") + ". " + df_news["summary"].fillna("")).tolist()
    labels=[]
    for i in range(0,len(texts),16):
        preds = clf(texts[i:i+16], truncation=True)
        for p in preds:
            lab = (p.get("label") or "NEUTRAL").upper()
            labels.append(1.0 if "POS" in lab else -1.0 if "NEG" in lab else 0.0)
    df = df_news.copy(); df["score"] = labels[:len(df)]
    return df.groupby("date", as_index=False)["score"].mean().rename(columns={"score":"sentiment"})

# -------------- features ----------------
def make_features(df_px, df_sent):
    df = df_px.copy()
    df["sma_10"] = SMAIndicator(df["close"], 10, fillna=True).sma_indicator()
    df["rsi_14"] = RSIIndicator(df["close"], 14, fillna=True).rsi()
    macd = MACD(df["close"], 26, 12, 9, fillna=True)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
    for k in (1,2,3): df[f"lag_{k}"] = df["close"].shift(k)
    df["date_only"] = df["date"].dt.floor("D")
    if not df_sent.empty:
        ds = df_sent.copy(); ds["date_only"] = pd.to_datetime(ds["date"]).dt.floor("D")
        df = df.merge(ds[["date_only","sentiment"]], on="date_only", how="left")
    else:
        df["sentiment"] = 0.0
    df["sentiment"] = df["sentiment"].fillna(0.0)
    df = df.dropna().reset_index(drop=True)
    df["target"] = df["close"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    feat_cols = ["open","high","low","close","volume",
                 "sma_10","rsi_14","macd","macd_signal","macd_hist",
                 "lag_1","lag_2","lag_3","sentiment"]
    X = df[feat_cols].values.astype(float)
    y = df["target"].values.astype(float)
    return df, X, y, feat_cols

# --------------- models ------------------
def build_lstm_shallow(input_dim, lr=1e-3, units=96):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    m = Sequential([
        LSTM(units, input_shape=(1, input_dim), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    m.compile(optimizer=Adam(lr), loss="mse")
    return m

# ================== MAIN =================
if __name__ == "__main__":
    TICKER = "AAPL"    # fixed baseline ticker
    EPOCHS = 40
    BATCH  = 32

    end = datetime.utcnow().date()
    start = end - timedelta(days=730)  # ~2 years
    sd, ed = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    print(f"Baseline (shallow 1-step LSTM) on {TICKER} | {sd} → {ed}")
    df_px = fetch_prices(TICKER, sd, ed)
    if df_px.empty: raise SystemExit("No price data returned.")

    df_news = fetch_news(TICKER, sd, ed, max_articles=200)
    df_sent = finbert_daily_sentiment(df_news) if not df_news.empty else pd.DataFrame({"date": [], "sentiment": []})

    df, X, y, feat_cols = make_features(df_px, df_sent)
    n = len(X); n_tr = int(n*0.6); n_va = int(n*0.2)
    # chronological split
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xva, yva = X[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va]
    Xte, yte = X[n_tr+n_va:], y[n_tr+n_va:]

    # scale on train only
    xsc, ysc = StandardScaler(), MinMaxScaler()
    Xtr_s, Xva_s, Xte_s = xsc.fit_transform(Xtr), xsc.transform(Xva), xsc.transform(Xte)
    ytr_s, yva_s        = ysc.fit_transform(ytr.reshape(-1,1)).ravel(), ysc.transform(yva.reshape(-1,1)).ravel()

    # LSTM expects (samples, timesteps=1, features)
    Xtr_seq = Xtr_s.reshape(-1, 1, Xtr_s.shape[1])
    Xva_seq = Xva_s.reshape(-1, 1, Xva_s.shape[1])
    Xte_seq = Xte_s.reshape(-1, 1, Xte_s.shape[1])

    # LSTM (shallow) with early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    lstm = build_lstm_shallow(input_dim=Xtr_s.shape[1], lr=1e-3, units=64)
    es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    lstm.fit(Xtr_seq, ytr_s, validation_data=(Xva_seq, yva_s),
             epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=0)

    # LSTM-only predictions (USD)
    ypred_lstm_s = lstm.predict(Xte_seq, verbose=0).ravel()
    ypred_lstm   = ysc.inverse_transform(ypred_lstm_s.reshape(-1,1)).ravel()
    compute_metrics(yte, ypred_lstm, "LSTM-only (baseline)")

    # ---------- Hybrid (LSTM → XGB) ----------
    import xgboost as xgb
    p_tr_s = lstm.predict(Xtr_seq, verbose=0).ravel()
    p_va_s = lstm.predict(Xva_seq, verbose=0).ravel()
    Xtr_stack = np.hstack([Xtr_s, p_tr_s.reshape(-1,1)])
    Xva_stack = np.hstack([Xva_s, p_va_s.reshape(-1,1)])

    xgbm = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=4
    )
    xgbm.fit(Xtr_stack, ytr_s, eval_set=[(Xva_stack, yva_s)], verbose=False)

    p_te_s = lstm.predict(Xte_seq, verbose=0).ravel()
    Xte_stack = np.hstack([Xte_s, p_te_s.reshape(-1,1)])
    ypred_hybrid_s = xgbm.predict(Xte_stack)
    ypred_hybrid   = ysc.inverse_transform(ypred_hybrid_s.reshape(-1,1)).ravel()
    compute_metrics(yte, ypred_hybrid, "Hybrid (LSTM→XGBoost) baseline")

    print(">>> DONE baseline_eval.py")
