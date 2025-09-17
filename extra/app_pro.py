# app_tuned.py
import os, math, time, json, random, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser

import streamlit as st
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

# optional sentiment
from transformers import pipeline
from transformers.pipelines.base import PipelineException

# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or "YOUR_FALLBACK_KEY"
BASE = "https://api.polygon.io"

# ------------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------------
def set_global_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        import xgboost as xgb
        xgb.set_config(verbosity=0)  # keeps logs quiet
    except Exception:
        pass

# ------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------
def to_utc_datestr(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")

def from_unix_ms(ms): return datetime.utcfromtimestamp(ms / 1000.0)

def metrics_dict(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2}

def print_terminal_metrics(name, d):
    print(f"\n=== {name} ===")
    for k,v in d.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

# ------------------------------------------------------------------------------------
# Data fetch (Polygon)
# ------------------------------------------------------------------------------------
def fetch_agg_daily(ticker: str, start_date: str, end_date: str):
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    res = js.get("results", [])
    if not res:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    df = pd.DataFrame(res)
    df["date"] = pd.to_datetime(df["t"].apply(from_unix_ms))
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["date","open","high","low","close","volume"]].sort_values("date").reset_index(drop=True)

def fetch_news(ticker: str, start_date: str, end_date: str, max_articles=400):
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
    if not got:
        return pd.DataFrame(columns=["date","headline","summary"])
    rows=[]
    for it in got[:max_articles]:
        pu = it.get("published_utc"); 
        if not pu: continue
        try: d = dateparser.parse(pu).date()
        except Exception: continue
        rows.append({"date": pd.Timestamp(d), "headline": it.get("title") or "", "summary": it.get("description") or ""})
    df = pd.DataFrame(rows); df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

# ------------------------------------------------------------------------------------
# Sentiment
# ------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_finbert():
    try:
        return pipeline("text-classification", model="ProsusAI/finbert")
    except Exception:
        return None

def finbert_daily_sentiment(df_news, finbert):
    if df_news.empty or finbert is None:
        return pd.DataFrame({"date": [], "sentiment": []})
    texts=(df_news["headline"].fillna("")+". "+df_news["summary"].fillna("")).tolist()
    labels=[]
    for i in range(0, len(texts), 16):
        chunk = texts[i:i+16]
        try:
            preds = finbert(chunk, truncation=True)
        except PipelineException:
            preds = [{"label":"NEUTRAL"} for _ in chunk]
        for p in preds:
            lab = (p.get("label") or "NEUTRAL").upper()
            labels.append(1.0 if "POS" in lab else -1.0 if "NEG" in lab else 0.0)
    df = df_news.copy(); df["score"] = labels[:len(df)]
    daily = df.groupby("date", as_index=False)["score"].mean().rename(columns={"score":"sentiment"})
    return daily

# ------------------------------------------------------------------------------------
# Features
# ------------------------------------------------------------------------------------
def make_features(df_px: pd.DataFrame, df_sent: pd.DataFrame):
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

def build_sequences(X_2d, y_vec, seq_len):
    X_seq, y_out = [], []
    for i in range(seq_len-1, len(X_2d)):
        X_seq.append(X_2d[i-seq_len+1:i+1, :])
        y_out.append(y_vec[i])
    return np.array(X_seq), np.array(y_out)

# ------------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------------
def build_lstm(input_dim, seq_len, units=64, lr=1e-3, dropout=0.2):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    m = Sequential([
        LSTM(units, input_shape=(seq_len, input_dim)),
        Dropout(dropout),
        Dense(1)
    ])
    m.compile(optimizer=Adam(lr), loss="mse")
    return m

def train_lstm_stack_xgb(X_tr, y_tr, X_va, y_va, seq_len, units, epochs, batch, lr):
    # scale (train-only fit)
    xs, ys = StandardScaler(), MinMaxScaler()
    Xtr_s, Xva_s = xs.fit_transform(X_tr), xs.transform(X_va)
    ytr_s, yva_s = ys.fit_transform(y_tr.reshape(-1,1)).ravel(), ys.transform(y_va.reshape(-1,1)).ravel()

    # sequences
    Xtr_seq, ytr_s_seq = build_sequences(Xtr_s, ytr_s, seq_len)
    Xva_seq, yva_s_seq = build_sequences(Xva_s, yva_s, seq_len)

    # LSTM
    from tensorflow.keras.callbacks import EarlyStopping
    lstm = build_lstm(Xtr_s.shape[1], seq_len, units=units, lr=lr)
    es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    lstm.fit(Xtr_seq, ytr_s_seq, validation_data=(Xva_seq, yva_s_seq),
             epochs=epochs, batch_size=batch, callbacks=[es], verbose=0)

    # stacked XGB
    import xgboost as xgb
    p_tr = lstm.predict(Xtr_seq, verbose=0).ravel()
    p_va = lstm.predict(Xva_seq, verbose=0).ravel()
    Xtr_last = Xtr_seq[:, -1, :]
    Xva_last = Xva_seq[:, -1, :]
    Xtr_stack = np.hstack([Xtr_last, p_tr.reshape(-1,1)])
    Xva_stack = np.hstack([Xva_last, p_va.reshape(-1,1)])

    xgbm = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=4
    )
    xgbm.fit(Xtr_stack, ytr_s_seq, eval_set=[(Xva_stack, yva_s_seq)], verbose=False)

    pack = {"lstm": lstm, "xgb": xgbm, "x_scaler": xs, "y_scaler": ys, "seq_len": seq_len}
    return pack

def predict_hybrid(pack, X):
    xs, ys = pack["x_scaler"], pack["y_scaler"]
    seq_len = pack["seq_len"]
    Xs = xs.transform(X)
    # Build rolling sequences
    X_seq, _ = build_sequences(Xs, np.zeros(len(Xs)), seq_len)  # dummy y
    p_l = pack["lstm"].predict(X_seq, verbose=0).ravel()
    X_last = X_seq[:, -1, :]
    X_stack = np.hstack([X_last, p_l.reshape(-1,1)])
    p_s = pack["xgb"].predict(X_stack)
    p = ys.inverse_transform(p_s.reshape(-1,1)).ravel()
    return p

# ------------------------------------------------------------------------------------
# CV tuning (per ticker)
# ------------------------------------------------------------------------------------
def cv_search(X, y, seq_grid=(5,10), unit_grid=(32,64), depth_grid=(4,5), n_splits=5,
              epochs=40, batch=32, lr=1e-3, use_progress=False):
    """
    TimeSeriesSplit CV. Returns best config + fold metrics table.
    """
    # split once at top level (same folds reused for configs)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    configs = []
    for sl in seq_grid:
        for un in unit_grid:
            for md in depth_grid:
                configs.append({"seq_len": sl, "units": un, "max_depth": md})

    results = []
    prog = st.progress(0) if use_progress else None
    done = 0; total = len(configs)*n_splits

    for cfg in configs:
        fold_metrics = []
        for fold, (tr, va) in enumerate(tscv.split(X, y), 1):
            Xtr, ytr = X[tr], y[tr]
            Xva, yva = X[va], y[va]
            pack = train_lstm_stack_xgb(
                Xtr, ytr, Xva, yva,
                seq_len=cfg["seq_len"], units=cfg["units"],
                epochs=epochs, batch=batch, lr=lr
            )
            # Predict on validation (align sequences)
            xs = pack["x_scaler"]; ys = pack["y_scaler"]; L = cfg["seq_len"]
            Xva_s = xs.transform(Xva)
            Xva_seq, _ = build_sequences(Xva_s, np.zeros(len(Xva_s)), L)
            yva_s_full = ys.transform(yva.reshape(-1,1)).ravel()
            yva_usd_seq = yva[(L-1):]
            p_val = predict_hybrid(pack, Xva)  # returns aligned preds
            p_val = p_val  # already USD
            m = metrics_dict(yva_usd_seq, p_val)
            fold_metrics.append(m)
            done += 1
            if prog: prog.progress(min(1.0, done/total))
        # aggregate
        mean = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0].keys()}
        std  = {k+"_std": float(np.std([fm[k] for fm in fold_metrics])) for k in fold_metrics[0].keys()}
        results.append({**cfg, **mean, **std})

    # pick best by lowest RMSE (you can switch to highest R2)
    best = sorted(results, key=lambda r: r["RMSE"])[0]
    return best, results

# ------------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="Hybrid (CV-tuned per ticker)", layout="wide")
st.title("📊 Hybrid Stock Forecast — Per-Ticker CV Tuning (LSTM → XGBoost)")

with st.sidebar:
    st.subheader("Settings")

    preset = st.selectbox("Choose a ticker", ["AAPL", "MSFT", "NVDA", "TSLA"], index=0)
    custom = st.text_input("…or type another ticker", value="").upper().strip()
    ticker = (custom or preset).upper()

    # default to last 2 years
    default_end = datetime.utcnow().date()
    default_start = default_end - timedelta(days=730)
    sd = st.date_input("Start date", value=default_start)
    ed = st.date_input("End date", value=default_end)

    use_sent = st.checkbox("Use FinBERT sentiment", value=True)
    tune_sent = st.checkbox("Use sentiment during tuning (slower)", value=False)

    st.markdown("---")
    st.subheader("Auto-tune")
    do_tune = st.checkbox("Auto-tune per ticker (TimeSeriesSplit CV)", value=True)
    n_splits = st.slider("CV folds", 3, 8, 5)
    epochs = st.slider("LSTM epochs (per fold)", 10, 80, 40, step=5)
    seq_grid = st.multiselect("Seq window grid", [3,5,7,10,12], default=[5,10])
    unit_grid = st.multiselect("LSTM units grid", [32,64,96], default=[32,64])
    depth_grid = st.multiselect("XGB max_depth grid", [3,4,5,6], default=[4,5])

    st.markdown("---")
    st.subheader("Quick run (no tuning)")
    quick_seq = st.slider("Seq window", 3, 15, 5)
    quick_units = st.selectbox("LSTM units", [32,64,96], index=1)

    st.markdown("---")
    go_btn = st.button("Run")

st.caption(f"Polygon API key source: {'env var' if os.getenv('POLYGON_API_KEY') else 'inline fallback'}")

if go_btn:
    try:
        set_global_seeds(42)
        start_date = to_utc_datestr(datetime.combine(sd, datetime.min.time()))
        end_date   = to_utc_datestr(datetime.combine(ed, datetime.min.time()))

        with st.spinner("Fetching prices…"):
            df_px = fetch_agg_daily(ticker, start_date, end_date)
        if df_px.empty:
            st.error("No price data returned. Check ticker/dates or API plan limits.")
            st.stop()

        if use_sent or tune_sent:
            with st.spinner("Fetching news…"):
                df_news = fetch_news(ticker, start_date, end_date, max_articles=400)
            finbert = load_finbert() if (use_sent or tune_sent) else None
            if tune_sent:
                df_sent_tune = finbert_daily_sentiment(df_news, finbert)
            else:
                df_sent_tune = pd.DataFrame({"date": [], "sentiment": []})
            df_sent_final = finbert_daily_sentiment(df_news, finbert) if use_sent else pd.DataFrame({"date": [], "sentiment": []})
        else:
            df_sent_tune = df_sent_final = pd.DataFrame({"date": [], "sentiment": []})

        # Features
        df_all, X_all, y_all, feat_cols = make_features(df_px, df_sent_final if do_tune else df_sent_final)
        n = len(X_all)
        if n < 120:
            st.warning(f"Only {n} samples after features. Consider longer date range.")
        # Chronological 60/20/20 split for final reporting
        n_tr = int(n*0.6); n_va = int(n*0.2)
        Xtr, ytr = X_all[:n_tr], y_all[:n_tr]
        Xva, yva = X_all[n_tr:n_tr+n_va], y_all[n_tr:n_tr+n_va]
        Xte, yte = X_all[n_tr+n_va:], y_all[n_tr+n_va:]

        st.write(f"Split → Train: {len(Xtr)}, Val: {len(Xva)}, Test: {len(Xte)}")

        # --------------------------------------------
        # CV tuning (per ticker)
        # --------------------------------------------
        best_cfg = None
        if do_tune:
            # Use sentiment in tuning or not
            df_all_tune, X_all_tune, y_all_tune, _ = make_features(df_px, df_sent_tune)
            with st.spinner("Running CV search (per ticker)…"):
                best_cfg, grid_tab = cv_search(
                    X_all_tune, y_all_tune,
                    seq_grid=seq_grid or [5,10],
                    unit_grid=unit_grid or [32,64],
                    depth_grid=depth_grid or [4,5],
                    n_splits=n_splits, epochs=epochs, batch=32, lr=1e-3,
                    use_progress=True
                )
            st.success("CV complete.")
            st.subheader("Best config (by RMSE)")
            st.json(best_cfg)
            with st.expander("All CV results"):
                st.dataframe(pd.DataFrame(grid_tab))

        # --------------------------------------------
        # Final train on Train+Val with best (or quick) config, evaluate on Test
        # --------------------------------------------
        cfg = best_cfg or {"seq_len": int(quick_seq), "units": int(quick_units)}
        st.info(f"Using config → seq_len={cfg['seq_len']}, units={cfg['units']}")

        # Train on Train only, validate on Val (to mimic baseline), then test
        pack = train_lstm_stack_xgb(
            Xtr, ytr, Xva, yva,
            seq_len=int(cfg["seq_len"]),
            units=int(cfg["units"]),
            epochs=epochs if do_tune else 40,
            batch=32, lr=1e-3
        )

        # Align predictions with test (drop first seq_len-1 points)
        L = int(cfg["seq_len"])
        yte_usd_seq = yte[(L-1):]
        ypred_h = predict_hybrid(pack, Xte)
        # LSTM-alone for comparison
        xs, ys = pack["x_scaler"], pack["y_scaler"]
        Xte_s = xs.transform(Xte)
        Xte_seq, _ = build_sequences(Xte_s, np.zeros(len(Xte_s)), L)
        from tensorflow.keras.models import Model
        ypred_lstm_s = pack["lstm"].predict(Xte_seq, verbose=0).ravel()
        ypred_lstm = ys.inverse_transform(ypred_lstm_s.reshape(-1,1)).ravel()

        m_lstm  = metrics_dict(yte_usd_seq, ypred_lstm)
        m_hyb   = metrics_dict(yte_usd_seq, ypred_h)

        # Show metrics
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("📊 Test Metrics (USD) — LSTM-only")
            st.json({k: round(v,4) for k,v in m_lstm.items()})
        with c2:
            st.subheader("📊 Test Metrics (USD) — Hybrid (LSTM→XGB)")
            st.json({k: round(v,4) for k,v in m_hyb.items()})

        # Print to terminal too (for screenshots / resume)
        print_terminal_metrics(f"{ticker} — LSTM-only (Test)", m_lstm)
        print_terminal_metrics(f"{ticker} — Hybrid (Test)", m_hyb)

        # Plot test True vs Pred
        test_idx = df_all.iloc[n_tr+n_va:].index[(L-1):]
        df_plot = pd.DataFrame({
            "date": df_all.loc[test_idx, "date"].values,
            "True": yte_usd_seq,
            "LSTM": ypred_lstm,
            "Hybrid": ypred_h
        })
        fig = go.Figure()
        for col in ["True","LSTM","Hybrid"]:
            fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot[col], mode="lines", name=col))
        fig.update_layout(title="🧪 Test: True vs Predicted (Next-Day Close)", xaxis_title="Date", yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)

        # Optional: forecast next N days (naive roll) — reuse your previous logic if you want

        # Recent headlines preview
        with st.expander("Recent headlines (sample)"):
            try:
                df_news_show = fetch_news(ticker, start_date, end_date, 100)
                st.dataframe(df_news_show.tail(15), use_container_width=True)
            except Exception:
                st.info("News preview unavailable.")

    except requests.HTTPError as e:
        st.error(f"HTTP error from Polygon: {e}")
    except Exception as e:
        st.exception(e)
