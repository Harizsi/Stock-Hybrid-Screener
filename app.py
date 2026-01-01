import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Enhanced Stock Predictor", layout="wide")

# =========================
# Technical Indicators
# =========================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series, period=20, std_dev=2):
    mid = sma(series, period)
    std = series.rolling(period).std()
    return mid + std * std_dev, mid, mid - std * std_dev

def atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = np.abs(df["High"] - df["Close"].shift())
    lc = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def obv(df):
    return (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

# =========================
# Feature Engineering
# =========================

def add_features(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    df["EMA5"] = ema(df["Close"], 5)
    df["EMA10"] = ema(df["Close"], 10)
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["RSI"] = rsi(df["Close"])
    df["MACD"], _, df["MACD_hist"] = macd(df["Close"])

    bb_u, bb_m, bb_l = bollinger_bands(df["Close"])
    df["BB_position"] = (df["Close"] - bb_l) / (bb_u - bb_l)

    close_col = df["Close"]

    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]

    # df["ATR_percent"] = df["ATR"] / close_col

    df["ATR"] = atr(df)
    df["ATR_pct"] = df["ATR"] / df["Close"]

    df["Volume_MA"] = df["Volume"].rolling(20).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_MA"]

    df["OBV"] = obv(df)
    df["OBV_EMA"] = ema(df["OBV"], 20)

    df["ret_1"] = df["Close"].pct_change()
    df["volatility"] = df["ret_1"].rolling(10).std()

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df

FEATURES = [
    "EMA20","EMA50","RSI","MACD","MACD_hist",
    "BB_position","ATR_pct","Volume_ratio",
    "OBV_EMA","ret_1","volatility"
]

# =========================
# Model
# =========================

def train_model(df, model_type):
    df = df.dropna()
    if len(df) < 200:
        return None, None

    X = df[FEATURES]
    y = df["target"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    clf = (
        GradientBoostingClassifier(random_state=42)
        if model_type == "gradient_boosting"
        else RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:,1]

    return model, {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "auc": roc_auc_score(y_test, prob)
    }

# =========================
# UI
# =========================

st.title("📈 Enhanced Stock Prediction System")

tab1, tab2 = st.tabs(["Single Stock", "Multi Screener"])


# ---------- TAB 1 ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        ticker = st.text_input("Ticker", "AAPL")

    with c2:
        period = st.selectbox("History Period", ["6mo","1y","2y","5y"])

    with c3:
        interval = st.selectbox(
            "Time Frame",
            ["1d","1h","30m","15m","5m"]
        )

    with c4:
        model_type = st.selectbox(
            "Model",
            ["random_forest","gradient_boosting"]
        )

    if st.button("Analyze", type="primary"):
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False
        )
        
        if df.empty:
            st.error("No data")
            st.stop()

        df = add_features(df)
        model, metrics = train_model(df, model_type)

        if model is None:
            st.error("Not enough data")
            st.stop()

        last_X = df[FEATURES].iloc[[-1]]
        base_prob = model.predict_proba(last_X)[0][1]

        # ---- Candlestick Chart ----
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"]
            )
        ])
        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Model Metrics")
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        st.metric("AUC", f"{metrics['auc']:.2f}")

        # ---- Next 5 Candle Projection ----
        st.subheader("🔮 Next 5 Candle Direction (Probabilistic)")

        projections = []
        for i in range(1, 6):
            decay = 0.85 ** (i - 1)
            prob = base_prob * decay + 0.5 * (1 - decay)
            direction = "UP" if prob >= 0.5 else "DOWN"

            projections.append({
                "Candle +": i,
                "Direction": direction,
                "Confidence": f"{prob:.1%}"
            })

        st.dataframe(pd.DataFrame(projections), use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    tickers = st.text_area(
        "Tickers (comma separated)",
        "AAPL,MSFT,GOOGL,AMZN,TSLA"
    )

    period_multi = st.selectbox("Period", ["6mo","1y","2y"])
    interval_multi = st.selectbox("Time Frame", ["1d","1h","30m"])
    model = st.selectbox("Models", ["random_forest","gradient_boosting"])

    if st.button("Run Screener", type="primary"):
        rows = []

        for t in [x.strip() for x in tickers.split(",") if x.strip()]:
            df = yf.download(t, period=period_multi, interval=interval_multi, progress=False)
            if df.empty:
                continue

            df = add_features(df)
            model, metrics = train_model(df, model)
            if model:
                prob = model.predict_proba(df[FEATURES].iloc[[-1]])[0][1]
                rows.append({
                    "Ticker": t,
                    "Next Candle Prob ↑": f"{prob:.1%}",
                    "Accuracy": f"{metrics['accuracy']:.1%}",
                    "AUC": f"{metrics['auc']:.2f}"
                })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)


### 🎓 About This System
st.markdown("""
**Accuracy Tips:**
- Use 2+ years of data
- Random Forest usually best
- Look for AUC > 0.60
- Compare to buy-and-hold
""")
