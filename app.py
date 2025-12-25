import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Create two tabs
tab1, tab2 = st.tabs(["Single Stock Analysis", "Multi-Stock Analysis"])

# =========================
# Indicator Functions
# =========================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# =========================
# Layer 1: Rules
# =========================

def layer1_signals(df):
    signals = {}

    signals["trend"] = int(df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1])
    signals["momentum"] = int(40 <= df["RSI"].iloc[-1] <= 70)
    signals["volume"] = int(df["Volume"].iloc[-1] > df["Volume_MA"].iloc[-1])

    return signals


def rule_decision(signals):
    score = sum(signals.values())

    if score >= 3:
        return "BUY", score
    elif score == 2:
        return "HOLD", score
    else:
        return "SELL", score


# =========================
# Layer 2: ML Model
# =========================

@st.cache_resource
def train_ml_model(df):
    """
    Trains a direction classifier.
    Cached so it does NOT retrain on every interaction.
    """

    data = df.copy()

    # Feature engineering
    data["return"] = data["Close"].pct_change()
    data["volatility"] = data["return"].rolling(10).std()
    data["trend_strength"] = data["EMA20"] - data["EMA50"]

    # IMPORTANT FIX: ensure 1D arrays
    data["volume_ratio"] = data["Volume"].values / data["Volume_MA"].values

    # Target: next-day direction
    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    features = [
        "trend_strength",
        "RSI",
        "volume_ratio",
        "return",
        "volatility"
    ]

    data = data.dropna()

    X = data[features]
    y = data["target"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])

    model.fit(X, y)

    return model, features



def ml_probability(model, features, latest_row):
    X = latest_row[features].values.reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    return prob

def run_screener(tickers, period="1y"):
    results = []

    for ticker in tickers:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            continue  # skip invalid tickers

        # Flatten columns in case of MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Indicators
        df["EMA20"] = ema(df["Close"], 20)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI"] = rsi(df["Close"], 14)
        df["Volume_MA"] = df["Volume"].rolling(20).mean()

        # Layer 1 signals
        signals = layer1_signals(df)
        rule_result, score = rule_decision(signals)

        # ML
        model, features = train_ml_model(df)
        latest = df.copy()
        latest["return"] = latest["Close"].pct_change()
        latest["volatility"] = latest["return"].rolling(10).std()
        latest["trend_strength"] = latest["EMA20"] - latest["EMA50"]
        latest["volume_ratio"] = latest["Volume"].values / latest["Volume_MA"].values
        latest = latest.dropna()
        prob_up = ml_probability(model, features, latest.iloc[-1])

        # Hybrid decision
        if rule_result == "BUY" and prob_up > 0.6:
            final_decision = "BUY"
        elif rule_result == "SELL":
            final_decision = "SELL"
        else:
            final_decision = "HOLD"

        results.append({
            "Ticker": ticker,
            "Decision": final_decision,
            "Rule Score": score,
            "ML Prob": round(prob_up, 2),
            "Trend": signals["trend"],
            "Momentum": signals["momentum"],
            "Volume": signals["volume"]
        })

    return pd.DataFrame(results)



# ===================================================================================================
# Streamlit UI
# ===================================================================================================

# ======= Tab 1 ==========
with tab1:
    st.set_page_config("Hybrid Stock Predictor", layout="centered")
    st.title("📊 Stock Hybrid Screener")
    st.caption("Scan stock using rules + ML hybrid system")
    with st.expander("ℹ️ Explanation of Metrics"):
        st.markdown("""
        **Final Decision**:  
        The hybrid recommendation based on **Layer 1 rules** (trend, momentum, volume) + **ML probability**.  
        - BUY → strong bullish signal  
        - HOLD → uncertain or moderate signal  
        - SELL → bearish signal  

        **Rule Score**:  
        Sum of the Layer 1 rule signals:  
        - Trend (EMA20 > EMA50) = 1 if True else 0  
        - Momentum (RSI in healthy range 40–70) = 1 if True else 0  
        - Volume (current volume > 20-day average) = 1 if True else 0  
        **Total score** ranges from 0–3, higher score → stronger BUY signal  

        **ML Probability (Up)**:  
        Probability (0–1) predicted by the ML model that the stock will move up the next day.  
        - 0.68 means the model estimates a 68% chance of upward movement  

        **Rule Signals**:  
        Individual signals used in Layer 1 rules:  
        - trend → 1 if uptrend (EMA20 > EMA50), else 0  
        - momentum → 1 if RSI healthy (40–70), else 0  
        - volume → 1 if volume above 20-day MA, else 0  

        **Position Fraction**:  
        Portion of your total capital recommended to invest based on ML confidence:  
        - 0% → very low confidence (do not invest)  
        - 30% → low confidence  
        - 50% → medium confidence  
        - 100% → high confidence  

        **Shares to Buy (Next Open)**:  
        Number of shares you could buy at the next day’s opening price based on **Position Fraction** and your total capital.  
        - 0.00 → either the hybrid system suggests SELL/HOLD, or your position fraction is 0 (low confidence)
        """)


    ticker = st.text_input("Stock Ticker", "AAPL")
    period = st.selectbox("Data Period", ["1y", "2y", "3y"], index=1)

    if st.button("Analyze"):
        
        df = yf.download(ticker, period=period)
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)


        if df.empty:
            st.error("Invalid ticker or no data.")
            st.stop()

        # Indicators
        df["EMA20"] = ema(df["Close"], 20)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI"] = rsi(df["Close"], 14)
        df["Volume_MA"] = df["Volume"].rolling(20).mean()

        # ===== Layer 1 =====
        signals = layer1_signals(df)
        rule_result, score = rule_decision(signals)

        # ===== Layer 2 =====
        model, features = train_ml_model(df)

        latest = df.copy()
        latest["return"] = latest["Close"].pct_change()
        latest["volatility"] = latest["return"].rolling(10).std()
        latest["trend_strength"] = latest["EMA20"] - latest["EMA50"]
        latest["volume_ratio"] = latest["Volume"] / latest["Volume_MA"]

        latest = latest.dropna()
        prob_up = ml_probability(model, features, latest.iloc[-1])

        # ===== Hybrid Decision =====
        if rule_result == "BUY" and prob_up > 0.6:
            final_decision = "BUY"
        elif rule_result == "SELL":
            final_decision = "SELL"
        else:
            final_decision = "HOLD"

        # ===== Output =====
        st.subheader("🧠 Hybrid Decision")
        st.metric("Final Decision", final_decision)
        st.metric("Rule Score", score)
        st.metric("ML Probability (Up)", f"{prob_up:.2f}")

        st.write("### Rule Signals")
        st.write(signals)

        # ===== Confidence-based Position Sizing =====
        capital = 10000  # total capital, can be user input later

        if prob_up < 0.6:
            position_fraction = 0
        elif prob_up < 0.7:
            position_fraction = 0.3
        elif prob_up < 0.8:
            position_fraction = 0.5
        else:
            position_fraction = 1.0

        shares_to_buy = (capital * position_fraction) / df["Open"].iloc[-1] if final_decision=="BUY" else 0

        st.metric("Position Fraction", f"{position_fraction*100:.0f}%")
        st.metric("Shares to Buy (Next Open)", f"{shares_to_buy:.2f}")

        # ===== Add explanation =====
        with st.expander("ℹ️ Explanation for Position Sizing"):
            st.markdown("""
            **Position Fraction**: Portion of your total capital suggested to invest in this stock based on ML confidence.
            - 0% → Hold / do not buy  
            - 30% → Small position (low confidence)  
            - 50% → Medium position (moderate confidence)  
            - 100% → Full position (high confidence)  

            **Shares to Buy (Next Open)**: Number of shares to purchase at the **next day’s opening price** based on your position fraction and total capital.  
            - Example: If Position Fraction = 50% and capital = $10,000, the app calculates how many shares you can buy at the next open price.
            """)

        # ===== Chart =====
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["Close"], label="Close")
        ax.plot(df["EMA20"], label="EMA20")
        ax.plot(df["EMA50"], label="EMA50")
        ax.legend()
        st.pyplot(fig)

        # ===== Candlestick Chart =====
        st.subheader("📊 Candlestick Chart")

        fig = go.Figure()

        # Candles
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ))

        # EMA overlays
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["EMA20"],
            mode="lines",
            name="EMA20"
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["EMA50"],
            mode="lines",
            name="EMA50"
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)


        st.caption("ML does not predict price. It estimates probability given similar past conditions.")

    
    # ======= Tab 2 ==========
    with tab2:
        st.title("📊 Stock Hybrid Screener")
        st.caption("Scan multiple stocks at once using rules + ML")
        # Add explanation notes
        with st.expander("ℹ️ Column Explanation"):
            st.markdown("""
            **Decision**: Final hybrid recommendation based on rules + ML probability (BUY / HOLD / SELL)  
            **Rule Score**: Sum of Layer 1 rule signals (Trend + Momentum + Volume). 3 = strong BUY, 2 = HOLD, 1 or 0 = SELL  
            **ML Prob**: Probability (0–1) predicted by the ML model that the stock will go up next day  
            **Trend**: 1 if EMA20 > EMA50 (uptrend), 0 otherwise  
            **Momentum**: 1 if RSI is in healthy range (40–70), 0 otherwise  
            **Volume**: 1 if current volume > 20-day average, 0 otherwise  
            """)

        tickers_input = st.text_area(
            "Enter tickers (comma-separated, e.g., AAPL, MSFT, TSLA)",
            value="AAPL, MSFT, TSLA",
            key="multi"
        )
        period_multi = st.selectbox("Select data period", ["6mo", "1y", "2y"], index=1, key="multi_period")

        if st.button("Run Screener", key="multi_btn"):
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            if not tickers:
                st.warning("Please enter at least one ticker")
            else:
                screener_df = run_screener(tickers, period_multi)
                if screener_df.empty:
                    st.error("No valid tickers found or data unavailable")
                else:
                    # Optional: highlight BUY/SELL
                    def highlight_decisions(val):
                        color = ""
                        if val == "BUY":
                            color = "background-color: lightgreen"
                        elif val == "SELL":
                            color = "background-color: salmon"
                        return color

                    st.dataframe(
                        screener_df.style.applymap(highlight_decisions, subset=["Decision"])
                    )


        
