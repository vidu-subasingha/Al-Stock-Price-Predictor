import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
from xgboost import XGBRegressor
import yfinance as yf

st.set_page_config(page_title="AlphaPredict | ML Stock Forecaster", page_icon="📈", layout="wide")

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna().sort_index()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Hist_Vol_20"] = data["Log_Return"].rolling(20).std()
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    data["SMA_Ratio"] = data["SMA_10"] / data["SMA_50"]

    roll_mean = data["Close"].rolling(20).mean()
    roll_std = data["Close"].rolling(20).std()
    data["BB_Upper"] = roll_mean + (2 * roll_std)
    data["BB_Lower"] = roll_mean - (2 * roll_std)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / (roll_mean + 1e-9)

    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["Vol_Ratio"] = data["Volume"] / (data["Volume"].rolling(20).mean() + 1e-9)
    data["Target_Next_Close"] = data["Close"].shift(-1)
    return data

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ Model Settings")
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    col1, col2 = st.columns(2)
    start_d = col1.date_input("Start Date", datetime.date(2019, 1, 1))
    end_d = col2.date_input("End Date", datetime.date.today())
    test_ratio = st.slider("Test Split Ratio", 0.10, 0.35, 0.20, 0.05)
    rf_trees = st.slider("RF Estimators", 50, 300, 150, 25)
    xgb_lr = st.select_slider("XGBoost Learning Rate", options=[0.01, 0.02, 0.03, 0.05, 0.1], value=0.03)

st.title(f"📈 Stock Price Predictor: {ticker}")

raw = load_data(ticker, start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d"))
if raw.empty or len(raw) < 100:
    st.error("Insufficient market data for this ticker/date range.")
    st.stop()

featured = engineer_features(raw)
feat_cols = ["SMA_10", "SMA_50", "SMA_Ratio", "BB_Upper", "BB_Lower", "BB_Width", "Hist_Vol_20", "RSI_14", "MACD", "MACD_Signal", "Vol_Ratio", "Close", "Volume"]

latest_row = featured[feat_cols].iloc[[-1]]
clean = featured.dropna(subset=feat_cols + ["Target_Next_Close"])

split_idx = int(len(clean) * (1 - test_ratio))
train, test = clean.iloc[:split_idx], clean.iloc[split_idx:]

rf = RandomForestRegressor(n_estimators=rf_trees, max_depth=6, random_state=42, n_jobs=-1).fit(train[feat_cols], train["Target_Next_Close"])
xgb = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=xgb_lr, subsample=0.8, random_state=42).fit(train[feat_cols], train["Target_Next_Close"])

rf_preds = rf.predict(test[feat_cols])
xgb_preds = xgb.predict(test[feat_cols])
rf_next = float(rf.predict(latest_row)[0])
xgb_next = float(xgb.predict(latest_row)[0])
curr_price = float(raw["Close"].iloc[-1])

# Top KPI Cards
k1, k2, k3 = st.columns(3)
k1.metric("Current Close", f"${curr_price:.2f}")
k2.metric("RF Next-Day Prediction", f"${rf_next:.2f}", delta=f"{rf_next - curr_price:+.2f}")
k3.metric("XGBoost Next-Day Prediction", f"${xgb_next:.2f}", delta=f"{xgb_next - curr_price:+.2f}")

# Visualizations
tab1, tab2, tab3 = st.tabs(["📊 Out-of-Sample Test Evaluation", "🕯️ Technical Candlesticks", "🧬 Feature Importance"])

with tab1:
    fig_eval = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Actual vs Forecast", "Residual Errors"), row_heights=[0.7, 0.3])
    fig_eval.add_trace(go.Scatter(x=test.index, y=test["Target_Next_Close"], name="Actual Close", line=dict(color="#1f77b4")), row=1, col=1)
    fig_eval.add_trace(go.Scatter(x=test.index, y=rf_preds, name="RF Predicted", line=dict(color="#2ca02c", dash="dash")), row=1, col=1)
    fig_eval.add_trace(go.Scatter(x=test.index, y=xgb_preds, name="XGB Predicted", line=dict(color="#d62728", dash="dot")), row=1, col=1)
    fig_eval.add_trace(go.Scatter(x=test.index, y=rf_preds - test["Target_Next_Close"].values, name="RF Residual", line=dict(color="#2ca02c")), row=2, col=1)
    fig_eval.add_trace(go.Scatter(x=test.index, y=xgb_preds - test["Target_Next_Close"].values, name="XGB Residual", line=dict(color="#d62728")), row=2, col=1)
    fig_eval.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig_eval, use_container_width=True)

with tab2:
    sub = featured.tail(120)
    fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25])
    fig_c.add_trace(go.Candlestick(x=sub.index, open=sub["Open"], high=sub["High"], low=sub["Low"], close=sub["Close"], name="OHLC"), row=1, col=1)
    fig_c.add_trace(go.Scatter(x=sub.index, y=sub["BB_Upper"], line=dict(color="rgba(100,100,255,0.4)", dash="dot"), name="BB Upper"), row=1, col=1)
    fig_c.add_trace(go.Scatter(x=sub.index, y=sub["BB_Lower"], line=dict(color="rgba(100,100,255,0.4)", dash="dot"), fill="tonexty", fillcolor="rgba(173,216,230,0.15)", name="BB Lower"), row=1, col=1)
    fig_c.add_trace(go.Bar(x=sub.index, y=sub["Volume"], marker_color="royalblue", name="Volume"), row=2, col=1)
    fig_c.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_c, use_container_width=True)

with tab3:
    imp_df = pd.DataFrame({"Feature": feat_cols, "Importance": xgb.feature_importances_}).sort_values("Importance", ascending=True)
    fig_imp = go.Figure(go.Bar(x=imp_df["Importance"], y=imp_df["Feature"], orientation="h", marker_color="#d62728"))
    fig_imp.update_layout(title="XGBoost Feature Importance", template="plotly_white", height=400)
    st.plotly_chart(fig_imp, use_container_width=True)