import numpy as np
import pandas as pd

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicators without lookahead leakage."""
    data = df.copy()

    # Returns and Historical Volatility
    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Hist_Vol_20"] = data["Log_Return"].rolling(20).std()

    # Moving Averages
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    data["SMA_Ratio"] = data["SMA_10"] / data["SMA_50"]

    # Bollinger Bands (20-day)
    roll_mean = data["Close"].rolling(20).mean()
    roll_std = data["Close"].rolling(20).std()
    data["BB_Upper"] = roll_mean + (2 * roll_std)
    data["BB_Lower"] = roll_mean - (2 * roll_std)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / (roll_mean + 1e-9)

    # Momentum (RSI 14)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12-day EMA - 26-day EMA)
    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # Volume Dynamics
    vol_sma = data["Volume"].rolling(20).mean()
    data["Vol_Ratio"] = data["Volume"] / (vol_sma + 1e-9)

    # Target: Next-day Close Price (t + 1)
    data["Target_Next_Close"] = data["Close"].shift(-1)
    return data.dropna()

def chronological_train_test_split(df: pd.DataFrame, test_pct: float = 0.20):
    """Sequential time-series split preserving chronological order."""
    feature_cols = [
        "SMA_10", "SMA_50", "SMA_Ratio", "BB_Upper", "BB_Lower",
        "BB_Width", "Hist_Vol_20", "RSI_14", "MACD", "MACD_Signal",
        "Vol_Ratio", "Close", "Volume"
    ]
    split_idx = int(len(df) * (1 - test_pct))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return (
        train[feature_cols], test[feature_cols],
        train["Target_Next_Close"], test["Target_Next_Close"],
        feature_cols, test
    )