import pandas as pd
import yfinance as yf

def load_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches OHLCV records via yfinance and flattens headers."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().sort_index()
    expected = ["Open", "High", "Low", "Close", "Volume"]
    return df[[c for c in expected if c in df.columns]]