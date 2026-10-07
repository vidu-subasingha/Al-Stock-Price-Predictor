# 📈 AI Stock Price Predictor

An end-to-end quantitative financial machine learning pipeline and interactive dashboard that forecasts equity price movements using engineered technical indicators, sequential chronological cross-validation, and tree ensemble models (**Random Forest** and **XGBoost**).

## Features
- **Automated Data Ingestion**: Historical OHLCV via `yfinance`.
- **Technical Feature Engineering**: RSI, MACD, Bollinger Bands, Moving Average Ratios, and rolling volatility.
- **No Lookahead Bias**: Sequential time-series splitting.
- **Evaluation Metrics**: MAE, RMSE, MAPE, $R^2$, and Directional Hit Accuracy.
- **Interactive Dashboard**: Streamlit interface with real-time ticker selection and Plotly charts.

## Running the Project