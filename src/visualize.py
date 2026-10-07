import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

def export_residual_and_forecast_plot(test_df: pd.DataFrame, rf_pred: np.ndarray, xgb_pred: np.ndarray, ticker: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.5, 1]}, sharex=True)
    dates = test_df.index
    y_true = test_df["Target_Next_Close"]

    # Upper: Forecast curves
    ax1.plot(dates, y_true, label="Actual Price", color="#1f77b4", linewidth=2.0)
    ax1.plot(dates, rf_pred, label="Random Forest", color="#2ca02c", linestyle="--", alpha=0.85)
    ax1.plot(dates, xgb_pred, label="XGBoost", color="#d62728", linestyle=":", linewidth=1.8, alpha=0.9)
    ax1.set_title(f"{ticker} Forecast Performance (Out-of-Sample Window)", fontsize=13, weight="bold")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")

    # Lower: Residual errors
    ax2.plot(dates, rf_pred - y_true.values, label="RF Error", color="#2ca02c", alpha=0.5, linewidth=1.0)
    ax2.plot(dates, xgb_pred - y_true.values, label="XGB Error", color="#d62728", alpha=0.5, linewidth=1.0)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.set_ylabel("Residual ($y_{pred} - y_{true}$)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("forecast_residuals.png", dpi=300)
    plt.close()

def export_feature_importance_plot(model, feature_names: list, model_title: str):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances[sorted_idx], y=[feature_names[i] for i in sorted_idx], palette="viridis")
    plt.title(f"Feature Importance ({model_title})", fontsize=12, weight="bold")
    plt.xlabel("Importance Weight")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    plt.close()