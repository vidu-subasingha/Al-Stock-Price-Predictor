import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def train_baseline_rf(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    rf = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    xgb = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.8, random_state=42)
    xgb.fit(X_train, y_train)
    return xgb

def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true.values - y_pred) / y_true.values)) * 100
    r2 = r2_score(y_true, y_pred)

    # Directional Hit Accuracy
    prev_close = y_true.shift(1).dropna()
    actual_delta = np.sign(y_true.iloc[1:].values - prev_close.values)
    pred_delta = np.sign(y_pred[1:] - prev_close.values)
    dir_acc = np.mean(actual_delta == pred_delta) * 100

    print(f"\n--- {model_name} Metrics ---")
    print(f"MAE:                  ${mae:.2f}")
    print(f"RMSE:                 ${rmse:.2f}")
    print(f"MAPE:                 {mape:.2f}%")
    print(f"R²:                   {r2:.4f}")
    print(f"Directional Accuracy: {dir_acc:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "Dir_Acc": dir_acc}