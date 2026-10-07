from src.data_loader import load_ticker_data
from src.features import generate_features, chronological_train_test_split
from src.models import train_baseline_rf, train_xgboost, evaluate_model
from src.visualize import export_residual_and_forecast_plot, export_feature_importance_plot

def main():
    ticker = "AAPL"
    print(f"[1/5] Ingesting market data for {ticker}...")
    raw_df = load_ticker_data(ticker, "2019-01-01", "2024-01-01")

    print("[2/5] Calculating technical indicators and lag features...")
    featured_df = generate_features(raw_df)

    print("[3/5] Performing sequential train/test split...")
    X_train, X_test, y_train, y_test, feats, test_df = chronological_train_test_split(featured_df, test_pct=0.20)

    print("[4/5] Training Random Forest and XGBoost...")
    rf = train_baseline_rf(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    rf_preds = rf.predict(X_test)
    xgb_preds = xgb.predict(X_test)

    print("[5/5] Computing out-of-sample metrics & exporting plots...")
    evaluate_model(y_test, rf_preds, "Random Forest")
    evaluate_model(y_test, xgb_preds, "XGBoost")

    export_residual_and_forecast_plot(test_df, rf_preds, xgb_preds, ticker)
    export_feature_importance_plot(xgb, feats, "XGBoost")
    print("Pipeline execution complete! Generated: forecast_residuals.png, feature_importance.png")

if __name__ == "__main__":
    main()