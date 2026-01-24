import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# CONFIGURATION
# =========================
TRAIN_END_DATE = "2021-06-30"
TEST_START_DATE = "2021-07-01"
TEST_END_DATE = "2021-12-31"

def run_forecasting_task():
    print("Loading data for Task 1 (Price Prediction)...")
    try:
        X_train_full = pd.read_csv('train_features_with_time.csv', index_col=0, parse_dates=True)
        X_test_full  = pd.read_csv('test_features_with_time.csv',  index_col=0, parse_dates=True)
        raw_hourly   = pd.read_csv('hourly_raw.csv', index_col='date', parse_dates=True)
    except FileNotFoundError:
        print("Error: Data files not found. Run dataset_load.py first.")
        return

    # Combine to handle custom split
    full_features = pd.concat([X_train_full, X_test_full]).sort_index()
    full_raw = raw_hourly.loc[full_features.index]
    
    # Target: 4H Log Returns (converted to Price for final MSE)
    # We predict Log Return first as it is stationary
    targets = np.log(full_raw['close'] / full_raw['close'].shift(1))
    
    # SPLIT
    train_mask = (full_features.index <= TRAIN_END_DATE) & (targets.notna())
    test_mask  = (full_features.index >= TEST_START_DATE) & (full_features.index <= TEST_END_DATE) & (targets.notna())
    
    X_train = full_features[train_mask]
    y_train = targets[train_mask]
    
    X_test = full_features[test_mask]
    y_test = targets[test_mask]
    
    print(f"Train Samples: {len(X_train)} (Up to {TRAIN_END_DATE})")
    print(f"Test Samples : {len(X_test)} ({TEST_START_DATE} to {TEST_END_DATE})")

    # MLflow Tracking
    mlflow.set_experiment("Task1_Price_Forecasting")
    
    with mlflow.start_run(run_name="XGB_Regressor_MSE"):
        # MODEL
        params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        
        mlflow.log_params(params)
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # PREDICT
        preds_log_ret = model.predict(X_test)
        
        # METRICS (Log Returns)
        mse_log = mean_squared_error(y_test, preds_log_ret)
        mae_log = mean_absolute_error(y_test, preds_log_ret)
        
        print(f"Log Return MSE: {mse_log:.8f}")
        
        # RECONSTRUCT PRICE (For Requirement MSE on Price?) 
        # Requirement: "MSE on test set price predictions"
        # We start with Close Price at T-1
        initial_prices = full_raw.loc[X_test.index, 'close'].shift(1).fillna(method='bfill') # Need T-1 price
        # Check alignment logic carefully
        # y_test is log(P_t / P_{t-1})
        # Prediction is est_log_ret
        # Est_Price_t = P_{t-1} * exp(est_log_ret)
        
        # Let's align carefully using raw index
        # To get P_{t-1}, shift close by 1
        prev_close = full_raw['close'].shift(1).loc[X_test.index]
        
        pred_prices = prev_close * np.exp(preds_log_ret)
        actual_prices = full_raw.loc[X_test.index, 'close']
        
        # Clean NaNs if any (first row might be nan due to shift)
        valid = prev_close.notna()
        
        price_mse = mean_squared_error(actual_prices[valid], pred_prices[valid])
        price_rmse = np.sqrt(price_mse)
        
        print(f"Price MSE: {price_mse:.4f}")
        print(f"Price RMSE: {price_rmse:.4f}")
        
        mlflow.log_metric("mse_log_return", mse_log)
        mlflow.log_metric("mae_log_return", mae_log)
        mlflow.log_metric("mse_price", price_mse)
        mlflow.log_metric("rmse_price", price_rmse)
        
        # PLOT
        plt.figure(figsize=(12, 6))
        plt.plot(actual_prices.index, actual_prices, label='Actual Price', color='black', alpha=0.6)
        plt.plot(pred_prices.index, pred_prices, label='Predicted Price', color='blue', alpha=0.6, linestyle='--')
        plt.title(f"Task 1: Price Forecast (RMSE: ${price_rmse:.2f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = "task1_price_forecast.png"
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
        
        mlflow.log_artifact(plot_path)
        
        print("✅ Task 1 Completed.")

if __name__ == "__main__":
    run_forecasting_task()
