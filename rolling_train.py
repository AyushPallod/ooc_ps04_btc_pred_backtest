
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from dateutil.relativedelta import relativedelta

# =========================
# CONFIG
# =========================
INITIAL_TRAIN_END = "2021-06-30"
TEST_START = "2021-07-01"
TEST_END = "2022-03-31"
RETRAIN_STEP_MONTHS = 1

# =========================
# LOAD DATA
# =========================
print("Loading data for Rolling Training...")
try:
    X_train_full = pd.read_csv('train_features_with_time.csv', index_col=0, parse_dates=True)
    X_test_full  = pd.read_csv('test_features_with_time.csv',  index_col=0, parse_dates=True)
    raw_hourly   = pd.read_csv('hourly_raw.csv', index_col='date', parse_dates=True)
except FileNotFoundError:
    print("Error: Features not found. Please run dataset_load.py first.")
    exit()

# Combine for rolling window
full_features = pd.concat([X_train_full, X_test_full]).sort_index()
full_raw = raw_hourly.loc[full_features.index]
targets = full_raw['Target_10d_Up']

valid_mask = targets.notna()
full_features_valid = full_features[valid_mask]
targets_valid = targets[valid_mask]

# =========================
# ROLLING WALK-FORWARD LOOP
# =========================
current_train_end = pd.Timestamp(INITIAL_TRAIN_END)
final_test_end = pd.Timestamp(TEST_END)

# =========================
# MLFLOW & TRAINING
# =========================
import mlflow

mlflow.set_experiment("Task2_Rolling_Training")

with mlflow.start_run(run_name="Rolling_XGB"):
    predictions = []
    print(f"Starting Rolling Validation from {current_train_end} to {final_test_end}")
    
    total_logloss = 0
    count = 0
    
    model = None 

    while current_train_end < final_test_end:
        next_step_end = current_train_end + relativedelta(months=RETRAIN_STEP_MONTHS)
        if next_step_end > final_test_end:
            next_step_end = final_test_end + relativedelta(days=1) 
            
        print(f"  Training up to {current_train_end.date()}... Predicting until {next_step_end.date()}")
        
        train_mask = full_features_valid.index <= current_train_end
        test_window_mask = (full_features.index > current_train_end) & (full_features.index <= next_step_end)
        
        if not test_window_mask.any():
            break
            
        X_tr = full_features_valid[train_mask]
        y_tr = targets_valid[train_mask]
        
        X_te = full_features[test_window_mask] 
        
        pos = y_tr.sum()
        neg = len(y_tr) - pos
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        model.fit(X_tr, y_tr, verbose=False)
        
        probs = model.predict_proba(X_te)[:, 1]
        
        step_preds = pd.DataFrame(index=X_te.index)
        step_preds['Prob_Up_10d'] = probs
        predictions.append(step_preds)
        
        current_train_end = next_step_end
        count += 1

    all_predictions = pd.concat(predictions).sort_index()
    all_predictions.to_csv('predictions_rolling.csv')
    print(f"Saved rolling predictions: {len(all_predictions)} rows.")

    # =========================
    # EXPLAINABILITY (Feature Importance)
    # =========================
    print("Calculating Feature Importance on Final Model...")

    # 1. Gain-based Importance
    try:
        importance_gain = model.get_booster().get_score(importance_type='gain')
        importance_df = pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
        importance_df = importance_df.sort_values(by='Gain', ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'][::-1], importance_df['Gain'][::-1])
        plt.title("Top 20 Features (Gain)")
        plt.xlabel("Gain")
        plt.tight_layout()
        plt.savefig('feature_importance_gain.png')
        print("Saved feature_importance_gain.png")
        mlflow.log_artifact('feature_importance_gain.png')
    except Exception as e:
        print(f"Could not plot importance: {e}")

    print("✅ Rolling Validation Complete.")