import os
import subprocess
import sys

def run_step(script_name, description):
    print(f"\n========================================")
    print(f"STEP: {description}")
    print(f"SCRIPT: {script_name}")
    print(f"========================================")
    
    ret = subprocess.call([sys.executable, script_name])
    
    if ret != 0:
        print(f"❌ Error in {script_name}. Pipeline stopped.")
        sys.exit(ret)
    else:
        print(f"✅ {script_name} completed successfully.")

def main():
    print("🚀 Starting End-to-End Pipeline")
    print("Current Working Directory:", os.getcwd())
    
    # 1. Data Preparation
    run_step("dataset_load.py", "Data Loading & Feature Engineering")
    
    # 2. Task 1: Price Forecasting
    run_step("task1_forecasting.py", "Task 1: Price Prediction (MSE)")
    
    # 3. Task 2: Trading Model Training
    run_step("rolling_train.py", "Task 2: Training Conviction Model (Rolling)")
    
    # 4. Task 2: Backtesting
    run_step("backtest_strategy.py", "Task 2: Strategy Execution & Backtest")
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("Check 'mlruns' folder or run 'mlflow ui' to view results.")
    print("Check 'trade_log_*.csv' for execution details.")

if __name__ == "__main__":
    main()
