# Project Report: BTC/USDT Forecasting & Trading Strategy

This report details the implementation of the machine learning forecasting model, the rolling retraining framework, and the algorithmic trading strategy used for BTC/USDT.

---

## 1. Task 1: Forecasting XGBoost Model

### **Objective**
To predict the future movement of Bitcoin prices using a machine learning approach, specifically focusing on Log Returns to ensure stationarity, which are then reconstructed into price predictions.

### **Data & Features**
- **Input Data:** `hourly_raw.csv` (OHLCV data) combined with engineered features (`train_features_with_time.csv`, `test_features_with_time.csv`).
- **Target Variable:** 4-Hour Log Returns
  $$ \text{Log Return} = \ln\left(\frac{P_t}{P_{t-1}}\right) $$
- **Data Split:**
  - **Training:** Data up to `2021-06-30`
  - **Testing:** `2021-07-01` to `2021-12-31`

### **Model Architecture & Hyperparameters**
- **Algorithm:** XGBoost Regressor (`XGBRegressor`)
- **Hyperparameters:**
  - `n_estimators`: 500
  - `max_depth`: 6
  - `learning_rate`: 0.05
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `objective`: `reg:squarederror`
  - `random_state`: 42

### **Prediction & Reconstruction**
1.  **Prediction:** The model predicts the **Log Return** for the next period.
2.  **Price Reconstruction:**
    To evaluate the model against the user requirement of "MSE on Price", the predicted log returns are converted back to price levels using the previous period's Close price:
    $$ \hat{P}_t = P_{t-1} \times e^{\text{Predicted Log Return}} $$

### **Metrics**
The model is evaluated on:
- **MSE (Log Returns):** Direct model loss.
- **Price MSE & RMSE:** Measures the error in the reconstructed price vs. actual price, which is the primary business metric.

---

## 2. Rolling Retrain Framework

### **Objective**
To adapt the model to changing market conditions by periodically retraining it on the most recent data using a Walk-Forward Validation approach.

### **Framework Details**
- **Initial Training Window:** Up to `2021-06-30`
- **Test Period:** `2021-07-01` to `2022-03-31`
- **Retraining Frequency:** Every **1 Month**
- **Target:** Classification (`Target_10d_Up`) - Predicting if price will be up in 10 days.

### **Walk-Forward Loop Logic**
1.  **Train:** Model is trained on all available data up to `Current Train End`.
2.  **Predict:** Model makes predictions for the next 1 month (the "Test Window").
3.  **Step:** `Current Train End` is advanced by 1 month.
4.  **Repeat:** The process repeats until the end of the final test period.

### **Model Configuration (Rolling)**
- **Algorithm:** XGBoost Classifier
- **Hyperparameters:** Same as Task 1 but adapted for classification (Objective: `binary:logistic`, Eval Metric: `logloss`).
- **Class Imbalance:** Handled using `scale_pos_weight = Negative Samples / Positive Samples`.
- **Output:** A continuous probability score (`Prob_Up_10d`) representing the likelihood of an upward trend.

---

## 3. Trading Strategy

### **Overview**
A metric-driven strategy that combines the **Rolling ML Predictions** with **Technical Indicators** to execute Long and Short trades on BTC/USDT.

### **Data Preparation**
- **Resampling:** Raw hourly data is resampled to **4-Hour** candles.
- **Regime Filter:** A Daily **EMA 200** is used to define the market regime.
  - Price > Daily EMA 200 $\rightarrow$ **BULL Regime**
  - Price < Daily EMA 200 $\rightarrow$ **BEAR Regime**

### **Entry Logic & Position Sizing**

The strategy employs two distinct sizing methods depending on the trade direction, optimizing for **Growth** in Bull markets and **Safety** in Bear markets.

#### A. Long Setup (Bull Market)
*   **Condition:** Market must be in **BULL Regime**.
*   **Trigger:**
    *   Price crosses above **SMA 20** (or **EMA 9** in Hyper Mode).
    *   **ML Confirmation:** ML Prediction must **NOT** be Bearish (Prob > 0.4).
*   **Position Sizing: Target Exposure Method**
    Instead of fixed risk, long trades aim for a target percentage of equity based on conviction.
    *   **Weak Bull (Prob < 0.6):** Target **50% Equity** exposure.
    *   **Strong Bull (Prob > 0.6):** Target **100% Equity** exposure.
    *   **Super Bull (Prob > 0.7):** Target **150% Equity** exposure (Leverage) [Hyper Mode Only].
    *   *Formula:* $\text{Size} = \frac{\text{Equity} \times \text{Target Exposure}}{\text{Entry Price}}$

#### B. Short Setup (Bear Market)
*   **Condition:** Market must be in **BEAR Regime** and Shorts allowed.
*   **Trigger:**
    *   Price breaks below **Bollinger Band Lower** (Std 2.0).
    *   **ML Confirmation:** ML Prediction must **NOT** be Bullish (Prob < 0.6).
*   **Position Sizing: Volatility-Adjusted Risk Method**
    Shorts utilize a defensive, risk-based approach to limit downsides.
    *   **Risk Per Trade:** Fixed at **2%** of current Equity.
    *   **Stop Distance:** Calculated as **2.5 x ATR** (Average True Range).
    *   *Formula:* $\text{Size} = \frac{\text{Equity} \times 0.02}{2.5 \times \text{ATR}}$
    *   *Constraint:* Max position size capped at **50% Equity** for safety.

### **Risk Management**

#### Stop Loss
- **Long:** Initial Hard Stop is set at **4.0 x ATR** below entry price.
- **Short:** Initial Hard Stop is set at **2.5 x ATR** above entry price.

#### Trailing Stop
- **Logic:** The stop loss is trailed to lock in profits.
- **Sensitivity:**
  - **Bull Regime:** Loose Trail (4.0 x ATR) to allow volatility.
  - **Bear Regime:** Tight Trail (2.5 x ATR) for quick defense.

#### Pyramiding (Hyper Mode / Super Bull)
- **Logic:** Adds to the winning position if the trend is strong and profitable.
- **Condition:** Position is profitable (>5%) AND Leverage is < 1.0.
- **Action:** Adds 0.5x Risk magnitude to the position.
- **Margin Boost:** If ML Probability is "Super Bull" (>0.75), additional margin is used to boost size up to 1.4x leverage.

### **Exit Logic**

1.  **Stop Loss:** Triggered if price hits the Hard Stop or Trailing Stop.
2.  **Signal Exit (Trend Break):**
    *   **Long:** Price closes below **SMA 20** (or EMA 9) **AND** ML is Bearish.
    *   **Short:** ML Signal flips to Bullish.

### **Performance Metrics**
- **Total Return:** Net profit percentage.
- **Sharpe Ratio:** Risk-adjusted return (Mean Return / Std Dev).
- **Sortino Ratio:** Downside risk-adjusted return.
- **Max Drawdown:** Maximum peak-to-trough decline in equity.

---

### **Strategy Performance Update (2022-01-01 to 2022-03-31)**

The following table summarizes the strategy performance for the period of Q1 2022, tested with the default parameters (Non-Hyper Mode).

| Metric | Result |
| :--- | :--- |
| **Final Equity** | **$10,651.11** |
| **Total Return** | **+6.51%** |
| **Buy & Hold Return** | -8.05% |
| **Max Drawdown** | -2.64% |
| **Sharpe Ratio** | 2.68 |
| **Sortino Ratio** | 1.76 |
| **Total Trades** | 6 |

> [!NOTE]
> The strategy significantly outperformed Buy & Hold during this bearish/volatile period, achieving a positive return of 6.51% while Bitcoin declined by 8.05%.

![Strategy Equity Curve](final_backtest_2022-01-01_2022-03-31.png)
