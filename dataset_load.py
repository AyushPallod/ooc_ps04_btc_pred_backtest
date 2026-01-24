import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =========================
# 1. LOAD & MERGE RAW DATA
# =========================
files = [
    "BTC-2017min.csv",
    "BTC-2018min.csv",
    "BTC-2019min.csv",
    "BTC-2020min.csv",
    "BTC-2021min.csv"
]

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

full_df = pd.concat(dfs, axis=0)
full_df['date'] = pd.to_datetime(full_df['date'])
full_df = full_df.sort_values('date').set_index('date')

# =========================
# 2. RESAMPLE TO HOURLY
# =========================
df = full_df.resample('H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'Volume BTC': 'sum'
})

df = df.ffill().dropna()

# =========================
# 3. TECHNICAL INDICATORS
# =========================

# --- Trend ---
df['SMA_50'] = df['close'].rolling(50).mean()
df['SMA_200'] = df['close'].rolling(200).mean()
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']

# --- RSI (Wilder) ---
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# --- ATR (Wilder) ---
high_low = df['high'] - df['low']
high_close = (df['high'] - df['close'].shift()).abs()
low_close = (df['low'] - df['close'].shift()).abs()

true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = true_range.ewm(com=13, adjust=False).mean()

# --- Bollinger Bands ---
df['MA20'] = df['close'].rolling(20).mean()
df['STD20'] = df['close'].rolling(20).std()
df['Bollinger_High'] = df['MA20'] + 2 * df['STD20']
df['Bollinger_Low'] = df['MA20'] - 2 * df['STD20']

# --- Volume Normalization ---
df['Volume_Ratio'] = df['Volume BTC'] / df['Volume BTC'].rolling(20).mean()

# --- Returns ---
df['Log_Returns'] = np.log(df['close'] / df['close'].shift())

# --- Lags ---
df['Close_t-1'] = df['close'].shift(1)
df['Close_t-2'] = df['close'].shift(2)

# =========================
# 4. TARGETS
# =========================

# Next hour direction (optional)
df['Target_Up'] = (df['close'].shift(-1) > df['close']).astype(int)

# 10-day trend continuation (PRIMARY)
df['Target_10d_Up'] = (df['close'].shift(-240) >= df['close'] * 1.03).astype(int)

df = df.dropna()

# =========================
# 5. CYCLICAL FEATURES
# =========================
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

# =========================
# 6. TRAIN / TEST SPLIT
# =========================
split_date = "2021-06-30"
train_df = df.loc[:split_date].copy()
test_df = df.loc[split_date:].copy()

# =========================
# 7. FEATURE SEPARATION
# =========================
scale_features = [
    'open', 'high', 'low', 'close',
    'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
    'EMA_50', 'EMA_200', 'MACD',
    'RSI', 'ATR',
    'Bollinger_High', 'Bollinger_Low',
    'Volume_Ratio',
    'Log_Returns',
    'Close_t-1', 'Close_t-2'
]

passthrough_features = [
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    'is_weekend'
]

# =========================
# 8. SCALING (NO LEAKAGE)
# =========================
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_df[scale_features])
test_scaled = scaler.transform(test_df[scale_features])

train_scaled_df = pd.DataFrame(train_scaled, columns=scale_features, index=train_df.index)
test_scaled_df = pd.DataFrame(test_scaled, columns=scale_features, index=test_df.index)

train_final = pd.concat([train_scaled_df, train_df[passthrough_features]], axis=1)
test_final = pd.concat([test_scaled_df, test_df[passthrough_features]], axis=1)

# =========================
# 9. SAVE OUTPUTS
# =========================
df.to_csv("hourly_raw.csv")
train_final.to_csv("train_features_with_time.csv")
test_final.to_csv("test_features_with_time.csv")

print("Dataset generation complete.")
print(f"Train samples: {len(train_final)} | Test samples: {len(test_final)}")
