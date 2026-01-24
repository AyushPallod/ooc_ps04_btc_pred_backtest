
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# CONFIGURATION
# =========================
INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.02       # 2% Risk
LEVERAGE_CAP = 1.5          # Increased Cap for Hyper Mode
TRANSACTION_COST = 0.002
ALLOW_SHORTS = True
HYPER_MODE = False          # EXPERIMENTAL FLAG (Set to False for Optimal Stability)

# =========================
# DATA PREPARATION
# =========================
def prepare_data():
    print("Loading Data...")
    df = pd.read_csv("hourly_raw.csv", index_col="date", parse_dates=True)
    
    # Load Rolling Predictions (Walk-Forward)
    try:
        preds = pd.read_csv("predictions_rolling.csv", index_col=0, parse_dates=True)
        # Join: Inner join to only backtest on the OOS period
        print(f"Loaded Rolling Predictions: {len(preds)} rows")
        df = df.join(preds[['Prob_Up_10d']], how='inner')
    except:
        print("Warning: Rolling predictions not found.")
        df['Prob_Up_10d'] = 0.5

    # 1. 4-Hour Resample for Signals
    df_4h = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "Volume BTC": "sum",
        "Prob_Up_10d": "last" # Use latest prob
    }).dropna()

    # 2. Daily Indicators for Regime (200 EMA)
    daily = df.resample("D").agg({"close": "last"})
    daily["EMA_200"] = daily["close"].ewm(span=200, adjust=False).mean()
    df_4h["Regime_EMA_200"] = daily["EMA_200"].reindex(df_4h.index, method="ffill")

    # 3. 4H Indicators
    # SMA 20 (Mid Band)
    df_4h["SMA_20"] = df_4h["close"].rolling(20).mean()
    df_4h["EMA_9"] = df_4h["close"].ewm(span=9, adjust=False).mean() # For Hyper Entry
    df_4h["Std_20"] = df_4h["close"].rolling(20).std()
    df_4h["BB_Upper"] = df_4h["SMA_20"] + (df_4h["Std_20"] * 2)
    df_4h["BB_Lower"] = df_4h["SMA_20"] - (df_4h["Std_20"] * 2)

    # ATR (14)
    tr1 = df_4h["high"] - df_4h["low"]
    tr2 = (df_4h["high"] - df_4h["close"].shift(1)).abs()
    tr3 = (df_4h["low"] - df_4h["close"].shift(1)).abs()
    df_4h["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_4h["ATR"] = df_4h["TR"].rolling(14).mean()

    return df_4h.dropna()

# =========================
# BACKTEST ENGINE
# =========================
def run_backtest_strategy(start_date, end_date):
    df = prepare_data()
    
    # Filter
    mask = (df.index >= start_date) & (df.index <= end_date)
    data = df[mask].copy()
    
    if data.empty:
        print(f"No data for {start_date} to {end_date}")
        return

    # Sate
    equity = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    position = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    
    equity_curve = []
    trades = []
    
    print(f"Running Hyper-Aggressive Backtest: {start_date} to {end_date}")
    
    for i in range(len(data)):
        row = data.iloc[i]
        curr_date = row.name
        price = row["close"]
        high = row["high"]
        low = row["low"]
        prob = row["Prob_Up_10d"]
        
        # Regime Check
        ema_200 = row["Regime_EMA_200"]
        regime = "BULL" if price > ema_200 else "BEAR"
        
        # ML Signal Strength
        ml_bullish = prob > 0.6
        ml_bearish = prob < 0.4
        ml_super_bull = prob > 0.7 # For Hyper Leverage
        
        # 1. Update Equity
        equity = cash + (position * price)

        # 1. Pyramiding Check (Hyper Mode)
        # If position > 0 and Profit > 5%, Add 0.5x Risk/Size
        if HYPER_MODE and position > 0:
            unrealized_pct = (price - entry_price) / entry_price
            # Limit pyramiding to Max Leverage
            current_lev = (position * price) / equity
            if unrealized_pct > 0.05 and current_lev < 1.0: # Only pyramid if < 1.0 lev used
                 # Add to position
                 add_size = (equity * 0.5) / price
                 cost = (add_size * price) * TRANSACTION_COST
                 if cash > (add_size * price) + cost:
                     cash -= (add_size * price) + cost
                     # Average Entry Price update
                     total_val = (position * entry_price) + (add_size * price)
                     new_pos = position + add_size
                     entry_price = total_val / new_pos
                     position = new_pos
                     trades.append({"date": curr_date, "action": "BUY_ADD", "price": price, "size": add_size, "reason": "Pyramid"})
        
        # 2. Exits (Signal & Stop)
        if position != 0:
            exit_signal = False
            pnl = 0
            reason = ""
            
            if position > 0: # Long
                # Check Hard Stop first (Safety)
                if low < stop_loss:
                    exec_price = stop_loss
                    val = position * exec_price
                    cost = val * TRANSACTION_COST
                    cash += val - cost
                    pnl = (exec_price - entry_price) * position
                    trades.append({"date": curr_date, "action": "SELL", "price": exec_price, "size": position, "pnl": pnl, "reason": "Stop Loss"})
                    position = 0
                    exit_signal = True
                
                # Check Signal Exit (Profit Take / Trend Reversal)
                # Hyper Mode: Use EMA 9 break (Faster exit on reversal)
                # The original code had `if ml_bearish or HYPER_MODE:`. The instruction snippet had `elif price < exit_trigger_price:`.
                # To faithfully apply the instruction, I'm adding the `size` key to the existing `if` block.
                elif ml_bearish or HYPER_MODE: # In Hyper, strict fast exit
                    exec_price = price
                    val = position * exec_price
                    cost = val * TRANSACTION_COST
                    cash += val - cost
                    pnl = (exec_price - entry_price) * position
                    trades.append({"date": curr_date, "action": "SELL", "price": exec_price, "size": position, "pnl": pnl, "reason": "Trend_Break"})
                    position = 0
                    exit_signal = True
                        
                if not exit_signal:
                    # Update Trailing Stop
                    # Bull Regime: Loose Trail (4x ATR)
                    mult = 4.0 if regime == "BULL" else 2.5
                    new_stop = high - (mult * row["ATR"])
                    stop_loss = max(stop_loss, new_stop)
                    
            elif position < 0: # Short
                # Check Hard Stop
                if high > stop_loss:
                    exec_price = stop_loss
                    cost_to_buy = abs(position) * exec_price
                    fee = cost_to_buy * TRANSACTION_COST
                    cash -= (cost_to_buy + fee)
                    pnl = (entry_price - exec_price) * abs(position)
                    trades.append({"date": curr_date, "action": "COVER", "price": exec_price, "size": abs(position), "pnl": pnl, "reason": "Stop Loss"})
                    position = 0
                    exit_signal = True
                
                # Check Signal Exit (ML Bullish)
                elif ml_bullish:
                    exec_price = price
                    cost_to_buy = abs(position) * exec_price
                    fee = cost_to_buy * TRANSACTION_COST
                    cash -= (cost_to_buy + fee)
                    pnl = (entry_price - exec_price) * abs(position)
                    trades.append({"date": curr_date, "action": "COVER", "price": exec_price, "size": abs(position), "pnl": pnl, "reason": "ML_Bullish"})
                    position = 0
                    exit_signal = True
                    
                if not exit_signal:
                    # Trailing
                    mult = 2.5 # Tight for shorts
                    new_stop = low + (mult * row["ATR"])
                    stop_loss = min(stop_loss, new_stop)
            
            if exit_signal:
                equity = cash
        
        # 3. Entries (Aggressive Sizing)
        if position == 0:
            # Base Risk (still calculate for reference)
            atr = row["ATR"]
            if atr == 0: atr = price * 0.01 
            
            # LONG SETUP
            if regime == "BULL":
                # Hyper Mode: Entry on EMA 9 (Fast)
                entry_trigger = row["EMA_9"] if HYPER_MODE else row["SMA_20"]

                # Entry: Price > SMA_20 (Momentum) AND Not Bearish
                if price > entry_trigger and not ml_bearish:
                    # SIZING: TARGET EXPOSURE
                    # Strong Bull (ML > 0.6): 100% Equity
                    # Weak Bull (ML < 0.6): 50% Equity
                    target_exposure = 1.0
                    if HYPER_MODE and ml_super_bull:
                        target_exposure = 1.5 # 1.5x Leverage if Super Bull
                    elif ml_bullish:
                         target_exposure = 1.0 # Normal Strong Bull
                    else:
                         target_exposure = 0.5 # Weak Bull
                    
                    val_to_buy = equity * target_exposure
                    size = val_to_buy / price
                    
                    # Cap Leverage
                    max_size = (equity * LEVERAGE_CAP) / price
                    size = min(size, max_size)
                    
                    cost = (size * price) * TRANSACTION_COST
                    if cash > (size * price) + cost:
                        cash -= (size * price) + cost
                        position = size
                        entry_price = price
                        # Initial Stop: Wide (4x ATR)
                        stop_loss = price - (4.0 * atr)
                        trades.append({"date": curr_date, "action": "BUY", "price": price, "size": size, "reason": "Bull_Entry"})
            
            # SHORT SETUP
            elif regime == "BEAR" and ALLOW_SHORTS:
                # Entry: Breakdown AND Not Bullish
                if price < row["BB_Lower"] and not ml_bullish:
                    # SIZING: RISK BASED (Defensive)
                    # We don't want 100% Short exposure usually.
                    risk_amt = equity * RISK_PER_TRADE
                    stop_dist = 2.5 * atr
                    size = risk_amt / stop_dist
                    
                    # Cap at 50% Equity for Shorts
                    max_size = (equity * 0.5) / price
                    size = min(size, max_size)
                    
                    vol_val = size * price
                    cost = vol_val * TRANSACTION_COST
                    cash += vol_val - cost
                    position = -size
                    entry_price = price
                    stop_loss = price + stop_dist
                    trades.append({"date": curr_date, "action": "SHORT", "price": price, "size": size, "reason": "Bear_Breakout"})

        equity_curve.append({"date": curr_date, "equity": equity, "price": price, "regime": regime})

    # =========================
    # VISUALIZATION & METRICS
    # =========================
    res = pd.DataFrame(equity_curve).set_index("date")
    res["BuyHold"] = res["price"] / res["price"].iloc[0] * INITIAL_CAPITAL
    
    final_eq = res["equity"].iloc[-1]
    total_ret = (final_eq / INITIAL_CAPITAL) - 1
    
    # Drawdown
    res["DD"] = (res["equity"] - res["equity"].cummax()) / res["equity"].cummax()
    max_dd = res["DD"].min()
    
    # Risk Metrics (Sharpe/Sortino)
    # 4H data, assume risk-free=0
    # Annualize factor: 365 * 6 = 2190 bars per year
    returns = res["equity"].pct_change().dropna()
    mean_ret = returns.mean() * 2190
    std_ret = returns.std() * np.sqrt(2190)
    
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    
    # Sortino (Downside Deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(2190)
    sortino = mean_ret / downside_std if downside_std > 0 else 0

    # Save Trade Log
    tr_df = pd.DataFrame(trades)
    log_name = f"trade_log_{start_date}_{end_date}.csv"
    if not tr_df.empty:
        tr_df.to_csv(log_name)
    
    # --- REPORTING ---
    print(f"\n╔════════════════════════════════════════════════╗")
    print(f"║ STRATEGY RESULTS ({start_date})            ║")
    print(f"╠════════════════════════════════════════════════╣")
    print(f"║ Final Equity   : ${final_eq:,.2f}               ║")
    print(f"║ Total Return   : {total_ret*100:+.2f}%                  ║")
    print(f"║ Max Drawdown   : {max_dd*100:.2f}%                  ║")
    print(f"║ Sharpe Ratio   : {sharpe:.2f}                     ║")
    print(f"║ Sortino Ratio  : {sortino:.2f}                     ║")
    print(f"║ Total Trades   : {len(trades)}                      ║")
    print(f"╚════════════════════════════════════════════════╝")

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Ax1: Equity Curve
    ax1.plot(res.index, res["equity"], label="Strategy", color="blue", linewidth=1.5)
    ax1.plot(res.index, res["BuyHold"], label="Buy & Hold BTC", color="gray", alpha=0.5, linestyle="--")
    
    # Trades
    if not tr_df.empty:
        tr_df = tr_df.set_index("date") 
        # Need to re-join to ensure alignment if not using entire index
        # Simple scatter using existing dates
        buys = tr_df[tr_df["action"].astype(str).str.contains("BUY")]
        shorts = tr_df[tr_df["action"].astype(str).str.contains("SHORT")]
        
        # Re-calc for plotting on equity curve:
        # We need equity value at that timestamp.
        # res has equity.
        valid_buys = buys.index.intersection(res.index)
        valid_shorts = shorts.index.intersection(res.index)
        
        ax1.scatter(valid_buys, res.loc[valid_buys, "equity"], marker="^", color="lime", s=100, edgecolors='black', label="Buy", zorder=5)
        ax1.scatter(valid_shorts, res.loc[valid_shorts, "equity"], marker="v", color="red", s=100, edgecolors='black', label="Short", zorder=5)

    ax1.set_title(f"Task 2: Strategy Equity Curve ({start_date} - {end_date})", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Equity ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Ax2: Drawdown
    ax2.fill_between(res.index, res["DD"] * 100, 0, color="red", alpha=0.3, label="Drawdown %")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    
    fname = f"final_backtest_{start_date}_{end_date}.png"
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved Chart: {fname}")

# Run
run_backtest_strategy('2021-07-01', '2022-03-31')
run_backtest_strategy('2022-01-01', '2022-03-31')
