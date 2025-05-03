import pandas as pd
import ta
import pytz
from datetime import timedelta
from utils import basic_failure_reason
from db import log_backtesting_sql


def calculate_strategy_indicators(candle):
    """Calculate necessary indicators for the strategy (row-based only)."""
    body = abs(candle['close'] - candle['open'])
    full_range = candle['high'] - candle['low']
    # Use the precomputed supertrend value from the DataFrame
    supertrend = candle['supertrend']
    return body, full_range, supertrend


def generate_strategy_signal(candle):
    """Generate trading signal based on strategy criteria."""
    if (
        candle['rsi'] > 65 and
        candle['macd'] > candle['macd_signal'] + 7 and
        candle['close'] > candle['ema_20'] * 1.001
    ):
        return "BUY CALL", "High" if candle['rsi'] > 70 else "Medium"
    elif (
        candle['rsi'] < 35 and
        candle['macd'] < candle['macd_signal'] - 5 and
        candle['close'] < candle['ema_20'] * 0.999
    ):
        return "BUY PUT", "High" if candle['rsi'] < 30 else "Medium"
    return "NO TRADE", "Medium"


def execute_trade(candle, next_df, lot_size):
    """Execute trade and calculate outcomes."""
    price = candle['close']
    atr = candle['atr']
    stoploss = int(round(atr))
    target = int(round(1.5 * atr))
    target2 = int(round(2.0 * atr))
    target3 = int(round(2.5 * atr))

    low_hit = next_df['low'] <= (price - stoploss)
    high_hit1 = next_df['high'] >= (price + target)
    high_hit2 = next_df['high'] >= (price + target2)
    high_hit3 = next_df['high'] >= (price + target3)

    if low_hit.any():
        return "Stoploss Hit", -stoploss * lot_size, 0
    else:
        lots_hit = 0
        pnl = 0
        if high_hit1.any():
            pnl += target * lot_size
            lots_hit += 1
        if high_hit2.any():
            pnl += target2 * lot_size
            lots_hit += 1
        if high_hit3.any():
            pnl += target3 * lot_size
            lots_hit += 1
        return f"{lots_hit} Targets Hit", pnl, lots_hit


def log_strategy_signal(index_name, signal_data):
    """Log the strategy signal to a dedicated table."""
    log_backtesting_sql(index_name, signal_data)


def execute_supertrend_macd_rsi_ema_strategy(df, index_name, lot_size):
    last_signal = "NO TRADE"
    confirmation_counter = 0
    total_signals = 0
    successful_signals = 0
    daily_pnl = {}
    total_pnl = 0
    total_wins = 0
    total_losses = 0
    win_amount = 0
    loss_amount = 0
    targets_hit_count = {}
    stoploss_count = {}

    for idx in range(50, len(df) - 24):
        candle = df.iloc[idx]
        body, full_range, supertrend = calculate_strategy_indicators(candle)

        if full_range == 0 or body / full_range < 0.6 or supertrend < 0.5:
            continue

        ist_time_check = candle['time'].tz_localize("UTC").tz_convert("Asia/Kolkata")
        if ist_time_check.hour >= 14 and ist_time_check.minute >= 45:
            continue

        current_signal, confidence = generate_strategy_signal(candle)

        if current_signal == last_signal and current_signal != "NO TRADE":
            confirmation_counter += 1
        else:
            confirmation_counter = 0

        if confirmation_counter == 2:
            next_df = df.iloc[idx + 1: idx + 25]
            outcome, pnl, targets_hit = execute_trade(candle, next_df, lot_size)

            utc_time = candle['time'].tz_localize("UTC")
            ist_time = utc_time.astimezone(pytz.timezone("Asia/Kolkata"))
            signal_time = ist_time.strftime("%Y-%m-%d %H:%M:%S")
            date_str = ist_time.date().isoformat()

            total_pnl += pnl
            daily_pnl[date_str] = daily_pnl.get(date_str, 0) + pnl
            targets_hit_count[date_str] = targets_hit_count.get(date_str, 0) + targets_hit

            option_chain_confirmation = "Yes" if confidence == "High" else "No"

            failure_reason = basic_failure_reason(
                candle['rsi'], candle['macd'], candle['macd_signal'], candle['close'],
                candle['ema_20'], targets_hit, outcome
            )

            signal_data = {
                "signal_time": signal_time,
                "signal": current_signal,
                "price": candle['close'],
                "rsi": candle['rsi'],
                "macd": candle['macd'],
                "macd_signal": candle['macd_signal'],
                "ema_20": candle['ema_20'],
                "atr": candle['atr'],
                "confidence": confidence,
                "rsi_reason": "",
                "macd_reason": "",
                "price_reason": "",
                "trade_type": "Intraday",
                "option_chain_confirmation": option_chain_confirmation,
                "outcome": outcome,
                "pnl": pnl,
                "targets_hit": targets_hit,
                "stoploss_count": stoploss_count.get(date_str, 0),
                "failure_reason": failure_reason
            }

            log_strategy_signal(index_name, signal_data)

            confirmation_counter = 0
            total_signals += 1

        last_signal = current_signal

    accuracy = (successful_signals / total_signals * 100) if total_signals else 0
    avg_profit = (win_amount / total_wins) if total_wins else 0
    avg_loss = (loss_amount / total_losses) if total_losses else 0
    win_ratio = (total_wins / total_signals * 100) if total_signals else 0

    print(f"\n📊 Accuracy for {index_name}: {accuracy:.2f}% ({successful_signals}/{total_signals})")
    print(f"✅ Wins: {total_wins}, ❌ Losses: {total_losses}, 💰 Net P&L: ₹{total_pnl:.2f}")
    print("📈 Daily P&L Summary:")
    for d, p in daily_pnl.items():
        print(f"{d}: ₹{p:.2f}")

    return accuracy, total_pnl, total_wins, total_losses, win_amount, loss_amount, win_ratio 