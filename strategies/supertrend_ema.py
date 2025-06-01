from datetime import datetime
from db import log_strategy_sql
import numpy as np
import pandas as pd
from indicators.supertrend import Supertrend, calculate_supertrend_live

def strategy_supertrend_ema(candle, index_name, future_data=None, price_to_ema_ratio=None):
    """Supertrend with EMA confirmation."""
    signal = "NO TRADE"
    confidence = "Low"
    trade_type = "Intraday"
    pnl = 0.0
    targets_hit = 0
    stoploss_count = 0
    outcome = "Pending"
    failure_reason = ""

    supertrend_instance = get_supertrend_instance(index_name)
    supertrend_data = supertrend_instance.update(candle)
    
    if price_to_ema_ratio is None and 'ema_20' in candle:
        price_to_ema_ratio = (candle['close'] / candle['ema_20'] - 1) * 100

    # Use the Supertrend instance to maintain state
    supertrend_value = supertrend_data['value']
    supertrend_direction = supertrend_data['direction']
    
    # Enhanced signal generation with band confirmation
    if 'ema_20' in candle:
        if candle['close'] > candle['ema_20'] and supertrend_direction > 0:
            signal = "BUY CALL"
            # Higher confidence when price is above both EMA and upper band
            if (price_to_ema_ratio > 0.5 and 
                candle['close'] > supertrend_value and 
                candle['close'] > supertrend_data['upperband']):
                confidence = "High"
            else:
                confidence = "Medium"
        elif candle['close'] < candle['ema_20'] and supertrend_direction < 0:
            signal = "BUY PUT"
            # Higher confidence when price is below both EMA and lower band
            if (price_to_ema_ratio < -0.5 and 
                candle['close'] < supertrend_value and 
                candle['close'] < supertrend_data['lowerband']):
                confidence = "High"
            else:
                confidence = "Medium"

    if signal != "NO TRADE" and future_data is not None and not future_data.empty:
        price = candle['close']
        atr = candle['atr'] if 'atr' in candle else candle['high'] - candle['low']
        stop_loss = int(round(atr))
        target1 = int(round(1.5 * atr))
        target2 = int(round(2.0 * atr))
        target3 = int(round(2.5 * atr))

        for _, future_candle in future_data.iterrows():
            high = future_candle['high']
            low = future_candle['low']
            if signal == "BUY CALL":
                if low <= (price - stop_loss):
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Price dropped to SL at {price - stop_loss}"
                    break
                if high >= (price + target1) and targets_hit == 0:
                    targets_hit = 1
                    pnl = target1
                if high >= (price + target2) and targets_hit == 1:
                    targets_hit = 2
                    pnl = target2
                if high >= (price + target3) and targets_hit == 2:
                    targets_hit = 3
                    pnl = target3
                    outcome = "Win"
                    break
            elif signal == "BUY PUT":
                if high >= (price + stop_loss):
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Price hit SL at {price + stop_loss}"
                    break
                if low <= (price - target1) and targets_hit == 0:
                    targets_hit = 1
                    pnl = target1
                if low <= (price - target2) and targets_hit == 1:
                    targets_hit = 2
                    pnl = target2
                if low <= (price - target3) and targets_hit == 2:
                    targets_hit = 3
                    pnl = target3
                    outcome = "Win"
                    break
        if outcome == "Pending":
            outcome = "Win" if targets_hit > 0 else "Pending"

    # Enhanced signal data with Supertrend bands
    signal_data = {
        "signal_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "index_name": index_name,
        "signal": signal,
        "price": candle['close'],
        "ema_20": candle.get('ema_20', 0),
        "atr": candle.get('atr', 0),
        "price_to_ema_ratio": price_to_ema_ratio or 0,
        "confidence": confidence,
        "trade_type": trade_type,
        "stop_loss": stop_loss if 'stop_loss' in locals() else 0,
        "target": target1 if 'target1' in locals() else 0,
        "target2": target2 if 'target2' in locals() else 0,
        "target3": target3 if 'target3' in locals() else 0,
        "outcome": outcome,
        "pnl": pnl,
        "targets_hit": targets_hit,
        "stoploss_count": stoploss_count,
        "failure_reason": failure_reason,
        "supertrend_value": supertrend_value,
        "supertrend_direction": supertrend_direction,
        "supertrend_upperband": supertrend_data['upperband'],
        "supertrend_lowerband": supertrend_data['lowerband']
    }
    log_strategy_sql('supertrend_ema', signal_data)

    return {
        "signal": signal,
        "confidence": confidence,
        "trade_type": trade_type,
        "outcome": outcome,
        "pnl": pnl,
        "targets_hit": targets_hit,
        "stoploss_count": stoploss_count,
        "supertrend_value": supertrend_value,
        "supertrend_direction": supertrend_direction,
        "supertrend_upperband": supertrend_data['upperband'],
        "supertrend_lowerband": supertrend_data['lowerband']
    }
