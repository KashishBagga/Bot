from datetime import datetime
from db import log_strategy_sql
import numpy as np
import pandas as pd

def calculate_supertrend(candle, prev_candle=None, prev_supertrend=None, period=7, multiplier=3):
    tr1 = abs(candle['high'] - candle['low'])
    tr2 = abs(candle['high'] - candle['close'])
    tr3 = abs(candle['low'] - candle['close'])
    true_range = max(tr1, tr2, tr3)
    atr = candle.get('atr', true_range)

    hl2 = (candle['high'] + candle['low']) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)

    final_upperband = basic_upperband
    final_lowerband = basic_lowerband

    if prev_candle and prev_supertrend:
        if prev_candle['close'] <= prev_supertrend['upperband']:
            final_upperband = min(basic_upperband, prev_supertrend['upperband'])
        if prev_candle['close'] >= prev_supertrend['lowerband']:
            final_lowerband = max(basic_lowerband, prev_supertrend['lowerband'])

    if candle['close'] > final_upperband:
        direction = 1
        supertrend = final_lowerband
    elif candle['close'] < final_lowerband:
        direction = -1
        supertrend = final_upperband
    else:
        if prev_supertrend:
            direction = prev_supertrend['direction']
            supertrend = prev_supertrend['value']
        else:
            direction = 1 if candle['close'] > (final_upperband + final_lowerband) / 2 else -1
            supertrend = final_lowerband if direction == 1 else final_upperband

    return {
        "value": supertrend,
        "direction": direction,
        "upperband": final_upperband,
        "lowerband": final_lowerband
    }

def strategy_supertrend_ema(candle, index_name, future_data=None, price_to_ema_ratio=None):
    signal = "NO TRADE"
    confidence = "Low"
    trade_type = "Intraday"
    pnl = 0.0
    targets_hit = 0
    stoploss_count = 0
    outcome = "Pending"
    failure_reason = ""

    if price_to_ema_ratio is None and 'ema_20' in candle:
        price_to_ema_ratio = (candle['close'] / candle['ema_20'] - 1) * 100

    supertrend_data = calculate_supertrend(
        candle=candle
    )
    supertrend_value = supertrend_data['value']
    supertrend_direction = supertrend_data['direction']

    if 'ema_20' in candle:
        if candle['close'] > candle['ema_20'] and supertrend_direction > 0:
            signal = "BUY CALL"
            if price_to_ema_ratio > 0.5 and candle['close'] > supertrend_value:
                confidence = "High"
            else:
                confidence = "Medium"
        elif candle['close'] < candle['ema_20'] and supertrend_direction < 0:
            signal = "BUY PUT"
            if price_to_ema_ratio < -0.5 and candle['close'] < supertrend_value:
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
        "failure_reason": failure_reason
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
        "supertrend_direction": supertrend_direction
    }
