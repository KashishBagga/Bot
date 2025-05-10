from datetime import datetime
from db import log_strategy_sql

def strategy_insidebar_rsi(candle, prev_candle, index_name, future_data=None, rsi_level=None):
    """Inside bar with RSI confirmation."""
    signal = "NO TRADE"
    confidence = "Low"
    rsi_reason = macd_reason = price_reason = ""
    trade_type = "Intraday"
    
    # Performance tracking variables
    pnl = 0.0
    targets_hit = 0
    stoploss_count = 0
    outcome = "Pending"
    failure_reason = ""

    is_inside = candle['high'] < prev_candle['high'] and candle['low'] > prev_candle['low']

    # RSI levels for confidence
    rsi_level = rsi_level or (
        "Extreme Oversold" if candle['rsi'] < 30 else
        "Oversold" if candle['rsi'] < 40 else
        "Neutral" if candle['rsi'] < 60 else
        "Overbought" if candle['rsi'] < 70 else
        "Extreme Overbought"
    )

    # Signal generation logic
    if is_inside and candle['rsi'] > 60:
        signal = "BUY CALL"
        confidence = "High" if candle['rsi'] > 70 else "Medium"
        rsi_reason = f"RSI {candle['rsi']:.2f} > 60 ({rsi_level})"
        price_reason = "Inside bar pattern"
    elif is_inside and candle['rsi'] < 40:
        signal = "BUY PUT"
        confidence = "High" if candle['rsi'] < 30 else "Medium"
        rsi_reason = f"RSI {candle['rsi']:.2f} < 40 ({rsi_level})"
        price_reason = "Inside bar pattern"

    # If we have a trade signal and future data, calculate performance
    if signal != "NO TRADE" and future_data is not None and not future_data.empty:
        price = candle['close']
        
        # Calculate ATR-based stop loss and targets
        atr = candle['atr'] if 'atr' in candle else candle['high'] - candle['low']
        stop_loss = int(round(atr))
        target1 = int(round(1.5 * atr))
        target2 = int(round(2.0 * atr))
        target3 = int(round(2.5 * atr))
        
        # Check future prices to see if targets or stop loss were hit
        if signal == "BUY CALL":
            # For buy calls, check if price went up to targets or down to stop loss
            max_future_price = future_data['high'].max()
            min_future_price = future_data['low'].min()
            
            # Check if stop loss was hit
            if min_future_price <= (price - stop_loss):
                outcome = "Failure"
                pnl = -stop_loss
                stoploss_count = 1
                failure_reason = f"Stop loss hit at {price - stop_loss}"
            else:
                outcome = "Success"
                # Check which targets were hit
                if max_future_price >= (price + target1):
                    targets_hit += 1
                    pnl += target1
                if max_future_price >= (price + target2):
                    targets_hit += 1
                    pnl += (target2 - target1)
                if max_future_price >= (price + target3):
                    targets_hit += 1
                    pnl += (target3 - target2)
                
        elif signal == "BUY PUT":
            # For buy puts, check if price went down to targets or up to stop loss
            max_future_price = future_data['high'].max()
            min_future_price = future_data['low'].min()
            
            # Check if stop loss was hit
            if max_future_price >= (price + stop_loss):
                outcome = "Failure"
                pnl = -stop_loss
                stoploss_count = 1
                failure_reason = f"Stop loss hit at {price + stop_loss}"
            else:
                outcome = "Success"
                # Check which targets were hit
                if min_future_price <= (price - target1):
                    targets_hit += 1
                    pnl += target1
                if min_future_price <= (price - target2):
                    targets_hit += 1
                    pnl += (target2 - target1)
                if min_future_price <= (price - target3):
                    targets_hit += 1
                    pnl += (target3 - target2)

    # Log the strategy signal to its dedicated table
    signal_data = {
        "signal_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "index_name": index_name,
        "signal": signal,
        "price": candle['close'],
        "rsi": candle['rsi'],
        "rsi_level": rsi_level,
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
    log_strategy_sql('insidebar_rsi', signal_data)

    return {
        "signal": signal,
        "confidence": confidence,
        "rsi_reason": rsi_reason,
        "macd_reason": macd_reason,
        "price_reason": price_reason,
        "trade_type": trade_type,
        "outcome": outcome,
        "pnl": pnl,
        "targets_hit": targets_hit,
        "stoploss_count": stoploss_count
    }
