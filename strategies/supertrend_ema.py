from datetime import datetime
from db import log_strategy_sql

def strategy_supertrend_ema(candle, index_name, future_data=None, price_to_ema_ratio=None):
    """Supertrend with EMA confirmation."""
    signal = "NO TRADE"
    confidence = "Low"
    trade_type = "Intraday"
    
    # Performance tracking variables
    pnl = 0.0
    targets_hit = 0
    stoploss_count = 0
    outcome = "Pending"
    failure_reason = ""

    # Calculate EMA distance if not provided
    if price_to_ema_ratio is None and 'ema_20' in candle:
        price_to_ema_ratio = (candle['close'] / candle['ema_20'] - 1) * 100

    supertrend_value = 1  # Placeholder for actual supertrend calculation
    
    # Signal generation based on Supertrend and EMA
    if 'ema_20' in candle:
        if candle['close'] > candle['ema_20'] and supertrend_value > 0:
            signal = "BUY CALL"
            # Higher confidence when price is further above EMA (showing momentum)
            if price_to_ema_ratio and price_to_ema_ratio > 0.5:
                confidence = "High"
            else:
                confidence = "Medium"
        elif candle['close'] < candle['ema_20'] and supertrend_value < 0:
            signal = "BUY PUT"
            # Higher confidence when price is further below EMA (showing momentum)
            if price_to_ema_ratio and price_to_ema_ratio < -0.5:
                confidence = "High"
            else:
                confidence = "Medium"

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
                failure_reason = f"Price dropped below EMA, stop loss hit at {price - stop_loss}"
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
                failure_reason = f"Price moved above EMA, stop loss hit at {price + stop_loss}"
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
        "ema_20": candle['ema_20'] if 'ema_20' in candle else 0,
        "atr": candle['atr'] if 'atr' in candle else 0,
        "price_to_ema_ratio": price_to_ema_ratio if price_to_ema_ratio is not None else 0,
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
        "stoploss_count": stoploss_count
    }
