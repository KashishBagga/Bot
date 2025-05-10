from datetime import datetime
from db import log_strategy_sql

def strategy_insidebar_bollinger(candle, prev_candle, index_name, future_data=None, bollinger_width=None, price_to_band_ratio=None, inside_bar_size=None):
    signal = "NO TRADE"
    confidence = "Low"
    rsi_reason = macd_reason = price_reason = ""
    trade_type = "Intraday"
    option_chain_confirmation = "No"

    is_inside = candle['high'] < prev_candle['high'] and candle['low'] > prev_candle['low']
    
    if is_inside and candle['close'] < candle['bollinger_lower']:
        signal = "BUY CALL"
        confidence = "Medium"
        price_reason = "Inside bar near lower Bollinger Band"
    elif is_inside and candle['close'] > candle['bollinger_upper']:
        signal = "BUY PUT"
        confidence = "Medium"
        price_reason = "Inside bar near upper Bollinger Band"

    # Log the strategy signal to its dedicated table
    signal_data = {
        "signal_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "index_name": index_name,
        "signal": signal,
        "price": candle['close'],
        "bollinger_width": bollinger_width if bollinger_width is not None else 0,
        "price_to_band_ratio": price_to_band_ratio if price_to_band_ratio is not None else 0,
        "inside_bar_size": inside_bar_size if inside_bar_size is not None else 0,
        "confidence": confidence,
        "price_reason": price_reason,
        "trade_type": trade_type,
        "outcome": "Pending",
        "pnl": 0.0,
        "targets_hit": 0,
        "stoploss_count": 0,
        "failure_reason": ""
    }
    log_strategy_sql('insidebar_bollinger', signal_data)

    return {
        "signal": signal,
        "confidence": confidence,
        "rsi_reason": rsi_reason,
        "macd_reason": macd_reason,
        "price_reason": price_reason,
        "trade_type": trade_type,
        "option_chain_confirmation": option_chain_confirmation
    }
