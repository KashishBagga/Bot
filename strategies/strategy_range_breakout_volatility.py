from datetime import datetime
from db import log_strategy_sql

def strategy_range_breakout_volatility(candle, range_high, range_low, index_name, future_data=None, volatility_rank=None, range_width=None, breakout_size=None):
    signal = "NO TRADE"
    confidence = "Low"
    rsi_reason = macd_reason = price_reason = ""
    trade_type = "Intraday"
    option_chain_confirmation = "No"

    if candle['close'] > range_high and candle['atr'] > 10:
        signal = "BUY CALL"
        confidence = "High"
        price_reason = "Breakout with high ATR"
    elif candle['close'] < range_low and candle['atr'] > 10:
        signal = "BUY PUT"
        confidence = "High"
        price_reason = "Breakdown with high ATR"

    # Log the strategy signal to its dedicated table
    signal_data = {
        "signal_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "index_name": index_name,
        "signal": signal,
        "price": candle['close'],
        "atr": candle['atr'] if 'atr' in candle else 0,
        "volatility_rank": volatility_rank if volatility_rank is not None else 0,
        "range_width": range_width if range_width is not None else (range_high - range_low),
        "breakout_size": breakout_size if breakout_size is not None else 0,
        "confidence": confidence,
        "price_reason": price_reason,
        "trade_type": trade_type,
        "outcome": "Pending",
        "pnl": 0.0,
        "targets_hit": 0,
        "stoploss_count": 0,
        "failure_reason": ""
    }
    log_strategy_sql('range_breakout_volatility', signal_data)

    return {
        "signal": signal,
        "confidence": confidence,
        "rsi_reason": rsi_reason,
        "macd_reason": macd_reason,
        "price_reason": price_reason,
        "trade_type": trade_type,
        "option_chain_confirmation": option_chain_confirmation
    }
