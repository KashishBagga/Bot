def basic_failure_reason(rsi, macd, macd_signal, close, ema_20, targets_hit, outcome):
    if "Stoploss" in outcome:
        reasons = []
        if rsi < 68:
            reasons.append("Weak RSI")
        if macd < macd_signal + 8:
            reasons.append("Weak MACD crossover")
        if close < ema_20 * 1.003:
            reasons.append("Low EMA strength")
        if not reasons:
            return "Sudden reversal or volatility"
        return ", ".join(reasons)
    elif targets_hit == 0:
        return "No momentum after entry"
    return "" 