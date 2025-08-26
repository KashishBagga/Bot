import pandas as pd

def calculate_macd(
    df: pd.DataFrame,
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9
):
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()

    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    hist = macd - signal

    return macd, signal, hist
