import pandas as pd


def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Calculate the MACD indicator for a DataFrame."""
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal 