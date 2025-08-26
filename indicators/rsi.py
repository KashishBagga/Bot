import pandas as pd

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Prevent division by zero
    loss = loss.replace(0, 1e-10)

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Fill early NaNs with neutral RSI
    return rsi.fillna(50)
