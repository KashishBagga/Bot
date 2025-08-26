import pandas as pd

def calculate_ema(series: pd.Series, span: int = 20) -> pd.Series:
    """
    Exponential Moving Average (EMA)
    """
    return series.ewm(span=span, adjust=False).mean()
