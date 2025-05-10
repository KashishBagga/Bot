import pandas as pd


def calculate_ema(series, span=20):
    """Calculate the Exponential Moving Average (EMA) of a series."""
    return series.ewm(span=span, adjust=False).mean() 