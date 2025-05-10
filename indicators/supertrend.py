import pandas as pd


def calculate_supertrend(df, period=7, multiplier=3):
    """Calculate the Supertrend indicator for a DataFrame."""
    atr = df['atr']
    hl2 = (df['high'] + df['low']) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    final_upperband = basic_upperband.copy()
    final_lowerband = basic_lowerband.copy()

    # Create new arrays for final bands
    new_final_upperband = final_upperband.copy()
    new_final_lowerband = final_lowerband.copy()

    for i in range(1, len(df)):
        if df['close'][i-1] <= final_upperband[i-1]:
            new_final_upperband[i] = min(basic_upperband[i], final_upperband[i-1])
        else:
            new_final_upperband[i] = basic_upperband[i]

        if df['close'][i-1] >= final_lowerband[i-1]:
            new_final_lowerband[i] = max(basic_lowerband[i], final_lowerband[i-1])
        else:
            new_final_lowerband[i] = basic_lowerband[i]

    supertrend = pd.Series(index=df.index, data=False)
    for i in range(1, len(df)):
        if df['close'][i] > new_final_upperband[i]:
            supertrend[i] = True
        elif df['close'][i] < new_final_lowerband[i]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i-1]

    return supertrend 