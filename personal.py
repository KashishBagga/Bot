import yfinance as yf
import pandas as pd
import numpy as np
import ta
import math



# 🚀 Settings
INDEX_SYMBOL = "^NSEI"
INTERVALS = ["1m", "3m", "5m", "15m"]
PERIOD = "15d"
SWING_LOOKBACK = 30   # Number of swings for S/R detection

# Supertrend Calculation
def calculate_supertrend(df, period=10, multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    supertrend = [True] * len(df)

    for i in range(1, len(df.index)):
        if df['close'][i] > upperband[i-1]:
            supertrend[i] = True
        elif df['close'][i] < lowerband[i-1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i-1]

            if supertrend[i] and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]

            if not supertrend[i] and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]

    df['supertrend'] = supertrend
    return df

# Fetch historical data
def fetch_data(symbol, interval, period="15d"):
    try:
        if interval in ["1m", "3m"]:
            actual_period = "5d"
        else:
            actual_period = period

        df = yf.download(symbol, interval=interval, period=actual_period, progress=False)
        
        if df.empty:
            print(f"⚠️ Warning: No data fetched for {interval} interval.")
            return None

        # Yahoo sometimes returns weird formats, fix them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip().lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]

        return df

    except Exception as e:
        print(f"Error fetching {interval} data: {e}")
        return None

# Add indicators
def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df = calculate_supertrend(df)
    return df

# Find support and resistance
def find_support_resistance(df):
    supports = []
    resistances = []

    for i in range(SWING_LOOKBACK, len(df)):
        local_min = np.min(df['low'][i-SWING_LOOKBACK:i])
        local_max = np.max(df['high'][i-SWING_LOOKBACK:i])
        supports.append(local_min)
        resistances.append(local_max)

    df = df.iloc[SWING_LOOKBACK:]
    df['support'] = supports
    df['resistance'] = resistances
    return df

# Detect volume spike
def detect_volume_spike(df):
    avg_vol = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > 1.2 * avg_vol
    return df

# Strike price calculator
def get_strikes(price):
    strike_step = 50 if INDEX_SYMBOL == "^NSEBANK" else 50
    atm = round(price / strike_step) * strike_step
    itm = atm - strike_step  # Slight ITM
    return atm, itm

# MAIN function
def main_engine():
    print(f"🚀 Fetching and processing data for {INDEX_SYMBOL}...")

    all_data = {}
    for interval in INTERVALS:
        df = fetch_data(INDEX_SYMBOL, interval, PERIOD)
        
        # 🚨 Check if df exists and has required columns
        if df is None or not all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            print(f"⚠️ Skipping {interval} interval due to missing essential data.")
            continue
        
        df = add_indicators(df)
        df = find_support_resistance(df)
        df = detect_volume_spike(df)
        all_data[interval] = df
        print(f"✅ {interval} data ready.")

    # 🚨 Check if we have usable data before proceeding
    if not all_data:
        print("❌ No valid data available to proceed.")
        exit()

    df_for_price = all_data.get('5m') or all_data.get('15m')
    latest_close = df_for_price['close'].iloc[-1]
    atm_strike, itm_strike = get_strikes(latest_close)

    print("\n🔎 Latest Market Snapshot:")
    print(f"Current Index Price: {latest_close:.2f}")
    print(f"Recommended ATM Strike: {atm_strike}")
    print(f"Recommended Slight ITM Strike: {itm_strike}")

    return all_data, atm_strike, itm_strike



def detect_trade_signals(all_data, atm_strike, itm_strike):
    print("\n🚀 Checking for Trade Signals...")

    signal_found = False

    df_5m = all_data['5m']
    df_1m = all_data['1m']

    latest = df_5m.iloc[-1]

    price = latest['close']
    support = latest['support']
    resistance = latest['resistance']
    volume_spike = latest['volume_spike']
    rsi = latest['rsi']
    macd = latest['macd']
    macd_signal = latest['macd_signal']
    ema_20 = latest['ema_20']

    action = None

    # 🔥 Buy CALL Signal
    if (price > resistance) and volume_spike and (rsi > 60) and (macd > macd_signal) and (price > ema_20):
        action = 'BUY CALL'

    # 🔥 Buy PUT Signal
    elif (price < support) and volume_spike and (rsi < 40) and (macd < macd_signal) and (price < ema_20):
        action = 'BUY PUT'

    if action:
        signal_found = True

        # Dynamic SL and Targets
        atr = latest['atr']
        stoploss = round(atr)
        target1 = round(2 * atr)
        target2 = round(3 * atr)
        target3 = round(4 * atr)

        print("\n✅ Trade Opportunity Detected:")

        print(f"Index: BANKNIFTY")
        print(f"Action: {action}")
        print(f"Current Price: {price:.2f}")
        print(f"Recommended ATM Strike: {atm_strike}")
        print(f"Recommended Slight ITM Strike: {itm_strike}")
        print(f"Suggested Stoploss: {stoploss} points")
        print(f"Target 1: {target1} points")
        print(f"Target 2: {target2} points")
        print(f"Target 3: {target3} points")
        print(f"Volume Spike: {'Yes' if volume_spike else 'No'}")

    if not signal_found:
        print("\n❌ No Trade Opportunity detected based on current market conditions.")


if __name__ == "__main__":
    all_data, atm_strike, itm_strike = main_engine()
    detect_trade_signals(all_data, atm_strike, itm_strike)
