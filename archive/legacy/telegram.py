import yfinance as yf
import pandas as pd
import numpy as np
import ta
import requests
import time

# 🔥 Your bot token and both group chat IDs
BOT_TOKEN = '7233653035:AAHVNm4ESq5_s9fq-qFbUNN3bHXYerpMsBw'
FREE_CHAT_ID = '-4760811451'   # Free group
VIP_CHAT_ID = '-4753610116'    # (use another id when VIP group ready)

def fetch_data(symbol):
    try:
        data = yf.download(tickers=symbol, period='5d', interval='5m', progress=False)
        if data.empty:
            raise ValueError(f"No data fetched for {symbol}")
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def detect_columns(data):
    close_col = [col for col in data.columns if 'close' in col.lower()][0]
    high_col = [col for col in data.columns if 'high' in col.lower()][0]
    low_col = [col for col in data.columns if 'low' in col.lower()][0]
    open_col = [col for col in data.columns if 'open' in col.lower()][0]
    volume_col = [col for col in data.columns if 'volume' in col.lower()][0] if any('volume' in col.lower() for col in data.columns) else None
    return open_col, high_col, low_col, close_col, volume_col

def calculate_indicators(data, open_col, high_col, low_col, close_col, volume_col):
    if len(data) < 50:
        raise ValueError("Not enough data to calculate indicators")

    data['rsi'] = ta.momentum.RSIIndicator(data[close_col], window=14).rsi()
    macd = ta.trend.MACD(data[close_col])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['ema_20'] = ta.trend.EMAIndicator(data[close_col], window=20).ema_indicator()
    data['atr'] = ta.volatility.AverageTrueRange(high=data[high_col], low=data[low_col], close=data[close_col], window=14).average_true_range()

    data = data.dropna()  # 🚀 Drop NaN rows after indicators calculated
    return data

def send_telegram_signal(message, bot_token, chat_id):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print(f"✅ Signal sent to Telegram Chat ID {chat_id}")
        else:
            print(f"Failed to send Telegram Signal: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram signal: {e}")

# def generate_signal(data, close_col, index_name, bot_token, free_chat_id, vip_chat_id):
    latest = data.iloc[-1]

    signal = "NO TRADE"
    if (
        latest['rsi'] > 60 
        and latest['macd'] > latest['macd_signal'] 
        and latest[close_col] > latest['ema_20']
    ):
        signal = "BUY CALL"
    elif (
        latest['rsi'] < 40 
        and latest['macd'] < latest['macd_signal'] 
        and latest[close_col] < latest['ema_20']
    ):
        signal = "BUY PUT"

    if signal != "NO TRADE":
        price = latest[close_col]
        atr = latest['atr']
        strike_price = int(round(price / 50) * 50)
        stoploss = int(round(atr))
        target = int(round(2 * atr))

        # Free group message
        free_message = f"""
🚀 {index_name} - {signal}
Strike Price: {strike_price}
Stoploss: {stoploss} points
Target: {target} points
"""
        print(free_message)
        send_telegram_signal(free_message, bot_token, free_chat_id)

        # VIP group message (only if RSI stronger or MACD gap big)
        if (latest['rsi'] > 65 or latest['rsi'] < 35) and abs(latest['macd'] - latest['macd_signal']) > 5:
            vip_message = f"""
🌟 VIP SIGNAL 🌟
{index_name} - {signal}
ATM Strike: {strike_price}
Stoploss: {stoploss} pts
Target: {target} pts
Confidence Level: HIGH 🔥
"""
            send_telegram_signal(vip_message, bot_token, vip_chat_id)

    else:
        print(f"No Trade Signal for {index_name} currently.")

def generate_signal(data_5m, data_15m, close_col_5m, close_col_15m, index_name, bot_token, free_chat_id, vip_chat_id):
    latest_5m = data_5m.iloc[-1]
    latest_15m = data_15m.iloc[-1]

    signal = "NO TRADE"
    if (
        latest_5m['rsi'] > 60 
        and latest_5m['macd'] > latest_5m['macd_signal'] 
        and latest_5m[close_col_5m] > latest_5m['ema_20']
    ):
        signal = "BUY CALL"
    elif (
        latest_5m['rsi'] < 40 
        and latest_5m['macd'] < latest_5m['macd_signal'] 
        and latest_5m[close_col_5m] < latest_5m['ema_20']
    ):
        signal = "BUY PUT"

    if signal != "NO TRADE":
        price = latest_5m[close_col_5m]
        atr = latest_5m['atr']
        strike_price = int(round(price / 50) * 50)
        stoploss = int(round(atr))
        target = int(round(2 * atr))

        # 🚀 Always send Free Signal
        free_message = f"""
🚀 {index_name} - {signal}
Strike Price: {strike_price}
Stoploss: {stoploss} points
Target: {target} points
"""
        print(free_message)
        send_telegram_signal(free_message, bot_token, free_chat_id)

        # 🚀 Send VIP High Signal (5-min only confirmation)
        vip_message_high = f"""
🌟 HIGH SIGNAL 🌟
{index_name} - {signal}
ATM Strike: {strike_price}
Stoploss: {stoploss} pts
Target: {target} pts
5-Min Breakout Confirmed ✅
Confidence Level: HIGH 🔥
"""
        send_telegram_signal(vip_message_high, bot_token, vip_chat_id)

        # 🚀 If 15-min also confirms, send VIP Very High Signal
        if (
            (signal == "BUY CALL" and latest_15m['rsi'] > 60 and latest_15m['macd'] > latest_15m['macd_signal'])
            or
            (signal == "BUY PUT" and latest_15m['rsi'] < 40 and latest_15m['macd'] < latest_15m['macd_signal'])
        ):
            vip_message_very_high = f"""
🌟🌟 VERY HIGH SIGNAL 🌟🌟
{index_name} - {signal}
ATM Strike: {strike_price}
Stoploss: {stoploss} pts
Target: {target} pts
5-Min + 15-Min Strong Confirmed ✅✅
Confidence Level: VERY HIGH 🔥🚀
"""
            send_telegram_signal(vip_message_very_high, bot_token, vip_chat_id)

    else:
        print(f"No Trade Signal for {index_name} currently.")

# def run_bot():
#     symbols = {
#         "^NSEI": "NIFTY50",
#         "^NSEBANK": "BANKNIFTY"
#     }

#     for symbol, index_name in symbols.items():
#         data = fetch_data(symbol)
#         if not data.empty:
#             try:
#                 if isinstance(data.columns, pd.MultiIndex):
#                     data.columns = ['_'.join(col).strip().lower() for col in data.columns]
#                 else:
#                     data.columns = [col.lower() for col in data.columns]

#                 open_col, high_col, low_col, close_col, volume_col = detect_columns(data)
#                 data = calculate_indicators(data, open_col, high_col, low_col, close_col, volume_col)
#                 generate_signal(data, close_col, index_name, BOT_TOKEN, FREE_CHAT_ID, VIP_CHAT_ID)

#             except Exception as e:
#                 print(f"Error processing {index_name}: {e}")

def run_bot():
    symbols = {
        "^NSEI": "NIFTY50",
        "^NSEBANK": "BANKNIFTY"
    }

    for symbol, index_name in symbols.items():
        try:
            # 🚀 Fetch 5-min data
            data_5m = fetch_data(symbol)
            if isinstance(data_5m.columns, pd.MultiIndex):
                data_5m.columns = ['_'.join(col).strip().lower() for col in data_5m.columns]
            else:
                data_5m.columns = [col.lower() for col in data_5m.columns]

            open_5m, high_5m, low_5m, close_5m, volume_5m = detect_columns(data_5m)
            data_5m = calculate_indicators(data_5m, open_5m, high_5m, low_5m, close_5m, volume_5m)

            # 🚀 Fetch 15-min data
            data_15m = yf.download(tickers=symbol, period='5d', interval='15m', progress=False)
            if isinstance(data_15m.columns, pd.MultiIndex):
                data_15m.columns = ['_'.join(col).strip().lower() for col in data_15m.columns]
            else:
                data_15m.columns = [col.lower() for col in data_15m.columns]

            open_15m, high_15m, low_15m, close_15m, volume_15m = detect_columns(data_15m)
            data_15m = calculate_indicators(data_15m, open_15m, high_15m, low_15m, close_15m, volume_15m)

            # 🚀 Generate Signal based on 5m + 15m
            generate_signal(data_5m, data_15m, close_5m, close_15m, index_name, BOT_TOKEN, FREE_CHAT_ID, VIP_CHAT_ID)

        except Exception as e:
            print(f"Error processing {index_name}: {e}")

if __name__ == "__main__":
    print("🚀 Bot Started! Auto-checking every 5 minutes...")

    while True:
        run_bot()
        print("✅ Waiting 5 minutes before next check...\n")
        time.sleep(300)  # 300 seconds = 5 minutes
