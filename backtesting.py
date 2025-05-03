import requests
import pandas as pd
import ta
import time
from fyers_apiv3 import fyersModel
import webbrowser
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import backoff
from dataclasses import dataclass
import json
import os
from dotenv import load_dotenv
import math
import pytz
import schedule
import threading
from db import log_trade_sql, log_backtesting_sql
from strategies.supertrend_macd_rsi_ema import execute_supertrend_macd_rsi_ema_strategy
from utils import basic_failure_reason

"""
In order to get started with Fyers API we would like you to do the following things first.
1. Checkout our API docs :   https://myapi.fyers.in/docsv3
2. Create an APP using our API dashboard :   https://myapi.fyers.in/dashboard/

Once you have created an APP you can start using the below SDK 
"""

# Load environment variables
load_dotenv()

# Get Fyers API credentials from environment variables
redirect_uri = os.getenv("FYERS_REDIRECT_URI")
client_id = os.getenv("FYERS_CLIENT_ID")
secret_key = os.getenv("FYERS_SECRET_KEY")
grant_type = os.getenv("FYERS_GRANT_TYPE")
response_type = os.getenv("FYERS_RESPONSE_TYPE")
state = os.getenv("FYERS_STATE")
auth_code = os.getenv("FYERS_AUTH_CODE")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the trading bot"""
    client_id: str = os.getenv('FYERS_CLIENT_ID')
    secret_key: str = os.getenv('FYERS_SECRET_KEY')
    redirect_uri: str = os.getenv('FYERS_REDIRECT_URI')
    grant_type: str = os.getenv('FYERS_GRANT_TYPE')
    response_type: str = os.getenv('FYERS_RESPONSE_TYPE')
    state: str = os.getenv('FYERS_STATE')
    symbols: Dict[str, str] = None
    resolution: str = "5"
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_period: int = 20
    atr_period: int = 14
    confirmation_count: int = 2
    rsi_upper: float = 65
    rsi_lower: float = 35
    macd_threshold: float = 5
    price_threshold: float = 0.001

    def __post_init__(self):
        # Validate required environment variables
        required_vars = ['client_id', 'secret_key', 'redirect_uri', 'grant_type', 'response_type', 'state']
        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        if self.symbols is None:
            self.symbols = {
                "NSE:NIFTY50-INDEX": "NIFTY50",
                "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
            }

# Initialize Fyers session
appSession = fyersModel.SessionModel(
    client_id=client_id,
    redirect_uri=redirect_uri,
    response_type=response_type,
    state=state,
    secret_key=secret_key,
    grant_type=grant_type
)

generateTokenUrl = appSession.generate_authcode()
print((generateTokenUrl))  
# webbrowser.open(generateTokenUrl,new=1)

appSession.set_token(auth_code)
response = appSession.generate_token()

try: 
    access_token = response["access_token"]
except Exception as e:
    print(e,response)  

fyers = fyersModel.FyersModel(token=access_token, is_async=False, client_id=client_id, log_path="")
# print(fyers.get_profile()) 

# # CONNECTION ESTABLISHED ABOVE

# 🔥 Fetch candles
def fetch_candles(symbol, fyers, resolution="5", range_from="2025-04-02", range_to="2025-05-02"):
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1"
    }
    response = fyers.history(data)
    candles = response.get('candles')
    if not candles:
        raise Exception(f"No candle data fetched for {symbol}")
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# 🔥 Calculate indicators
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

def calculate_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df = calculate_supertrend(df)
    return df

# 🔥 Generate and log signals
def is_volume_spike(candle, df, idx):
    if idx < 20:
        return False
    avg_volume = df['volume'][idx-20:idx].mean()
    return candle['volume'] > avg_volume * 1.5

def is_long_wick_candle(candle):
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['close'], candle['open'])
    lower_wick = min(candle['close'], candle['open']) - candle['low']
    return upper_wick > body or lower_wick > body

def generate_signal_all(df, index_name, lot_size):
    return execute_supertrend_macd_rsi_ema_strategy(df, index_name, lot_size)



def run_bot(fyers):
    symbols = {
        "NSE:NIFTY50-INDEX": "NIFTY50",
        "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
    }

    lot_sizes = {
        "NIFTY50": 50,
        "BANKNIFTY": 15
    }

    for symbol, index_name in symbols.items():
        try:
            df = fetch_candles(symbol, fyers)
            if df.empty:
                print(f"❗ No candles available for {index_name}")
                continue
            # Debugging: Check the type of df['close'] before calculating indicators
            print(f"Type of df['close'] before indicators: {type(df['close'])}")
            df = calculate_indicators(df)
            # Debugging: Check the type of df['close'] after calculating indicators
            print(f"Type of df['close'] after indicators: {type(df['close'])}")
            accuracy, total_pnl, total_wins, total_losses, win_amount, loss_amount, win_ratio = generate_signal_all(
                df,
                index_name,
                lot_sizes[index_name]
            )
            print(f"✅✅✅ Historical Data Processing Finished for {index_name} ✅✅✅")
        except Exception as e:
            print(f"❌ Error processing {index_name}: {e}")

# 🔥 Entry point
if __name__ == "__main__":
    print("🚀 Bot Started - Auto Refresh Every 5 Minutes!")

    run_bot(fyers)
