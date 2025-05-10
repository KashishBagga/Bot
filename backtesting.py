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
from strategies.strategy_breakout_rsi import strategy_breakout_rsi
from strategies.strategy_ema_crossover import strategy_ema_crossover
from strategies.strategy_donchian_breakout import strategy_donchian_breakout
from strategies.insidebar_rsi import strategy_insidebar_rsi
from strategies.supertrend_ema import strategy_supertrend_ema
from strategies.strategy_insidebar_bollinger import strategy_insidebar_bollinger
from strategies.strategy_range_breakout_volatility import strategy_range_breakout_volatility
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
access_token = os.getenv("FYERS_ACCESS_TOKEN")

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
            # Updated to use the correct symbol format based on testing
            self.symbols = {
                "NSE:NIFTY50-INDEX": "NIFTY50",
                "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
            }

# Initialize Fyers session
if not access_token:
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
        print(f"Error generating token: {e}")
        print(f"Response: {response}")
        exit(1)
else:
    print("Using access token from .env file")

fyers = fyersModel.FyersModel(token=access_token, is_async=False, client_id=client_id, log_path="")
# print(fyers.get_profile()) 

# # CONNECTION ESTABLISHED ABOVE

# 🔥 Fetch candles
@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def fetch_candles(symbol, fyers, resolution="5", range_from="2025-04-02", range_to="2025-05-02"):
    """
    Fetch historical candle data from Fyers API with improved error handling
    
    Args:
        symbol: The symbol to fetch data for (e.g., "NSE:NIFTY50-INDEX")
        fyers: The Fyers API client
        resolution: The timeframe resolution (1, 5, 15, D, W, M)
        range_from: Start date in YYYY-MM-DD format or timestamp
        range_to: End date in YYYY-MM-DD format or timestamp
    
    Returns:
        DataFrame with OHLCV data
    """
    # Set default date range if not provided - use a shorter recent period
    if not range_from:
        range_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not range_to:
        range_to = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching data for {symbol} from {range_from} to {range_to} with resolution {resolution}")
    
    # Try a simpler approach with direct date strings
    try:
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",  # Unix timestamp
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1"
        }
        
        print(f"Request data: {data}")
        response = fyers.history(data)
        
        # Check if response contains error
        if isinstance(response, dict) and 'code' in response and response['code'] != 200:
            error_code = response.get('code')
            error_message = response.get('message', 'Unknown error')
            print(f"API Error: {error_code} - {error_message}")
            
            # Try a shorter date range (last 7 days)
            print("Trying with a shorter date range (last 7 days)...")
            short_range_from = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            short_range_to = datetime.now().strftime('%Y-%m-%d')
            
            data = {
                "symbol": symbol,
                "resolution": resolution,
                "date_format": "1",
                "range_from": short_range_from,
                "range_to": short_range_to,
                "cont_flag": "1"
            }
            
            print(f"New request data: {data}")
            response = fyers.history(data)
            
            if isinstance(response, dict) and 'code' in response and response['code'] != 200:
                raise Exception(f"Failed to fetch data: {response.get('message')}")
        
        # Check if candles exist in the response
        candles = response.get('candles')
        if not candles:
            raise Exception(f"No candle data fetched for {symbol}")
            
        print(f"✅ Successfully fetched {len(candles)} candles for {symbol}")
        df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
        
    except Exception as e:
        print(f"❗ Error fetching candles: {e}")
        raise

# 🔥 Generate and log signals
def generate_signal_all(df, index_name, lot_size):
    # Pre-calculate indicators needed for all strategies
    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    
    # Calculate RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # Calculate MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Pre-calculate Bollinger Bands for insidebar_bollinger strategy
    bb = ta.volatility.BollingerBands(df['close'])
    df['bollinger_upper'] = bb.bollinger_hband()
    df['bollinger_lower'] = bb.bollinger_lband()
    df['bollinger_mid'] = bb.bollinger_mavg()
    
    # Pre-calculate ATR for stop-loss and target calculations
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # Calculate volatility rank (percentile of current ATR relative to recent history)
    # Use a safer approach that doesn't rely on negative indexing
    df['volatility_rank'] = 50.0  # Default value
    for i in range(20, len(df)):
        window = df['atr'].iloc[i-20:i]
        if window.max() > window.min():
            df.loc[df.index[i], 'volatility_rank'] = (df['atr'].iloc[i] - window.min()) / (window.max() - window.min()) * 100

    # Execute each strategy
    strategies = [
        ('supertrend_macd_rsi_ema', execute_supertrend_macd_rsi_ema_strategy),
        ('breakout_rsi', strategy_breakout_rsi),
        ('ema_crossover', strategy_ema_crossover),
        ('donchian_breakout', strategy_donchian_breakout),
        ('insidebar_rsi', strategy_insidebar_rsi),
        ('supertrend_ema', strategy_supertrend_ema),
        ('insidebar_bollinger', strategy_insidebar_bollinger),
        ('range_breakout_volatility', strategy_range_breakout_volatility)
    ]

    # Call DataFrame-wide strategies only once
    for strategy_name, strategy_function in strategies:
        if strategy_name == 'supertrend_macd_rsi_ema':
            strategy_function(df, index_name, lot_size)

    # For all other strategies, loop over each row
    for idx in range(len(df) - 20):  # Leave room for future data analysis
        candle = df.iloc[idx]
        prev_candle = df.iloc[idx - 1] if idx > 0 else candle
        prev_high = df['high'].shift(1).iloc[idx] if idx > 0 else candle['high']
        prev_low = df['low'].shift(1).iloc[idx] if idx > 0 else candle['low']
        high_20 = df['high'].rolling(window=20).max().iloc[idx]
        low_20 = df['low'].rolling(window=20).min().iloc[idx]
        
        # Get future data for analyzing stop-loss/target hits
        future_data = df.iloc[idx+1:idx+20] if idx < len(df) - 20 else df.iloc[idx+1:]
        
        # Enhanced strategy calls with additional metrics
        
        # Breakout RSI strategy
        if idx > 0:  # Ensure we have previous data for breakout comparison
            breakout_strength = ((candle['high'] - prev_high) / prev_high * 100) if candle['high'] > prev_high else \
                               ((candle['low'] - prev_low) / prev_low * 100) if candle['low'] < prev_low else 0
            rsi_alignment = "Confirming" if (candle['high'] > prev_high and candle['rsi'] > df['rsi'].iloc[idx-1]) or \
                           (candle['low'] < prev_low and candle['rsi'] < df['rsi'].iloc[idx-1]) else "Diverging"
            strategy_breakout_rsi(candle, prev_high, prev_low, index_name, future_data=future_data, 
                              breakout_strength=breakout_strength, rsi_alignment=rsi_alignment)
        
        # EMA crossover strategy
        if idx > 0:  # Need previous data for crossover detection
            crossover_strength = abs(candle['ema_9'] - candle['ema_21']) / candle['ema_21'] * 100
            momentum = "Strong" if abs(candle['close'] - candle['open']) > candle['atr'] else "Moderate"
            strategy_ema_crossover(candle, index_name, future_data=future_data,
                               crossover_strength=crossover_strength, momentum=momentum)
        
        # Donchian breakout strategy
        channel_width = high_20 - low_20
        breakout_size = ((candle['close'] - high_20) / high_20 * 100) if candle['close'] > high_20 else \
                        ((candle['close'] - low_20) / low_20 * 100) if candle['close'] < low_20 else 0
        volume_ratio = candle['volume'] / df['volume'].rolling(window=10).mean().iloc[idx] \
                     if 'volume' in df.columns and idx >= 10 else 1.0
        strategy_donchian_breakout(candle, high_20, low_20, index_name, future_data=future_data,
                               channel_width=channel_width, breakout_size=breakout_size, volume_ratio=volume_ratio)
        
        # Inside bar RSI strategy
        if idx > 0:  # Need previous candle for inside bar pattern
            rsi_level = "Extreme Oversold" if candle['rsi'] < 30 else \
                       "Oversold" if candle['rsi'] < 40 else \
                       "Neutral" if candle['rsi'] < 60 else \
                       "Overbought" if candle['rsi'] < 70 else \
                       "Extreme Overbought"
            strategy_insidebar_rsi(candle, prev_candle, index_name, future_data=future_data, rsi_level=rsi_level)
        
        # Supertrend EMA strategy
        if 'ema_20' in candle:
            price_to_ema_ratio = (candle['close'] / candle['ema_20'] - 1) * 100  # % distance from EMA
            strategy_supertrend_ema(candle, index_name, future_data=future_data, price_to_ema_ratio=price_to_ema_ratio)
        
        # Inside bar Bollinger strategy
        if idx > 0 and 'bollinger_upper' in candle and 'bollinger_lower' in candle:
            bollinger_width = (candle['bollinger_upper'] - candle['bollinger_lower']) / candle['bollinger_mid'] * 100
            price_to_band_ratio = 100 * (candle['close'] - candle['bollinger_lower']) / \
                               (candle['bollinger_upper'] - candle['bollinger_lower']) if \
                               candle['bollinger_upper'] > candle['bollinger_lower'] else 50
            inside_bar_size = (candle['high'] - candle['low']) / (prev_candle['high'] - prev_candle['low']) * 100 \
                            if (prev_candle['high'] - prev_candle['low']) > 0 else 100
            strategy_insidebar_bollinger(candle, prev_candle, index_name, future_data=future_data,
                                     bollinger_width=bollinger_width, price_to_band_ratio=price_to_band_ratio,
                                     inside_bar_size=inside_bar_size)
        
        # Range breakout volatility strategy
        if 'atr' in candle and 'volatility_rank' in candle:
            range_width = high_20 - low_20
            breakout_size = ((candle['close'] - high_20) / high_20 * 100) if candle['close'] > high_20 else \
                           ((candle['close'] - low_20) / low_20 * 100) if candle['close'] < low_20 else 0
            strategy_range_breakout_volatility(candle, high_20, low_20, index_name, future_data=future_data,
                                           volatility_rank=candle['volatility_rank'], range_width=range_width,
                                           breakout_size=breakout_size)

    # Calculate performance metrics for the entire backtest
    successful_signals = sum(1 for i in range(len(df)) if 'signal' in df.iloc[i] and df.iloc[i]['signal'] != "NO TRADE")
    total_pnl = sum(df['pnl'].fillna(0)) if 'pnl' in df.columns else 0
    total_targets_hit = sum(df['targets_hit'].fillna(0)) if 'targets_hit' in df.columns else 0
    total_stoploss_hit = sum(df['stoploss_count'].fillna(0)) if 'stoploss_count' in df.columns else 0
    
    # Log results using log_backtesting_sql (once per DataFrame)
    log_backtesting_sql(index_name, {
        'signal_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'index_name': index_name,
        'signal': 'Completed',
        'strike_price': int(round(df['close'].iloc[-1] / 50) * 50),
        'stop_loss': int(round(df['atr'].iloc[-1])) if 'atr' in df else 0,
        'target': int(round(df['atr'].iloc[-1] * 1.5)) if 'atr' in df else 0,
        'target2': int(round(df['atr'].iloc[-1] * 2.0)) if 'atr' in df else 0,
        'target3': int(round(df['atr'].iloc[-1] * 2.5)) if 'atr' in df else 0,
        'price': df['close'].iloc[-1],
        'rsi': df['rsi'].iloc[-1] if 'rsi' in df else 0,
        'macd': df['macd'].iloc[-1] if 'macd' in df else 0,
        'macd_signal': df['macd_signal'].iloc[-1] if 'macd_signal' in df else 0,
        'ema_20': df['ema_20'].iloc[-1] if 'ema_20' in df else 0,
        'atr': df['atr'].iloc[-1] if 'atr' in df else 0,
        'outcome': 'Success' if total_pnl > 0 else 'Failure',
        'rsi_reason': '',
        'macd_reason': '',
        'price_reason': '',
        'confidence': 'High' if total_pnl > 0 and successful_signals > 0 else 'Low',
        'trade_type': 'Backtest',
        'option_chain_confirmation': 'No',
        'pnl': total_pnl,
        'targets_hit': total_targets_hit,
        'stoploss_count': total_stoploss_hit,
        'failure_reason': 'Negative overall PnL' if total_pnl <= 0 else ''
    })

def run_bot(fyers):
    # Updated to use the correct symbol format based on testing
    symbols = {
        "NSE:NIFTY50-INDEX": "NIFTY50",
        "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
    }

    lot_sizes = {
        "NIFTY50": 50,
        "BANKNIFTY": 15
    }
    
    # Use a more recent time period (last 30 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print(f"Running backtesting from {start_date} to {end_date}")

    for symbol, index_name in symbols.items():
        try:
            print(f"\nProcessing {index_name} from {start_date} to {end_date}")
            
            # Try to fetch data with progressive fallbacks
            try:
                # Start with daily resolution
                df = fetch_candles(symbol, fyers, resolution="D", range_from=start_date, range_to=end_date)
            except Exception as e:
                print(f"Error fetching data: {e}")
                
                try:
                    # Try with an even shorter date range (last 7 days)
                    shorter_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                    print(f"Trying with last 7 days: {shorter_start} to {end_date}")
                    df = fetch_candles(symbol, fyers, resolution="D", range_from=shorter_start, range_to=end_date)
                except Exception as e2:
                    print(f"All attempts failed: {e2}")
                    print(f"❗ Could not fetch data for {index_name}, skipping...")
                    continue
            
            if df.empty:
                print(f"❗ No candles available for {index_name}")
                continue
                
            # Process the data
            generate_signal_all(
                df,
                index_name,
                lot_sizes[index_name]
            )
            print(f"✅✅✅ Historical Data Processing Finished for {index_name} ✅✅✅")
        except Exception as e:
            print(f"❌ Error processing {index_name}: {e}")
            
    print("\n✅ Backtesting completed.")

# 🔥 Entry point
if __name__ == "__main__":
    print("🚀 Bot Started - Auto Refresh Every 5 Minutes!")

    run_bot(fyers)
