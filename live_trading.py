import time
import pandas as pd
import ta
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
import webbrowser
import logging
import math
import pytz
import schedule
import threading
from dotenv import load_dotenv
import os
from db import log_trade_sql


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


def safe_float(val):
    """Safely convert value to float with rounding and edge case handling."""
    try:
        f = float(val)
        return round(f, 2) if not (math.isnan(f) or math.isinf(f)) else 0.0
    except (ValueError, TypeError):
        return 0.0

# === 🧠 Analyze Signal ===
def analyze_signal(df):
    """Analyze market data and return a signal with indicators and reasons."""
    try:
        df = df.copy()

        # Compute indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().apply(safe_float)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd().apply(safe_float)
        df['macd_signal'] = macd.macd_signal().apply(safe_float)
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator().apply(safe_float)
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().apply(safe_float)

        latest = df.iloc[-1]

        rsi = safe_float(latest.get('rsi'))
        macd_val = safe_float(latest.get('macd'))
        macd_signal = safe_float(latest.get('macd_signal'))
        ema_20 = safe_float(latest.get('ema_20'))
        price = safe_float(latest.get('close'))
        atr = safe_float(latest.get('atr'))

        # Default signal
        signal = "None"
        confidence = "Low"
        trade_type = "Intraday"
        rsi_reason = macd_reason = price_reason = ""
        option_chain_confirmation = "No"

        if (
            rsi > 65 and
            macd_val > macd_signal + 7 and
            price > ema_20 * 1.001
        ):
            signal = "BUY CALL"
            rsi_reason = f"RSI {rsi:.2f} > 65"
            macd_reason = f"MACD {macd_val:.2f} > MACD Signal +7 ({macd_signal + 7:.2f})"
            price_reason = f"Price {price:.2f} > EMA {ema_20:.2f}"
            confidence = "High" if rsi > 70 else "Medium"
        elif (
            rsi < 35 and
            macd_val < macd_signal - 5 and
            price < ema_20 * 0.999
        ):
            signal = "BUY PUT"
            rsi_reason = f"RSI {rsi:.2f} < 35"
            macd_reason = f"MACD {macd_val:.2f} < MACD Signal -5 ({macd_signal - 5:.2f})"
            price_reason = f"Price {price:.2f} < EMA {ema_20:.2f}"
            confidence = "High" if rsi < 30 else "Medium"

        option_chain_confirmation = "Yes" if confidence == "High" else "No"

        return {
            "signal": signal,
            "price": price,
            "rsi": rsi,
            "macd": macd_val,
            "macd_signal": macd_signal,
            "ema_20": ema_20,
            "atr": atr,
            "confidence": confidence,
            "rsi_reason": rsi_reason,
            "macd_reason": macd_reason,
            "price_reason": price_reason,
            "trade_type": trade_type,
            "option_chain_confirmation": option_chain_confirmation
        }

    except Exception as e:
        logger.error(f"Error in analyze_signal: {e}")
        return {
            "signal": "None",
            "price": 0.0,
            "rsi": 0.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "ema_20": 0.0,
            "atr": 0.0,
            "confidence": "Low",
            "rsi_reason": "",
            "macd_reason": "",
            "price_reason": "",
            "trade_type": "Intraday",
            "option_chain_confirmation": "No"
        }


# === 📊 Fetch 1-minute candles ===
def fetch_candles(fyers, symbol):
    data = {
        "symbol": symbol,
        "resolution": "1",
        "date_format": "1",
        "range_from": datetime.now().strftime("%Y-%m-%d"),
        "range_to": datetime.now().strftime("%Y-%m-%d"),
        "cont_flag": "1"
    }
    res = fyers.history(data)
    candles = res.get("candles", [])
    if not candles:
        return pd.DataFrame()
    return pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"]).assign(
        time=lambda df_: pd.to_datetime(df_['time'], unit='s')
    )
# === 📈 Log valid trade ===
def log_trade(index_name, signal_data):
    try:
        signal_time = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
        signal_data['signal_time'] = signal_time
        log_trade_sql(index_name, signal_data)
        logger.info(f"✅ Trade Logged: {signal_data.get('signal')} at {signal_data.get('price')}")
    except Exception as e:
        logger.error(f"Error logging trade: {e}")



def is_market_open():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    return now.weekday() < 5 and now.replace(hour=9, minute=15) <= now <= now.replace(hour=15, minute=30)

# === 🚀 Main Loop ===
def run_realtime_bot():
    logger.info("🔁 Starting real-time signal detection every 1 minute...")
    symbols = {
        "NSE:NIFTY50-INDEX": "NIFTY50",
        "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
    }

    while True:
        if not is_market_open():
            logger.info("Market is closed. Sleeping for 60 seconds.")
            time.sleep(60)
            continue

        for symbol, index_name in symbols.items():
            try:
                df = fetch_candles(fyers, symbol)
                if df.empty:
                    logger.warning(f"No data for {index_name}")
                    continue
                signal_data = analyze_signal(df)
                if signal_data["signal"] != "None":
                    log_trade(index_name, signal_data)
            except Exception as e:
                logger.error(f"❌ Error processing {index_name}: {e}")
        time.sleep(60)



if __name__ == "__main__":
    run_realtime_bot()
