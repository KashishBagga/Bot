"""
Configuration settings for the trading bot application.
Centralizes all environment variables and application settings.
"""
import os
from dotenv import load_dotenv
import logging
import pytz

# Load environment variables
load_dotenv()

# Timezone configuration
TIMEZONE = pytz.timezone("Asia/Kolkata")

# Trading symbols and timeframes
SYMBOLS = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX"
]

TIMEFRAMES = [
    "1min",
    "3min", 
    "5min",
    "15min",
    "30min",
    "60min",
    "240min",
    "1D"
]

# Fyers API configuration
FYERS_REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI")
FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
FYERS_SECRET_KEY = os.getenv("FYERS_SECRET_KEY")
FYERS_GRANT_TYPE = os.getenv("FYERS_GRANT_TYPE")
FYERS_RESPONSE_TYPE = os.getenv("FYERS_RESPONSE_TYPE")
FYERS_STATE = os.getenv("FYERS_STATE")
FYERS_AUTH_CODE = os.getenv("FYERS_AUTH_CODE")

# Database configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "trading_signals.db")

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_FREE_GROUP_ID = os.getenv("TELEGRAM_FREE_GROUP_ID", "")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.getenv("LOG_FILE", "logs/trading_bot.log")

# Trading settings
TRADING_HOURS = {
    "start": {"hour": 9, "minute": 15},
    "end": {"hour": 15, "minute": 30}
}

# Default strategy parameters
DEFAULT_STRATEGY_PARAMS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "ema_period": 20,
    "atr_period": 14
}

def setup_logging(logger_name=None):
    """Configure and return a logger instance."""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Create handlers
    file_handler = logging.FileHandler(LOG_FILE)
    console_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 