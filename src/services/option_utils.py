import datetime
import os
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel
import pandas as pd

# Load environment variables
load_dotenv()

# Fyers API credentials
CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")

# Initialize Fyers client
fyers = fyersModel.FyersModel(token=ACCESS_TOKEN, is_async=False, client_id=CLIENT_ID, log_path="")


def get_nearest_expiry(date=None):
    """Get the nearest Thursday expiry date from a given date (or today)."""
    if date is None:
        date = datetime.date.today()
    else:
        date = pd.to_datetime(date).date() if not isinstance(date, datetime.date) else date
    # Find next Thursday
    days_ahead = (3 - date.weekday()) % 7  # 3 = Thursday
    expiry = date + datetime.timedelta(days=days_ahead)
    return expiry


def construct_option_symbol(index_name, expiry, strike, option_type):
    """Construct Fyers option symbol for NIFTY/BANKNIFTY."""
    # expiry: datetime.date
    # strike: int
    # option_type: 'CE' or 'PE'
    # Example: NSE:NIFTY24APR18000CE
    month_map = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    year_short = str(expiry.year)[-2:]
    month_str = month_map[expiry.month - 1]
    day_str = f"{expiry.day:02d}"
    if index_name.upper() == 'NIFTY50' or index_name.upper() == 'NIFTY':
        symbol = f"NSE:NIFTY{year_short}{month_str}{day_str}{int(strike)}{option_type}"
    elif index_name.upper() == 'BANKNIFTY':
        symbol = f"NSE:BANKNIFTY{year_short}{month_str}{day_str}{int(strike)}{option_type}"
    else:
        raise ValueError(f"Unsupported index: {index_name}")
    return symbol


def fetch_option_ohlcv(symbol, range_from, range_to, resolution="5"):
    """Fetch OHLCV data for a given option contract symbol from Fyers."""
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1"
    }
    response = fyers.history(data)
    candles = response.get('candles', [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df 