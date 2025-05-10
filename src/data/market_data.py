"""
Market data module.
Handles fetching and processing market data from various sources.
"""
import pandas as pd
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
import pytz
from src.config.settings import TIMEZONE

class MarketData:
    """Market data fetcher and processor."""
    
    def __init__(self, fyers_client=None):
        """Initialize the market data handler with optional Fyers client."""
        self.fyers = fyers_client
    
    def fetch_fyers_candles(self, symbol, resolution="1", days_back=1):
        """Fetch candle data from Fyers.
        
        Args:
            symbol: The trading symbol to fetch data for
            resolution: Candle resolution (1, 5, 15, 30, 60, D, W, M)
            days_back: Number of days to look back
            
        Returns:
            pandas.DataFrame: OHLCV data with datetime index
        """
        if not self.fyers:
            raise ValueError("Fyers client not initialized")
        
        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": start_date,
            "range_to": end_date,
            "cont_flag": "1"
        }
        
        try:
            res = self.fyers.history(data)
            candles = res.get("candles", [])
            
            if not candles:
                return pd.DataFrame()
                
            df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching candles: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, df):
        """Preprocess market data.
        
        Args:
            df: pandas.DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: Preprocessed data
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert index to Indian timezone if not already
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC").tz_convert(TIMEZONE)
        
        # Ensure all numeric columns are float
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Calculate additional columns
        df["hl2"] = (df["high"] + df["low"]) / 2
        df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
        df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        
        return df
    
    def get_latest_candle(self, symbol, resolution="1"):
        """Get the latest candle for a symbol.
        
        Args:
            symbol: The trading symbol
            resolution: Candle resolution
            
        Returns:
            dict: Latest candle data
        """
        df = self.fetch_fyers_candles(symbol, resolution, days_back=1)
        
        if df.empty:
            return {}
            
        latest = df.iloc[-1].to_dict()
        latest["time"] = df.index[-1]
        
        return latest
    
    def is_market_open(self):
        """Check if the market is currently open.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        now = datetime.now(TIMEZONE)
        
        # Check if it's a weekday (0=Monday, 4=Friday)
        if now.weekday() > 4:  # Weekend
            return False
            
        # Check market hours (9:15 AM to 3:30 PM)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    @staticmethod  
    def resample_data(df, timeframe):
        """Resample data to a different timeframe.
        
        Args:
            df: pandas.DataFrame with OHLCV data
            timeframe: Target timeframe (e.g., '5min', '1H', '1D')
            
        Returns:
            pandas.DataFrame: Resampled data
        """
        if df.empty:
            return df
            
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex for resampling")
            
        # Resample the data
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Drop rows with NaN values that might appear at the edges
        resampled.dropna(inplace=True)
        
        return resampled 