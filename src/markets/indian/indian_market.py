"""
Indian market implementation for NSE/BSE trading.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from zoneinfo import ZoneInfo
import requests
from typing import Set

from src.adapters.market_interface import (
    MarketInterface, MarketConfig, MarketType, AssetType, Contract
)



class IndianHolidayManager:
    """Manage Indian market holidays"""
    
    def __init__(self):
        self.tz = ZoneInfo('Asia/Kolkata')
        self.holidays_cache = set()
    
    def is_holiday(self, date=None):
        """Check if given date is a market holiday"""
        if date is None:
            from src.core.timezone_utils import now; date = now()
        
        # Simple holiday check - in real implementation, use proper holiday API
        date_str = date.strftime('%Y-%m-%d')
        
        # Mock holidays for testing
        mock_holidays = {
            '2024-01-26', '2024-03-08', '2024-03-29', '2024-04-11',
            '2024-04-17', '2024-05-01', '2024-06-17', '2024-08-15',
            '2024-08-26', '2024-10-02', '2024-10-12', '2024-11-01',
            '2024-11-15', '2024-12-25'
        }
        
        return date_str in mock_holidays


class IndianMarket(MarketInterface):
    """Indian market implementation for NSE/BSE."""
    
    def __init__(self):
        self.holiday_manager = IndianHolidayManager()
        config = MarketConfig(
            market_type=MarketType.INDIAN_STOCKS,
            timezone="Asia/Kolkata",
            trading_hours={"start": "09:15", "end": "15:30"},
            trading_days=[0, 1, 2, 3, 4],  # Monday to Friday
            lot_sizes={
                "NSE:NIFTY50-INDEX": 50,
                "NSE:NIFTYBANK-INDEX": 25,
                "NSE:FINNIFTY-INDEX": 40,
                "NSE:RELIANCE-EQ": 1,
                "NSE:HDFCBANK-EQ": 1
            },
            tick_sizes={
                "NSE:NIFTY50-INDEX": 0.05,
                "NSE:NIFTYBANK-INDEX": 0.05,
                "NSE:FINNIFTY-INDEX": 0.05,
                "NSE:RELIANCE-EQ": 0.05,
                "NSE:HDFCBANK-EQ": 0.05
            },
            commission_rates={
                "NSE:NIFTY50-INDEX": 0.0001,
                "NSE:NIFTYBANK-INDEX": 0.0001,
                "NSE:FINNIFTY-INDEX": 0.0001,
                "NSE:RELIANCE-EQ": 0.0003,
                "NSE:HDFCBANK-EQ": 0.0003
            },
            margin_requirements={
                "NSE:NIFTY50-INDEX": 0.1,
                "NSE:NIFTYBANK-INDEX": 0.1,
                "NSE:FINNIFTY-INDEX": 0.1,
                "NSE:RELIANCE-EQ": 0.2,
                "NSE:HDFCBANK-EQ": 0.2
            },
            currency="INR"
        )
        super().__init__(config)
        self._data_provider = None
        self.tz = ZoneInfo("Asia/Kolkata")
    
    def is_market_open(self, timestamp: Optional[datetime] = None) -> bool:
        if timestamp is None:
            timestamp = datetime.now(self.tz)
        else:
            timestamp = timestamp.astimezone(self.tz)
        
        # Check if it's a trading day
        if timestamp.weekday() >= 5:  # Saturday/Sunday
            return False
        
        # Check trading hours
        market_start = timestamp.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = timestamp.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= timestamp <= market_end
    
    def get_trading_hours(self) -> Dict[str, str]:
        return self.config.trading_hours
    
    def get_contract_info(self, symbol: str) -> Optional[Contract]:
        # Parse symbol to determine asset type
        if "INDEX" in symbol:
            asset_type = AssetType.STOCK  # Index treated as stock
        elif "EQ" in symbol:
            asset_type = AssetType.STOCK
        else:
            asset_type = AssetType.STOCK  # Default
        
        return Contract(
            symbol=symbol,
            underlying=symbol,
            asset_type=asset_type,
            lot_size=self.get_lot_size(symbol),
            tick_size=self.get_tick_size(symbol),
            margin_requirement=self.get_margin_requirement(symbol),
            commission_rate=self.get_commission_rate(symbol)
        )
    
    def get_lot_size(self, symbol: str) -> int:
        return self.config.lot_sizes.get(symbol, 1)
    
    def get_tick_size(self, symbol: str) -> float:
        return self.config.tick_sizes.get(symbol, 0.05)
    
    def get_commission_rate(self, symbol: str) -> float:
        return self.config.commission_rates.get(symbol, 0.0003)
    
    def get_margin_requirement(self, symbol: str) -> float:
        return self.config.margin_requirements.get(symbol, 0.2)
    
    def normalize_symbol(self, symbol: str) -> str:
        # Ensure proper NSE format
        if not symbol.startswith("NSE:"):
            symbol = f"NSE:{symbol}"
        return symbol
    
    def validate_symbol(self, symbol: str) -> bool:
        return symbol in self.config.lot_sizes
    
    def get_data_provider(self):
        """Get the data provider for this market (singleton pattern)."""
        if self._data_provider is None:
            import os
            demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
            print(f"🔧 Data provider mode: {'Demo' if demo_mode else 'Live'}")
            
            if demo_mode:
                from src.adapters.data.mock_data_provider import MockDataProvider
                self._data_provider = MockDataProvider()
                print("🎭 Using Mock Data Provider")
            else:
                from src.adapters.data.fyers_data_provider import FyersDataProvider
                self._data_provider = FyersDataProvider()
                print("🔐 Using Fyers Data Provider")
        return self._data_provider
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            data_provider = self.get_data_provider()
            return data_provider.get_current_price(symbol)
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
    
    def get_current_prices_batch(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Get current prices for multiple symbols in a single request."""
        try:
            data_provider = self.get_data_provider()
            return data_provider.get_current_prices_batch(symbols)
        except Exception as e:
            print(f"Error getting batch prices: {e}")
            return {symbol: None for symbol in symbols}
    
    def _get_price_with_retry(self, symbol: str, max_retries: int = 3) -> Optional[float]:
        """Get price with retry logic"""
        for attempt in range(max_retries):
            try:
                # Mock API call with timeout
                # response = requests.get(url, timeout=10)
                # return response.json().get('price')
                
                # Mock implementation
                return 19500.0 + (attempt * 100)
                
            except requests.exceptions.Timeout:
                logger.warning(f"⚠️ API timeout for {symbol}, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
            except Exception as e:
                logger.error(f"❌ API error for {symbol}: {e}")
                break
        
        return None
        """Get current price for a symbol."""
        try:
            data_provider = self.get_data_provider()
            quotes = data_provider.get_current_price(symbol)
            if quotes is not None:
                return quotes
            return None
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_default_symbols(self):
        """Get default symbols for Indian trading."""
        return ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX', 'NSE:FINNIFTY-INDEX', 'NSE:RELIANCE-EQ', 'NSE:HDFCBANK-EQ']

    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol."""
        try:
            data_provider = self.get_data_provider()
            if data_provider:
                return data_provider.get_historical_data(symbol, start_date, end_date, interval)
            return None
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return None
