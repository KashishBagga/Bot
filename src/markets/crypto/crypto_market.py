"""
Crypto market implementation for cryptocurrency trading.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from src.core.timezone_utils import timezone_manager, now

from src.adapters.market_interface import (
    MarketInterface, MarketConfig, MarketType, AssetType, Contract
)



class CryptoMarketPerformanceTracker:
    """Track fill rates and latency per symbol"""
    
    def __init__(self):
        self.fill_rates = {}
        self.latencies = {}
        self.api_errors = {}
    
    def record_fill_rate(self, symbol, fill_rate):
        """Record fill rate for symbol"""
        if symbol not in self.fill_rates:
            self.fill_rates[symbol] = []
        self.fill_rates[symbol].append(fill_rate)
    
    def record_latency(self, symbol, latency):
        """Record API latency for symbol"""
        if symbol not in self.latencies:
            self.latencies[symbol] = []
        self.latencies[symbol].append(latency)
    
    def record_api_error(self, symbol, error_type):
        """Record API error for symbol"""
        key = f"{symbol}_{error_type}"
        self.api_errors[key] = self.api_errors.get(key, 0) + 1
    
    def get_performance_stats(self, symbol):
        """Get performance statistics for symbol"""
        fill_rates = self.fill_rates.get(symbol, [])
        latencies = self.latencies.get(symbol, [])
        
        return {
            'avg_fill_rate': sum(fill_rates) / len(fill_rates) if fill_rates else 0,
            'avg_latency': sum(latencies) / len(latencies) if latencies else 0,
            'total_errors': sum(self.api_errors.values()),
            'sample_size': len(fill_rates)
        }


class CryptoMarket(MarketInterface):
    """Crypto market implementation - 24/7 trading."""
    
    def __init__(self):
        self.performance_tracker = CryptoMarketPerformanceTracker()
        config = MarketConfig(
            market_type=MarketType.CRYPTO,
            timezone="UTC",  # Crypto markets are global
            trading_hours={"start": "00:00", "end": "23:59"},  # 24/7
            trading_days=[0, 1, 2, 3, 4, 5, 6],  # All days
            lot_sizes={
                "BTCUSDT": 0.001,
                "ETHUSDT": 0.01,
                "BNBUSDT": 0.1,
                "ADAUSDT": 1.0,
                "SOLUSDT": 0.1,
                "DOTUSDT": 0.1,
                "MATICUSDT": 1.0,
                "AVAXUSDT": 0.1
            },
            tick_sizes={
                "BTCUSDT": 0.01,
                "ETHUSDT": 0.01,
                "BNBUSDT": 0.01,
                "ADAUSDT": 0.0001,
                "SOLUSDT": 0.01,
                "DOTUSDT": 0.001,
                "MATICUSDT": 0.0001,
                "AVAXUSDT": 0.01
            },
            commission_rates={
                "BTCUSDT": 0.001,  # 0.1% maker/taker
                "ETHUSDT": 0.001,
                "BNBUSDT": 0.001,
                "ADAUSDT": 0.001,
                "SOLUSDT": 0.001,
                "DOTUSDT": 0.001,
                "MATICUSDT": 0.001,
                "AVAXUSDT": 0.001
            },
            margin_requirements={
                "BTCUSDT": 0.05,  # 5% margin for crypto
                "ETHUSDT": 0.05,
                "BNBUSDT": 0.05,
                "ADAUSDT": 0.05,
                "SOLUSDT": 0.05,
                "DOTUSDT": 0.05,
                "MATICUSDT": 0.05,
                "AVAXUSDT": 0.05
            },
            currency="USDT"
        )
        super().__init__(config)
        self.tz = ZoneInfo("UTC")
    
    def is_market_open(self, timestamp: Optional[datetime] = None) -> bool:
        # Crypto markets are always open
        return True
    
    def get_trading_hours(self) -> Dict[str, str]:
        return self.config.trading_hours
    
    def get_contract_info(self, symbol: str) -> Optional[Contract]:
        return Contract(
            symbol=symbol,
            underlying=symbol,
            asset_type=AssetType.CRYPTO,
            lot_size=self.get_lot_size(symbol),
            tick_size=self.get_tick_size(symbol),
            margin_requirement=self.get_margin_requirement(symbol),
            commission_rate=self.get_commission_rate(symbol)
        )
    
    def get_lot_size(self, symbol: str) -> float:
        return self.config.lot_sizes.get(symbol, 0.001)
    
    def get_tick_size(self, symbol: str) -> float:
        return self.config.tick_sizes.get(symbol, 0.01)
    
    def get_commission_rate(self, symbol: str) -> float:
        return self.config.commission_rates.get(symbol, 0.001)
    
    def get_margin_requirement(self, symbol: str) -> float:
        return self.config.margin_requirements.get(symbol, 0.05)
    
    def normalize_symbol(self, symbol: str) -> str:
        # Ensure proper crypto format (e.g., BTCUSDT)
        symbol = symbol.upper()
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        return symbol
    
    def validate_symbol(self, symbol: str) -> bool:
        return symbol in self.config.lot_sizes
    
    def get_data_provider(self):
        """Get the data provider for this market."""
        from src.adapters.data.crypto_data_provider import CryptoDataProvider
        return CryptoDataProvider()
    
    def get_default_symbols(self):
        """Get default symbols for crypto trading."""
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']


    def _api_call_with_retry(self, func, *args, max_retries: int = 3, timeout: int = 10, **kwargs):
        """Wrapper for API calls with timeout and retry"""
        for attempt in range(max_retries):
            start_time = time.time()
            try:
                # Make API call with timeout
                result = func(*args, timeout=timeout, **kwargs)
                
                # Record latency
                latency = time.time() - start_time
                self.performance_tracker.record_latency('api', latency)
                
                return result
                
            except requests.exceptions.Timeout:
                self.performance_tracker.record_api_error('api', 'timeout')
                logger.warning(f"⚠️ API timeout, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            except requests.exceptions.RequestException as e:
                self.performance_tracker.record_api_error('api', 'request_error')
                logger.error(f"❌ API request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
                
            except Exception as e:
                logger.error(f"❌ Unexpected API error: {e}")
                break
        
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            data_provider = self.get_data_provider()
            if data_provider:
                # Use direct price API call instead of latest_data
                return data_provider.get_current_price(symbol)
            return None
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None

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

    def get_current_prices_batch(self, symbols):
        """Get current prices for multiple symbols."""
        try:
            return self.data_provider.get_current_prices_batch(symbols)
        except Exception as e:
            logger.error(f"Error getting batch prices: {e}")
            return {}
