"""
Enhanced Real-Time Data Manager with WebSocket Integration
=========================================================
Combines WebSocket real-time data with fallback to REST API
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from threading import Lock
from dataclasses import dataclass

from src.core.fyers_websocket_manager import get_websocket_manager, MarketData

logger = logging.getLogger(__name__)

@dataclass
class DataFreshness:
    """Track data freshness requirements."""
    current_price_max_age: float = 1.0  # 1 second for WebSocket data
    historical_data_max_age: float = 300.0  # 5 minutes
    quotes_max_age: float = 0.5  # 0.5 seconds for WebSocket data

class EnhancedRealTimeDataManager:
    """Enhanced real-time data manager with WebSocket integration."""
    
    def __init__(self, data_provider, symbols: List[str]):
        self.data_provider = data_provider
        self.symbols = symbols
        self.freshness = DataFreshness()
        self._lock = Lock()
        
        # Initialize WebSocket manager
        self.websocket_manager = get_websocket_manager(symbols)
        self.websocket_manager.add_data_callback(self._on_websocket_data)
        
        # Track last fetch times for fallback
        self._last_historical_fetch = {}
        self._last_historical = {}
        
        # Track WebSocket data
        self._websocket_data: Dict[str, MarketData] = {}
        
        logger.info("🔄 Enhanced Real-Time Data Manager initialized with WebSocket integration")
    
    def _on_websocket_data(self, market_data: MarketData):
        """Callback for WebSocket market data updates."""
        with self._lock:
            self._websocket_data[market_data.symbol] = market_data
            logger.debug(f"📡 WebSocket data updated: {market_data.symbol} = {market_data.ltp}")
    
    def get_current_price(self, symbol: str, force_fresh: bool = True) -> Optional[float]:
        """Get current price with WebSocket priority."""
        # First try WebSocket data
        websocket_price = self.websocket_manager.get_current_price(symbol)
        if websocket_price is not None:
            return websocket_price
        
        # Fallback to REST API if WebSocket data not available
        try:
            price = self.data_provider.get_current_price(symbol)
            if price is not None:
                logger.debug(f"🔄 Fallback price fetched for {symbol}: {price}")
                return price
        except Exception as e:
            logger.error(f"Error fetching fallback price for {symbol}: {e}")
        
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30, force_fresh: bool = False) -> Optional[Any]:
        """Get historical data with controlled caching."""
        with self._lock:
            now = time.time()
            last_fetch = self._last_historical_fetch.get(symbol, 0)
            
            # Only cache historical data for 5 minutes (for technical indicators)
            if force_fresh or (now - last_fetch) > self.freshness.historical_data_max_age:
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    data = self.data_provider.get_historical_data(symbol, start_date, end_date, "1h")
                    if data is not None:
                        self._last_historical[symbol] = data
                        self._last_historical_fetch[symbol] = now
                        logger.debug(f"🔄 Fresh historical data fetched for {symbol}")
                        return data
                    else:
                        return self._last_historical.get(symbol)
                except Exception as e:
                    logger.error(f"Error fetching fresh historical data for {symbol}: {e}")
                    return self._last_historical.get(symbol)
            else:
                # Use cached historical data (acceptable for technical indicators)
                return self._last_historical.get(symbol)
    
    def get_quotes(self, symbols: list, force_fresh: bool = True) -> Optional[Dict]:
        """Get quotes with WebSocket priority."""
        quotes = {}
        
        # Try to get WebSocket data first
        for symbol in symbols:
            websocket_data = self.websocket_manager.get_live_data(symbol)
            if websocket_data:
                quotes[symbol] = {
                    'ltp': websocket_data.ltp,
                    'volume': websocket_data.volume,
                    'change': websocket_data.change,
                    'change_percent': websocket_data.change_percent,
                    'timestamp': websocket_data.timestamp.isoformat()
                }
        
        # If we have all symbols from WebSocket, return them
        if len(quotes) == len(symbols):
            return quotes
        
        # Fallback to REST API for missing symbols
        try:
            rest_quotes = self.data_provider.get_quotes(symbols)
            if rest_quotes:
                quotes.update(rest_quotes)
        except Exception as e:
            logger.error(f"Error fetching fallback quotes: {e}")
        
        return quotes if quotes else None
    
    def get_live_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get complete live market data from WebSocket."""
        return self.websocket_manager.get_live_data(symbol)
    
    def get_all_live_data(self) -> Dict[str, MarketData]:
        """Get all live market data from WebSocket."""
        return self.websocket_manager.get_all_live_data()
    
    def start_websocket(self):
        """Start the WebSocket connection."""
        self.websocket_manager.start()
        logger.info("🚀 WebSocket connection started")
    
    def stop_websocket(self):
        """Stop the WebSocket connection."""
        self.websocket_manager.stop()
        logger.info("🛑 WebSocket connection stopped")
    
    def get_connection_status(self) -> Dict[str, any]:
        """Get WebSocket connection status."""
        return self.websocket_manager.get_connection_status()
    
    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            self._last_historical_fetch.clear()
            self._last_historical.clear()
            logger.info("🗑️ All cached data cleared")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status for monitoring."""
        with self._lock:
            now = time.time()
            return {
                "historical_cache_age": {symbol: now - fetch_time for symbol, fetch_time in self._last_historical_fetch.items()},
                "cached_historical": len(self._last_historical),
                "websocket_data_count": len(self._websocket_data),
                "websocket_connection": self.websocket_manager.is_connected
            }

    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        current_prices = {}
        for symbol in self.symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                current_prices[symbol] = price
        return current_prices
