#!/usr/bin/env python3
"""
Profit-Optimized Cache System
Balances performance with profit accuracy
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class ProfitOptimizedCache:
    """Cache system optimized for trading profitability."""
    
    def __init__(self, 
                 indicator_ttl: int = 30,  # 30 seconds for indicators
                 strategy_ttl: int = 60,   # 60 seconds for strategies
                 max_cache_size: int = 100):
        self.indicator_ttl = indicator_ttl
        self.strategy_ttl = strategy_ttl
        self.max_cache_size = max_cache_size
        
        # Separate caches for different data types
        self.indicator_cache = {}
        self.strategy_cache = {}
        self.cache_timestamps = {}
        
        # Market event tracking
        self.last_market_event = time.time()
        self.market_volatility_threshold = 0.02  # 2% price change
        self.last_prices = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'profit_impact_events': 0
        }
        
        logger.info(f"💰 Profit-Optimized Cache initialized (TTL: {indicator_ttl}s indicators, {strategy_ttl}s strategies)")
    
    def _is_cache_valid(self, cache_key: str, cache_type: str) -> bool:
        """Check if cache entry is valid based on TTL and market conditions."""
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        current_time = time.time()
        
        # Check TTL
        ttl = self.indicator_ttl if cache_type == 'indicator' else self.strategy_ttl
        if current_time - cache_time > ttl:
            return False
        
        # Check for recent market events
        if current_time - self.last_market_event < 10:  # 10 seconds after market event
            logger.debug(f"🚨 Cache invalidated due to recent market event")
            return False
        
        return True
    
    def _detect_market_event(self, symbol: str, current_price: float) -> bool:
        """Detect significant market events that should invalidate cache."""
        if symbol not in self.last_prices:
            self.last_prices[symbol] = current_price
            return False
        
        last_price = self.last_prices[symbol]
        price_change = abs(current_price - last_price) / last_price
        
        if price_change > self.market_volatility_threshold:
            self.last_market_event = time.time()
            self.stats['profit_impact_events'] += 1
            logger.warning(f"🚨 Market event detected for {symbol}: {price_change:.2%} price change")
            return True
        
        self.last_prices[symbol] = current_price
        return False
    
    def get_indicator_cache(self, symbol: str, data_hash: str) -> Optional[Any]:
        """Get cached indicators if valid."""
        with self._lock:
            cache_key = f"indicator_{symbol}_{data_hash}"
            
            if self._is_cache_valid(cache_key, 'indicator'):
                self.stats['hits'] += 1
                logger.debug(f"✅ Indicator cache hit for {symbol}")
                return self.indicator_cache.get(cache_key)
            
            self.stats['misses'] += 1
            return None
    
    def set_indicator_cache(self, symbol: str, data_hash: str, indicators: Any):
        """Cache indicators with profit optimization."""
        with self._lock:
            cache_key = f"indicator_{symbol}_{data_hash}"
            
            # Clean cache if it's getting too large
            if len(self.indicator_cache) >= self.max_cache_size:
                self._clean_old_cache()
            
            self.indicator_cache[cache_key] = indicators
            self.cache_timestamps[cache_key] = time.time()
            logger.debug(f"💾 Cached indicators for {symbol}")
    
    def get_strategy_cache(self, strategy_name: str, symbol: str, data_hash: str) -> Optional[Any]:
        """Get cached strategy result if valid."""
        with self._lock:
            cache_key = f"strategy_{strategy_name}_{symbol}_{data_hash}"
            
            if self._is_cache_valid(cache_key, 'strategy'):
                self.stats['hits'] += 1
                logger.debug(f"✅ Strategy cache hit for {strategy_name} on {symbol}")
                return self.strategy_cache.get(cache_key)
            
            self.stats['misses'] += 1
            return None
    
    def set_strategy_cache(self, strategy_name: str, symbol: str, data_hash: str, result: Any):
        """Cache strategy result with profit optimization."""
        with self._lock:
            cache_key = f"strategy_{strategy_name}_{symbol}_{data_hash}"
            
            # Clean cache if it's getting too large
            if len(self.strategy_cache) >= self.max_cache_size:
                self._clean_old_cache()
            
            self.strategy_cache[cache_key] = result
            self.cache_timestamps[cache_key] = time.time()
            logger.debug(f"💾 Cached strategy result for {strategy_name} on {symbol}")
    
    def invalidate_on_market_event(self, symbol: str, current_price: float):
        """Invalidate cache when significant market events occur."""
        if self._detect_market_event(symbol, current_price):
            with self._lock:
                # Remove all cache entries for this symbol
                keys_to_remove = [k for k in self.cache_timestamps.keys() if symbol in k]
                for key in keys_to_remove:
                    self.indicator_cache.pop(key, None)
                    self.strategy_cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)
                
                self.stats['invalidations'] += len(keys_to_remove)
                logger.warning(f"🚨 Cache invalidated for {symbol} due to market event")
    
    def _clean_old_cache(self):
        """Clean old cache entries to prevent memory bloat."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, timestamp in self.cache_timestamps.items():
            # Remove entries older than 5 minutes
            if current_time - timestamp > 300:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.indicator_cache.pop(key, None)
            self.strategy_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
        
        if keys_to_remove:
            logger.debug(f"🗑️ Cleaned {len(keys_to_remove)} old cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'invalidations': self.stats['invalidations'],
            'profit_impact_events': self.stats['profit_impact_events'],
            'cache_size': len(self.cache_timestamps),
            'indicator_cache_size': len(self.indicator_cache),
            'strategy_cache_size': len(self.strategy_cache)
        }
    
    def clear_cache(self):
        """Clear all caches."""
        with self._lock:
            self.indicator_cache.clear()
            self.strategy_cache.clear()
            self.cache_timestamps.clear()
            self.last_prices.clear()
            logger.info("🗑️ Profit-optimized cache cleared")
    
    def adjust_ttl_for_volatility(self, symbol: str, volatility: float):
        """Adjust cache TTL based on market volatility."""
        if volatility > 0.05:  # High volatility
            self.indicator_ttl = 15  # 15 seconds
            self.strategy_ttl = 30   # 30 seconds
            logger.info(f"📈 High volatility detected for {symbol}, reduced TTL")
        elif volatility < 0.01:  # Low volatility
            self.indicator_ttl = 60  # 60 seconds
            self.strategy_ttl = 120  # 120 seconds
            logger.info(f"📉 Low volatility detected for {symbol}, increased TTL")
        else:  # Normal volatility
            self.indicator_ttl = 30  # 30 seconds
            self.strategy_ttl = 60   # 60 seconds
