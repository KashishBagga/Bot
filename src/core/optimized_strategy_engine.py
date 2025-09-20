#!/usr/bin/env python3
"""
Optimized Strategy Engine with Shared Indicator Cache
Eliminates redundant calculations and improves performance
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from enum import Enum
import time

# Import strategies
from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.strategies.simple_ema_strategy import SimpleEmaStrategy
from src.strategies.supertrend_ema import SupertrendEma
from src.core.technical_indicators import calculate_all_indicators, validate_indicators

logger = logging.getLogger(__name__)

class OptimizedStrategyEngine:
    """Optimized strategy engine with shared indicator cache and vectorized processing."""
    
    def __init__(self, symbols: List[str], confidence_cutoff: float = 25.0):
        self.symbols = symbols
        self.confidence_cutoff = confidence_cutoff
        self.tz = ZoneInfo("Asia/Kolkata")
        
        # Initialize strategies
        self.strategies = {
            'ema_crossover_enhanced': EmaCrossoverEnhanced(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma(),
            'simple_ema': SimpleEmaStrategy(),
            'supertrend_ema': SupertrendEma()
        }
        
        # Performance tracking
        self.performance_stats = {
            'indicator_calculation_time': 0.0,
            'strategy_processing_time': 0.0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Indicator cache to avoid recalculation
        self.indicator_cache = {}
        self.cache_timestamp = {}
        self.cache_ttl = 60  # Cache TTL in seconds
        
        logger.info(f"🚀 Optimized Strategy Engine initialized for {len(symbols)} symbols")
    
    def _get_cached_indicators(self, symbol: str, data_hash: str) -> Optional[pd.DataFrame]:
        """Get cached indicators if available and not expired."""
        cache_key = f"{symbol}_{data_hash}"
        
        if cache_key in self.indicator_cache:
            cache_time = self.cache_timestamp.get(cache_key, 0)
            if time.time() - cache_time < self.cache_ttl:
                self.performance_stats['cache_hits'] += 1
                logger.debug(f"✅ Cache hit for {symbol}")
                return self.indicator_cache[cache_key]
            else:
                # Cache expired, remove it
                del self.indicator_cache[cache_key]
                del self.cache_timestamp[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        return None
    
    def _cache_indicators(self, symbol: str, data_hash: str, indicators: pd.DataFrame):
        """Cache calculated indicators."""
        cache_key = f"{symbol}_{data_hash}"
        self.indicator_cache[cache_key] = indicators.copy()
        self.cache_timestamp[cache_key] = time.time()
        logger.debug(f"💾 Cached indicators for {symbol}")
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of data for cache key."""
        # Use last few rows and shape for hash
        last_values = data.tail(5).values.tobytes()
        shape = str(data.shape).encode()
        return str(hash(last_values + shape))
    
    def _get_or_calculate_indicators(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Get indicators from cache or calculate them."""
        data_hash = self._calculate_data_hash(data)
        
        # Try to get from cache first
        cached_indicators = self._get_cached_indicators(symbol, data_hash)
        if cached_indicators is not None:
            return cached_indicators
        
        # Calculate indicators
        start_time = time.time()
        indicators = calculate_all_indicators(data)
        calculation_time = time.time() - start_time
        
        self.performance_stats['indicator_calculation_time'] += calculation_time
        
        # Validate and cache
        if validate_indicators(indicators):
            self._cache_indicators(symbol, data_hash, indicators)
            logger.debug(f"✅ Calculated and cached indicators for {symbol} in {calculation_time:.3f}s")
            return indicators
        else:
            logger.warning(f"⚠️ Invalid indicators for {symbol}")
            return data  # Return original data if indicators are invalid
    
    def _process_strategy_optimized(self, strategy_name: str, strategy, data: pd.DataFrame, 
                                  symbol: str, current_price: float) -> Optional[Dict]:
        """Process a single strategy with optimized data handling."""
        try:
            start_time = time.time()
            
            # Use pre-calculated indicators
            signal_result = strategy.analyze(data)
            
            processing_time = time.time() - start_time
            self.performance_stats['strategy_processing_time'] += processing_time
            
            if not signal_result or signal_result.get('signal') in ['NO TRADE', 'ERROR']:
                return None
            
            signal_type = signal_result.get('signal')
            confidence = signal_result.get('confidence', signal_result.get('confidence_score', 0))
            
            # Check confidence threshold
            if confidence < self.confidence_cutoff:
                logger.debug(f"⚠️ {strategy_name} signal for {symbol} rejected: confidence {confidence} < {self.confidence_cutoff}")
                return None
            
            # Create optimized signal
            signal = {
                'symbol': symbol,
                'strategy': strategy_name,
                'signal': signal_type,
                'confidence': confidence,
                'price': current_price,
                'timestamp': datetime.now(self.tz).isoformat(),
                'timeframe': '5m',
                'strength': 'moderate',
                'confirmed': True,
                'processing_time': processing_time,
                'indicator_values': {
                    'ema_9': data['ema_9'].iloc[-1] if 'ema_9' in data.columns else None,
                    'ema_21': data['ema_21'].iloc[-1] if 'ema_21' in data.columns else None,
                    'rsi': data['rsi'].iloc[-1] if 'rsi' in data.columns else None,
                    'atr': data['atr'].iloc[-1] if 'atr' in data.columns else None
                }
            }
            
            logger.debug(f"✅ {strategy_name} signal for {symbol}: {signal_type} (confidence: {confidence}) in {processing_time:.3f}s")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error in {strategy_name} analysis for {symbol}: {e}")
            return None
    
    def generate_signals_for_all_symbols(self, historical_data: Dict[str, pd.DataFrame], 
                                       current_prices: Dict[str, float]) -> List[Dict]:
        """Generate signals for all symbols with optimized processing."""
        start_time = time.time()
        all_signals = []
        
        for symbol in self.symbols:
            if symbol not in historical_data or symbol not in current_prices:
                continue
                
            data = historical_data[symbol]
            current_price = current_prices[symbol]
            
            if data is None or len(data) < 50:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(data) if data is not None else 0} candles")
                continue
            
            # Get or calculate indicators (with caching)
            data_with_indicators = self._get_or_calculate_indicators(symbol, data)
            
            # Process all strategies with shared indicators
            for strategy_name, strategy in self.strategies.items():
                signal = self._process_strategy_optimized(
                    strategy_name, strategy, data_with_indicators, symbol, current_price
                )
                if signal:
                    all_signals.append(signal)
        
        # Sort by confidence
        all_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        total_time = time.time() - start_time
        self.performance_stats['total_processing_time'] = total_time
        
        logger.info(f"📊 Generated {len(all_signals)} signals in {total_time:.3f}s")
        logger.info(f"📈 Performance: Indicators: {self.performance_stats['indicator_calculation_time']:.3f}s, "
                   f"Strategies: {self.performance_stats['strategy_processing_time']:.3f}s, "
                   f"Cache hits: {self.performance_stats['cache_hits']}, "
                   f"Cache misses: {self.performance_stats['cache_misses']}")
        
        return all_signals
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def clear_cache(self):
        """Clear the indicator cache."""
        self.indicator_cache.clear()
        self.cache_timestamp.clear()
        logger.info("🗑️ Indicator cache cleared")
