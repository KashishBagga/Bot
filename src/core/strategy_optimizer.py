#!/usr/bin/env python3
"""
Strategy Optimizer - Eliminates Redundant Calculations
Fixes the core performance issues in strategy processing
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
from zoneinfo import ZoneInfo
import time

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """Optimizes strategy processing by eliminating redundant calculations."""
    
    def __init__(self):
        self.indicator_cache = {}
        self.strategy_cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        
    def _get_cache_key(self, data: pd.DataFrame, strategy_name: str) -> str:
        """Generate cache key for data and strategy."""
        # Use last few rows and data shape for cache key
        last_values = data.tail(3).values.tobytes()
        shape = str(data.shape).encode()
        return f"{strategy_name}_{hash(last_values + shape)}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.strategy_cache:
            return False
        
        cache_time = self.strategy_cache[cache_key].get('timestamp', 0)
        return time.time() - cache_time < self.cache_ttl
    
    def optimize_strategy_processing(self, strategies: Dict, data: pd.DataFrame, 
                                   symbol: str, current_price: float) -> List[Dict]:
        """Process all strategies with shared indicators and caching."""
        signals = []
        
        # Pre-calculate all indicators once
        if 'indicators_calculated' not in data.columns:
            data = self._ensure_indicators_calculated(data)
        
        # Process each strategy
        for strategy_name, strategy in strategies.items():
            try:
                # Check cache first
                cache_key = self._get_cache_key(data, strategy_name)
                
                if self._is_cache_valid(cache_key):
                    cached_result = self.strategy_cache[cache_key]['result']
                    if cached_result and cached_result.get('signal') not in ['NO TRADE', 'ERROR']:
                        signal = self._create_signal_from_cached_result(
                            cached_result, symbol, current_price, strategy_name
                        )
                        if signal:
                            signals.append(signal)
                        continue
                
                # Process strategy
                start_time = time.time()
                signal_result = strategy.analyze(data)
                processing_time = time.time() - start_time
                
                # Cache the result
                self.strategy_cache[cache_key] = {
                    'result': signal_result,
                    'timestamp': time.time()
                }
                
                # Create signal if valid
                if signal_result and signal_result.get('signal') not in ['NO TRADE', 'ERROR']:
                    signal = self._create_signal_from_result(
                        signal_result, symbol, current_price, strategy_name, processing_time
                    )
                    if signal:
                        signals.append(signal)
                        
            except Exception as e:
                logger.error(f"❌ Error in {strategy_name} analysis: {e}")
                continue
        
        return signals
    
    def _ensure_indicators_calculated(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required indicators are calculated."""
        # Check if indicators are already calculated
        required_indicators = ['ema_9', 'ema_21', 'rsi', 'atr', 'macd']
        missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
        
        if not missing_indicators:
            return data
        
        # Calculate missing indicators efficiently
        df = data.copy()
        
        # Calculate EMAs if missing
        if 'ema_9' not in df.columns:
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        if 'ema_21' not in df.columns:
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # Calculate RSI if missing
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.inf)
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50)
        
        # Calculate ATR if missing
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(14, min_periods=1).mean()
            df['atr'] = df['atr'].fillna(df['close'].mean() * 0.02)
        
        # Calculate MACD if missing
        if 'macd' not in df.columns:
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Mark as calculated
        df['indicators_calculated'] = True
        
        return df
    
    def _create_signal_from_result(self, signal_result: Dict, symbol: str, 
                                 current_price: float, strategy_name: str, 
                                 processing_time: float) -> Optional[Dict]:
        """Create signal from strategy result."""
        signal_type = signal_result.get('signal')
        confidence = signal_result.get('confidence', signal_result.get('confidence_score', 0))
        
        if confidence < 25:  # Confidence threshold
            return None
        
        return {
            'symbol': symbol,
            'strategy': strategy_name,
            'signal': signal_type,
            'confidence': confidence,
            'price': current_price,
            'timestamp': datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            'timeframe': '5m',
            'strength': 'moderate',
            'confirmed': True,
            'processing_time': processing_time,
            'reasoning': signal_result.get('reasoning', '')
        }
    
    def _create_signal_from_cached_result(self, signal_result: Dict, symbol: str,
                                        current_price: float, strategy_name: str) -> Optional[Dict]:
        """Create signal from cached result."""
        return self._create_signal_from_result(signal_result, symbol, current_price, strategy_name, 0.0)
    
    def clear_cache(self):
        """Clear all caches."""
        self.indicator_cache.clear()
        self.strategy_cache.clear()
        logger.info("🗑️ Strategy caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'indicator_cache_size': len(self.indicator_cache),
            'strategy_cache_size': len(self.strategy_cache)
        }
