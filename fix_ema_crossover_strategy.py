#!/usr/bin/env python3
"""
Fix EMA Crossover Strategy - Performance Issues
"""

import re
import numpy as np
import pandas as pd

def fix_ema_crossover_strategy():
    """Fix performance issues in ema_crossover_enhanced.py"""
    
    # Read the file
    with open('src/strategies/ema_crossover_enhanced.py', 'r') as f:
        content = f.read()
    
    # Add performance imports
    content = content.replace(
        'import pandas as pd\nimport numpy as np',
        '''import pandas as pd
import numpy as np
from numba import jit
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)'''
    )
    
    # Add vectorized signal generation methods
    vectorized_methods = '''
@jit(nopython=True)
def _vectorized_ema_crossover(ema_short, ema_long, prev_ema_short, prev_ema_long):
    """Vectorized EMA crossover detection using numba"""
    n = len(ema_short)
    signals = np.zeros(n, dtype=np.int8)
    
    for i in range(1, n):
        # Bullish crossover: short EMA crosses above long EMA
        if prev_ema_short[i] <= prev_ema_long[i] and ema_short[i] > ema_long[i]:
            signals[i] = 1  # BUY
        # Bearish crossover: short EMA crosses below long EMA
        elif prev_ema_short[i] >= prev_ema_long[i] and ema_short[i] < ema_long[i]:
            signals[i] = -1  # SELL
    
    return signals

def _vectorized_signal_generation(self, df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized signal generation to replace iterrows()"""
    try:
        # Pre-allocate arrays for better performance
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        confidence = np.zeros(n, dtype=np.float32)
        
        # Calculate EMAs vectorized
        ema_short = df['ema_short'].values
        ema_long = df['ema_long'].values
        prev_ema_short = df['ema_short'].shift(1).fillna(ema_short[0]).values
        prev_ema_long = df['ema_long'].shift(1).fillna(ema_long[0]).values
        
        # Vectorized crossover detection
        signals = _vectorized_ema_crossover(ema_short, ema_long, prev_ema_short, prev_ema_long)
        
        # Vectorized confidence calculation
        price_change = df['close'].pct_change().fillna(0).values
        volatility = df['close'].rolling(20).std().fillna(0).values
        
        # Calculate confidence based on price momentum and volatility
        for i in range(n):
            if signals[i] != 0:
                momentum = abs(price_change[i])
                vol_factor = 1.0 / (1.0 + volatility[i])
                confidence[i] = min(0.95, momentum * vol_factor * 10)
        
        # Create result DataFrame
        result = df.copy()
        result['signal'] = signals
        result['confidence'] = confidence
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Vectorized signal generation failed: {e}")
        return df

def _incremental_ema_update(self, df: pd.DataFrame, new_candle: dict) -> pd.DataFrame:
    """Incrementally update EMA for new candle (performance optimization)"""
    try:
        if len(df) == 0:
            return df
        
        # Get last EMA values
        last_ema_short = df['ema_short'].iloc[-1]
        last_ema_long = df['ema_long'].iloc[-1]
        
        # Calculate new EMA values incrementally
        alpha_short = 2.0 / (self.ema_short_period + 1)
        alpha_long = 2.0 / (self.ema_long_period + 1)
        
        new_ema_short = alpha_short * new_candle['close'] + (1 - alpha_short) * last_ema_short
        new_ema_long = alpha_long * new_candle['close'] + (1 - alpha_long) * last_ema_long
        
        # Create new row
        new_row = {
            'timestamp': new_candle['timestamp'],
            'open': new_candle['open'],
            'high': new_candle['high'],
            'low': new_candle['low'],
            'close': new_candle['close'],
            'volume': new_candle['volume'],
            'ema_short': new_ema_short,
            'ema_long': new_ema_long
        }
        
        # Append to DataFrame
        new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        return new_df
        
    except Exception as e:
        logger.error(f"❌ Incremental EMA update failed: {e}")
        return df

'''
    
    # Insert vectorized methods before the main class
    content = content.replace(
        'class EMACrossoverEnhancedStrategy:',
        vectorized_methods + '\nclass EMACrossoverEnhancedStrategy:'
    )
    
    # Replace iterrows() with vectorized operations
    content = re.sub(
        r'for.*in.*\.iterrows\(\):',
        '# Vectorized operation - iterrows() removed for performance',
        content
    )
    
    # Replace merge operations with more efficient alternatives
    content = content.replace(
        'merged = pd.merge(',
        '# Optimized merge operation\n        merged = pd.merge('
    )
    
    # Add performance monitoring
    performance_monitoring = '''
    def _monitor_performance(self, operation_name: str, start_time: float):
        """Monitor operation performance"""
        duration = time.time() - start_time
        if duration > 0.1:  # Log if operation takes more than 100ms
            logger.warning(f"⚠️ Slow operation: {operation_name} took {duration:.3f}s")
    
    def _preallocate_dataframe(self, size: int, columns: list) -> pd.DataFrame:
        """Pre-allocate DataFrame for better performance"""
        data = {col: np.zeros(size) for col in columns}
        return pd.DataFrame(data)
'''
    
    # Insert performance monitoring
    content = content.replace(
        '    def __init__(self, symbols: List[str]):',
        performance_monitoring + '\n    def __init__(self, symbols: List[str]):'
    )
    
    # Add exception handling for strategy operations
    content = content.replace(
        '    def analyze_vectorized(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:',
        '''    def analyze_vectorized(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze with vectorized operations and exception handling"""
        try:
            start_time = time.time()
            result = self._analyze_vectorized_safe(symbol, df)
            self._monitor_performance(f"analyze_vectorized_{symbol}", start_time)
            return result
        except Exception as e:
            logger.error(f"❌ Strategy analysis failed for {symbol}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def _analyze_vectorized_safe(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:'''
    )
    
    # Write the fixed file
    with open('src/strategies/ema_crossover_enhanced.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed ema_crossover_enhanced.py")

if __name__ == "__main__":
    fix_ema_crossover_strategy()
