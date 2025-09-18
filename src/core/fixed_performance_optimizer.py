#!/usr/bin/env python3
"""
Fixed Performance Optimizer
Corrected Numba JIT compilation and ML model evaluation issues
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import time
import psutil
import gc
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import Numba, fallback to pure Pandas if not available
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
    logger.info("✅ Numba available - using JIT compilation")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("⚠️ Numba not available - using pure Pandas fallback")
    
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization"""
    execution_time: float
    memory_usage: float
    throughput: float
    optimization_level: str

# Standalone Numba functions (not class methods)
@njit
def numba_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized EMA calculation"""
    alpha = 2.0 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        # Numerically stable formula: ema_new = ema_old + alpha * (price - ema_old)
        ema[i] = ema[i-1] + alpha * (prices[i] - ema[i-1])
    
    return ema

@njit
def numba_signal_generation(close: np.ndarray, ema_short: np.ndarray, 
                           ema_long: np.ndarray, signals: np.ndarray, 
                           confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-optimized signal generation"""
    for i in range(1, len(close)):
        # Bullish crossover
        if ema_short[i-1] <= ema_long[i-1] and ema_short[i] > ema_long[i]:
            signals[i] = 1
            confidence[i] = min(0.95, abs(ema_short[i] - ema_long[i]) / close[i] * 100)
        
        # Bearish crossover
        elif ema_short[i-1] >= ema_long[i-1] and ema_short[i] < ema_long[i]:
            signals[i] = -1
            confidence[i] = min(0.95, abs(ema_short[i] - ema_long[i]) / close[i] * 100)
    
    return signals, confidence

class FixedPerformanceOptimizer:
    """Fixed performance optimizer with corrected Numba usage"""
    
    def __init__(self):
        self.performance_history = []
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.optimization_level = "numba" if NUMBA_AVAILABLE else "pandas"
        
    def optimize_ema_calculation(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Optimized EMA calculation with numerical stability"""
        try:
            if len(prices) == 0:
                return np.array([])
            
            # Use float64 for numerical stability
            prices = prices.astype(np.float64)
            
            if self.optimization_level == "numba" and NUMBA_AVAILABLE:
                return numba_ema(prices, period)
            else:
                return self._pandas_ema(prices, period)
                
        except Exception as e:
            logger.error(f"❌ EMA calculation failed: {e}")
            return np.array([])
    
    def _pandas_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Pure Pandas EMA calculation"""
        try:
            series = pd.Series(prices)
            ema = series.ewm(span=period, adjust=False).mean()
            return ema.values
        except Exception as e:
            logger.error(f"❌ Pandas EMA calculation failed: {e}")
            return np.array([])
    
    def optimize_signal_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimized signal generation with vectorized operations"""
        try:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Pre-allocate arrays for better performance
            n = len(data)
            signals = np.zeros(n, dtype=np.int8)
            confidence = np.zeros(n, dtype=np.float32)
            
            # Vectorized operations
            if 'close' in data.columns and 'ema_short' in data.columns and 'ema_long' in data.columns:
                close = data['close'].values.astype(np.float64)
                ema_short = data['ema_short'].values.astype(np.float64)
                ema_long = data['ema_long'].values.astype(np.float64)
                
                # Vectorized crossover detection
                if self.optimization_level == "numba" and NUMBA_AVAILABLE:
                    signals, confidence = numba_signal_generation(
                        close, ema_short, ema_long, signals, confidence
                    )
                else:
                    signals, confidence = self._pandas_signal_generation(
                        close, ema_short, ema_long, signals, confidence
                    )
            
            # Create result DataFrame
            result = data.copy()
            result['signal'] = signals
            result['confidence'] = confidence
            
            # Performance metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            throughput = n / execution_time if execution_time > 0 else 0
            
            # Store performance metrics
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_usage,
                throughput=throughput,
                optimization_level=self.optimization_level
            )
            self.performance_history.append(metrics)
            
            # Check memory usage
            if memory_usage > self.memory_threshold * 100:  # 100MB threshold
                logger.warning(f"⚠️ High memory usage: {memory_usage:.1f}MB")
                self._optimize_memory_usage()
            
            logger.debug(f"✅ Signal generation: {throughput:.0f} points/sec, {memory_usage:.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Signal generation optimization failed: {e}")
            return data
    
    def _pandas_signal_generation(self, close: np.ndarray, ema_short: np.ndarray, 
                                 ema_long: np.ndarray, signals: np.ndarray, 
                                 confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pure Pandas signal generation"""
        try:
            # Use Pandas for vectorized operations
            df = pd.DataFrame({
                'close': close,
                'ema_short': ema_short,
                'ema_long': ema_long
            })
            
            # Calculate crossovers
            df['ema_short_prev'] = df['ema_short'].shift(1)
            df['ema_long_prev'] = df['ema_long'].shift(1)
            
            # Bullish crossover
            bullish_mask = (df['ema_short_prev'] <= df['ema_long_prev']) & (df['ema_short'] > df['ema_long'])
            signals[bullish_mask] = 1
            
            # Bearish crossover
            bearish_mask = (df['ema_short_prev'] >= df['ema_long_prev']) & (df['ema_short'] < df['ema_long'])
            signals[bearish_mask] = -1
            
            # Calculate confidence
            crossover_mask = bullish_mask | bearish_mask
            confidence[crossover_mask] = np.minimum(
                0.95, 
                np.abs(df['ema_short'] - df['ema_long']) / df['close'] * 100
            )[crossover_mask]
            
            return signals, confidence
            
        except Exception as e:
            logger.error(f"❌ Pandas signal generation failed: {e}")
            return signals, confidence
    
    def _optimize_memory_usage(self):
        """Optimize memory usage"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            memory_info = psutil.Process().memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            logger.info(f"🧹 Memory optimization: {memory_mb:.1f}MB")
            
        except Exception as e:
            logger.error(f"❌ Memory optimization failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        try:
            if not self.performance_history:
                return {}
            
            recent_metrics = self.performance_history[-10:]  # Last 10 operations
            
            avg_execution_time = np.mean([m.execution_time for m in recent_metrics])
            avg_memory_usage = np.mean([m.memory_usage for m in recent_metrics])
            avg_throughput = np.mean([m.throughput for m in recent_metrics])
            
            return {
                'avg_execution_time': avg_execution_time,
                'avg_memory_usage': avg_memory_usage,
                'avg_throughput': avg_throughput,
                'optimization_level': self.optimization_level,
                'numba_available': NUMBA_AVAILABLE,
                'total_operations': len(self.performance_history)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance metrics: {e}")
            return {}
    
    def benchmark_operations(self, data_size: int = 10000) -> Dict[str, Any]:
        """Benchmark operations for performance testing"""
        try:
            # Generate test data
            np.random.seed(42)
            test_data = pd.DataFrame({
                'close': np.random.uniform(19000, 20000, data_size),
                'ema_short': np.random.uniform(19000, 20000, data_size),
                'ema_long': np.random.uniform(19000, 20000, data_size)
            })
            
            # Benchmark signal generation
            start_time = time.time()
            result = self.optimize_signal_generation(test_data)
            signal_time = time.time() - start_time
            
            # Benchmark EMA calculation
            start_time = time.time()
            ema_result = self.optimize_ema_calculation(test_data['close'].values, 20)
            ema_time = time.time() - start_time
            
            return {
                'data_size': data_size,
                'signal_generation_time': signal_time,
                'ema_calculation_time': ema_time,
                'total_time': signal_time + ema_time,
                'throughput': data_size / (signal_time + ema_time),
                'optimization_level': self.optimization_level,
                'numba_available': NUMBA_AVAILABLE
            }
            
        except Exception as e:
            logger.error(f"❌ Benchmarking failed: {e}")
            return {}

# Global fixed performance optimizer instance
fixed_performance_optimizer = FixedPerformanceOptimizer()

# Convenience functions
def optimize_ema_calculation(prices: np.ndarray, period: int) -> np.ndarray:
    """Optimized EMA calculation"""
    return fixed_performance_optimizer.optimize_ema_calculation(prices, period)

def optimize_signal_generation(data: pd.DataFrame) -> pd.DataFrame:
    """Optimized signal generation"""
    return fixed_performance_optimizer.optimize_signal_generation(data)

def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics"""
    return fixed_performance_optimizer.get_performance_metrics()

def benchmark_operations(data_size: int = 10000) -> Dict[str, Any]:
    """Benchmark operations"""
    return fixed_performance_optimizer.benchmark_operations(data_size)
