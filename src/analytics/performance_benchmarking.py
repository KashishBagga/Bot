#!/usr/bin/env python3
"""
Performance Benchmarking System
Comprehensive performance analysis and optimization
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import psutil
import threading
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Comprehensive performance benchmarking system"""
    
    def __init__(self):
        self.benchmark_results = {}
        self.performance_history = []
        
    def benchmark_strategy_execution(self, strategy_name: str, data_size: int = 1000) -> Dict[str, Any]:
        """Benchmark strategy execution performance"""
        try:
            from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
            
            # Create test data
            dates = pd.date_range('2024-01-01', periods=data_size, freq='1H')
            data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.uniform(19000, 20000, data_size),
                'high': np.random.uniform(19000, 20000, data_size),
                'low': np.random.uniform(19000, 20000, data_size),
                'close': np.random.uniform(19000, 20000, data_size),
                'volume': np.random.uniform(1000, 10000, data_size)
            })
            
            strategy = EmaCrossoverEnhanced({})
            
            # Benchmark execution
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            result = strategy.analyze_vectorized('TEST', data)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            throughput = data_size / execution_time
            
            benchmark_result = {
                'strategy': strategy_name,
                'data_size': data_size,
                'execution_time': execution_time,
                'memory_usage_mb': memory_usage,
                'throughput_points_per_second': throughput,
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ Strategy benchmark: {strategy_name} - {throughput:.0f} points/sec")
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"❌ Strategy benchmark failed: {e}")
            return {}
    
    def benchmark_database_operations(self) -> Dict[str, Any]:
        """Benchmark database operations"""
        try:
            from src.models.enhanced_database import EnhancedTradingDatabase
            
            db = EnhancedTradingDatabase('benchmark_test.db')
            
            # Benchmark insert operations
            start_time = time.time()
            
            for i in range(1000):
                signal_data = {
                    'signal_id': f'benchmark_{i}',
                    'symbol': 'NSE:NIFTY50-INDEX',
                    'strategy': 'benchmark_strategy',
                    'signal_type': 'BUY',
                    'confidence': 0.8,
                    'price': 19500.0,
                    'timestamp': datetime.now(),
                    'timeframe': '1h',
                    'strength': 'STRONG',
                    'indicator_values': {'ema_20': 19500, 'ema_50': 19400},
                    'market_condition': 'TRENDING',
                    'volatility': 0.02,
                    'position_size': 100.0,
                    'stop_loss_price': 19400.0,
                    'take_profit_price': 19600.0
                }
                db.save_entry_signal('indian', signal_data)
            
            insert_time = time.time() - start_time
            
            # Benchmark query operations
            start_time = time.time()
            
            for i in range(100):
                signals = db.get_entry_signals('indian', 'NSE:NIFTY50-INDEX', limit=10)
            
            query_time = time.time() - start_time
            
            # Cleanup
            os.remove('benchmark_test.db')
            
            benchmark_result = {
                'insert_operations': 1000,
                'insert_time': insert_time,
                'insert_throughput': 1000 / insert_time,
                'query_operations': 100,
                'query_time': query_time,
                'query_throughput': 100 / query_time,
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ Database benchmark: {1000/insert_time:.0f} inserts/sec, {100/query_time:.0f} queries/sec")
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"❌ Database benchmark failed: {e}")
            return {}
    
    def benchmark_websocket_performance(self) -> Dict[str, Any]:
        """Benchmark WebSocket performance"""
        try:
            from src.core.fyers_websocket_manager import get_websocket_manager
            
            symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"]
            ws_manager = get_websocket_manager(symbols)
            
            # Start WebSocket
            ws_manager.start()
            time.sleep(2)  # Wait for connection
            
            if not ws_manager.is_connected:
                return {}
            
            # Benchmark data reception
            start_time = time.time()
            data_count = 0
            message_count = 0
            
            while time.time() - start_time < 10:  # 10 seconds
                data = ws_manager.get_all_live_data()
                if data:
                    data_count += len(data)
                    message_count += 1
                time.sleep(0.01)  # 10ms intervals
            
            duration = time.time() - start_time
            ws_manager.stop()
            
            benchmark_result = {
                'duration': duration,
                'data_points_received': data_count,
                'messages_received': message_count,
                'data_rate_per_second': data_count / duration,
                'message_rate_per_second': message_count / duration,
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ WebSocket benchmark: {data_count/duration:.1f} data points/sec")
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"❌ WebSocket benchmark failed: {e}")
            return {}
    
    def benchmark_memory_efficiency(self) -> Dict[str, Any]:
        """Benchmark memory efficiency"""
        try:
            import gc
            
            process = psutil.Process()
            
            # Test memory allocation patterns
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Create and destroy large objects
            memory_usage = []
            for i in range(10):
                # Create large DataFrame
                large_df = pd.DataFrame({
                    'data': np.random.randn(10000, 100)
                })
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_usage.append(current_memory)
                
                # Delete and garbage collect
                del large_df
                gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024
            
            benchmark_result = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'peak_memory_mb': max(memory_usage),
                'memory_growth_mb': final_memory - initial_memory,
                'memory_efficiency': (final_memory - initial_memory) / max(memory_usage) * 100,
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ Memory benchmark: {benchmark_result['memory_efficiency']:.1f}% efficiency")
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"❌ Memory benchmark failed: {e}")
            return {}
    
    def benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """Benchmark concurrent operations"""
        try:
            from src.advanced_systems.advanced_risk_management import AdvancedRiskManager
            
            risk_manager = AdvancedRiskManager()
            results = []
            
            def worker_thread(thread_id: int):
                """Worker thread for concurrent testing"""
                start_time = time.time()
                
                for i in range(50):
                    symbol = f'SYMBOL_{thread_id}_{i}'
                    risk_manager.add_position(symbol, 100.0, 19500.0, datetime.now())
                    risk_manager.update_position_price(symbol, 19500.0 + i)
                    risk_manager.check_risk_limits()
                
                end_time = time.time()
                results.append(end_time - start_time)
            
            # Start multiple threads
            threads = []
            start_time = time.time()
            
            for i in range(5):  # 5 concurrent threads
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            benchmark_result = {
                'threads': 5,
                'operations_per_thread': 50,
                'total_operations': 250,
                'total_time': total_time,
                'operations_per_second': 250 / total_time,
                'avg_thread_time': sum(results) / len(results),
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ Concurrent benchmark: {250/total_time:.1f} operations/sec")
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"❌ Concurrent benchmark failed: {e}")
            return {}
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        logger.info("🚀 Starting comprehensive performance benchmarking...")
        
        benchmarks = [
            ('strategy_execution', lambda: self.benchmark_strategy_execution('EMA_Crossover', 5000)),
            ('database_operations', self.benchmark_database_operations),
            ('websocket_performance', self.benchmark_websocket_performance),
            ('memory_efficiency', self.benchmark_memory_efficiency),
            ('concurrent_operations', self.benchmark_concurrent_operations)
        ]
        
        results = {}
        for benchmark_name, benchmark_func in benchmarks:
            logger.info(f"📊 Running {benchmark_name} benchmark...")
            results[benchmark_name] = benchmark_func()
        
        # Store results
        self.benchmark_results = results
        self.performance_history.append({
            'timestamp': datetime.now(),
            'results': results
        })
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(results)
        
        logger.info(f"�� Performance benchmark completed. Score: {performance_score:.1f}/100")
        
        return {
            'results': results,
            'performance_score': performance_score,
            'timestamp': datetime.now()
        }
    
    def _calculate_performance_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        score = 0
        max_score = 100
        
        # Strategy execution score (25 points)
        if 'strategy_execution' in results and results['strategy_execution']:
            throughput = results['strategy_execution'].get('throughput_points_per_second', 0)
            if throughput > 1000:
                score += 25
            elif throughput > 500:
                score += 20
            elif throughput > 100:
                score += 15
        
        # Database operations score (25 points)
        if 'database_operations' in results and results['database_operations']:
            insert_throughput = results['database_operations'].get('insert_throughput', 0)
            if insert_throughput > 100:
                score += 25
            elif insert_throughput > 50:
                score += 20
            elif insert_throughput > 10:
                score += 15
        
        # WebSocket performance score (25 points)
        if 'websocket_performance' in results and results['websocket_performance']:
            data_rate = results['websocket_performance'].get('data_rate_per_second', 0)
            if data_rate > 10:
                score += 25
            elif data_rate > 5:
                score += 20
            elif data_rate > 1:
                score += 15
        
        # Memory efficiency score (25 points)
        if 'memory_efficiency' in results and results['memory_efficiency']:
            efficiency = results['memory_efficiency'].get('memory_efficiency', 0)
            if efficiency > 80:
                score += 25
            elif efficiency > 60:
                score += 20
            elif efficiency > 40:
                score += 15
        
        return score

def main():
    """Run performance benchmarking"""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\n" + "="*80)
    print("📊 PERFORMANCE BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\n🎯 OVERALL PERFORMANCE SCORE: {results['performance_score']:.1f}/100")
    
    print(f"\n📋 DETAILED BENCHMARKS:")
    for benchmark_name, result in results['results'].items():
        if result:
            print(f"\n   {benchmark_name.upper()}:")
            for metric, value in result.items():
                if metric != 'timestamp':
                    print(f"     {metric}: {value}")
    
    print("\n" + "="*80)
    
    return results['performance_score'] >= 70  # 70% performance score required

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
