#!/usr/bin/env python3
"""
Comprehensive System Testing Framework
Tests all production systems under load and real market conditions
"""

import sys
import os
import time
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemLoadTester:
    """Comprehensive system load testing"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        
    def test_websocket_connections(self) -> bool:
        """Test WebSocket connections under load"""
        try:
            from src.core.fyers_websocket_manager import get_websocket_manager
            
            symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"]
            ws_manager = get_websocket_manager(symbols)
            
            # Test connection
            ws_manager.start()
            time.sleep(5)
            
            if not ws_manager.is_connected:
                return False
            
            # Test data flow
            data_count = 0
            start_time = time.time()
            
            while time.time() - start_time < 10:  # 10 seconds
                data = ws_manager.get_all_live_data()
                if data:
                    data_count += len(data)
                time.sleep(0.1)
            
            ws_manager.stop()
            
            # Performance metrics
            self.performance_metrics['websocket_data_rate'] = data_count / 10
            logger.info(f"✅ WebSocket test: {data_count} data points in 10s")
            
            return data_count > 0
            
        except Exception as e:
            logger.error(f"❌ WebSocket test failed: {e}")
            return False
    
    def test_strategy_performance(self) -> bool:
        """Test strategy performance under load"""
        try:
            from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
            
            # Create large dataset
            dates = pd.date_range('2024-01-01', periods=10000, freq='1H')
            data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.uniform(19000, 20000, 10000),
                'high': np.random.uniform(19000, 20000, 10000),
                'low': np.random.uniform(19000, 20000, 10000),
                'close': np.random.uniform(19000, 20000, 10000),
                'volume': np.random.uniform(1000, 10000, 10000)
            })
            
            strategy = EmaCrossoverEnhanced({})
            
            # Test performance
            start_time = time.time()
            result = strategy.analyze_vectorized('TEST', data)
            duration = time.time() - start_time
            
            # Performance metrics
            self.performance_metrics['strategy_execution_time'] = duration
            self.performance_metrics['data_points_processed'] = len(data)
            self.performance_metrics['strategy_throughput'] = len(data) / duration
            
            logger.info(f"✅ Strategy test: {len(data)} points in {duration:.3f}s")
            
            return duration < 5.0  # Should process 10k points in under 5 seconds
            
        except Exception as e:
            logger.error(f"❌ Strategy test failed: {e}")
            return False
    
    def test_database_performance(self) -> bool:
        """Test database performance under load"""
        try:
            from src.models.enhanced_database import EnhancedTradingDatabase
            
            db = EnhancedTradingDatabase('test_system.db')
            
            # Test bulk operations
            start_time = time.time()
            
            # Insert 1000 test records
            for i in range(1000):
                signal_data = {
                    'signal_id': f'test_{i}',
                    'symbol': 'NSE:NIFTY50-INDEX',
                    'strategy': 'test_strategy',
                    'signal_type': 'BUY',
                    'confidence': 0.8,
                    'price': 19500.0 + i,
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
            
            duration = time.time() - start_time
            
            # Performance metrics
            self.performance_metrics['database_insert_time'] = duration
            self.performance_metrics['database_throughput'] = 1000 / duration
            
            logger.info(f"✅ Database test: 1000 inserts in {duration:.3f}s")
            
            # Cleanup
            os.remove('test_system.db')
            
            return duration < 10.0  # Should insert 1000 records in under 10 seconds
            
        except Exception as e:
            logger.error(f"❌ Database test failed: {e}")
            return False
    
    def test_risk_management_load(self) -> bool:
        """Test risk management under load"""
        try:
            from src.advanced_systems.advanced_risk_management import AdvancedRiskManager
            
            risk_manager = AdvancedRiskManager()
            
            # Test concurrent risk checks
            start_time = time.time()
            
            for i in range(100):
                risk_manager.add_position(f'SYMBOL_{i}', 100.0, 19500.0, datetime.now())
                risk_manager.update_position_price(f'SYMBOL_{i}', 19500.0 + i)
                risk_manager.check_risk_limits()
            
            duration = time.time() - start_time
            
            # Performance metrics
            self.performance_metrics['risk_check_time'] = duration
            self.performance_metrics['risk_throughput'] = 100 / duration
            
            logger.info(f"✅ Risk management test: 100 operations in {duration:.3f}s")
            
            return duration < 5.0  # Should handle 100 operations in under 5 seconds
            
        except Exception as e:
            logger.error(f"❌ Risk management test failed: {e}")
            return False
    
    def test_memory_usage(self) -> bool:
        """Test memory usage under load"""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create large datasets
            large_data = []
            for i in range(100):
                data = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
                    'price': np.random.uniform(19000, 20000, 1000)
                })
                large_data.append(data)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Cleanup
            del large_data
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Performance metrics
            self.performance_metrics['initial_memory_mb'] = initial_memory
            self.performance_metrics['peak_memory_mb'] = peak_memory
            self.performance_metrics['final_memory_mb'] = final_memory
            self.performance_metrics['memory_growth_mb'] = peak_memory - initial_memory
            
            logger.info(f"✅ Memory test: {initial_memory:.1f}MB -> {peak_memory:.1f}MB -> {final_memory:.1f}MB")
            
            return (peak_memory - initial_memory) < 500  # Should not grow more than 500MB
            
        except Exception as e:
            logger.error(f"❌ Memory test failed: {e}")
            return False
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all system tests"""
        logger.info("🚀 Starting comprehensive system testing...")
        
        tests = [
            ('websocket_connections', self.test_websocket_connections),
            ('strategy_performance', self.test_strategy_performance),
            ('database_performance', self.test_database_performance),
            ('risk_management_load', self.test_risk_management_load),
            ('memory_usage', self.test_memory_usage)
        ]
        
        results = {}
        for test_name, test_func in tests:
            logger.info(f"🧪 Running {test_name} test...")
            results[test_name] = test_func()
        
        # Calculate overall score
        passed_tests = sum(1 for result in results.values() if result)
        total_tests = len(results)
        success_rate = passed_tests / total_tests
        
        self.test_results = {
            'results': results,
            'performance_metrics': self.performance_metrics,
            'success_rate': success_rate,
            'passed_tests': passed_tests,
            'total_tests': total_tests
        }
        
        logger.info(f"📊 System test results: {passed_tests}/{total_tests} passed ({success_rate:.1%})")
        
        return self.test_results

def main():
    """Run comprehensive system testing"""
    tester = SystemLoadTester()
    results = tester.run_comprehensive_test()
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE SYSTEM TEST RESULTS")
    print("="*80)
    
    print(f"\n🎯 OVERALL RESULTS:")
    print(f"   Success Rate: {results['success_rate']:.1%}")
    print(f"   Passed Tests: {results['passed_tests']}/{results['total_tests']}")
    
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, result in results['results'].items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    for metric, value in results['performance_metrics'].items():
        print(f"   {metric}: {value}")
    
    print("\n" + "="*80)
    
    return results['success_rate'] >= 0.8  # 80% success rate required

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
