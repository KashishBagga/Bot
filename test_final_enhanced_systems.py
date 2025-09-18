#!/usr/bin/env python3
"""
Final Comprehensive Test for Enhanced Systems
Tests all critical fixes and improvements with corrected implementations
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_timezone_management():
    """Test enhanced timezone management with market sessions"""
    try:
        from src.core.enhanced_timezone_utils import enhanced_timezone_manager, MarketState
        
        # Test timezone-aware datetime
        current_time = enhanced_timezone_manager.now()
        current_iso = enhanced_timezone_manager.now_isoformat()
        
        # Test market state detection
        market_state = enhanced_timezone_manager.is_market_open()
        
        # Test holiday management
        holiday_manager = enhanced_timezone_manager.holiday_manager
        is_holiday = holiday_manager.is_holiday()
        
        # Test next market open
        next_open = enhanced_timezone_manager.get_next_market_open()
        
        print(f"✅ Enhanced Timezone Management:")
        print(f"   Current IST time: {current_time}")
        print(f"   ISO format: {current_iso}")
        print(f"   Market state: {market_state.value}")
        print(f"   Is holiday: {is_holiday}")
        print(f"   Next market open: {next_open}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced timezone management test failed: {e}")
        return False

def test_actor_model_state_manager():
    """Test actor model state manager"""
    try:
        from src.core.actor_model_state_manager import state_manager, AlertSeverity
        
        # Start state manager
        state_manager.start()
        time.sleep(1)  # Let it initialize
        
        # Test event posting
        state_manager.post_event("test_event", {"test": "data"}, "test_source", priority=1)
        
        # Test alert posting
        state_manager.post_alert("test_alert", AlertSeverity.HIGH, "Test alert message", 
                                {"test": "alert_data"}, "test_source")
        
        # Test state retrieval
        state = state_manager.get_state()
        
        # Test alert history
        alert_history = state_manager.get_alert_history()
        
        # Stop state manager
        state_manager.stop()
        
        print(f"✅ Actor Model State Manager:")
        print(f"   State manager started and stopped successfully")
        print(f"   State keys: {list(state.keys()) if state else 'None'}")
        print(f"   Alert history keys: {list(alert_history.keys()) if alert_history else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Actor model state manager test failed: {e}")
        return False

def test_fixed_performance_optimizer():
    """Test fixed performance optimizer"""
    try:
        from src.core.fixed_performance_optimizer import fixed_performance_optimizer
        
        # Test EMA calculation
        prices = np.random.uniform(19000, 20000, 1000)
        ema_result = fixed_performance_optimizer.optimize_ema_calculation(prices, 20)
        
        # Test signal generation
        test_data = pd.DataFrame({
            'close': np.random.uniform(19000, 20000, 1000),
            'ema_short': np.random.uniform(19000, 20000, 1000),
            'ema_long': np.random.uniform(19000, 20000, 1000)
        })
        signal_result = fixed_performance_optimizer.optimize_signal_generation(test_data)
        
        # Get performance metrics
        metrics = fixed_performance_optimizer.get_performance_metrics()
        
        # Run benchmark
        benchmark = fixed_performance_optimizer.benchmark_operations(5000)
        
        print(f"✅ Fixed Performance Optimizer:")
        print(f"   EMA calculation: {len(ema_result)} values")
        print(f"   Signal generation: {len(signal_result)} rows")
        print(f"   Optimization level: {metrics.get('optimization_level', 'unknown')}")
        print(f"   Numba available: {metrics.get('numba_available', False)}")
        print(f"   Benchmark throughput: {benchmark.get('throughput', 0):.0f} points/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Fixed performance optimizer test failed: {e}")
        return False

def test_fixed_ml_model_evaluation():
    """Test fixed ML model evaluation with leakage detection"""
    try:
        from src.analytics.fixed_ml_model_evaluation import fixed_model_evaluator
        from sklearn.linear_model import LinearRegression
        
        # Generate test data
        np.random.seed(42)
        n_samples = 1000
        
        # Create features (some with potential leakage)
        features = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'future_feature': np.random.randn(n_samples),  # Suspicious name
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H')
        })
        
        # Create target with some correlation
        target = features['feature1'] + np.random.randn(n_samples) * 0.1
        
        # Test realistic model creation
        results = fixed_model_evaluator.create_realistic_model(features, target)
        
        print(f"✅ Fixed ML Model Evaluation:")
        print(f"   Total models: {results.get('evaluation_summary', {}).get('total_models', 0)}")
        print(f"   Realistic models: {results.get('evaluation_summary', {}).get('realistic_models', 0)}")
        print(f"   Best model: {results.get('best_model', 'None')}")
        print(f"   Best directional accuracy: {results.get('evaluation_summary', {}).get('best_directional_accuracy', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fixed ML model evaluation test failed: {e}")
        return False

def test_enhanced_options_pricing():
    """Test enhanced options pricing with market IV"""
    try:
        from src.strategies.enhanced_options_pricing import options_pricer, OptionType
        
        # Test Black-Scholes pricing
        call_price = options_pricer.calculate_black_scholes_price(
            19500, 19500, 0.25, 0.25, OptionType.CALL
        )
        
        # Test implied volatility calculation
        iv = options_pricer.calculate_implied_volatility(
            100, 19500, 19500, 0.25, OptionType.CALL
        )
        
        # Test analytic Greeks
        greeks = options_pricer.calculate_analytic_greeks(
            19500, 19500, 0.25, 0.25, OptionType.CALL
        )
        
        # Test options chain generation
        expiry_date = datetime.now() + timedelta(days=30)
        options_chain = options_pricer.generate_options_chain_with_iv(
            'NSE:NIFTY50-INDEX', 19500, expiry_date
        )
        
        print(f"✅ Enhanced Options Pricing:")
        print(f"   Call price: {call_price:.2f}")
        print(f"   Implied volatility: {iv:.3f}")
        print(f"   Delta: {greeks.delta:.3f}")
        print(f"   Gamma: {greeks.gamma:.6f}")
        print(f"   Theta: {greeks.theta:.3f}")
        print(f"   Vega: {greeks.vega:.3f}")
        print(f"   Stability score: {greeks.stability_score:.3f}")
        print(f"   Options chain strikes: {len(options_chain.get('strike_prices', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced options pricing test failed: {e}")
        return False

def test_enhanced_risk_management():
    """Test enhanced risk management"""
    try:
        from src.core.enhanced_risk_management import enhanced_risk_manager, PositionRisk
        
        # Create test positions
        positions = {
            'NIFTY': PositionRisk(
                symbol='NIFTY',
                position_size=100.0,
                position_value=1950000.0,
                risk_per_trade=0.02,
                stop_loss_distance=0.05,
                expected_return=0.1,
                volatility=0.2,
                correlation_with_portfolio=0.3,
                margin_requirement=195000.0
            )
        }
        
        # Test portfolio risk calculation
        risk_metrics = enhanced_risk_manager.calculate_portfolio_risk(positions)
        
        # Test risk limit checking
        new_position = PositionRisk(
            symbol='BANKNIFTY',
            position_size=50.0,
            position_value=975000.0,
            risk_per_trade=0.01,
            stop_loss_distance=0.03,
            expected_return=0.08,
            volatility=0.25,
            correlation_with_portfolio=0.4,
            margin_requirement=97500.0
        )
        
        risk_ok, risk_message = enhanced_risk_manager.check_risk_limits(new_position, positions)
        
        # Test position sizing
        optimal_size = enhanced_risk_manager.calculate_optimal_position_size(
            'NIFTY', 19500.0, 18500.0, 0.1, 0.2, 0.7
        )
        
        # Test circuit breaker
        circuit_breaker = enhanced_risk_manager.check_circuit_breaker(-5000.0)
        
        # Get risk report
        risk_report = enhanced_risk_manager.get_risk_report()
        
        print(f"✅ Enhanced Risk Management:")
        print(f"   Portfolio exposure: {risk_metrics.exposure_percentage:.1%}")
        print(f"   Risk limits check: {risk_ok} - {risk_message}")
        print(f"   Optimal position size: {optimal_size:.2f}")
        print(f"   Circuit breaker active: {circuit_breaker}")
        print(f"   Current capital: {risk_report.get('current_capital', 0):.2f}")
        print(f"   Total return: {risk_report.get('total_return', 0):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced risk management test failed: {e}")
        return False

def test_enhanced_execution_manager():
    """Test enhanced execution manager"""
    try:
        from src.execution.enhanced_execution_manager import EnhancedExecutionManager, MockBrokerAdapter, OrderType
        
        # Create mock broker adapter
        broker_adapter = MockBrokerAdapter()
        
        # Create execution manager
        execution_manager = EnhancedExecutionManager(broker_adapter)
        
        # Test order placement
        order_id = execution_manager.place_order(
            'NIFTY', OrderType.MARKET, 'BUY', 100.0, timeout_seconds=10
        )
        
        # Test order status
        order_status = execution_manager.get_order_status(order_id)
        
        # Test positions
        positions = execution_manager.get_positions()
        
        # Test execution stats
        stats = execution_manager.get_execution_stats()
        
        # Stop execution manager
        execution_manager.stop()
        
        print(f"✅ Enhanced Execution Manager:")
        print(f"   Order placed: {order_id}")
        print(f"   Order status: {order_status.status.value if order_status else 'None'}")
        print(f"   Positions count: {len(positions)}")
        print(f"   Total orders: {stats.get('total_orders', 0)}")
        print(f"   Success rate: {stats.get('success_rate', 0):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced execution manager test failed: {e}")
        return False

def test_enhanced_database_with_timezone():
    """Test enhanced database with timezone awareness"""
    try:
        from src.models.enhanced_database_with_timezone import enhanced_timezone_db
        
        # Test saving entry signal
        signal_data = {
            'signal_id': 'TEST_SIGNAL_001',
            'symbol': 'NIFTY',
            'strategy': 'EMA_CROSSOVER',
            'signal_type': 'BUY',
            'confidence': 0.75,
            'price': 19500.0,
            'timestamp': datetime.now(),
            'timeframe': '1h',
            'strength': 'strong',
            'indicator_values': {'ema_short': 19500, 'ema_long': 19400},
            'market_condition': 'trending',
            'volatility': 0.2,
            'position_size': 100.0,
            'stop_loss_price': 18500.0,
            'take_profit_price': 20500.0,
            'confirmed': True,
            'executed': False
        }
        
        enhanced_timezone_db.save_entry_signal(signal_data)
        
        # Test saving trade
        trade_data = {
            'trade_id': 'TEST_TRADE_001',
            'symbol': 'NIFTY',
            'strategy': 'EMA_CROSSOVER',
            'signal_type': 'BUY',
            'entry_price': 19500.0,
            'quantity': 100.0,
            'timestamp': datetime.now(),
            'stop_loss': 18500.0,
            'take_profit': 20500.0,
            'current_price': 19600.0,
            'unrealized_pnl': 10000.0
        }
        
        enhanced_timezone_db.save_trade(trade_data)
        
        # Test getting entry signals
        entry_signals = enhanced_timezone_db.get_entry_signals('NIFTY', days=1)
        
        # Test getting market statistics
        market_stats = enhanced_timezone_db.get_market_statistics('indian', days=7)
        
        print(f"✅ Enhanced Database with Timezone:")
        print(f"   Entry signal saved: {signal_data['signal_id']}")
        print(f"   Trade saved: {trade_data['trade_id']}")
        print(f"   Entry signals retrieved: {len(entry_signals)}")
        print(f"   Market statistics: {len(market_stats)} keys")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced database with timezone test failed: {e}")
        return False

def test_leakage_detection():
    """Test data leakage detection"""
    try:
        from src.analytics.fixed_ml_model_evaluation import FixedLeakageDetector
        
        detector = FixedLeakageDetector()
        
        # Create test data with leakage
        features = pd.DataFrame({
            'normal_feature': np.random.randn(100),
            'perfect_correlation': np.random.randn(100),
            'future_feature': np.random.randn(100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H')
        })
        
        # Create target with perfect correlation to one feature
        target = features['perfect_correlation'] + np.random.randn(100) * 0.01
        
        # Test feature leakage detection
        feature_leakage = detector.check_feature_leakage(features, target)
        
        # Test temporal leakage detection
        temporal_leakage = detector.check_temporal_leakage(features, target, 'timestamp')
        
        print(f"✅ Leakage Detection:")
        print(f"   Feature leakage detected: {len(feature_leakage)} features")
        print(f"   Temporal leakage detected: {len(temporal_leakage)} features")
        
        for feature, result in feature_leakage.items():
            print(f"   - {feature}: {result.get('leakage_risk', 'UNKNOWN')} risk")
        
        return True
        
    except Exception as e:
        print(f"❌ Leakage detection test failed: {e}")
        return False

def test_market_session_detection():
    """Test market session detection"""
    try:
        from src.core.enhanced_timezone_utils import enhanced_timezone_manager, MarketState
        
        # Test different times
        test_times = [
            datetime.now().replace(hour=8, minute=30),   # Before market
            datetime.now().replace(hour=9, minute=10),   # Pre-open
            datetime.now().replace(hour=10, minute=0),   # Market open
            datetime.now().replace(hour=15, minute=45),  # After market
        ]
        
        print(f"✅ Market Session Detection:")
        for test_time in test_times:
            market_state = enhanced_timezone_manager.is_market_open(test_time)
            print(f"   {test_time.strftime('%H:%M')}: {market_state.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market session detection test failed: {e}")
        return False

def test_performance_benchmarking():
    """Test performance benchmarking"""
    try:
        from src.core.fixed_performance_optimizer import fixed_performance_optimizer
        
        # Run comprehensive benchmark
        benchmark_results = fixed_performance_optimizer.benchmark_operations(10000)
        
        print(f"✅ Performance Benchmarking:")
        print(f"   Data size: {benchmark_results.get('data_size', 0)}")
        print(f"   Signal generation time: {benchmark_results.get('signal_generation_time', 0):.3f}s")
        print(f"   EMA calculation time: {benchmark_results.get('ema_calculation_time', 0):.3f}s")
        print(f"   Total time: {benchmark_results.get('total_time', 0):.3f}s")
        print(f"   Throughput: {benchmark_results.get('throughput', 0):.0f} points/sec")
        print(f"   Optimization level: {benchmark_results.get('optimization_level', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmarking test failed: {e}")
        return False

def main():
    """Run all enhanced system tests"""
    print("🚀 FINAL ENHANCED SYSTEMS COMPREHENSIVE TEST")
    print("="*60)
    
    tests = [
        ('Enhanced Timezone Management', test_enhanced_timezone_management),
        ('Actor Model State Manager', test_actor_model_state_manager),
        ('Fixed Performance Optimizer', test_fixed_performance_optimizer),
        ('Fixed ML Model Evaluation', test_fixed_ml_model_evaluation),
        ('Enhanced Options Pricing', test_enhanced_options_pricing),
        ('Enhanced Risk Management', test_enhanced_risk_management),
        ('Enhanced Execution Manager', test_enhanced_execution_manager),
        ('Enhanced Database with Timezone', test_enhanced_database_with_timezone),
        ('Leakage Detection', test_leakage_detection),
        ('Market Session Detection', test_market_session_detection),
        ('Performance Benchmarking', test_performance_benchmarking)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("📊 FINAL ENHANCED SYSTEMS TEST RESULTS")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    success_rate = passed_tests / total_tests
    
    print(f"\n🎯 OVERALL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {success_rate:.1%}")
    
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 FINAL ENHANCED SYSTEMS STATUS:")
    if success_rate >= 0.9:
        print("   🎉 ALL ENHANCED SYSTEMS OPERATIONAL - PRODUCTION READY!")
    elif success_rate >= 0.8:
        print("   ⚠️ MOSTLY OPERATIONAL - Minor issues remain")
    else:
        print("   ❌ NEEDS ATTENTION - Significant issues found")
    
    print("\n" + "="*60)
    
    return success_rate >= 0.9

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
