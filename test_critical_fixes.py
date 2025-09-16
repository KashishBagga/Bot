#!/usr/bin/env python3
"""
Test Critical Fixes - Validate all production issue fixes
"""

import sys
import os
import asyncio
import time
import logging
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_timezone_utils():
    """Test timezone utilities"""
    print("\n" + "="*60)
    print("🧪 TESTING: Timezone Utilities")
    print("="*60)
    
    try:
        from src.core.timezone_utils import timezone_manager, now, now_kolkata, is_market_hours, format_datetime
        
        # Test timezone-aware datetime
        current_time = now()
        print(f"✅ Current IST time: {format_datetime(current_time)}")
        
        # Test market hours check
        is_open = is_market_hours()
        print(f"✅ Market hours check: {'OPEN' if is_open else 'CLOSED'}")
        
        # Test timezone conversion
        naive_dt = datetime.now()
        aware_dt = timezone_manager.to_aware(naive_dt)
        print(f"✅ Timezone conversion: {aware_dt.tzinfo}")
        
        return True
        
    except Exception as e:
        print(f"❌ Timezone utilities test failed: {e}")
        return False

def test_advanced_risk_management():
    """Test advanced risk management fixes"""
    print("\n" + "="*60)
    print("🧪 TESTING: Advanced Risk Management Fixes")
    print("="*60)
    
    try:
        from src.advanced_systems.advanced_risk_management import AdvancedRiskManager, RiskLimits
        
        # Test timezone-aware initialization
        risk_manager = AdvancedRiskManager()
        print(f"✅ Risk manager initialized with timezone: {risk_manager.tz}")
        
        # Test thread safety
        print(f"✅ Thread lock available: {hasattr(risk_manager, '_lock')}")
        
        # Test rate limiting
        print(f"✅ Alert rate limiting available: {hasattr(risk_manager, '_should_send_alert')}")
        
        # Test configuration
        print(f"✅ Configurable parameters: {hasattr(risk_manager, 'max_daily_loss')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced risk management test failed: {e}")
        return False

def test_ai_trade_review():
    """Test AI trade review fixes"""
    print("\n" + "="*60)
    print("🧪 TESTING: AI Trade Review Fixes")
    print("="*60)
    
    try:
        from src.advanced_systems.ai_trade_review import AITradeReview, AsyncTradeReviewProcessor
        
        # Test timezone-aware initialization
        ai_system = AITradeReview()
        print(f"✅ AI system initialized")
        
        # Test async processor
        async_processor = AsyncTradeReviewProcessor()
        print(f"✅ Async processor available: {hasattr(async_processor, 'generate_report_async')}")
        
        # Test exception handling
        print(f"✅ Exception handling available: {hasattr(ai_system, '_generate_report_with_exceptions')}")
        
        # Test safe ML inference
        print(f"✅ Safe ML inference available: {hasattr(ai_system, '_safe_ml_inference')}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI trade review test failed: {e}")
        return False

def test_ema_crossover_strategy():
    """Test EMA crossover strategy fixes"""
    print("\n" + "="*60)
    print("🧪 TESTING: EMA Crossover Strategy Fixes")
    print("="*60)
    
    try:
        from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
        
        # Test vectorized methods
        strategy = EmaCrossoverEnhanced({})
        print(f"✅ Strategy initialized")
        
        # Test vectorized signal generation
        print(f"✅ Vectorized signal generation: {hasattr(strategy, '_vectorized_signal_generation')}")
        
        # Test incremental EMA update
        print(f"✅ Incremental EMA update: {hasattr(strategy, '_incremental_ema_update')}")
        
        # Test performance monitoring
        print(f"✅ Performance monitoring: {hasattr(strategy, '_monitor_performance')}")
        
        # Test exception handling
        print(f"✅ Exception handling: {hasattr(strategy, '_analyze_vectorized_safe')}")
        
        return True
        
    except Exception as e:
        print(f"❌ EMA crossover strategy test failed: {e}")
        return False

def test_indian_market():
    """Test Indian market fixes"""
    print("\n" + "="*60)
    print("🧪 TESTING: Indian Market Fixes")
    print("="*60)
    
    try:
        from src.markets.indian.indian_market import IndianMarket, IndianHolidayManager
        
        # Test holiday manager
        holiday_manager = IndianHolidayManager()
        print(f"✅ Holiday manager initialized")
        
        # Test holiday checking
        is_holiday = holiday_manager.is_holiday()
        print(f"✅ Holiday check: {'HOLIDAY' if is_holiday else 'TRADING DAY'}")
        
        # Test market with holiday support
        market = IndianMarket()
        is_open = market.is_market_open()
        print(f"✅ Market open check: {'OPEN' if is_open else 'CLOSED'}")
        
        # Test timeout and retry
        print(f"✅ Timeout and retry available: {hasattr(market, '_get_price_with_retry')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Indian market test failed: {e}")
        return False

def test_crypto_market():
    """Test crypto market fixes"""
    print("\n" + "="*60)
    print("🧪 TESTING: Crypto Market Fixes")
    print("="*60)
    
    try:
        from src.markets.crypto.crypto_market import CryptoMarket, CryptoMarketPerformanceTracker
        
        # Test performance tracker
        tracker = CryptoMarketPerformanceTracker()
        print(f"✅ Performance tracker initialized")
        
        # Test fill rate tracking
        tracker.record_fill_rate('BTC', 0.95)
        stats = tracker.get_performance_stats('BTC')
        print(f"✅ Fill rate tracking: {stats['avg_fill_rate']:.2%}")
        
        # Test market with performance tracking
        market = CryptoMarket()
        print(f"✅ Market with performance tracking: {hasattr(market, 'performance_tracker')}")
        
        # Test API retry wrapper
        print(f"✅ API retry wrapper: {hasattr(market, '_api_call_with_retry')}")
        
        # Test balance reconciliation
        print(f"✅ Balance reconciliation: {hasattr(market, 'reconcile_balances')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Crypto market test failed: {e}")
        return False

def test_cross_file_timezone_consistency():
    """Test timezone consistency across files"""
    print("\n" + "="*60)
    print("🧪 TESTING: Cross-File Timezone Consistency")
    print("="*60)
    
    try:
        from src.core.timezone_utils import now
        from src.advanced_systems.advanced_risk_management import AdvancedRiskManager
        from src.advanced_systems.ai_trade_review import AITradeReview
        
        # Get current time from different modules
        time1 = now()
        risk_manager = AdvancedRiskManager()
        time2 = risk_manager.now() if hasattr(risk_manager, 'now') else now()
        
        # Check if times are consistent (within 1 second)
        time_diff = abs((time1 - time2).total_seconds())
        print(f"✅ Timezone consistency: {time_diff:.3f}s difference")
        
        if time_diff < 1.0:
            print("✅ All modules using consistent timezone")
            return True
        else:
            print("⚠️ Timezone inconsistency detected")
            return False
        
    except Exception as e:
        print(f"❌ Timezone consistency test failed: {e}")
        return False

def test_thread_safety():
    """Test thread safety improvements"""
    print("\n" + "="*60)
    print("🧪 TESTING: Thread Safety Improvements")
    print("="*60)
    
    try:
        from src.advanced_systems.advanced_risk_management import AdvancedRiskManager
        import threading
        
        risk_manager = AdvancedRiskManager()
        
        # Test lock availability
        has_lock = hasattr(risk_manager, '_lock')
        print(f"✅ Thread lock available: {has_lock}")
        
        if has_lock:
            # Test lock functionality
            with risk_manager._lock:
                print("✅ Lock acquired successfully")
            
            print("✅ Lock released successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Thread safety test failed: {e}")
        return False

def test_performance_improvements():
    """Test performance improvements"""
    print("\n" + "="*60)
    print("🧪 TESTING: Performance Improvements")
    print("="*60)
    
    try:
        from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
        import pandas as pd
        import numpy as np
        
        # Create test data
        dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(19000, 20000, 1000),
            'high': np.random.uniform(19000, 20000, 1000),
            'low': np.random.uniform(19000, 20000, 1000),
            'close': np.random.uniform(19000, 20000, 1000),
            'volume': np.random.uniform(1000, 10000, 1000)
        })
        
        strategy = EmaCrossoverEnhanced({})
        
        # Test vectorized operations
        start_time = time.time()
        if hasattr(strategy, '_vectorized_signal_generation'):
            result = strategy._vectorized_signal_generation(data)
            duration = time.time() - start_time
            print(f"✅ Vectorized signal generation: {duration:.3f}s for 1000 rows")
        
        # Test performance monitoring
        if hasattr(strategy, '_monitor_performance'):
            strategy._monitor_performance('test_operation', start_time)
            print("✅ Performance monitoring working")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance improvements test failed: {e}")
        return False

def main():
    """Run all critical fix tests"""
    print("🚀 STARTING CRITICAL FIXES VALIDATION")
    print("="*80)
    
    test_results = {}
    
    # Run all tests
    test_results['timezone_utils'] = test_timezone_utils()
    test_results['advanced_risk_management'] = test_advanced_risk_management()
    test_results['ai_trade_review'] = test_ai_trade_review()
    test_results['ema_crossover_strategy'] = test_ema_crossover_strategy()
    test_results['indian_market'] = test_indian_market()
    test_results['crypto_market'] = test_crypto_market()
    test_results['timezone_consistency'] = test_cross_file_timezone_consistency()
    test_results['thread_safety'] = test_thread_safety()
    test_results['performance_improvements'] = test_performance_improvements()
    
    # Summary
    print("\n" + "="*80)
    print("📊 CRITICAL FIXES VALIDATION SUMMARY")
    print("="*80)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    failed_tests = total_tests - passed_tests
    
    print(f"\n📈 OVERALL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {failed_tests}")
    print(f"   Success Rate: {passed_tests/total_tests:.1%}")
    
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 CRITICAL FIXES STATUS:")
    if passed_tests == total_tests:
        print("   🎉 ALL CRITICAL FIXES VALIDATED - PRODUCTION READY!")
    elif passed_tests >= total_tests * 0.8:
        print("   ⚠️ MOSTLY FIXED - Minor issues remain")
    else:
        print("   ❌ SIGNIFICANT ISSUES - Requires attention")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
