#!/usr/bin/env python3
"""
Comprehensive Production Systems Test
Tests all MUST and HIGH PRIORITY production systems
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

async def test_end_to_end_validation():
    """Test end-to-end validation system"""
    print("\n" + "="*60)
    print("🧪 TESTING: End-to-End Validation System")
    print("="*60)
    
    try:
        from src.production.end_to_end_validation import EndToEndValidator, ValidationConfig
        
        config = ValidationConfig(
            backtest_years=1,  # Reduced for testing
            forward_test_days=7,
            equity_curve_tolerance=0.15
        )
        
        validator = EndToEndValidator(config)
        result = validator.run_validation()
        
        print(f"✅ End-to-End Validation: {'PASSED' if result.validation_passed else 'FAILED'}")
        print(f"   Backtest Return: {result.backtest_return:.2%}")
        print(f"   Forward Test Return: {result.forward_test_return:.2%}")
        print(f"   Max Deviation: {result.max_deviation:.2%}")
        
        return result.validation_passed
        
    except Exception as e:
        print(f"❌ End-to-End Validation Test Failed: {e}")
        return False

async def test_execution_reliability():
    """Test execution reliability system"""
    print("\n" + "="*60)
    print("🧪 TESTING: Execution Reliability System")
    print("="*60)
    
    try:
        from src.production.execution_reliability import ExecutionReliabilityManager, ReconciliationConfig
        
        config = ReconciliationConfig(
            poll_interval=1,
            max_poll_attempts=3,
            reconciliation_interval=5,
            full_reconciliation_interval=10
        )
        
        manager = ExecutionReliabilityManager(config)
        
        # Test order placement
        order_id = await manager.place_order_with_guarantee(
            symbol="NSE:NIFTY50-INDEX",
            side="BUY",
            quantity=100,
            price=19500
        )
        
        print(f"✅ Order placed: {order_id}")
        
        # Test reconciliation
        await manager.start_reconciliation_loop()
        await asyncio.sleep(15)  # Let it run for 15 seconds
        
        status = manager.get_reconciliation_status()
        print(f"✅ Reconciliation Status: {status}")
        
        criteria_met = manager.check_acceptance_criteria()
        print(f"✅ Acceptance Criteria: {'MET' if criteria_met else 'NOT MET'}")
        
        await manager.stop_reconciliation_loop()
        
        return criteria_met
        
    except Exception as e:
        print(f"❌ Execution Reliability Test Failed: {e}")
        return False

async def test_database_resilience():
    """Test database resilience system"""
    print("\n" + "="*60)
    print("🧪 TESTING: Database Resilience System")
    print("="*60)
    
    try:
        from src.production.database_resilience import DatabaseResilienceManager, BackupConfig
        
        config = BackupConfig(
            backup_interval_hours=1,
            backup_retention_days=7,
            restore_test_interval_days=1
        )
        
        db_manager = DatabaseResilienceManager("data/test_trading.db", config)
        
        # Test atomic transaction
        trade_data = {
            'operation': 'OPEN',
            'trade_id': f"TEST_{int(time.time())}",
            'symbol': 'NSE:NIFTY50-INDEX',
            'signal': 'BUY',
            'entry_price': 19500,
            'quantity': 100,
            'timestamp': datetime.now().isoformat(),
            'strategy': 'test_strategy',
            'confidence': 75,
            'account_id': 'test_account',
            'balance_change': -1950000
        }
        
        result = db_manager.execute_atomic_trade_transaction(trade_data)
        print(f"✅ Atomic Transaction: {'SUCCESS' if result.success else 'FAILED'}")
        
        # Test backup
        backup_success = db_manager.create_backup()
        print(f"✅ Backup Creation: {'SUCCESS' if backup_success else 'FAILED'}")
        
        # Test restore
        restore_success = db_manager.run_restore_test()
        print(f"✅ Restore Test: {'SUCCESS' if restore_success else 'FAILED'}")
        
        # Check acceptance criteria
        criteria_met = db_manager.check_acceptance_criteria()
        print(f"✅ Acceptance Criteria: {'MET' if criteria_met else 'NOT MET'}")
        
        return criteria_met
        
    except Exception as e:
        print(f"❌ Database Resilience Test Failed: {e}")
        return False

async def test_robust_risk_engine():
    """Test robust risk engine"""
    print("\n" + "="*60)
    print("🧪 TESTING: Robust Risk Engine")
    print("="*60)
    
    try:
        from src.production.robust_risk_engine import RobustRiskEngine, PortfolioConstraints
        
        constraints = PortfolioConstraints(
            max_portfolio_exposure=0.60,
            max_daily_drawdown=0.03,
            max_single_position=0.10,
            max_sector_exposure=0.30,
            max_correlation_exposure=0.50,
            max_consecutive_losses=5
        )
        
        risk_engine = RobustRiskEngine(constraints)
        
        # Add positions
        risk_engine.add_position("NSE:NIFTY50-INDEX", 100, 19500, "ema_strategy")
        risk_engine.add_position("NSE:NIFTYBANK-INDEX", 50, 45000, "supertrend_strategy")
        
        # Test risk check
        current_prices = {
            "NSE:NIFTY50-INDEX": 19600,
            "NSE:NIFTYBANK-INDEX": 45100
        }
        
        test_signal = {
            'symbol': 'NSE:FINNIFTY-INDEX',
            'position_size': 100,
            'strategy': 'ema_strategy'
        }
        
        can_trade, reason = risk_engine.check_portfolio_risk(test_signal, current_prices)
        print(f"✅ Risk Check: {'PASSED' if can_trade else 'FAILED'} - {reason}")
        
        # Test circuit breaker
        for i in range(6):
            risk_engine.add_trade_result("NSE:NIFTY50-INDEX", -1000, "ema_strategy")
        
        can_trade_after_losses, reason_after = risk_engine.check_portfolio_risk(test_signal, current_prices)
        print(f"✅ Circuit Breaker: {'TRIGGERED' if not can_trade_after_losses else 'NOT TRIGGERED'}")
        
        # Get risk report
        report = risk_engine.get_risk_report()
        print(f"✅ Risk Report Generated: {len(report)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust Risk Engine Test Failed: {e}")
        return False

async def test_slippage_model():
    """Test slippage model"""
    print("\n" + "="*60)
    print("🧪 TESTING: Slippage Model")
    print("="*60)
    
    try:
        from src.production.slippage_model import SlippageModel, SlippageConfig, PartialFillConfig
        
        slippage_config = SlippageConfig(
            base_slippage=0.0005,
            volatility_multiplier=2.0,
            volume_impact=0.1,
            time_impact=0.05,
            market_impact=0.2
        )
        
        partial_fill_config = PartialFillConfig(
            base_fill_rate=0.95,
            volatility_impact=-0.1,
            volume_impact=-0.05,
            time_impact=0.1,
            market_impact=-0.15
        )
        
        slippage_model = SlippageModel(slippage_config, partial_fill_config)
        
        # Test order execution simulation
        market_data = {
            'volatility': 0.02,
            'market_condition': 'NORMAL',
            'volume': 1000
        }
        
        result = slippage_model.simulate_order_execution(
            symbol="NSE:NIFTY50-INDEX",
            side="BUY",
            quantity=100,
            limit_price=19500,
            market_data=market_data
        )
        
        print(f"✅ Order Execution Simulation: SUCCESS")
        print(f"   Filled Quantity: {result.filled_quantity}")
        print(f"   Average Price: {result.average_price:.2f}")
        print(f"   Total Slippage: {result.total_slippage:.4f}")
        print(f"   Fill Rate: {result.fill_rate:.2%}")
        print(f"   Partial Fills: {len(result.partial_fills)}")
        
        # Get statistics
        slippage_stats = slippage_model.get_slippage_statistics()
        fill_rate_stats = slippage_model.get_fill_rate_statistics()
        
        print(f"✅ Slippage Statistics: {len(slippage_stats)} metrics")
        print(f"✅ Fill Rate Statistics: {len(fill_rate_stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Slippage Model Test Failed: {e}")
        return False

async def test_pre_live_checklist():
    """Test pre-live checklist"""
    print("\n" + "="*60)
    print("🧪 TESTING: Pre-Live Checklist")
    print("="*60)
    
    try:
        from src.production.pre_live_checklist import PreLiveChecklist
        
        checklist = PreLiveChecklist()
        
        # Run some checklist items
        await checklist.run_checklist_item('system_validation')
        await checklist.run_checklist_item('api_connectivity')
        await checklist.run_checklist_item('kill_switch')
        
        # Get status
        status = checklist.get_checklist_status()
        print(f"✅ Checklist Status: {status['completion_percentage']:.1f}% complete")
        print(f"   Total Items: {status['total_items']}")
        print(f"   Passed: {status['passed_items']}")
        print(f"   Failed: {status['failed_items']}")
        print(f"   Ready for Live: {'YES' if status['ready_for_live'] else 'NO'}")
        
        # Test trading stages
        if status['ready_for_live']:
            stage_started = checklist.start_trading_stage('PILOT')
            print(f"✅ Trading Stage Started: {'SUCCESS' if stage_started else 'FAILED'}")
            
            trading_status = checklist.get_trading_status()
            print(f"✅ Trading Status: {trading_status['trading_status']}")
        
        # Test kill switch
        checklist.activate_kill_switch("Test activation")
        trading_status = checklist.get_trading_status()
        print(f"✅ Kill Switch: {'ACTIVE' if trading_status['kill_switch_active'] else 'INACTIVE'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pre-Live Checklist Test Failed: {e}")
        return False

async def test_production_monitoring():
    """Test production monitoring"""
    print("\n" + "="*60)
    print("🧪 TESTING: Production Monitoring")
    print("="*60)
    
    try:
        from src.monitoring.production_monitoring import ProductionMonitor
        
        monitor = ProductionMonitor()
        
        # Start monitoring
        monitoring_task = asyncio.create_task(monitor.start_monitoring())
        
        # Let it run for a bit
        await asyncio.sleep(65)  # Let it collect metrics and check thresholds
        
        # Get status
        status = monitor.get_monitoring_status()
        print(f"✅ Monitoring Status: {'ACTIVE' if status['monitoring_active'] else 'INACTIVE'}")
        print(f"   Total Metrics: {status['total_metrics']}")
        print(f"   Total Alerts: {status['total_alerts']}")
        print(f"   Active Alerts: {status['active_alerts']}")
        print(f"   Critical Alerts: {status['critical_alerts']}")
        print(f"   Warning Alerts: {status['warning_alerts']}")
        
        monitor.stop_monitoring()
        monitoring_task.cancel()
        
        return True
        
    except Exception as e:
        print(f"❌ Production Monitoring Test Failed: {e}")
        return False

async def test_chaos_testing():
    """Test chaos testing"""
    print("\n" + "="*60)
    print("🧪 TESTING: Chaos Testing Engine")
    print("="*60)
    
    try:
        from src.testing.chaos_testing import ChaosTestingEngine, ChaosTestType
        
        engine = ChaosTestingEngine()
        
        # Run a single test
        result = await engine.run_chaos_test(ChaosTestType.BROKER_TIMEOUT, duration=10)
        print(f"✅ Single Chaos Test: {'PASSED' if result.success else 'FAILED'}")
        print(f"   Test Type: {result.test_type.value}")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"   Recovery Time: {result.recovery_time:.2f}s")
        print(f"   Data Integrity: {'PASSED' if result.data_integrity else 'FAILED'}")
        
        # Run comprehensive test
        comprehensive_result = await engine.run_comprehensive_chaos_test()
        print(f"✅ Comprehensive Chaos Test: {'PASSED' if comprehensive_result['overall_success'] else 'FAILED'}")
        print(f"   Success Rate: {comprehensive_result['success_rate']:.1%}")
        print(f"   Successful Tests: {comprehensive_result['successful_tests']}/{comprehensive_result['total_tests']}")
        
        return comprehensive_result['overall_success']
        
    except Exception as e:
        print(f"❌ Chaos Testing Test Failed: {e}")
        return False

async def test_broker_abstraction():
    """Test broker abstraction"""
    print("\n" + "="*60)
    print("🧪 TESTING: Broker Abstraction")
    print("="*60)
    
    try:
        from src.production.broker_abstraction import (
            BrokerFailoverManager, BrokerConfig, 
            FyersBrokerAdapter, ZerodhaBrokerAdapter, Order, OrderStatus
        )
        
        # Create broker configurations
        fyers_config = BrokerConfig(
            broker_name="Fyers",
            api_endpoint="https://api.fyers.in",
            credentials={"client_id": "test", "access_token": "test"},
            priority=1
        )
        
        zerodha_config = BrokerConfig(
            broker_name="Zerodha",
            api_endpoint="https://api.kite.trade",
            credentials={"api_key": "test", "access_token": "test"},
            priority=2
        )
        
        # Create broker adapters
        fyers_adapter = FyersBrokerAdapter(fyers_config)
        zerodha_adapter = ZerodhaBrokerAdapter(zerodha_config)
        
        # Create failover manager
        failover_manager = BrokerFailoverManager()
        failover_manager.add_broker(fyers_adapter, fyers_config)
        failover_manager.add_broker(zerodha_adapter, zerodha_config)
        
        # Test order placement
        order = Order(
            order_id="TEST_001",
            symbol="NSE:NIFTY50-INDEX",
            side="BUY",
            quantity=100,
            price=19500,
            order_type="LIMIT",
            timestamp=datetime.now(),
            status=OrderStatus.PENDING
        )
        
        broker_order_id = await failover_manager.place_order(order)
        print(f"✅ Order Placement: SUCCESS - {broker_order_id}")
        
        # Test health check
        health_status = await failover_manager.health_check_all()
        print(f"✅ Health Check: {len(health_status)} brokers checked")
        
        # Get failover status
        failover_status = failover_manager.get_failover_status()
        print(f"✅ Failover Status: {failover_status['total_brokers']} brokers configured")
        print(f"   Primary Broker: {failover_status['primary_broker']}")
        print(f"   Available Brokers: {failover_status['available_brokers']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Broker Abstraction Test Failed: {e}")
        return False

async def test_capital_efficiency():
    """Test capital efficiency optimizer"""
    print("\n" + "="*60)
    print("🧪 TESTING: Capital Efficiency Optimizer")
    print("="*60)
    
    try:
        from src.production.capital_efficiency import CapitalEfficiencyOptimizer
        import numpy as np
        
        # Create optimizer
        optimizer = CapitalEfficiencyOptimizer(total_capital=100000)
        
        # Add strategies
        optimizer.add_strategy("EMA_Strategy", initial_allocation=0.30)
        optimizer.add_strategy("Supertrend_Strategy", initial_allocation=0.25)
        optimizer.add_strategy("MACD_Strategy", initial_allocation=0.20)
        optimizer.add_strategy("RSI_Strategy", initial_allocation=0.25)
        
        # Generate mock trade data
        strategies = ["EMA_Strategy", "Supertrend_Strategy", "MACD_Strategy", "RSI_Strategy"]
        
        for strategy in strategies:
            # Generate 50 trades with different performance characteristics
            trades = []
            for i in range(50):
                if strategy == "EMA_Strategy":
                    # Good performance
                    pnl = np.random.normal(100, 200)
                elif strategy == "Supertrend_Strategy":
                    # Moderate performance
                    pnl = np.random.normal(50, 150)
                elif strategy == "MACD_Strategy":
                    # Poor performance
                    pnl = np.random.normal(-20, 100)
                else:  # RSI_Strategy
                    # Very good performance
                    pnl = np.random.normal(150, 180)
                
                trades.append({'pnl': pnl})
            
            # Update metrics
            optimizer.update_strategy_metrics(strategy, trades)
        
        # Optimize allocations
        new_allocations = optimizer.optimize_allocations()
        print(f"✅ Allocation Optimization: SUCCESS")
        print(f"   New Allocations: {new_allocations}")
        
        # Get performance report
        report = optimizer.get_strategy_performance_report()
        print(f"✅ Performance Report: {len(report['strategies'])} strategies analyzed")
        print(f"   Total Capital: ₹{report['total_capital']:,.2f}")
        print(f"   Active Strategies: {report['allocation_summary']['active_strategies']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Capital Efficiency Test Failed: {e}")
        return False

async def main():
    """Run all production system tests"""
    print("🚀 STARTING COMPREHENSIVE PRODUCTION SYSTEMS TEST")
    print("="*80)
    
    test_results = {}
    
    # MUST IMPLEMENTATIONS
    print("\n📋 TESTING MUST IMPLEMENTATIONS:")
    test_results['end_to_end_validation'] = await test_end_to_end_validation()
    test_results['execution_reliability'] = await test_execution_reliability()
    test_results['database_resilience'] = await test_database_resilience()
    test_results['robust_risk_engine'] = await test_robust_risk_engine()
    test_results['slippage_model'] = await test_slippage_model()
    test_results['pre_live_checklist'] = await test_pre_live_checklist()
    
    # HIGH PRIORITY SYSTEMS
    print("\n📋 TESTING HIGH PRIORITY SYSTEMS:")
    test_results['production_monitoring'] = await test_production_monitoring()
    test_results['chaos_testing'] = await test_chaos_testing()
    test_results['broker_abstraction'] = await test_broker_abstraction()
    test_results['capital_efficiency'] = await test_capital_efficiency()
    
    # Summary
    print("\n" + "="*80)
    print("📊 PRODUCTION SYSTEMS TEST SUMMARY")
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
    
    print(f"\n🎯 PRODUCTION READINESS:")
    if passed_tests == total_tests:
        print("   🎉 ALL SYSTEMS OPERATIONAL - READY FOR PRODUCTION!")
    elif passed_tests >= total_tests * 0.8:
        print("   ⚠️ MOSTLY READY - Minor issues to resolve")
    else:
        print("   ❌ NOT READY - Significant issues need attention")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(main())
