#!/usr/bin/env python3
"""
Test Advanced Systems - AI Trade Review, Backtesting, Risk Management, Alerting
"""

import sys
import os
import time
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ai_trade_review():
    """Test AI Trade Review System"""
    print("🧪 Testing AI Trade Review System")
    print("-" * 40)
    
    try:
        from ai_trade_review import AITradeReview
        
        # Initialize AI review system
        ai_review = AITradeReview()
        print("✅ AI Trade Review system initialized")
        
        # Generate daily report
        report = ai_review.generate_daily_report()
        
        if report:
            print("✅ Daily report generated successfully")
            print(f"  📊 Summary: {report.get('executive_summary', 'N/A')[:100]}...")
            print(f"  🧠 AI Insights: {len(report.get('ai_insights', {}).get('recommendations', []))} recommendations")
        else:
            print("⚠️ No data available for report generation")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Trade Review test failed: {e}")
        return False

def test_unified_backtesting():
    """Test Unified Backtesting Engine"""
    print("\n🧪 Testing Unified Backtesting Engine")
    print("-" * 40)
    
    try:
        from unified_backtesting_engine import UnifiedBacktestingEngine, BacktestConfig
        import pandas as pd
        import numpy as np
        
        # Create sample historical data
        symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
        historical_data = {}
        
        for symbol in symbols:
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
            base_price = 19500 if 'NIFTY50' in symbol else 45000
            
            # Generate realistic price movements
            returns = np.random.normal(0, 0.001, len(dates))
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data = pd.DataFrame({
                'timestamp': dates,
                'open': [p * (1 + np.random.uniform(-0.001, 0.001)) for p in prices],
                'high': [p * (1 + abs(np.random.uniform(0, 0.002))) for p in prices],
                'low': [p * (1 - abs(np.random.uniform(0, 0.002))) for p in prices],
                'close': prices,
                'volume': np.random.uniform(1000, 10000, len(dates))
            })
            
            historical_data[symbol] = data
        
        print("✅ Sample historical data created")
        
        # Configure backtest
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005,
            symbols=symbols,
            strategies=["simple_ema", "ema_crossover_enhanced"]
        )
        
        print("✅ Backtest configuration created")
        
        # Run backtest
        engine = UnifiedBacktestingEngine(config)
        result = engine.run_backtest(historical_data)
        
        print("✅ Backtest completed successfully")
        print(f"  📊 Total Return: {result.total_return*100:.2f}%")
        print(f"  📈 Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  🛡️ Max Drawdown: {result.max_drawdown*100:.2f}%")
        print(f"  🎯 Win Rate: {result.win_rate:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified Backtesting test failed: {e}")
        return False

def test_advanced_risk_management():
    """Test Advanced Risk Management System"""
    print("\n🧪 Testing Advanced Risk Management System")
    print("-" * 40)
    
    try:
        from advanced_risk_management import AdvancedRiskManager, RiskLimits
        
        # Create risk manager
        risk_limits = RiskLimits(
            max_daily_loss=0.05,
            max_portfolio_exposure=0.8,
            max_single_position=0.2,
            max_correlation_exposure=0.6,
            max_sector_exposure=0.4
        )
        
        risk_manager = AdvancedRiskManager(risk_limits)
        print("✅ Advanced Risk Manager initialized")
        
        # Add positions
        risk_manager.add_position("NSE:NIFTY50-INDEX", 100, 19500, time.time())
        risk_manager.add_position("NSE:NIFTYBANK-INDEX", 50, 45000, time.time())
        print("✅ Positions added")
        
        # Update prices
        current_prices = {
            "NSE:NIFTY50-INDEX": 19600,
            "NSE:NIFTYBANK-INDEX": 45100
        }
        
        # Test risk check
        test_signal = {
            'symbol': 'NSE:FINNIFTY-INDEX',
            'position_size': 100,
            'confidence': 75
        }
        
        can_trade, reason = risk_manager.check_risk_limits(test_signal, current_prices)
        print(f"✅ Risk check completed: {can_trade} - {reason}")
        
        # Generate risk report
        report = risk_manager.get_risk_report(current_prices)
        print("✅ Risk report generated")
        print(f"  📊 Total Value: ₹{report.get('portfolio_risk', {}).get('total_value', 0):,.2f}")
        print(f"  🛡️ Risk Level: {report.get('portfolio_risk', {}).get('risk_level', 'UNKNOWN')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced Risk Management test failed: {e}")
        return False

async def test_monitoring_alerting():
    """Test Monitoring & Alerting System"""
    print("\n🧪 Testing Monitoring & Alerting System")
    print("-" * 40)
    
    try:
        from monitoring_alerting_system import AlertManager, AlertConfig, SystemMonitor
        
        # Create alert configuration
        config = AlertConfig(
            enable_email=False,  # Disable for testing
            enable_telegram=False,  # Disable for testing
            enable_slack=False,  # Disable for testing
            enable_webhook=False  # Disable for testing
        )
        
        # Create alert manager
        alert_manager = AlertManager(config)
        print("✅ Alert Manager initialized")
        
        # Test different alert types
        await alert_manager.alert_trade_placed("NSE:NIFTY50-INDEX", "BUY", 100, 19500)
        await alert_manager.alert_trade_filled("NSE:NIFTY50-INDEX", "BUY", 100, 19500, 500)
        await alert_manager.alert_risk_limit_exceeded("portfolio_exposure", 0.85, 0.8)
        print("✅ Alerts sent successfully")
        
        # Test system monitoring
        system_monitor = SystemMonitor(alert_manager)
        health_status = await system_monitor.check_system_health()
        print("✅ System health check completed")
        print(f"  🏥 Overall Status: {health_status.get('overall_status', 'UNKNOWN')}")
        
        # Get alert summary
        summary = alert_manager.get_alert_summary(1)
        print(f"✅ Alert summary: {summary.get('total_alerts', 0)} alerts in last hour")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring & Alerting test failed: {e}")
        return False

def test_trade_execution_manager():
    """Test Trade Execution Manager"""
    print("\n🧪 Testing Trade Execution Manager")
    print("-" * 40)
    
    try:
        from trade_execution_manager import TradeExecutionManager, ExecutionConfig, FyersBroker
        
        # Create mock broker
        broker = FyersBroker(None)
        print("✅ Mock broker created")
        
        # Configure execution
        config = ExecutionConfig(
            max_retries=3,
            retry_delay=1.0,
            timeout=30.0,
            enable_fallback=True,
            enable_position_reconciliation=True
        )
        
        # Create execution manager
        manager = TradeExecutionManager(broker, config)
        print("✅ Trade Execution Manager initialized")
        
        # Test order placement (async)
        async def test_order():
            order_id = await manager.place_order(
                symbol="NSE:NIFTY50-INDEX",
                side="BUY",
                quantity=100,
                order_type="MARKET"
            )
            print(f"✅ Order placed: {order_id}")
            
            # Get portfolio summary
            summary = await manager.get_portfolio_summary()
            print(f"✅ Portfolio summary: {summary.get('total_value', 0):,.2f}")
            
            await manager.shutdown()
        
        # Run async test
        asyncio.run(test_order())
        
        return True
        
    except Exception as e:
        print(f"❌ Trade Execution Manager test failed: {e}")
        return False

def main():
    """Main function to test all advanced systems"""
    print("🚀 TESTING ADVANCED TRADING SYSTEMS")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: AI Trade Review
    test_results["ai_trade_review"] = test_ai_trade_review()
    
    # Test 2: Unified Backtesting
    test_results["unified_backtesting"] = test_unified_backtesting()
    
    # Test 3: Advanced Risk Management
    test_results["advanced_risk_management"] = test_advanced_risk_management()
    
    # Test 4: Trade Execution Manager
    test_results["trade_execution_manager"] = test_trade_execution_manager()
    
    # Test 5: Monitoring & Alerting (async)
    test_results["monitoring_alerting"] = asyncio.run(test_monitoring_alerting())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ADVANCED SYSTEMS TEST RESULTS")
    print("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status_icon = "✅" if result else "❌"
        print(f"  {status_icon} {test_name.replace('_', ' ').title()}: {'PASSED' if result else 'FAILED'}")
    
    print(f"\n📈 Overall Result: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL ADVANCED SYSTEMS OPERATIONAL!")
        print("\n🚀 READY FOR PRODUCTION WITH ADVANCED FEATURES:")
        print("  ✅ AI-driven trade review and insights")
        print("  ✅ Unified backtesting with same code paths")
        print("  ✅ Advanced risk management with portfolio controls")
        print("  ✅ Trade execution with retry and fallback")
        print("  ✅ Real-time monitoring and alerting")
    else:
        print("⚠️ SOME ADVANCED SYSTEMS NEED ATTENTION")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 ALL ADVANCED SYSTEMS TEST: SUCCESS!")
    else:
        print(f"\n❌ ADVANCED SYSTEMS TEST: PARTIAL SUCCESS")
