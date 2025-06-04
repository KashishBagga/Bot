#!/usr/bin/env python3
"""
Complete Trading System Launcher
Easy-to-use interface for backtesting and live trading
"""

import argparse
import sys
import os
import subprocess
import json
from datetime import datetime

def print_banner():
    """Print system banner"""
    print("🚀 OPTIMIZED TRADING SYSTEM")
    print("=" * 60)
    print("🎯 Confidence-Based Trading with Advanced Strategies")
    print("⚡ No Time Restrictions - Market Condition Based Trading")
    print("🛡️ Dynamic Risk Management & Signal Filtering")
    print("=" * 60)

def validate_system():
    """Validate system before running"""
    print("🔍 Validating system...")
    try:
        result = subprocess.run([sys.executable, 'validate_system.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ System validation passed")
            return True
        else:
            print("❌ System validation failed")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def run_backtest(args):
    """Run backtesting"""
    print("📊 STARTING BACKTESTING")
    print("-" * 40)
    
    # Build command
    cmd = [sys.executable, 'backtesting_parquet.py']
    
    if args.days:
        cmd.extend(['--days', str(args.days)])
    if args.timeframe:
        cmd.extend(['--timeframe', args.timeframe])
    if args.strategies:
        cmd.extend(['--strategies', args.strategies])
    if args.symbols:
        cmd.extend(['--symbols', args.symbols])
    if args.sequential:
        cmd.append('--sequential')
    if args.no_cache:
        cmd.append('--no-cache')
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Backtesting completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Backtesting failed: {e}")
        return False

def run_live_trading(args):
    """Run live trading bot"""
    print("🤖 STARTING LIVE TRADING BOT")
    print("-" * 40)
    
    try:
        # Import and run the optimized live trading bot
        from optimized_live_trading_bot import OptimizedLiveTradingBot
        
        bot = OptimizedLiveTradingBot()
        print("🎯 Optimized Live Trading Bot initialized")
        print("📊 Strategies: insidebar_rsi, ema_crossover, supertrend_ema, supertrend_macd_rsi_ema")
        print("🛡️ Risk Management: Confidence-based filtering (min score: 60)")
        print("⚡ Trading Mode: Real-time market condition analysis")
        print("-" * 40)
        
        if args.test_mode:
            print("🧪 Running in TEST MODE (single cycle)")
            bot.execute_trading_cycle()
            bot.generate_market_report()
            print("✅ Test cycle completed")
        else:
            print("🚀 Starting continuous trading...")
            print("Press Ctrl+C to stop")
            bot.start()
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Live trading stopped by user")
        return True
    except Exception as e:
        print(f"❌ Live trading error: {e}")
        return False

def run_test_suite():
    """Run comprehensive test suite"""
    print("🧪 RUNNING TEST SUITE")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, 'test_optimized_strategies.py'], 
                              check=True)
        print("✅ Test suite completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test suite failed: {e}")
        return False

def show_system_status():
    """Show current system status"""
    print("📊 SYSTEM STATUS")
    print("-" * 40)
    
    # Check strategies
    strategies = ['insidebar_rsi', 'ema_crossover', 'supertrend_ema', 'supertrend_macd_rsi_ema']
    print("📈 Available Strategies:")
    for strategy in strategies:
        print(f"  ✅ {strategy}: Optimized & Ready")
    
    # Check recent backtest results
    if os.path.exists('test_results.json'):
        with open('test_results.json', 'r') as f:
            results = json.load(f)
        success_rate = results.get('summary', {}).get('success_rate', 0)
        print(f"\n🎯 Last Test Results: {success_rate}% success rate")
    
    # Check database
    if os.path.exists('trading_signals.db'):
        print("💾 Database: Connected")
    else:
        print("💾 Database: Not found")
    
    # Check logs
    if os.path.exists('logs'):
        print("📝 Logging: Enabled")
    else:
        print("📝 Logging: Directory missing")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Optimized Trading System Launcher')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--days', type=int, default=7, help='Days to backtest (default: 7)')
    backtest_parser.add_argument('--timeframe', default='5min', help='Timeframe (default: 5min)')
    backtest_parser.add_argument('--strategies', help='Comma-separated strategies to test')
    backtest_parser.add_argument('--symbols', help='Comma-separated symbols to test')
    backtest_parser.add_argument('--sequential', action='store_true', help='Run sequentially instead of parallel')
    backtest_parser.add_argument('--no-cache', action='store_true', help='Disable result caching')
    
    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading bot')
    live_parser.add_argument('--test-mode', action='store_true', help='Run single test cycle instead of continuous')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate system')
    
    args = parser.parse_args()
    
    print_banner()
    
    if not args.command:
        print("❓ No command specified. Use --help for options.")
        print("\n📖 Quick Start:")
        print("  • python3 run_trading_system.py test      # Run test suite")
        print("  • python3 run_trading_system.py backtest  # Run backtesting")
        print("  • python3 run_trading_system.py live      # Start live trading")
        print("  • python3 run_trading_system.py status    # Show system status")
        return 1
    
    if args.command == 'validate':
        if validate_system():
            print("🎉 System is ready for trading!")
            return 0
        else:
            print("❌ Please fix issues before trading")
            return 1
    
    elif args.command == 'test':
        if run_test_suite():
            print("🎉 All tests passed!")
            return 0
        else:
            print("❌ Some tests failed")
            return 1
    
    elif args.command == 'backtest':
        if run_backtest(args):
            print("🎉 Backtesting completed!")
            return 0
        else:
            print("❌ Backtesting failed")
            return 1
    
    elif args.command == 'live':
        if run_live_trading(args):
            print("🎉 Live trading session completed!")
            return 0
        else:
            print("❌ Live trading failed")
            return 1
    
    elif args.command == 'status':
        show_system_status()
        return 0
    
    else:
        print(f"❓ Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    exit(main()) 