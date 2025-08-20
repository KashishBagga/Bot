#!/usr/bin/env python3
"""
Unified Backtesting Command
Single entry point for all backtesting operations using parquet data only
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from backtest_config import BacktestConfig, print_backtest_config

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🚀 {description}")
    print(f"💻 Command: {' '.join(command)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {command[0]}")
        return False

def setup_data(symbols=None, timeframes=None):
    """Setup 20-year parquet data"""
    command = ["python3", "setup_20_year_parquet_data.py"]
    
    if symbols:
        command.extend(["--symbols", ",".join(symbols)])
    
    if timeframes:
        command.extend(["--timeframes", ",".join(timeframes)])
    
    return run_command(command, "Setting up 20-year parquet data")

def run_comprehensive_backtest(days=30, timeframe="15min", symbols=None, strategies=None, save_db=True):
    """Run comprehensive backtesting using parquet data"""
    command = ["python3", "all_strategies_parquet.py"]
    command.extend(["--days", str(days)])
    command.extend(["--timeframe", timeframe])
    
    if not save_db:
        command.append("--no-save")
    
    if symbols:
        command.extend(["--symbols", ",".join(symbols)])
    
    if strategies:
        command.extend(["--strategies", ",".join(strategies)])
    
    return run_command(command, f"Running comprehensive backtest ({days} days, {timeframe})")

def run_fast_backtest(days=30, timeframe="15min", symbols=None, strategies=None):
    """Run ultra-fast backtesting using parquet data"""
    command = ["python3", "backtesting_parquet.py"]
    command.extend(["--days", str(days)])
    command.extend(["--timeframe", timeframe])
    
    if symbols:
        command.extend(["--symbols", ",".join(symbols)])
    
    if strategies:
        command.extend(["--strategies", ",".join(strategies)])
    
    return run_command(command, f"Running ultra-fast backtest ({days} days, {timeframe})")

def check_data_status():
    """Check parquet data status"""
    print("🔍 CHECKING PARQUET DATA STATUS")
    print("=" * 60)
    
    status = BacktestConfig.check_setup_status()
    
    if not status["parquet_data_available"]:
        print("❌ No parquet data found!")
        print("\n💡 SETUP REQUIRED:")
        print("   Run: python3 backtest.py setup")
        return False
    
    print(f"✅ Parquet data available!")
    print(f"📊 Symbols: {status['total_symbols']}")
    print(f"📁 Files: {status['total_files']}")
    print(f"💾 Size: {status['total_size_mb']:.1f} MB")
    
    if status["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in status["recommendations"]:
            print(f"   • {rec}")
    
    print("\n📈 AVAILABLE SYMBOLS:")
    symbols = BacktestConfig.get_available_symbols()
    for symbol in symbols:
        timeframes = BacktestConfig.get_available_timeframes(symbol)
        print(f"   {symbol}: {len(timeframes)} timeframes")
    
    return True

def show_examples():
    """Show usage examples"""
    print("📚 BACKTESTING EXAMPLES")
    print("=" * 60)
    print()
    
    print("🔧 SETUP (First time only):")
    print("   python3 backtest.py setup")
    print("   python3 backtest.py setup --symbols NIFTY50,BANKNIFTY")
    print()
    
    print("🚀 QUICK BACKTESTS:")
    print("   python3 backtest.py run --days 7")
    print("   python3 backtest.py run --days 30 --timeframe 5min")
    print("   python3 backtest.py run --symbols NIFTY50")
    print()
    
    print("⚡ ULTRA-FAST BACKTESTS:")
    print("   python3 backtest.py fast --days 30")
    print("   python3 backtest.py fast --days 90 --timeframe 1hour")
    print()
    
    print("🎯 STRATEGY-SPECIFIC:")
    print("   python3 backtest.py run --strategies ema_crossover,supertrend_ema")
    print("   python3 backtest.py fast --strategies insidebar_rsi --days 60")
    print()
    
    print("📊 STATUS & INFO:")
    print("   python3 backtest.py status")
    print("   python3 backtest.py config")
    print("   python3 backtest.py examples")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Unified backtesting command (parquet data only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Setup data:     python3 backtest.py setup
  Quick backtest: python3 backtest.py run --days 30
  Fast backtest:  python3 backtest.py fast --days 30
  Check status:   python3 backtest.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup 20-year parquet data')
    setup_parser.add_argument('--symbols', type=str, help='Comma-separated symbols (default: all major indices)')
    setup_parser.add_argument('--timeframes', type=str, help='Comma-separated timeframes (default: all)')
    
    # Run command (comprehensive backtest)
    run_parser = subparsers.add_parser('run', help='Run comprehensive backtesting')
    run_parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    run_parser.add_argument('--timeframe', type=str, default='15min', help='Timeframe')
    run_parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    run_parser.add_argument('--strategies', type=str, help='Comma-separated strategies')
    run_parser.add_argument('--no-save', action='store_true', help="Don't save to database")
    
    # Fast command (ultra-fast backtest)
    fast_parser = subparsers.add_parser('fast', help='Run ultra-fast backtesting')
    fast_parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    fast_parser.add_argument('--timeframe', type=str, default='15min', help='Timeframe')
    fast_parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    fast_parser.add_argument('--strategies', type=str, help='Comma-separated strategies')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check data status')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    
    # Examples command
    examples_parser = subparsers.add_parser('examples', help='Show usage examples')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("🤖 UNIFIED BACKTESTING SYSTEM")
    print("📊 Parquet Data Only - No API Calls")
    print("=" * 60)
    
    if args.command == 'setup':
        symbols = args.symbols.split(',') if args.symbols else None
        timeframes = args.timeframes.split(',') if args.timeframes else None
        success = setup_data(symbols=symbols, timeframes=timeframes)
        sys.exit(0 if success else 1)
    
    elif args.command == 'run':
        symbols = args.symbols.split(',') if args.symbols else None
        strategies = args.strategies.split(',') if args.strategies else None
        success = run_comprehensive_backtest(
            days=args.days,
            timeframe=args.timeframe,
            symbols=symbols,
            strategies=strategies,
            save_db=not args.no_save
        )
        sys.exit(0 if success else 1)
    
    elif args.command == 'fast':
        symbols = args.symbols.split(',') if args.symbols else None
        strategies = args.strategies.split(',') if args.strategies else None
        success = run_fast_backtest(
            days=args.days,
            timeframe=args.timeframe,
            symbols=symbols,
            strategies=strategies
        )
        sys.exit(0 if success else 1)
    
    elif args.command == 'status':
        check_data_status()
    
    elif args.command == 'config':
        print_backtest_config()
    
    elif args.command == 'examples':
        show_examples()

if __name__ == "__main__":
    main() 