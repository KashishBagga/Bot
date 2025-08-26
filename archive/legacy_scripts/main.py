"""
Main module for the trading bot.
Handles command-line arguments and runs the appropriate mode.
"""
import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import modules after setting path
from src.config.settings import setup_logging
from src.strategies import get_available_strategies, get_strategy_class
from src.models.database import db
from src.services.telegram_service import TelegramService

# Setup logging
logger = setup_logging()

def run_backtest_mode(args):
    """Run the bot in backtest mode."""
    from src.core.backtest import Backtester
    
    logger.info("Starting trading bot in backtest mode")
    
    # Load strategies
    strategies = []
    for strategy_name in args.strategies:
        strategy_class = get_strategy_class(strategy_name)
        if strategy_class:
            strategies.append(strategy_class())
            logger.info(f"Loaded strategy: {strategy_name}")
        else:
            logger.error(f"Strategy not found: {strategy_name}")
    
    if not strategies:
        logger.error("No valid strategies to run. Exiting.")
        return
    
    # Create backtester
    backtester = Backtester(strategies)
    
    # Run backtest
    results = backtester.run(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe
    )
    
    # Display results
    for strategy_name, result in results.items():
        logger.info(f"Backtest results for {strategy_name}:")
        logger.info(f"  Total trades: {result['total_trades']}")
        logger.info(f"  Win rate: {result['win_rate']:.2%}")
        logger.info(f"  Profit factor: {result['profit_factor']:.2f}")
        logger.info(f"  Average profit: {result['avg_profit']:.2f}")
        logger.info(f"  Max drawdown: {result['max_drawdown']:.2f}")
    
    logger.info("Backtest completed")

def run_realtime_mode(args):
    """Run the bot in real-time mode."""
    from src.api.fyers import FyersAPI
    
    logger.info("Starting trading bot in real-time mode")
    
    # Initialize API
    api = FyersAPI()
    
    # Check if authentication is required
    if not api.is_authenticated():
        logger.info("Authentication required. Starting authentication process.")
        api.authenticate()
    
    if not api.is_authenticated():
        logger.error("Authentication failed. Exiting.")
        return
    
    logger.info("Authentication successful. Starting trading.")
    
    # Initialize Telegram service
    telegram = TelegramService()
    
    # Load strategies
    strategies = []
    for strategy_name in args.strategies:
        strategy_class = get_strategy_class(strategy_name)
        if strategy_class:
            strategies.append(strategy_class())
            logger.info(f"Loaded strategy: {strategy_name}")
        else:
            logger.error(f"Strategy not found: {strategy_name}")
    
    if not strategies:
        logger.error("No valid strategies to run. Exiting.")
        return
    
    # Start the trading loop
    from src.core.trading_loop import TradingLoop
    
    trading_loop = TradingLoop(api, strategies, telegram)
    trading_loop.start(symbols=args.symbols, timeframe=args.timeframe)

def run_test_mode(args):
    """Run the bot in test mode for development."""
    logger.info("Starting trading bot in test mode")
    
    # Load strategies
    strategies = []
    for strategy_name in args.strategies:
        strategy_class = get_strategy_class(strategy_name)
        if strategy_class:
            strategies.append(strategy_class())
            logger.info(f"Loaded strategy: {strategy_name}")
        else:
            logger.error(f"Strategy not found: {strategy_name}")
    
    if not strategies:
        logger.error("No valid strategies to run. Exiting.")
        return
    
    # Generate sample data
    from test_strategy import generate_sample_data
    
    df = generate_sample_data(days=10)
    
    # Run each strategy on the sample data
    for strategy in strategies:
        logger.info(f"Testing strategy: {strategy.name}")
        df_with_indicators = strategy.calculate_indicators(df)
        
        # Find signals
        signals = []
        for i in range(50, len(df_with_indicators)):
            data_slice = df_with_indicators.iloc[:i+1]
            signal = strategy.analyze(data_slice)
            if signal and signal.get('signal') != 'None':
                signals.append(signal)
        
        logger.info(f"Found {len(signals)} signals for strategy {strategy.name}")
        
        # Log signals to console for review
        if signals:
            logger.info(f"Sample signals for {strategy.name}:")
            for i, signal in enumerate(signals[:5]):  # Show first 5 signals
                logger.info(f"  Signal {i+1}: {signal.get('signal')} at {signal.get('price', 0):.2f}")
        
    logger.info("Test completed")

def main():
    """Main function to parse args and run the bot."""
    parser = argparse.ArgumentParser(description='Trading Bot')
    
    # Mode selection
    parser.add_argument('--mode', choices=['backtest', 'realtime', 'test'], default='test',
                        help='Bot operation mode')
    
    # Common arguments
    parser.add_argument('--symbols', nargs='+', default=['NIFTY'],
                        help='Symbols to trade')
    parser.add_argument('--timeframe', default='5minute',
                        help='Timeframe for the trading data')
    parser.add_argument('--strategies', nargs='+', default=None,
                        help='Strategies to use')
    
    # Backtest-specific arguments
    parser.add_argument('--start-date', default='2023-01-01',
                        help='Start date for backtest')
    parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date for backtest')
    
    args = parser.parse_args()
    
    # If no strategies provided, use all available
    if not args.strategies:
        args.strategies = get_available_strategies()
    
    # Run the appropriate mode
    if args.mode == 'backtest':
        run_backtest_mode(args)
    elif args.mode == 'realtime':
        run_realtime_mode(args)
    elif args.mode == 'test':
        run_test_mode(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main() 