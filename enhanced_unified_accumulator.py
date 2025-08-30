#!/usr/bin/env python3
"""
Enhanced Unified Options Data Accumulator
Single database solution with multi-symbol support and comprehensive analytics.
"""

import os
import sys
import time
import signal
import argparse
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.fyers import FyersClient
from src.models.unified_database_enhanced import UnifiedDatabaseEnhanced

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('enhanced_unified_accumulator')

class EnhancedUnifiedAccumulator:
    """Enhanced unified options data accumulator with multi-symbol support."""
    
    def __init__(self, symbols: List[str], interval: int = 30):
        self.symbols = symbols
        self.interval = interval
        self.running = False
        
        # Initialize enhanced unified database
        self.database = UnifiedDatabaseEnhanced()
        
        # Initialize Fyers client
        logger.info("🔐 Initializing Fyers client...")
        try:
            # Load access token from .env
            from dotenv import load_dotenv
            load_dotenv()
            
            access_token = os.getenv('FYERS_ACCESS_TOKEN')
            if not access_token:
                raise Exception("FYERS_ACCESS_TOKEN not found in .env file")
            
            logger.info(f"✅ Got access token: {access_token[:20]}...")
            
            # Initialize Fyers client
            self.fyers_client = FyersClient()
            self.fyers_client.access_token = access_token
            if not self.fyers_client.initialize_client():
                raise Exception("Failed to initialize Fyers client")
            logger.info("✅ Fyers client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            raise
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        logger.info("🛑 Received interrupt signal")
        self.running = False
    
    def accumulate_options_data(self, symbol: str) -> bool:
        """Accumulate options data for a single symbol."""
        try:
            # Get raw option chain data from Fyers API
            raw_option_chain = self.fyers_client.get_option_chain(symbol)
            
            if not raw_option_chain:
                logger.warning(f"⚠️ Could not fetch raw option chain for {symbol}")
                return False
            
            # Save to enhanced unified database
            success = self.database.save_raw_options_chain(raw_option_chain)
            
            if success:
                logger.info(f"✅ Enhanced data accumulated for {symbol}")
            else:
                logger.warning(f"⚠️ Failed to save enhanced data for {symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error accumulating options data for {symbol}: {e}")
            return False
    
    def run(self):
        """Main accumulation loop with multi-symbol support."""
        logger.info("🚀 Starting ENHANCED UNIFIED options data accumulation")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(f"⏰ Interval: {self.interval} seconds")
        logger.info(f"🗄️ Database: Enhanced unified schema with analytics")
        
        self.running = True
        
        while self.running:
            try:
                start_time = time.time()
                
                # Accumulate data for each symbol
                for symbol in self.symbols:
                    logger.info(f"📊 Fetching enhanced options data for {symbol}...")
                    success = self.accumulate_options_data(symbol)
                    
                    if success:
                        logger.info(f"✅ Successfully accumulated enhanced data for {symbol}")
                    else:
                        logger.warning(f"⚠️ Failed to accumulate enhanced data for {symbol}")
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                
                if sleep_time > 0:
                    logger.info(f"⏰ Sleeping for {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                    
            except KeyboardInterrupt:
                logger.info("🛑 Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in main loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        logger.info("✅ Enhanced unified options data accumulation stopped")
    
    def get_database_summary(self) -> Dict:
        """Get enhanced database summary."""
        return self.database.get_database_summary()
    
    def get_analytics_data(self, symbol: str, start_date: str = None, end_date: str = None) -> Dict:
        """Get analytics data for a symbol."""
        return self.database.get_analytics_data(symbol, start_date, end_date)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Unified Options Data Accumulator')
    parser.add_argument('--symbols', nargs='+', 
                       default=['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX', 'NSE:FINNIFTY-INDEX'],
                       help='Trading symbols to accumulate data for')
    parser.add_argument('--interval', type=int, default=30,
                       help='Accumulation interval in seconds (default: 30)')
    parser.add_argument('--summary', action='store_true',
                       help='Show database summary and exit')
    parser.add_argument('--analytics', type=str, metavar='SYMBOL',
                       help='Show analytics data for a specific symbol')
    parser.add_argument('--start-date', type=str,
                       help='Start date for analytics (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date for analytics (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        # Create accumulator
        accumulator = EnhancedUnifiedAccumulator(args.symbols, args.interval)
        
        if args.summary:
            # Show database summary
            summary = accumulator.get_database_summary()
            print("\n📊 ENHANCED UNIFIED DATABASE SUMMARY")
            print("=" * 60)
            print(f"Raw Options Chain Records: {summary.get('raw_options_chain_count', 0):,}")
            print(f"Individual Options Records: {summary.get('options_data_count', 0):,}")
            print(f"Market Summary Records: {summary.get('market_summary_count', 0):,}")
            print(f"OHLC Candles Records: {summary.get('ohlc_candles_count', 0):,}")
            print(f"Alert Records: {summary.get('alerts_count', 0):,}")
            print(f"Quality Log Records: {summary.get('data_quality_log_count', 0):,}")
            print(f"Greeks Analysis Records: {summary.get('greeks_analysis_count', 0):,}")
            print(f"Volatility Surface Records: {summary.get('volatility_surface_count', 0):,}")
            print(f"Strategy Signals Records: {summary.get('strategy_signals_count', 0):,}")
            print(f"Performance Metrics Records: {summary.get('performance_metrics_count', 0):,}")
            print(f"Symbols: {summary.get('symbols', [])}")
            print(f"Average Quality Score: {summary.get('avg_quality_score', 0):.2f}")
            print(f"Market Open Records: {summary.get('market_open_records', 0):,}")
            print(f"Unacknowledged Alerts: {summary.get('unacknowledged_alerts', 0):,}")
            
            date_range = summary.get('date_range', {})
            if date_range.get('start') and date_range.get('end'):
                print(f"Date Range: {date_range['start']} to {date_range['end']}")
            
            return
        
        if args.analytics:
            # Show analytics data
            analytics = accumulator.get_analytics_data(args.analytics, args.start_date, args.end_date)
            print(f"\n📈 ANALYTICS DATA FOR {args.analytics}")
            print("=" * 50)
            
            market_data = analytics.get('market_data', [])
            if market_data:
                print(f"📊 Market Data Records: {len(market_data)}")
                print("Latest Market Data:")
                for i, record in enumerate(market_data[:5]):
                    print(f"  {i+1}. {record}")
            
            options_data = analytics.get('options_data', [])
            if options_data:
                print(f"📋 Options Data Records: {len(options_data)}")
                print("Latest Options Data:")
                for i, record in enumerate(options_data[:5]):
                    print(f"  {i+1}. {record}")
            
            alerts = analytics.get('alerts', [])
            if alerts:
                print(f"⚠️ Alert Records: {len(alerts)}")
                print("Latest Alerts:")
                for i, alert in enumerate(alerts[:5]):
                    print(f"  {i+1}. {alert}")
            
            return
        
        # Run accumulation
        accumulator.run()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 