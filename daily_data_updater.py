#!/usr/bin/env python3
"""
Daily Historical Data Updater
=============================

This script automatically runs every day when markets open to update
historical data from Fyers API. It ensures your local parquet files
are always up to date with the latest market data.

Features:
- Automatic market hours detection (9:15 AM - 3:30 PM IST, Mon-Fri)
- Smart data gap detection and filling
- Rate limiting to avoid API issues
- Comprehensive logging and error handling
- Can be scheduled with cron or systemd

Usage:
    python3 daily_data_updater.py --symbols NSE:NIFTY50-INDEX,NSE:NIFTYBANK-INDEX --timeframes 5min,15min,1D
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import pytz

# Add src to path
sys.path.append('src')

from live_paper_trading import LivePaperTradingSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'daily_data_updater.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DailyDataUpdater:
    """Automated daily historical data updater that runs during market hours."""
    
    def __init__(self, symbols: List[str], timeframes: List[str]):
        """Initialize the daily updater."""
        self.symbols = symbols
        self.timeframes = timeframes
        self.tz = pytz.timezone("Asia/Kolkata")
        self.base_path = Path("historical_data_20yr")
        
        # Ensure base directory exists
        self.base_path.mkdir(exist_ok=True)
        
        # Market hours (IST)
        self.market_start = {"hour": 9, "minute": 15}
        self.market_end = {"hour": 15, "minute": 30}
        
        logger.info("✅ Daily Data Updater initialized")
        logger.info(f"📊 Symbols: {', '.join(symbols)}")
        logger.info(f"⏰ Timeframes: {', '.join(timeframes)}")
    
    def is_market_open(self, current_time: Optional[datetime] = None) -> bool:
        """Check if market is open (market timezone = Asia/Kolkata)."""
        if current_time is None:
            current_time = datetime.now(self.tz)
        else:
            current_time = current_time.astimezone(self.tz) if current_time.tzinfo else current_time.replace(tzinfo=self.tz)
        
        if current_time.weekday() >= 5:  # Saturday/Sunday
            logger.debug(f"❌ Market closed: Weekend ({current_time.strftime('%A')})")
            return False
        
        market_start = current_time.replace(hour=self.market_start["hour"], minute=self.market_start["minute"], second=0, microsecond=0)
        market_end = current_time.replace(hour=self.market_end["hour"], minute=self.market_end["minute"], second=0, microsecond=0)
        
        is_open = market_start <= current_time <= market_end
        if not is_open:
            logger.debug(f"❌ Market closed: {current_time.strftime('%H:%M:%S')} (Market: {market_start.strftime('%H:%M')}-{market_end.strftime('%H:%M')})")
        else:
            logger.debug(f"✅ Market open: {current_time.strftime('%H:%M:%S')}")
        
        return is_open
    
    def wait_for_market_open(self) -> bool:
        """Wait for market to open, return True if market opened, False if timeout."""
        logger.info("⏰ Waiting for market to open...")
        
        timeout_minutes = 60  # Wait up to 1 hour
        start_time = datetime.now(self.tz)
        
        while not self.is_market_open():
            current_time = datetime.now(self.tz)
            elapsed = (current_time - start_time).total_seconds() / 60
            
            if elapsed > timeout_minutes:
                logger.warning(f"⚠️ Timeout waiting for market to open after {timeout_minutes} minutes")
                return False
            
            # Log progress every 5 minutes
            if int(elapsed) % 5 == 0:
                logger.info(f"⏳ Still waiting... Elapsed: {int(elapsed)} minutes")
            
            time.sleep(60)  # Check every minute
        
        logger.info("🎉 Market is now open!")
        return True
    
    def get_missing_dates(self, symbol: str, timeframe: str) -> List[str]:
        """Get list of missing dates for a symbol and timeframe."""
        try:
            # Map symbol to directory name
            symbol_mapping = {
                'NSE:NIFTY50-INDEX': 'NSE_NIFTY50-INDEX',
                'NSE:NIFTYBANK-INDEX': 'NSE_NIFTYBANK-INDEX',
                'NSE:FINNIFTY-INDEX': 'NSE_FINNIFTY-INDEX'
            }
            
            symbol_key = symbol_mapping.get(symbol)
            if not symbol_key:
                logger.warning(f"⚠️ No mapping for symbol: {symbol}")
                return []
            
            # Check if parquet file exists
            parquet_path = self.base_path / symbol_key / timeframe / f"{symbol_key}_{timeframe}_complete.parquet"
            
            if not parquet_path.exists():
                logger.info(f"📁 Creating new data file for {symbol} {timeframe}")
                return self._get_all_missing_dates()
            
            # Read existing data
            df = pd.read_parquet(parquet_path)
            latest_date = df['timestamp'].max().date()
            
            # Calculate missing dates
            today = datetime.now(self.tz).date()
            missing_dates = []
            
            current_date = latest_date + timedelta(days=1)
            while current_date <= today:
                # Skip weekends
                if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                    missing_dates.append(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=1)
            
            if missing_dates:
                logger.info(f"📅 Missing dates for {symbol} {timeframe}: {len(missing_dates)} dates")
                logger.info(f"📅 From {latest_date} to {today}")
            else:
                logger.info(f"✅ {symbol} {timeframe} data is up to date")
            
            return missing_dates
            
        except Exception as e:
            logger.error(f"❌ Error getting missing dates for {symbol} {timeframe}: {e}")
            return []
    
    def _get_all_missing_dates(self) -> List[str]:
        """Get all missing dates from a reasonable start date."""
        start_date = datetime(2024, 1, 1).date()
        today = datetime.now(self.tz).date()
        missing_dates = []
        
        current_date = start_date
        while current_date <= today:
            if current_date.weekday() < 5:  # Skip weekends
                missing_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        logger.info(f"📅 Creating complete dataset from {start_date} to {today}")
        return missing_dates
    
    def update_symbol_timeframe(self, trading_system: LivePaperTradingSystem, symbol: str, timeframe: str) -> bool:
        """Update historical data for a specific symbol and timeframe."""
        try:
            logger.info(f"🔄 Updating {symbol} {timeframe}")
            
            # Get missing dates
            missing_dates = self.get_missing_dates(symbol, timeframe)
            
            if not missing_dates:
                logger.info(f"✅ {symbol} {timeframe} is already up to date")
                return True
            
            # Fetch data for each missing date
            all_new_data = []
            
            for date in missing_dates:
                # For intraday timeframes, fetch the specific date
                if timeframe in ['1min', '3min', '5min', '15min', '30min', '60min', '240min']:
                    new_data = self._fetch_historical_data(trading_system, symbol, timeframe, date, date)
                    if new_data is not None:
                        all_new_data.append(new_data)
                        logger.info(f"✅ Fetched {len(new_data)} candles for {date}")
                    
                    # Add small delay to avoid rate limiting
                    time.sleep(0.5)
                
                # For daily timeframe, fetch in larger chunks
                elif timeframe == '1D':
                    # Fetch monthly chunks for daily data
                    start_date = datetime.strptime(date, '%Y-%m-%d')
                    end_date = min(start_date + timedelta(days=30), datetime.now())
                    
                    new_data = self._fetch_historical_data(
                        trading_system, symbol, timeframe, 
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if new_data is not None:
                        all_new_data.append(new_data)
                        logger.info(f"✅ Fetched {len(new_data)} daily candles")
                    
                    # Add delay for daily data
                    time.sleep(1)
            
            if not all_new_data:
                logger.warning(f"⚠️ No new data fetched for {symbol} {timeframe}")
                return False
            
            # Combine all new data
            combined_new_data = pd.concat(all_new_data, ignore_index=True)
            
            # Update parquet file
            success = self._update_parquet_file(symbol, timeframe, combined_new_data)
            
            if success:
                logger.info(f"✅ Successfully updated {symbol} {timeframe}")
            else:
                logger.error(f"❌ Failed to update {symbol} {timeframe}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error updating {symbol} {timeframe}: {e}")
            return False
    
    def _fetch_historical_data(self, trading_system: LivePaperTradingSystem, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from Fyers API for a specific date range."""
        try:
            logger.info(f"📡 Fetching {timeframe} data for {symbol} from {start_date} to {end_date}")
            
            # Map timeframe to resolution
            timeframe_mapping = {
                '1min': '1',
                '3min': '3',
                '5min': '5',
                '15min': '15',
                '30min': '30',
                '60min': '60',
                '240min': '240',
                '1D': '1D'
            }
            
            resolution = timeframe_mapping.get(timeframe, '5')
            
            # Fetch data from Fyers
            data = trading_system.data_manager.get_historical_data(
                symbol=symbol,
                resolution=resolution,
                date_format=1,
                range_from=start_date,
                range_to=end_date,
                cont_flag=1
            )
            
            if not data or 'candles' not in data or len(data['candles']) == 0:
                logger.warning(f"⚠️ No data received for {symbol} {timeframe} {start_date} to {end_date}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
            
            # Ensure all timestamps are timezone-aware
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Kolkata')
            
            logger.info(f"✅ Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol} {timeframe} {start_date} to {end_date}: {e}")
            return None
    
    def _update_parquet_file(self, symbol: str, timeframe: str, new_data: pd.DataFrame):
        """Update the parquet file with new data."""
        try:
            # Map symbol to directory name
            symbol_mapping = {
                'NSE:NIFTY50-INDEX': 'NSE_NIFTY50-INDEX',
                'NSE:NIFTYBANK-INDEX': 'NSE_NIFTYBANK-INDEX',
                'NSE:FINNIFTY-INDEX': 'NSE_FINNIFTY-INDEX'
            }
            
            symbol_key = symbol_mapping.get(symbol)
            if not symbol_key:
                logger.error(f"❌ No mapping for symbol: {symbol}")
                return False
            
            # Create directory structure
            data_dir = self.base_path / symbol_key / timeframe
            data_dir.mkdir(parents=True, exist_ok=True)
            
            parquet_path = data_dir / f"{symbol_key}_{timeframe}_complete.parquet"
            
            if parquet_path.exists():
                # Read existing data
                existing_df = pd.read_parquet(parquet_path)
                logger.info(f"📁 Existing data: {len(existing_df)} candles")
                
                # Ensure existing data timestamps are timezone-aware
                if existing_df['timestamp'].dt.tz is None:
                    existing_df['timestamp'] = existing_df['timestamp'].dt.tz_localize('Asia/Kolkata')
                
                # Ensure new data timestamps are timezone-aware
                if new_data['timestamp'].dt.tz is None:
                    new_data['timestamp'] = new_data['timestamp'].dt.tz_localize('Asia/Kolkata')
                
                # Combine with new data
                combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                
                # Remove duplicates based on timestamp
                combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                logger.info(f"📁 Combined data: {len(combined_df)} candles")
            else:
                # Ensure new data timestamps are timezone-aware
                if new_data['timestamp'].dt.tz is None:
                    new_data['timestamp'] = new_data['timestamp'].dt.tz_localize('Asia/Kolkata')
                
                combined_df = new_data
                logger.info(f"📁 New data file: {len(combined_df)} candles")
            
            # Save updated data
            combined_df.to_parquet(parquet_path, index=False)
            
            # Update metadata
            metadata = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": combined_df['timestamp'].min().strftime('%Y-%m-%d'),
                "end_date": combined_df['timestamp'].max().strftime('%Y-%m-%d'),
                "total_candles": len(combined_df),
                "last_updated": datetime.now(self.tz).strftime('%Y-%m-%d %H:%M:%S'),
                "data_type": "real"
            }
            
            metadata_path = data_dir / f"{symbol_key}_{timeframe}_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Updated {symbol} {timeframe}: {len(combined_df)} total candles")
            logger.info(f"📅 Date range: {metadata['start_date']} to {metadata['end_date']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating parquet file for {symbol} {timeframe}: {e}")
            return False
    
    def run_daily_update(self) -> Dict[str, bool]:
        """Run the daily update process."""
        try:
            logger.info("🚀 Starting daily historical data update")
            logger.info(f"⏰ Current time: {datetime.now(self.tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # Check if market is open
            if not self.is_market_open():
                logger.info("⏰ Market is not open, waiting for market hours...")
                if not self.wait_for_market_open():
                    logger.error("❌ Failed to wait for market to open")
                    return {}
            
            # Initialize trading system
            logger.info("🔐 Initializing trading system for data access...")
            trading_system = LivePaperTradingSystem(initial_capital=30000, data_provider='fyers')
            logger.info("✅ Trading system initialized successfully")
            
            # Update all specified data
            results = {}
            
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    key = f"{symbol}_{timeframe}"
                    logger.info(f"🔄 Processing {key}")
                    
                    success = self.update_symbol_timeframe(trading_system, symbol, timeframe)
                    results[key] = success
                    
                    # Add delay between requests
                    time.sleep(2)
            
            # Summary
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            
            logger.info(f"📊 Daily Update Summary: {successful}/{total} successful")
            
            for key, success in results.items():
                status = "✅" if success else "❌"
                logger.info(f"{status} {key}")
            
            if all(results.values()):
                logger.info("🎉 All daily updates completed successfully!")
            else:
                logger.warning("⚠️ Some updates failed. Check the logs for details.")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Fatal error in daily update: {e}")
            return {}

def main():
    """Main function to run the daily data updater."""
    parser = argparse.ArgumentParser(description='Daily historical data updater')
    parser.add_argument('--symbols', nargs='+', 
                       default=['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX'],
                       help='Symbols to update')
    parser.add_argument('--timeframes', nargs='+',
                       default=['5min', '15min', '1D'],
                       help='Timeframes to update')
    parser.add_argument('--wait-for-market', action='store_true',
                       help='Wait for market to open before updating')
    
    args = parser.parse_args()
    
    try:
        updater = DailyDataUpdater(args.symbols, args.timeframes)
        
        # Run daily update
        results = updater.run_daily_update()
        
        # Exit with appropriate code
        if all(results.values()):
            logger.info("🎉 Daily update completed successfully")
            sys.exit(0)
        else:
            logger.warning("⚠️ Daily update completed with some failures")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 