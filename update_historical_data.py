#!/usr/bin/env python3
"""
Historical Data Update Script
============================

This script fetches missing historical data from Fyers API and updates
the local parquet files to ensure no gaps in historical data.

Usage:
    python3 update_historical_data.py --symbols NSE:NIFTY50-INDEX,NSE:NIFTYBANK-INDEX --timeframes 5min,15min,1D
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')

from live_paper_trading import LivePaperTradingSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('historical_data_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HistoricalDataUpdater:
    """Updates historical data by fetching missing data from Fyers API."""
    
    def __init__(self):
        """Initialize the updater with the existing trading system."""
        try:
            # Use the existing trading system's data manager
            self.trading_system = LivePaperTradingSystem(initial_capital=30000, data_provider='fyers')
            self.data_manager = self.trading_system.data_manager
            self.base_path = Path("historical_data_20yr")
            
            # Ensure base directory exists
            self.base_path.mkdir(exist_ok=True)
            
            logger.info("✅ Historical Data Updater initialized with existing trading system")
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading system: {e}")
            raise
    
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
            today = datetime.now().date()
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
        today = datetime.now().date()
        missing_dates = []
        
        current_date = start_date
        while current_date <= today:
            if current_date.weekday() < 5:  # Skip weekends
                missing_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        logger.info(f"📅 Creating complete dataset from {start_date} to {today}")
        return missing_dates
    
    def fetch_historical_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
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
            data = self.data_manager.get_historical_data(
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
    
    def update_parquet_file(self, symbol: str, timeframe: str, new_data: pd.DataFrame):
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
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
    
    def update_symbol_timeframe(self, symbol: str, timeframe: str) -> bool:
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
                    new_data = self.fetch_historical_data(symbol, timeframe, date, date)
                    if new_data is not None:
                        all_new_data.append(new_data)
                        logger.info(f"✅ Fetched {len(new_data)} candles for {date}")
                    
                    # Add small delay to avoid rate limiting
                    import time
                    time.sleep(0.5)
                
                # For daily timeframe, fetch in larger chunks
                elif timeframe == '1D':
                    # Fetch monthly chunks for daily data
                    start_date = datetime.strptime(date, '%Y-%m-%d')
                    end_date = min(start_date + timedelta(days=30), datetime.now())
                    
                    new_data = self.fetch_historical_data(
                        symbol, timeframe, 
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if new_data is not None:
                        all_new_data.append(new_data)
                        logger.info(f"✅ Fetched {len(new_data)} daily candles")
                    
                    # Add delay for daily data
                    import time
                    time.sleep(1)
            
            if not all_new_data:
                logger.warning(f"⚠️ No new data fetched for {symbol} {timeframe}")
                return False
            
            # Combine all new data
            combined_new_data = pd.concat(all_new_data, ignore_index=True)
            
            # Update parquet file
            success = self.update_parquet_file(symbol, timeframe, combined_new_data)
            
            if success:
                logger.info(f"✅ Successfully updated {symbol} {timeframe}")
            else:
                logger.error(f"❌ Failed to update {symbol} {timeframe}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error updating {symbol} {timeframe}: {e}")
            return False
    
    def update_all_data(self, symbols: List[str], timeframes: List[str]) -> Dict[str, bool]:
        """Update all specified symbols and timeframes."""
        results = {}
        
        logger.info(f"🚀 Starting historical data update for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                logger.info(f"🔄 Processing {key}")
                
                success = self.update_symbol_timeframe(symbol, timeframe)
                results[key] = success
                
                # Add delay between requests
                import time
                time.sleep(2)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"📊 Update Summary: {successful}/{total} successful")
        
        for key, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"{status} {key}")
        
        return results

def main():
    """Main function to run the historical data updater."""
    parser = argparse.ArgumentParser(description='Update historical data from Fyers API')
    parser.add_argument('--symbols', nargs='+', 
                       default=['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX'],
                       help='Symbols to update')
    parser.add_argument('--timeframes', nargs='+',
                       default=['5min', '15min', '1D'],
                       help='Timeframes to update')
    parser.add_argument('--complete', action='store_true',
                       help='Complete missing data from Aug 25, 2025 to today')
    
    args = parser.parse_args()
    
    try:
        updater = HistoricalDataUpdater()
        
        if args.complete:
            logger.info("🔄 Running complete data update from Aug 25, 2025 to today")
        
        # Update all specified data
        results = updater.update_all_data(args.symbols, args.timeframes)
        
        # Check if all updates were successful
        if all(results.values()):
            logger.info("🎉 All historical data updates completed successfully!")
        else:
            logger.warning("⚠️ Some updates failed. Check the logs for details.")
        
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 