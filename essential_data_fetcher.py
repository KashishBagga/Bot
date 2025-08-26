#!/usr/bin/env python3
"""
Essential Data Fetcher
Fetches essential timeframe data from 2020-01-01 to today for backtesting
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.api.fyers import FyersClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('essential_data_fetching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Essential symbols and timeframes for backtesting
ESSENTIAL_SYMBOLS = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX"
]

ESSENTIAL_TIMEFRAMES = [
    "5min",
    "15min", 
    "30min",
    "1D"
]

# Timeframe to resolution mapping for Fyers API
TIMEFRAME_TO_RESOLUTION = {
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1D": "1D"
}

class EssentialDataFetcher:
    def __init__(self):
        """Initialize the essential data fetcher"""
        self.fyers = FyersClient()
        self.base_dir = Path("historical_data_20yr")
        self.start_date = "2020-01-01"  # Start from 2020 for testing
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create base directory
        self.base_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Essential Data Fetcher Initialized")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"📁 Base Directory: {self.base_dir}")
        
    def create_data_directories(self):
        """Create directory structure for essential symbols and timeframes"""
        for symbol in ESSENTIAL_SYMBOLS:
            symbol_dir = self.base_dir / symbol.replace(":", "_")
            symbol_dir.mkdir(exist_ok=True)
            
            for timeframe in ESSENTIAL_TIMEFRAMES:
                timeframe_dir = symbol_dir / timeframe
                timeframe_dir.mkdir(exist_ok=True)
                
        logger.info("✅ Directory structure created for essential symbols and timeframes")
    
    def fetch_data_for_symbol_timeframe(self, symbol, timeframe):
        """Fetch data for a specific symbol and timeframe"""
        logger.info(f"🔄 Fetching {symbol} {timeframe}")
        
        # Convert timeframe to resolution
        resolution = TIMEFRAME_TO_RESOLUTION.get(timeframe)
        if resolution is None:
            logger.error(f"❌ Invalid timeframe: {timeframe}")
            return None
        
        try:
            # Fetch data for the entire period
            response = self.fyers.get_historical_data(
                symbol=symbol,
                resolution=resolution,
                range_from=self.start_date,
                range_to=self.end_date
            )
            
            if response and 'candles' in response:
                candles = response['candles']
                if candles:
                    # Convert to DataFrame
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    logger.info(f"✅ Fetched {len(df):,} candles for {symbol} {timeframe}")
                    return df
                else:
                    logger.warning(f"⚠️ No data for {symbol} {timeframe}")
                    return None
            else:
                logger.warning(f"⚠️ No data for {symbol} {timeframe}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol} {timeframe}: {e}")
            return None
    
    def save_data(self, symbol, timeframe, data):
        """Save data to parquet file"""
        if data is None or data.empty:
            logger.warning(f"⚠️ No data to save for {symbol} {timeframe}")
            return
        
        symbol_dir = self.base_dir / symbol.replace(":", "_") / timeframe
        parquet_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_complete.parquet"
        
        try:
            # Save the dataset
            data.to_parquet(parquet_file, index=False)
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': data['timestamp'].min().strftime('%Y-%m-%d'),
                'end_date': data['timestamp'].max().strftime('%Y-%m-%d'),
                'total_candles': len(data),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metadata_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_metadata.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Data saved: {parquet_file}")
            logger.info(f"📊 {metadata['total_candles']:,} candles from {metadata['start_date']} to {metadata['end_date']}")
            
        except Exception as e:
            logger.error(f"❌ Error saving data for {symbol} {timeframe}: {e}")
    
    def fetch_all_essential_data(self):
        """Fetch all essential data"""
        logger.info("🚀 Starting essential data fetch...")
        
        # Create directory structure
        self.create_data_directories()
        
        total_combinations = len(ESSENTIAL_SYMBOLS) * len(ESSENTIAL_TIMEFRAMES)
        current_combination = 0
        
        for symbol in ESSENTIAL_SYMBOLS:
            for timeframe in ESSENTIAL_TIMEFRAMES:
                current_combination += 1
                logger.info(f"📊 Progress: {current_combination}/{total_combinations}")
                
                try:
                    # Fetch data
                    data = self.fetch_data_for_symbol_timeframe(symbol, timeframe)
                    
                    if data is not None:
                        # Save data
                        self.save_data(symbol, timeframe, data)
                    else:
                        logger.warning(f"⚠️ No data available for {symbol} {timeframe}")
                    
                    # Rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {symbol} {timeframe}: {e}")
                    continue
        
        logger.info("🎉 Essential data fetch completed!")
    
    def print_data_summary(self):
        """Print summary of fetched data"""
        logger.info("📊 DATA SUMMARY:")
        logger.info("=" * 60)
        
        for symbol in ESSENTIAL_SYMBOLS:
            for timeframe in ESSENTIAL_TIMEFRAMES:
                symbol_dir = self.base_dir / symbol.replace(":", "_") / timeframe
                parquet_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_complete.parquet"
                
                if parquet_file.exists():
                    try:
                        df = pd.read_parquet(parquet_file)
                        if not df.empty:
                            start_date = df['timestamp'].min().strftime('%Y-%m-%d')
                            end_date = df['timestamp'].max().strftime('%Y-%m-%d')
                            count = len(df)
                            logger.info(f"{symbol:<20} {timeframe:<8} {start_date:<12} {end_date:<12} {count:>8,} candles")
                        else:
                            logger.info(f"{symbol:<20} {timeframe:<8} {'N/A':<12} {'N/A':<12} {'0':>8} candles")
                    except Exception as e:
                        logger.info(f"{symbol:<20} {timeframe:<8} {'ERROR':<12} {'ERROR':<12} {'0':>8} candles")
                else:
                    logger.info(f"{symbol:<20} {timeframe:<8} {'MISSING':<12} {'MISSING':<12} {'0':>8} candles")

def main():
    """Main function"""
    try:
        fetcher = EssentialDataFetcher()
        
        # Fetch essential data
        fetcher.fetch_all_essential_data()
        
        # Print summary
        fetcher.print_data_summary()
        
        logger.info("🎉 Essential data fetch completed successfully!")
        logger.info("💾 Data is now stored in parquet files for backtesting")
        
    except Exception as e:
        logger.error(f"❌ Fatal error in essential data fetch: {e}")
        raise

if __name__ == "__main__":
    main() 