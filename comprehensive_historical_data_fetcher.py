#!/usr/bin/env python3
"""
Comprehensive Historical Data Fetcher
Fetches all timeframe data from 2000-01-01 to today and stores permanently in parquet files
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
from src.config.settings import SYMBOLS, TIMEFRAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_historical_fetching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Timeframe to resolution mapping for Fyers API
TIMEFRAME_TO_RESOLUTION = {
    "1min": 1,
    "2min": 2,
    "3min": 3,
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "20min": 20,
    "30min": 30,
    "45min": 45,
    "60min": 60,
    "120min": 120,
    "180min": 180,
    "240min": 240,
    "1D": "1D"
}

class ComprehensiveHistoricalDataFetcher:
    def __init__(self):
        """Initialize the comprehensive data fetcher"""
        self.fyers = FyersClient()
        self.base_dir = Path("historical_data_20yr")
        self.start_date = "2000-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create base directory
        self.base_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Comprehensive Historical Data Fetcher Initialized")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"📁 Base Directory: {self.base_dir}")
        
    def create_data_directories(self):
        """Create directory structure for all symbols and timeframes"""
        for symbol in SYMBOLS:
            symbol_dir = self.base_dir / symbol.replace(":", "_")
            symbol_dir.mkdir(exist_ok=True)
            
            for timeframe in TIMEFRAMES:
                timeframe_dir = symbol_dir / timeframe
                timeframe_dir.mkdir(exist_ok=True)
                
        logger.info("✅ Directory structure created for all symbols and timeframes")
    
    def get_existing_data_info(self, symbol, timeframe):
        """Check what data already exists for a symbol/timeframe"""
        symbol_dir = self.base_dir / symbol.replace(":", "_") / timeframe
        parquet_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_complete.parquet"
        
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)
                if not df.empty:
                    start_date = df['timestamp'].min().strftime('%Y-%m-%d')
                    end_date = df['timestamp'].max().strftime('%Y-%m-%d')
                    count = len(df)
                    logger.info(f"📊 Existing data for {symbol} {timeframe}: {start_date} to {end_date} ({count:,} candles)")
                    return start_date, end_date, count
            except Exception as e:
                logger.warning(f"⚠️ Error reading existing data for {symbol} {timeframe}: {e}")
        
        return None, None, 0
    
    def fetch_data_in_chunks(self, symbol, timeframe, start_date, end_date, chunk_days=365):
        """Fetch data in yearly chunks to avoid API limits"""
        logger.info(f"🔄 Fetching {symbol} {timeframe} from {start_date} to {end_date}")
        
        # Convert timeframe to resolution
        resolution = TIMEFRAME_TO_RESOLUTION.get(timeframe)
        if resolution is None:
            logger.error(f"❌ Invalid timeframe: {timeframe}")
            return pd.DataFrame()
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_days), end_dt)
            
            chunk_start = current_start.strftime("%Y-%m-%d")
            chunk_end = current_end.strftime("%Y-%m-%d")
            
            logger.info(f"📅 Fetching chunk: {chunk_start} to {chunk_end}")
            
            try:
                # Fetch data for this chunk using correct parameters
                response = self.fyers.get_historical_data(
                    symbol=symbol,
                    resolution=resolution,
                    range_from=chunk_start,
                    range_to=chunk_end
                )
                
                if response and 'candles' in response:
                    # Convert response to DataFrame
                    candles = response['candles']
                    if candles:
                        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        all_data.append(df)
                        logger.info(f"✅ Chunk fetched: {len(df):,} candles")
                    else:
                        logger.warning(f"⚠️ No data for chunk {chunk_start} to {chunk_end}")
                else:
                    logger.warning(f"⚠️ No data for chunk {chunk_start} to {chunk_end}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error fetching chunk {chunk_start} to {chunk_end}: {e}")
                time.sleep(5)  # Longer delay on error
            
            current_start = current_end
        
        if all_data:
            # Combine all chunks
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=['timestamp'])
            combined_data = combined_data.sort_values('timestamp')
            
            logger.info(f"🎉 Total data fetched: {len(combined_data):,} candles")
            return combined_data
        else:
            logger.warning(f"⚠️ No data fetched for {symbol} {timeframe}")
            return pd.DataFrame()
    
    def save_data_permanently(self, symbol, timeframe, data):
        """Save data permanently in parquet format"""
        if data.empty:
            logger.warning(f"⚠️ No data to save for {symbol} {timeframe}")
            return
        
        symbol_dir = self.base_dir / symbol.replace(":", "_") / timeframe
        parquet_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_complete.parquet"
        
        try:
            # Save the complete dataset
            data.to_parquet(parquet_file, index=False)
            
            # Also save metadata
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
            logger.info(f"📊 Metadata: {metadata['total_candles']:,} candles from {metadata['start_date']} to {metadata['end_date']}")
            
        except Exception as e:
            logger.error(f"❌ Error saving data for {symbol} {timeframe}: {e}")
    
    def fetch_all_historical_data(self):
        """Fetch all historical data for all symbols and timeframes"""
        logger.info("🚀 Starting comprehensive historical data fetch...")
        
        # Create directory structure
        self.create_data_directories()
        
        total_combinations = len(SYMBOLS) * len(TIMEFRAMES)
        current_combination = 0
        
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                current_combination += 1
                logger.info(f"📊 Progress: {current_combination}/{total_combinations}")
                
                try:
                    # Check existing data
                    existing_start, existing_end, existing_count = self.get_existing_data_info(symbol, timeframe)
                    
                    # Determine what needs to be fetched
                    if existing_start and existing_end:
                        # Check if we need to update
                        if existing_start <= self.start_date and existing_end >= self.end_date:
                            logger.info(f"✅ {symbol} {timeframe}: Data already complete ({existing_count:,} candles)")
                            continue
                        else:
                            logger.info(f"🔄 {symbol} {timeframe}: Updating existing data")
                    
                    # Fetch complete data
                    data = self.fetch_data_in_chunks(symbol, timeframe, self.start_date, self.end_date)
                    
                    if not data.empty:
                        # Save permanently
                        self.save_data_permanently(symbol, timeframe, data)
                    else:
                        logger.warning(f"⚠️ No data available for {symbol} {timeframe}")
                    
                    # Rate limiting between symbols
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {symbol} {timeframe}: {e}")
                    continue
        
        logger.info("🎉 Comprehensive historical data fetch completed!")
    
    def verify_data_completeness(self):
        """Verify that all data has been fetched and saved correctly"""
        logger.info("🔍 Verifying data completeness...")
        
        verification_report = []
        
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                symbol_dir = self.base_dir / symbol.replace(":", "_") / timeframe
                parquet_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_complete.parquet"
                
                if parquet_file.exists():
                    try:
                        df = pd.read_parquet(parquet_file)
                        if not df.empty:
                            start_date = df['timestamp'].min().strftime('%Y-%m-%d')
                            end_date = df['timestamp'].max().strftime('%Y-%m-%d')
                            count = len(df)
                            
                            verification_report.append({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'start_date': start_date,
                                'end_date': end_date,
                                'candles': count,
                                'status': '✅ COMPLETE' if start_date <= self.start_date and end_date >= self.end_date else '⚠️ PARTIAL'
                            })
                        else:
                            verification_report.append({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'start_date': 'N/A',
                                'end_date': 'N/A',
                                'candles': 0,
                                'status': '❌ EMPTY'
                            })
                    except Exception as e:
                        verification_report.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'start_date': 'ERROR',
                            'end_date': 'ERROR',
                            'candles': 0,
                            'status': f'❌ ERROR: {e}'
                        })
                else:
                    verification_report.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'start_date': 'N/A',
                        'end_date': 'N/A',
                        'candles': 0,
                        'status': '❌ MISSING'
                    })
        
        # Print verification report
        logger.info("📊 DATA VERIFICATION REPORT:")
        logger.info("=" * 80)
        
        for item in verification_report:
            logger.info(f"{item['symbol']:<20} {item['timeframe']:<8} {item['start_date']:<12} {item['end_date']:<12} {item['candles']:>8,} {item['status']}")
        
        # Summary
        complete = sum(1 for item in verification_report if 'COMPLETE' in item['status'])
        total = len(verification_report)
        
        logger.info("=" * 80)
        logger.info(f"📈 SUMMARY: {complete}/{total} datasets complete ({complete/total*100:.1f}%)")
        
        return verification_report

def main():
    """Main function to run the comprehensive data fetcher"""
    try:
        fetcher = ComprehensiveHistoricalDataFetcher()
        
        # Fetch all historical data
        fetcher.fetch_all_historical_data()
        
        # Verify completeness
        fetcher.verify_data_completeness()
        
        logger.info("🎉 Comprehensive historical data fetch completed successfully!")
        logger.info("💾 All data is now permanently stored in parquet files")
        logger.info("🚀 You can now run backtests without fetching data from API")
        
    except Exception as e:
        logger.error(f"❌ Fatal error in comprehensive data fetch: {e}")
        raise

if __name__ == "__main__":
    main() 