#!/usr/bin/env python3
"""
Fetch All Real Historical Data
Fetches real historical data from 2000-01-01 to today for all timeframes
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
        logging.FileHandler('fetch_all_real_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# All symbols and timeframes
ALL_SYMBOLS = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX"
]

ALL_TIMEFRAMES = [
    "1min",
    "3min", 
    "5min",
    "15min",
    "30min",
    "60min",
    "240min",
    "1D"
]

# Timeframe to resolution mapping for Fyers API
TIMEFRAME_TO_RESOLUTION = {
    "1min": 1,
    "3min": 3,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "60min": 60,
    "240min": 240,
    "1D": "1D"
}

class RealDataFetcher:
    def __init__(self):
        """Initialize the real data fetcher"""
        self.fyers = FyersClient()
        self.base_dir = Path("historical_data_20yr")
        self.start_date = "2000-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create base directory
        self.base_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Real Data Fetcher Initialized")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"📁 Base Directory: {self.base_dir}")
        logger.info(f"📊 Symbols: {ALL_SYMBOLS}")
        logger.info(f"⏰ Timeframes: {ALL_TIMEFRAMES}")
    
    def setup_fyers_client(self):
        """Setup and authenticate Fyers client"""
        logger.info("🔐 Setting up Fyers client...")
        
        try:
            # Check if we have the required environment variables
            if not self.fyers.client_id or not self.fyers.secret_key:
                logger.error("❌ Missing Fyers credentials. Please set FYERS_CLIENT_ID and FYERS_SECRET_KEY in .env file")
                return False
            
            # Generate auth URL and open it
            logger.info("🌐 Opening authentication URL...")
            self.fyers.open_auth_url()
            
            # Get auth code from user
            auth_code = input("🔑 Please enter the authorization code from the URL: ").strip()
            
            if not auth_code:
                logger.error("❌ No authorization code provided")
                return False
            
            # Set auth code and generate access token
            self.fyers.set_auth_code(auth_code)
            
            if not self.fyers.generate_access_token():
                logger.error("❌ Failed to generate access token")
                return False
            
            # Initialize client
            if not self.fyers.initialize_client():
                logger.error("❌ Failed to initialize Fyers client")
                return False
            
            logger.info("✅ Fyers client setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up Fyers client: {e}")
            return False
    
    def create_data_directories(self):
        """Create directory structure for all symbols and timeframes"""
        for symbol in ALL_SYMBOLS:
            symbol_dir = self.base_dir / symbol.replace(":", "_")
            symbol_dir.mkdir(exist_ok=True)
            
            for timeframe in ALL_TIMEFRAMES:
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
    
    def fetch_data_in_chunks(self, symbol, timeframe, start_date, end_date, chunk_days=30):
        """Fetch data in monthly chunks to avoid API limits"""
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
                # Fetch data for this chunk
                response = self.fyers.get_historical_data(
                    symbol=symbol,
                    resolution=resolution,
                    range_from=chunk_start,
                    range_to=chunk_end
                )
                
                if response and 'candles' in response:
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
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error fetching chunk {chunk_start} to {chunk_end}: {e}")
                time.sleep(10)  # Longer delay on error
            
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
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_type': 'real'
            }
            
            metadata_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_metadata.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Data saved: {parquet_file}")
            logger.info(f"📊 Metadata: {metadata['total_candles']:,} candles from {metadata['start_date']} to {metadata['end_date']}")
            
        except Exception as e:
            logger.error(f"❌ Error saving data for {symbol} {timeframe}: {e}")
    
    def fetch_all_real_data(self):
        """Fetch all real historical data for all symbols and timeframes"""
        logger.info("🚀 Starting comprehensive real data fetch...")
        
        # Create directory structure
        self.create_data_directories()
        
        total_combinations = len(ALL_SYMBOLS) * len(ALL_TIMEFRAMES)
        current_combination = 0
        
        for symbol in ALL_SYMBOLS:
            for timeframe in ALL_TIMEFRAMES:
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
                    time.sleep(5)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {symbol} {timeframe}: {e}")
                    continue
        
        logger.info("🎉 Comprehensive real data fetch completed!")
    
    def verify_data_completeness(self):
        """Verify that all data has been fetched and saved correctly"""
        logger.info("🔍 Verifying data completeness...")
        
        verification_report = []
        
        for symbol in ALL_SYMBOLS:
            for timeframe in ALL_TIMEFRAMES:
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
        logger.info("📊 REAL DATA VERIFICATION REPORT:")
        logger.info("=" * 100)
        
        for item in verification_report:
            logger.info(f"{item['symbol']:<20} {item['timeframe']:<8} {item['start_date']:<12} {item['end_date']:<12} {item['candles']:>8,} {item['status']}")
        
        # Summary
        complete = sum(1 for item in verification_report if 'COMPLETE' in item['status'])
        total = len(verification_report)
        
        logger.info("=" * 100)
        logger.info(f"📈 SUMMARY: {complete}/{total} datasets complete ({complete/total*100:.1f}%)")
        
        return verification_report

def main():
    """Main function to run the real data fetcher"""
    try:
        fetcher = RealDataFetcher()
        
        # Setup Fyers client
        if not fetcher.setup_fyers_client():
            logger.error("❌ Failed to setup Fyers client. Exiting.")
            return
        
        # Fetch all real data
        fetcher.fetch_all_real_data()
        
        # Verify completeness
        fetcher.verify_data_completeness()
        
        logger.info("🎉 Real data fetch completed successfully!")
        logger.info("💾 All real data is now permanently stored in parquet files")
        logger.info("🚀 You can now run backtests without fetching data from API")
        
    except Exception as e:
        logger.error(f"❌ Fatal error in real data fetch: {e}")
        raise

if __name__ == "__main__":
    main() 