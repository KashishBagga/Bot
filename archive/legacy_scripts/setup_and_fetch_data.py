#!/usr/bin/env python3
"""
Setup and Fetch Historical Data
Properly initializes Fyers client and fetches historical data from 2000-01-01 to today
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
        logging.FileHandler('setup_and_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Essential symbols and timeframes
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

# Timeframe to resolution mapping
TIMEFRAME_TO_RESOLUTION = {
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1D": "1D"
}

class DataFetcher:
    def __init__(self):
        """Initialize the data fetcher"""
        self.fyers = FyersClient()
        self.base_dir = Path("historical_data_20yr")
        self.start_date = "2000-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create base directory
        self.base_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Data Fetcher Initialized")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"📁 Base Directory: {self.base_dir}")
    
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
        """Create directory structure"""
        for symbol in ESSENTIAL_SYMBOLS:
            symbol_dir = self.base_dir / symbol.replace(":", "_")
            symbol_dir.mkdir(exist_ok=True)
            
            for timeframe in ESSENTIAL_TIMEFRAMES:
                timeframe_dir = symbol_dir / timeframe
                timeframe_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Directory structure created")
    
    def fetch_data_for_symbol_timeframe(self, symbol, timeframe):
        """Fetch data for a specific symbol and timeframe"""
        logger.info(f"🔄 Fetching {symbol} {timeframe}")
        
        resolution = TIMEFRAME_TO_RESOLUTION.get(timeframe)
        if resolution is None:
            logger.error(f"❌ Invalid timeframe: {timeframe}")
            return None
        
        try:
            # Fetch data in smaller chunks to avoid API limits
            chunk_days = 30  # Fetch 30 days at a time
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            
            all_data = []
            current_start = start_dt
            
            while current_start < end_dt:
                current_end = min(current_start + timedelta(days=chunk_days), end_dt)
                
                chunk_start = current_start.strftime("%Y-%m-%d")
                chunk_end = current_end.strftime("%Y-%m-%d")
                
                logger.info(f"📅 Fetching chunk: {chunk_start} to {chunk_end}")
                
                try:
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
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"❌ Error fetching chunk {chunk_start} to {chunk_end}: {e}")
                    time.sleep(5)
                
                current_start = current_end
            
            if all_data:
                # Combine all chunks
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=['timestamp'])
                combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"🎉 Total data fetched: {len(combined_data):,} candles")
                return combined_data
            else:
                logger.warning(f"⚠️ No data fetched for {symbol} {timeframe}")
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
    
    def fetch_all_data(self):
        """Fetch all data"""
        logger.info("🚀 Starting data fetch...")
        
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
                    
                    # Rate limiting between symbols
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {symbol} {timeframe}: {e}")
                    continue
        
        logger.info("🎉 Data fetch completed!")
    
    def print_data_summary(self):
        """Print summary of fetched data"""
        logger.info("📊 DATA SUMMARY:")
        logger.info("=" * 80)
        
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
        fetcher = DataFetcher()
        
        # Setup Fyers client
        if not fetcher.setup_fyers_client():
            logger.error("❌ Failed to setup Fyers client. Exiting.")
            return
        
        # Fetch all data
        fetcher.fetch_all_data()
        
        # Print summary
        fetcher.print_data_summary()
        
        logger.info("🎉 Data fetch completed successfully!")
        logger.info("💾 All data is now stored in parquet files for backtesting")
        logger.info("🚀 You can now run backtests without fetching data from API")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    main() 