#!/usr/bin/env python3
"""
Complete Missing Data
Fetches missing data and completes any gaps in historical data
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
        logging.FileHandler('complete_missing_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Missing data to fetch
MISSING_DATA = [
    ("NSE:NIFTYBANK-INDEX", "60min"),
    ("NSE:NIFTYBANK-INDEX", "240min"),
]

# Timeframe to resolution mapping
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

class MissingDataFetcher:
    def __init__(self):
        """Initialize the missing data fetcher"""
        self.fyers = FyersClient()
        self.base_dir = Path("historical_data_20yr")
        self.start_date = "2000-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"🚀 Missing Data Fetcher Initialized")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
    
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
    
    def fetch_data_in_chunks(self, symbol, timeframe, start_date, end_date, chunk_days=30):
        """Fetch data in monthly chunks"""
        logger.info(f"🔄 Fetching {symbol} {timeframe} from {start_date} to {end_date}")
        
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
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error fetching chunk {chunk_start} to {chunk_end}: {e}")
                time.sleep(10)
            
            current_start = current_end
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=['timestamp'])
            combined_data = combined_data.sort_values('timestamp')
            
            logger.info(f"🎉 Total data fetched: {len(combined_data):,} candles")
            return combined_data
        else:
            logger.warning(f"⚠️ No data fetched for {symbol} {timeframe}")
            return pd.DataFrame()
    
    def save_data(self, symbol, timeframe, data):
        """Save data to parquet file"""
        if data.empty:
            logger.warning(f"⚠️ No data to save for {symbol} {timeframe}")
            return
        
        symbol_dir = self.base_dir / symbol.replace(":", "_") / timeframe
        parquet_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_complete.parquet"
        
        try:
            data.to_parquet(parquet_file, index=False)
            
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
            logger.info(f"📊 {metadata['total_candles']:,} candles from {metadata['start_date']} to {metadata['end_date']}")
            
        except Exception as e:
            logger.error(f"❌ Error saving data for {symbol} {timeframe}: {e}")
    
    def fetch_missing_data(self):
        """Fetch all missing data"""
        logger.info("🚀 Starting to fetch missing data...")
        
        for symbol, timeframe in MISSING_DATA:
            logger.info(f"📊 Fetching missing data for {symbol} {timeframe}")
            
            try:
                data = self.fetch_data_in_chunks(symbol, timeframe, self.start_date, self.end_date)
                
                if not data.empty:
                    self.save_data(symbol, timeframe, data)
                else:
                    logger.warning(f"⚠️ No data available for {symbol} {timeframe}")
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol} {timeframe}: {e}")
                continue
        
        logger.info("🎉 Missing data fetch completed!")
    
    def check_data_completeness(self):
        """Check completeness of all data"""
        logger.info("🔍 Checking data completeness...")
        
        all_symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
        all_timeframes = ["1min", "3min", "5min", "15min", "30min", "60min", "240min", "1D"]
        
        for symbol in all_symbols:
            for timeframe in all_timeframes:
                symbol_dir = self.base_dir / symbol.replace(":", "_") / timeframe
                parquet_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_complete.parquet"
                
                if parquet_file.exists():
                    try:
                        df = pd.read_parquet(parquet_file)
                        if not df.empty:
                            start_date = df['timestamp'].min().strftime('%Y-%m-%d')
                            end_date = df['timestamp'].max().strftime('%Y-%m-%d')
                            count = len(df)
                            logger.info(f"✅ {symbol:<20} {timeframe:<8} {start_date:<12} {end_date:<12} {count:>8,} candles")
                        else:
                            logger.warning(f"❌ {symbol:<20} {timeframe:<8} {'EMPTY':<12} {'EMPTY':<12} {'0':>8} candles")
                    except Exception as e:
                        logger.error(f"❌ {symbol:<20} {timeframe:<8} {'ERROR':<12} {'ERROR':<12} {'0':>8} candles")
                else:
                    logger.warning(f"❌ {symbol:<20} {timeframe:<8} {'MISSING':<12} {'MISSING':<12} {'0':>8} candles")

def main():
    """Main function"""
    try:
        fetcher = MissingDataFetcher()
        
        # Setup Fyers client
        if not fetcher.setup_fyers_client():
            logger.error("❌ Failed to setup Fyers client. Exiting.")
            return
        
        # Fetch missing data
        fetcher.fetch_missing_data()
        
        # Check completeness
        fetcher.check_data_completeness()
        
        logger.info("🎉 Missing data fetch completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    main() 