#!/usr/bin/env python3
"""
Fetch 240min Data Only
Fetches only the missing 240min data for NIFTYBANK
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SingleDataFetcher:
    def __init__(self):
        """Initialize the single data fetcher"""
        self.fyers = FyersClient()
        self.base_dir = Path("historical_data_20yr")
        self.start_date = "2000-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"🎯 Single Data Fetcher - 240min NIFTYBANK")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
    
    def setup_fyers_client(self):
        """Setup and authenticate Fyers client"""
        logger.info("🔐 Setting up Fyers client...")
        
        try:
            if not self.fyers.client_id or not self.fyers.secret_key:
                logger.error("❌ Missing Fyers credentials")
                return False
            
            logger.info("🌐 Opening authentication URL...")
            self.fyers.open_auth_url()
            
            auth_code = input("🔑 Please enter the authorization code from the URL: ").strip()
            
            if not auth_code:
                logger.error("❌ No authorization code provided")
                return False
            
            self.fyers.set_auth_code(auth_code)
            
            if not self.fyers.generate_access_token():
                logger.error("❌ Failed to generate access token")
                return False
            
            if not self.fyers.initialize_client():
                logger.error("❌ Failed to initialize Fyers client")
                return False
            
            logger.info("✅ Fyers client setup completed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up Fyers client: {e}")
            return False
    
    def fetch_240min_data(self):
        """Fetch 240min data for NIFTYBANK"""
        symbol = "NSE:NIFTYBANK-INDEX"
        timeframe = "240min"
        
        logger.info(f"🔄 Fetching {symbol} {timeframe}")
        
        try:
            # Fetch data in chunks
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            
            all_data = []
            current_start = start_dt
            chunk_days = 30
            
            while current_start < end_dt:
                current_end = min(current_start + timedelta(days=chunk_days), end_dt)
                
                chunk_start = current_start.strftime("%Y-%m-%d")
                chunk_end = current_end.strftime("%Y-%m-%d")
                
                logger.info(f"📅 Fetching chunk: {chunk_start} to {chunk_end}")
                
                try:
                    response = self.fyers.get_historical_data(
                        symbol=symbol,
                        resolution=240,  # 240min
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
                
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def save_data(self, data):
        """Save data to parquet file"""
        if data.empty:
            logger.warning(f"⚠️ No data to save")
            return
        
        symbol = "NSE:NIFTYBANK-INDEX"
        timeframe = "240min"
        
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
            logger.error(f"❌ Error saving data: {e}")
    
    def verify_completion(self):
        """Verify that all data is now complete"""
        logger.info("🔍 Verifying final completion...")
        
        total_files = 0
        complete_files = 0
        
        for symbol in ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]:
            for timeframe in ["1min", "3min", "5min", "15min", "30min", "60min", "240min", "1D"]:
                total_files += 1
                symbol_dir = self.base_dir / symbol.replace(":", "_") / timeframe
                parquet_file = symbol_dir / f"{symbol.replace(':', '_')}_{timeframe}_complete.parquet"
                
                if parquet_file.exists():
                    try:
                        df = pd.read_parquet(parquet_file)
                        if not df.empty:
                            complete_files += 1
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
        
        logger.info("=" * 80)
        logger.info(f"📈 FINAL COMPLETION: {complete_files}/{total_files} files ({complete_files/total_files*100:.1f}%)")
        
        if complete_files == total_files:
            logger.info("🎉 ALL DATA COMPLETE!")
        else:
            logger.info(f"⚠️ Still missing {total_files - complete_files} files")

def main():
    """Main function"""
    try:
        fetcher = SingleDataFetcher()
        
        # Setup Fyers client
        if not fetcher.setup_fyers_client():
            logger.error("❌ Failed to setup Fyers client. Exiting.")
            return
        
        # Fetch 240min data
        data = fetcher.fetch_240min_data()
        
        if not data.empty:
            # Save data
            fetcher.save_data(data)
        
        # Verify completion
        fetcher.verify_completion()
        
        logger.info("🎉 240min data fetch completed!")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    main() 