#!/usr/bin/env python3
"""
Quick Data Fetch - Get essential data for NIFTY50 and NIFTYBANK
Fetch data for recent trading periods only
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.api.fyers import FyersClient

class QuickDataFetcher:
    """Quick data fetcher for essential data"""
    
    def __init__(self):
        self.fyers = FyersClient()
        if not self.fyers.initialize_client():
            raise Exception("Failed to initialize Fyers client")
        
        self.data_dir = "data/parquet"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('quick_data_fetch.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def fetch_recent_data(self):
        """Fetch recent data for NIFTY50 and NIFTYBANK"""
        symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
        timeframes = ['5min', '15min', '30min', '1D']
        
        # Recent date ranges (last 2 years)
        date_ranges = [
            ('2023-01-01', '2023-12-31'),
            ('2024-01-01', '2024-12-31'),
            ('2025-01-01', '2025-12-31')
        ]
        
        for symbol in symbols:
            for timeframe in timeframes:
                self.logger.info(f"Fetching {timeframe} data for {symbol}")
                
                all_data = []
                for start_date, end_date in date_ranges:
                    try:
                        data = self.fyers.get_historical_data(
                            symbol=symbol,
                            resolution=timeframe,
                            date_format=1,
                            range_from=start_date,
                            range_to=end_date,
                            cont_flag=1
                        )
                        
                        if data and 'candles' in data and data['candles']:
                            df = pd.DataFrame(data['candles'], columns=[
                                'datetime', 'open', 'high', 'low', 'close', 'volume'
                            ])
                            df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                            df.set_index('datetime', inplace=True)
                            
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            df = df.dropna()
                            if not df.empty:
                                all_data.append(df)
                                self.logger.info(f"Got {len(df)} records for {start_date} to {end_date}")
                        
                    except Exception as e:
                        self.logger.error(f"Error fetching {symbol} {timeframe} {start_date}-{end_date}: {e}")
                
                if all_data:
                    # Merge all data
                    merged_df = pd.concat(all_data, ignore_index=False)
                    merged_df = merged_df.sort_index()
                    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
                    
                    # Save to parquet
                    os.makedirs(os.path.join(self.data_dir, timeframe), exist_ok=True)
                    symbol_clean = symbol.replace(':', '_').replace('-', '_')
                    filename = f"{symbol_clean}_{timeframe}_complete.parquet"
                    filepath = os.path.join(self.data_dir, timeframe, filename)
                    
                    merged_df.to_parquet(filepath, index=True, compression='snappy')
                    self.logger.info(f"✅ Saved {len(merged_df)} records to {filepath}")
                else:
                    self.logger.warning(f"No data available for {symbol} {timeframe}")

def main():
    fetcher = QuickDataFetcher()
    fetcher.fetch_recent_data()
    print("✅ Quick data fetch completed!")

if __name__ == "__main__":
    main() 