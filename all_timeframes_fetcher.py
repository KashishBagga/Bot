#!/usr/bin/env python3
"""
All Timeframes Data Fetcher
Fetch all timeframes for NIFTY50 and NIFTYBANK from recent periods
"""
import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.api.fyers import FyersClient

class AllTimeframesFetcher:
    """Fetch all timeframes for NIFTY50 and NIFTYBANK"""
    
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
                logging.FileHandler('all_timeframes_fetch.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # All timeframes to fetch
        self.timeframes = {
            '1min': '1',
            '3min': '3', 
            '5min': '5',
            '15min': '15',
            '30min': '30',
            '60min': '60',
            '240min': '240',
            '1D': '1D'
        }
        
        # Symbols to fetch
        self.symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
        
        # Recent date ranges to try (last 6 months)
        self.date_ranges = [
            ('2025-03-01', '2025-08-25'),  # Last 6 months
            ('2024-09-01', '2025-02-28'),  # Previous 6 months
            ('2024-03-01', '2024-08-31'),  # Earlier 6 months
        ]
    
    def fetch_timeframe_data(self, symbol: str, timeframe: str, tf_code: str):
        """Fetch data for a specific timeframe"""
        self.logger.info(f"Fetching {timeframe} data for {symbol}")
        
        all_data = []
        
        for start_date, end_date in self.date_ranges:
            try:
                self.logger.info(f"  Trying {start_date} to {end_date}")
                
                data = self.fyers.get_historical_data(
                    symbol=symbol,
                    resolution=tf_code,
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
                        self.logger.info(f"    Got {len(df)} records for {start_date} to {end_date}")
                    else:
                        self.logger.warning(f"    No valid data for {start_date} to {end_date}")
                else:
                    self.logger.warning(f"    No data received for {start_date} to {end_date}")
                    
            except Exception as e:
                self.logger.error(f"    Error fetching {symbol} {timeframe} {start_date}-{end_date}: {e}")
        
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
            return True
        else:
            self.logger.warning(f"❌ No data available for {symbol} {timeframe}")
            return False
    
    def fetch_all_timeframes(self):
        """Fetch all timeframes for all symbols"""
        self.logger.info("🚀 Starting all timeframes data fetch")
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info(f"Timeframes: {list(self.timeframes.keys())}")
        
        results = {}
        
        for symbol in self.symbols:
            results[symbol] = {}
            for timeframe, tf_code in self.timeframes.items():
                success = self.fetch_timeframe_data(symbol, timeframe, tf_code)
                results[symbol][timeframe] = success
        
        # Summary
        self.logger.info("\n📊 FETCH SUMMARY")
        self.logger.info("=" * 50)
        
        for symbol in self.symbols:
            self.logger.info(f"\n{symbol}:")
            for timeframe, success in results[symbol].items():
                status = "✅" if success else "❌"
                self.logger.info(f"  {status} {timeframe}")
        
        # Check what we have
        self.logger.info("\n📁 AVAILABLE DATA FILES:")
        self.logger.info("=" * 50)
        
        for timeframe in self.timeframes.keys():
            tf_dir = os.path.join(self.data_dir, timeframe)
            if os.path.exists(tf_dir):
                files = [f for f in os.listdir(tf_dir) if f.endswith('_complete.parquet')]
                if files:
                    self.logger.info(f"\n{timeframe}:")
                    for file in files:
                        filepath = os.path.join(tf_dir, file)
                        try:
                            df = pd.read_parquet(filepath)
                            self.logger.info(f"  ✅ {file} ({len(df)} records)")
                        except Exception as e:
                            self.logger.error(f"  ❌ {file} (error: {e})")
                else:
                    self.logger.info(f"\n{timeframe}: No complete files found")

def main():
    fetcher = AllTimeframesFetcher()
    fetcher.fetch_all_timeframes()
    print("\n🎉 All timeframes fetch completed!")

if __name__ == "__main__":
    main() 