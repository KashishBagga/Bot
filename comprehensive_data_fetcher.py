#!/usr/bin/env python3
"""
Comprehensive Historical Data Fetcher
Fetch data from 1990 to 2025 for all timeframes and store in parquet format
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import argparse
from typing import List, Dict, Optional
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.fyers import FyersClient
from src.config.settings import FYERS_CLIENT_ID, FYERS_SECRET_KEY

class ComprehensiveDataFetcher:
    """Fetch comprehensive historical data from 1990 to 2025"""
    
    def __init__(self):
        self.fyers = FyersClient()
        # Initialize the Fyers client
        if not self.fyers.initialize_client():
            raise Exception("Failed to initialize Fyers client")
        
        self.data_dir = "data/parquet"
        self.timeframes = {
            '1min': 1,
            '2min': 2,
            '3min': 3,
            '5min': 5,
            '10min': 10,
            '15min': 15,
            '20min': 20,
            '30min': 30,
            '45min': 45,
            '60min': 60,
            '120min': 120,
            '180min': 180,
            '240min': 240,
            '1D': 1440
        }
        
        # Setup logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_fetching.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create data directory structure
        self._create_data_directories()
    
    def _create_data_directories(self):
        """Create directory structure for data storage"""
        for timeframe in self.timeframes.keys():
            timeframe_dir = os.path.join(self.data_dir, timeframe)
            os.makedirs(timeframe_dir, exist_ok=True)
            self.logger.info(f"Created directory: {timeframe_dir}")
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols to fetch data for"""
        return [
            'NSE:NIFTY50-INDEX',
            'NSE:NIFTYBANK-INDEX',
            'NSE:NIFTYIT-INDEX',
            'NSE:NIFTYPHARMA-INDEX',
            'NSE:NIFTYAUTO-INDEX',
            'NSE:NIFTYFMCG-INDEX',
            'NSE:NIFTYMETAL-INDEX',
            'NSE:NIFTYREALTY-INDEX',
            'NSE:NIFTYMEDIA-INDEX',
            'NSE:NIFTYPVTBANK-INDEX',
            'NSE:NIFTYPSUBANK-INDEX',
            'NSE:NIFTYCONSUMERDURABLES-INDEX',
            'NSE:NIFTYENERGY-INDEX',
            'NSE:NIFTYINFRA-INDEX',
            'NSE:NIFTYFINANCIALSERVICES-INDEX'
        ]
    
    def calculate_date_ranges(self, start_date: str = "1990-01-01", end_date: str = "2025-12-31") -> List[Dict]:
        """Calculate date ranges for fetching data in chunks"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Fetch data in 6-month chunks to avoid API limits
        chunk_size = timedelta(days=180)
        date_ranges = []
        
        current_start = start
        while current_start < end:
            current_end = min(current_start + chunk_size, end)
            date_ranges.append({
                'start': current_start.strftime("%Y-%m-%d"),
                'end': current_end.strftime("%Y-%m-%d")
            })
            current_start = current_end + timedelta(days=1)
        
        return date_ranges
    
    def fetch_historical_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for a specific symbol and timeframe"""
        try:
            self.logger.info(f"Fetching {timeframe} data for {symbol} from {start_date} to {end_date}")
            
            # Get timeframe value
            tf_value = self.timeframes[timeframe]
            
            # Fetch data from Fyers
            data = self.fyers.get_historical_data(
                symbol=symbol,
                resolution=timeframe,
                date_format=1,
                range_from=start_date,
                range_to=end_date,
                cont_flag=1
            )
            
            if not data or 'candles' not in data:
                self.logger.warning(f"No data received for {symbol} {timeframe} {start_date} to {end_date}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['candles'], columns=[
                'datetime', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any invalid data
            df = df.dropna()
            
            if df.empty:
                self.logger.warning(f"Empty DataFrame for {symbol} {timeframe} {start_date} to {end_date}")
                return None
            
            self.logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")
            return None
    
    def save_to_parquet(self, df: pd.DataFrame, symbol: str, timeframe: str, start_date: str, end_date: str):
        """Save DataFrame to parquet file"""
        try:
            # Create filename
            symbol_clean = symbol.replace(':', '_').replace('-', '_')
            filename = f"{symbol_clean}_{timeframe}_{start_date}_to_{end_date}.parquet"
            filepath = os.path.join(self.data_dir, timeframe, filename)
            
            # Save to parquet
            df.to_parquet(filepath, index=True, compression='snappy')
            self.logger.info(f"Saved {len(df)} records to {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving parquet file for {symbol} {timeframe}: {e}")
            return None
    
    def merge_parquet_files(self, symbol: str, timeframe: str):
        """Merge multiple parquet files for a symbol and timeframe"""
        try:
            timeframe_dir = os.path.join(self.data_dir, timeframe)
            symbol_clean = symbol.replace(':', '_').replace('-', '_')
            
            # Find all parquet files for this symbol and timeframe
            pattern = f"{symbol_clean}_{timeframe}_*.parquet"
            files = []
            
            for file in os.listdir(timeframe_dir):
                if file.startswith(f"{symbol_clean}_{timeframe}_") and file.endswith('.parquet'):
                    files.append(os.path.join(timeframe_dir, file))
            
            if not files:
                self.logger.warning(f"No parquet files found for {symbol} {timeframe}")
                return None
            
            # Sort files by date
            files.sort()
            
            # Read and merge all files
            dfs = []
            for file in files:
                try:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                    self.logger.info(f"Loaded {len(df)} records from {file}")
                except Exception as e:
                    self.logger.error(f"Error reading {file}: {e}")
            
            if not dfs:
                self.logger.warning(f"No valid DataFrames to merge for {symbol} {timeframe}")
                return None
            
            # Merge all DataFrames
            merged_df = pd.concat(dfs, ignore_index=False)
            merged_df = merged_df.sort_index()
            
            # Remove duplicates
            merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
            
            # Save merged file
            merged_filename = f"{symbol_clean}_{timeframe}_complete.parquet"
            merged_filepath = os.path.join(timeframe_dir, merged_filename)
            merged_df.to_parquet(merged_filepath, index=True, compression='snappy')
            
            self.logger.info(f"Merged {len(merged_df)} records to {merged_filepath}")
            
            # Clean up individual files
            for file in files:
                try:
                    os.remove(file)
                    self.logger.info(f"Removed {file}")
                except Exception as e:
                    self.logger.error(f"Error removing {file}: {e}")
            
            return merged_filepath
            
        except Exception as e:
            self.logger.error(f"Error merging parquet files for {symbol} {timeframe}: {e}")
            return None
    
    def fetch_comprehensive_data(self, symbols: Optional[List[str]] = None, 
                               timeframes: Optional[List[str]] = None,
                               start_date: str = "1990-01-01", 
                               end_date: str = "2025-12-31"):
        """Fetch comprehensive historical data for all symbols and timeframes"""
        
        if symbols is None:
            symbols = self.get_symbols()
        
        if timeframes is None:
            timeframes = list(self.timeframes.keys())
        
        # Calculate date ranges
        date_ranges = self.calculate_date_ranges(start_date, end_date)
        
        self.logger.info(f"Starting comprehensive data fetch:")
        self.logger.info(f"Symbols: {len(symbols)}")
        self.logger.info(f"Timeframes: {len(timeframes)}")
        self.logger.info(f"Date ranges: {len(date_ranges)}")
        
        total_requests = len(symbols) * len(timeframes) * len(date_ranges)
        current_request = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                self.logger.info(f"Processing {symbol} - {timeframe}")
                
                for date_range in date_ranges:
                    current_request += 1
                    progress = (current_request / total_requests) * 100
                    
                    self.logger.info(f"Progress: {progress:.1f}% ({current_request}/{total_requests})")
                    
                    # Fetch data
                    df = self.fetch_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=date_range['start'],
                        end_date=date_range['end']
                    )
                    
                    if df is not None and not df.empty:
                        # Save to parquet
                        self.save_to_parquet(
                            df=df,
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=date_range['start'],
                            end_date=date_range['end']
                        )
                    
                    # Rate limiting - wait between requests
                    time.sleep(0.5)
                
                # Merge files for this symbol and timeframe
                self.merge_parquet_files(symbol, timeframe)
        
        self.logger.info("Comprehensive data fetch completed!")
    
    def verify_data_completeness(self):
        """Verify that all data has been fetched and merged correctly"""
        self.logger.info("Verifying data completeness...")
        
        symbols = self.get_symbols()
        timeframes = list(self.timeframes.keys())
        
        missing_data = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                symbol_clean = symbol.replace(':', '_').replace('-', '_')
                filename = f"{symbol_clean}_{timeframe}_complete.parquet"
                filepath = os.path.join(self.data_dir, timeframe, filename)
                
                if not os.path.exists(filepath):
                    missing_data.append(f"{symbol} - {timeframe}")
                    continue
                
                try:
                    df = pd.read_parquet(filepath)
                    self.logger.info(f"✅ {symbol} {timeframe}: {len(df)} records")
                except Exception as e:
                    missing_data.append(f"{symbol} - {timeframe} (Error: {e})")
        
        if missing_data:
            self.logger.warning("Missing or corrupted data:")
            for item in missing_data:
                self.logger.warning(f"  • {item}")
        else:
            self.logger.info("✅ All data verified successfully!")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Historical Data Fetcher")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to fetch")
    parser.add_argument("--timeframes", nargs="+", help="Specific timeframes to fetch")
    parser.add_argument("--start-date", default="1990-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    
    args = parser.parse_args()
    
    fetcher = ComprehensiveDataFetcher()
    
    if args.verify_only:
        fetcher.verify_data_completeness()
    else:
        fetcher.fetch_comprehensive_data(
            symbols=args.symbols,
            timeframes=args.timeframes,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Verify after fetching
        fetcher.verify_data_completeness()

if __name__ == "__main__":
    main() 