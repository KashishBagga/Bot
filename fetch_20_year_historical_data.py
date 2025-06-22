#!/usr/bin/env python3
"""
20-Year Historical Data Fetcher for Fyers
Comprehensive data fetching across all timeframes for backtesting
"""

import os
import sys
import time
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
import pickle
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import backoff

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DataRequest:
    """Data request configuration"""
    symbol: str
    symbol_name: str
    timeframe: str
    resolution: str
    start_date: str
    end_date: str
    chunk_size_days: int

class HistoricalDataFetcher:
    """20-Year Historical Data Fetcher with advanced features"""
    
    def __init__(self, output_dir: str = "historical_data_20yr"):
        """Initialize the historical data fetcher"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Fyers API setup
        self.setup_fyers()
        
        # Timeframe configurations
        self.timeframe_configs = {
            '1min': {'resolution': '1', 'chunk_days': 30},    # 30 days chunks
            '3min': {'resolution': '3', 'chunk_days': 60},    # 60 days chunks
            '5min': {'resolution': '5', 'chunk_days': 90},    # 90 days chunks
            '15min': {'resolution': '15', 'chunk_days': 180}, # 180 days chunks
            '30min': {'resolution': '30', 'chunk_days': 365}, # 1 year chunks
            '1hour': {'resolution': '60', 'chunk_days': 730}, # 2 year chunks
            '4hour': {'resolution': '240', 'chunk_days': 1460}, # 4 year chunks
            '1day': {'resolution': 'D', 'chunk_days': 2920},   # 8 year chunks
            '1week': {'resolution': 'W', 'chunk_days': 7300}   # 20 year chunks
        }
        
        # Symbols to fetch
        self.symbols = {
            'NSE:NIFTY50-INDEX': 'NIFTY50',
            'NSE:NIFTYBANK-INDEX': 'BANKNIFTY',
            'NSE:NIFTYFIN-INDEX': 'NIFTYFIN',
            'NSE:NIFTYIT-INDEX': 'NIFTYIT',
            'NSE:NIFTYPHARMA-INDEX': 'NIFTYPHARMA',
            'NSE:NIFTYMETAL-INDEX': 'NIFTYMETAL',
            'NSE:NIFTYAUTO-INDEX': 'NIFTYAUTO',
            'NSE:NIFTYREALTY-INDEX': 'NIFTYREALTY'
        }
        
        # Progress tracking
        self.progress = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'total_candles_fetched': 0,
            'start_time': None,
            'errors': []
        }
        
        # Rate limiting
        self.request_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'fetch_20yr_data.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('HistoricalDataFetcher')
        
    def setup_fyers(self):
        """Setup Fyers API client"""
        try:
            client_id = os.getenv("FYERS_CLIENT_ID")
            access_token = os.getenv("FYERS_ACCESS_TOKEN")
            
            if not client_id or not access_token:
                raise ValueError("Missing Fyers API credentials in .env file")
            
            self.fyers = fyersModel.FyersModel(
                token=access_token, 
                is_async=False, 
                client_id=client_id, 
                log_path=""
            )
            
            # Test connection
            profile = self.fyers.get_profile()
            if profile.get('code') == 200:
                self.logger.info(f"✅ Fyers API connected successfully")
                self.logger.info(f"User: {profile.get('data', {}).get('name', 'Unknown')}")
            else:
                raise Exception(f"Fyers API test failed: {profile}")
                
        except Exception as e:
            self.logger.error(f"❌ Fyers API setup failed: {e}")
            raise
    
    def rate_limit(self):
        """Implement rate limiting to avoid API throttling"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3, base=2)
    def fetch_data_chunk(self, request: DataRequest) -> Optional[pd.DataFrame]:
        """Fetch a single chunk of historical data with retry logic"""
        self.rate_limit()
        
        try:
            data = {
                "symbol": request.symbol,
                "resolution": request.resolution,
                "date_format": "1",  # Unix timestamp
                "range_from": request.start_date,
                "range_to": request.end_date,
                "cont_flag": "1"
            }
            
            self.logger.debug(f"Fetching {request.symbol_name} {request.timeframe} "
                            f"from {request.start_date} to {request.end_date}")
            
            response = self.fyers.history(data)
            
            # Check response
            if isinstance(response, dict):
                if response.get('code') == 200:
                    candles = response.get('candles', [])
                    if candles:
                        df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('time', inplace=True)
                        
                        self.logger.debug(f"✅ Fetched {len(df)} candles for "
                                        f"{request.symbol_name} {request.timeframe}")
                        return df
                    else:
                        self.logger.warning(f"No candles in response for {request.symbol_name} "
                                          f"{request.timeframe}")
                        return pd.DataFrame()
                else:
                    error_msg = response.get('message', 'Unknown error')
                    self.logger.error(f"API Error {response.get('code')}: {error_msg}")
                    return None
            else:
                self.logger.error(f"Unexpected response format: {type(response)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception fetching data: {e}")
            raise
    
    def generate_date_chunks(self, start_date: str, end_date: str, chunk_days: int) -> List[Tuple[str, str]]:
        """Generate date chunks for data fetching"""
        chunks = []
        current_start = datetime.strptime(start_date, '%Y-%m-%d')
        final_end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_start < final_end:
            current_end = min(current_start + timedelta(days=chunk_days), final_end)
            chunks.append((
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d')
            ))
            current_start = current_end + timedelta(days=1)
        
        return chunks
    
    def fetch_symbol_timeframe(self, symbol: str, symbol_name: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch complete historical data for a symbol and timeframe"""
        config = self.timeframe_configs[timeframe]
        
        # Calculate date range (20 years back)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=20*365)).strftime('%Y-%m-%d')
        
        self.logger.info(f"📊 Fetching {symbol_name} {timeframe} data from {start_date} to {end_date}")
        
        # Generate date chunks
        date_chunks = self.generate_date_chunks(start_date, end_date, config['chunk_days'])
        
        all_data = []
        successful_chunks = 0
        
        for chunk_start, chunk_end in date_chunks:
            request = DataRequest(
                symbol=symbol,
                symbol_name=symbol_name,
                timeframe=timeframe,
                resolution=config['resolution'],
                start_date=chunk_start,
                end_date=chunk_end,
                chunk_size_days=config['chunk_days']
            )
            
            try:
                chunk_data = self.fetch_data_chunk(request)
                self.progress['completed_requests'] += 1
                
                if chunk_data is not None and not chunk_data.empty:
                    all_data.append(chunk_data)
                    successful_chunks += 1
                    self.progress['total_candles_fetched'] += len(chunk_data)
                    
                    self.logger.debug(f"✅ Chunk {successful_chunks}/{len(date_chunks)} completed")
                else:
                    self.logger.warning(f"⚠️ Empty data for chunk {chunk_start} to {chunk_end}")
                    
            except Exception as e:
                self.progress['failed_requests'] += 1
                error_msg = f"Failed to fetch {symbol_name} {timeframe} chunk {chunk_start}-{chunk_end}: {e}"
                self.progress['errors'].append(error_msg)
                self.logger.error(error_msg)
                continue
        
        # Combine all chunks
        if all_data:
            combined_df = pd.concat(all_data, axis=0)
            combined_df = combined_df.sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]  # Remove duplicates
            
            self.logger.info(f"✅ {symbol_name} {timeframe}: {len(combined_df)} candles "
                           f"({successful_chunks}/{len(date_chunks)} chunks)")
            return combined_df
        else:
            self.logger.error(f"❌ No data fetched for {symbol_name} {timeframe}")
            return None
    
    def save_data(self, df: pd.DataFrame, symbol_name: str, timeframe: str):
        """Save data in multiple formats"""
        symbol_dir = self.output_dir / symbol_name
        symbol_dir.mkdir(exist_ok=True)
        
        base_filename = f"{symbol_name}_{timeframe}_20yr"
        
        # Save as compressed parquet (best for analysis)
        parquet_file = symbol_dir / f"{base_filename}.parquet"
        df.to_parquet(parquet_file, compression='gzip')
        
        # Save as CSV (human readable)
        csv_file = symbol_dir / f"{base_filename}.csv"
        df.to_csv(csv_file)
        
        # Save as compressed pickle (fastest loading)
        pickle_file = symbol_dir / f"{base_filename}.pkl.gz"
        with gzip.open(pickle_file, 'wb') as f:
            pickle.dump(df, f)
        
        # Save metadata
        metadata = {
            'symbol': symbol_name,
            'timeframe': timeframe,
            'total_candles': len(df),
            'date_range': {
                'start': df.index.min().isoformat() if not df.empty else None,
                'end': df.index.max().isoformat() if not df.empty else None
            },
            'file_info': {
                'parquet_size_mb': parquet_file.stat().st_size / 1024 / 1024,
                'csv_size_mb': csv_file.stat().st_size / 1024 / 1024,
                'pickle_size_mb': pickle_file.stat().st_size / 1024 / 1024
            },
            'created_at': datetime.now().isoformat()
        }
        
        metadata_file = symbol_dir / f"{base_filename}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"💾 Saved {symbol_name} {timeframe} data: "
                        f"{len(df)} candles, {metadata['file_info']['parquet_size_mb']:.1f}MB")
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add essential technical indicators"""
        if df.empty:
            return df
        
        try:
            # EMA indicators
            df['ema_9'] = df['close'].ewm(span=9).mean()
            df['ema_21'] = df['close'].ewm(span=21).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_200'] = df['close'].ewm(span=200).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['close'].rolling(bb_period).mean()
            bb_std_val = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
            
            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def print_progress(self):
        """Print current progress"""
        if self.progress['start_time']:
            elapsed = datetime.now() - self.progress['start_time']
            completion_rate = self.progress['completed_requests'] / max(self.progress['total_requests'], 1)
            estimated_total = elapsed / max(completion_rate, 0.001)
            remaining = estimated_total - elapsed
            
            print(f"\n📊 PROGRESS UPDATE")
            print(f"{'='*50}")
            print(f"Completed: {self.progress['completed_requests']}/{self.progress['total_requests']} requests")
            print(f"Failed: {self.progress['failed_requests']} requests")
            print(f"Candles fetched: {self.progress['total_candles_fetched']:,}")
            print(f"Elapsed time: {elapsed}")
            print(f"Estimated remaining: {remaining}")
            print(f"Completion: {completion_rate*100:.1f}%")
            print(f"{'='*50}")
    
    def fetch_all_data(self, symbols: Dict[str, str] = None, timeframes: List[str] = None):
        """Fetch all historical data"""
        symbols = symbols or self.symbols
        timeframes = timeframes or list(self.timeframe_configs.keys())
        
        self.progress['total_requests'] = len(symbols) * len(timeframes)
        self.progress['start_time'] = datetime.now()
        
        self.logger.info(f"🚀 Starting 20-year historical data fetch")
        self.logger.info(f"📊 Symbols: {list(symbols.values())}")
        self.logger.info(f"⏱️ Timeframes: {timeframes}")
        self.logger.info(f"📋 Total requests: {self.progress['total_requests']}")
        
        results = {}
        
        for symbol, symbol_name in symbols.items():
            results[symbol_name] = {}
            
            for timeframe in timeframes:
                try:
                    # Fetch data
                    df = self.fetch_symbol_timeframe(symbol, symbol_name, timeframe)
                    
                    if df is not None and not df.empty:
                        # Add technical indicators
                        df = self.add_technical_indicators(df)
                        
                        # Save data
                        self.save_data(df, symbol_name, timeframe)
                        
                        results[symbol_name][timeframe] = {
                            'success': True,
                            'candles': len(df),
                            'date_range': [df.index.min(), df.index.max()]
                        }
                    else:
                        results[symbol_name][timeframe] = {
                            'success': False,
                            'error': 'No data fetched'
                        }
                        
                except Exception as e:
                    error_msg = f"Failed to fetch {symbol_name} {timeframe}: {e}"
                    self.logger.error(error_msg)
                    results[symbol_name][timeframe] = {
                        'success': False,
                        'error': str(e)
                    }
                
                # Print progress every 5 requests
                if self.progress['completed_requests'] % 5 == 0:
                    self.print_progress()
        
        return results
    
    def generate_summary_report(self, results: dict):
        """Generate comprehensive summary report"""
        report_file = self.output_dir / 'fetch_summary_report.json'
        
        summary = {
            'fetch_completed_at': datetime.now().isoformat(),
            'total_duration': str(datetime.now() - self.progress['start_time']),
            'statistics': {
                'total_requests': self.progress['total_requests'],
                'successful_requests': self.progress['completed_requests'] - self.progress['failed_requests'],
                'failed_requests': self.progress['failed_requests'],
                'success_rate': (self.progress['completed_requests'] - self.progress['failed_requests']) / self.progress['total_requests'] * 100,
                'total_candles_fetched': self.progress['total_candles_fetched']
            },
            'results': results,
            'errors': self.progress['errors']
        }
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print(f"\n🎉 20-YEAR DATA FETCH COMPLETED!")
        print(f"{'='*60}")
        print(f"Total Duration: {summary['total_duration']}")
        print(f"Success Rate: {summary['statistics']['success_rate']:.1f}%")
        print(f"Total Candles: {summary['statistics']['total_candles_fetched']:,}")
        print(f"Data Location: {self.output_dir}")
        print(f"Summary Report: {report_file}")
        print(f"{'='*60}")
        
        return summary


def main():
    """Main execution function"""
    print("📈 20-YEAR HISTORICAL DATA FETCHER")
    print("="*60)
    print("🎯 Fetching 20 years of data across all timeframes")
    print("📊 Timeframes: 1min, 3min, 5min, 15min, 30min, 1hour, 4hour, 1day, 1week")
    print("💾 Output formats: Parquet (compressed), CSV, Pickle")
    print("🔧 Features: Technical indicators, metadata, progress tracking")
    print("="*60)
    
    # Confirm the operation
    estimated_time_hours = 8  # Conservative estimate
    print(f"⚠️  This will take approximately {estimated_time_hours} hours to complete.")
    print(f"   The script will fetch data in chunks with rate limiting.")
    print(f"   You can interrupt and resume later if needed.")
    
    confirmation = input("\nDo you want to proceed with the 20-year data fetch? (y/N): ").strip().lower()
    if confirmation not in ['y', 'yes']:
        print("❌ Data fetch cancelled")
        return 1
    
    try:
        # Initialize fetcher
        fetcher = HistoricalDataFetcher()
        
        # Fetch all data
        results = fetcher.fetch_all_data()
        
        # Generate summary report
        fetcher.generate_summary_report(results)
        
        print("\n✅ Historical data fetch completed successfully!")
        print(f"📂 Data saved in: {fetcher.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during data fetch: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 