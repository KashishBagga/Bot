#!/usr/bin/env python3
"""
20-Year Parquet Data Setup
Fetch 20 years of historical data and store in optimized parquet format for backtesting
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging
from pathlib import Path
import backoff
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ParquetDataSetup:
    """Setup 20-year historical data in parquet format"""
    
    def __init__(self, data_dir: str = "data/parquet"):
        """Initialize the parquet data setup"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Fyers API setup
        self.setup_fyers()
        
        # Timeframe configurations (optimized for 20-year data)
        self.timeframe_configs = {
            '1min': {'resolution': '1', 'chunk_days': 15, 'priority': 1},      # High frequency, small chunks
            '3min': {'resolution': '3', 'chunk_days': 30, 'priority': 2},      
            '5min': {'resolution': '5', 'chunk_days': 45, 'priority': 3},      
            '15min': {'resolution': '15', 'chunk_days': 90, 'priority': 4},    
            '30min': {'resolution': '30', 'chunk_days': 180, 'priority': 5},   
            '1hour': {'resolution': '60', 'chunk_days': 365, 'priority': 6},   
            '4hour': {'resolution': '240', 'chunk_days': 730, 'priority': 7},  
            '1day': {'resolution': 'D', 'chunk_days': 1825, 'priority': 8}     # 5 year chunks for daily
        }
        
        # Symbols to fetch (comprehensive list)
        self.symbols = {
            # Major Indices
            'NSE:NIFTY50-INDEX': 'NIFTY50',
            'NSE:NIFTYBANK-INDEX': 'BANKNIFTY',
            'NSE:NIFTYFIN-INDEX': 'NIFTYFIN',
            'NSE:NIFTYIT-INDEX': 'NIFTYIT',
            'NSE:NIFTYPHARMA-INDEX': 'NIFTYPHARMA',
            'NSE:NIFTYMETAL-INDEX': 'NIFTYMETAL',
            'NSE:NIFTYAUTO-INDEX': 'NIFTYAUTO',
            'NSE:NIFTYREALTY-INDEX': 'NIFTYREALTY',
            'NSE:NIFTYFMCG-INDEX': 'NIFTYFMCG',
            'NSE:NIFTYENERGY-INDEX': 'NIFTYENERGY'
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
        
        # Rate limiting (conservative for large data fetch)
        self.request_delay = 0.5  # 500ms between requests
        self.last_request_time = 0
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'parquet_setup.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ParquetDataSetup')
        
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
                user_name = profile.get('data', {}).get('name', 'Unknown')
                self.logger.info(f"👤 User: {user_name}")
            else:
                raise Exception(f"Fyers API test failed: {profile}")
                
        except Exception as e:
            self.logger.error(f"❌ Fyers API setup failed: {e}")
            raise
    
    def rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3, base=2)
    def fetch_data_chunk(self, symbol: str, symbol_name: str, timeframe: str, 
                        start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch a single chunk of historical data"""
        self.rate_limit()
        
        try:
            config = self.timeframe_configs[timeframe]
            
            data = {
                "symbol": symbol,
                "resolution": config['resolution'],
                "date_format": "1",  # Unix timestamp
                "range_from": start_date,
                "range_to": end_date,
                "cont_flag": "1"
            }
            
            response = self.fyers.history(data)
            
            if isinstance(response, dict) and response.get('code') == 200:
                candles = response.get('candles', [])
                if candles:
                    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    # Add technical indicators for faster backtesting
                    df = self.add_basic_indicators(df)
                    
                    self.progress['total_candles_fetched'] += len(df)
                    return df
                else:
                    return pd.DataFrame()
            else:
                error_msg = response.get('message', 'Unknown error') if isinstance(response, dict) else str(response)
                self.logger.warning(f"API Error for {symbol_name} {timeframe}: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception fetching {symbol_name} {timeframe}: {e}")
            raise
    
    def add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to speed up backtesting"""
        if len(df) < 50:  # Need minimum data for indicators
            return df
        
        try:
            # EMA calculations
            df['ema_9'] = df['close'].ewm(span=9).mean()
            df['ema_21'] = df['close'].ewm(span=21).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            
            # RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD calculation
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            df['bb_middle'] = sma_20
            
            # SuperTrend (simplified)
            hl2 = (df['high'] + df['low']) / 2
            atr = df[['high', 'low', 'close']].apply(
                lambda x: max(x['high'] - x['low'], 
                             abs(x['high'] - x['close']), 
                             abs(x['low'] - x['close'])), axis=1
            ).rolling(window=10).mean()
            
            multiplier = 3.0
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            df['supertrend_upper'] = upper_band
            df['supertrend_lower'] = lower_band
            
        except Exception as e:
            self.logger.warning(f"Error adding indicators: {e}")
        
        return df
    
    def generate_date_chunks(self, start_date: str, end_date: str, chunk_days: int) -> List[tuple]:
        """Generate date chunks for data fetching"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_days), end_dt)
            chunks.append((
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d')
            ))
            current_start = current_end + timedelta(days=1)
        
        return chunks
    
    def fetch_symbol_timeframe(self, symbol: str, symbol_name: str, timeframe: str) -> bool:
        """Fetch complete data for a symbol-timeframe combination"""
        try:
            # Calculate 20-year date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=20*365)  # 20 years
            
            # Create output directory
            symbol_dir = self.data_dir / symbol_name.replace(':', '_')
            symbol_dir.mkdir(exist_ok=True)
            
            output_file = symbol_dir / f"{timeframe}.parquet"
            
            # Check if file already exists and is recent
            if output_file.exists():
                existing_df = pd.read_parquet(output_file)
                if len(existing_df) > 1000:  # Has substantial data
                    self.logger.info(f"⏭️  Skipping {symbol_name} {timeframe} (already exists with {len(existing_df)} records)")
                    return True
            
            self.logger.info(f"📥 Fetching {symbol_name} {timeframe} data (20 years)...")
            
            config = self.timeframe_configs[timeframe]
            chunks = self.generate_date_chunks(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                config['chunk_days']
            )
            
            all_data = []
            successful_chunks = 0
            
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                try:
                    chunk_df = self.fetch_data_chunk(
                        symbol, symbol_name, timeframe, chunk_start, chunk_end
                    )
                    
                    if chunk_df is not None and not chunk_df.empty:
                        all_data.append(chunk_df)
                        successful_chunks += 1
                        
                        # Progress update
                        progress = (i + 1) / len(chunks) * 100
                        self.logger.info(f"  📊 {symbol_name} {timeframe}: {progress:.1f}% "
                                       f"({successful_chunks}/{len(chunks)} chunks, {len(chunk_df)} candles)")
                    else:
                        self.logger.warning(f"  ⚠️  No data for chunk {chunk_start} to {chunk_end}")
                    
                    self.progress['completed_requests'] += 1
                    
                except Exception as e:
                    self.logger.error(f"  ❌ Chunk failed {chunk_start}-{chunk_end}: {e}")
                    self.progress['failed_requests'] += 1
                    continue
            
            if all_data:
                # Combine all chunks
                combined_df = pd.concat(all_data, ignore_index=False)
                combined_df = combined_df.sort_index()
                
                # Remove duplicates
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                
                # Save to parquet with compression
                combined_df.to_parquet(
                    output_file,
                    compression='snappy',
                    engine='pyarrow'
                )
                
                file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                self.logger.info(f"✅ Saved {symbol_name} {timeframe}: {len(combined_df)} records, "
                               f"{file_size:.1f}MB")
                return True
            else:
                self.logger.error(f"❌ No data collected for {symbol_name} {timeframe}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch {symbol_name} {timeframe}: {e}")
            return False
    
    def fetch_all_data(self, symbols: Dict[str, str] = None, timeframes: List[str] = None):
        """Fetch all historical data"""
        symbols = symbols or self.symbols
        timeframes = timeframes or list(self.timeframe_configs.keys())
        
        # Sort timeframes by priority (daily first, then lower frequencies)
        timeframes = sorted(timeframes, key=lambda tf: self.timeframe_configs[tf]['priority'], reverse=True)
        
        self.progress['start_time'] = datetime.now()
        self.progress['total_requests'] = len(symbols) * len(timeframes)
        
        self.logger.info(f"🚀 Starting 20-Year Parquet Data Setup")
        self.logger.info(f"📊 Symbols: {len(symbols)} ({', '.join(symbols.values())})")
        self.logger.info(f"⏰ Timeframes: {len(timeframes)} ({', '.join(timeframes)})")
        self.logger.info(f"📈 Total combinations: {self.progress['total_requests']}")
        self.logger.info(f"💾 Output directory: {self.data_dir}")
        
        results = {}
        
        # Process each symbol-timeframe combination
        for symbol, symbol_name in symbols.items():
            results[symbol_name] = {}
            
            for timeframe in timeframes:
                success = self.fetch_symbol_timeframe(symbol, symbol_name, timeframe)
                results[symbol_name][timeframe] = success
                
                # Print progress
                progress = (self.progress['completed_requests'] + self.progress['failed_requests']) / self.progress['total_requests'] * 100
                self.logger.info(f"📊 Overall Progress: {progress:.1f}%")
        
        # Generate summary
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results: dict):
        """Generate comprehensive summary report"""
        elapsed = datetime.now() - self.progress['start_time']
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📋 20-YEAR PARQUET DATA SETUP SUMMARY")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"⏱️  Duration: {elapsed}")
        self.logger.info(f"📊 Total Requests: {self.progress['total_requests']}")
        self.logger.info(f"✅ Successful: {self.progress['completed_requests']}")
        self.logger.info(f"❌ Failed: {self.progress['failed_requests']}")
        self.logger.info(f"📈 Total Candles: {self.progress['total_candles_fetched']:,}")
        
        # Calculate total data size
        total_size = 0
        total_files = 0
        
        for symbol_dir in self.data_dir.iterdir():
            if symbol_dir.is_dir():
                for parquet_file in symbol_dir.glob('*.parquet'):
                    total_size += parquet_file.stat().st_size
                    total_files += 1
        
        total_size_mb = total_size / (1024 * 1024)
        
        self.logger.info(f"💾 Total Files: {total_files}")
        self.logger.info(f"💾 Total Size: {total_size_mb:.1f} MB")
        
        # Per-symbol summary
        self.logger.info(f"\n📊 PER-SYMBOL SUMMARY:")
        for symbol_name, timeframes in results.items():
            successful_tfs = sum(1 for success in timeframes.values() if success)
            total_tfs = len(timeframes)
            self.logger.info(f"  📈 {symbol_name}: {successful_tfs}/{total_tfs} timeframes")
        
        # Save detailed report
        report_file = self.data_dir / 'setup_report.json'
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': elapsed.total_seconds(),
            'progress': self.progress,
            'results': results,
            'total_size_mb': total_size_mb,
            'total_files': total_files
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"📄 Detailed report saved: {report_file}")
        self.logger.info(f"{'='*80}")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup 20-year historical data in parquet format')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (default: all major indices)')
    parser.add_argument('--timeframes', type=str, help='Comma-separated timeframes (default: all)')
    parser.add_argument('--data-dir', type=str, default='data/parquet', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = ParquetDataSetup(data_dir=args.data_dir)
    
    # Process symbols
    symbols = None
    if args.symbols:
        symbol_list = [s.strip() for s in args.symbols.split(',')]
        symbols = {}
        for symbol in symbol_list:
            if symbol.upper() == 'NIFTY50':
                symbols['NSE:NIFTY50-INDEX'] = 'NIFTY50'
            elif symbol.upper() == 'BANKNIFTY':
                symbols['NSE:NIFTYBANK-INDEX'] = 'BANKNIFTY'
            else:
                # Assume it's an index
                symbols[f'NSE:{symbol.upper()}-INDEX'] = symbol.upper()
    
    # Process timeframes
    timeframes = None
    if args.timeframes:
        timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    
    try:
        # Run the setup
        results = setup.fetch_all_data(symbols=symbols, timeframes=timeframes)
        
        print(f"\n🎉 20-Year Parquet Data Setup Complete!")
        print(f"📁 Data location: {setup.data_dir}")
        print(f"🚀 Ready for ultra-fast backtesting!")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Setup interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 