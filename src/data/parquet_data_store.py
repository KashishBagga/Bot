"""
Parquet Data Store System
Efficiently stores and retrieves historical market data using parquet files.
Supports multiple timeframes and large datasets (5+ years).
"""
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import hashlib
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

class ParquetDataStore:
    """Efficient parquet-based data storage for market data."""
    
    # Available timeframes for storage
    TIMEFRAMES = {
        '1min': '1T',      # 1 minute
        '3min': '3T',      # 3 minutes
        '5min': '5T',      # 5 minutes
        '15min': '15T',    # 15 minutes
        '30min': '30T',    # 30 minutes
        '1hour': '1H',     # 1 hour
        '4hour': '4H',     # 4 hours
        '1day': '1D'       # 1 day
    }
    
    def __init__(self, data_dir: str = "data/parquet"):
        """Initialize the parquet data store.
        
        Args:
            data_dir: Directory to store parquet files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _get_symbol_dir(self, symbol: str) -> Path:
        """Get directory for a symbol."""
        # Clean symbol name for directory
        clean_symbol = symbol.replace(':', '_').replace('-', '_')
        symbol_dir = self.data_dir / clean_symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir
    
    def _get_timeframe_file(self, symbol: str, timeframe: str) -> Path:
        """Get parquet file path for symbol and timeframe."""
        symbol_dir = self._get_symbol_dir(symbol)
        return symbol_dir / f"{timeframe}.parquet"
    
    def fetch_and_store_data(self, fyers, symbols: Dict[str, str], 
                           timeframes_to_fetch: List[str] = None, years_back: int = 5):
        """Fetch historical data directly for each timeframe and store separately.
        
        Args:
            fyers: Fyers API client
            symbols: Dict mapping symbol codes to display names
            timeframes_to_fetch: List of timeframes to fetch (None for all)
            years_back: Years of historical data to fetch
        """
        # Default to all timeframes if none specified
        if timeframes_to_fetch is None:
            timeframes_to_fetch = list(self.TIMEFRAMES.keys())
        
        # Convert timeframe names to API resolution values
        timeframe_to_resolution = {
            '1min': '1', '3min': '3', '5min': '5', '15min': '15', '30min': '30',
            '1hour': '60', '4hour': '240', '1day': 'D'
        }
        
        print(f"🚀 Setting up parquet data store with DIRECT API fetching:")
        print(f"  📅 Period: {years_back} years")
        print(f"  📈 Symbols: {list(symbols.values())}")
        print(f"  🎯 Timeframes: {timeframes_to_fetch}")
        print(f"  ✨ Strategy: Fetch each timeframe directly from API (most accurate)")
        
        # Calculate date range
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=years_back * 365 + 1)).strftime('%Y-%m-%d')
        
        print(f"📅 Date range: {start_date} to {end_date}")
        
        for symbol, name in symbols.items():
            print(f"\n📈 Processing {name} ({symbol})...")
            
            # Process each timeframe separately
            for timeframe in timeframes_to_fetch:
                if timeframe not in timeframe_to_resolution:
                    print(f"  ⚠️ Skipping unknown timeframe: {timeframe}")
                    continue
                
                resolution = timeframe_to_resolution[timeframe]
                print(f"\n  🎯 Fetching {timeframe} timeframe (resolution: {resolution})...")
                
                # Check if timeframe file already exists
                timeframe_file = self._get_timeframe_file(symbol, timeframe)
                if timeframe_file.exists():
                    print(f"    ✅ {timeframe} data already exists, skipping")
                    continue
                
                try:
                    # Determine if we need chunked fetching
                    small_resolutions = ['1', '3', '5', '15', '30', '60', '120', '180', '240']
                    use_chunks = resolution in small_resolutions
                    chunk_days = 90 if use_chunks else 365
                    
                    if use_chunks:
                        print(f"    📦 Using chunked fetching (max {chunk_days} days per request)")
                        df = self._fetch_candles_chunked(fyers, symbol, resolution, 
                                                       start_date, end_date, chunk_days)
                    else:
                        print(f"    📊 Fetching all data in single request")
                        df = self._fetch_candles(fyers, symbol, resolution, 
                                               start_date, end_date)
                    
                    if df.empty:
                        print(f"    ❗ No data available for {timeframe}")
                        continue
                    
                    print(f"    ✅ Fetched {len(df)} candles for {timeframe}")
                    
                    # Add technical indicators
                    df = self._add_indicators(df)
                    
                    # Store this timeframe
                    self._store_timeframe_data(symbol, timeframe, df)
                    print(f"    💾 Stored {timeframe}: {len(df)} candles")
                    
                except Exception as e:
                    print(f"    ❌ Failed to fetch {timeframe}: {e}")
                    continue
            
            # Update metadata for this symbol
            available_timeframes = self.get_available_timeframes(symbol)
            if available_timeframes:
                # Use the largest dataset to represent the symbol
                largest_tf = available_timeframes[0]  # Start with first available
                largest_count = 0
                
                for tf in available_timeframes:
                    tf_file = self._get_timeframe_file(symbol, tf)
                    if tf_file.exists():
                        try:
                            tf_df = pd.read_parquet(tf_file)
                            if len(tf_df) > largest_count:
                                largest_count = len(tf_df)
                                largest_tf = tf
                        except:
                            continue
                
                self._update_metadata(symbol, name, start_date, end_date, largest_count)
        
        self._save_metadata()
        print(f"\n✅ Parquet data store setup completed!")
        print(f"🎉 All timeframes fetched directly from API for maximum accuracy!")
        self._print_storage_info()
    
    def _fetch_candles(self, fyers, symbol: str, resolution: str, 
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch candles from Fyers API."""
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": start_date,
            "range_to": end_date,
            "cont_flag": "1"
        }
        
        response = fyers.history(data)
        
        if isinstance(response, dict) and 'code' in response and response['code'] != 200:
            raise Exception(f"API Error: {response.get('message', 'Unknown error')}")
        
        candles = response.get('candles')
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def _fetch_candles_chunked(self, fyers, symbol: str, resolution: str, 
                              start_date: str, end_date: str, chunk_days: int) -> pd.DataFrame:
        """Fetch candles in chunks to handle API limitations."""
        all_chunks = []
        current_start = datetime.strptime(start_date, '%Y-%m-%d')
        final_end = datetime.strptime(end_date, '%Y-%m-%d')
        
        chunk_count = 0
        while current_start < final_end:
            chunk_count += 1
            current_end = min(current_start + timedelta(days=chunk_days), final_end)
            
            chunk_start_str = current_start.strftime('%Y-%m-%d')
            chunk_end_str = current_end.strftime('%Y-%m-%d')
            
            print(f"  📦 Chunk {chunk_count}: {chunk_start_str} to {chunk_end_str}")
            
            try:
                chunk_df = self._fetch_candles(fyers, symbol, resolution, 
                                             chunk_start_str, chunk_end_str)
                if not chunk_df.empty:
                    all_chunks.append(chunk_df)
                    print(f"    ✅ Got {len(chunk_df)} candles")
                else:
                    print(f"    ⚠️ No data for this chunk")
            except Exception as e:
                print(f"    ❌ Chunk failed: {e}")
                # Continue with next chunk instead of failing completely
            
            current_start = current_end + timedelta(days=1)
        
        if not all_chunks:
            return pd.DataFrame()
        
        # Combine all chunks
        combined_df = pd.concat(all_chunks, ignore_index=False)
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        print(f"  🔗 Combined {len(all_chunks)} chunks into {len(combined_df)} total candles")
        return combined_df
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        if df.empty or len(df) < 50:  # Need enough data for indicators
            return df
        
        df = df.copy()
        
        try:
            # EMA indicators
            df['ema_9'] = EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema_21'] = EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
            df['ema'] = df['ema_20']  # Default EMA
            
            # RSI
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            
            # MACD
            macd = MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # ATR
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            
            # Bollinger Bands
            bb = BollingerBands(df['close'])
            df['bollinger_upper'] = bb.bollinger_hband()
            df['bollinger_lower'] = bb.bollinger_lband()
            df['bollinger_mid'] = bb.bollinger_mavg()
            
        except Exception as e:
            print(f"    ⚠️ Warning: Could not calculate some indicators: {e}")
        
        return df
    
    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample dataframe to a different timeframe."""
        if df.empty:
            return df
        
        pandas_freq = self.TIMEFRAMES[timeframe]
        
        # OHLCV aggregation
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Add indicator columns to aggregation
        indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        for col in indicator_cols:
            agg_dict[col] = 'last'  # Use last value for indicators
        
        resampled = df.resample(pandas_freq).agg(agg_dict)
        resampled = resampled.dropna()  # Remove rows with NaN values
        
        return resampled
    
    def _store_all_timeframes(self, symbol: str, name: str, base_df: pd.DataFrame, base_resolution: str):
        """Generate and store data for all timeframes."""
        base_timeframe = f"{base_resolution}min"
        
        # Store base timeframe
        if base_timeframe in self.TIMEFRAMES:
            self._store_timeframe_data(symbol, base_timeframe, base_df)
            print(f"  💾 Stored {base_timeframe}: {len(base_df)} candles")
        
        # Generate and store higher timeframes
        for timeframe in self.TIMEFRAMES:
            if timeframe == base_timeframe:
                continue  # Already stored
            
            # Only generate higher timeframes
            base_minutes = int(base_resolution)
            target_minutes = self._timeframe_to_minutes(timeframe)
            
            if target_minutes <= base_minutes:
                continue  # Can't generate lower timeframes
            
            try:
                resampled_df = self._resample_to_timeframe(base_df, timeframe)
                if not resampled_df.empty:
                    self._store_timeframe_data(symbol, timeframe, resampled_df)
                    print(f"  💾 Stored {timeframe}: {len(resampled_df)} candles")
                else:
                    print(f"  ⚠️ No data for {timeframe} after resampling")
            except Exception as e:
                print(f"  ❌ Failed to generate {timeframe}: {e}")
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe to minutes for comparison."""
        mapping = {
            '1min': 1, '3min': 3, '5min': 5, '15min': 15, '30min': 30,
            '1hour': 60, '4hour': 240, '1day': 1440
        }
        return mapping.get(timeframe, 0)
    
    def _store_timeframe_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Store dataframe for a specific timeframe."""
        file_path = self._get_timeframe_file(symbol, timeframe)
        
        # Reset index to make 'time' a column for parquet storage
        df_to_store = df.reset_index()
        
        # Write to parquet with compression
        table = pa.Table.from_pandas(df_to_store)
        pq.write_table(table, file_path, compression='snappy')
    
    def load_data(self, symbol: str, timeframe: str, days_back: Optional[int] = None) -> pd.DataFrame:
        """Load data for a specific symbol and timeframe.
        
        Args:
            symbol: Symbol to load
            timeframe: Timeframe to load (e.g., '1min', '5min', '1day')
            days_back: Number of days to load (None for all data)
            
        Returns:
            DataFrame with the requested data
        """
        file_path = self._get_timeframe_file(symbol, timeframe)
        
        if not file_path.exists():
            print(f"❌ No data found for {symbol} at {timeframe}")
            return pd.DataFrame()
        
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)
            
            # Set time as index
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Filter by days_back if specified
            if days_back is not None:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                df = df[df.index >= cutoff_date]
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data for {symbol} at {timeframe}: {e}")
            return pd.DataFrame()
    
    def load_multi_timeframe_data(self, symbol: str, timeframes: List[str], 
                                 days_back: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load data for multiple timeframes.
        
        Args:
            symbol: Symbol to load
            timeframes: List of timeframes to load
            days_back: Number of days to load (None for all data)
            
        Returns:
            Dict mapping timeframe to DataFrame
        """
        result = {}
        for timeframe in timeframes:
            df = self.load_data(symbol, timeframe, days_back)
            if not df.empty:
                result[timeframe] = df
        return result
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        symbols = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name != '__pycache__':
                # Convert directory name back to symbol
                symbol = item.name
                
                # Handle NSE symbols specifically
                if symbol.startswith('NSE_') and symbol.endswith('_INDEX'):
                    # NSE_NIFTY50_INDEX -> NSE:NIFTY50-INDEX
                    # NSE_NIFTYBANK_INDEX -> NSE:NIFTYBANK-INDEX
                    symbol = symbol.replace('_INDEX', '-INDEX')
                    symbol = symbol.replace('NSE_', 'NSE:', 1)
                elif symbol.startswith('NSE_') and symbol.endswith('_EQ'):
                    # NSE_RELIANCE_EQ -> NSE:RELIANCE-EQ
                    symbol = symbol.replace('_EQ', '-EQ')
                    symbol = symbol.replace('NSE_', 'NSE:', 1)
                else:
                    # Generic fallback
                    symbol = symbol.replace('_', ':')
                
                symbols.append(symbol)
        return symbols
    
    def get_available_timeframes(self, symbol: str) -> List[str]:
        """Get available timeframes for a symbol."""
        symbol_dir = self._get_symbol_dir(symbol)
        timeframes = []
        for file_path in symbol_dir.glob("*.parquet"):
            timeframe = file_path.stem
            if timeframe in self.TIMEFRAMES:
                timeframes.append(timeframe)
        return sorted(timeframes, key=lambda x: self._timeframe_to_minutes(x))
    
    def _has_complete_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Check if we already have complete data for the symbol and date range."""
        symbol_key = symbol
        if symbol_key in self.metadata:
            stored_start = self.metadata[symbol_key].get('start_date')
            stored_end = self.metadata[symbol_key].get('end_date')
            
            if stored_start and stored_end:
                return (stored_start <= start_date and stored_end >= end_date)
        
        return False
    
    def _update_metadata(self, symbol: str, name: str, start_date: str, 
                        end_date: str, candles_count: int):
        """Update metadata for a symbol."""
        symbol_dir = self._get_symbol_dir(symbol)
        total_size = sum(f.stat().st_size for f in symbol_dir.glob("*.parquet"))
        
        self.metadata[symbol] = {
            'name': name,
            'start_date': start_date,
            'end_date': end_date,
            'base_candles_count': candles_count,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'timeframes': self.get_available_timeframes(symbol),
            'last_updated': datetime.now().isoformat()
        }
    
    def _print_storage_info(self):
        """Print information about stored data."""
        print(f"\n📊 Parquet Data Store Summary:")
        
        total_size_mb = 0
        total_symbols = len(self.metadata)
        
        for symbol, info in self.metadata.items():
            size_mb = info.get('total_size_mb', 0)
            total_size_mb += size_mb
            timeframes = info.get('timeframes', [])
            
            print(f"  • {info['name']} ({symbol})")
            print(f"    📅 Period: {info['start_date']} to {info['end_date']}")
            print(f"    📊 Base candles: {info['base_candles_count']:,}")
            print(f"    🎯 Timeframes: {len(timeframes)} ({', '.join(timeframes)})")
            print(f"    💾 Size: {size_mb} MB")
        
        print(f"\n🎉 Total: {total_symbols} symbols, {total_size_mb:.2f} MB")
        print(f"📁 Storage directory: {self.data_dir}")
    
    def get_storage_info(self) -> Dict:
        """Get detailed storage information."""
        info = {
            'total_symbols': len(self.metadata),
            'total_size_mb': 0,
            'storage_directory': str(self.data_dir),
            'symbols': []
        }
        
        for symbol, data in self.metadata.items():
            size_mb = data.get('total_size_mb', 0)
            info['total_size_mb'] += size_mb
            
            info['symbols'].append({
                'symbol': symbol,
                'name': data['name'],
                'date_range': f"{data['start_date']} to {data['end_date']}",
                'base_candles_count': data['base_candles_count'],
                'timeframes': data.get('timeframes', []),
                'size_mb': size_mb,
                'last_updated': data['last_updated']
            })
        
        info['total_size_mb'] = round(info['total_size_mb'], 2)
        return info
    
    def clear_data(self, symbol: str = None):
        """Clear stored data.
        
        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            symbol_dir = self._get_symbol_dir(symbol)
            if symbol_dir.exists():
                for file_path in symbol_dir.glob("*.parquet"):
                    file_path.unlink()
                symbol_dir.rmdir()
                
                if symbol in self.metadata:
                    del self.metadata[symbol]
                
                print(f"✅ Cleared data for {symbol}")
        else:
            # Clear all data
            for item in self.data_dir.iterdir():
                if item.is_dir():
                    for file_path in item.rglob("*.parquet"):
                        file_path.unlink()
                    item.rmdir()
            
            self.metadata.clear()
            print("✅ Cleared all data")
        
        self._save_metadata() 