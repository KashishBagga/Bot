#!/usr/bin/env python3
"""
Sync Parquet Data
Fetch any missing historical data from Fyers for all symbols/timeframes and update parquet files.
- Does NOT compute indicators; stores raw OHLCV only
- Merges and de-duplicates by timestamp
- Safe to run repeatedly
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Ensure src is importable
sys.path.append(str(Path(__file__).parent / "src"))

from src.api.fyers import FyersClient
from src.data.parquet_data_store import ParquetDataStore

# Resolution mapping for Fyers
RESOLUTION_MAP = {
	'1min': '1', '3min': '3', '5min': '5', '15min': '15', '30min': '30',
	'1hour': '60', '4hour': '240', '1day': 'D'
}
SMALL_RESOLUTIONS = {'1', '3', '5', '15', '30', '60', '120', '180', '240'}

RAW_COLUMNS = ["open", "high", "low", "close", "volume"]

def _read_parquet_indexed(file_path: Path) -> pd.DataFrame:
	if not file_path.exists():
		return pd.DataFrame()
	try:
		df = pd.read_parquet(file_path)
		if 'time' in df.columns:
			df['time'] = pd.to_datetime(df['time'])
			df = df.set_index('time')
		else:
			df.index = pd.to_datetime(df.index)
			df.index.name = 'time'
		return df
	except Exception as e:
		print(f"❌ Error reading {file_path}: {e}")
		return pd.DataFrame()

def _store_raw_parquet(file_path: Path, df: pd.DataFrame):
	# Keep only raw columns in this writer
	cols = [c for c in RAW_COLUMNS if c in df.columns]
	if not cols:
		# Nothing to store
		return
	to_store = df[cols].reset_index()
	to_store.to_parquet(file_path, compression='snappy', engine='pyarrow', index=False)

def _fetch_history(fyers, symbol: str, resolution: str, start_date: str, end_date: str) -> pd.DataFrame:
	data = {
		"symbol": symbol,
		"resolution": resolution,
		"date_format": "1",
		"range_from": start_date,
		"range_to": end_date,
		"cont_flag": "1",
	}
	resp = fyers.history(data)
	if not isinstance(resp, dict) or resp.get('code') != 200:
		msg = resp.get('message', 'Unknown error') if isinstance(resp, dict) else str(resp)
		raise RuntimeError(f"API Error for {symbol} {resolution}: {msg}")
	candles = resp.get('candles', [])
	if not candles:
		return pd.DataFrame()
	df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
	df['time'] = pd.to_datetime(df['time'], unit='s')
	return df.set_index('time')

def _fetch_history_chunked(fyers, symbol: str, resolution: str, start_date: str, end_date: str, chunk_days: int) -> pd.DataFrame:
	all_chunks: List[pd.DataFrame] = []
	cur_start = datetime.strptime(start_date, '%Y-%m-%d')
	final_end = datetime.strptime(end_date, '%Y-%m-%d')
	while cur_start <= final_end:
		cur_end = min(cur_start + timedelta(days=chunk_days), final_end)
		try:
			chunk = _fetch_history(
				fyers, symbol, resolution,
				cur_start.strftime('%Y-%m-%d'), cur_end.strftime('%Y-%m-%d')
			)
			if not chunk.empty:
				all_chunks.append(chunk)
		except Exception as e:
			print(f"  ❌ Chunk {cur_start:%Y-%m-%d} - {cur_end:%Y-%m-%d} failed: {e}")
		cur_start = cur_end + timedelta(days=1)
	if not all_chunks:
		return pd.DataFrame()
	combined = pd.concat(all_chunks).sort_index()
	return combined[~combined.index.duplicated(keep='last')]

def sync_symbol_timeframe(fyers, store: ParquetDataStore, symbol: str, timeframe: str) -> bool:
	file_path = store._get_timeframe_file(symbol, timeframe)
	resolution = RESOLUTION_MAP.get(timeframe)
	if not resolution:
		print(f"⚠️ Unknown timeframe {timeframe}; skipping")
		return False
	# Determine range to fetch
	existing = _read_parquet_indexed(file_path)
	if existing.empty:
		# Fresh fetch for last 5 years
		end_date = (datetime.now() - timedelta(days=0)).strftime('%Y-%m-%d')
		start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
	else:
		last_ts = existing.index.max()
		# Re-fetch from previous day to handle incomplete last day, will de-dup
		start_date = (last_ts - timedelta(days=1)).strftime('%Y-%m-%d')
		end_date = (datetime.now() - timedelta(days=0)).strftime('%Y-%m-%d')
	# Skip if up-to-date (no new days)
	if existing is not None and not existing.empty:
		if pd.to_datetime(end_date) <= existing.index.max().normalize():
			return True
	# Fetch
	try:
		if resolution in SMALL_RESOLUTIONS:
			chunk_days = 90
			fetched = _fetch_history_chunked(fyers, symbol, resolution, start_date, end_date, chunk_days)
		else:
			fetched = _fetch_history(fyers, symbol, resolution, start_date, end_date)
		if fetched.empty:
			print(f"  ⚠️ No new data for {symbol} {timeframe}")
			return True
		# Merge raw only
		if existing.empty:
			merged = fetched
		else:
			merged = pd.concat([existing[RAW_COLUMNS] if all(c in existing.columns for c in RAW_COLUMNS) else existing, fetched])
			merged = merged.sort_index()
			merged = merged[~merged.index.duplicated(keep='last')]
		_store_raw_parquet(file_path, merged)
		return True
	except Exception as e:
		print(f"❌ Failed sync for {symbol} {timeframe}: {e}")
		return False

def main():
	import argparse
	parser = argparse.ArgumentParser(description="Sync parquet data from Fyers (raw OHLCV only)")
	parser.add_argument('--symbols', type=str, help='Comma-separated symbols (e.g., NSE:NIFTYBANK-INDEX,NSE:NIFTY50-INDEX). Default: all existing parquet symbols.')
	parser.add_argument('--timeframes', type=str, help='Comma-separated timeframes. Default: all known timeframes.')
	parser.add_argument('--data-dir', type=str, default='data/parquet', help='Parquet data directory')
	args = parser.parse_args()

	store = ParquetDataStore(data_dir=args.data_dir)
	# Determine symbols
	if args.symbols:
		symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
	else:
		# Use existing symbols; if none, fall back to major indices
		symbols = store.get_available_symbols()
		if not symbols:
			symbols = ['NSE:NIFTYBANK-INDEX', 'NSE:NIFTY50-INDEX']
	# Determine timeframes
	if args.timeframes:
		timeframes = [tf.strip() for tf in args.timeframes.split(',') if tf.strip()]
	else:
		timeframes = list(RESOLUTION_MAP.keys())

	# Initialize Fyers
	client = FyersClient()
	if not client.initialize_client():
		print("❌ Could not initialize Fyers client. Ensure credentials and FYERS_AUTH_CODE are set.")
		return 1
	fyers = client.fyers

	print(f"\n🚀 Syncing parquet data")
	print(f"📁 Dir: {store.data_dir}")
	print(f"📈 Symbols: {symbols}")
	print(f"🕒 Timeframes: {timeframes}")

	success = True
	for symbol in symbols:
		print(f"\n▶️ {symbol}")
		for tf in timeframes:
			ok = sync_symbol_timeframe(fyers, store, symbol, tf)
			status = "✅" if ok else "❌"
			print(f"  {status} {tf}")

	# Update metadata end_date to today for any symbol/timeframe we touched
	for symbol in symbols:
		try:
			# Name is not tracked here; use symbol as name for now
			store._update_metadata(symbol, symbol, start_date='unknown', end_date=datetime.now().strftime('%Y-%m-%d'), candles_count=0)
		except Exception:
			pass
	store._save_metadata()
	print("\n🎉 Sync complete!")
	return 0 if success else 1

if __name__ == '__main__':
	sys.exit(main()) 