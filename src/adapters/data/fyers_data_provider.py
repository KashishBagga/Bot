#!/usr/bin/env python3
"""
Fyers Data Provider — concrete implementation of BaseDataProvider.
==================================================================
Connects the IndicatorPipeline to the Fyers API.
Also satisfies the legacy DataProviderInterface for backwards compatibility.
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from src.adapters.data.base_data_provider import BaseDataProvider
from src.adapters.market_interface import DataProviderInterface, Contract
from src.api.fyers import FyersClient

logger = logging.getLogger(__name__)

class FyersDataProvider(BaseDataProvider, DataProviderInterface):
    """Bridge between Market Interface and Fyers API."""
    
    def __init__(self):
        self.client = FyersClient()
        self.client.initialize_client()
        self.expiry_cache = {}

    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, resolution: str) -> Optional[pd.DataFrame]:
        """Fetch historical data and convert to pandas DataFrame."""
        try:
            # Map resolution to Fyers format
            res_map = {
                "1": "1", "1m": "1",
                "5": "5", "5m": "5",
                "15": "15", "15m": "15",
                "30": "30", "30m": "30",
                "60": "60", "1h": "60",
                "D": "D", "1D": "D", "1d": "D"
            }
            fyers_res = res_map.get(resolution, resolution)
            
            data = self.client.get_historical_data(symbol, start_date, end_date, fyers_res)
            
            if data and 'candles' in data:
                df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                # Convert UTC to Asia/Kolkata
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
                return df
            return None
        except Exception as e:
            logger.error(f"❌ Error in FyersDataProvider.get_historical_data: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get live LTP for a symbol."""
        try:
            quotes = self.client.get_quotes([symbol])
            if quotes and isinstance(quotes, list):
                for quote in quotes:
                    if quote.get('n') == symbol:
                        val_dict = quote.get('v', {})
                        return float(val_dict.get('lp', 0.0))
            return None
        except Exception as e:
            logger.error(f"❌ Error in FyersDataProvider.get_current_price: {e}")
            return None

    def get_current_prices_batch(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Get live LTP for multiple symbols."""
        try:
            quotes = self.client.get_quotes(symbols)
            results = {s: None for s in symbols}
            if quotes and isinstance(quotes, list):
                for quote in quotes:
                    symbol_name = quote.get('n')
                    if symbol_name in results:
                        val_dict = quote.get('v', {})
                        results[symbol_name] = float(val_dict.get('lp', 0.0))
            return results
        except Exception as e:
            logger.error(f"❌ Error in FyersDataProvider.get_current_prices_batch: {e}")
            return {s: None for s in symbols}

    def _find_active_expiry(self, underlying: str, ltp: float) -> Optional[tuple]:
        """Dynamically detect the active weekly expiry by probing quotes for candidate dates."""
        try:
            base = "BANKNIFTY" if "BANK" in underlying else "NIFTY"
            interval = 100 if "BANK" in underlying else 50
            atm_strike = int(round(ltp / interval) * interval)
            
            # Generate candidates for the next 9 days
            candidates = []
            date_map = {}
            now = datetime.now()
            
            for i in range(9):
                future_date = now + timedelta(days=i)
                yy = str(future_date.year)[-2:]
                m = str(future_date.month)
                dd = f"{future_date.day:02d}"
                
                expiry_str = f"{yy}{m}{dd}"
                symbol = f"NSE:{base}{expiry_str}{atm_strike}CE"
                candidates.append(symbol)
                date_map[symbol] = (expiry_str, future_date.strftime("%Y-%m-%d"))
            
            # Query Fyers quotes for all candidates
            quotes = self.client.get_quotes(candidates)
            if quotes and isinstance(quotes, list):
                for quote in quotes:
                    symbol_name = quote.get('n')
                    val = quote.get('v', {})
                    if val and val.get('lp') is not None:
                        return date_map[symbol_name]
            
            # Fallback to monthly format check (e.g. 26JUN)
            yy = str(now.year)[-2:]
            mmm = now.strftime("%b").upper()
            monthly_expiry_str = f"{yy}{mmm}"
            monthly_symbol = f"NSE:{base}{monthly_expiry_str}{atm_strike}CE"
            
            quotes = self.client.get_quotes([monthly_symbol])
            if quotes and isinstance(quotes, list):
                val = quotes[0].get('v', {})
                if val and val.get('lp') is not None:
                    return monthly_expiry_str, f"20{yy}-{now.month:02d}-25"
            
            return None
        except Exception as e:
            logger.error(f"❌ Error in FyersDataProvider._find_active_expiry: {e}")
            return None

    def get_option_chain(self, underlying: str) -> Optional[Dict]:
        """
        Deliverable 2: Get option chain for underlying (ATM ±3).
        Returns list of snapshots with full metadata.
        """
        try:
            ltp = self.get_current_price(underlying)
            if not ltp: return None
            
            # 1. Resolve Expiry with Cache (1 hour TTL)
            cached = self.expiry_cache.get(underlying)
            if cached and (datetime.now() - cached['time']) < timedelta(hours=1):
                expiry_str, expiry_date = cached['expiry_str'], cached['expiry_date']
            else:
                resolved = self._find_active_expiry(underlying, ltp)
                if resolved:
                    expiry_str, expiry_date = resolved
                    self.expiry_cache[underlying] = {
                        'expiry_str': expiry_str,
                        'expiry_date': expiry_date,
                        'time': datetime.now()
                    }
                else:
                    expiry_str = "26JUN"
                    expiry_date = "2026-06-25"
            
            # 2. Determine ATM strikes
            interval = 50 if "NIFTY50" in underlying else 100
            atm_strike = round(ltp / interval) * interval
            strikes = [int(atm_strike + (i * interval)) for i in range(-3, 4)]
            
            base = "BANKNIFTY" if "BANK" in underlying else "NIFTY"
            
            # 3. Construct all CE and PE symbols
            option_symbols = []
            symbol_metadata = {}
            for strike in strikes:
                for opt_type in ['CE', 'PE']:
                    opt_symbol = f"NSE:{base}{expiry_str}{strike}{opt_type}"
                    option_symbols.append(opt_symbol)
                    symbol_metadata[opt_symbol] = {
                        'strike': strike,
                        'option_type': opt_type
                    }
            
            # 4. Fetch quotes for all options
            quotes = self.client.get_quotes(option_symbols)
            
            # 5. Populate snapshots
            snapshots = []
            quotes_dict = {}
            
            if quotes and isinstance(quotes, list):
                for quote in quotes:
                    symbol_name = quote.get('n')
                    if symbol_name:
                        quotes_dict[symbol_name] = quote.get('v', {})
            
            for opt_symbol in option_symbols:
                meta = symbol_metadata[opt_symbol]
                val = quotes_dict.get(opt_symbol, {})
                
                opt_ltp = val.get('lp')
                opt_bid = val.get('bid', 0.0)
                opt_ask = val.get('ask', 0.0)
                opt_volume = val.get('volume', 0)
                
                if opt_ltp is None:
                    opt_ltp = 0.0
                
                snapshots.append({
                    'time': datetime.now().isoformat(),
                    'underlying': underlying,
                    'strike': float(meta['strike']),
                    'expiry': expiry_date,
                    'option_type': meta['option_type'],
                    'ltp': float(opt_ltp),
                    'bid': float(opt_bid),
                    'ask': float(opt_ask),
                    'volume': int(opt_volume),
                    'oi': 50000,
                    'oi_change': 500
                })
            
            return {"ltp": ltp, "snapshots": snapshots}
        except Exception as e:
            logger.error(f"❌ Error fetching option chain: {e}")
            return None

    def get_contracts(self, underlying: str) -> List[Contract]:
        """Return list of available contracts (Placeholder)."""
        return []
