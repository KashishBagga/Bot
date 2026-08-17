#!/usr/bin/env python3
"""
Fyers Data Provider — concrete implementation of BaseDataProvider.
==================================================================
Connects the IndicatorPipeline to the Fyers API.
Also satisfies the legacy DataProviderInterface for backwards compatibility.
"""

import pandas as pd
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any
from src.adapters.data.base_data_provider import BaseDataProvider
from src.adapters.market_interface import DataProviderInterface, Contract
from src.api.fyers import FyersClient

logger = logging.getLogger(__name__)

RES_MAP = {
    "1": "1", "1m": "1",
    "5": "5", "5m": "5",
    "15": "15", "15m": "15",
    "30": "30", "30m": "30",
    "60": "60", "1h": "60",
    "D": "D", "1D": "D", "1d": "D"
}
BAR_SECONDS = {"1": 60, "5": 300, "15": 900, "30": 1800, "60": 3600, "D": 86400}

class FyersDataProvider(BaseDataProvider, DataProviderInterface):
    """Bridge between Market Interface and Fyers API."""

    def __init__(self):
        self.client = FyersClient()
        self.client.initialize_client()
        self.expiry_cache = {}
        from src.models.postgres_database import PostgresDatabase
        self.db = PostgresDatabase()

    def _fetch_from_fyers(self, symbol: str, start_date: datetime, end_date: datetime,
                           fyers_res: str) -> Optional[pd.DataFrame]:
        """Raw Fyers historical-data call, parsed into a clean DataFrame. No caching —
        callers decide what (if anything) gets persisted."""
        data = self.client.get_historical_data(symbol, start_date, end_date, fyers_res)
        if not data or 'candles' not in data:
            return None
        df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        # Convert UTC to Asia/Kolkata
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        # Data hygiene: Fyers can return duplicate rows (the same forming
        # candle repeated across calls), out-of-order rows, or NaN candles.
        # Dedup (keep the freshest copy), sort chronologically, drop NaN OHLC
        # so indicators are never computed on a corrupt/unsorted series.
        df = df[~df.index.duplicated(keep='last')].sort_index()
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        return df if not df.empty else None

    def get_historical_data(self, symbol: str, start_date: datetime,
                          end_date: datetime, resolution: str) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV as a DataFrame, backed by the local `candles`
        cache (src/models/postgres_database.py) so repeat requests over the
        same historical range — backtests especially — replay from Postgres
        instead of re-hitting the Fyers API every time. Only the gap between
        what's cached and `end_date` (the "tail") is ever fetched live; a
        fully-cached range makes zero Fyers calls."""
        try:
            fyers_res = RES_MAP.get(resolution, resolution)
            bar_td = timedelta(seconds=BAR_SECONDS.get(fyers_res, 300))

            cached_rows = self.db.get_candles(symbol, fyers_res, start_date, end_date)

            head_gap = not cached_rows or (cached_rows[0]["time"] - start_date) > bar_td
            if head_gap:
                # No usable cache for this range at all — fetch it in full and
                # seed the cache so subsequent calls hit it.
                df = self._fetch_from_fyers(symbol, start_date, end_date, fyers_res)
                if df is not None:
                    self.db.save_candles(symbol, fyers_res, self._df_to_rows(df))
                return df

            tail_gap = (end_date - cached_rows[-1]["time"]) > bar_td
            if not tail_gap:
                # Cache fully covers the requested range — no Fyers call needed.
                return self._rows_to_df(cached_rows)

            # Only the tail is stale/missing — fetch from the last cached bar
            # onward (this naturally re-fetches and re-upserts the last,
            # still-forming bar too, which is harmless — see save_candles).
            fresh = self._fetch_from_fyers(symbol, cached_rows[-1]["time"], end_date, fyers_res)
            if fresh is not None:
                self.db.save_candles(symbol, fyers_res, self._df_to_rows(fresh))
            merged = self._rows_to_df(cached_rows)
            if fresh is not None:
                merged = pd.concat([merged, fresh])
                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
            return merged if not merged.empty else None
        except Exception as e:
            logger.error(f"❌ Error in FyersDataProvider.get_historical_data: {e}")
            return None

    @staticmethod
    def _df_to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
        return [
            {"time": ts.to_pydatetime(), "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close),
             "volume": int(r.volume) if pd.notna(r.volume) else None}
            for ts, r in df.iterrows()
        ]

    @staticmethod
    def _rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert('Asia/Kolkata')
        df.set_index("time", inplace=True)
        df.index.name = "timestamp"
        return df

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
        """Dynamically detect the active weekly expiry by probing quotes or history for candidate dates."""
        try:
            from src.core.options_execution_engine import ExpiryResolver
            base = "BANKNIFTY" if "BANK" in underlying else "NIFTY"
            interval = 100 if "BANK" in underlying else 50
            atm_strike = int(round(ltp / interval) * interval)
            
            # Generate candidates for the next 9 days
            candidates = []
            date_map = {}
            now = datetime.now()
            
            for i in range(9):
                future_date = (now + timedelta(days=i)).date()
                expiry_str = ExpiryResolver.date_to_fyers_expiry(future_date)
                symbol = f"NSE:{base}{expiry_str}{atm_strike}CE"
                candidates.append(symbol)
                date_map[symbol] = (expiry_str, future_date.strftime("%Y-%m-%d"))
            
            # 1. Try quotes endpoint (fastest, but subject to 429 throttle)
            quotes = self.client.get_quotes(candidates)
            if quotes and isinstance(quotes, list):
                for quote in quotes:
                    symbol_name = quote.get('n')
                    val = quote.get('v', {})
                    if val and val.get('lp') is not None:
                        return date_map[symbol_name]
            
            # Fallback to monthly format check (e.g. 26AUG instead of expired 26JUL)
            def get_last_tuesday(yr, mo):
                nm = mo + 1 if mo < 12 else 1
                ny = yr if mo < 12 else yr + 1
                last_d = date(ny, nm, 1) - timedelta(days=1)
                sub = (last_d.weekday() - 1) % 7
                return last_d - timedelta(days=sub)

            now_date = now.date()
            lt_this_month = get_last_tuesday(now.year, now.month)
            if now_date > lt_this_month:
                # This month's monthly expired -> use next month
                active_monthly_date = get_last_tuesday(
                    now.year if now.month < 12 else now.year + 1,
                    now.month + 1 if now.month < 12 else 1
                )
            else:
                active_monthly_date = lt_this_month

            monthly_expiry_str = ExpiryResolver.date_to_fyers_expiry(active_monthly_date)
            monthly_symbol = f"NSE:{base}{monthly_expiry_str}{atm_strike}CE"
            
            quotes = self.client.get_quotes([monthly_symbol])
            if quotes and isinstance(quotes, list):
                val = quotes[0].get('v', {})
                if val and val.get('lp') is not None and float(val.get('lp', 0.0)) > 0:
                    return monthly_expiry_str, active_monthly_date.strftime("%Y-%m-%d")
            
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
                    from src.core.options_execution_engine import ExpiryResolver
                    now = datetime.now()
                    if "BANK" in underlying:
                        def get_last_tuesday(yr, mo):
                            nm = mo + 1 if mo < 12 else 1
                            ny = yr if mo < 12 else yr + 1
                            last_d = date(ny, nm, 1) - timedelta(days=1)
                            return last_d - timedelta(days=(last_d.weekday() - 1) % 7)
                        fallback_date = get_last_tuesday(now.year, now.month)
                        if now.date() > fallback_date:
                            fallback_date = get_last_tuesday(
                                now.year if now.month < 12 else now.year + 1,
                                now.month + 1 if now.month < 12 else 1
                            )
                    else:
                        days_until_tuesday = (1 - now.weekday()) % 7 or 7
                        fallback_date = (now + timedelta(days=days_until_tuesday)).date()
                    expiry_str = ExpiryResolver.date_to_fyers_expiry(fallback_date)
                    expiry_date = fallback_date.strftime("%Y-%m-%d")

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
                    # PLACEHOLDER — the quotes endpoint used above has no OI field.
                    # Never read oi/oi_change from this snapshot for OI-based decisions
                    # (PCR, max pain, OI walls) — real OI comes only from
                    # src.warehouse.option_warehouse.OptionWarehouse, which uses the
                    # depth endpoint and persists to option_snapshots.
                    'oi': 50000,
                    'oi_change': 500
                })
            
            return {"ltp": ltp, "snapshots": snapshots, "expiry_str": expiry_str, "expiry_date": expiry_date}
        except Exception as e:
            logger.error(f"❌ Error fetching option chain: {e}")
            return None

    def get_contracts(self, underlying: str) -> List[Contract]:
        """Return list of available contracts (Placeholder)."""
        return []
