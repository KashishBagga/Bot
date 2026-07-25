import logging
import re
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from src.models.postgres_database import PostgresDatabase

logger = logging.getLogger("OptionExecution")

@dataclass
class OptionContract:
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'CE' or 'PE'
    premium: float
    delta: float = 0.5
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    resolved_at: datetime = None

class ExpiryResolver:
    """Resolves weekly or monthly expiry dates using database snapshots with fallbacks."""
    
    MONTH_MAP = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    
    WEEKLY_MONTH_MAP = {
        '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'O': 10, 'N': 11, 'D': 12
    }

    def __init__(self, db: PostgresDatabase):
        self.db = db

    def parse_expiry_to_date(self, expiry: str) -> Optional[date]:
        """Parses Fyers expiry string (e.g. '26723' or '26JUL') into a datetime.date object."""
        try:
            # 0. YYYY-MM-DD format (e.g. 2026-07-23)
            date_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", expiry)
            if date_match:
                return date(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))

            # 1. Weekly format (e.g. 26723 -> Year 2026, Month 7, Day 23)
            weekly_match = re.match(r"^(\d{2})([1-9OND])(\d{2})$", expiry)
            if weekly_match:
                yy = int(weekly_match.group(1))
                m_char = weekly_match.group(2)
                dd = int(weekly_match.group(3))
                year = 2000 + yy
                month = self.WEEKLY_MONTH_MAP.get(m_char, int(m_char) if m_char.isdigit() else 1)
                return date(year, month, dd)

            # 2. Monthly format (e.g. 26JUL -> Year 2026, Month July, last Thursday)
            monthly_match = re.match(r"^(\d{2})([A-Z]{3})$", expiry)
            if monthly_match:
                yy = int(monthly_match.group(1))
                mmm = monthly_match.group(2)
                year = 2000 + yy
                month = self.MONTH_MAP.get(mmm, 1)
                # Find last Thursday of the month
                # Start from last day of the month and count backwards
                next_month = month + 1 if month < 12 else 1
                next_month_year = year if month < 12 else year + 1
                first_of_next = date(next_month_year, next_month, 1)
                last_day = first_of_next - timedelta(days=1)
                
                # Backtrack to Thursday (weekday 3 in python, Mon=0 ... Sun=6)
                days_to_subtract = (last_day.weekday() - 3) % 7
                last_thursday = last_day - timedelta(days=days_to_subtract)
                return last_thursday
                
            return None
        except Exception as e:
            logger.error(f"Error parsing expiry string {expiry}: {e}")
            return None

    def date_to_fyers_expiry(self, d: date) -> str:
        """Converts a date object to Fyers weekly or monthly expiry string."""
        yy = str(d.year)[-2:]
        month = d.month
        
        # Check if d is the last Thursday of the month
        next_month = month + 1 if month < 12 else 1
        next_month_year = d.year if month < 12 else d.year + 1
        first_of_next = date(next_month_year, next_month, 1)
        last_day = first_of_next - timedelta(days=1)
        
        days_to_subtract = (last_day.weekday() - 3) % 7
        last_thursday = last_day - timedelta(days=days_to_subtract)
        
        if d == last_thursday:
            months = ["", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
            mmm = months[month]
            return f"{yy}{mmm}"
        else:
            month_chars = {10: 'O', 11: 'N', 12: 'D'}
            m_char = month_chars.get(month, str(month))
            dd = f"{d.day:02d}"
            return f"{yy}{m_char}{dd}"

    def get_active_expiry(self, underlying: str, data_provider=None) -> str:
        """Retrieves the closest active expiry from database or calls Fyers as fallback."""
        try:
            # 1. Attempt to query Postgres for expiry cache
            with self.db._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        SELECT DISTINCT expiry FROM option_snapshots
                        WHERE underlying = %s AND time >= NOW() - INTERVAL '12 hours'
                    ''', (underlying,))
                    rows = cursor.fetchall()
                    
            if rows:
                expiries = [row[0] for row in rows]
                parsed_expiries = []
                for exp in expiries:
                    d = self.parse_expiry_to_date(exp)
                    if d:
                        parsed_expiries.append((d, exp))
                
                # Sort by date and select nearest future date
                today = date.today()
                future_expiries = [(d, exp) for d, exp in parsed_expiries if d >= today]
                if future_expiries:
                    future_expiries.sort(key=lambda x: x[0])
                    fyers_exp_code = self.date_to_fyers_expiry(future_expiries[0][0])
                    logger.info(f"Resolved active expiry from DB: {fyers_exp_code} ({future_expiries[0][0]})")
                    return fyers_exp_code

            # 2. Fallback to Fyers Data Provider API
            if data_provider:
                # Query option chain to force resolve
                chain = data_provider.get_option_chain(underlying)
                if chain and 'expiry_str' in chain:
                    logger.info(f"Resolved active expiry from API: {chain['expiry_str']}")
                    return chain['expiry_str']
                # Direct cache inspection
                cached = data_provider.expiry_cache.get(underlying)
                if cached:
                    return cached['expiry_str']
                    
            # BUG FIX: Dynamically compute next Thursday instead of hardcoding "26JUL"
            # Nifty/BankNifty weekly options expire every Thursday
            _today = date.today()
            days_until_thursday = (3 - _today.weekday()) % 7  # 3=Thursday (Mon=0)
            if days_until_thursday == 0:
                days_until_thursday = 7  # Today is Thursday — use NEXT Thursday
            _next_thu = _today + timedelta(days=days_until_thursday)
            _yy = str(_next_thu.year)[2:]
            _m = _next_thu.month
            # Fyers weekly month encoding: 1-9 as digits, Oct='O', Nov='N', Dec='D'
            _month_chars = {10: 'O', 11: 'N', 12: 'D'}
            _m_char = _month_chars.get(_m, str(_m))
            _dd = f"{_next_thu.day:02d}"
            dynamic_expiry = f"{_yy}{_m_char}{_dd}"
            logger.warning(
                f"[ExpiryResolver] Both DB and API failed for {underlying}. "
                f"Falling back to dynamically computed expiry: {dynamic_expiry} ({_next_thu})"
            )
            return dynamic_expiry
        except Exception as e:
            logger.error(f"Error resolving active expiry for {underlying}: {e}")
            # Even on exception, compute a valid next-Thursday dynamically
            try:
                _today = date.today()
                days_until = (3 - _today.weekday()) % 7 or 7
                _thu = _today + timedelta(days=days_until)
                _mc = {10: 'O', 11: 'N', 12: 'D'}.get(_thu.month, str(_thu.month))
                return f"{str(_thu.year)[2:]}{_mc}{_thu.day:02d}"
            except Exception:
                return "26JUL"  # True last resort — should never reach here

class StrikeSelector:
    """Selects strikes based on underlying price, call/put type, and strategy config."""
    
    def __init__(self, underlying: str):
        self.underlying = underlying
        self.interval = 100 if "BANK" in underlying else 50

    def get_atm_strike(self, price: float) -> int:
        return int(round(price / self.interval) * self.interval)

    def select_strike(self, price: float, option_type: str, policy: str = "ATM") -> int:
        """
        Policy options:
          - 'ATM' (At The Money)
          - 'ITM_1' (1 strike In The Money)
          - 'ITM_2' (2 strikes In The Money)
          - 'OTM_1' (1 strike Out of The Money)
        """
        atm = self.get_atm_strike(price)
        
        if policy == "ATM":
            return atm
            
        elif policy == "ITM_1":
            if option_type == "CE":
                return atm - self.interval
            else: # PE
                return atm + self.interval
                
        elif policy == "ITM_2":
            if option_type == "CE":
                return atm - (2 * self.interval)
            else: # PE
                return atm + (2 * self.interval)
                
        elif policy == "OTM_1":
            if option_type == "CE":
                return atm + self.interval
            else: # PE
                return atm - self.interval
                
        logger.warning(f"Unknown strike policy '{policy}', defaulting to ATM.")
        return atm

class PremiumResolver:
    """Fetches option premium from database cache or falls back to Fyers API quotes."""
    
    def __init__(self, db: PostgresDatabase, data_provider):
        self.db = db
        self.data_provider = data_provider

    def resolve_premium(self, underlying: str, strike: float, option_type: str, expiry: str, symbol: str) -> Tuple[float, float, float, int]:
        """Returns (premium/LTP, bid, ask, volume)."""
        # 1. Try DB Warehouse lookup (TimescaleDB cache)
        try:
            db_expiry = expiry
            parsed_date = ExpiryResolver(self.db).parse_expiry_to_date(expiry)
            if parsed_date:
                db_expiry = parsed_date.strftime("%Y-%m-%d")
                
            with self.db._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        SELECT ltp, bid, ask, volume, time FROM option_snapshots
                        WHERE underlying = %s AND strike = %s AND option_type = %s AND expiry = %s
                        ORDER BY time DESC LIMIT 1
                    ''', (underlying, strike, option_type, db_expiry))
                    row = cursor.fetchone()
            
            if row:
                ltp, bid, ask, volume, timestamp = row
                # Check cache freshness (within 15 seconds)
                if (datetime.now(timestamp.tzinfo) - timestamp) < timedelta(seconds=15):
                    logger.debug(f"Retrieved cached premium from DB: LTP={ltp} for {symbol}")
                    return float(ltp), float(bid or 0.0), float(ask or 0.0), int(volume or 0)
        except Exception as e:
            logger.error(f"Error querying option premium from database: {e}")

        # 2. Cache miss or stale -> Query Fyers client API
        try:
            if self.data_provider and self.data_provider.client:
                quotes = self.data_provider.client.get_quotes([symbol])
                if quotes and isinstance(quotes, list):
                    for quote in quotes:
                        if quote.get('n') == symbol:
                            val = quote.get('v', {})
                            ltp = float(val.get('lp', 0.0))
                            bid = float(val.get('bid', 0.0))
                            ask = float(val.get('ask', 0.0))
                            volume = int(val.get('volume', 0))
                            logger.info(f"Retrieved live premium from Fyers API: LTP={ltp} for {symbol}")
                            return ltp, bid, ask, volume
        except Exception as e:
            logger.error(f"Failed to query live quote for option {symbol}: {e}")

        # Both the DB warehouse and the live API failed. DO NOT fabricate a
        # premium — a synthetic price (the old dummy 100.0) silently sizes and
        # "fills" a real order against a made-up number. Fail loudly so the
        # caller skips the trade instead.
        raise ValueError(
            f"Could not resolve a real premium for {symbol} "
            f"(DB warehouse miss/stale AND live quote failed)"
        )

class OptionExecutionEngine:
    """Main facade engine coordinating option resolution."""
    
    def __init__(self, db: PostgresDatabase, data_provider, strike_policy: str = "ATM"):
        self.db = db
        self.data_provider = data_provider
        self.strike_policy = strike_policy
        self.expiry_resolver = ExpiryResolver(db)

    def resolve(self, signal: Dict, index_ltp: float) -> OptionContract:
        """Resolves an index signal dict into a tradable OptionContract."""
        index_symbol = signal['symbol']
        signal_type = signal['signal']  # e.g., 'BUY CALL' or 'BUY PUT'
        option_type = "CE" if "CALL" in signal_type else "PE"
        
        # 1. Resolve Expiry
        expiry = self.expiry_resolver.get_active_expiry(index_symbol, self.data_provider)
        
        # 2. Select Strike
        selector = StrikeSelector(index_symbol)
        strike = selector.select_strike(index_ltp, option_type, self.strike_policy)
        
        # 3. Construct Symbol
        base = "BANKNIFTY" if "BANK" in index_symbol else "NIFTY"
        option_symbol = f"NSE:{base}{expiry}{strike}{option_type}"
        
        # 4. Resolve Premium
        resolver = PremiumResolver(self.db, self.data_provider)
        premium, bid, ask, volume = resolver.resolve_premium(index_symbol, strike, option_type, expiry, option_symbol)
        
        # 5. Delta mapping approximation (ATM is 0.50, ITM_1 is ~0.60, etc.)
        delta = 0.50
        if self.strike_policy == "ITM_1":
            delta = 0.60
        elif self.strike_policy == "ITM_2":
            delta = 0.70
        elif self.strike_policy == "OTM_1":
            delta = 0.40
            
        contract = OptionContract(
            symbol=option_symbol,
            strike=float(strike),
            expiry=expiry,
            option_type=option_type,
            premium=premium,
            delta=delta,
            bid=bid,
            ask=ask,
            volume=volume,
            resolved_at=datetime.now()
        )
        logger.info(f"Resolved index signal into option contract: {contract}")
        return contract
