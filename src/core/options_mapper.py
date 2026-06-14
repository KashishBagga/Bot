import datetime
import math

class OptionsMapper:
    """
    Resolves Index signals into tradable Option Contracts.
    Example: NSE:NIFTY50-INDEX BUY CALL -> NSE:NIFTY26MAY22000CE
    """
    
    @staticmethod
    def get_atm_strike(symbol: str, current_price: float) -> int:
        """Rounds the current price to the nearest strike interval."""
        if "NIFTYBANK" in symbol:
            interval = 100
        elif "NIFTY50" in symbol or "NIFTY" in symbol:
            interval = 50
        else:
            interval = 50 # Default
            
        return int(round(current_price / interval) * interval)
        
    @staticmethod
    def resolve_option_symbol(index_symbol: str, current_price: float, signal_type: str, expiry_str: str = None) -> str:
        """
        Maps an index symbol to an option symbol.
        expiry_str should be the Fyers formatted expiry (e.g., '26MAY' for May 2026 monthly, or '26509' for May 9 2026 weekly).
        """
        if "INDEX" not in index_symbol:
            return index_symbol # It's already a tradable equity/futures
            
        atm_strike = OptionsMapper.get_atm_strike(index_symbol, current_price)
        
        # Determine option type
        opt_type = "CE" if "CALL" in signal_type else "PE"
        
        # Base symbol extraction (NSE:NIFTY50-INDEX -> NIFTY)
        base = "NIFTY"
        if "BANK" in index_symbol:
            base = "BANKNIFTY"
        
        # If expiry is not provided, default to a placeholder (User MUST configure this in live)
        if not expiry_str:
            # Fallback to current year and month for a generic monthly expiry
            now = datetime.datetime.now()
            year_short = str(now.year)[-2:]
            month_str = now.strftime("%b").upper()
            expiry_str = f"{year_short}{month_str}"
            
        option_symbol = f"NSE:{base}{expiry_str}{atm_strike}{opt_type}"
        return option_symbol

    @staticmethod
    def get_lot_size(symbol: str) -> int:
        if "BANKNIFTY" in symbol or "NIFTYBANK" in symbol:
            return 15
        elif "NIFTY" in symbol:
            return 25
        return 1
