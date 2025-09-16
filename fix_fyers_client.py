#!/usr/bin/env python3
"""
Fix FyersClient by adding missing methods
"""

# Read the file
with open('src/api/fyers.py', 'r') as f:
    content = f.read()

# Add get_current_price method after get_quotes method
get_quotes_end = content.find('def get_option_chain(self, symbol: str, expiry_date: str = None):')

if get_quotes_end != -1:
    # Insert the new method
    new_method = '''
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a single symbol.
        
        Args:
            symbol: Trading symbol (e.g., "NSE:NIFTY50-INDEX")
            
        Returns:
            float: Current price or None if error
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None
        
        # Apply rate limiting
        self._rate_limit()
        
        try:
            response = self.fyers.quotes({"symbols": symbol})
            
            if response and response.get("code") == 200:
                data = response.get("data", {})
                if symbol in data:
                    symbol_data = data[symbol]
                    # Try different price fields
                    price = (symbol_data.get("v") or  # Last traded price
                            symbol_data.get("lp") or  # Last price
                            symbol_data.get("c") or   # Close price
                            symbol_data.get("o"))     # Open price
                    
                    if price:
                        return float(price)
            
            logger.warning(f"No price data available for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                           interval: str = "1h") -> Optional[dict]:
        """Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
            
        Returns:
            dict: Historical data or None if error
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None
        
        # Apply rate limiting
        self._rate_limit()
        
        try:
            # Format dates for Fyers API
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            data = {
                "symbol": symbol,
                "resolution": interval,
                "date_format": "1",
                "range_from": start_str,
                "range_to": end_str,
                "cont_flag": "1"
            }
            
            response = self.fyers.history(data)
            
            if response and response.get("code") == 200:
                return response.get("data", {})
            else:
                logger.warning(f"No historical data available for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

'''
    
    # Insert the new method
    content = content[:get_quotes_end] + new_method + content[get_quotes_end:]
    
    # Write back to file
    with open('src/api/fyers.py', 'w') as f:
        f.write(content)
    
    print("✅ Added get_current_price and get_historical_data methods to FyersClient")
else:
    print("❌ Could not find insertion point for new methods")
