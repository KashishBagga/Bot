"""
Fyers API client module with fixed methods.
Handles authentication and API requests to the Fyers trading platform.
"""
import webbrowser
import logging
import time
from fyers_apiv3 import fyersModel
from src.config.settings import (
    FYERS_CLIENT_ID, 
    FYERS_SECRET_KEY, 
    FYERS_REDIRECT_URI,
    FYERS_RESPONSE_TYPE,
    FYERS_GRANT_TYPE,
    FYERS_STATE,
    FYERS_AUTH_CODE,
    setup_logging
)
from datetime import datetime, timedelta
import re
from typing import Optional

# Set up logger
logger = logging.getLogger(__name__)

class FyersClient:
    """Fyers API client for authentication and trading."""
    
    def __init__(self):
        """Initialize the Fyers API client."""
        self.client_id = FYERS_CLIENT_ID
        self.secret_key = FYERS_SECRET_KEY
        self.redirect_uri = FYERS_REDIRECT_URI
        self.response_type = FYERS_RESPONSE_TYPE
        self.grant_type = FYERS_GRANT_TYPE
        self.state = FYERS_STATE
        self.auth_code = FYERS_AUTH_CODE
        self.access_token = None
        self.fyers = None
        
        # Rate limiting
        self.last_api_call = 0
        self.min_call_interval = 0.5  # Minimum 0.5 seconds between API calls to avoid rate limits
        
        # Initialize session model
        self.session = fyersModel.SessionModel(
            client_id=self.client_id,
            redirect_uri=self.redirect_uri,
            response_type=self.response_type,
            state=self.state,
            secret_key=self.secret_key,
            grant_type=self.grant_type
        )
        
        # Try to get access token from environment
        self._load_access_token()
        
        # Initialize Fyers model if we have access token
        if self.access_token:
            self._initialize_fyers()
    
    def _load_access_token(self):
        """Load access token from environment variables."""
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            access_token = os.getenv('FYERS_ACCESS_TOKEN')
            if access_token:
                self.access_token = access_token
                logger.info("🔑 Access token loaded from environment")
            else:
                logger.warning("⚠️ No access token found in environment")
        except Exception as e:
            logger.error(f"❌ Error loading access token: {e}")
    
    def _initialize_fyers(self):
        """Initialize Fyers model with access token."""
        try:
            self.fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                token=self.access_token,
                log_path="logs/"
            )
            logger.info("✅ Fyers model initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Fyers model: {e}")
    
    def _rate_limit(self):
        """Apply rate limiting to API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    def get_current_price(self, symbol: str):
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
    
    def get_historical_data(self, symbol: str, start_date, end_date, interval: str = "1h"):
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
    
    def get_quotes(self, symbols):
        """Get live quotes for symbols.
        
        Args:
            symbols: List of symbols (e.g., ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"])
            
        Returns:
            dict: Quotes data
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None
        
        # Apply rate limiting
        self._rate_limit()
        
        try:
            # Convert list to comma-separated string for Fyers API
            if isinstance(symbols, list):
                symbols_str = ','.join(symbols)
            else:
                symbols_str = symbols
                
            response = self.fyers.quotes({"symbols": symbols_str})
            logger.debug(f"Quotes fetched for {len(symbols) if isinstance(symbols, list) else 1} symbols")
            return response
        except Exception as e:
            logger.error(f"Error fetching quotes: {e}")
            return None
