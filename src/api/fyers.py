"""
Fyers API client module.
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
from typing import Optional, List, Dict

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
        
        # Load access token from environment
        import os
        self.access_token = os.getenv("FYERS_ACCESS_TOKEN")
        if self.access_token:
            logger.info("🔑 Access token loaded from environment")
        else:
            logger.warning("⚠️ No access token found in environment")        # Rate limiting
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
    
    def _rate_limit(self):
        """Implement rate limiting to prevent 429 errors."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        # Increased minimum interval to reduce API pressure
        min_interval = max(self.min_call_interval, 0.5)  # At least 500ms between calls
        
        if time_since_last_call < min_interval:
            sleep_time = min_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    def generate_auth_url(self):
        """Generate the authorization URL for authentication.
        
        Returns:
            str: Authorization URL
        """
        return self.session.generate_authcode()
    
    def open_auth_url(self):
        """Open the authorization URL in the default web browser."""
        auth_url = self.generate_auth_url()
        logger.info(f"Opening auth URL: {auth_url}")
        webbrowser.open(auth_url, new=1)
    
    def set_auth_code(self, auth_code):
        """Set the authentication code received after user authorization.
        
        Args:
            auth_code: Authorization code from the redirect URL
        """
        self.auth_code = auth_code
        self.session.set_token(auth_code)
    
    def generate_access_token(self):
        """Generate and set the access token using the authentication code.
        
        Returns:
            bool: True if token generation was successful, False otherwise
        """
        try:
            self.session.set_token(self.auth_code)
            response = self.session.generate_token()
            
            if 'access_token' in response:
                self.access_token = response['access_token']
                logger.info("Access token generated successfully")
                return True
            else:
                logger.error(f"Failed to generate access token: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error generating access token: {e}")
            return False
    

    def initialize_client(self):
        """Initialize the Fyers client with the access token.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if not self.access_token:
            logger.warning("No access token available for initialization")
            return False
        
        try:
            self.fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                token=self.access_token,
                log_path="logs/"
            )
            
            # Test the connection by getting the profile
            profile = self.fyers.get_profile()
            if 'code' in profile and profile['code'] == 200:
                logger.info("✅ Fyers client initialized successfully")
                return True
            else:
                logger.error(f"❌ Failed to initialize Fyers client: {profile}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error initializing Fyers client: {e}")
            return False

    def get_profile(self):
        """Get the user profile.
        
        Returns:
            dict: User profile data
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None
        
        # Apply rate limiting
        self._rate_limit()
        
        try:
            response = self.fyers.get_profile()
            return response
        except Exception as e:
            logger.error(f"Error fetching profile: {e}")
    def get_quotes(self, symbols: List[str]) -> Optional[Dict]:
        """Get quotes for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            dict: Quotes data or None if error
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None
        
        # Apply rate limiting
        self._rate_limit()
        
        try:
            # Format symbols for Fyers API
            formatted_symbols = []
            for symbol in symbols:
                # Convert NSE:SYMBOL-EQ to NSE:SYMBOL
                if "-EQ" in symbol:
                    formatted_symbol = symbol.replace("-EQ", "")
                elif "-INDEX" in symbol:
                    formatted_symbol = symbol
                else:
                    formatted_symbol = symbol
                formatted_symbols.append(formatted_symbol)
            
            data = {"symbols": ",".join(formatted_symbols)}
            response = self.fyers.quotes(data)
            
            if response and response.get("code") == 200:
                return response.get("d", {})
            else:
                logger.warning(f"No quotes data available for {symbols}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching quotes for {symbols}: {e}")
            return None


    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> Optional[Dict]:
        """Get historical data for a symbol."""
        try:
            if not self.fyers:
                logger.error("❌ Fyers client not initialized")
                return None
            
            # Convert dates to YYYY-MM-DD format
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            # Map interval to Fyers format
            interval_map = {
                "1m": "1",
                "5m": "5", 
                "15m": "15",
                "30m": "30",
                "1h": "60",
                "1d": "D"
            }
            
            fyers_interval = interval_map.get(interval, "60")
            
            # Make API call
            data = {
                "symbol": symbol,
                "resolution": fyers_interval,
                "date_format": "1",
                "range_from": start_date_str,
                "range_to": end_date_str,
                "cont_flag": "1"
            }
            
            # Try the correct Fyers API method
            try:
                response = self.fyers.history(data)
                
                if response and response.get("s") == "ok":
                    return response
                else:
                    logger.error(f"❌ Historical data request failed: {response}")
                    return None
            except Exception as api_error:
                logger.error(f"❌ Fyers API error: {api_error}")
                # Try alternative method
                try:
                    # Alternative: use the quotes method with historical data
                    alt_response = self.fyers.quotes(data)
                    return alt_response
                except Exception as alt_error:
                    logger.error(f"❌ Alternative method also failed: {alt_error}")
                    return None
                
        except Exception as e:
            logger.error(f"❌ Error getting historical data: {e}")
            return None


# Create an instance for direct imports
fyers_client = FyersClient()
