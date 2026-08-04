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

# Patch Fyers SDK bug where network exceptions cause 'UnboundLocalError: local variable response referenced before assignment'
try:
    _orig_get_call = fyersModel.FyersServiceSync.get_call
    def _safe_get_call(self, api, header, data=None, data_flag=False):
        try:
            return _orig_get_call(self, api, header, data=data, data_flag=data_flag)
        except UnboundLocalError:
            if hasattr(self, 'api_logger') and self.api_logger:
                self.api_logger.error("Fyers SDK UnboundLocalError caught (network failure before response)")
            return {"code": -1, "message": "Network connection failure before HTTP response", "s": "error"}
    fyersModel.FyersServiceSync.get_call = _safe_get_call
except Exception:
    pass

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
        
        # Load access token from tokens directory or environment fallback
        import os
        import json
        from datetime import date
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        token_dir = os.path.join(project_root, "tokens")
        today_str = date.today().strftime('%Y-%m-%d')
        token_path = os.path.join(token_dir, f"token_{today_str}.json")
        
        if os.path.exists(token_path):
            try:
                with open(token_path, 'r') as f:
                    data = json.load(f)
                    self.access_token = data.get("access_token")
                if self.access_token:
                    logger.info(f"🔑 Loaded access token from local JSON cache: {token_path}")
            except Exception as e:
                logger.error(f"❌ Failed to read token cache: {e}")
                
        if not self.access_token:
            self.access_token = os.getenv("FYERS_ACCESS_TOKEN")
            if self.access_token:
                logger.info("🔑 Access token loaded from environment fallback")
            else:
                logger.warning("⚠️ No access token found in JSON cache or environment")        # Rate limiting
        self.last_api_call = 0
        self.min_call_interval = 1.0  # Minimum 1.0 seconds between API calls to avoid 429 rate limits
        
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
        
        min_interval = max(self.min_call_interval, 1.0)  # At least 1s between calls
        
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
                
                # Cache token in JSON file
                try:
                    import os
                    import json
                    from datetime import date
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    token_dir = os.path.join(project_root, "tokens")
                    os.makedirs(token_dir, exist_ok=True)
                    today_str = date.today().strftime('%Y-%m-%d')
                    token_path = os.path.join(token_dir, f"token_{today_str}.json")
                    with open(token_path, 'w') as f:
                        json.dump({"access_token": self.access_token, "date": today_str}, f)
                    logger.info(f"💾 Saved access token to local JSON cache: {token_path}")
                except Exception as ex:
                    logger.error(f"❌ Failed to save token cache: {ex}")
                    
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
    def get_quotes(self, symbols: List[str], _retries: int = 3) -> Optional[Dict]:
        """Get quotes for multiple symbols with automatic 429 retry.
        
        Args:
            symbols: List of trading symbols
            _retries: Internal retry count for 429 backoff (do not set manually)
            
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
            
            # ── 429 Rate Limit: back off and retry ───────────────────────
            resp_code = response.get("code") if response else None
            if resp_code == 429 and _retries > 0:
                backoff = 2 ** (3 - _retries)  # 1s, 2s, 4s
                logger.warning(
                    f"⚠️ Rate limited (429) for {symbols}, backing off {backoff}s "
                    f"({_retries} retries left)"
                )
                time.sleep(backoff)
                return self.get_quotes(symbols, _retries=_retries - 1)
            
            # Non-retryable failure — log the actual response for diagnosis
            logger.warning(
                f"No quotes data available for {symbols} "
                f"(code={resp_code}, message={response.get('message', 'N/A') if response else 'no response'})"
            )
            return None
                
        except Exception as e:
            logger.error(f"Error fetching quotes for {symbols}: {e}")
            return None


    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str, _retries: int = 3) -> Optional[Dict]:
        """Get historical data for a symbol with rate limiting and automatic retry."""
        try:
            if not self.fyers:
                logger.error("❌ Fyers client not initialized")
                return None
            
            self._rate_limit()

            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            interval_map = {
                "1": "1", "1m": "1", "5": "5", "5m": "5",
                "15": "15", "15m": "15", "30": "30", "30m": "30",
                "60": "60", "1h": "60", "D": "D", "1d": "D"
            }
            fyers_interval = interval_map.get(interval, interval)
            
            data = {
                "symbol": symbol,
                "resolution": fyers_interval,
                "date_format": "1",
                "range_from": start_date_str,
                "range_to": end_date_str,
                "cont_flag": "1"
            }
            
            for attempt in range(1, _retries + 1):
                try:
                    response = self.fyers.history(data)
                    if response and isinstance(response, dict) and response.get("s") == "ok":
                        return response
                    
                    resp_code = response.get("code") if isinstance(response, dict) else None
                    if resp_code == 429 and attempt < _retries:
                        backoff = 2 ** attempt
                        logger.warning(f"⚠️ Rate limited (429) on historical data for {symbol}, backing off {backoff}s...")
                        time.sleep(backoff)
                        continue
                    
                    logger.error(f"❌ Historical data request failed for {symbol} (attempt {attempt}/{_retries}): {response}")
                    if attempt < _retries:
                        time.sleep(1)
                        continue
                    return None
                except Exception as api_error:
                    logger.error(f"❌ Fyers API history error for {symbol} (attempt {attempt}/{_retries}): {api_error}")
                    if attempt < _retries:
                        time.sleep(1)
                        continue
                    return None
            return None
        except Exception as e:
            logger.error(f"❌ Error getting historical data: {e}")
            return None


    def get_market_depth(self, symbol: str) -> Optional[Dict]:
        """Get full market depth for a symbol, including real OI data.

        Uses the Fyers `depth` endpoint which returns:
            oi, pdoi (prev-day OI), oipercent, ltp, volume, bid/ask ladders.

        Args:
            symbol: Fyers-format symbol e.g. "NSE:NIFTY2680424600CE"

        Returns:
            Dict with keys: ltp, volume, oi, pdoi, oi_change_pct, bid, ask, or None on error.
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None

        self._rate_limit()

        try:
            response = self.fyers.depth({"symbol": symbol, "ohlcv_flag": 1})
            if not response or response.get("s") != "ok":
                logger.warning(f"Depth request failed for {symbol}: {response}")
                return None

            data = response.get("d", {}).get(symbol, {})
            if not data:
                return None

            best_bid = data["bids"][0]["price"] if data.get("bids") else 0.0
            best_ask = data["ask"][0]["price"] if data.get("ask") else 0.0
            oi = data.get("oi", 0)
            pdoi = data.get("pdoi", 0)
            oi_change = oi - pdoi

            return {
                "ltp": float(data.get("ltp", 0.0)),
                "volume": int(data.get("v", 0)),
                "oi": int(oi),
                "pdoi": int(pdoi),
                "oi_change": int(oi_change),
                "oi_change_pct": float(data.get("oipercent", 0.0)),
                "bid": float(best_bid),
                "ask": float(best_ask),
                "high": float(data.get("h", 0.0)),
                "low": float(data.get("l", 0.0)),
                "prev_close": float(data.get("c", 0.0)),
            }

        except Exception as e:
            logger.error(f"Error fetching market depth for {symbol}: {e}")
            return None


# Create an instance for direct imports
fyers_client = FyersClient()
