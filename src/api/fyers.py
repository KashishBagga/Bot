"""
Fyers API client module.
Handles authentication and API requests to the Fyers trading platform.
"""
import webbrowser
import logging
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

# Set up logger
logger = setup_logging('fyers_api')

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
        
        # Initialize session model
        self.session = fyersModel.SessionModel(
            client_id=self.client_id,
            redirect_uri=self.redirect_uri,
            response_type=self.response_type,
            state=self.state,
            secret_key=self.secret_key,
            grant_type=self.grant_type
        )
    
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
            success = self.generate_access_token()
            if not success:
                return False
        
        try:
            self.fyers = fyersModel.FyersModel(
                token=self.access_token,
                is_async=False,
                client_id=self.client_id,
                log_path=""
            )
            
            # Test the connection by getting the profile
            profile = self.fyers.get_profile()
            if 'code' in profile and profile['code'] == 200:
                logger.info("Fyers client initialized successfully")
                return True
            else:
                logger.error(f"Failed to initialize Fyers client: {profile}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Fyers client: {e}")
            return False
    
    def get_profile(self):
        """Get the user profile.
        
        Returns:
            dict: User profile data
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None
            
        return self.fyers.get_profile()
    
    def get_funds(self):
        """Get the user's fund details.
        
        Returns:
            dict: Fund details
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None
            
        return self.fyers.funds()
    
    def get_positions(self):
        """Get the user's current positions.
        
        Returns:
            dict: Position details
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None
            
        return self.fyers.positions()
    
    def place_order(self, symbol, transaction_type, quantity, product_type="INTRADAY", 
                   limit_price=0, stop_price=0, order_type="MARKET"):
        """Place an order.
        
        Args:
            symbol: Trading symbol (e.g., "NSE:RELIANCE-EQ")
            transaction_type: 1 for Buy, -1 for Sell
            quantity: Number of shares to trade
            product_type: INTRADAY, CNC, MARGIN
            limit_price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            order_type: MARKET, LIMIT, STOPLIMIT, STOPMARKET
            
        Returns:
            dict: Order response
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None
        
        data = {
            "symbol": symbol,
            "qty": quantity,
            "type": order_type,
            "side": transaction_type,
            "productType": product_type,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": "False",
        }
        
        try:
            response = self.fyers.place_order(data)
            logger.info(f"Order placed: {response}")
            return response
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
            
    def get_order_status(self, order_id):
        """Get the status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            dict: Order status
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None
            
        try:
            response = self.fyers.orderbook()
            if response and 'orderBook' in response:
                for order in response['orderBook']:
                    if order['id'] == order_id:
                        return order
            return None
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None

    def get_historical_data(self, symbol, resolution, date_format=1, range_from=None, range_to=None, cont_flag=1):
        """Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "NSE:NIFTY50-INDEX")
            resolution: Timeframe (1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 120, 180, 240, 1D)
            date_format: Date format (1 for timestamp)
            range_from: Start date (YYYY-MM-DD)
            range_to: End date (YYYY-MM-DD)
            cont_flag: Continuous flag
            
        Returns:
            dict: Historical data
        """
        if not self.fyers:
            logger.error("Fyers client not initialized")
            return None
        
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": date_format,
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": cont_flag
        }
        
        try:
            response = self.fyers.history(data)
            logger.info(f"Historical data fetched for {symbol}")
            return response
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

# Create an instance for direct imports
fyers_client = FyersClient() 