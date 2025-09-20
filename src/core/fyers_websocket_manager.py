import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if demo mode is enabled
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
"""
Fyers WebSocket Manager for Real-Time Market Data
================================================
Implements real-time market data streaming using Fyers WebSocket API v3
with enhanced error handling, automatic reconnection, and health monitoring
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import json

from fyers_apiv3.FyersWebsocket import data_ws
from src.config.settings import FYERS_CLIENT_ID
import os

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Real-time market data structure."""
    symbol: str
    ltp: float
    volume: int
    timestamp: datetime
    change: float = 0.0
    change_percent: float = 0.0

class FyersWebSocketManager:
    """Manages Fyers WebSocket connection for real-time market data."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.access_token = self._get_access_token()
        self.data_callbacks: List[Callable] = []
        self.live_data: Dict[str, MarketData] = {}
        self.is_connected = False
        self.is_running = False
        self._lock = threading.Lock()
        
        # Enhanced error handling and reconnection
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.reconnect_delay = 5  # seconds
        self.last_connection_time = None
        self.last_data_time = None
        self.health_check_interval = 30  # seconds
        self.health_check_thread = None
        
        # Initialize WebSocket
        self.fyers_socket = None
        self._initialize_socket()
        
        logger.info(f"🔄 Fyers WebSocket Manager initialized for {len(symbols)} symbols")
    
    def _get_access_token(self) -> str:
        """Get access token from environment."""
        try:
            # Try to get from FyersClient first
            from src.api.fyers import FyersClient
            client = FyersClient()
            if client.access_token:
                logger.info("🔑 Access token loaded from FyersClient")
                return client.access_token
            
            # Fallback to environment variable
            access_token = os.getenv('FYERS_ACCESS_TOKEN')
            if access_token:
                logger.info("🔑 Access token loaded from environment")
                return access_token
            
            logger.error("❌ No access token available")
            raise ValueError("No access token available")
            
        except Exception as e:
            logger.error(f"❌ Failed to get access token: {e}")
            raise
    
    def _initialize_socket(self):
        """Initialize the Fyers WebSocket socket."""
        try:
            self.fyers_socket = data_ws.FyersDataSocket(
                access_token=self.access_token,
                log_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs/websocket/"),
                
            )
            
            # Set up callbacks
            self.fyers_socket.websocket_data = self._on_message
            self.fyers_socket.on_connect = self._on_connect
            self.fyers_socket.on_error = self._on_error
            self.fyers_socket.on_close = self._on_close
            
            logger.info("✅ Fyers WebSocket socket initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Fyers WebSocket: {e}")
            raise
    
    def _on_connect(self):
        """Callback when WebSocket connects."""
        logger.info("🔗 Fyers WebSocket connected")
        self.is_connected = True
        self.connection_attempts = 0
        self.last_connection_time = datetime.now()
        
        # Subscribe to symbols
        try:
            self.fyers_socket.subscribe(
                symbols=self.symbols,
                data_type="SymbolUpdate"
            )
            logger.info(f"📡 Subscribed to {len(self.symbols)} symbols")
            
            # Keep the socket running
            self.fyers_socket.keep_running()
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to symbols: {e}")
            self._handle_connection_error()
    
    def _on_message(self, message):
        """Callback when WebSocket receives a message."""
        try:
            if isinstance(message, dict) and 'symbol' in message:
                symbol = message['symbol']
                ltp = float(message.get('ltp', 0))
                volume = int(message.get('volume', 0))
                timestamp = datetime.now()
                
                # Calculate change if we have previous data
                change = 0.0
                change_percent = 0.0
                if symbol in self.live_data:
                    prev_ltp = self.live_data[symbol].ltp
                    if prev_ltp > 0:
                        change = ltp - prev_ltp
                        change_percent = (change / prev_ltp) * 100
                
                # Create market data object
                market_data = MarketData(
                    symbol=symbol,
                    ltp=ltp,
                    volume=volume,
                    timestamp=timestamp,
                    change=change,
                    change_percent=change_percent
                )
                
                # Thread-safe update
                with self._lock:
                    self.live_data[symbol] = market_data
                    self.last_data_time = timestamp
                
                # Notify callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(market_data)
                    except Exception as e:
                        logger.error(f"❌ Error in data callback: {e}")
                
                logger.debug(f"📊 {symbol}: {ltp} ({change:+.2f}, {change_percent:+.2f}%)")
                
        except Exception as e:
            logger.error(f"❌ Error processing WebSocket message: {e}")
    
    def _on_error(self, error):
        """Callback when WebSocket encounters an error."""
        logger.error(f"❌ Fyers WebSocket error: {error}")
        self.is_connected = False
        self._handle_connection_error()
    
    def _on_close(self):
        """Callback when WebSocket connection closes."""
        logger.warning("🔌 Fyers WebSocket connection closed")
        self.is_connected = False
        
        # Attempt reconnection if we're still running
        if self.is_running:
            logger.info("🔄 Attempting to reconnect...")
            self._attempt_reconnection()
    
    def _handle_connection_error(self):
        """Handle connection errors with exponential backoff."""
        self.connection_attempts += 1
        
        if self.connection_attempts <= self.max_connection_attempts:
            delay = self.reconnect_delay * (2 ** (self.connection_attempts - 1))
            logger.warning(f"🔄 Connection attempt {self.connection_attempts}/{self.max_connection_attempts}, retrying in {delay}s...")
            
            def delayed_reconnect():
                time.sleep(delay)
                if self.is_running:
                    self._attempt_reconnection()
            
            threading.Thread(target=delayed_reconnect, daemon=True).start()
        else:
            logger.error(f"❌ Max connection attempts ({self.max_connection_attempts}) reached. Stopping reconnection.")
            self.is_running = False
    
    def _attempt_reconnection(self):
        """Attempt to reconnect the WebSocket."""
        try:
            logger.info("🔄 Attempting WebSocket reconnection...")
            self._initialize_socket()
            self.start()
        except Exception as e:
            logger.error(f"❌ Reconnection failed: {e}")
            self._handle_connection_error()
    
    def _start_health_check(self):
        """Start health check monitoring."""
        def health_check_loop():
            while self.is_running:
                time.sleep(self.health_check_interval)
                
                if not self.is_connected:
                    logger.warning("⚠️ WebSocket health check: Not connected")
                    continue
                
                # Check if we're receiving data
                if self.last_data_time:
                    time_since_data = (datetime.now() - self.last_data_time).total_seconds()
                    if time_since_data > 60:  # No data for 1 minute
                        logger.warning(f"⚠️ WebSocket health check: No data for {time_since_data:.0f}s")
                        self._handle_connection_error()
                else:
                    logger.warning("⚠️ WebSocket health check: No data received yet")
        
        self.health_check_thread = threading.Thread(target=health_check_loop, daemon=True)
        self.health_check_thread.start()
        logger.info("🏥 Health check monitoring started")
    
    def start(self):
        """Start the WebSocket connection."""
        if self.is_running:
            logger.warning("⚠️ WebSocket is already running")
            return
        
        self.is_running = True
        self.connection_attempts = 0
        
        try:
            # Start WebSocket in a separate thread
            def run_websocket():
                try:
                    self.fyers_socket.websocket_data = self._on_message
                    self.fyers_socket.on_connect = self._on_connect
                    self.fyers_socket.on_error = self._on_error
                    self.fyers_socket.on_close = self._on_close
                    self.fyers_socket.keep_running()
                except Exception as e:
                    logger.error(f"❌ WebSocket thread error: {e}")
                    self._handle_connection_error()
            
            websocket_thread = threading.Thread(target=run_websocket, daemon=True)
            websocket_thread.start()
            
            # Start health check
            self._start_health_check()
            
            logger.info("🚀 Fyers WebSocket started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket: {e}")
            self.is_running = False
            raise
    
    def stop(self):
        """Stop the WebSocket connection."""
        self.is_running = False
        if self.fyers_socket:
            try:
                # Fyers WebSocket auto-closes when is_running=False
                pass
            except Exception as e:
                logger.error(f"❌ Error closing WebSocket: {e}")
        logger.info("🛑 Fyers WebSocket stopped")
    
    def add_data_callback(self, callback: Callable):
        """Add a callback function for market data updates."""
        self.data_callbacks.append(callback)
        logger.info(f"📞 Added data callback: {callback.__name__}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        with self._lock:
            if symbol in self.live_data:
                return self.live_data[symbol].ltp
        return None
    
    def get_all_live_data(self) -> Dict[str, MarketData]:
        """Get all live market data."""
        with self._lock:
            return self.live_data.copy()
    
    def get_connection_status(self) -> Dict:
        """Get detailed connection status."""
        return {
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'connection_attempts': self.connection_attempts,
            'last_connection_time': self.last_connection_time,
            'last_data_time': self.last_data_time,
            'live_data_count': len(self.live_data),
            'symbols_subscribed': len(self.symbols)
        }

# Global WebSocket manager instance
_websocket_manager = None

def get_websocket_manager(symbols: List[str]) -> FyersWebSocketManager:
    """Get or create a global WebSocket manager instance."""
    global _websocket_manager
    
    if _websocket_manager is None:
        _websocket_manager = FyersWebSocketManager(symbols)
    
    return _websocket_manager

def stop_websocket():
    """Stop the global WebSocket manager."""
    global _websocket_manager
    
    if _websocket_manager:
        _websocket_manager.stop()
        _websocket_manager = None
