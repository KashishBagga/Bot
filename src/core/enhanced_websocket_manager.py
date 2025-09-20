#!/usr/bin/env python3
"""
Enhanced WebSocket Manager with Auto-Reconnection and Health Monitoring
"""

import os
import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import json
from fyers_apiv3.FyersWebsocket import data_ws

logger = logging.getLogger(__name__)

class EnhancedWebSocketManager:
    """Enhanced WebSocket manager with auto-reconnection and health monitoring."""
    
    def __init__(self, access_token: str, symbols: List[str], max_retries: int = 5):
        self.access_token = access_token
        self.symbols = symbols
        self.max_retries = max_retries
        self.retry_count = 0
        self.is_connected = False
        self.is_running = False
        self.fyers_socket = None
        self.data_callbacks = []
        self.last_heartbeat = datetime.now()
        self.reconnect_delay = 5  # seconds
        self.health_check_interval = 30  # seconds
        self.connection_lock = threading.Lock()
        self.health_thread = None
        
        # Connection statistics
        self.connection_stats = {
            'total_connections': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'messages_received': 0,
            'last_message_time': None,
            'uptime_start': None
        }
        
    def add_data_callback(self, callback: Callable[[Dict], None]):
        """Add a callback for data events."""
        self.data_callbacks.append(callback)
        logger.info(f"📞 Added data callback: {callback.__name__}")
    
    def start(self):
        """Start WebSocket connection with auto-reconnection."""
        logger.info("🚀 Starting Enhanced WebSocket Manager")
        self.is_running = True
        self._connect_with_retry()
        self._start_health_monitoring()
    
    def stop(self):
        """Stop WebSocket connection."""
        logger.info("�� Stopping Enhanced WebSocket Manager")
        self.is_running = False
        
        if self.health_thread:
            self.health_thread.join(timeout=5)
        
        with self.connection_lock:
            if self.fyers_socket:
                try:
                    self.fyers_socket.close_connection()
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
                finally:
                    self.fyers_socket = None
                    self.is_connected = False
    
    def _connect_with_retry(self):
        """Connect with retry logic."""
        while self.is_running and self.retry_count < self.max_retries:
            try:
                self._initialize_connection()
                if self.is_connected:
                    logger.info("✅ WebSocket connected successfully")
                    self.retry_count = 0
                    self.connection_stats['successful_connections'] += 1
                    self.connection_stats['uptime_start'] = datetime.now()
                    break
                else:
                    raise Exception("Connection failed")
                    
            except Exception as e:
                self.retry_count += 1
                self.connection_stats['failed_connections'] += 1
                logger.error(f"❌ WebSocket connection attempt {self.retry_count} failed: {e}")
                
                if self.retry_count < self.max_retries:
                    logger.info(f"⏳ Retrying in {self.reconnect_delay} seconds...")
                    time.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 2, 60)  # Exponential backoff
                else:
                    logger.error("❌ Max retry attempts reached. WebSocket connection failed.")
                    break
    
    def _initialize_connection(self):
        """Initialize WebSocket connection."""
        with self.connection_lock:
            self.connection_stats['total_connections'] += 1
            
            # Ensure log directory exists
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs/websocket/")
            os.makedirs(log_path, exist_ok=True)
            
            self.fyers_socket = data_ws.FyersDataSocket(
                access_token=self.access_token,
                log_path=log_path,
                litemode=False
            )
            
            # Set up callbacks
            self.fyers_socket.websocket_data = self._on_message
            self.fyers_socket.on_connect = self._on_connect
            self.fyers_socket.on_error = self._on_error
            self.fyers_socket.on_close = self._on_close
            
            # Start connection
            self.fyers_socket.connect()
            
            # Wait for connection
            connection_timeout = 10
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < connection_timeout:
                time.sleep(0.1)
            
            if self.is_connected:
                self._subscribe_to_symbols()
    
    def _subscribe_to_symbols(self):
        """Subscribe to market data for symbols."""
        try:
            if self.fyers_socket and self.is_connected:
                # Subscribe to symbols
                self.fyers_socket.subscribe(symbol=self.symbols, data_type="SymbolUpdate")
                logger.info(f"📡 Subscribed to {len(self.symbols)} symbols")
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
    
    def _on_connect(self, ws):
        """Handle WebSocket connection."""
        self.is_connected = True
        self.last_heartbeat = datetime.now()
        logger.info("🔗 WebSocket connected")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            self.last_heartbeat = datetime.now()
            self.connection_stats['messages_received'] += 1
            self.connection_stats['last_message_time'] = datetime.now()
            
            # Parse message
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message
            
            # Call all registered callbacks
            for callback in self.data_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in data callback {callback.__name__}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"❌ WebSocket error: {error}")
        self.is_connected = False
        
        # Attempt reconnection if still running
        if self.is_running:
            threading.Thread(target=self._reconnect, daemon=True).start()
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        self.is_connected = False
        logger.warning(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")
        
        # Attempt reconnection if still running
        if self.is_running:
            threading.Thread(target=self._reconnect, daemon=True).start()
    
    def _reconnect(self):
        """Reconnect WebSocket."""
        logger.info("🔄 Attempting WebSocket reconnection...")
        time.sleep(self.reconnect_delay)
        if self.is_running:
            self._connect_with_retry()
    
    def _start_health_monitoring(self):
        """Start health monitoring thread."""
        self.health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        self.health_thread.start()
        logger.info("🏥 Health monitoring started")
    
    def _health_monitor(self):
        """Monitor WebSocket health."""
        while self.is_running:
            try:
                time.sleep(self.health_check_interval)
                
                if not self.is_running:
                    break
                
                # Check connection health
                time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
                
                if time_since_heartbeat > 60:  # 1 minute without heartbeat
                    logger.warning("⚠️ WebSocket health check: No heartbeat for 1 minute")
                    if self.is_connected:
                        self.is_connected = False
                        threading.Thread(target=self._reconnect, daemon=True).start()
                
                elif not self.is_connected:
                    logger.warning("⚠️ WebSocket health check: Not connected")
                    threading.Thread(target=self._reconnect, daemon=True).start()
                
                else:
                    logger.debug("✅ WebSocket health check: OK")
                    
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        uptime = None
        if self.connection_stats['uptime_start']:
            uptime = (datetime.now() - self.connection_stats['uptime_start']).total_seconds()
        
        return {
            **self.connection_stats,
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'current_uptime_seconds': uptime,
            'retry_count': self.retry_count,
            'subscribed_symbols': len(self.symbols)
        }

# Global instance
_enhanced_websocket_manager = None

def get_enhanced_websocket_manager(access_token: str, symbols: List[str]) -> EnhancedWebSocketManager:
    """Get or create enhanced WebSocket manager instance."""
    global _enhanced_websocket_manager
    
    if _enhanced_websocket_manager is None:
        _enhanced_websocket_manager = EnhancedWebSocketManager(access_token, symbols)
    
    return _enhanced_websocket_manager
