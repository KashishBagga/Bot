#!/usr/bin/env python3
"""
Fyers WebSocket Data Manager
============================

Real-time data streaming using Fyers WebSocket API for live trading.
Provides low-latency, efficient data feeds for fast signal generation.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from zoneinfo import ZoneInfo
import websocket
from fyers_apiv3.FyersWebsocket import FyersWebsocket

from src.config.settings import setup_logging

logger = setup_logging('fyers_websocket')

class FyersWebSocketManager:
    """Real-time WebSocket data manager for Fyers API"""
    
    def __init__(self, access_token: str, client_id: str):
        """Initialize WebSocket manager"""
        self.access_token = access_token
        self.client_id = client_id
        self.ws = None
        self.is_connected = False
        self.is_running = False
        
        # Data storage
        self.live_prices = {}
        self.live_candles = {}
        self.option_chains = {}
        
        # Callbacks
        self.price_callbacks = []
        self.candle_callbacks = []
        self.option_callbacks = []
        
        # Threading
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._ws_thread = None
        
        # Timezone
        self.tz = ZoneInfo("Asia/Kolkata")
        
        logger.info("🔌 Fyers WebSocket Manager initialized")
    
    def add_price_callback(self, callback: Callable):
        """Add callback for price updates"""
        self.price_callbacks.append(callback)
    
    def add_candle_callback(self, callback: Callable):
        """Add callback for candle updates"""
        self.candle_callbacks.append(callback)
    
    def add_option_callback(self, callback: Callable):
        """Add callback for option chain updates"""
        self.option_callbacks.append(callback)
    
    def connect(self, symbols: List[str]):
        """Connect to Fyers WebSocket and subscribe to symbols"""
        try:
            # Initialize WebSocket
            self.ws = FyersWebsocket(
                access_token=f"{self.client_id}:{self.access_token}",
                log_path="",
                litemode=True,
                write_to_file=False,
                on_connect=self._on_connect,
                on_close=self._on_close,
                on_error=self._on_error,
                on_message=self._on_message
            )
            
            # Start WebSocket connection
            self.ws.connect()
            self.is_running = True
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not self.is_connected:
                raise Exception("WebSocket connection timeout")
            
            # Subscribe to symbols
            self._subscribe_symbols(symbols)
            
            logger.info(f"✅ WebSocket connected and subscribed to {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            return False
    
    def _subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbol data"""
        try:
            # Subscribe to quotes (LTP)
            quote_symbols = [{"symbol": symbol} for symbol in symbols]
            self.ws.subscribe_quotes(quote_symbols)
            
            # Subscribe to candles (1-minute)
            candle_symbols = [{"symbol": symbol, "resolution": "1"} for symbol in symbols]
            self.ws.subscribe_candles(candle_symbols)
            
            logger.info(f"📡 Subscribed to {len(symbols)} symbols for real-time data")
            
        except Exception as e:
            logger.error(f"❌ Symbol subscription failed: {e}")
    
    def _on_connect(self):
        """WebSocket connection callback"""
        self.is_connected = True
        logger.info("🔌 WebSocket connected successfully")
    
    def _on_close(self):
        """WebSocket close callback"""
        self.is_connected = False
        logger.warning("🔌 WebSocket connection closed")
        
        # Attempt reconnection if still running
        if self.is_running and not self._stop_event.is_set():
            logger.info("🔄 Attempting WebSocket reconnection...")
            threading.Timer(5.0, self._reconnect).start()
    
    def _on_error(self, error):
        """WebSocket error callback"""
        logger.error(f"❌ WebSocket error: {error}")
    
    def _on_message(self, message):
        """WebSocket message callback"""
        try:
            data = json.loads(message)
            
            if 'd' in data:
                for item in data['d']:
                    if 'symbol' in item and 'ltp' in item:
                        # Price update
                        symbol = item['symbol']
                        price = float(item['ltp'])
                        timestamp = datetime.now(self.tz)
                        
                        with self._lock:
                            self.live_prices[symbol] = {
                                'price': price,
                                'timestamp': timestamp
                            }
                        
                        # Trigger callbacks
                        for callback in self.price_callbacks:
                            try:
                                callback(symbol, price, timestamp)
                            except Exception as e:
                                logger.error(f"Price callback error: {e}")
                    
                    elif 'symbol' in item and 'candles' in item:
                        # Candle update
                        symbol = item['symbol']
                        candles = item['candles']
                        
                        with self._lock:
                            self.live_candles[symbol] = candles
                        
                        # Trigger callbacks
                        for callback in self.candle_callbacks:
                            try:
                                callback(symbol, candles)
                            except Exception as e:
                                logger.error(f"Candle callback error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error processing WebSocket message: {e}")
    
    def _reconnect(self):
        """Attempt to reconnect WebSocket"""
        if self._stop_event.is_set():
            return
        
        try:
            logger.info("🔄 Reconnecting WebSocket...")
            self.ws.connect()
        except Exception as e:
            logger.error(f"❌ WebSocket reconnection failed: {e}")
            # Retry after 30 seconds
            threading.Timer(30.0, self._reconnect).start()
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get latest live price for symbol"""
        with self._lock:
            if symbol in self.live_prices:
                return self.live_prices[symbol]['price']
        return None
    
    def get_live_candles(self, symbol: str) -> Optional[List]:
        """Get latest candles for symbol"""
        with self._lock:
            return self.live_candles.get(symbol)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all live prices"""
        with self._lock:
            return {symbol: data['price'] for symbol, data in self.live_prices.items()}
    
    def disconnect(self):
        """Disconnect WebSocket"""
        self.is_running = False
        self._stop_event.set()
        
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        self.is_connected = False
        logger.info("🔌 WebSocket disconnected")
    
    def is_healthy(self) -> bool:
        """Check if WebSocket is healthy"""
        return self.is_connected and self.is_running and not self._stop_event.is_set() 