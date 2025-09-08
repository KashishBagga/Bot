"""
Enhanced WebSocket management with automatic reconnection.
"""

import asyncio
import websockets
import json
import logging
import time
import threading
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import queue

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

class WebSocketManager:
    """Enhanced WebSocket manager with automatic reconnection."""
    
    def __init__(self, url: str, 
                 reconnect_interval: float = 5.0,
                 max_reconnect_attempts: int = 10,
                 heartbeat_interval: float = 30.0,
                 message_timeout: float = 10.0):
        
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.heartbeat_interval = heartbeat_interval
        self.message_timeout = message_timeout
        
        self.state = ConnectionState.DISCONNECTED
        self.websocket = None
        self.reconnect_attempts = 0
        self.last_heartbeat = 0
        self.message_handlers = {}
        self.subscriptions = set()
        
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._reconnect_thread = None
        self._heartbeat_thread = None
        self._message_queue = queue.Queue()
        
        logger.info(f"WebSocket manager initialized for {url}")
    
    def add_message_handler(self, message_type: str, handler: Callable):
        """Add a message handler for specific message types."""
        with self._lock:
            if message_type not in self.message_handlers:
                self.message_handlers[message_type] = []
            self.message_handlers[message_type].append(handler)
    
    def remove_message_handler(self, message_type: str, handler: Callable):
        """Remove a message handler."""
        with self._lock:
            if message_type in self.message_handlers:
                try:
                    self.message_handlers[message_type].remove(handler)
                except ValueError:
                    pass
    
    async def connect(self):
        """Connect to the WebSocket."""
        with self._lock:
            if self.state == ConnectionState.CONNECTED:
                return True
            
            self.state = ConnectionState.CONNECTING
        
        try:
            logger.info(f"Connecting to WebSocket: {self.url}")
            self.websocket = await websockets.connect(
                self.url,
                ping_interval=self.heartbeat_interval,
                ping_timeout=self.message_timeout,
                close_timeout=self.message_timeout
            )
            
            with self._lock:
                self.state = ConnectionState.CONNECTED
                self.reconnect_attempts = 0
                self.last_heartbeat = time.time()
            
            logger.info("✅ WebSocket connected successfully")
            
            # Start heartbeat thread
            self._start_heartbeat()
            
            # Resubscribe to previous subscriptions
            await self._resubscribe()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            with self._lock:
                self.state = ConnectionState.FAILED
            
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket."""
        with self._lock:
            self.state = ConnectionState.DISCONNECTED
            self._stop_event.set()
        
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None
        
        # Stop background threads
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5.0)
        
        logger.info("WebSocket disconnected")
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send a message through the WebSocket."""
        if self.state != ConnectionState.CONNECTED or not self.websocket:
            logger.warning("Cannot send message: WebSocket not connected")
            return False
        
        try:
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            await self._handle_connection_error()
            return False
    
    async def subscribe(self, subscription: Dict[str, Any]):
        """Subscribe to a data stream."""
        if await self.send_message(subscription):
            with self._lock:
                self.subscriptions.add(json.dumps(subscription, sort_keys=True))
            logger.info(f"Subscribed to: {subscription}")
    
    async def unsubscribe(self, subscription: Dict[str, Any]):
        """Unsubscribe from a data stream."""
        if await self.send_message(subscription):
            with self._lock:
                self.subscriptions.discard(json.dumps(subscription, sort_keys=True))
            logger.info(f"Unsubscribed from: {subscription}")
    
    async def listen(self):
        """Listen for incoming messages."""
        if not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                if self._stop_event.is_set():
                    break
                
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            await self._handle_connection_error()
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
            await self._handle_connection_error()
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming messages."""
        message_type = data.get('type', 'unknown')
        
        # Update heartbeat
        with self._lock:
            self.last_heartbeat = time.time()
        
        # Call registered handlers
        if message_type in self.message_handlers:
            for handler in self.message_handlers[message_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
    
    async def _handle_connection_error(self):
        """Handle connection errors and initiate reconnection."""
        with self._lock:
            if self.state == ConnectionState.CONNECTED:
                self.state = ConnectionState.RECONNECTING
        
        logger.warning("WebSocket connection error, initiating reconnection...")
        
        # Start reconnection in background
        if not self._reconnect_thread or not self._reconnect_thread.is_alive():
            self._reconnect_thread = threading.Thread(target=self._reconnect_loop)
            self._reconnect_thread.daemon = True
            self._reconnect_thread.start()
    
    def _reconnect_loop(self):
        """Background reconnection loop."""
        while not self._stop_event.is_set():
            with self._lock:
                if self.state != ConnectionState.RECONNECTING:
                    break
                
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error("Max reconnection attempts reached")
                    self.state = ConnectionState.FAILED
                    break
            
            time.sleep(self.reconnect_interval)
            
            with self._lock:
                self.reconnect_attempts += 1
            
            logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
            
            # Try to reconnect
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                success = loop.run_until_complete(self.connect())
                if success:
                    # Start listening again
                    loop.run_until_complete(self.listen())
                    break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
            finally:
                loop.close()
    
    def _start_heartbeat(self):
        """Start heartbeat monitoring thread."""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self._heartbeat_thread.daemon = True
        self._heartbeat_thread.start()
    
    def _heartbeat_loop(self):
        """Background heartbeat monitoring loop."""
        while not self._stop_event.is_set():
            time.sleep(self.heartbeat_interval)
            
            with self._lock:
                if self.state != ConnectionState.CONNECTED:
                    continue
                
                time_since_heartbeat = time.time() - self.last_heartbeat
                if time_since_heartbeat > self.heartbeat_interval * 2:
                    logger.warning("Heartbeat timeout, connection may be dead")
                    asyncio.create_task(self._handle_connection_error())
    
    async def _resubscribe(self):
        """Resubscribe to previous subscriptions after reconnection."""
        with self._lock:
            subscriptions = list(self.subscriptions)
        
        for subscription_str in subscriptions:
            try:
                subscription = json.loads(subscription_str)
                await self.send_message(subscription)
            except Exception as e:
                logger.error(f"Failed to resubscribe: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get WebSocket connection status."""
        with self._lock:
            return {
                'state': self.state.value,
                'url': self.url,
                'reconnect_attempts': self.reconnect_attempts,
                'last_heartbeat': self.last_heartbeat,
                'subscriptions': len(self.subscriptions),
                'message_handlers': len(self.message_handlers)
            }
