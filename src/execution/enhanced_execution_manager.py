#!/usr/bin/env python3
"""
Enhanced Execution Manager with Reconciliation
Guaranteed order execution with retry logic and position reconciliation
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import queue
import json
from concurrent.futures import ThreadPoolExecutor
import requests

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class Order:
    """Order with comprehensive tracking"""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # BUY/SELL
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float
    average_price: float
    timestamp: datetime
    timeout_seconds: int
    retry_count: int
    max_retries: int
    broker_order_id: Optional[str]
    error_message: Optional[str]

@dataclass
class Position:
    """Position with reconciliation data"""
    symbol: str
    quantity: float
    average_price: float
    unrealized_pnl: float
    last_updated: datetime
    broker_position_id: Optional[str]

class EnhancedExecutionManager:
    """Enhanced execution manager with guaranteed reconciliation"""
    
    def __init__(self, broker_adapter, reconciliation_interval: int = 60):
        self.broker_adapter = broker_adapter
        self.reconciliation_interval = reconciliation_interval
        
        # Order management
        self.pending_orders = {}
        self.filled_orders = {}
        self.cancelled_orders = {}
        
        # Position tracking
        self.positions = {}
        self.broker_positions = {}
        
        # Threading
        self.order_queue = queue.Queue()
        self.reconciliation_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Monitoring
        self.order_timeout = 30  # 30 seconds default timeout
        self.max_retries = 3
        self.reconciliation_running = False
        
        # Statistics
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'timeout_orders': 0,
            'avg_fill_time': 0.0,
            'success_rate': 0.0
        }
        
        # Start background threads
        self._start_background_threads()
    
    def _start_background_threads(self):
        """Start background threads for order processing and reconciliation"""
        try:
            # Order processing thread
            self.order_thread = threading.Thread(target=self._order_processing_loop, daemon=True)
            self.order_thread.start()
            
            # Reconciliation thread
            self.reconciliation_thread = threading.Thread(target=self._reconciliation_loop, daemon=True)
            self.reconciliation_thread.start()
            
            # Order monitoring thread
            self.monitoring_thread = threading.Thread(target=self._order_monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("✅ Enhanced execution manager background threads started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start background threads: {e}")
    
    def place_order(self, symbol: str, order_type: OrderType, side: str, 
                   quantity: float, price: Optional[float] = None,
                   stop_price: Optional[float] = None, timeout_seconds: int = 30) -> str:
        """Place order with guaranteed execution tracking"""
        try:
            # Generate unique order ID
            order_id = f"{symbol}_{side}_{int(time.time())}_{hash(str(time.time()))}"
            
            # Create order
            order = Order(
                order_id=order_id,
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.PENDING,
                filled_quantity=0.0,
                average_price=0.0,
                timestamp=datetime.now(),
                timeout_seconds=timeout_seconds,
                retry_count=0,
                max_retries=self.max_retries,
                broker_order_id=None,
                error_message=None
            )
            
            # Add to pending orders
            self.pending_orders[order_id] = order
            
            # Queue for processing
            self.order_queue.put(order)
            
            logger.info(f"📋 Order queued: {order_id} - {side} {quantity} {symbol} @ {price}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"❌ Failed to place order: {e}")
            return None
    
    def _order_processing_loop(self):
        """Background order processing loop"""
        logger.info("🔄 Order processing loop started")
        
        while True:
            try:
                # Get order from queue
                order = self.order_queue.get(timeout=1.0)
                
                # Process order
                self._process_order(order)
                
                # Mark task as done
                self.order_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Order processing loop error: {e}")
                time.sleep(1)
    
    def _process_order(self, order: Order):
        """Process individual order with retry logic"""
        try:
            logger.info(f"🔄 Processing order: {order.order_id}")
            
            # Submit order to broker
            success, broker_order_id, error_message = self._submit_order_to_broker(order)
            
            if success:
                order.status = OrderStatus.SUBMITTED
                order.broker_order_id = broker_order_id
                logger.info(f"✅ Order submitted: {order.order_id} -> {broker_order_id}")
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = error_message
                logger.error(f"❌ Order rejected: {order.order_id} - {error_message}")
                
                # Move to cancelled orders
                self.cancelled_orders[order.order_id] = order
                del self.pending_orders[order.order_id]
                
                # Update statistics
                self.execution_stats['rejected_orders'] += 1
                
        except Exception as e:
            logger.error(f"❌ Order processing failed: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            
            # Move to cancelled orders
            self.cancelled_orders[order.order_id] = order
            del self.pending_orders[order.order_id]
    
    def _submit_order_to_broker(self, order: Order) -> Tuple[bool, Optional[str], Optional[str]]:
        """Submit order to broker with retry logic"""
        try:
            for attempt in range(order.max_retries):
                try:
                    # Submit to broker
                    broker_order_id = self.broker_adapter.place_order(
                        symbol=order.symbol,
                        order_type=order.order_type.value,
                        side=order.side,
                        quantity=order.quantity,
                        price=order.price,
                        stop_price=order.stop_price
                    )
                    
                    if broker_order_id:
                        return True, broker_order_id, None
                    else:
                        logger.warning(f"⚠️ Broker returned empty order ID for {order.order_id}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Broker submission attempt {attempt + 1} failed: {e}")
                    
                    if attempt < order.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        return False, None, str(e)
            
            return False, None, "Max retries exceeded"
            
        except Exception as e:
            logger.error(f"❌ Broker submission failed: {e}")
            return False, None, str(e)
    
    def _order_monitoring_loop(self):
        """Background order monitoring loop"""
        logger.info("🔄 Order monitoring loop started")
        
        while True:
            try:
                current_time = datetime.now()
                
                # Check for timeout orders
                timeout_orders = []
                for order_id, order in self.pending_orders.items():
                    if order.status == OrderStatus.SUBMITTED:
                        elapsed_time = (current_time - order.timestamp).total_seconds()
                        if elapsed_time > order.timeout_seconds:
                            timeout_orders.append(order_id)
                
                # Handle timeout orders
                for order_id in timeout_orders:
                    self._handle_order_timeout(order_id)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Order monitoring loop error: {e}")
                time.sleep(5)
    
    def _handle_order_timeout(self, order_id: str):
        """Handle order timeout"""
        try:
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
                
                # Cancel order
                success = self._cancel_order(order)
                
                if success:
                    order.status = OrderStatus.CANCELLED
                    order.error_message = "Order timeout"
                    logger.warning(f"⏰ Order timeout: {order_id}")
                else:
                    order.status = OrderStatus.TIMEOUT
                    order.error_message = "Timeout and cancellation failed"
                    logger.error(f"❌ Order timeout and cancellation failed: {order_id}")
                
                # Move to cancelled orders
                self.cancelled_orders[order_id] = order
                del self.pending_orders[order_id]
                
                # Update statistics
                self.execution_stats['timeout_orders'] += 1
                
        except Exception as e:
            logger.error(f"❌ Order timeout handling failed: {e}")
    
    def _cancel_order(self, order: Order) -> bool:
        """Cancel order with broker"""
        try:
            if order.broker_order_id:
                success = self.broker_adapter.cancel_order(order.broker_order_id)
                if success:
                    logger.info(f"✅ Order cancelled: {order.order_id}")
                    return True
                else:
                    logger.error(f"❌ Order cancellation failed: {order.order_id}")
                    return False
            else:
                logger.warning(f"⚠️ No broker order ID to cancel: {order.order_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Order cancellation error: {e}")
            return False
    
    def _reconciliation_loop(self):
        """Background reconciliation loop"""
        logger.info("🔄 Reconciliation loop started")
        
        while True:
            try:
                # Perform reconciliation
                self._perform_reconciliation()
                
                # Wait for next reconciliation
                time.sleep(self.reconciliation_interval)
                
            except Exception as e:
                logger.error(f"❌ Reconciliation loop error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _perform_reconciliation(self):
        """Perform position reconciliation with broker"""
        try:
            logger.info("🔄 Performing position reconciliation...")
            
            # Get positions from broker
            broker_positions = self.broker_adapter.get_positions()
            
            # Update broker positions
            self.broker_positions = broker_positions
            
            # Reconcile with local positions
            reconciliation_results = self._reconcile_positions(broker_positions)
            
            # Log reconciliation results
            if reconciliation_results['discrepancies']:
                logger.warning(f"⚠️ Position discrepancies found: {reconciliation_results['discrepancies']}")
            else:
                logger.info("✅ Position reconciliation successful - no discrepancies")
            
            # Update order statuses
            self._update_order_statuses()
            
        except Exception as e:
            logger.error(f"❌ Position reconciliation failed: {e}")
    
    def _reconcile_positions(self, broker_positions: Dict[str, Position]) -> Dict[str, Any]:
        """Reconcile local and broker positions"""
        try:
            discrepancies = []
            
            # Check for missing positions
            for symbol, broker_pos in broker_positions.items():
                if symbol not in self.positions:
                    discrepancies.append(f"Missing local position: {symbol}")
                    # Add to local positions
                    self.positions[symbol] = broker_pos
                else:
                    local_pos = self.positions[symbol]
                    # Check for quantity discrepancies
                    if abs(local_pos.quantity - broker_pos.quantity) > 0.01:
                        discrepancies.append(f"Quantity discrepancy for {symbol}: local={local_pos.quantity}, broker={broker_pos.quantity}")
                        # Update local position
                        self.positions[symbol] = broker_pos
            
            # Check for extra positions
            for symbol, local_pos in self.positions.items():
                if symbol not in broker_positions:
                    discrepancies.append(f"Extra local position: {symbol}")
                    # Remove from local positions
                    del self.positions[symbol]
            
            return {
                'discrepancies': discrepancies,
                'reconciled_positions': len(self.positions),
                'broker_positions': len(broker_positions)
            }
            
        except Exception as e:
            logger.error(f"❌ Position reconciliation failed: {e}")
            return {'discrepancies': [f"Reconciliation error: {e}"], 'reconciled_positions': 0, 'broker_positions': 0}
    
    def _update_order_statuses(self):
        """Update order statuses from broker"""
        try:
            for order_id, order in self.pending_orders.items():
                if order.broker_order_id:
                    # Get order status from broker
                    status, filled_qty, avg_price = self.broker_adapter.get_order_status(order.broker_order_id)
                    
                    if status:
                        # Update order status
                        if status == 'FILLED':
                            order.status = OrderStatus.FILLED
                            order.filled_quantity = filled_qty
                            order.average_price = avg_price
                            
                            # Move to filled orders
                            self.filled_orders[order_id] = order
                            del self.pending_orders[order_id]
                            
                            # Update statistics
                            self.execution_stats['filled_orders'] += 1
                            
                            logger.info(f"✅ Order filled: {order_id} - {filled_qty} @ {avg_price}")
                            
                        elif status == 'PARTIALLY_FILLED':
                            order.status = OrderStatus.PARTIALLY_FILLED
                            order.filled_quantity = filled_qty
                            order.average_price = avg_price
                            
                            logger.info(f"🔄 Order partially filled: {order_id} - {filled_qty} @ {avg_price}")
                            
                        elif status == 'CANCELLED':
                            order.status = OrderStatus.CANCELLED
                            
                            # Move to cancelled orders
                            self.cancelled_orders[order_id] = order
                            del self.pending_orders[order_id]
                            
                            # Update statistics
                            self.execution_stats['cancelled_orders'] += 1
                            
                            logger.info(f"❌ Order cancelled: {order_id}")
                            
                        elif status == 'REJECTED':
                            order.status = OrderStatus.REJECTED
                            
                            # Move to cancelled orders
                            self.cancelled_orders[order_id] = order
                            del self.pending_orders[order_id]
                            
                            # Update statistics
                            self.execution_stats['rejected_orders'] += 1
                            
                            logger.info(f"❌ Order rejected: {order_id}")
                
        except Exception as e:
            logger.error(f"❌ Order status update failed: {e}")
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        try:
            # Check all order dictionaries
            for order_dict in [self.pending_orders, self.filled_orders, self.cancelled_orders]:
                if order_id in order_dict:
                    return order_dict[order_id]
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get order status: {e}")
            return None
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.positions.copy()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        try:
            total_orders = self.execution_stats['total_orders']
            if total_orders > 0:
                self.execution_stats['success_rate'] = self.execution_stats['filled_orders'] / total_orders
            
            return self.execution_stats.copy()
            
        except Exception as e:
            logger.error(f"❌ Failed to get execution stats: {e}")
            return {}
    
    def stop(self):
        """Stop execution manager"""
        try:
            # Cancel all pending orders
            for order_id in list(self.pending_orders.keys()):
                self._cancel_order(self.pending_orders[order_id])
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("✅ Enhanced execution manager stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop execution manager: {e}")

# Mock broker adapter for testing
class MockBrokerAdapter:
    """Mock broker adapter for testing"""
    
    def __init__(self):
        self.orders = {}
        self.positions = {}
        self.order_counter = 0
    
    def place_order(self, symbol: str, order_type: str, side: str, 
                   quantity: float, price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> str:
        """Place order"""
        try:
            self.order_counter += 1
            broker_order_id = f"BROKER_{self.order_counter}"
            
            # Simulate order placement
            self.orders[broker_order_id] = {
                'symbol': symbol,
                'order_type': order_type,
                'side': side,
                'quantity': quantity,
                'price': price,
                'stop_price': stop_price,
                'status': 'SUBMITTED',
                'filled_quantity': 0.0,
                'average_price': 0.0,
                'timestamp': datetime.now()
            }
            
            return broker_order_id
            
        except Exception as e:
            logger.error(f"❌ Mock broker order placement failed: {e}")
            return None
    
    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order"""
        try:
            if broker_order_id in self.orders:
                self.orders[broker_order_id]['status'] = 'CANCELLED'
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Mock broker order cancellation failed: {e}")
            return False
    
    def get_order_status(self, broker_order_id: str) -> Tuple[Optional[str], float, float]:
        """Get order status"""
        try:
            if broker_order_id in self.orders:
                order = self.orders[broker_order_id]
                return order['status'], order['filled_quantity'], order['average_price']
            return None, 0.0, 0.0
            
        except Exception as e:
            logger.error(f"❌ Mock broker order status failed: {e}")
            return None, 0.0, 0.0
    
    def get_positions(self) -> Dict[str, Position]:
        """Get positions"""
        return self.positions.copy()

# Global execution manager instance
execution_manager = None

def initialize_execution_manager(broker_adapter):
    """Initialize global execution manager"""
    global execution_manager
    execution_manager = EnhancedExecutionManager(broker_adapter)
    return execution_manager

def get_execution_manager():
    """Get global execution manager"""
    return execution_manager
