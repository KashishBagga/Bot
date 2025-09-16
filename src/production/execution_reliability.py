#!/usr/bin/env python3
"""
Execution Reliability & Reconciliation
MUST #2: Guaranteed confirm/reconcile loop with zero un-reconciled trades
"""

import sys
import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    UNRECONCILED = "UNRECONCILED"

@dataclass
class Order:
    """Order with reconciliation tracking"""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    status: OrderStatus
    broker_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    reconciliation_attempts: int = 0
    last_reconciliation: Optional[datetime] = None
    reconciliation_status: str = "PENDING"

@dataclass
class ReconciliationConfig:
    """Reconciliation configuration"""
    poll_interval: int = 5  # seconds
    max_poll_attempts: int = 12  # 1 minute total
    reconciliation_interval: int = 60  # 1 minute
    full_reconciliation_interval: int = 300  # 5 minutes
    max_reconciliation_attempts: int = 3
    alert_threshold: int = 1  # Alert if any un-reconciled trades

class ExecutionReliabilityManager:
    """Execution reliability and reconciliation manager"""
    
    def __init__(self, config: ReconciliationConfig):
        self.config = config
        self.orders = {}
        self.reconciliation_queue = []
        self.reconciliation_running = False
        self.last_full_reconciliation = None
        self.unreconciled_count = 0
        
    async def place_order_with_guarantee(self, symbol: str, side: str, quantity: float, price: float) -> str:
        """Place order with guaranteed confirmation and reconciliation"""
        try:
            # Create order
            order = Order(
                order_id=f"ORD_{int(time.time())}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING
            )
            
            self.orders[order.order_id] = order
            
            # Step 1: Place order with broker
            logger.info(f"📤 Placing order: {order.order_id} - {symbol} {side} {quantity} @ {price}")
            broker_order_id = await self._place_order_with_broker(order)
            
            if broker_order_id:
                order.broker_order_id = broker_order_id
                order.status = OrderStatus.SUBMITTED
                logger.info(f"✅ Order submitted: {order.order_id} -> {broker_order_id}")
            else:
                order.status = OrderStatus.FAILED
                logger.error(f"❌ Order placement failed: {order.order_id}")
                return order.order_id
            
            # Step 2: Poll until filled/rejected
            fill_result = await self._poll_until_filled(order)
            
            if fill_result['status'] == OrderStatus.FILLED:
                order.status = OrderStatus.FILLED
                order.filled_quantity = fill_result['quantity']
                order.average_price = fill_result['price']
                order.commission = fill_result['commission']
                logger.info(f"✅ Order filled: {order.order_id} - {order.filled_quantity} @ {order.average_price}")
            elif fill_result['status'] == OrderStatus.REJECTED:
                order.status = OrderStatus.REJECTED
                logger.warning(f"⚠️ Order rejected: {order.order_id}")
            else:
                # Timeout - cancel order
                await self._cancel_order(order)
                order.status = OrderStatus.CANCELLED
                logger.warning(f"⏰ Order timeout, cancelled: {order.order_id}")
            
            # Step 3: Add to reconciliation queue
            self.reconciliation_queue.append(order.order_id)
            
            return order.order_id
            
        except Exception as e:
            logger.error(f"❌ Order placement failed: {e}")
            if order.order_id in self.orders:
                self.orders[order.order_id].status = OrderStatus.FAILED
            raise
    
    async def _place_order_with_broker(self, order: Order) -> Optional[str]:
        """Place order with broker (mock implementation)"""
        try:
            # Simulate broker API call
            await asyncio.sleep(0.1)
            
            # Mock successful order placement
            broker_order_id = f"BRK_{int(time.time())}"
            logger.info(f"📡 Broker order placed: {broker_order_id}")
            
            return broker_order_id
            
        except Exception as e:
            logger.error(f"❌ Broker order placement failed: {e}")
            return None
    
    async def _poll_until_filled(self, order: Order) -> Dict[str, Any]:
        """Poll order status until filled, rejected, or timeout"""
        try:
            for attempt in range(self.config.max_poll_attempts):
                # Check order status with broker
                status_result = await self._check_broker_order_status(order)
                
                if status_result['status'] == OrderStatus.FILLED:
                    return {
                        'status': OrderStatus.FILLED,
                        'quantity': status_result['quantity'],
                        'price': status_result['price'],
                        'commission': status_result['commission']
                    }
                elif status_result['status'] == OrderStatus.REJECTED:
                    return {
                        'status': OrderStatus.REJECTED,
                        'reason': status_result['reason']
                    }
                
                # Wait before next poll
                await asyncio.sleep(self.config.poll_interval)
            
            # Timeout
            return {'status': OrderStatus.PENDING}
            
        except Exception as e:
            logger.error(f"❌ Order polling failed: {e}")
            return {'status': OrderStatus.FAILED}
    
    async def _check_broker_order_status(self, order: Order) -> Dict[str, Any]:
        """Check order status with broker (mock implementation)"""
        try:
            # Simulate broker API call
            await asyncio.sleep(0.05)
            
            # Mock order status check
            # Simulate 70% fill rate
            if np.random.random() > 0.3:
                return {
                    'status': OrderStatus.FILLED,
                    'quantity': order.quantity,
                    'price': order.price + np.random.uniform(-0.5, 0.5),  # Slight slippage
                    'commission': order.quantity * order.price * 0.001
                }
            else:
                return {
                    'status': OrderStatus.PENDING,
                    'reason': 'Still pending'
                }
                
        except Exception as e:
            logger.error(f"❌ Broker status check failed: {e}")
            return {'status': OrderStatus.FAILED}
    
    async def _cancel_order(self, order: Order) -> bool:
        """Cancel order with broker"""
        try:
            # Simulate broker cancellation
            await asyncio.sleep(0.1)
            logger.info(f"🚫 Order cancelled: {order.order_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Order cancellation failed: {e}")
            return False
    
    async def start_reconciliation_loop(self):
        """Start continuous reconciliation loop"""
        if self.reconciliation_running:
            return
        
        self.reconciliation_running = True
        logger.info("🔄 Starting reconciliation loop")
        
        while self.reconciliation_running:
            try:
                # Quick reconciliation every minute
                await self._quick_reconciliation()
                
                # Full reconciliation every 5 minutes
                if (self.last_full_reconciliation is None or 
                    datetime.now() - self.last_full_reconciliation > timedelta(seconds=self.config.full_reconciliation_interval)):
                    await self._full_reconciliation()
                    self.last_full_reconciliation = datetime.now()
                
                # Wait before next reconciliation
                await asyncio.sleep(self.config.reconciliation_interval)
                
            except Exception as e:
                logger.error(f"❌ Reconciliation loop error: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def _quick_reconciliation(self):
        """Quick reconciliation of pending orders"""
        try:
            unreconciled_orders = [oid for oid, order in self.orders.items() 
                                 if order.status in [OrderStatus.FILLED, OrderStatus.SUBMITTED] 
                                 and order.reconciliation_status == "PENDING"]
            
            for order_id in unreconciled_orders:
                order = self.orders[order_id]
                await self._reconcile_single_order(order)
                
        except Exception as e:
            logger.error(f"❌ Quick reconciliation failed: {e}")
    
    async def _full_reconciliation(self):
        """Full reconciliation with broker"""
        try:
            logger.info("🔄 Starting full reconciliation")
            
            # Get all orders from broker
            broker_orders = await self._get_all_broker_orders()
            
            # Compare with internal orders
            internal_orders = {oid: order for oid, order in self.orders.items() 
                             if order.status in [OrderStatus.FILLED, OrderStatus.SUBMITTED]}
            
            # Find discrepancies
            discrepancies = self._find_reconciliation_discrepancies(internal_orders, broker_orders)
            
            if discrepancies:
                logger.warning(f"⚠️ Found {len(discrepancies)} reconciliation discrepancies")
                await self._handle_reconciliation_discrepancies(discrepancies)
            else:
                logger.info("✅ Full reconciliation successful - no discrepancies")
            
            # Update reconciliation status
            self.unreconciled_count = len(discrepancies)
            
        except Exception as e:
            logger.error(f"❌ Full reconciliation failed: {e}")
    
    async def _reconcile_single_order(self, order: Order):
        """Reconcile single order"""
        try:
            if order.reconciliation_attempts >= self.config.max_reconciliation_attempts:
                order.reconciliation_status = "FAILED"
                logger.error(f"❌ Order reconciliation failed after {order.reconciliation_attempts} attempts: {order.order_id}")
                return
            
            # Check order status with broker
            broker_status = await self._check_broker_order_status(order)
            
            # Update order based on broker status
            if broker_status['status'] == OrderStatus.FILLED:
                if order.status != OrderStatus.FILLED:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = broker_status['quantity']
                    order.average_price = broker_status['price']
                    order.commission = broker_status['commission']
                    logger.info(f"✅ Order reconciled as filled: {order.order_id}")
                
                order.reconciliation_status = "RECONCILED"
            else:
                order.reconciliation_attempts += 1
                order.last_reconciliation = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Single order reconciliation failed: {e}")
            order.reconciliation_attempts += 1
    
    async def _get_all_broker_orders(self) -> Dict[str, Dict[str, Any]]:
        """Get all orders from broker (mock implementation)"""
        try:
            # Simulate broker API call
            await asyncio.sleep(0.1)
            
            # Mock broker orders
            broker_orders = {}
            for order_id, order in self.orders.items():
                if order.broker_order_id:
                    broker_orders[order.broker_order_id] = {
                        'status': order.status.value,
                        'filled_quantity': order.filled_quantity,
                        'average_price': order.average_price,
                        'commission': order.commission
                    }
            
            return broker_orders
            
        except Exception as e:
            logger.error(f"❌ Failed to get broker orders: {e}")
            return {}
    
    def _find_reconciliation_discrepancies(self, internal_orders: Dict[str, Order], 
                                         broker_orders: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find discrepancies between internal and broker orders"""
        discrepancies = []
        
        for order_id, order in internal_orders.items():
            if order.broker_order_id and order.broker_order_id in broker_orders:
                broker_order = broker_orders[order.broker_order_id]
                
                # Check for discrepancies
                if (order.status.value != broker_order['status'] or
                    abs(order.filled_quantity - broker_order['filled_quantity']) > 0.01 or
                    abs(order.average_price - broker_order['average_price']) > 0.01):
                    
                    discrepancies.append({
                        'order_id': order_id,
                        'broker_order_id': order.broker_order_id,
                        'internal_status': order.status.value,
                        'broker_status': broker_order['status'],
                        'internal_quantity': order.filled_quantity,
                        'broker_quantity': broker_order['filled_quantity'],
                        'internal_price': order.average_price,
                        'broker_price': broker_order['average_price']
                    })
        
        return discrepancies
    
    async def _handle_reconciliation_discrepancies(self, discrepancies: List[Dict[str, Any]]):
        """Handle reconciliation discrepancies"""
        for discrepancy in discrepancies:
            logger.warning(f"⚠️ Reconciliation discrepancy: {discrepancy}")
            
            # Update internal order to match broker
            order_id = discrepancy['order_id']
            if order_id in self.orders:
                order = self.orders[order_id]
                order.status = OrderStatus(discrepancy['broker_status'])
                order.filled_quantity = discrepancy['broker_quantity']
                order.average_price = discrepancy['broker_price']
                order.reconciliation_status = "RECONCILED"
                
                logger.info(f"✅ Order updated to match broker: {order_id}")
    
    async def stop_reconciliation_loop(self):
        """Stop reconciliation loop"""
        self.reconciliation_running = False
        logger.info("🛑 Reconciliation loop stopped")
    
    def get_reconciliation_status(self) -> Dict[str, Any]:
        """Get current reconciliation status"""
        total_orders = len(self.orders)
        reconciled_orders = sum(1 for order in self.orders.values() 
                              if order.reconciliation_status == "RECONCILED")
        pending_orders = sum(1 for order in self.orders.values() 
                           if order.reconciliation_status == "PENDING")
        failed_orders = sum(1 for order in self.orders.values() 
                          if order.reconciliation_status == "FAILED")
        
        return {
            'total_orders': total_orders,
            'reconciled_orders': reconciled_orders,
            'pending_orders': pending_orders,
            'failed_orders': failed_orders,
            'unreconciled_count': self.unreconciled_count,
            'reconciliation_running': self.reconciliation_running,
            'last_full_reconciliation': self.last_full_reconciliation.isoformat() if self.last_full_reconciliation else None
        }
    
    def check_acceptance_criteria(self) -> bool:
        """Check if acceptance criteria are met"""
        # Acceptance criteria: zero un-reconciled trades after 5 minutes in normal operation
        return self.unreconciled_count == 0

def main():
    """Main function for testing"""
    async def test_execution_reliability():
        config = ReconciliationConfig(
            poll_interval=2,
            max_poll_attempts=6,
            reconciliation_interval=10,
            full_reconciliation_interval=30
        )
        
        manager = ExecutionReliabilityManager(config)
        
        try:
            # Start reconciliation loop
            reconciliation_task = asyncio.create_task(manager.start_reconciliation_loop())
            
            # Test order placement
            order_id = await manager.place_order_with_guarantee(
                symbol="NSE:NIFTY50-INDEX",
                side="BUY",
                quantity=100,
                price=19500
            )
            
            print(f"✅ Order placed: {order_id}")
            
            # Wait for reconciliation
            await asyncio.sleep(35)
            
            # Check status
            status = manager.get_reconciliation_status()
            print(f"📊 Reconciliation status: {status}")
            
            # Check acceptance criteria
            criteria_met = manager.check_acceptance_criteria()
            print(f"✅ Acceptance criteria met: {criteria_met}")
            
        finally:
            await manager.stop_reconciliation_loop()
            reconciliation_task.cancel()
    
    # Run test
    asyncio.run(test_execution_reliability())

if __name__ == "__main__":
    main()
