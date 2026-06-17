#!/usr/bin/env python3
"""
Trade Execution Manager with Reliability Features
Handles order execution, retries, fallbacks, and position reconciliation
"""

import sys
import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from abc import ABC, abstractmethod

from src.models.postgres_database import PostgresDatabase
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # BUY or SELL
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    timestamp: datetime
    status: OrderStatus
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    rejection_reason: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class ExecutionConfig:
    """Execution configuration"""
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    timeout: float = 30.0
    enable_fallback: bool = True
    enable_position_reconciliation: bool = True
    reconciliation_interval: int = 300  # 5 minutes
    max_slippage: float = 0.01  # 1%
    enable_circuit_breaker: bool = True
    max_failures_per_minute: int = 10

class BrokerInterface(ABC):
    """Abstract broker interface"""
    
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> float:
        pass

class FyersBroker(BrokerInterface):
    """Fyers broker implementation"""
    
    def __init__(self, fyers_client):
        self.fyers_client = fyers_client
        self.orders = {}
    
    async def place_order(self, order: Order) -> str:
        """Place order with Fyers"""
        try:
            # Simulate API call
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Mock order placement
            order_id = f"FYERS_{int(time.time())}"
            order.order_id = order_id
            order.status = OrderStatus.SUBMITTED
            
            self.orders[order_id] = order
            
            logger.info(f"✅ Order placed: {order_id} - {order.symbol} {order.side} {order.quantity}")
            return order_id
            
        except Exception as e:
            logger.error(f"❌ Failed to place order: {e}")
            order.status = OrderStatus.FAILED
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with Fyers"""
        try:
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELLED
                logger.info(f"✅ Order cancelled: {order_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel order: {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Fyers"""
        try:
            if order_id in self.orders:
                # Simulate order execution
                order = self.orders[order_id]
                if order.status == OrderStatus.SUBMITTED:
                    # Simulate random execution
                    if np.random.random() > 0.3:  # 70% execution rate
                        order.status = OrderStatus.FILLED
                        order.filled_quantity = order.quantity
                        order.average_price = order.price or 100.0
                        order.commission = order.quantity * order.average_price * 0.001
                
                return order.status
            return OrderStatus.FAILED
            
        except Exception as e:
            logger.error(f"❌ Failed to get order status: {order_id}: {e}")
            return OrderStatus.FAILED
    
    async def get_positions(self) -> Dict[str, float]:
        """Get current positions from Fyers"""
        try:
            # Mock positions
            return {
                "NSE:NIFTY50-INDEX": 100.0,
                "NSE:NIFTYBANK-INDEX": 50.0
            }
        except Exception as e:
            logger.error(f"❌ Failed to get positions: {e}")
            return {}
    
    async def get_account_balance(self) -> float:
        """Get account balance from Fyers"""
        try:
            # Mock balance
            return 100000.0
        except Exception as e:
            logger.error(f"❌ Failed to get account balance: {e}")
            return 0.0

class TradeExecutionManager:
    """Advanced trade execution manager with reliability features"""
    
    def __init__(self, broker: BrokerInterface, config: ExecutionConfig):
        self.broker = broker
        self.config = config
        self.orders = {}
        self.positions = {}
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_breaker_active = False
        self.reconciliation_task = None
        self.active_trade_features = {} # trade_id -> features
        self.db = PostgresDatabase()
        
        # Start position reconciliation
        if config.enable_position_reconciliation:
            self._start_reconciliation()
    
    async def place_order(self, symbol: str, side: str, quantity: float, 
                         order_type: OrderType = OrderType.MARKET, 
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         features: Optional[Dict] = None) -> str:
        """Place order with retry logic and fallback"""
        
        # Check circuit breaker
        if self.circuit_breaker_active:
            raise Exception("Circuit breaker active - trading halted")
        
        # Create order
        order = Order(
            order_id="",
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            timestamp=datetime.now(),
            status=OrderStatus.PENDING,
            max_retries=self.config.max_retries
        )
        
        # Try to place order with retries
        for attempt in range(self.config.max_retries + 1):
            try:
                order_id = await self._place_order_with_retry(order, attempt)
                if order_id:
                    self.orders[order_id] = order
                    if features:
                        features['trade_id'] = order_id
                        features['entry_price'] = price or 0.0
                        features['entry_time'] = datetime.now().isoformat()
                        self.active_trade_features[order_id] = features
                        # Save trade performance initiation
                        self.db.save_trade_performance({
                            'trade_id': order_id,
                            'entry_time': features['entry_time'],
                            'strategy': features['strategy_name'],
                            'symbol': symbol,
                            'entry_price': features['entry_price'],
                            'features': features
                        })
                        
                        # Update original signal as executed
                        if 'signal_id' in features:
                            self.db.save_signal({
                                'signal_id': features['signal_id'],
                                'executed': True
                            })
                    return order_id
                    
            except Exception as e:
                logger.warning(f"⚠️ Order placement attempt {attempt + 1} failed: {e}")
                order.retry_count = attempt + 1
                
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                    await asyncio.sleep(delay)
                else:
                    self._handle_order_failure(order, str(e))
                    raise
        
        return None
    
    async def _place_order_with_retry(self, order: Order, attempt: int) -> str:
        """Place order with timeout and error handling"""
        try:
            # Set timeout
            order_id = await asyncio.wait_for(
                self.broker.place_order(order),
                timeout=self.config.timeout
            )
            
            # Reset failure count on success
            self.failure_count = 0
            self.circuit_breaker_active = False
            
            return order_id
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Order timeout after {self.config.timeout}s")
            raise Exception("Order placement timeout")
            
        except Exception as e:
            self._increment_failure_count()
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with retry logic"""
        try:
            for attempt in range(self.config.max_retries + 1):
                success = await self.broker.cancel_order(order_id)
                if success:
                    if order_id in self.orders:
                        self.orders[order_id].status = OrderStatus.CANCELLED
                    return True
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status with error handling"""
        try:
            status = await self.broker.get_order_status(order_id)
            
            # Update local order
            if order_id in self.orders:
                self.orders[order_id].status = status
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get order status {order_id}: {e}")
            return OrderStatus.FAILED
    
    async def monitor_orders(self):
        """Monitor all pending orders"""
        while True:
            try:
                pending_orders = [oid for oid, order in self.orders.items() 
                                if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]]
                
                for order_id in pending_orders:
                    status = await self.get_order_status(order_id)
                    
                    if status == OrderStatus.FILLED:
                        await self._handle_filled_order(order_id)
                    elif status == OrderStatus.REJECTED:
                        await self._handle_rejected_order(order_id)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Error monitoring orders: {e}")
                await asyncio.sleep(10)
    
    async def _handle_filled_order(self, order_id: str):
        """Handle filled order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            logger.info(f"✅ Order filled: {order_id} - {order.symbol} {order.side} {order.filled_quantity}")
            
            # Update positions
            self._update_positions(order)
            
            # Start MFE/MAE tracking for this order if it's an entry
            # In a real system, we'd check if this opens a new position
    
    async def _handle_rejected_order(self, order_id: str):
        """Handle rejected order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            logger.warning(f"❌ Order rejected: {order_id} - {order.rejection_reason}")
    
    def _handle_order_failure(self, order: Order, error: str):
        """Handle order failure"""
        logger.error(f"❌ Order failed after {order.retry_count} retries: {error}")
        order.status = OrderStatus.FAILED
        order.rejection_reason = error
        
        self._increment_failure_count()
    
    def _increment_failure_count(self):
        """Increment failure count and check circuit breaker"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        # Check circuit breaker
        if self.config.enable_circuit_breaker:
            if self.failure_count >= self.config.max_failures_per_minute:
                self.circuit_breaker_active = True
                logger.error(f"🚨 Circuit breaker activated - {self.failure_count} failures in last minute")
    
    def _update_positions(self, order: Order):
        """Update internal position tracking"""
        symbol = order.symbol
        quantity = order.filled_quantity
        
        if order.side == "BUY":
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity

    async def close_position(self, symbol: str, reason: str = "MANUAL"):
        """Close position and record exit analytics"""
        if symbol in self.positions and abs(self.positions[symbol]) > 0:
            side = "SELL" if self.positions[symbol] > 0 else "BUY"
            qty = abs(self.positions[symbol])
            
            # Find active trade_id
            active_trade_id = next((tid for tid, feat in self.active_trade_features.items() 
                                  if self.orders[tid].symbol == symbol), None)
            
            order_id = await self.place_order(symbol, side, qty)
            
            if order_id and active_trade_id:
                features = self.active_trade_features[active_trade_id]
                entry_time = datetime.fromisoformat(features['entry_time'])
                exit_time = datetime.now()
                duration = int((exit_time - entry_time).total_seconds())
                
                analysis = {
                    'trade_id': active_trade_id,
                    'exit_reason': reason,
                    'time_in_trade': duration,
                    'mfe': features.get('mfe', 0.0),
                    'mae': features.get('mae', 0.0)
                }
                
                # Update features for final save
                features['exit_time'] = exit_time.isoformat()
                features['exit_price'] = self.orders[order_id].average_price
                features['pnl'] = features['mfe'] # Placeholder for actual PnL calculation
                features['win_loss'] = 'WIN' if features['pnl'] > 0 else 'LOSS'
                
                # Update performance in Postgres
                self.db.save_trade_performance({
                    'trade_id': active_trade_id,
                    'exit_time': features['exit_time'],
                    'exit_price': features['exit_price'],
                    'pnl': features['pnl'],
                    'exit_reason': reason,
                    'mfe': features['mfe'],
                    'mae': features['mae']
                })
                
                # Cleanup active tracking
                del self.active_trade_features[active_trade_id]

    async def update_mfe_mae(self, symbol: str, current_price: float):
        """Update MFE and MAE for all active trades of this symbol"""
        for order_id, features in self.active_trade_features.items():
            order = self.orders.get(order_id)
            if order and order.symbol == symbol and order.status == OrderStatus.FILLED:
                entry = features['entry_price']
                side = order.side # BUY or SELL
                
                if side == "BUY":
                    favorable = current_price - entry
                    adverse = current_price - entry
                    features['mfe'] = max(features.get('mfe', 0), favorable)
                    features['mae'] = min(features.get('mae', 0), adverse)
                else: # SELL
                    favorable = entry - current_price
                    adverse = entry - current_price
                    features['mfe'] = max(features.get('mfe', 0), favorable)
                    features['mae'] = min(features.get('mae', 0), adverse)
                
                # Persist updates periodically
                self.db.save_trade_performance({
                    'trade_id': order_id,
                    'mfe': features['mfe'],
                    'mae': features['mae']
                })
    
    async def reconcile_positions(self):
        """Reconcile positions with broker"""
        try:
            logger.info("🔄 Starting position reconciliation...")
            
            # Get positions from broker
            broker_positions = await self.broker.get_positions()
            
            # Compare with internal positions
            discrepancies = []
            for symbol, broker_qty in broker_positions.items():
                internal_qty = self.positions.get(symbol, 0)
                if abs(broker_qty - internal_qty) > 0.01:  # Allow small rounding differences
                    discrepancies.append({
                        'symbol': symbol,
                        'broker_qty': broker_qty,
                        'internal_qty': internal_qty,
                        'difference': broker_qty - internal_qty
                    })
            
            if discrepancies:
                logger.warning(f"⚠️ Position discrepancies found: {discrepancies}")
                # Update internal positions to match broker
                for disc in discrepancies:
                    self.positions[disc['symbol']] = disc['broker_qty']
            else:
                logger.info("✅ Position reconciliation successful - no discrepancies")
            
        except Exception as e:
            logger.error(f"❌ Position reconciliation failed: {e}")
    
    def _start_reconciliation(self):
        """Start periodic position reconciliation"""
        async def reconciliation_loop():
            while True:
                await asyncio.sleep(self.config.reconciliation_interval)
                await self.reconcile_positions()
        
        # Start reconciliation task
        self.reconciliation_task = asyncio.create_task(reconciliation_loop())
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            # Get account balance
            balance = await self.broker.get_account_balance()
            
            # Get current positions
            broker_positions = await self.broker.get_positions()
            
            # Calculate portfolio value
            total_value = balance
            for symbol, quantity in broker_positions.items():
                # In real implementation, get current price
                current_price = 100.0  # Mock price
                total_value += quantity * current_price
            
            return {
                'account_balance': balance,
                'positions': broker_positions,
                'total_value': total_value,
                'open_orders': len([o for o in self.orders.values() if o.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]]),
                'circuit_breaker_active': self.circuit_breaker_active,
                'failure_count': self.failure_count
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get portfolio summary: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown execution manager"""
        if self.reconciliation_task:
            self.reconciliation_task.cancel()
        
        # Cancel all pending orders
        pending_orders = [oid for oid, order in self.orders.items() 
                         if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]]
        
        for order_id in pending_orders:
            await self.cancel_order(order_id)
        
        logger.info("🛑 Trade execution manager shutdown complete")

def main():
    """Main function for testing"""
    async def test_execution_manager():
        # Create mock broker
        broker = FyersBroker(None)
        
        # Configure execution
        config = ExecutionConfig(
            max_retries=3,
            retry_delay=1.0,
            timeout=30.0,
            enable_fallback=True,
            enable_position_reconciliation=True
        )
        
        # Create execution manager
        manager = TradeExecutionManager(broker, config)
        
        try:
            # Test order placement
            order_id = await manager.place_order(
                symbol="NSE:NIFTY50-INDEX",
                side="BUY",
                quantity=100,
                order_type=OrderType.MARKET
            )
            
            print(f"✅ Order placed: {order_id}")
            
            # Monitor order
            for i in range(10):
                status = await manager.get_order_status(order_id)
                print(f"Order status: {status}")
                
                if status == OrderStatus.FILLED:
                    break
                
                await asyncio.sleep(1)
            
            # Get portfolio summary
            summary = await manager.get_portfolio_summary()
            print(f"Portfolio summary: {summary}")
            
        finally:
            await manager.shutdown()
    
    # Run test
    asyncio.run(test_execution_manager())

if __name__ == "__main__":
    main()
