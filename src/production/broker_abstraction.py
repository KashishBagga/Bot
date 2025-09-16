#!/usr/bin/env python3
"""
Broker Abstraction & Multi-Broker Failover
HIGH PRIORITY #3: Broker abstraction with failover logic
"""

import sys
import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Protocol
from dataclasses import dataclass
from enum import Enum
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrokerStatus(Enum):
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    MAINTENANCE = "MAINTENANCE"
    UNKNOWN = "UNKNOWN"

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    order_type: str
    timestamp: datetime
    status: OrderStatus
    broker_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0

@dataclass
class BrokerConfig:
    """Broker configuration"""
    broker_name: str
    api_endpoint: str
    credentials: Dict[str, str]
    timeout: int = 30
    retry_attempts: int = 3
    priority: int = 1  # Lower number = higher priority
    enabled: bool = True

class IBrokerAdapter(Protocol):
    """Broker adapter interface"""
    
    async def place_order(self, order: Order) -> str:
        """Place order and return broker order ID"""
        ...
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        ...
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        ...
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        ...
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance"""
        ...
    
    async def health_check(self) -> bool:
        """Check broker health"""
        ...

class FyersBrokerAdapter:
    """Fyers broker adapter"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.broker_name = config.broker_name
        self.api_endpoint = config.api_endpoint
        self.credentials = config.credentials
        self.timeout = config.timeout
        self.retry_attempts = config.retry_attempts
        self.error_count = 0
        self.last_error_time = None
        self.status = BrokerStatus.ACTIVE
    
    async def place_order(self, order: Order) -> str:
        """Place order with Fyers"""
        try:
            # Mock Fyers order placement
            await asyncio.sleep(0.1)
            
            # Simulate occasional failures
            if self.error_count > 5:
                raise Exception("Broker temporarily unavailable")
            
            broker_order_id = f"FYERS_{int(time.time())}"
            logger.info(f"📤 Order placed with Fyers: {broker_order_id}")
            
            return broker_order_id
            
        except Exception as e:
            self._record_error()
            logger.error(f"❌ Fyers order placement failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with Fyers"""
        try:
            # Mock Fyers order cancellation
            await asyncio.sleep(0.1)
            logger.info(f"🚫 Order cancelled with Fyers: {order_id}")
            return True
            
        except Exception as e:
            self._record_error()
            logger.error(f"❌ Fyers order cancellation failed: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Fyers"""
        try:
            # Mock Fyers order status check
            await asyncio.sleep(0.05)
            return OrderStatus.FILLED
            
        except Exception as e:
            self._record_error()
            logger.error(f"❌ Fyers order status check failed: {e}")
            return OrderStatus.FAILED
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from Fyers"""
        try:
            # Mock Fyers positions
            await asyncio.sleep(0.1)
            return [
                {
                    'symbol': 'NSE:NIFTY50-INDEX',
                    'quantity': 100,
                    'average_price': 19500,
                    'market_value': 1960000
                }
            ]
            
        except Exception as e:
            self._record_error()
            logger.error(f"❌ Fyers positions retrieval failed: {e}")
            return []
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance from Fyers"""
        try:
            # Mock Fyers account balance
            await asyncio.sleep(0.1)
            return {
                'cash': 50000.0,
                'margin': 100000.0,
                'available': 45000.0
            }
            
        except Exception as e:
            self._record_error()
            logger.error(f"❌ Fyers account balance retrieval failed: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check Fyers broker health"""
        try:
            # Mock health check
            await asyncio.sleep(0.1)
            
            # Simulate health check failure if too many errors
            if self.error_count > 10:
                self.status = BrokerStatus.FAILED
                return False
            
            self.status = BrokerStatus.ACTIVE
            return True
            
        except Exception as e:
            self._record_error()
            self.status = BrokerStatus.FAILED
            logger.error(f"❌ Fyers health check failed: {e}")
            return False
    
    def _record_error(self):
        """Record broker error"""
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        if self.error_count > 5:
            self.status = BrokerStatus.FAILED
            logger.warning(f"⚠️ Broker {self.broker_name} marked as failed")

class ZerodhaBrokerAdapter:
    """Zerodha broker adapter"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.broker_name = config.broker_name
        self.api_endpoint = config.api_endpoint
        self.credentials = config.credentials
        self.timeout = config.timeout
        self.retry_attempts = config.retry_attempts
        self.error_count = 0
        self.last_error_time = None
        self.status = BrokerStatus.ACTIVE
    
    async def place_order(self, order: Order) -> str:
        """Place order with Zerodha"""
        try:
            # Mock Zerodha order placement
            await asyncio.sleep(0.1)
            broker_order_id = f"ZERODHA_{int(time.time())}"
            logger.info(f"📤 Order placed with Zerodha: {broker_order_id}")
            return broker_order_id
            
        except Exception as e:
            self._record_error()
            logger.error(f"❌ Zerodha order placement failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with Zerodha"""
        try:
            await asyncio.sleep(0.1)
            logger.info(f"🚫 Order cancelled with Zerodha: {order_id}")
            return True
            
        except Exception as e:
            self._record_error()
            logger.error(f"❌ Zerodha order cancellation failed: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Zerodha"""
        try:
            await asyncio.sleep(0.05)
            return OrderStatus.FILLED
            
        except Exception as e:
            self._record_error()
            logger.error(f"❌ Zerodha order status check failed: {e}")
            return OrderStatus.FAILED
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from Zerodha"""
        try:
            await asyncio.sleep(0.1)
            return []
            
        except Exception as e:
            self._record_error()
            logger.error(f"❌ Zerodha positions retrieval failed: {e}")
            return []
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance from Zerodha"""
        try:
            await asyncio.sleep(0.1)
            return {
                'cash': 75000.0,
                'margin': 150000.0,
                'available': 70000.0
            }
            
        except Exception as e:
            self._record_error()
            logger.error(f"❌ Zerodha account balance retrieval failed: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check Zerodha broker health"""
        try:
            await asyncio.sleep(0.1)
            
            if self.error_count > 10:
                self.status = BrokerStatus.FAILED
                return False
            
            self.status = BrokerStatus.ACTIVE
            return True
            
        except Exception as e:
            self._record_error()
            self.status = BrokerStatus.FAILED
            logger.error(f"❌ Zerodha health check failed: {e}")
            return False
    
    def _record_error(self):
        """Record broker error"""
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        if self.error_count > 5:
            self.status = BrokerStatus.FAILED
            logger.warning(f"⚠️ Broker {self.broker_name} marked as failed")

class BrokerFailoverManager:
    """Broker failover manager"""
    
    def __init__(self):
        self.brokers = {}
        self.primary_broker = None
        self.failover_active = False
        self.failover_threshold = 5  # errors in 5 minutes
        self.failover_window = timedelta(minutes=5)
        self.order_routing = {}
        
    def add_broker(self, broker_adapter: IBrokerAdapter, config: BrokerConfig):
        """Add broker to failover manager"""
        self.brokers[config.broker_name] = {
            'adapter': broker_adapter,
            'config': config,
            'error_count': 0,
            'last_error_time': None,
            'status': BrokerStatus.ACTIVE
        }
        
        # Set primary broker if not set
        if self.primary_broker is None:
            self.primary_broker = config.broker_name
        
        logger.info(f"✅ Broker added: {config.broker_name}")
    
    async def place_order(self, order: Order) -> str:
        """Place order with failover logic"""
        try:
            # Try primary broker first
            if self.primary_broker and self._is_broker_available(self.primary_broker):
                try:
                    broker_order_id = await self._place_order_with_broker(self.primary_broker, order)
                    self.order_routing[order.order_id] = self.primary_broker
                    return broker_order_id
                except Exception as e:
                    logger.warning(f"⚠️ Primary broker failed: {e}")
                    self._record_broker_error(self.primary_broker)
            
            # Try fallback brokers
            fallback_brokers = self._get_available_brokers(exclude=self.primary_broker)
            
            for broker_name in fallback_brokers:
                try:
                    broker_order_id = await self._place_order_with_broker(broker_name, order)
                    self.order_routing[order.order_id] = broker_name
                    logger.info(f"🔄 Order routed to fallback broker: {broker_name}")
                    return broker_order_id
                except Exception as e:
                    logger.warning(f"⚠️ Fallback broker {broker_name} failed: {e}")
                    self._record_broker_error(broker_name)
            
            # All brokers failed
            raise Exception("All brokers failed - order cannot be placed")
            
        except Exception as e:
            logger.error(f"❌ Order placement failed with all brokers: {e}")
            raise
    
    async def _place_order_with_broker(self, broker_name: str, order: Order) -> str:
        """Place order with specific broker"""
        broker_info = self.brokers[broker_name]
        adapter = broker_info['adapter']
        
        return await adapter.place_order(order)
    
    def _is_broker_available(self, broker_name: str) -> bool:
        """Check if broker is available"""
        if broker_name not in self.brokers:
            return False
        
        broker_info = self.brokers[broker_name]
        
        # Check if broker is enabled
        if not broker_info['config'].enabled:
            return False
        
        # Check broker status
        if broker_info['status'] == BrokerStatus.FAILED:
            return False
        
        # Check error count in time window
        if broker_info['last_error_time']:
            time_since_error = datetime.now() - broker_info['last_error_time']
            if time_since_error < self.failover_window:
                if broker_info['error_count'] >= self.failover_threshold:
                    return False
        
        return True
    
    def _get_available_brokers(self, exclude: Optional[str] = None) -> List[str]:
        """Get list of available brokers"""
        available_brokers = []
        
        for broker_name, broker_info in self.brokers.items():
            if broker_name == exclude:
                continue
            
            if self._is_broker_available(broker_name):
                available_brokers.append(broker_name)
        
        # Sort by priority
        available_brokers.sort(key=lambda x: self.brokers[x]['config'].priority)
        
        return available_brokers
    
    def _record_broker_error(self, broker_name: str):
        """Record broker error"""
        if broker_name not in self.brokers:
            return
        
        broker_info = self.brokers[broker_name]
        broker_info['error_count'] += 1
        broker_info['last_error_time'] = datetime.now()
        
        # Check if broker should be marked as failed
        if broker_info['error_count'] >= self.failover_threshold:
            broker_info['status'] = BrokerStatus.FAILED
            logger.warning(f"⚠️ Broker {broker_name} marked as failed")
            
            # Activate failover if primary broker failed
            if broker_name == self.primary_broker:
                self.failover_active = True
                logger.warning("🚨 Primary broker failed - failover activated")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with failover logic"""
        try:
            # Get broker that handled the order
            broker_name = self.order_routing.get(order_id)
            if not broker_name:
                logger.error(f"❌ No broker found for order: {order_id}")
                return False
            
            # Try to cancel with the same broker
            if self._is_broker_available(broker_name):
                broker_info = self.brokers[broker_name]
                adapter = broker_info['adapter']
                return await adapter.cancel_order(order_id)
            else:
                logger.warning(f"⚠️ Original broker {broker_name} unavailable for cancellation")
                return False
                
        except Exception as e:
            logger.error(f"❌ Order cancellation failed: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status with failover logic"""
        try:
            # Get broker that handled the order
            broker_name = self.order_routing.get(order_id)
            if not broker_name:
                logger.error(f"❌ No broker found for order: {order_id}")
                return OrderStatus.FAILED
            
            # Try to get status from the same broker
            if self._is_broker_available(broker_name):
                broker_info = self.brokers[broker_name]
                adapter = broker_info['adapter']
                return await adapter.get_order_status(order_id)
            else:
                logger.warning(f"⚠️ Original broker {broker_name} unavailable for status check")
                return OrderStatus.FAILED
                
        except Exception as e:
            logger.error(f"❌ Order status check failed: {e}")
            return OrderStatus.FAILED
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from all brokers"""
        try:
            all_positions = []
            
            for broker_name, broker_info in self.brokers.items():
                if self._is_broker_available(broker_name):
                    try:
                        adapter = broker_info['adapter']
                        positions = await adapter.get_positions()
                        for pos in positions:
                            pos['broker'] = broker_name
                        all_positions.extend(positions)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get positions from {broker_name}: {e}")
            
            return all_positions
            
        except Exception as e:
            logger.error(f"❌ Position retrieval failed: {e}")
            return []
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance from all brokers"""
        try:
            total_balance = {
                'cash': 0.0,
                'margin': 0.0,
                'available': 0.0
            }
            
            for broker_name, broker_info in self.brokers.items():
                if self._is_broker_available(broker_name):
                    try:
                        adapter = broker_info['adapter']
                        balance = await adapter.get_account_balance()
                        for key, value in balance.items():
                            if key in total_balance:
                                total_balance[key] += value
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get balance from {broker_name}: {e}")
            
            return total_balance
            
        except Exception as e:
            logger.error(f"❌ Balance retrieval failed: {e}")
            return {}
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all brokers"""
        try:
            health_status = {}
            
            for broker_name, broker_info in self.brokers.items():
                try:
                    adapter = broker_info['adapter']
                    is_healthy = await adapter.health_check()
                    health_status[broker_name] = is_healthy
                    
                    # Update broker status
                    broker_info['status'] = BrokerStatus.ACTIVE if is_healthy else BrokerStatus.FAILED
                    
                except Exception as e:
                    logger.warning(f"⚠️ Health check failed for {broker_name}: {e}")
                    health_status[broker_name] = False
                    broker_info['status'] = BrokerStatus.FAILED
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {}
    
    def get_failover_status(self) -> Dict[str, Any]:
        """Get failover status"""
        return {
            'primary_broker': self.primary_broker,
            'failover_active': self.failover_active,
            'total_brokers': len(self.brokers),
            'available_brokers': len(self._get_available_brokers()),
            'broker_status': {
                name: {
                    'status': info['status'].value,
                    'error_count': info['error_count'],
                    'last_error_time': info['last_error_time'].isoformat() if info['last_error_time'] else None,
                    'enabled': info['config'].enabled,
                    'priority': info['config'].priority
                }
                for name, info in self.brokers.items()
            }
        }

def main():
    """Main function for testing"""
    async def test_broker_failover():
        # Create broker configurations
        fyers_config = BrokerConfig(
            broker_name="Fyers",
            api_endpoint="https://api.fyers.in",
            credentials={"client_id": "test", "access_token": "test"},
            priority=1
        )
        
        zerodha_config = BrokerConfig(
            broker_name="Zerodha",
            api_endpoint="https://api.kite.trade",
            credentials={"api_key": "test", "access_token": "test"},
            priority=2
        )
        
        # Create broker adapters
        fyers_adapter = FyersBrokerAdapter(fyers_config)
        zerodha_adapter = ZerodhaBrokerAdapter(zerodha_config)
        
        # Create failover manager
        failover_manager = BrokerFailoverManager()
        failover_manager.add_broker(fyers_adapter, fyers_config)
        failover_manager.add_broker(zerodha_adapter, zerodha_config)
        
        # Test order placement
        order = Order(
            order_id="TEST_001",
            symbol="NSE:NIFTY50-INDEX",
            side="BUY",
            quantity=100,
            price=19500,
            order_type="LIMIT",
            timestamp=datetime.now(),
            status=OrderStatus.PENDING
        )
        
        try:
            broker_order_id = await failover_manager.place_order(order)
            print(f"✅ Order placed: {broker_order_id}")
        except Exception as e:
            print(f"❌ Order placement failed: {e}")
        
        # Test health check
        health_status = await failover_manager.health_check_all()
        print(f"📊 Health status: {health_status}")
        
        # Get failover status
        failover_status = failover_manager.get_failover_status()
        print(f"🔄 Failover status: {json.dumps(failover_status, indent=2, default=str)}")
    
    # Run test
    asyncio.run(test_broker_failover())

if __name__ == "__main__":
    main()
