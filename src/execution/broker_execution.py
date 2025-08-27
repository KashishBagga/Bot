#!/usr/bin/env python3
"""
Broker Execution Layer
Handles real order placement and management for live trading
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.option_contract import OptionContract, OptionType

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"


@dataclass
class OrderRequest:
    """Order request structure."""
    symbol: str
    quantity: int
    side: OrderSide
    order_type: OrderType
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    product: str = "NRML"
    validity: str = "DAY"
    disclosed_quantity: Optional[int] = None
    tag: Optional[str] = None


@dataclass
class OrderResponse:
    """Order response structure."""
    order_id: str
    status: OrderStatus
    filled_quantity: int = 0
    average_price: float = 0.0
    message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BrokerAPI(ABC):
    """Abstract base class for broker APIs."""
    
    @abstractmethod
    def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place an order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get order status."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def get_margins(self) -> Dict:
        """Get margin information."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass


class ZerodhaBrokerAPI(BrokerAPI):
    """Zerodha Kite Connect broker API."""
    
    def __init__(self, api_key: str, api_secret: str, access_token: str = None):
        """Initialize Zerodha broker API."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.kite = None
        self.connected = False
        
        try:
            from kiteconnect import KiteConnect
            self.KiteConnect = KiteConnect
        except ImportError:
            logger.error("❌ KiteConnect not installed. Install with: pip install kiteconnect")
            raise
    
    def connect(self) -> bool:
        """Connect to Zerodha API."""
        try:
            self.kite = self.KiteConnect(api_key=self.api_key)
            
            if self.access_token:
                self.kite.set_access_token(self.access_token)
                self.connected = True
                logger.info("✅ Connected to Zerodha API")
                return True
            else:
                # Generate login URL for manual authentication
                login_url = self.kite.login_url()
                logger.info(f"🔗 Please login at: {login_url}")
                logger.info("After login, call set_access_token() with the token")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Zerodha: {e}")
            return False
    
    def set_access_token(self, access_token: str):
        """Set access token after manual login."""
        self.access_token = access_token
        if self.kite:
            self.kite.set_access_token(access_token)
            self.connected = True
            logger.info("✅ Access token set successfully")
    
    def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place an order on Zerodha."""
        if not self.connected or not self.kite:
            return OrderResponse(
                order_id="",
                status=OrderStatus.REJECTED,
                message="Not connected to broker"
            )
        
        try:
            # Convert order request to Zerodha format
            kite_order = {
                "tradingsymbol": order_request.symbol,
                "quantity": order_request.quantity,
                "transaction_type": order_request.side.value,
                "order_type": order_request.order_type.value,
                "product": order_request.product,
                "validity": order_request.validity
            }
            
            if order_request.price:
                kite_order["price"] = order_request.price
            
            if order_request.trigger_price:
                kite_order["trigger_price"] = order_request.trigger_price
            
            if order_request.disclosed_quantity:
                kite_order["disclosed_quantity"] = order_request.disclosed_quantity
            
            if order_request.tag:
                kite_order["tag"] = order_request.tag
            
            # Place order
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                **kite_order
            )
            
            logger.info(f"✅ Order placed successfully: {order_id}")
            
            return OrderResponse(
                order_id=str(order_id),
                status=OrderStatus.PENDING,
                message="Order placed successfully"
            )
            
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            return OrderResponse(
                order_id="",
                status=OrderStatus.REJECTED,
                message=str(e)
            )
    
    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get order status from Zerodha."""
        if not self.connected or not self.kite:
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message="Not connected to broker"
            )
        
        try:
            # Get order history
            orders = self.kite.orders()
            
            # Find the specific order
            for order in orders:
                if str(order['order_id']) == order_id:
                    # Map Zerodha status to our enum
                    status_map = {
                        'PENDING': OrderStatus.PENDING,
                        'COMPLETE': OrderStatus.COMPLETE,
                        'REJECTED': OrderStatus.REJECTED,
                        'CANCELLED': OrderStatus.CANCELLED,
                        'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED
                    }
                    
                    status = status_map.get(order['status'], OrderStatus.PENDING)
                    
                    return OrderResponse(
                        order_id=order_id,
                        status=status,
                        filled_quantity=int(order.get('filled_quantity', 0)),
                        average_price=float(order.get('average_price', 0)),
                        message=order.get('status_message', ''),
                        timestamp=datetime.now()
                    )
            
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message="Order not found"
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting order status: {e}")
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message=str(e)
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order on Zerodha."""
        if not self.connected or not self.kite:
            return False
        
        try:
            self.kite.cancel_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=order_id
            )
            logger.info(f"✅ Order cancelled successfully: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cancelling order: {e}")
            return False
    
    def get_positions(self) -> List[Dict]:
        """Get current positions from Zerodha."""
        if not self.connected or not self.kite:
            return []
        
        try:
            positions = self.kite.positions()
            
            # Format positions
            formatted_positions = []
            for position in positions.get('net', []):
                if position['quantity'] != 0:  # Only non-zero positions
                    formatted_positions.append({
                        'symbol': position['tradingsymbol'],
                        'quantity': position['quantity'],
                        'average_price': position['average_price'],
                        'pnl': position['pnl'],
                        'product': position['product']
                    })
            
            return formatted_positions
            
        except Exception as e:
            logger.error(f"❌ Error getting positions: {e}")
            return []
    
    def get_margins(self) -> Dict:
        """Get margin information from Zerodha."""
        if not self.connected or not self.kite:
            return {}
        
        try:
            margins = self.kite.margins()
            return {
                'equity': margins.get('equity', {}),
                'commodity': margins.get('commodity', {}),
                'net': margins.get('net', {})
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting margins: {e}")
            return {}
    
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        return self.connected


class PaperBrokerAPI(BrokerAPI):
    """Paper trading broker API for testing."""
    
    def __init__(self):
        """Initialize paper trading broker."""
        self.orders = {}
        self.positions = {}
        self.order_counter = 0
        self.connected = True
    
    def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place a paper order."""
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter}"
        
        # Simulate order processing
        time.sleep(0.1)  # Simulate network delay
        
        # Simulate fill (always fill at requested price or market price)
        fill_price = order_request.price or 100.0  # Default price for paper trading
        filled_quantity = order_request.quantity
        
        # Update positions
        symbol = order_request.symbol
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        if order_request.side == OrderSide.BUY:
            self.positions[symbol] += filled_quantity
        else:
            self.positions[symbol] -= filled_quantity
        
        # Store order
        self.orders[order_id] = {
            'order_request': order_request,
            'status': OrderStatus.COMPLETE,
            'filled_quantity': filled_quantity,
            'average_price': fill_price,
            'timestamp': datetime.now()
        }
        
        logger.info(f"✅ Paper order placed: {order_id} - {order_request.side.value} {filled_quantity} {symbol} @ {fill_price}")
        
        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.COMPLETE,
            filled_quantity=filled_quantity,
            average_price=fill_price,
            message="Paper order filled",
            timestamp=datetime.now()
        )
    
    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get paper order status."""
        if order_id not in self.orders:
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message="Order not found"
            )
        
        order = self.orders[order_id]
        return OrderResponse(
            order_id=order_id,
            status=order['status'],
            filled_quantity=order['filled_quantity'],
            average_price=order['average_price'],
            message="Paper order",
            timestamp=order['timestamp']
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order."""
        if order_id in self.orders:
            self.orders[order_id]['status'] = OrderStatus.CANCELLED
            logger.info(f"✅ Paper order cancelled: {order_id}")
            return True
        return False
    
    def get_positions(self) -> List[Dict]:
        """Get paper positions."""
        positions = []
        for symbol, quantity in self.positions.items():
            if quantity != 0:
                positions.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'average_price': 100.0,  # Default for paper trading
                    'pnl': 0.0,
                    'product': 'NRML'
                })
        return positions
    
    def get_margins(self) -> Dict:
        """Get paper margins."""
        return {
            'equity': {
                'available': 1000000.0,
                'used': 0.0,
                'net': 1000000.0
            },
            'net': {
                'available': 1000000.0,
                'used': 0.0,
                'net': 1000000.0
            }
        }
    
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        return self.connected


class BrokerExecution:
    """Main broker execution class."""
    
    def __init__(self, broker_api: BrokerAPI, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize broker execution."""
        self.broker_api = broker_api
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.order_history = []
        self.slippage_tracker = {}
        
        logger.info("🎯 Broker Execution initialized")
    
    def place_option_order(self, contract: OptionContract, quantity: int, side: OrderSide, 
                          order_type: OrderType = OrderType.MARKET, price: Optional[float] = None,
                          stop_loss: Optional[float] = None, target: Optional[float] = None) -> OrderResponse:
        """Place an option order with retry logic."""
        
        # Create order request
        order_request = OrderRequest(
            symbol=contract.symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            price=price,
            product="NRML"
        )
        
        # Add stop loss and target if provided
        if stop_loss:
            order_request.trigger_price = stop_loss
        
        # Place order with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.broker_api.place_order(order_request)
                
                if response.status != OrderStatus.REJECTED:
                    # Track slippage
                    self._track_slippage(contract, side, response.average_price, price)
                    
                    # Store in history
                    self.order_history.append({
                        'order_id': response.order_id,
                        'contract': contract,
                        'quantity': quantity,
                        'side': side,
                        'order_type': order_type,
                        'requested_price': price,
                        'filled_price': response.average_price,
                        'status': response.status,
                        'timestamp': response.timestamp
                    })
                    
                    logger.info(f"✅ Option order placed: {response.order_id}")
                    return response
                else:
                    logger.warning(f"⚠️ Order rejected (attempt {attempt + 1}): {response.message}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        
            except Exception as e:
                logger.error(f"❌ Error placing order (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # All retries failed
        logger.error(f"❌ Failed to place order after {self.max_retries} attempts")
        return OrderResponse(
            order_id="",
            status=OrderStatus.REJECTED,
            message="Max retries exceeded"
        )
    
    def place_stop_loss_order(self, contract: OptionContract, quantity: int, side: OrderSide,
                             stop_price: float) -> OrderResponse:
        """Place a stop loss order."""
        return self.place_option_order(
            contract=contract,
            quantity=quantity,
            side=side,
            order_type=OrderType.STOP_LOSS_MARKET,
            price=stop_price
        )
    
    def place_target_order(self, contract: OptionContract, quantity: int, side: OrderSide,
                          target_price: float) -> OrderResponse:
        """Place a target order."""
        return self.place_option_order(
            contract=contract,
            quantity=quantity,
            side=side,
            order_type=OrderType.LIMIT,
            price=target_price
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        return self.broker_api.cancel_order(order_id)
    
    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get order status."""
        return self.broker_api.get_order_status(order_id)
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        return self.broker_api.get_positions()
    
    def get_margins(self) -> Dict:
        """Get margin information."""
        return self.broker_api.get_margins()
    
    def check_margin_before_order(self, contract: OptionContract, quantity: int, 
                                 side: OrderSide, price: Optional[float] = None) -> bool:
        """Check if we have sufficient margin before placing order."""
        try:
            margins = self.get_margins()
            
            # Calculate required margin
            if side == OrderSide.BUY:
                # For buying options, we need the premium amount
                required_margin = (price or contract.ask) * quantity
            else:
                # For selling options, we need margin based on underlying value
                # This is a simplified calculation
                required_margin = contract.strike * quantity * 0.15  # 15% margin
            
            available_margin = margins.get('net', {}).get('available', 0)
            
            if required_margin <= available_margin:
                logger.info(f"✅ Sufficient margin: Required ₹{required_margin:,.2f}, Available ₹{available_margin:,.2f}")
                return True
            else:
                logger.warning(f"⚠️ Insufficient margin: Required ₹{required_margin:,.2f}, Available ₹{available_margin:,.2f}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking margin: {e}")
            return False
    
    def _track_slippage(self, contract: OptionContract, side: OrderSide, 
                       filled_price: float, requested_price: Optional[float]):
        """Track slippage for analysis."""
        if requested_price:
            slippage = (filled_price - requested_price) / requested_price * 100
            
            key = f"{contract.symbol}_{side.value}"
            if key not in self.slippage_tracker:
                self.slippage_tracker[key] = []
            
            self.slippage_tracker[key].append(slippage)
            
            logger.info(f"📊 Slippage tracked: {slippage:+.2f}% for {key}")
    
    def get_slippage_stats(self) -> Dict:
        """Get slippage statistics."""
        stats = {}
        for key, slippages in self.slippage_tracker.items():
            if slippages:
                stats[key] = {
                    'count': len(slippages),
                    'avg_slippage': sum(slippages) / len(slippages),
                    'max_slippage': max(slippages),
                    'min_slippage': min(slippages)
                }
        return stats
    
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        return self.broker_api.is_connected()


# Factory function to create broker APIs
def create_broker_api(broker_type: str, **kwargs) -> BrokerAPI:
    """Create broker API based on type."""
    if broker_type.lower() == 'zerodha':
        return ZerodhaBrokerAPI(**kwargs)
    elif broker_type.lower() == 'paper':
        return PaperBrokerAPI()
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")


# Example usage
if __name__ == "__main__":
    # Example: Create paper broker for testing
    paper_broker = create_broker_api('paper')
    execution = BrokerExecution(paper_broker)
    
    # Example: Create a test option contract
    from src.models.option_contract import OptionContract, OptionType
    from datetime import datetime, timedelta
    
    contract = OptionContract(
        symbol="NIFTY25AUG25000CE",
        underlying="NSE:NIFTY50-INDEX",
        strike=25000,
        expiry=datetime.now() + timedelta(days=7),
        option_type=OptionType.CALL,
        lot_size=50,
        bid=100,
        ask=110,
        last=105
    )
    
    # Place a test order
    response = execution.place_option_order(
        contract=contract,
        quantity=50,  # 1 lot
        side=OrderSide.BUY,
        order_type=OrderType.MARKET
    )
    
    print(f"Order response: {response}")
    
    # Get positions
    positions = execution.get_positions()
    print(f"Positions: {positions}")
    
    # Get slippage stats
    slippage_stats = execution.get_slippage_stats()
    print(f"Slippage stats: {slippage_stats}") 