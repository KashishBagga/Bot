"""
Core market interface for multi-asset trading system.
Defines the contract that all market implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class MarketType(Enum):
    """Supported market types."""
    INDIAN_STOCKS = "indian_stocks"
    CRYPTO = "crypto"
    US_STOCKS = "us_stocks"
    FOREX = "forex"
    COMMODITIES = "commodities"


class AssetType(Enum):
    """Supported asset types."""
    STOCK = "stock"
    OPTION = "option"
    FUTURE = "future"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"


@dataclass
class MarketConfig:
    """Market configuration."""
    market_type: MarketType
    timezone: str
    trading_hours: Dict[str, str]  # {"start": "09:15", "end": "15:30"}
    trading_days: List[int]  # 0=Monday, 6=Sunday
    lot_sizes: Dict[str, int]  # Symbol -> lot size mapping
    tick_sizes: Dict[str, float]  # Symbol -> tick size mapping
    commission_rates: Dict[str, float]  # Symbol -> commission rate
    margin_requirements: Dict[str, float]  # Symbol -> margin requirement
    currency: str  # Base currency for the market


@dataclass
class Contract:
    """Generic contract representation."""
    symbol: str
    underlying: str
    asset_type: AssetType
    expiry: Optional[datetime] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # "CALL" or "PUT"
    lot_size: int = 1
    tick_size: float = 0.01
    margin_requirement: float = 0.0
    commission_rate: float = 0.001


@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None


class MarketInterface(ABC):
    """Abstract base class for market implementations."""
    
    def __init__(self, config: MarketConfig):
        self.config = config
        self.market_type = config.market_type
    
    @abstractmethod
    def is_market_open(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if market is open at given timestamp."""
        pass
    
    @abstractmethod
    def get_trading_hours(self) -> Dict[str, str]:
        """Get trading hours for the market."""
        pass
    
    @abstractmethod
    def get_contract_info(self, symbol: str) -> Optional[Contract]:
        """Get contract information for a symbol."""
        pass
    
    @abstractmethod
    def get_lot_size(self, symbol: str) -> int:
        """Get lot size for a symbol."""
        pass
    
    @abstractmethod
    def get_tick_size(self, symbol: str) -> float:
        """Get tick size for a symbol."""
        pass
    
    @abstractmethod
    def get_commission_rate(self, symbol: str) -> float:
        """Get commission rate for a symbol."""
        pass
    
    @abstractmethod
    def get_margin_requirement(self, symbol: str) -> float:
        """Get margin requirement for a symbol."""
        pass
    
    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for the market."""
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists in the market."""
        pass


class DataProviderInterface(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, resolution: str) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol."""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        pass
    
    @abstractmethod
    def get_option_chain(self, underlying: str) -> Optional[Dict]:
        """Get option chain for underlying (if applicable)."""
        pass
    
    @abstractmethod
    def get_contracts(self, underlying: str) -> List[Contract]:
        """Get available contracts for underlying."""
        pass


class ExecutionInterface(ABC):
    """Abstract base class for execution providers."""
    
    @abstractmethod
    def place_order(self, contract: Contract, quantity: int, 
                   order_type: str, price: Optional[float] = None) -> str:
        """Place an order."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status."""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """Get account information."""
        pass
