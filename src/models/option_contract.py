#!/usr/bin/env python3
"""
Option Contract Models
Core classes for options trading infrastructure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

class OptionType(Enum):
    CALL = "CE"
    PUT = "PE"

class StrikeSelection(Enum):
    ATM = "atm"  # At-the-money
    ITM = "itm"  # In-the-money
    OTM = "otm"  # Out-of-the-money
    DELTA = "delta"  # Based on delta target

@dataclass
class OptionContract:
    """Represents an option contract with all necessary details."""
    
    # Contract details
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: OptionType
    lot_size: int
    
    # Market data
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    
    # Greeks (computed)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    def __post_init__(self):
        """Validate contract details after initialization."""
        if self.strike <= 0:
            raise ValueError(f"Invalid strike price: {self.strike}")
        if self.lot_size <= 0:
            raise ValueError(f"Invalid lot size: {self.lot_size}")
        
        # Don't raise on expired contracts for historical/backtest data
        # Instead warn for near-term or past expiries only if this is a live contract
        try:
            now = datetime.now(self.expiry.tzinfo) if self.expiry.tzinfo else datetime.utcnow()
        except Exception:
            now = datetime.utcnow()
        
        # Optional warning, not exception for historical data
        if self.expiry < now - timedelta(days=1):
            logging.warning(f"Initialized expired contract (historical): {self.symbol} exp {self.expiry}")
        
        # Normalize Greeks to reasonable ranges
        self.normalize_greeks()
    
    def normalize_greeks(self):
        """Normalize Greeks to reasonable ranges."""
        # Clamp delta to [-1, 1]
        if hasattr(self, 'delta') and self.delta is not None:
            self.delta = max(min(self.delta, 1.0), -1.0)
        
        # Ensure gamma is non-negative
        if hasattr(self, 'gamma') and self.gamma is not None:
            self.gamma = max(self.gamma, 0.0)
        
        # Theta can be negative (time decay)
        if hasattr(self, 'theta') and self.theta is not None:
            # No clamping needed for theta
            pass
        
        # Vega should be positive (volatility sensitivity)
        if hasattr(self, 'vega') and self.vega is not None:
            self.vega = max(self.vega, 0.0)
    
    @property
    def mid_price(self) -> float:
        """Get mid price (bid + ask) / 2."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last if self.last > 0 else 0.0
    
    @property
    def ltp(self) -> float:
        """Alias for last traded price."""
        return self.last
    
    @property
    def last_price(self) -> float:
        """Alias for last traded price."""
        return self.last
    
    @property
    def spread(self) -> float:
        """Get bid-ask spread."""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0
    
    @property
    def days_to_expiry(self) -> float:
        """Get days to expiry (fractional, may be negative for expired contracts)."""
        try:
            now = datetime.now(self.expiry.tzinfo) if self.expiry.tzinfo else datetime.utcnow()
        except Exception:
            now = datetime.utcnow()
        return (self.expiry - now).total_seconds() / 86400.0  # may be negative for expired contracts
    
    def is_atm(self, underlying_price: float, tolerance: float = 50.0) -> bool:
        """Check if option is at-the-money."""
        return abs(self.strike - underlying_price) <= float(tolerance)
    
    def is_itm(self, underlying_price: float) -> bool:
        """Check if option is in-the-money."""
        if self.option_type == OptionType.CALL:
            return self.strike < underlying_price
        else:  # PUT
            return self.strike > underlying_price
    
    def is_otm(self, underlying_price: float) -> bool:
        """Check if option is out-of-the-money."""
        if self.option_type == OptionType.CALL:
            return self.strike > underlying_price
        else:  # PUT
            return self.strike < underlying_price
    
    def get_moneyness(self, underlying_price: float) -> float:
        """Get moneyness (strike/underlying ratio)."""
        return self.strike / underlying_price
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'underlying': self.underlying,
            'strike': self.strike,
            'expiry': self.expiry.isoformat(),
            'option_type': self.option_type.value,
            'lot_size': self.lot_size,
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'implied_volatility': self.implied_volatility,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OptionContract':
        """Create from dictionary with timezone-safe expiry handling."""
        # Handle expiry with timezone safety
        expiry = pd.to_datetime(data['expiry'])
        if expiry.tzinfo is None and 'tz' in data:
            # Optional tz handling
            expiry = expiry.tz_localize(data['tz'])
        
        return cls(
            symbol=data['symbol'],
            underlying=data['underlying'],
            strike=float(data['strike']),
            expiry=expiry.to_pydatetime(),
            option_type=OptionType(data['option_type']),
            lot_size=int(data['lot_size']),
            bid=float(data.get('bid', 0)),
            ask=float(data.get('ask', 0)),
            last=float(data.get('last', 0)),
            volume=int(data.get('volume', 0)),
            open_interest=int(data.get('open_interest', 0)),
            implied_volatility=float(data.get('implied_volatility', 0)),
            delta=float(data.get('delta', 0)),
            gamma=float(data.get('gamma', 0)),
            theta=float(data.get('theta', 0)),
            vega=float(data.get('vega', 0))
        )

class OptionChain:
    """Represents a complete option chain for an underlying."""
    
    def __init__(self, underlying: str, timestamp: datetime):
        self.underlying = underlying
        self.timestamp = timestamp
        self.contracts: List[OptionContract] = []
        self.underlying_price: float = 0.0
    
    def add_contract(self, contract: OptionContract):
        """Add a contract to the chain."""
        self.contracts.append(contract)
    
    def get_contracts_by_expiry(self, expiry: datetime) -> List[OptionContract]:
        """Get all contracts for a specific expiry (timezone-safe comparison)."""
        # Compare by date only to avoid timezone issues
        target_date = pd.to_datetime(expiry).date()
        return [c for c in self.contracts if pd.to_datetime(c.expiry).date() == target_date]
    
    def get_contracts_by_type(self, option_type: OptionType) -> List[OptionContract]:
        """Get all contracts of a specific type."""
        return [c for c in self.contracts if c.option_type == option_type]
    
    def get_atm_contracts(self, option_type: OptionType, expiry: datetime, underlying_price: Optional[float] = None) -> List[OptionContract]:
        """Get at-the-money contracts for given type and expiry."""
        contracts = [c for c in self.contracts 
                    if c.option_type == option_type and c.expiry == expiry]
        
        if underlying_price is None:
            underlying_price = self.underlying_price
        
        if not contracts or not underlying_price:
            return []
        
        # Find closest to ATM
        atm_contracts = []
        min_distance = float('inf')
        
        for contract in contracts:
            distance = abs(contract.strike - underlying_price)
            if distance < min_distance:
                min_distance = distance
                atm_contracts = [contract]
            elif distance == min_distance:
                atm_contracts.append(contract)
        
        return atm_contracts
    
    def get_contract_by_delta(self, option_type: OptionType, expiry: datetime, 
                            target_delta: float, tolerance: float = 0.05) -> Optional[OptionContract]:
        """Get contract closest to target delta (returns closest even if outside tolerance)."""
        contracts = [c for c in self.contracts 
                    if c.option_type == option_type and c.expiry == expiry and c.delta is not None]
        
        if not contracts:
            return None
        
        # Find contract with delta closest to target
        best_contract = min(contracts, key=lambda c: abs(abs(c.delta) - target_delta))
        
        # Log if outside tolerance but return closest anyway
        delta_diff = abs(abs(best_contract.delta) - target_delta)
        if delta_diff > tolerance:
            logger.debug(f"Delta out of tolerance ({delta_diff:.3f} > {tolerance}); returning closest anyway.")
        
        return best_contract
    
    def get_liquid_contracts(self, min_oi: int = 100, min_volume: int = 10) -> List[OptionContract]:
        """Get contracts with sufficient liquidity."""
        return [c for c in self.contracts 
                if c.open_interest >= min_oi and c.volume >= min_volume]
    
    def nearest_strikes(self, underlying_price: float, k: int = 20) -> List[OptionContract]:
        """Get k nearest strikes to underlying price."""
        return sorted(self.contracts, key=lambda c: abs(c.strike - underlying_price))[:k]
    
    def top_by_oi(self, n: int = 10) -> List[OptionContract]:
        """Get top n contracts by open interest."""
        return sorted(self.contracts, key=lambda c: c.open_interest or 0, reverse=True)[:n]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        if not self.contracts:
            return pd.DataFrame()
        
        data = [contract.to_dict() for contract in self.contracts]
        df = pd.DataFrame(data)
        df['expiry'] = pd.to_datetime(df['expiry'])
        df['underlying_price'] = self.underlying_price
        df['timestamp'] = self.timestamp
        
        return df

class OptionOrder:
    """Represents an option order with execution details."""
    
    def __init__(self, contract: OptionContract, quantity: int, 
                 order_type: str = "MARKET", price: float = 0.0):
        self.contract = contract
        self.quantity = quantity  # Number of lots
        self.order_type = order_type
        self.price = price  # Limit price if applicable
        self.executed_price = 0.0
        self.executed_quantity = 0
        self.status = "PENDING"
        self.timestamp = datetime.now()
    
    @property
    def total_quantity(self) -> int:
        """Get total quantity in shares (lots * lot_size)."""
        return self.quantity * self.contract.lot_size
    
    @property
    def notional_value(self) -> float:
        """Get notional value of the order."""
        if self.executed_price > 0:
            return self.executed_price * self.total_quantity
        return self.price * self.total_quantity if self.price > 0 else 0.0
    
    def execute(self, price: float, quantity: int = None, timestamp: datetime = None):
        """Execute the order with optional timestamp."""
        self.executed_price = price
        self.executed_quantity = quantity if quantity else self.quantity
        self.status = "EXECUTED"
        if timestamp:
            self.timestamp = timestamp
        else:
            # Use timezone-aware timestamp if possible
            try:
                self.timestamp = datetime.now().astimezone()
            except Exception:
                self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'contract_symbol': self.contract.symbol,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'price': self.price,
            'executed_price': self.executed_price,
            'executed_quantity': self.executed_quantity,
            'status': self.status,
            'timestamp': self.timestamp.isoformat(),
            'total_quantity': self.total_quantity,
            'notional_value': self.notional_value
        } 