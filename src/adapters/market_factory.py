"""
Market factory for creating market instances.
"""

from typing import Dict, Type
from src.adapters.market_interface import MarketInterface, MarketType
from src.markets.indian.indian_market import IndianMarket
from src.markets.crypto.crypto_market import CryptoMarket


class MarketFactory:
    """Factory for creating market instances."""
    
    _markets: Dict[MarketType, Type[MarketInterface]] = {
        MarketType.INDIAN_STOCKS: IndianMarket,
        MarketType.CRYPTO: CryptoMarket,
    }
    
    @classmethod
    def create_market(cls, market_type: MarketType) -> MarketInterface:
        """Create a market instance."""
        if market_type not in cls._markets:
            raise ValueError(f"Unsupported market type: {market_type}")
        
        market_class = cls._markets[market_type]
        return market_class()
    
    @classmethod
    def get_supported_markets(cls) -> list[MarketType]:
        """Get list of supported market types."""
        return list(cls._markets.keys())
    
    @classmethod
    def register_market(cls, market_type: MarketType, market_class: Type[MarketInterface]):
        """Register a new market type."""
        cls._markets[market_type] = market_class
