"""
Crypto market implementation for cryptocurrency trading.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.adapters.market_interface import (
    MarketInterface, MarketConfig, MarketType, AssetType, Contract
)


class CryptoMarket(MarketInterface):
    """Crypto market implementation - 24/7 trading."""
    
    def __init__(self):
        config = MarketConfig(
            market_type=MarketType.CRYPTO,
            timezone="UTC",  # Crypto markets are global
            trading_hours={"start": "00:00", "end": "23:59"},  # 24/7
            trading_days=[0, 1, 2, 3, 4, 5, 6],  # All days
            lot_sizes={
                "BTCUSDT": 0.001,
                "ETHUSDT": 0.01,
                "BNBUSDT": 0.1,
                "ADAUSDT": 1.0,
                "SOLUSDT": 0.1,
                "DOTUSDT": 0.1,
                "MATICUSDT": 1.0,
                "AVAXUSDT": 0.1
            },
            tick_sizes={
                "BTCUSDT": 0.01,
                "ETHUSDT": 0.01,
                "BNBUSDT": 0.01,
                "ADAUSDT": 0.0001,
                "SOLUSDT": 0.01,
                "DOTUSDT": 0.001,
                "MATICUSDT": 0.0001,
                "AVAXUSDT": 0.01
            },
            commission_rates={
                "BTCUSDT": 0.001,  # 0.1% maker/taker
                "ETHUSDT": 0.001,
                "BNBUSDT": 0.001,
                "ADAUSDT": 0.001,
                "SOLUSDT": 0.001,
                "DOTUSDT": 0.001,
                "MATICUSDT": 0.001,
                "AVAXUSDT": 0.001
            },
            margin_requirements={
                "BTCUSDT": 0.05,  # 5% margin for crypto
                "ETHUSDT": 0.05,
                "BNBUSDT": 0.05,
                "ADAUSDT": 0.05,
                "SOLUSDT": 0.05,
                "DOTUSDT": 0.05,
                "MATICUSDT": 0.05,
                "AVAXUSDT": 0.05
            },
            currency="USDT"
        )
        super().__init__(config)
        self.tz = ZoneInfo("UTC")
    
    def is_market_open(self, timestamp: Optional[datetime] = None) -> bool:
        # Crypto markets are always open
        return True
    
    def get_trading_hours(self) -> Dict[str, str]:
        return self.config.trading_hours
    
    def get_contract_info(self, symbol: str) -> Optional[Contract]:
        return Contract(
            symbol=symbol,
            underlying=symbol,
            asset_type=AssetType.CRYPTO,
            lot_size=self.get_lot_size(symbol),
            tick_size=self.get_tick_size(symbol),
            margin_requirement=self.get_margin_requirement(symbol),
            commission_rate=self.get_commission_rate(symbol)
        )
    
    def get_lot_size(self, symbol: str) -> float:
        return self.config.lot_sizes.get(symbol, 0.001)
    
    def get_tick_size(self, symbol: str) -> float:
        return self.config.tick_sizes.get(symbol, 0.01)
    
    def get_commission_rate(self, symbol: str) -> float:
        return self.config.commission_rates.get(symbol, 0.001)
    
    def get_margin_requirement(self, symbol: str) -> float:
        return self.config.margin_requirements.get(symbol, 0.05)
    
    def normalize_symbol(self, symbol: str) -> str:
        # Ensure proper crypto format (e.g., BTCUSDT)
        symbol = symbol.upper()
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        return symbol
    
    def validate_symbol(self, symbol: str) -> bool:
        return symbol in self.config.lot_sizes
    
    def get_data_provider(self):
        """Get the data provider for this market."""
        from src.adapters.data.crypto_data_provider import CryptoDataProvider
        return CryptoDataProvider()
    
    def get_default_symbols(self):
        """Get default symbols for crypto trading."""
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
