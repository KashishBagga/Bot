#!/usr/bin/env python3
"""
BaseDataProvider — Formal interface above the indicator pipeline.
=================================================================
All data providers implement this interface. The IndicatorPipeline
and trader loop depend only on this interface — never on a concrete
implementation.

Implementations:
    FyersDataProvider     — Live market data via Fyers API (src/adapters/data/fyers_data_provider.py)
    CSVReplayProvider     — Replay from exported CSV files (future)
    DatabaseReplayProvider — Replay from TimescaleDB option_snapshots (future)

Adding a new data source:
    1. Subclass BaseDataProvider
    2. Implement get_historical_data() and get_current_price()
    3. Pass the new instance to IndicatorPipeline / trader
    No other changes needed.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class BaseDataProvider(ABC):
    """
    Abstract data provider interface.
    The trader loop and pipeline depend only on this — never on FyersDataProvider directly.
    """

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        resolution: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol:     Market symbol, e.g. "NSE:NIFTY50-INDEX"
            start_date: Inclusive start datetime
            end_date:   Inclusive end datetime
            resolution: Candle interval: "1", "5", "15", "60", "D"

        Returns:
            pd.DataFrame with columns [open, high, low, close, volume]
            and a timezone-aware DatetimeIndex (Asia/Kolkata).
            None if data is unavailable or an error occurs.
        """
        ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest traded price (LTP) for a symbol.

        Returns:
            float LTP, or None if unavailable.
        """
        ...
