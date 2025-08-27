"""
Base strategy module.
Provides the foundation for all trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from src.models.unified_database import UnifiedDatabase
from src.core.indicators import add_technical_indicators
import sqlite3

class Strategy(ABC):
    """Base class for all trading strategies."""
    
    # Base minimum candles requirement - override in child strategies
    min_candles = 50
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """Initialize the strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.signals = []
        self.db = UnifiedDatabase()
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data meets minimum requirements for analysis.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            bool: True if data is valid for analysis
        """
        if len(data) < self.min_candles:
            logging.warning(f"❌ Insufficient data for {self.name}: {len(data)} < {self.min_candles}")
            return False
        
        # Check for required columns
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            logging.error(f"❌ Missing required columns for {self.name}: {missing}")
            return False
        
        # Check for NaN in last closed candle
        if len(data) >= 2:
            last_closed = data.iloc[-2]
        else:
            last_closed = data.iloc[-1]
            
        if pd.isna(last_closed['close']):
            logging.warning(f"❌ Last closed candle has NaN close for {self.name}")
            return False
        
        return True
    
    def get_closed_candle(self, data: pd.DataFrame) -> pd.Series:
        """Get the last closed candle (iloc[-2]) or fallback to current if insufficient data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            pd.Series: Last closed candle
        """
        if len(data) >= 2:
            return data.iloc[-2]  # Last closed candle
        else:
            return data.iloc[-1]  # Fallback for insufficient data
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and generate trading signals.
        
        IMPORTANT: This method should use closed candles only (iloc[-2]) for live trading.
        For backtesting, future_data can be used for P&L calculation.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            dict: Signal data with analysis results
        """
        pass
    
    def analyze_closed(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze using only closed candles - safe wrapper for live trading.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            dict: Signal data with analysis results
        """
        if not self.validate_data(data):
            return {'signal': 'NO TRADE', 'reason': 'insufficient data'}
        
        try:
            return self.analyze(data)
        except Exception as e:
            logging.error(f"❌ Strategy {self.name} error: {e}")
            return {'signal': 'ERROR', 'reason': str(e)}
    
    def analyze_optimized(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze market data with pre-calculated indicators.
        
        This is an optimized version that assumes indicators are already calculated.
        
        Args:
            data: Market data DataFrame with pre-calculated indicators
            
        Returns:
            dict or None: Trading signal or None if no signal
        """
        # Skip validation since indicators are pre-calculated
        if len(data) < 10:  # Minimum data requirement
            return None
            
        # Get the latest candle
        candle = self.get_closed_candle(data)
        
        # Use the data as-is since indicators are already calculated
        df = data
        
        # Call the strategy-specific analysis
        return self.analyze_strategy(df, candle)
    
    def analyze_strategy(self, df: pd.DataFrame, candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Strategy-specific analysis method.
        
        This method should be overridden by child classes to implement their specific logic.
        
        Args:
            df: Market data with indicators
            candle: Current candle data
            
        Returns:
            dict or None: Trading signal or None if no signal
        """
        # Default implementation - should be overridden
        return None
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate common indicators needed for the strategy.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            pandas.DataFrame: Data with indicators added
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Use the unified indicators function
        df = add_technical_indicators(df)
        
        # Allow child classes to add their own indicators
        return self.add_indicators(df)
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators.
        
        This method should be overridden by child classes to add their own indicators.
        
        Args:
            data: Market data DataFrame with common indicators
            
        Returns:
            pandas.DataFrame: Data with strategy-specific indicators added
        """
        return data
    
    def log_signal(self, symbol: str, signal_data: Dict[str, Any]) -> None:
        """Log a trading signal to the database.
        
        Args:
            symbol: Trading symbol
            signal_data: Signal data
        """
        # Add the signal to the in-memory list
        self.signals.append(signal_data)
        
        # Use the unified database to log the signal
        try:
            signal_data_with_symbol = {
                'symbol': symbol,
                **signal_data
            }
            self.db.log_strategy_trade(self.name, signal_data_with_symbol)
        except Exception as e:
            logging.error(f"Error logging signal: {e}")
    
    def format_signal(self, signal: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format a signal for output and logging.
        
        Args:
            signal: Signal type (e.g., "BUY", "SELL")
            data: Additional signal data
            
        Returns:
            dict: Formatted signal data
        """
        return {
            'signal': signal,
            'strategy': self.name,
            'timestamp': pd.Timestamp.now(),
            **data
        }
    
    def backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run a backtest on historical data.
        
        Args:
            data: Historical market data
            
        Returns:
            dict: Backtest results
        """
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Reset signals
        self.signals = []
        
        # Analyze each row to generate signals
        for i in range(len(df)):
            # Skip the first few rows as indicators need data to calculate
            if i < 50:
                continue
                
            # Get data up to this point
            current_data = df.iloc[:i+1]
            
            # Analyze and generate signals
            signal = self.analyze(current_data)
            
            # If a signal was generated, log it
            if signal and signal.get('signal') != 'None':
                # Use the actual timestamp from the data instead of current time
                if current_data.index.name == 'time' or 'time' in current_data.columns:
                    # If time is the index or a column
                    if current_data.index.name == 'time':
                        # Time is the index
                        signal['signal_time'] = current_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        # Time is a column
                        signal['signal_time'] = current_data.iloc[-1]['time'].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Default to index value if no explicit time column
                    signal['signal_time'] = current_data.index[-1]
                
                self.signals.append(signal)
        
        # Calculate performance metrics
        return self.calculate_performance()
    
    def calculate_performance(self) -> Dict[str, Any]:
        """Calculate performance metrics from backtest signals.
        
        Returns:
            dict: Performance metrics
        """
        if not self.signals:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_profit': 0,
                'max_drawdown': 0
            }
        
        # Basic metrics
        wins = sum(1 for s in self.signals if s.get('outcome') == 'Win')
        losses = sum(1 for s in self.signals if s.get('outcome') == 'Loss')
        total_trades = len(self.signals)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_profit = sum(s.get('pnl', 0) for s in self.signals if s.get('pnl', 0) > 0)
        total_loss = abs(sum(s.get('pnl', 0) for s in self.signals if s.get('pnl', 0) < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        avg_profit = sum(s.get('pnl', 0) for s in self.signals) / total_trades if total_trades > 0 else 0
        
        # Drawdown calculation
        equity_curve = []
        balance = 0
        peak = 0
        drawdown = 0
        max_drawdown = 0
        
        for signal in self.signals:
            pnl = signal.get('pnl', 0)
            balance += pnl
            equity_curve.append(balance)
            peak = max(peak, balance)
            drawdown = peak - balance
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'max_drawdown': max_drawdown,
            'signals': self.signals,
            'equity_curve': equity_curve
        }

    def log_signals_batch(self, signals: list) -> None:
        """Batch log trading signals to the database."""
        for signal_data in signals:
            self.log_signal(signal_data['symbol'], signal_data) 