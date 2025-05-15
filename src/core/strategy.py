"""
Base strategy module.
Provides the foundation for all trading strategies.
"""
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.core.indicators import indicators
from src.models.database import db
import sqlite3

class Strategy(ABC):
    """Base strategy class that all strategies should inherit from."""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """Initialize the strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.signals = []
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and generate trading signals.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            dict: Signal data with analysis results
        """
        pass
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate common indicators needed for the strategy.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            pandas.DataFrame: Data with indicators added
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate indicators without caching
        rsi_period = self.params.get('rsi_period', 14)
        df['rsi'] = indicators.rsi(df, period=rsi_period)
        
        macd_fast = self.params.get('macd_fast', 12)
        macd_slow = self.params.get('macd_slow', 26)
        macd_signal_period = self.params.get('macd_signal', 9)
        macd_data = indicators.macd(df, fast_period=macd_fast, 
                                   slow_period=macd_slow, 
                                   signal_period=macd_signal_period)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        ema_period = self.params.get('ema_period', 20)
        df['ema'] = indicators.ema(df, period=ema_period)
        
        atr_period = self.params.get('atr_period', 14)
        df['atr'] = indicators.atr(df, period=atr_period)
        
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
        
        # Ensure symbol is included in signal data, check table schema for correct field name
        try:
            # First try with the current schema that uses 'symbol'
            signal_data_with_symbol = {
                'symbol': symbol,
                **signal_data
            }
            db.log_strategy_trade(self.name, signal_data_with_symbol)
        except sqlite3.OperationalError as e:
            # If we get an error about missing column, try with legacy schema using 'index_name'
            if "no column named symbol" in str(e):
                signal_data_with_symbol = {
                    'index_name': symbol,
                    **signal_data
                }
                db.log_strategy_trade(self.name, signal_data_with_symbol)
            else:
                # Re-raise other database errors
                raise
    
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