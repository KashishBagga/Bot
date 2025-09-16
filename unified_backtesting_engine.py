#!/usr/bin/env python3
"""
Unified Backtesting Engine with Same Code Paths as Live Trading
Uses dependency injection to ensure backtest and live use identical logic
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str
    end_date: str
    initial_capital: float
    commission_rate: float
    slippage_rate: float
    symbols: List[str]
    strategies: List[str]
    timeframe: str = "1h"
    enable_slippage: bool = True
    enable_commission: bool = True
    enable_latency: bool = True

@dataclass
class BacktestResult:
    """Backtesting results"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float

class DataProvider(ABC):
    """Abstract data provider interface"""
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str, timestamp: datetime) -> float:
        pass
    
    @abstractmethod
    def get_options_chain(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        pass

class LiveDataProvider(DataProvider):
    """Live data provider for real trading"""
    
    def __init__(self, fyers_client):
        self.fyers_client = fyers_client
    
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
        """Get historical data from live API"""
        return self.fyers_client.get_historical_data(symbol, start_date, end_date, interval)
    
    def get_current_price(self, symbol: str, timestamp: datetime) -> float:
        """Get current price from live API"""
        return self.fyers_client.get_current_price(symbol)
    
    def get_options_chain(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """Get options chain from live API"""
        # Implement real options chain fetching
        return {}

class BacktestDataProvider(DataProvider):
    """Backtest data provider using historical data"""
    
    def __init__(self, historical_data: Dict[str, pd.DataFrame]):
        self.historical_data = historical_data
        self.current_timestamp = None
    
    def set_timestamp(self, timestamp: datetime):
        """Set current backtest timestamp"""
        self.current_timestamp = timestamp
    
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
        """Get historical data for backtesting"""
        if symbol in self.historical_data:
            data = self.historical_data[symbol].copy()
            # Filter data for the requested period
            data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
            return data
        return pd.DataFrame()
    
    def get_current_price(self, symbol: str, timestamp: datetime) -> float:
        """Get price at specific timestamp"""
        if symbol in self.historical_data:
            data = self.historical_data[symbol]
            # Find the closest timestamp
            closest_idx = data['timestamp'].searchsorted(timestamp)
            if closest_idx < len(data):
                return data.iloc[closest_idx]['close']
        return 0.0
    
    def get_options_chain(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """Get options chain for backtesting (mock for now)"""
        # In real implementation, this would use historical options data
        return {}

class OrderManager(ABC):
    """Abstract order manager interface"""
    
    @abstractmethod
    def place_order(self, symbol: str, order_type: str, quantity: float, price: float, timestamp: datetime) -> str:
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        pass

class LiveOrderManager(OrderManager):
    """Live order manager for real trading"""
    
    def __init__(self, broker_client):
        self.broker_client = broker_client
        self.orders = {}
    
    def place_order(self, symbol: str, order_type: str, quantity: float, price: float, timestamp: datetime) -> str:
        """Place real order with broker"""
        order_id = f"LIVE_{int(time.time())}"
        # Implement real order placement
        self.orders[order_id] = {
            'symbol': symbol,
            'type': order_type,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'status': 'PENDING'
        }
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel real order"""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELLED'
            return True
        return False
    
    def get_order_status(self, order_id: str) -> str:
        """Get real order status"""
        return self.orders.get(order_id, {}).get('status', 'UNKNOWN')

class BacktestOrderManager(OrderManager):
    """Backtest order manager with simulated execution"""
    
    def __init__(self, config: BacktestConfig, data_provider: BacktestDataProvider):
        self.config = config
        self.data_provider = data_provider
        self.orders = {}
        self.executed_trades = []
        self.capital = config.initial_capital
        self.positions = {}
    
    def place_order(self, symbol: str, order_type: str, quantity: float, price: float, timestamp: datetime) -> str:
        """Place simulated order"""
        order_id = f"BT_{int(time.time())}"
        
        # Apply slippage
        if self.config.enable_slippage:
            slippage = price * self.config.slippage_rate
            if order_type == 'BUY':
                price += slippage
            else:
                price -= slippage
        
        # Apply commission
        commission = 0
        if self.config.enable_commission:
            commission = abs(quantity * price * self.config.commission_rate)
        
        # Check if we have enough capital
        required_capital = abs(quantity * price) + commission
        if order_type == 'BUY' and required_capital > self.capital:
            logger.warning(f"Insufficient capital for order {order_id}")
            return order_id
        
        # Execute order
        self.orders[order_id] = {
            'symbol': symbol,
            'type': order_type,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'status': 'FILLED',
            'commission': commission
        }
        
        # Update capital and positions
        if order_type == 'BUY':
            self.capital -= required_capital
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:
            self.capital += (abs(quantity) * price) - commission
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        # Record trade
        self.executed_trades.append({
            'order_id': order_id,
            'symbol': symbol,
            'type': order_type,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'commission': commission
        })
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel simulated order"""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELLED'
            return True
        return False
    
    def get_order_status(self, order_id: str) -> str:
        """Get simulated order status"""
        return self.orders.get(order_id, {}).get('status', 'UNKNOWN')
    
    def get_portfolio_value(self, timestamp: datetime) -> float:
        """Get total portfolio value at timestamp"""
        total_value = self.capital
        
        for symbol, quantity in self.positions.items():
            if quantity != 0:
                current_price = self.data_provider.get_current_price(symbol, timestamp)
                total_value += quantity * current_price
        
        return total_value

class UnifiedTradingEngine:
    """Unified trading engine that works for both live and backtest"""
    
    def __init__(self, data_provider: DataProvider, order_manager: OrderManager, 
                 strategy_engine, risk_manager, database=None):
        self.data_provider = data_provider
        self.order_manager = order_manager
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.database = database
        self.is_backtest = isinstance(data_provider, BacktestDataProvider)
        
    def run_trading_cycle(self, symbols: List[str], timestamp: datetime) -> List[Dict]:
        """Run one trading cycle - same logic for live and backtest"""
        try:
            # Get current prices
            current_prices = {}
            for symbol in symbols:
                price = self.data_provider.get_current_price(symbol, timestamp)
                if price > 0:
                    current_prices[symbol] = price
            
            if not current_prices:
                return []
            
            # Get historical data for signal generation
            historical_data = {}
            for symbol in symbols:
                start_date = timestamp - timedelta(days=30)
                hist_data = self.data_provider.get_historical_data(symbol, start_date, timestamp, "1h")
                if not hist_data.empty:
                    historical_data[symbol] = hist_data
            
            if not historical_data:
                return []
            
            # Generate signals using same strategy engine
            signals = self.strategy_engine.generate_signals_for_all_symbols(historical_data, current_prices)
            
            # Process signals
            executed_trades = []
            for signal in signals:
                # Risk management check
                if not self.risk_manager.should_execute_signal(signal, current_prices):
                    continue
                
                # Execute trade
                symbol = signal['symbol']
                signal_type = signal['signal']
                quantity = signal.get('position_size', 100)
                price = current_prices[symbol]
                
                # Determine order type
                if 'BUY' in signal_type:
                    order_type = 'BUY'
                else:
                    order_type = 'SELL'
                
                # Place order
                order_id = self.order_manager.place_order(symbol, order_type, quantity, price, timestamp)
                
                # Log to database if available
                if self.database and not self.is_backtest:
                    self.database.save_entry_signal(
                        market="indian",
                        signal_id=order_id,
                        symbol=symbol,
                        strategy=signal['strategy'],
                        signal_type=signal_type,
                        confidence=signal['confidence'],
                        price=price,
                        timestamp=timestamp.isoformat(),
                        timeframe=signal.get('timeframe', '1h'),
                        strength=signal.get('strength', 'MEDIUM'),
                        indicator_values=signal.get('indicator_values', {}),
                        market_condition=signal.get('market_condition', 'UNKNOWN'),
                        volatility=signal.get('volatility', 0.0),
                        position_size=quantity,
                        stop_loss_price=signal.get('stop_loss_price', 0.0),
                        take_profit_price=signal.get('take_profit_price', 0.0)
                    )
                
                executed_trades.append({
                    'order_id': order_id,
                    'signal': signal,
                    'timestamp': timestamp
                })
            
            return executed_trades
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
            return []

class UnifiedBacktestingEngine:
    """Unified backtesting engine with same code paths as live trading"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results = []
        
    def run_backtest(self, historical_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """Run comprehensive backtest"""
        logger.info(f"🚀 Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Initialize components
        data_provider = BacktestDataProvider(historical_data)
        order_manager = BacktestOrderManager(self.config, data_provider)
        
        # Initialize strategy engine and risk manager
        from src.core.enhanced_strategy_engine import EnhancedStrategyEngine
        from src.core.risk_manager import RiskManager
        
        strategy_engine = EnhancedStrategyEngine(self.config.symbols)
        risk_manager = RiskManager()
        
        # Create unified trading engine
        trading_engine = UnifiedTradingEngine(
            data_provider=data_provider,
            order_manager=order_manager,
            strategy_engine=strategy_engine,
            risk_manager=risk_manager
        )
        
        # Run backtest
        start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        
        current_date = start_date
        while current_date <= end_date:
            # Set timestamp for data provider
            data_provider.set_timestamp(current_date)
            
            # Run trading cycle
            trades = trading_engine.run_trading_cycle(self.config.symbols, current_date)
            
            if trades:
                logger.info(f"📊 {current_date.strftime('%Y-%m-%d')}: {len(trades)} trades executed")
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Calculate results
        result = self._calculate_backtest_results(order_manager, start_date, end_date)
        
        logger.info(f"✅ Backtest completed. Total return: {result.total_return*100:.2f}%")
        
        return result
    
    def _calculate_backtest_results(self, order_manager: BacktestOrderManager, 
                                  start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        try:
            trades = order_manager.executed_trades
            
            if not trades:
                return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            # Calculate returns
            initial_capital = self.config.initial_capital
            final_capital = order_manager.get_portfolio_value(end_date)
            total_return = (final_capital - initial_capital) / initial_capital
            
            # Calculate annualized return
            days = (end_date - start_date).days
            annualized_return = (1 + total_return) ** (365 / days) - 1
            
            # Calculate trade statistics
            pnls = []
            durations = []
            
            for i, trade in enumerate(trades):
                # Calculate P&L for this trade (simplified)
                if i < len(trades) - 1:
                    next_trade = trades[i + 1]
                    if trade['symbol'] == next_trade['symbol']:
                        pnl = (next_trade['price'] - trade['price']) * trade['quantity']
                        pnls.append(pnl)
                        
                        duration = (next_trade['timestamp'] - trade['timestamp']).total_seconds() / 3600
                        durations.append(duration)
            
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = sum(1 for pnl in pnls if pnl > 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = np.mean([pnl for pnl in pnls if pnl > 0]) if any(pnl > 0 for pnl in pnls) else 0
            avg_loss = np.mean([pnl for pnl in pnls if pnl < 0]) if any(pnl < 0 for pnl in pnls) else 0
            profit_factor = abs(sum([pnl for pnl in pnls if pnl > 0]) / sum([pnl for pnl in pnls if pnl < 0])) if any(pnl < 0 for pnl in pnls) else float('inf')
            
            # Calculate risk metrics
            returns = np.array(pnls)
            volatility = np.std(returns) if len(returns) > 1 else 0
            sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
            
            # Calculate additional ratios
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            sortino_ratio = np.mean(returns) / np.std([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
            
            avg_trade_duration = np.mean(durations) if durations else 0
            
            return BacktestResult(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                avg_trade_duration=avg_trade_duration,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating backtest results: {e}")
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def display_results(self, result: BacktestResult):
        """Display backtest results"""
        print("\n" + "="*80)
        print("📊 UNIFIED BACKTESTING RESULTS")
        print("="*80)
        
        print(f"\n💰 PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Total Return: {result.total_return*100:.2f}%")
        print(f"Annualized Return: {result.annualized_return*100:.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {result.calmar_ratio:.2f}")
        
        print(f"\n📈 TRADE STATISTICS")
        print("-" * 40)
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.1f}%")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Average Trade Duration: {result.avg_trade_duration:.1f} hours")
        
        print(f"\n🛡️ RISK METRICS")
        print("-" * 40)
        print(f"Maximum Drawdown: {result.max_drawdown*100:.2f}%")
        print(f"Volatility: {result.volatility*100:.2f}%")
        
        print("\n" + "="*80)

def main():
    """Main function for testing"""
    # Create sample historical data
    symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"]
    historical_data = {}
    
    for symbol in symbols:
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
        base_price = 19500 if 'NIFTY50' in symbol else 45000 if 'NIFTYBANK' in symbol else 21000
        
        # Generate realistic price movements
        returns = np.random.normal(0, 0.001, len(dates))
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [p * (1 + np.random.uniform(-0.001, 0.001)) for p in prices],
            'high': [p * (1 + abs(np.random.uniform(0, 0.002))) for p in prices],
            'low': [p * (1 - abs(np.random.uniform(0, 0.002))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates))
        })
        
        historical_data[symbol] = data
    
    # Configure backtest
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        symbols=symbols,
        strategies=["simple_ema", "ema_crossover_enhanced"]
    )
    
    # Run backtest
    engine = UnifiedBacktestingEngine(config)
    result = engine.run_backtest(historical_data)
    
    # Display results
    engine.display_results(result)

if __name__ == "__main__":
    main()
