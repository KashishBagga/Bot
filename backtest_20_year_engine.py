#!/usr/bin/env python3
"""
20-Year Comprehensive Backtesting Engine
Advanced backtesting system for all trading strategies with detailed analytics
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from pathlib import Path
import sqlite3
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import strategy modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from all_strategies import *
from validate_historical_data import HistoricalDataValidator

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000.0
    position_size_percent: float = 2.0  # 2% of portfolio per trade
    max_positions: int = 10
    commission: float = 0.0015  # 0.15% per trade
    slippage: float = 0.001  # 0.1% slippage
    risk_free_rate: float = 0.06  # 6% annual risk-free rate
    
@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    strategy: str
    signal: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    commission_paid: float
    duration_hours: float
    exit_reason: str
    
@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    
class ComprehensiveBacktester:
    """Advanced backtesting engine for 20-year data"""
    
    def __init__(self, config: BacktestConfig = None, data_dir: str = "historical_data_20yr"):
        """Initialize the backtester"""
        self.config = config or BacktestConfig()
        self.data_validator = HistoricalDataValidator(data_dir)
        self.setup_logging()
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.position_sizes = []
        self.current_positions = {}
        
        # Strategy modules
        self.strategies = {
            'ema_crossover': analyze_ema_crossover,
            'insidebar_rsi': analyze_insidebar_rsi,
            'supertrend_ema': analyze_supertrend_ema,
            'supertrend_macd_rsi_ema': analyze_supertrend_macd_rsi_ema
        }
        
        # Results storage
        self.backtest_results = {}
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('Backtester')
    
    def calculate_position_size(self, price: float, portfolio_value: float) -> int:
        """Calculate position size based on risk management"""
        position_value = portfolio_value * (self.config.position_size_percent / 100)
        quantity = int(position_value / price)
        return max(1, quantity)  # Minimum 1 share
    
    def calculate_commission(self, price: float, quantity: int) -> float:
        """Calculate trading commission"""
        trade_value = price * quantity
        return trade_value * self.config.commission
    
    def apply_slippage(self, price: float, signal: str) -> float:
        """Apply slippage to price"""
        if signal == 'BUY':
            return price * (1 + self.config.slippage)
        else:
            return price * (1 - self.config.slippage)
    
    def backtest_strategy(
        self, 
        strategy_name: str, 
        symbol: str, 
        timeframe: str = '5min',
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        """Backtest a single strategy on a symbol"""
        
        self.logger.info(f"📊 Backtesting {strategy_name} on {symbol} ({timeframe})")
        
        # Load data
        df = self.data_validator.load_data(symbol, timeframe)
        if df is None or df.empty:
            self.logger.warning(f"No data available for {symbol} {timeframe}")
            return {'error': 'No data available'}
        
        # Filter date range if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        if df.empty:
            return {'error': 'No data in specified date range'}
        
        # Initialize strategy function
        strategy_func = self.strategies.get(strategy_name)
        if not strategy_func:
            return {'error': f'Strategy {strategy_name} not found'}
        
        # Initialize tracking variables
        portfolio_value = self.config.initial_capital
        cash = self.config.initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        equity_curve = [portfolio_value]
        dates = [df.index[0]]
        
        # Process each candle
        for i in range(len(df)):
            current_time = df.index[i]
            current_data = df.iloc[:i+1]  # Data up to current point
            
            if len(current_data) < 50:  # Need minimum data for indicators
                continue
            
            try:
                # Generate signal
                signal_result = strategy_func(current_data)
                
                if not signal_result or 'action' not in signal_result:
                    continue
                
                signal = signal_result['action']
                current_price = df.iloc[i]['close']
                
                # Position management
                if signal in ['BUY', 'LONG'] and position == 0:
                    # Enter long position
                    entry_price = self.apply_slippage(current_price, 'BUY')
                    position = self.calculate_position_size(entry_price, portfolio_value)
                    commission = self.calculate_commission(entry_price, position)
                    
                    cash -= (entry_price * position + commission)
                    entry_time = current_time
                    
                elif signal in ['SELL', 'SHORT'] and position > 0:
                    # Exit long position
                    exit_price = self.apply_slippage(current_price, 'SELL')
                    commission = self.calculate_commission(exit_price, position)
                    
                    # Calculate trade P&L
                    trade_value = exit_price * position - commission
                    cash += trade_value
                    
                    pnl = (exit_price - entry_price) * position - (2 * commission)
                    pnl_percent = (pnl / (entry_price * position)) * 100
                    
                    # Record trade
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        symbol=symbol,
                        strategy=strategy_name,
                        signal='LONG',
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=position,
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        commission_paid=2 * commission,
                        duration_hours=(current_time - entry_time).total_seconds() / 3600,
                        exit_reason='Signal'
                    )
                    trades.append(trade)
                    
                    position = 0
                    entry_price = 0
                    entry_time = None
                
                # Update portfolio value
                if position > 0:
                    portfolio_value = cash + (current_price * position)
                else:
                    portfolio_value = cash
                
                equity_curve.append(portfolio_value)
                dates.append(current_time)
                
            except Exception as e:
                self.logger.warning(f"Error processing {current_time}: {e}")
                continue
        
        # Close any remaining position
        if position > 0:
            exit_price = self.apply_slippage(df.iloc[-1]['close'], 'SELL')
            commission = self.calculate_commission(exit_price, position)
            trade_value = exit_price * position - commission
            cash += trade_value
            
            pnl = (exit_price - entry_price) * position - (2 * commission)
            pnl_percent = (pnl / (entry_price * position)) * 100
            
            trade = Trade(
                entry_time=entry_time,
                exit_time=df.index[-1],
                symbol=symbol,
                strategy=strategy_name,
                signal='LONG',
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=position,
                pnl=pnl,
                pnl_percent=pnl_percent,
                commission_paid=2 * commission,
                duration_hours=(df.index[-1] - entry_time).total_seconds() / 3600,
                exit_reason='End of data'
            )
            trades.append(trade)
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(
            trades, equity_curve, dates, self.config.initial_capital
        )
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'dates': dates,
            'performance': performance,
            'final_portfolio_value': portfolio_value,
            'total_return_pct': ((portfolio_value - self.config.initial_capital) / self.config.initial_capital) * 100
        }
    
    def calculate_performance_metrics(
        self, 
        trades: List[Trade], 
        equity_curve: List[float], 
        dates: List[datetime],
        initial_capital: float
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return PerformanceMetrics(
                total_return=0, annualized_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, calmar_ratio=0, total_trades=0, winning_trades=0,
                losing_trades=0, win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                largest_win=0, largest_loss=0, avg_trade_duration=0
            )
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else float('inf')
        
        # Return metrics
        final_value = equity_curve[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        # Time-based metrics
        if len(dates) > 1:
            time_period_years = (dates[-1] - dates[0]).days / 365.25
            annualized_return = ((final_value / initial_capital) ** (1 / time_period_years) - 1) * 100
        else:
            annualized_return = 0
            time_period_years = 1
        
        # Volatility and Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            excess_return = annualized_return - self.config.risk_free_rate * 100
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Maximum drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Average trade duration
        avg_trade_duration = np.mean([t.duration_hours for t in trades]) if trades else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_trade_duration
        )
    
    def backtest_all_strategies(
        self, 
        symbols: List[str] = None, 
        timeframes: List[str] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        """Backtest all strategies across all symbols and timeframes"""
        
        if symbols is None:
            symbols = self.data_validator.get_available_symbols()
        
        if timeframes is None:
            timeframes = ['5min', '15min', '1hour', '1day']
        
        self.logger.info(f"🚀 Starting comprehensive backtest...")
        self.logger.info(f"📊 Strategies: {list(self.strategies.keys())}")
        self.logger.info(f"📈 Symbols: {symbols}")
        self.logger.info(f"⏰ Timeframes: {timeframes}")
        
        results = {}
        total_combinations = len(self.strategies) * len(symbols) * len(timeframes)
        current_combination = 0
        
        for strategy_name in self.strategies.keys():
            results[strategy_name] = {}
            
            for symbol in symbols:
                results[strategy_name][symbol] = {}
                
                for timeframe in timeframes:
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100
                    
                    self.logger.info(f"⏳ Progress: {progress:.1f}% - {strategy_name} | {symbol} | {timeframe}")
                    
                    # Run backtest
                    result = self.backtest_strategy(
                        strategy_name, symbol, timeframe, start_date, end_date
                    )
                    
                    results[strategy_name][symbol][timeframe] = result
        
        # Store results
        self.backtest_results = results
        
        # Generate summary
        summary = self.generate_comprehensive_summary(results)
        
        return {
            'detailed_results': results,
            'summary': summary,
            'config': self.config.__dict__,
            'execution_time': datetime.now().isoformat()
        }
    
    def generate_comprehensive_summary(self, results: Dict) -> Dict:
        """Generate comprehensive summary of all backtest results"""
        
        summary = {
            'overall_performance': {},
            'strategy_rankings': {},
            'symbol_performance': {},
            'timeframe_analysis': {},
            'risk_metrics': {},
            'trade_statistics': {}
        }
        
        all_results = []
        strategy_performance = {}
        symbol_performance = {}
        timeframe_performance = {}
        
        # Collect all valid results
        for strategy in results:
            strategy_performance[strategy] = []
            
            for symbol in results[strategy]:
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = []
                
                for timeframe in results[strategy][symbol]:
                    result = results[strategy][symbol][timeframe]
                    
                    if 'performance' in result and result.get('performance'):
                        perf = result['performance']
                        
                        result_summary = {
                            'strategy': strategy,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'total_return': perf.total_return,
                            'annualized_return': perf.annualized_return,
                            'sharpe_ratio': perf.sharpe_ratio,
                            'max_drawdown': perf.max_drawdown,
                            'win_rate': perf.win_rate,
                            'total_trades': perf.total_trades,
                            'profit_factor': perf.profit_factor
                        }
                        
                        all_results.append(result_summary)
                        strategy_performance[strategy].append(result_summary)
                        symbol_performance[symbol].append(result_summary)
                        
                        if timeframe not in timeframe_performance:
                            timeframe_performance[timeframe] = []
                        timeframe_performance[timeframe].append(result_summary)
        
        if not all_results:
            return summary
        
        # Overall performance metrics
        summary['overall_performance'] = {
            'total_backtests': len(all_results),
            'avg_total_return': np.mean([r['total_return'] for r in all_results]),
            'avg_annualized_return': np.mean([r['annualized_return'] for r in all_results]),
            'avg_sharpe_ratio': np.mean([r['sharpe_ratio'] for r in all_results if r['sharpe_ratio'] != float('inf')]),
            'avg_max_drawdown': np.mean([r['max_drawdown'] for r in all_results]),
            'avg_win_rate': np.mean([r['win_rate'] for r in all_results]),
            'total_trades': sum([r['total_trades'] for r in all_results])
        }
        
        # Strategy rankings
        for strategy in strategy_performance:
            if strategy_performance[strategy]:
                strategy_results = strategy_performance[strategy]
                summary['strategy_rankings'][strategy] = {
                    'avg_return': np.mean([r['total_return'] for r in strategy_results]),
                    'avg_sharpe': np.mean([r['sharpe_ratio'] for r in strategy_results if r['sharpe_ratio'] != float('inf')]),
                    'avg_drawdown': np.mean([r['max_drawdown'] for r in strategy_results]),
                    'win_rate': np.mean([r['win_rate'] for r in strategy_results]),
                    'total_trades': sum([r['total_trades'] for r in strategy_results]),
                    'num_backtests': len(strategy_results)
                }
        
        # Symbol performance
        for symbol in symbol_performance:
            if symbol_performance[symbol]:
                symbol_results = symbol_performance[symbol]
                summary['symbol_performance'][symbol] = {
                    'avg_return': np.mean([r['total_return'] for r in symbol_results]),
                    'avg_sharpe': np.mean([r['sharpe_ratio'] for r in symbol_results if r['sharpe_ratio'] != float('inf')]),
                    'best_strategy': max(symbol_results, key=lambda x: x['total_return'])['strategy'],
                    'total_trades': sum([r['total_trades'] for r in symbol_results])
                }
        
        # Timeframe analysis
        for timeframe in timeframe_performance:
            if timeframe_performance[timeframe]:
                tf_results = timeframe_performance[timeframe]
                summary['timeframe_analysis'][timeframe] = {
                    'avg_return': np.mean([r['total_return'] for r in tf_results]),
                    'avg_sharpe': np.mean([r['sharpe_ratio'] for r in tf_results if r['sharpe_ratio'] != float('inf')]),
                    'avg_trades_per_backtest': np.mean([r['total_trades'] for r in tf_results]),
                    'num_backtests': len(tf_results)
                }
        
        return summary
    
    def save_results(self, results: Dict, filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert Trade objects to dictionaries for JSON serialization
        serializable_results = self.convert_results_for_json(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"📄 Results saved to: {filename}")
        return filename
    
    def convert_results_for_json(self, results: Dict) -> Dict:
        """Convert results with Trade objects to JSON-serializable format"""
        if 'detailed_results' in results:
            for strategy in results['detailed_results']:
                for symbol in results['detailed_results'][strategy]:
                    for timeframe in results['detailed_results'][strategy][symbol]:
                        result = results['detailed_results'][strategy][symbol][timeframe]
                        if 'trades' in result:
                            # Convert Trade objects to dictionaries
                            result['trades'] = [
                                {
                                    'entry_time': trade.entry_time.isoformat(),
                                    'exit_time': trade.exit_time.isoformat(),
                                    'symbol': trade.symbol,
                                    'strategy': trade.strategy,
                                    'signal': trade.signal,
                                    'entry_price': trade.entry_price,
                                    'exit_price': trade.exit_price,
                                    'quantity': trade.quantity,
                                    'pnl': trade.pnl,
                                    'pnl_percent': trade.pnl_percent,
                                    'commission_paid': trade.commission_paid,
                                    'duration_hours': trade.duration_hours,
                                    'exit_reason': trade.exit_reason
                                }
                                for trade in result['trades']
                            ]
                        if 'performance' in result and result['performance']:
                            # Convert PerformanceMetrics to dictionary
                            perf = result['performance']
                            result['performance'] = {
                                'total_return': perf.total_return,
                                'annualized_return': perf.annualized_return,
                                'volatility': perf.volatility,
                                'sharpe_ratio': perf.sharpe_ratio,
                                'max_drawdown': perf.max_drawdown,
                                'calmar_ratio': perf.calmar_ratio,
                                'total_trades': perf.total_trades,
                                'winning_trades': perf.winning_trades,
                                'losing_trades': perf.losing_trades,
                                'win_rate': perf.win_rate,
                                'avg_win': perf.avg_win,
                                'avg_loss': perf.avg_loss,
                                'profit_factor': perf.profit_factor,
                                'largest_win': perf.largest_win,
                                'largest_loss': perf.largest_loss,
                                'avg_trade_duration': perf.avg_trade_duration
                            }
        
        return results
    
    def print_summary_report(self, results: Dict):
        """Print comprehensive summary report"""
        summary = results.get('summary', {})
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE 20-YEAR BACKTESTING REPORT")
        print("="*80)
        
        # Overall performance
        overall = summary.get('overall_performance', {})
        if overall:
            print(f"\n🎯 OVERALL PERFORMANCE:")
            print(f"   Total Backtests: {overall.get('total_backtests', 0)}")
            print(f"   Average Total Return: {overall.get('avg_total_return', 0):.2f}%")
            print(f"   Average Annualized Return: {overall.get('avg_annualized_return', 0):.2f}%")
            print(f"   Average Sharpe Ratio: {overall.get('avg_sharpe_ratio', 0):.2f}")
            print(f"   Average Max Drawdown: {overall.get('avg_max_drawdown', 0):.2f}%")
            print(f"   Average Win Rate: {overall.get('avg_win_rate', 0):.2f}%")
            print(f"   Total Trades Executed: {overall.get('total_trades', 0):,}")
        
        # Strategy rankings
        strategy_rankings = summary.get('strategy_rankings', {})
        if strategy_rankings:
            print(f"\n🏆 STRATEGY RANKINGS (by Average Return):")
            sorted_strategies = sorted(
                strategy_rankings.items(), 
                key=lambda x: x[1].get('avg_return', 0), 
                reverse=True
            )
            for i, (strategy, metrics) in enumerate(sorted_strategies, 1):
                print(f"   {i}. {strategy.upper()}:")
                print(f"      Return: {metrics.get('avg_return', 0):.2f}%")
                print(f"      Sharpe: {metrics.get('avg_sharpe', 0):.2f}")
                print(f"      Win Rate: {metrics.get('win_rate', 0):.2f}%")
                print(f"      Total Trades: {metrics.get('total_trades', 0):,}")
        
        # Symbol performance
        symbol_performance = summary.get('symbol_performance', {})
        if symbol_performance:
            print(f"\n📈 TOP PERFORMING SYMBOLS:")
            sorted_symbols = sorted(
                symbol_performance.items(), 
                key=lambda x: x[1].get('avg_return', 0), 
                reverse=True
            )[:5]  # Top 5
            for i, (symbol, metrics) in enumerate(sorted_symbols, 1):
                print(f"   {i}. {symbol}:")
                print(f"      Avg Return: {metrics.get('avg_return', 0):.2f}%")
                print(f"      Best Strategy: {metrics.get('best_strategy', 'N/A')}")
                print(f"      Total Trades: {metrics.get('total_trades', 0):,}")
        
        # Timeframe analysis
        timeframe_analysis = summary.get('timeframe_analysis', {})
        if timeframe_analysis:
            print(f"\n⏰ TIMEFRAME ANALYSIS:")
            for timeframe, metrics in timeframe_analysis.items():
                print(f"   {timeframe.upper()}:")
                print(f"      Avg Return: {metrics.get('avg_return', 0):.2f}%")
                print(f"      Avg Sharpe: {metrics.get('avg_sharpe', 0):.2f}")
                print(f"      Avg Trades per Backtest: {metrics.get('avg_trades_per_backtest', 0):.1f}")
        
        print("\n" + "="*80)


def main():
    """Main execution function"""
    print("🚀 20-YEAR COMPREHENSIVE BACKTESTING ENGINE")
    print("="*60)
    
    # Initialize backtester
    config = BacktestConfig(
        initial_capital=100000.0,
        position_size_percent=2.0,
        commission=0.0015,
        slippage=0.001
    )
    
    backtester = ComprehensiveBacktester(config)
    
    # Check data availability
    validator = HistoricalDataValidator()
    symbols = validator.get_available_symbols()
    
    if not symbols:
        print("❌ No historical data found. Run fetch_20_year_historical_data.py first.")
        return 1
    
    print(f"📊 Found data for {len(symbols)} symbols")
    
    # Run comprehensive backtest
    print("🔄 Starting comprehensive backtesting...")
    
    # For demo, test with limited scope (can be expanded)
    test_symbols = symbols[:3]  # First 3 symbols
    test_timeframes = ['5min', '1hour', '1day']
    
    results = backtester.backtest_all_strategies(
        symbols=test_symbols,
        timeframes=test_timeframes
    )
    
    # Print summary
    backtester.print_summary_report(results)
    
    # Save results
    filename = backtester.save_results(results)
    
    print(f"\n✅ Backtesting completed!")
    print(f"📄 Results saved to: {filename}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 