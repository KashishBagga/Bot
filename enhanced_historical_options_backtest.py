#!/usr/bin/env python3
"""
Enhanced Historical Options Backtester
Per-strategy analysis with comprehensive risk metrics
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.unified_database import UnifiedDatabase
from src.models.option_contract import OptionContract, OptionChain, OptionType, StrikeSelection
from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
from src.strategies.supertrend_ema import SupertrendEma
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.data.historical_options_loader import HistoricalOptionsLoader
from src.core.option_signal_mapper import OptionSignalMapper
from src.core.options_pnl_calculator import OptionsPnLCalculator, PositionType
from simple_backtest import OptimizedBacktester
from src.data.local_data_loader import LocalDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_historical_options_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedHistoricalOptionsBacktester:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_risk_per_trade: float = 0.02,
        confidence_cutoff: float = 40.0,
        exposure_limit: float = 0.6,
        max_daily_loss_pct: float = 0.03,
        commission_bps: float = 1.0,
        slippage_bps: float = 5.0,
        expiry_type: str = "weekly",
        strike_selection: StrikeSelection = StrikeSelection.ATM,
        delta_target: float = 0.30,
        use_historical_options: bool = True,
        run_per_strategy: bool = True
    ):
        """Initialize enhanced historical options backtester."""
        self.initial_capital = float(initial_capital)
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.confidence_cutoff = float(confidence_cutoff)
        self.exposure_limit = float(exposure_limit)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.use_historical_options = use_historical_options
        self.run_per_strategy = run_per_strategy

        # Initialize components
        self.db = UnifiedDatabase()
        self.data_loader = LocalDataLoader()
        self.historical_options_loader = HistoricalOptionsLoader()
        self.signal_mapper = OptionSignalMapper(self.historical_options_loader)
        self.backtester = OptimizedBacktester()
        self.pnl_calculator = OptionsPnLCalculator()

        # Set options parameters
        self.signal_mapper.set_parameters(
            expiry_type=expiry_type,
            strike_selection=strike_selection,
            delta_target=delta_target
        )

        # Trading state
        self.open_positions: Dict[str, Dict] = {}
        self.closed_positions: List[Dict] = []
        self.daily_pnl = 0.0
        self.daily_loss_limit_hit = False
        self.equity_curve = []
        self.strategy_results = {}

        logger.info(f"🚀 Enhanced Historical Options Backtester initialized")
        logger.info(f"📊 Capital: ₹{initial_capital:,.2f} | Risk per trade: {max_risk_per_trade*100:.1f}%")
        logger.info(f"🔄 Mode: {'Historical Options' if use_historical_options else 'Index-based'}")
        logger.info(f"📈 Analysis: {'Per-strategy' if run_per_strategy else 'Combined'}")

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage in bps to a price."""
        factor = 1.0 + (self.slippage_bps / 10000.0) if is_buy else 1.0 - (self.slippage_bps / 10000.0)
        return float(price) * factor

    def calculate_option_position_size(self, entry_price: float, confidence: float, lot_size: int) -> float:
        """Calculate position size based on risk management rules."""
        risk_amount = self.initial_capital * self.max_risk_per_trade
        
        # Confidence multiplier: normalized around 50, capped
        confidence_multiplier = min(max(confidence / 50.0, 0.5), 1.5)
        adjusted_risk = risk_amount * confidence_multiplier
        
        # For buying options, risk = premium paid
        premium_per_lot = entry_price * lot_size
        
        if premium_per_lot <= 0:
            return 0.0
        
        # Calculate maximum lots based on risk
        max_lots = int(adjusted_risk / premium_per_lot)
        
        # Cap by available capital (use 80% of capital for safety)
        available_capital = self.initial_capital * 0.8
        max_affordable_lots = int(available_capital // premium_per_lot)
        max_lots = min(max_lots, max_affordable_lots)
        
        # Apply per-contract caps
        per_contract_max_lots = 10  # Maximum 10 lots per contract
        max_lots = min(max_lots, per_contract_max_lots)
        
        # Ensure minimum 1 lot if we can afford it
        if max_lots < 1:
            if premium_per_lot <= available_capital and adjusted_risk >= premium_per_lot * 0.5:
                return float(lot_size)  # 1 lot
            else:
                return 0.0
        
        return float(max_lots * lot_size)  # Return quantity in shares

    def should_open_option_position(self, option_signal: Dict) -> bool:
        """Check if we should open an option position."""
        # Respect daily-stop
        if self.daily_loss_limit_hit:
            return False

        # Confidence threshold
        if float(option_signal.get('confidence', 0.0)) < self.confidence_cutoff:
            return False

        return True

    def open_option_position(self, option_signal: Dict, timestamp: datetime) -> Optional[str]:
        """Open a new option position using historical data."""
        try:
            contract = option_signal.get('contract')
            if not contract:
                return None

            entry_price = float(option_signal.get('entry_price', 0))
            if entry_price <= 0:
                return None

            # Calculate position size using contract lot size
            position_size = self.calculate_option_position_size(
                entry_price, 
                float(option_signal.get('confidence', 0)),
                contract.lot_size
            )
            
            if position_size <= 0:
                return None

            # Apply slippage
            exec_price = self._apply_slippage(entry_price, is_buy=True)
            
            # Calculate entry cost using P&L calculator
            entry_data = self.pnl_calculator.calculate_entry_cost(
                PositionType.LONG,
                exec_price,
                int(position_size),
                contract.lot_size,
                self.commission_bps
            )

            position_id = f"{contract.symbol}_{int(time.time())}"
            position = {
                'id': position_id,
                'contract_symbol': contract.symbol,
                'underlying': contract.underlying,
                'strategy': option_signal['strategy'],
                'entry_time': timestamp,
                'entry_price': exec_price,
                'quantity': position_size,
                'lot_size': contract.lot_size,
                'strike': contract.strike,
                'expiry': contract.expiry,
                'option_type': contract.option_type.value,
                'entry_cost': entry_data['total_cost'],
                'commission': entry_data['commission'],
                'status': 'OPEN',
                'position_type': PositionType.LONG.value
            }

            self.open_positions[position_id] = position
            
            logger.info(f"✅ Opened {option_signal.get('signal_type', 'UNKNOWN')} option position: {position_id}")
            logger.info(f"   Contract: {contract.symbol} | Size: {position_size} | Cost: ₹{entry_data['total_cost']:,.2f}")
            
            return position_id

        except Exception as e:
            logger.error(f"❌ Error opening option position: {e}")
            return None

    def close_option_position(self, position_id: str, exit_price: float, exit_reason: str, timestamp: datetime) -> Optional[Dict]:
        """Close an option position using historical data."""
        if position_id not in self.open_positions:
            return None

        position = self.open_positions[position_id]
        
        # Apply slippage
        exec_price = self._apply_slippage(exit_price, is_buy=False)
        
        # Calculate exit value using P&L calculator
        exit_data = self.pnl_calculator.calculate_exit_value(
            PositionType.LONG,
            exec_price,
            int(position['quantity']),
            position['lot_size'],
            self.commission_bps
        )
        
        # Calculate P&L
        pnl_result = self.pnl_calculator.calculate_pnl(
            {'position_type': PositionType.LONG.value, 'total_cost': position['entry_cost'], 'quantity': position['quantity'], 'lots': position['quantity'] / position['lot_size']},
            {'total_received': exit_data['total_received'], 'quantity': position['quantity'], 'lots': position['quantity'] / position['lot_size']}
        )

        closed_position = {
            **position,
            'exit_price': exec_price,
            'exit_time': timestamp,
            'exit_reason': exit_reason,
            'pnl': pnl_result['pnl'],
            'returns_pct': pnl_result['returns_pct'],
            'exit_commission': exit_data['commission'],
            'total_commission': pnl_result['total_commission'],
            'status': 'CLOSED'
        }

        # Remove from open positions
        del self.open_positions[position_id]
        self.closed_positions.append(closed_position)

        # Accumulate daily P&L
        self.daily_pnl += pnl_result['pnl']

        logger.info(f"🔒 Closed position: {position_id}")
        logger.info(f"   P&L: ₹{pnl_result['pnl']:+.2f} ({pnl_result['returns_pct']:+.2f}%) | Reason: {exit_reason}")

        return closed_position

    def check_option_position_exits(self, current_price: float, symbol: str, timestamp: datetime) -> List[Dict]:
        """Check if any open positions should be closed using historical data."""
        closed_positions = []
        positions_to_close = []

        for position_id, position in list(self.open_positions.items()):
            if position['underlying'] != symbol:
                continue

            # Get historical options price for this contract
            if self.use_historical_options:
                historical_price = self.historical_options_loader.get_historical_options_price(
                    position['contract_symbol'], timestamp
                )
                
                if historical_price is None:
                    # If no historical data, use simulated price
                    historical_price = self._estimate_option_price(position, current_price)
            else:
                historical_price = self._estimate_option_price(position, current_price)

            if historical_price <= 0:
                continue

            # Premium-based exit logic
            entry_premium = position['entry_price']
            premium_change_pct = (historical_price - entry_premium) / entry_premium if entry_premium > 0 else 0

            # Exit conditions
            if premium_change_pct <= -0.5:  # 50% loss
                positions_to_close.append((position_id, historical_price, 'Stop Loss - Premium -50%'))
            elif premium_change_pct >= 1.0:  # 100% gain
                positions_to_close.append((position_id, historical_price, 'Target Hit - Premium +100%'))
            elif premium_change_pct <= -0.3:  # 30% loss
                positions_to_close.append((position_id, historical_price, 'Stop Loss - Premium -30%'))
            elif premium_change_pct >= 0.5:  # 50% gain
                positions_to_close.append((position_id, historical_price, 'Target Hit - Premium +50%'))

        # Process closures
        for position_id, exit_price, exit_reason in positions_to_close:
            closed_position = self.close_option_position(position_id, exit_price, exit_reason, timestamp)
            if closed_position:
                closed_positions.append(closed_position)

        return closed_positions

    def _estimate_option_price(self, position: Dict, underlying_price: float) -> float:
        """Estimate option price when historical data is not available."""
        try:
            # Simplified Black-Scholes approximation
            strike = position['strike']
            option_type = position['option_type']
            time_to_expiry = (position['expiry'] - datetime.now()).total_seconds() / (365 * 24 * 3600)
            
            if time_to_expiry <= 0:
                return 0.0

            # Assume 20% implied volatility
            iv = 0.20
            moneyness = underlying_price / strike
            
            if option_type.upper() == 'CALL':
                if moneyness > 1.1:  # ITM
                    premium = max(underlying_price - strike, 0) + (strike * iv * np.sqrt(time_to_expiry) * 0.4)
                elif moneyness < 0.9:  # OTM
                    premium = strike * iv * np.sqrt(time_to_expiry) * 0.3
                else:  # ATM
                    premium = strike * iv * np.sqrt(time_to_expiry) * 0.4
            else:  # PUT
                if moneyness < 0.9:  # ITM
                    premium = max(strike - underlying_price, 0) + (strike * iv * np.sqrt(time_to_expiry) * 0.4)
                elif moneyness > 1.1:  # OTM
                    premium = strike * iv * np.sqrt(time_to_expiry) * 0.3
                else:  # ATM
                    premium = strike * iv * np.sqrt(time_to_expiry) * 0.4

            return premium

        except Exception as e:
            logger.debug(f"Error estimating option price: {e}")
            return 0.0

    def run_strategy_backtest(self, strategy_name: str, strategy, df: pd.DataFrame, symbol: str) -> Dict:
        """Run backtest for a single strategy."""
        logger.info(f"🔄 Running backtest for strategy: {strategy_name}")
        
        # Reset state for this strategy
        self.open_positions = {}
        self.closed_positions = []
        self.daily_pnl = 0.0
        self.daily_loss_limit_hit = False
        self.equity_curve = []
        current_capital = self.initial_capital

        try:
            # Generate signals for this strategy
            if hasattr(strategy, 'analyze_vectorized'):
                signals_df = strategy.analyze_vectorized(df)
            else:
                logger.warning(f"Strategy {strategy_name} doesn't support vectorized analysis")
                return {}

            if signals_df.empty:
                logger.info(f"⚠️ No signals generated for {strategy_name}")
                return {}

            # Process signals chronologically
            for idx, row in signals_df.iterrows():
                try:
                    # Skip if daily loss limit hit
                    if self.daily_loss_limit_hit:
                        continue

                    # Create signal dict
                    signal = {
                        'timestamp': df.loc[idx, 'timestamp'],
                        'strategy': strategy_name,
                        'signal': row['signal'],
                        'price': float(row['price']),
                        'confidence': float(row.get('confidence_score', 50)),
                        'reasoning': str(row.get('reasoning', ''))[:200],
                        'symbol': symbol,
                        'capital': current_capital,
                        'max_risk_per_trade': self.max_risk_per_trade
                    }

                    current_price = signal['price']
                    current_time = signal['timestamp']
                    
                    # Load historical options chain for this date
                    if self.use_historical_options:
                        option_chain = self.historical_options_loader.load_historical_options_chain(
                            symbol, current_time.date()
                        )
                    else:
                        option_chain = None

                    option_signals = self.signal_mapper.map_multiple_signals(
                        [signal], current_price, current_time, option_chain
                    )
                    
                    for option_signal in option_signals:
                        if self.should_open_option_position(option_signal):
                            position_id = self.open_option_position(option_signal, current_time)
                            if position_id:
                                logger.debug(f"Opened position: {position_id}")

                    # Check exits using historical data
                    closed_positions = self.check_option_position_exits(current_price, symbol, current_time)
                    
                    # Check daily loss limit
                    if self.daily_pnl < -abs(self.max_daily_loss_pct * self.initial_capital):
                        self.daily_loss_limit_hit = True
                        logger.warning(f"🚫 Daily loss limit breached for {strategy_name}: PnL={self.daily_pnl:.2f}")

                    # Update current capital
                    current_capital = self.initial_capital + sum(p.get('pnl', 0) for p in self.closed_positions)

                    # Record equity curve
                    self.equity_curve.append({
                        'timestamp': current_time,
                        'equity': current_capital,
                        'open_positions': len(self.open_positions),
                        'daily_pnl': self.daily_pnl
                    })

                except Exception as e:
                    logger.error(f"❌ Error processing signal for {strategy_name}: {e}")

            # Close remaining positions at last price
            last_price = float(df['close'].iloc[-1])
            last_time = df['timestamp'].iloc[-1]
            
            for position_id in list(self.open_positions.keys()):
                self.close_option_position(position_id, last_price, 'Backtest End', last_time)

            # Calculate final results
            results = self._calculate_strategy_results(strategy_name)
            
            logger.info(f"✅ Completed backtest for {strategy_name}: {results['total_trades']} trades, {results['win_rate']:.1f}% win rate")
            
            return results

        except Exception as e:
            logger.error(f"❌ Error in strategy backtest for {strategy_name}: {e}")
            return {}

    def _calculate_strategy_results(self, strategy_name: str) -> Dict:
        """Calculate results for a strategy."""
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades else 0.0
        
        total_pnl = sum(p['pnl'] for p in self.closed_positions)
        avg_pnl = (total_pnl / total_trades) if total_trades else 0.0
        
        returns = ((self.initial_capital + total_pnl - self.initial_capital) / self.initial_capital) * 100

        # Calculate drawdown metrics
        equity_values = [e['equity'] for e in self.equity_curve] if self.equity_curve else [self.initial_capital]
        drawdown_metrics = self.pnl_calculator.calculate_drawdown_metrics(equity_values)
        
        # Calculate risk metrics
        risk_metrics = self.pnl_calculator.calculate_risk_metrics(self.closed_positions, self.initial_capital)

        return {
            'strategy_name': strategy_name,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'returns': returns,
            'closed_positions': self.closed_positions.copy(),
            'equity_curve': self.equity_curve.copy(),
            'drawdown_metrics': drawdown_metrics,
            'risk_metrics': risk_metrics
        }

    def run_enhanced_historical_options_backtest(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Dict:
        """Run enhanced historical options backtest with per-strategy analysis."""
        logger.info(f"🚀 Starting enhanced historical options backtest: {symbol} {timeframe}")
        logger.info(f"📅 Date range: {start_date.date()} to {end_date.date()}")
        
        # Create sample historical options data if needed
        if self.use_historical_options:
            logger.info("📊 Creating sample historical options data for testing...")
            self.historical_options_loader.create_sample_historical_data(symbol, start_date, end_date)
        
        # Load index data
        days_diff = (end_date - start_date).days
        df = self.data_loader.load_data(symbol, timeframe, days_diff * 24 * 60 // 5)  # Approximate candles
        
        if df is None or df.empty:
            logger.error(f"❌ No data available for {symbol} {timeframe}")
            return {}

        # Filter data to date range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # Add indicators
        df = self.backtester.add_indicators_optimized(df)
        logger.info(f"✅ Loaded {len(df)} candles with indicators")

        # Initialize strategies
        strategies = {
            'ema_crossover_enhanced': EmaCrossoverEnhanced(),
            'supertrend_ema': SupertrendEma(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
        }

        # Run per-strategy backtests
        if self.run_per_strategy:
            for strategy_name, strategy in strategies.items():
                try:
                    strategy_results = self.run_strategy_backtest(strategy_name, strategy, df, symbol)
                    if strategy_results:
                        self.strategy_results[strategy_name] = strategy_results
                except Exception as e:
                    logger.error(f"❌ Error running strategy {strategy_name}: {e}")
        else:
            # Run combined backtest (original method)
            logger.info("🔄 Running combined strategy backtest...")
            # Implementation for combined backtest would go here
            pass

        # Calculate combined results
        combined_results = self._calculate_combined_results()
        
        logger.info(f"✅ Enhanced historical options backtest completed")
        logger.info(f"📊 Strategies tested: {len(self.strategy_results)}")
        
        return combined_results

    def _calculate_combined_results(self) -> Dict:
        """Calculate combined results across all strategies."""
        if not self.strategy_results:
            return {}

        # Aggregate metrics across strategies
        total_trades = sum(r['total_trades'] for r in self.strategy_results.values())
        total_pnl = sum(r['total_pnl'] for r in self.strategy_results.values())
        total_winning_trades = sum(r['winning_trades'] for r in self.strategy_results.values())
        
        overall_win_rate = (total_winning_trades / total_trades * 100) if total_trades else 0.0
        overall_returns = (total_pnl / self.initial_capital) * 100

        # Find best and worst strategies
        strategy_performance = [(name, r['returns']) for name, r in self.strategy_results.items()]
        strategy_performance.sort(key=lambda x: x[1], reverse=True)
        
        best_strategy = strategy_performance[0] if strategy_performance else None
        worst_strategy = strategy_performance[-1] if strategy_performance else None

        return {
            'overall_results': {
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'win_rate': overall_win_rate,
                'returns': overall_returns,
                'strategies_tested': len(self.strategy_results)
            },
            'strategy_results': self.strategy_results,
            'best_strategy': best_strategy,
            'worst_strategy': worst_strategy,
            'initial_capital': self.initial_capital
        }

    def print_enhanced_results(self, results: Dict):
        """Print enhanced backtest results."""
        print("\n" + "=" * 80)
        print("📊 ENHANCED HISTORICAL OPTIONS BACKTEST RESULTS")
        print("=" * 80)
        
        if not results:
            print("❌ No results to display")
            return

        overall = results.get('overall_results', {})
        strategy_results = results.get('strategy_results', {})
        
        print(f"💰 Initial Capital: ₹{results.get('initial_capital', 0):,.2f}")
        print(f"📈 Overall Returns: {overall.get('returns', 0):+.2f}%")
        print(f"📊 Total Trades: {overall.get('total_trades', 0)}")
        print(f"🎯 Overall Win Rate: {overall.get('win_rate', 0):.1f}%")
        print(f"💵 Total P&L: ₹{overall.get('total_pnl', 0):+,.2f}")
        print(f"🔄 Strategies Tested: {overall.get('strategies_tested', 0)}")
        
        if results.get('best_strategy'):
            best_name, best_return = results['best_strategy']
            print(f"🏆 Best Strategy: {best_name} ({best_return:+.2f}%)")
        
        if results.get('worst_strategy'):
            worst_name, worst_return = results['worst_strategy']
            print(f"📉 Worst Strategy: {worst_name} ({worst_return:+.2f}%)")
        
        print("\n" + "-" * 80)
        print("📋 PER-STRATEGY BREAKDOWN")
        print("-" * 80)
        
        for strategy_name, strategy_data in strategy_results.items():
            print(f"\n🔹 {strategy_name.upper()}:")
            print(f"   Trades: {strategy_data['total_trades']} | Win Rate: {strategy_data['win_rate']:.1f}%")
            print(f"   P&L: ₹{strategy_data['total_pnl']:+,.2f} | Returns: {strategy_data['returns']:+.2f}%")
            
            # Drawdown metrics
            dd_metrics = strategy_data.get('drawdown_metrics', {})
            if dd_metrics:
                print(f"   Max Drawdown: {dd_metrics.get('max_drawdown_pct', 0):.2f}%")
                print(f"   Rolling 30d DD: {dd_metrics.get('rolling_30d_drawdown_pct', 0):.2f}%")
            
            # Risk metrics
            risk_metrics = strategy_data.get('risk_metrics', {})
            if risk_metrics:
                print(f"   Profit Factor: {risk_metrics.get('profit_factor', 0):.2f}")
                print(f"   Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
                print(f"   Max Consecutive Losses: {risk_metrics.get('max_consecutive_losses', 0)}")
        
        print("\n" + "=" * 80)
        print(f"🔄 Mode: {'Historical Options' if self.use_historical_options else 'Index-based'}")
        print(f"📈 Analysis: {'Per-strategy' if self.run_per_strategy else 'Combined'}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Enhanced Historical Options Backtesting System')
    parser.add_argument('--symbol', default='NSE:NIFTY50-INDEX', help='Underlying symbol')
    parser.add_argument('--timeframe', default='5min', help='Timeframe')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=30, help='Days to backtest (if start/end not specified)')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.02, help='Max risk per trade')
    parser.add_argument('--confidence', type=float, default=40.0, help='Min confidence to open trades')
    parser.add_argument('--exposure', type=float, default=0.6, help='Max portfolio exposure (0-1)')
    parser.add_argument('--daily_loss', type=float, default=0.03, help='Max daily loss percent (0-1)')
    parser.add_argument('--commission_bps', type=float, default=1.0, help='Commission in bps')
    parser.add_argument('--slippage_bps', type=float, default=5.0, help='Slippage in bps')
    parser.add_argument('--expiry', type=str, default='weekly', help='Expiry type (weekly/monthly)')
    parser.add_argument('--strike', type=str, default='atm', help='Strike selection (atm/otm/itm/delta)')
    parser.add_argument('--delta', type=float, default=0.30, help='Target delta for delta-based selection')
    parser.add_argument('--use_historical', action='store_true', help='Use historical options data')
    parser.add_argument('--per_strategy', action='store_true', default=True, help='Run per-strategy analysis')

    args = parser.parse_args()

    # Convert strike selection string to enum
    strike_selection_map = {
        'atm': StrikeSelection.ATM,
        'otm': StrikeSelection.OTM,
        'itm': StrikeSelection.ITM,
        'delta': StrikeSelection.DELTA
    }
    strike_selection = strike_selection_map.get(args.strike.lower(), StrikeSelection.ATM)

    # Parse dates
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)

    backtester = EnhancedHistoricalOptionsBacktester(
        initial_capital=args.capital,
        max_risk_per_trade=args.risk,
        confidence_cutoff=args.confidence,
        exposure_limit=args.exposure,
        max_daily_loss_pct=args.daily_loss,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        expiry_type=args.expiry,
        strike_selection=strike_selection,
        delta_target=args.delta,
        use_historical_options=args.use_historical,
        run_per_strategy=args.per_strategy
    )

    try:
        results = backtester.run_enhanced_historical_options_backtest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        backtester.print_enhanced_results(results)
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    main() 