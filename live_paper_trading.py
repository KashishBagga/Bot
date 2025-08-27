#!/usr/bin/env python3
"""
Live Paper Trading System
Uses real-time broker data to simulate options trading
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
import threading
from dataclasses import dataclass

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.unified_database import UnifiedDatabase
from src.models.option_contract import OptionContract, OptionChain, OptionType, StrikeSelection
from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
from src.strategies.supertrend_ema import SupertrendEma
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.data.local_data_loader import LocalDataLoader
from src.data.realtime_data_manager import RealTimeDataManager
from src.execution.broker_execution import PaperBrokerAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Represents a paper trade."""
    id: str
    timestamp: datetime
    contract_symbol: str
    underlying: str
    strategy: str
    signal_type: str
    entry_price: float
    quantity: int
    lot_size: int
    strike: float
    expiry: datetime
    option_type: str
    status: str = 'OPEN'
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class RejectedSignal:
    """Represents a rejected signal with reasoning."""
    id: str
    timestamp: datetime
    strategy: str
    signal_type: str
    underlying: str
    price: float
    confidence: float
    reasoning: str
    rejection_reason: str


class LivePaperTradingSystem:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_risk_per_trade: float = 0.02,
        confidence_cutoff: float = 40.0,
        exposure_limit: float = 0.6,
        max_daily_loss_pct: float = 0.03,
        commission_bps: float = 1.0,
        slippage_bps: float = 5.0,
        symbols: List[str] = None,
        data_provider: str = "paper"
    ):
        """Initialize live paper trading system."""
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.confidence_cutoff = float(confidence_cutoff)
        self.exposure_limit = float(exposure_limit)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.symbols = symbols or ['NSE:NIFTY50-INDEX']
        self.data_provider = data_provider

        # Initialize components
        self.db = UnifiedDatabase()
        self.data_loader = LocalDataLoader()
        
        # Ensure database is initialized
        logger.info("🗄️ Initializing database tables...")
        self.db.init_database()
        
        # Real-time data manager - use Fyers if credentials available, otherwise paper broker
        if data_provider == "fyers":
            try:
                from src.data.realtime_data_manager import create_data_provider
                from refresh_fyers_token import check_and_refresh_token
                
                # Get fresh token
                access_token = check_and_refresh_token()
                if access_token:
                    fyers_provider = create_data_provider('fyers', app_id="C607KIH6W0-100", access_token=access_token)
                    if fyers_provider.connect():
                        self.data_manager = RealTimeDataManager(fyers_provider)
                        logger.info("✅ Using Fyers live data with fresh token")
                    else:
                        self.data_manager = RealTimeDataManager([PaperBrokerAPI()])
                        logger.info("⚠️ Fyers connection failed, using paper broker")
                else:
                    self.data_manager = RealTimeDataManager([PaperBrokerAPI()])
                    logger.info("⚠️ Could not get Fyers token, using paper broker")
            except Exception as e:
                logger.error(f"❌ Error setting up Fyers: {e}")
                self.data_manager = RealTimeDataManager([PaperBrokerAPI()])
                logger.info("⚠️ Using paper broker as fallback")
        else:
            self.data_manager = RealTimeDataManager([PaperBrokerAPI()])

        # Trading state
        self.open_trades: Dict[str, PaperTrade] = {}
        self.closed_trades: List[PaperTrade] = []
        self.rejected_signals: List[RejectedSignal] = []
        self.daily_pnl = 0.0
        self.daily_loss_limit_hit = False
        self.max_drawdown = 0.0
        self.peak_capital = float(initial_capital)
        self.equity_curve = []
        self.is_running = False

        # ALL strategies run by default
        self.strategy_instances = {
            'ema_crossover_enhanced': EmaCrossoverEnhanced(),
            'supertrend_ema': SupertrendEma(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
        }

        logger.info(f"🚀 Live Paper Trading System initialized with ₹{initial_capital:,.2f} capital")
        logger.info(f"📊 Symbols: {self.symbols}")
        logger.info(f"📈 ALL Strategies Active: {list(self.strategy_instances.keys())}")
        logger.info(f"🔄 Data Provider: {data_provider}")

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to price."""
        slippage_multiplier = 1 + (self.slippage_bps / 10000)
        if is_buy:
            return price * slippage_multiplier
        else:
            return price / slippage_multiplier

    def _commission_amount(self, notional: float) -> float:
        """Calculate commission amount."""
        return notional * (self.commission_bps / 10000)

    def _current_total_exposure(self) -> float:
        """Calculate current total exposure."""
        total_exposure = 0.0
        for trade in self.open_trades.values():
            total_exposure += trade.entry_price * trade.quantity
        return total_exposure / self.current_capital

    def _should_open_trade(self, signal: Dict) -> Tuple[bool, str]:
        """Check if we should open a trade. Returns (should_open, reason)."""
        # Respect daily stop
        if self.daily_loss_limit_hit:
            return False, "Daily loss limit breached"

        # Confidence threshold
        if float(signal.get('confidence', 0.0)) < self.confidence_cutoff:
            return False, f"Confidence {signal.get('confidence', 0.0)} below threshold {self.confidence_cutoff}"

        # Exposure check
        current_exposure = self._current_total_exposure()
        if current_exposure >= self.exposure_limit:
            return False, f"Exposure {current_exposure:.2%} at limit {self.exposure_limit:.2%}"

        return True, "Signal accepted"

    def _select_option_contract(self, symbol: str, signal_type: str, current_price: float) -> Optional[OptionContract]:
        """Select appropriate option contract based on signal."""
        try:
            # Get live option chain
            option_chain = self.data_manager.get_option_chain(symbol, datetime.now())
            if not option_chain or not option_chain.contracts:
                return None

            # Filter by option type
            if 'CALL' in signal_type.upper():
                contracts = [c for c in option_chain.contracts if c.option_type == OptionType.CALL]
            elif 'PUT' in signal_type.upper():
                contracts = [c for c in option_chain.contracts if c.option_type == OptionType.PUT]
            else:
                return None

            if not contracts:
                return None

            # Select ATM contract (closest to current price)
            atm_contract = min(contracts, key=lambda c: abs(c.strike - current_price))
            
            # Ensure contract has valid price
            if atm_contract.ask <= 0:
                return None

            return atm_contract

        except Exception as e:
            logger.error(f"❌ Error selecting option contract: {e}")
            return None

    def _open_paper_trade(self, signal: Dict, option_contract: OptionContract, 
                         entry_price: float, timestamp: datetime) -> Optional[str]:
        """Open a paper trade."""
        try:
            # Calculate position size
            risk_amount = self.current_capital * self.max_risk_per_trade
            confidence = float(signal.get('confidence', 50))
            confidence_multiplier = min(max(confidence / 50.0, 0.5), 1.5)
            adjusted_risk = risk_amount * confidence_multiplier
            
            premium_per_lot = entry_price * option_contract.lot_size
            max_lots = int(adjusted_risk / premium_per_lot)
            
            # Cap by available capital
            available_capital = self.current_capital * 0.9
            max_affordable_lots = int(available_capital // premium_per_lot)
            max_lots = min(max_lots, max_affordable_lots, 10)  # Max 10 lots
            
            if max_lots < 1:
                if premium_per_lot <= available_capital:
                    max_lots = 1
                else:
                    return None

            # Apply slippage
            exec_price = self._apply_slippage(entry_price, is_buy=True)
            
            # Calculate commission
            notional = max_lots * exec_price * option_contract.lot_size
            commission = self._commission_amount(notional)
            
            # Deduct commission
            self.current_capital -= commission

            # Create trade
            trade_id = f"{option_contract.symbol}_{int(time.time())}"
            trade = PaperTrade(
                id=trade_id,
                timestamp=timestamp,
                contract_symbol=option_contract.symbol,
                underlying=option_contract.underlying,
                strategy=signal['strategy'],
                signal_type=signal['signal'],
                entry_price=exec_price,
                quantity=max_lots * option_contract.lot_size,
                lot_size=option_contract.lot_size,
                strike=option_contract.strike,
                expiry=option_contract.expiry,
                option_type=option_contract.option_type.value
            )

            self.open_trades[trade_id] = trade
            
            # Log trade to database
            self.db.save_open_option_position(trade)
            
            logger.info(f"✅ Opened {signal['signal']} paper trade: {trade_id}")
            logger.info(f"   Contract: {option_contract.symbol} | Size: {max_lots} lots | Premium: ₹{notional:,.2f}")
            
            return trade_id

        except Exception as e:
            logger.error(f"❌ Error opening paper trade: {e}")
            return None

    def _close_paper_trade(self, trade_id: str, exit_price: float, 
                          exit_reason: str, timestamp: datetime) -> Optional[PaperTrade]:
        """Close a paper trade."""
        if trade_id not in self.open_trades:
            return None

        trade = self.open_trades[trade_id]
        
        # Apply slippage
        exec_price = self._apply_slippage(exit_price, is_buy=False)
        
        # Calculate P&L
        entry_value = trade.entry_price * trade.quantity
        exit_value = exec_price * trade.quantity
        exit_commission = self._commission_amount(exit_value)
        
        pnl = (exit_value - entry_value) - exit_commission
        returns = (pnl / entry_value * 100) if entry_value > 0 else 0.0

        # Add P&L to capital
        self.current_capital += exit_value - exit_commission

        # Update trade
        trade.exit_price = exec_price
        trade.exit_time = timestamp
        trade.pnl = pnl
        trade.exit_reason = exit_reason
        trade.status = 'CLOSED'

        # Move to closed trades
        self.closed_trades.append(trade)
        del self.open_trades[trade_id]

        # Update database
        self.db.update_option_position_status(trade_id, 'CLOSED', pnl, exit_reason)

        # Update daily P&L
        self.daily_pnl += pnl

        # Check daily loss limit
        if self.daily_pnl < -(self.initial_capital * self.max_daily_loss_pct):
            self.daily_loss_limit_hit = True
            logger.warning(f"🚫 Daily loss limit breached: PnL={self.daily_pnl:.2f}")

        # Update drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        else:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, drawdown)

        logger.info(f"🔒 Closed paper trade: {trade_id}")
        logger.info(f"   P&L: ₹{pnl:+.2f} ({returns:+.2f}%) | Reason: {exit_reason}")
        
        return trade

    def _check_trade_exits(self, current_prices: Dict[str, float], timestamp: datetime) -> List[PaperTrade]:
        """Check for trade exits based on current prices."""
        closed_trades = []
        
        for trade_id, trade in list(self.open_trades.items()):
            current_price = current_prices.get(trade.contract_symbol)
            if current_price is None:
                continue

            # Premium-based exit logic
            entry_premium = trade.entry_price
            premium_change_pct = (current_price - entry_premium) / entry_premium if entry_premium > 0 else 0

            # Exit conditions
            exit_reason = None
            if premium_change_pct <= -0.5:  # 50% loss
                exit_reason = 'Stop Loss - Premium -50%'
            elif premium_change_pct >= 0.5:  # 50% gain
                exit_reason = 'Target Hit - Premium +50%'
            elif premium_change_pct <= -0.3:  # 30% loss
                exit_reason = 'Stop Loss - Premium -30%'
            elif premium_change_pct >= 0.25:  # 25% gain
                exit_reason = 'Target Hit - Premium +25%'

            if exit_reason:
                closed_trade = self._close_paper_trade(trade_id, current_price, exit_reason, timestamp)
                if closed_trade:
                    closed_trades.append(closed_trade)

        return closed_trades

    def _generate_signals(self, index_data: pd.DataFrame) -> List[Dict]:
        """Generate trading signals from ALL strategies."""
        signals = []
        
        for strategy_name, strategy in self.strategy_instances.items():
            try:
                if hasattr(strategy, 'analyze_vectorized'):
                    signals_df = strategy.analyze_vectorized(index_data)
                    if not signals_df.empty:
                        for idx, row in signals_df.iterrows():
                            signal = {
                                'timestamp': index_data.loc[idx, 'timestamp'],
                                'strategy': strategy_name,
                                'signal': row['signal'],
                                'price': float(row['price']),
                                'confidence': float(row.get('confidence_score', 50)),
                                'reasoning': str(row.get('reasoning', ''))[:200]
                            }
                            signals.append(signal)
            except Exception as e:
                logger.error(f"❌ Error in {strategy_name}: {e}")

        return signals

    def _log_rejected_signal(self, signal: Dict, rejection_reason: str):
        """Log rejected signal with reasoning."""
        rejected_signal = RejectedSignal(
            id=f"rejected_{int(time.time())}",
            timestamp=signal['timestamp'],
            strategy=signal['strategy'],
            signal_type=signal['signal'],
            underlying=signal.get('underlying', ''),
            price=signal['price'],
            confidence=signal['confidence'],
            reasoning=signal.get('reasoning', ''),
            rejection_reason=rejection_reason
        )
        
        self.rejected_signals.append(rejected_signal)
        
        # Log to database
        self.db.save_rejected_signal(rejected_signal)
        
        logger.debug(f"🚫 Rejected signal: {signal['strategy']} {signal['signal']} - {rejection_reason}")

    def _trading_loop(self):
        """Main trading loop."""
        logger.info("🔄 Starting live paper trading loop...")
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check if market is open (9:15 AM to 3:30 PM IST)
                if not self._is_market_open(current_time):
                    # Outside market hours - sleep longer
                    time.sleep(300)  # Sleep for 5 minutes
                    continue

                # Market is open - process trading
                logger.info(f"📊 Market is open - processing signals at {current_time.strftime('%H:%M:%S')}")
                
                # Get current index data
                for symbol in self.symbols:
                    try:
                        # Get real-time index price
                        index_price = self.data_manager.get_underlying_price(symbol)
                        if index_price is None:
                            logger.warning(f"⚠️ Could not get price for {symbol}")
                            continue

                        # Get recent index data for signal generation
                        index_data = self._get_recent_index_data(symbol)
                        if index_data is None or index_data.empty:
                            logger.warning(f"⚠️ Could not get data for {symbol}")
                            continue

                        # Generate signals from ALL strategies
                        signals = self._generate_signals(index_data)
                        
                        if signals:
                            logger.info(f"📈 Generated {len(signals)} signals for {symbol}")
                        
                        # Process new signals
                        for signal in signals:
                            # Check if we should open a trade
                            should_open, reason = self._should_open_trade(signal)
                            
                            if not should_open:
                                self._log_rejected_signal(signal, reason)
                                continue

                            # Select option contract
                            option_contract = self._select_option_contract(symbol, signal['signal'], index_price)
                            if not option_contract:
                                self._log_rejected_signal(signal, "No suitable option contract found")
                                continue

                            # Open trade
                            trade_id = self._open_paper_trade(
                                signal, 
                                option_contract,
                                option_contract.ask,  # Use ask price for buying
                                current_time
                            )

                        # Check for trade exits
                        if self.open_trades:
                            current_prices = {}
                            for trade in self.open_trades.values():
                                price = self.data_manager.get_underlying_price(trade.contract_symbol)
                                if price:
                                    current_prices[trade.contract_symbol] = price

                            if current_prices:
                                closed_trades = self._check_trade_exits(current_prices, current_time)
                                if closed_trades:
                                    logger.info(f"🔒 Closed {len(closed_trades)} trades")

                        # Update equity curve (less frequently)
                        if len(self.equity_curve) == 0 or (current_time - self.equity_curve[-1]['timestamp']).seconds > 300:
                            self.equity_curve.append({
                                'timestamp': current_time,
                                'capital': self.current_capital,
                                'open_trades': len(self.open_trades),
                                'daily_pnl': self.daily_pnl
                            })
                            
                            # Save to database
                            self.db.save_equity_point(current_time, self.current_capital, len(self.open_trades), self.daily_pnl)

                    except Exception as e:
                        logger.error(f"❌ Error processing {symbol}: {e}")

                # Sleep for 2 minutes between iterations during market hours
                time.sleep(120)

            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(60)

    def _is_market_open(self, current_time: datetime) -> bool:
        """Check if market is open."""
        # Convert to IST (UTC+5:30)
        ist_time = current_time + timedelta(hours=5, minutes=30)
        
        # Market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        if ist_time.weekday() >= 5:  # Saturday/Sunday
            return False
            
        market_start = ist_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = ist_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= ist_time <= market_end

    def _get_recent_index_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get recent index data for signal generation."""
        try:
            # Load last 1000 candles for signal generation
            df = self.data_loader.load_data(symbol, '5min', 1000)
            if df is None or df.empty:
                return None

            # Add technical indicators
            from simple_backtest import OptimizedBacktester
            backtester = OptimizedBacktester()
            df = backtester.add_indicators_optimized(df)
            
            return df

        except Exception as e:
            logger.error(f"❌ Error loading index data: {e}")
            return None

    def start_trading(self):
        """Start live paper trading."""
        logger.info("🚀 Starting live paper trading...")
        self.is_running = True
        
        # Start trading loop in a separate thread
        trading_thread = threading.Thread(target=self._trading_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
        logger.info("✅ Live paper trading started successfully")

    def stop_trading(self):
        """Stop live paper trading."""
        logger.info("🛑 Stopping live paper trading...")
        self.is_running = False
        
        # Close all open trades
        current_prices = {}
        for trade in self.open_trades.values():
            current_prices[trade.contract_symbol] = trade.entry_price * 0.5  # Assume 50% loss
        
        for trade_id in list(self.open_trades.keys()):
            self._close_paper_trade(trade_id, current_prices.get(trade_id, 0), 'Trading Stopped', datetime.now())

    def get_performance_report(self) -> Dict:
        """Get performance report."""
        total_trades = len(self.closed_trades)
        winning_trades = len([t for t in self.closed_trades if t.pnl > 0])
        losing_trades = len([t for t in self.closed_trades if t.pnl < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t.pnl for t in self.closed_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        returns = ((self.current_capital - self.initial_capital) / self.initial_capital * 100)
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'returns': returns,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_drawdown': self.max_drawdown * 100,
            'open_trades': len(self.open_trades),
            'daily_pnl': self.daily_pnl,
            'rejected_signals': len(self.rejected_signals)
        }

    def print_performance_report(self):
        """Print performance report."""
        report = self.get_performance_report()
        
        print("\n" + "=" * 60)
        print("📊 LIVE PAPER TRADING PERFORMANCE REPORT")
        print("=" * 60)
        print(f"💰 Capital: ₹{report['current_capital']:,.2f} / ₹{report['initial_capital']:,.2f}")
        print(f"📈 Returns: {report['returns']:+.2f}%")
        print(f"📊 Total Trades: {report['total_trades']}")
        print(f"✅ Wins: {report['winning_trades']} | ❌ Losses: {report['losing_trades']}")
        print(f"🎯 Win Rate: {report['win_rate']:.1f}%")
        print(f"💵 Total P&L: ₹{report['total_pnl']:+.2f}")
        print(f"📊 Avg P&L: ₹{report['avg_pnl']:+.2f}")
        print(f"📉 Max Drawdown: {report['max_drawdown']:.2f}%")
        print(f"📈 Open Trades: {report['open_trades']}")
        print(f"📊 Daily P&L: ₹{report['daily_pnl']:+.2f}")
        print(f"🚫 Rejected Signals: {report['rejected_signals']}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Live Paper Trading System')
    parser.add_argument('--symbols', nargs='+', default=['NSE:NIFTY50-INDEX'], help='Trading symbols')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.02, help='Max risk per trade')
    parser.add_argument('--confidence', type=float, default=40.0, help='Min confidence to open trades')
    parser.add_argument('--exposure', type=float, default=0.6, help='Max portfolio exposure (0-1)')
    parser.add_argument('--daily_loss', type=float, default=0.03, help='Max daily loss percent (0-1)')
    parser.add_argument('--commission_bps', type=float, default=1.0, help='Commission in bps')
    parser.add_argument('--slippage_bps', type=float, default=5.0, help='Slippage in bps')
    parser.add_argument('--data_provider', type=str, default='paper', help='Data provider (paper/fyers)')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode for a few minutes')

    args = parser.parse_args()

    # Initialize trading system
    trading_system = LivePaperTradingSystem(
        initial_capital=args.capital,
        max_risk_per_trade=args.risk,
        confidence_cutoff=args.confidence,
        exposure_limit=args.exposure,
        max_daily_loss_pct=args.daily_loss,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        symbols=args.symbols,
        data_provider=args.data_provider
    )

    try:
        # Start trading
        trading_system.start_trading()
        
        if args.test_mode:
            # Test mode - run for 5 minutes
            logger.info("🧪 Running in test mode for 5 minutes...")
            time.sleep(300)  # 5 minutes
            trading_system.stop_trading()
            trading_system.print_performance_report()
        else:
            # Continuous mode - run during market hours
            logger.info("🔄 Starting continuous paper trading during market hours...")
            logger.info("📅 Trading will run from 9:15 AM to 3:30 PM IST, Monday to Friday")
            logger.info("⏹️ Press Ctrl+C to stop trading")
            
            # Run continuously until interrupted
            while True:
                time.sleep(60)  # Check every minute
                
                # Check if it's a new trading day
                current_time = datetime.now()
                if trading_system._is_market_open(current_time):
                    # Reset daily limits at market open
                    if current_time.hour == 9 and current_time.minute == 15:
                        trading_system.daily_pnl = 0.0
                        trading_system.daily_loss_limit_hit = False
                        logger.info("🆕 New trading day started - resetting daily limits")
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
        trading_system.stop_trading()
        trading_system.print_performance_report()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        trading_system.stop_trading()


if __name__ == "__main__":
    main() 