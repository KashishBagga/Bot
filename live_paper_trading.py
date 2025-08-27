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
from typing import Dict, List, Optional, Tuple, Any
import json
import sqlite3
from pathlib import Path
import uuid
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
    def __init__(self, initial_capital: float = 100000.0, max_risk_per_trade: float = 0.02,
                 confidence_cutoff: float = 40.0, exposure_limit: float = 0.6,
                 max_daily_loss_pct: float = 0.03, commission_bps: float = 1.0,
                 slippage_bps: float = 5.0, symbols: List[str] = None, data_provider: str = 'paper'):
        """Initialize the live paper trading system."""
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.confidence_cutoff = float(confidence_cutoff)
        self.exposure_limit = float(exposure_limit)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.symbols = symbols or ['NSE:NIFTY50-INDEX']
        
        # Trading state
        self.open_trades = {}
        self.closed_trades = []
        self.daily_pnl = 0.0
        self.daily_loss_limit_hit = False
        self.is_running = False
        
        # Data caching for performance
        self.data_cache = {}  # symbol -> (data, last_update_time)
        self.cache_duration = 60  # seconds - refresh cache every minute
        self.max_cached_candles = 1000
        
        # Performance tracking
        self.total_signals_generated = 0
        self.total_signals_rejected = 0
        self.total_trades_executed = 0
        self.session_start_time = datetime.now()
        
        # Initialize components
        self.db = UnifiedDatabase()
        self.data_loader = LocalDataLoader()
        
        # Ensure database is initialized
        logger.info("🗄️ Initializing database tables...")
        self.db.init_database()
        
        # Real-time data manager - use Fyers if credentials available, otherwise paper broker
        if data_provider == 'fyers':
            try:
                from src.data.realtime_data_manager import FyersDataProvider
                from refresh_fyers_token import check_and_refresh_token
                
                # Get fresh token
                access_token = check_and_refresh_token()
                if access_token:
                    self.data_manager = FyersDataProvider(app_id="C607KIH6W0-100", access_token=access_token)
                    logger.info("✅ Using Fyers live data with fresh token")
                else:
                    self.data_manager = None
                    logger.warning("⚠️ Could not get Fyers token, using paper data")
            except Exception as e:
                logger.warning(f"⚠️ Fyers not available, falling back to paper data: {e}")
                self.data_manager = None
        else:
            self.data_manager = None
            logger.info("✅ Using paper data simulation")
        
        # Initialize strategies
        self.strategies = {
            'ema_crossover_enhanced': EmaCrossoverEnhanced(),
            'supertrend_ema': SupertrendEma(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
        }
        
        logger.info(f"🚀 Live Paper Trading System initialized with ₹{initial_capital:,.2f} capital")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(f"🎯 Strategies: {', '.join(self.strategies.keys())}")
        logger.info(f"📛 Risk: {max_risk_per_trade*100:.1f}% per trade, {exposure_limit*100:.1f}% max exposure")
        logger.info(f"📉 Daily loss limit: {max_daily_loss_pct*100:.1f}%")

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

    def _open_paper_trade(self, signal: Dict, option_contract: Any, entry_price: float, timestamp: datetime) -> Optional[str]:
        """Open a new paper trade with improved position sizing and risk management."""
        try:
            # Generate unique trade ID using UUID
            trade_id = str(uuid.uuid4())
            
            # Calculate position size with dynamic lot sizing
            position_size = self._calculate_dynamic_position_size(signal, entry_price)
            if position_size <= 0:
                logger.info(f"⚠️ Position size too small for {signal['strategy']}")
                return None
            
            # Calculate notional value
            notional_value = position_size * entry_price
            
            # Check exposure limits with symbol-wise tracking
            if not self._check_exposure_limits(signal['symbol'], notional_value):
                logger.warning(f"🚫 Exposure limit exceeded for {signal['symbol']}")
                return None
            
            # Apply slippage and commission
            execution_price = self._apply_slippage(entry_price, True)  # True for buy
            commission = self._commission_amount(notional_value)
            
            # Create trade object
            trade = PaperTrade(
                id=trade_id,
                timestamp=timestamp,
                contract_symbol=signal['symbol'],
                underlying=option_contract.underlying,
                strategy=signal['strategy'],
                signal_type=signal['signal'],
                entry_price=execution_price,
                quantity=position_size,
                lot_size=option_contract.lot_size,
                strike=option_contract.strike,
                expiry=option_contract.expiry,
                option_type=option_contract.option_type.value,
                status='OPEN',
                commission=commission,
                confidence=signal.get('confidence', 0),
                reasoning=signal.get('reasoning', ''),
                stop_loss=signal.get('stop_loss'),
                target1=signal.get('target1'),
                target2=signal.get('target2'),
                target3=signal.get('target3')
            )
            
            # Deduct capital
            self.current_capital -= (notional_value + commission)
            
            # Store trade
            self.open_trades[trade_id] = trade
            
            # Log to database
            self.db.save_open_option_position(trade)
            
            self.total_trades_executed += 1
            
            logger.info(f"✅ Opened trade {trade_id[:8]}... | {signal['signal']} {signal['strategy']}")
            logger.info(f"   Symbol: {signal['symbol']} | Size: {position_size} | Price: ₹{execution_price:.2f}")
            logger.info(f"   Notional: ₹{notional_value:,.2f} | Commission: ₹{commission:.2f}")
            
            return trade_id
            
        except Exception as e:
            logger.error(f"❌ Error opening trade: {e}")
            return None
    
    def _calculate_dynamic_position_size(self, signal: Dict, entry_price: float) -> int:
        """Calculate position size with dynamic lot sizing based on confidence and volatility."""
        try:
            # Base risk amount
            base_risk = self.current_capital * self.max_risk_per_trade
            
            # Confidence multiplier (0.5 to 2.0 based on confidence)
            confidence = signal.get('confidence', 50)
            confidence_multiplier = max(0.5, min(2.0, confidence / 50.0))
            
            # Volatility adjustment (if we have ATR data)
            volatility_multiplier = 1.0
            if 'atr' in signal:
                # Adjust based on ATR - higher volatility = smaller position
                atr = signal['atr']
                if atr and atr > 0:
                    volatility_multiplier = max(0.5, min(1.5, 1.0 / (atr / entry_price)))
            
            # Calculate adjusted risk
            adjusted_risk = base_risk * confidence_multiplier * volatility_multiplier
            
            # Calculate position size
            if signal.get('stop_loss'):
                risk_per_unit = abs(entry_price - signal['stop_loss'])
            else:
                # Default 2% risk per unit if no stop loss
                risk_per_unit = entry_price * 0.02
            
            if risk_per_unit <= 0:
                return 0
            
            position_size = int(adjusted_risk / risk_per_unit)
            
            # Ensure minimum position size
            min_position = 1
            if position_size < min_position:
                position_size = min_position if adjusted_risk >= risk_per_unit else 0
            
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0
    
    def _check_exposure_limits(self, symbol: str, new_notional: float) -> bool:
        """Check exposure limits with symbol-wise tracking."""
        try:
            # Calculate current exposure by symbol
            symbol_exposure = 0.0
            total_exposure = 0.0
            
            for trade in self.open_trades.values():
                trade_notional = trade.entry_price * trade.quantity
                total_exposure += trade_notional
                if trade.contract_symbol == symbol:
                    symbol_exposure += trade_notional
            
            # Add new trade
            total_exposure += new_notional
            symbol_exposure += new_notional
            
            # Check total exposure limit
            if total_exposure / self.current_capital > self.exposure_limit:
                logger.warning(f"🚫 Total exposure limit exceeded: {total_exposure/self.current_capital:.2%}")
                return False
            
            # Check symbol-specific exposure limit (50% of total limit)
            symbol_limit = self.exposure_limit * 0.5
            if symbol_exposure / self.current_capital > symbol_limit:
                logger.warning(f"🚫 Symbol exposure limit exceeded for {symbol}: {symbol_exposure/self.current_capital:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking exposure limits: {e}")
            return False

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

    def _check_trade_exits(self, current_prices: Dict[str, float], timestamp: datetime) -> List[Dict]:
        """Check for trade exits using vectorized operations and improved exit logic."""
        if not self.open_trades:
            return []
        
        closed_trades = []
        trades_to_close = []
        
        # Create DataFrame for vectorized operations
        trades_data = []
        for trade_id, trade in self.open_trades.items():
            if trade.contract_symbol in current_prices:
                trades_data.append({
                    'trade_id': trade_id,
                    'trade': trade,
                    'current_price': current_prices[trade.contract_symbol],
                    'entry_price': trade.entry_price,
                    'signal_type': trade.signal_type,
                    'stop_loss': trade.stop_loss,
                    'target1': trade.target1,
                    'target2': trade.target2,
                    'target3': trade.target3,
                    'entry_time': trade.timestamp
                })
        
        if not trades_data:
            return []
        
        # Convert to DataFrame for vectorized operations
        df = pd.DataFrame(trades_data)
        
        # Calculate price movements
        df['price_change_pct'] = (df['current_price'] - df['entry_price']) / df['entry_price']
        
        # Check exit conditions vectorized
        for _, row in df.iterrows():
            trade_id = row['trade_id']
            trade = row['trade']
            current_price = row['current_price']
            price_change_pct = row['price_change_pct']
            
            exit_reason = None
            should_exit = False
            
            # Check stop loss
            if trade.stop_loss:
                if (trade.signal_type == 'BUY CALL' and current_price <= trade.stop_loss) or \
                   (trade.signal_type == 'BUY PUT' and current_price >= trade.stop_loss):
                    exit_reason = 'Stop Loss'
                    should_exit = True
            
            # Check targets (priority order)
            if not should_exit and trade.target3 and abs(price_change_pct) >= 0.06:  # 6% target
                exit_reason = 'Target 3'
                should_exit = True
            elif not should_exit and trade.target2 and abs(price_change_pct) >= 0.04:  # 4% target
                exit_reason = 'Target 2'
                should_exit = True
            elif not should_exit and trade.target1 and abs(price_change_pct) >= 0.025:  # 2.5% target
                exit_reason = 'Target 1'
                should_exit = True
            
            # Check time-based exit (if trade is older than 4 hours)
            if not should_exit:
                trade_duration = (timestamp - trade.timestamp).total_seconds() / 3600
                if trade_duration > 4:  # 4 hours
                    exit_reason = 'Time Exit'
                    should_exit = True
            
            # Check trailing stop (if trade is in profit)
            if not should_exit and price_change_pct > 0.02:  # 2% profit
                # Implement trailing stop logic here
                # For now, use simple trailing stop
                trailing_stop_pct = 0.01  # 1% trailing stop
                if price_change_pct < trailing_stop_pct:
                    exit_reason = 'Trailing Stop'
                    should_exit = True
            
            if should_exit:
                trades_to_close.append((trade_id, current_price, exit_reason))
        
        # Close trades
        for trade_id, exit_price, exit_reason in trades_to_close:
            closed_trade = self._close_paper_trade(trade_id, exit_price, exit_reason, timestamp)
            if closed_trade:
                closed_trades.append(closed_trade)
        
        return closed_trades

    def _generate_signals(self, index_data: pd.DataFrame) -> List[Dict]:
        """Generate trading signals from ALL strategies."""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
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

    def _log_rejected_signal(self, signal: Dict, reason: str):
        """Log rejected signal with detailed reasoning for debugging."""
        try:
            rejected_signal = RejectedSignal(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                strategy=signal['strategy'],
                signal_type=signal['signal'],
                underlying=signal['symbol'],
                price=signal['price'],
                confidence=signal.get('confidence', 0),
                reasoning=signal.get('reasoning', ''),
                rejection_reason=reason
            )
            
            self.rejected_signals.append(rejected_signal)
            self.db.save_rejected_signal(rejected_signal)
            
            self.total_signals_rejected += 1
            
            logger.info(f"🚫 Rejected signal: {signal['strategy']} {signal['signal']} | Reason: {reason}")
            
        except Exception as e:
            logger.error(f"❌ Error logging rejected signal: {e}")
    
    def _log_signal_generation(self, symbol: str, signals: List[Dict]):
        """Log signal generation with detailed metrics."""
        try:
            logger.info(f"📈 Generated {len(signals)} signals for {symbol}")
            
            # Log signal details
            for signal in signals:
                logger.info(f"   {signal['strategy']}: {signal['signal']} @ ₹{signal['price']:.2f} "
                          f"(Confidence: {signal.get('confidence', 0):.1f})")
            
            self.total_signals_generated += len(signals)
            
        except Exception as e:
            logger.error(f"❌ Error logging signal generation: {e}")
    
    def _update_performance_metrics(self):
        """Update and log performance metrics."""
        try:
            # Calculate current metrics
            total_trades = len(self.closed_trades)
            winning_trades = len([t for t in self.closed_trades if t.pnl > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = sum(t.pnl for t in self.closed_trades)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            
            # Calculate drawdown
            peak_capital = self.initial_capital
            current_capital = self.current_capital
            for trade in self.closed_trades:
                if trade.pnl > 0:
                    current_capital += trade.pnl
                    peak_capital = max(peak_capital, current_capital)
            
            drawdown = ((peak_capital - current_capital) / peak_capital * 100) if peak_capital > 0 else 0
            
            # Log performance summary
            logger.info(f"📊 Performance Update:")
            logger.info(f"   Trades: {total_trades} | Wins: {winning_trades} | Win Rate: {win_rate:.1f}%")
            logger.info(f"   P&L: ₹{total_pnl:+.2f} | Avg: ₹{avg_pnl:+.2f}")
            logger.info(f"   Capital: ₹{self.current_capital:,.2f} | Drawdown: {drawdown:.2f}%")
            logger.info(f"   Signals: {self.total_signals_generated} generated, {self.total_signals_rejected} rejected")
            
            # Save to database
            self.db.save_performance_metrics(
                timestamp=datetime.now(),
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=total_trades - winning_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_pnl=avg_pnl,
                max_drawdown=drawdown
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    def _generate_session_report(self) -> Dict:
        """Generate comprehensive session report."""
        try:
            session_duration = datetime.now() - self.session_start_time
            
            # Calculate metrics
            total_trades = len(self.closed_trades)
            winning_trades = len([t for t in self.closed_trades if t.pnl > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = sum(t.pnl for t in self.closed_trades)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            
            # Strategy performance
            strategy_performance = {}
            for trade in self.closed_trades:
                strategy = trade.strategy
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'trades': 0, 'wins': 0, 'pnl': 0}
                
                strategy_performance[strategy]['trades'] += 1
                if trade.pnl > 0:
                    strategy_performance[strategy]['wins'] += 1
                strategy_performance[strategy]['pnl'] += trade.pnl
            
            # Calculate win rates for strategies
            for strategy in strategy_performance:
                trades = strategy_performance[strategy]['trades']
                wins = strategy_performance[strategy]['wins']
                strategy_performance[strategy]['win_rate'] = (wins / trades * 100) if trades > 0 else 0
            
            report = {
                'session_duration': str(session_duration),
                'total_signals_generated': self.total_signals_generated,
                'total_signals_rejected': self.total_signals_rejected,
                'total_trades_executed': self.total_trades_executed,
                'total_trades_closed': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital * 100),
                'strategy_performance': strategy_performance,
                'open_trades': len(self.open_trades)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating session report: {e}")
            return {}

    def _trading_loop(self):
        """Main trading loop with improved monitoring and performance tracking."""
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
                        index_price = self.data_manager.get_underlying_price(symbol) if self.data_manager else None
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
                            self._log_signal_generation(symbol, signals)
                        
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
                                price = self.data_manager.get_underlying_price(trade.contract_symbol) if self.data_manager else None
                                if price:
                                    current_prices[trade.contract_symbol] = price

                            if current_prices:
                                closed_trades = self._check_trade_exits(current_prices, current_time)
                                if closed_trades:
                                    logger.info(f"🔒 Closed {len(closed_trades)} trades")

                        # Update performance metrics every 10 trades
                        if len(self.closed_trades) % 10 == 0 and len(self.closed_trades) > 0:
                            self._update_performance_metrics()

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
            # Check cache first
            if symbol in self.data_cache and \
               (datetime.now() - self.data_cache[symbol][1]).total_seconds() < self.cache_duration:
                logger.debug(f"Using cached data for {symbol}")
                return self.data_cache[symbol][0]

            # Load last 1000 candles for signal generation
            df = self.data_loader.load_data(symbol, '5min', self.max_cached_candles)
            if df is None or df.empty:
                return None

            # Add technical indicators
            from simple_backtest import OptimizedBacktester
            backtester = OptimizedBacktester()
            df = backtester.add_indicators_optimized(df)
            
            # Cache the data
            self.data_cache[symbol] = (df, datetime.now())
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
        """Stop the trading system and generate final report."""
        logger.info("🛑 Stopping paper trading system...")
        self.is_running = False
        
        # Close all open positions
        if self.open_trades:
            logger.info(f"🔒 Closing {len(self.open_trades)} open positions...")
            for trade_id in list(self.open_trades.keys()):
                # Use current market price or last known price
                exit_price = self.open_trades[trade_id].entry_price * 0.5  # Assume 50% loss for manual close
                self._close_paper_trade(trade_id, exit_price, "Manual Close", datetime.now())
        
        # Generate final session report
        report = self._generate_session_report()
        
        # Print comprehensive session summary
        self._print_session_summary(report)
        
        logger.info("✅ Paper trading system stopped")
    
    def _print_session_summary(self, report: Dict):
        """Print comprehensive session summary."""
        if not report:
            return
        
        print("\n" + "=" * 80)
        print("📊 PAPER TRADING SESSION SUMMARY")
        print("=" * 80)
        print(f"⏱️ Session Duration: {report.get('session_duration', 'N/A')}")
        print(f"💰 Initial Capital: ₹{report.get('initial_capital', 0):,.2f}")
        print(f"💰 Final Capital: ₹{report.get('final_capital', 0):,.2f}")
        print(f"📈 Return: {report.get('return_pct', 0):+.2f}%")
        print()
        
        print("📊 SIGNAL GENERATION:")
        print(f"   Generated: {report.get('total_signals_generated', 0)}")
        print(f"   Rejected: {report.get('total_signals_rejected', 0)}")
        print(f"   Execution Rate: {((report.get('total_trades_executed', 0) / max(report.get('total_signals_generated', 1), 1)) * 100):.1f}%")
        print()
        
        print("📈 TRADE PERFORMANCE:")
        print(f"   Total Trades: {report.get('total_trades_closed', 0)}")
        print(f"   Winning Trades: {report.get('winning_trades', 0)}")
        print(f"   Losing Trades: {report.get('losing_trades', 0)}")
        print(f"   Win Rate: {report.get('win_rate', 0):.1f}%")
        print(f"   Total P&L: ₹{report.get('total_pnl', 0):+.2f}")
        print(f"   Average P&L: ₹{report.get('avg_pnl', 0):+.2f}")
        print()
        
        print("🎯 STRATEGY PERFORMANCE:")
        strategy_perf = report.get('strategy_performance', {})
        for strategy, perf in strategy_perf.items():
            status = "✅ PROFITABLE" if perf['pnl'] > 0 else "❌ LOSS"
            print(f"   {strategy}:")
            print(f"     Trades: {perf['trades']} | Win Rate: {perf['win_rate']:.1f}%")
            print(f"     P&L: ₹{perf['pnl']:+.2f} | Status: {status}")
        print()
        
        print("🔓 OPEN POSITIONS:")
        print(f"   Remaining: {report.get('open_trades', 0)}")
        print()
        
        print("=" * 80)


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