#!/usr/bin/env python3
"""
Options Trading Bot
Extended paper trading bot with options-specific functionality
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime, date
from typing import Dict, List, Optional, Tuple
import json
import sqlite3
from pathlib import Path

# timezone helper
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    try:
        import pytz
        ZoneInfo = None
    except Exception:
        ZoneInfo = None

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.unified_database import UnifiedDatabase
from src.models.option_contract import OptionContract, OptionChain, OptionType, StrikeSelection
from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
from src.strategies.supertrend_ema import SupertrendEma
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.data.option_chain_loader import OptionChainLoader
from src.core.option_signal_mapper import OptionSignalMapper
from simple_backtest import OptimizedBacktester
from src.data.local_data_loader import LocalDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('options_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptionsTradingBot:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_risk_per_trade: float = 0.02,
        confidence_cutoff: float = 40.0,
        exposure_limit: float = 0.6,
        max_daily_loss_pct: float = 0.03,
        max_holding_minutes: Optional[int] = 24 * 60,
        trailing_stop_pct: Optional[float] = 0.01,
        commission_bps: float = 1.0,
        slippage_bps: float = 5.0,
        timezone: str = "Asia/Kolkata",
        expiry_type: str = "weekly",
        strike_selection: StrikeSelection = StrikeSelection.ATM,
        delta_target: float = 0.30
    ):
        """Initialize options trading bot with enhanced risk & safety features."""
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.confidence_cutoff = float(confidence_cutoff)
        self.exposure_limit = float(exposure_limit)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.max_holding_minutes = max_holding_minutes
        self.trailing_stop_pct = trailing_stop_pct
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

        # timezone handling
        self.tz_name = timezone
        if ZoneInfo is not None:
            self.tzinfo = ZoneInfo(self.tz_name)
        else:
            import pytz as _pytz
            self.tzinfo = _pytz.timezone(self.tz_name)

        # runtime state
        self.open_positions: Dict[str, Dict] = {}
        self.closed_positions: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.daily_pnl = 0.0
        self.daily_loss_limit_hit = False
        self.max_drawdown = 0.0
        self.peak_capital = float(initial_capital)
        self.nav_at_start_of_day = float(initial_capital)
        self.last_day = self._now_tz().date()

        # Initialize strategies
        self.strategies = {
            'ema_crossover_enhanced': EmaCrossoverEnhanced(),
            'supertrend_ema': SupertrendEma(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
        }

        # Initialize data loaders and mappers
        self.db = UnifiedDatabase()
        self.data_loader = LocalDataLoader()
        self.option_loader = OptionChainLoader()
        self.signal_mapper = OptionSignalMapper(self.option_loader)
        self.backtester = OptimizedBacktester()

        # Set options-specific parameters
        self.signal_mapper.set_parameters(
            expiry_type=expiry_type,
            strike_selection=strike_selection,
            delta_target=delta_target
        )

        # Trading session state
        self.session_start = self._now_tz()
        self.is_running = False

        logger.info(f"🚀 Options Trading Bot initialized with ₹{initial_capital:,.2f} capital")
        logger.info(f"📊 Max risk per trade: {max_risk_per_trade*100:.1f}% | Confidence cutoff: {self.confidence_cutoff}")
        logger.info(f"📛 Exposure limit: {self.exposure_limit*100:.1f}%, Daily loss cap: {self.max_daily_loss_pct*100:.1f}%")
        logger.info(f"🕰️ Timezone: {self.tz_name} | Expiry: {expiry_type} | Strike: {strike_selection.value}")

    # -------------------------
    # small helpers
    # -------------------------
    def _now_tz(self) -> datetime:
        """Return timezone-aware now in configured timezone."""
        now = datetime.now().astimezone(self.tzinfo) if hasattr(datetime.now(), "astimezone") else datetime.now()
        try:
            return now.astimezone(self.tzinfo)
        except Exception:
            return now

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage in bps to a price."""
        factor = 1.0 + (self.slippage_bps / 10000.0) if is_buy else 1.0 - (self.slippage_bps / 10000.0)
        return float(price) * factor

    def _commission_amount(self, notional: float) -> float:
        """Return commission amount from notional based on bps."""
        return abs(notional) * (self.commission_bps / 10000.0)

    def _current_total_exposure(self) -> float:
        """Return total notional exposure across open positions."""
        tot = 0.0
        for pos in self.open_positions.values():
            tot += pos['premium_risk']  # Use premium risk for options
        nav = self.current_capital + tot
        if nav <= 0:
            return 1.0
        return tot / nav

    # -------------------------
    # data & signals
    # -------------------------
    def get_latest_data(self, symbol: str, timeframe: str, lookback_candles: int = 200) -> pd.DataFrame:
        """Get latest market data for analysis."""
        try:
            df = self.data_loader.load_data(symbol, timeframe, lookback_candles)

            if df is None or df.empty:
                logger.warning(f"⚠️ No data available for {symbol} {timeframe}")
                return pd.DataFrame()

            df = self.backtester.add_indicators_optimized(df)

            if len(df) < 100:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} candles")
                return pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def generate_option_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Generate option trading signals from index signals."""
        try:
            # Generate index signals first
            index_signals = []
            for strategy_name, strategy in self.strategies.items():
                try:
                    if hasattr(strategy, 'analyze_vectorized'):
                        signals_df = strategy.analyze_vectorized(df)
                        if not signals_df.empty:
                            for idx, row in signals_df.tail(10).iterrows():
                                signal = {
                                    'timestamp': df.loc[idx, 'timestamp'],
                                    'strategy': strategy_name,
                                    'signal': row['signal'],
                                    'price': float(row['price']),
                                    'confidence': float(row.get('confidence_score', 0)),
                                    'reasoning': str(row.get('reasoning', ''))[:200],
                                    'stop_loss': float(row.get('stop_loss', 0) or 0),
                                    'target1': float(row.get('target1', 0) or 0),
                                    'target2': float(row.get('target2', 0) or 0),
                                    'target3': float(row.get('target3', 0) or 0),
                                    'position_multiplier': float(row.get('position_multiplier', 1.0)),
                                    'symbol': symbol,
                                    'capital': self.current_capital,
                                    'max_risk_per_trade': self.max_risk_per_trade
                                }
                                index_signals.append(signal)
                except Exception as e:
                    logger.error(f"❌ Error in {strategy_name}: {e}")
                    continue

            # Map index signals to option signals
            current_price = float(df['close'].iloc[-1])
            current_time = df['timestamp'].iloc[-1]
            
            option_signals = self.signal_mapper.map_multiple_signals(
                index_signals, current_price, current_time
            )

            logger.info(f"✅ Generated {len(option_signals)} option signals from {len(index_signals)} index signals")
            return option_signals

        except Exception as e:
            logger.error(f"❌ Error generating option signals: {e}")
            return []

    # -------------------------
    # execution / risk checks
    # -------------------------
    def should_open_option_position(self, option_signal: Dict) -> bool:
        """Check if we should open an option position."""
        # respect daily-stop
        if self.daily_loss_limit_hit:
            logger.warning("🚫 Daily loss limit hit — no new positions today.")
            return False

        # unique contract constraint (avoid duplicates)
        contract_symbol = option_signal['contract'].symbol
        for pos in self.open_positions.values():
            if pos['contract_symbol'] == contract_symbol:
                return False

        # confidence threshold
        if float(option_signal.get('confidence', 0.0)) < self.confidence_cutoff:
            return False

        # exposure check - calculate hypothetical exposure including new position
        try:
            contract = option_signal.get('contract')
            qty = int(option_signal.get('quantity', 1))
            entry_px = float(option_signal.get('entry_price', getattr(contract, 'ask', 0.0)))
            new_premium = entry_px * qty * getattr(contract, 'lot_size', 1)
        except Exception:
            new_premium = 0.0

        current_premium = sum(p['premium_risk'] for p in self.open_positions.values())
        nav = max(self.current_capital + current_premium, 1.0)
        est_exposure_after = (current_premium + new_premium) / nav

        if est_exposure_after >= self.exposure_limit:
            logger.warning(f"🚫 Exposure would exceed limit: {est_exposure_after:.2%} >= {self.exposure_limit:.2%}")
            return False

        return True

    def open_option_position(self, option_signal: Dict) -> Optional[str]:
        """Open a new option position."""
        try:
            contract = option_signal['contract']
            quantity = option_signal['quantity']
            entry_price = option_signal['entry_price']

            # Calculate total premium cost
            total_premium = entry_price * quantity * contract.lot_size

            # Check capital availability
            if total_premium > self.current_capital * 0.8:
                logger.warning(f"🚫 Insufficient capital for option: required ₹{total_premium:,.2f}")
                return None

            # Apply commission
            commission = self._commission_amount(total_premium)
            total_cost = total_premium + commission

            if total_cost > self.current_capital:
                logger.warning(f"🚫 Insufficient capital after commission: required ₹{total_cost:,.2f}")
                return None

            # Generate position ID
            position_id = f"{contract.symbol}_{option_signal['strategy']}_{int(time.time())}"

            # Create position
            position = {
                'id': position_id,
                'contract_symbol': contract.symbol,
                'underlying': contract.underlying,
                'strategy': option_signal['strategy'],
                'signal_type': option_signal['signal_type'],
                'entry_price': entry_price,
                'quantity': quantity,  # Number of lots
                'total_quantity': quantity * contract.lot_size,  # Total shares
                'strike': contract.strike,
                'expiry': contract.expiry,
                'option_type': contract.option_type.value,
                'lot_size': contract.lot_size,
                'entry_time': self._now_tz(),
                'confidence': option_signal['confidence'],
                'reasoning': option_signal['reasoning'],
                'premium_risk': total_premium,
                'entry_commission': commission,  # Store entry commission separately
                'delta': contract.delta,
                'gamma': contract.gamma,
                'theta': contract.theta,
                'vega': contract.vega,
                'status': 'OPEN'
            }

            # Deduct capital
            self.current_capital -= total_cost

            # Store position
            self.open_positions[position_id] = position

            # Persist open position to database
            try:
                self.db.save_open_option_position({
                    'id': position_id,
                    'contract_symbol': contract.symbol,
                    'underlying': contract.underlying,
                    'strategy': option_signal['strategy'],
                    'entry_time': position['entry_time'].isoformat(),
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'premium_risk': total_premium,
                    'strike': contract.strike,
                    'expiry': contract.expiry.isoformat(),
                    'option_type': contract.option_type.value,
                    'lot_size': contract.lot_size
                })
            except Exception as e:
                logger.warning(f"Failed to save open position to DB: {e}")

            # Safe signal type extraction
            signal_type = option_signal.get('signal_type') or option_signal.get('signal') or 'UNKNOWN'
            
            logger.info(f"✅ Opened {signal_type} option position: {position_id}")
            logger.info(f"   Contract: {contract.symbol} (Strike: {contract.strike}, Expiry: {contract.expiry.strftime('%Y-%m-%d')})")
            logger.info(f"   Quantity: {quantity} lots ({position['total_quantity']} shares)")
            logger.info(f"   Premium: ₹{entry_price:.2f} per share, Total: ₹{total_premium:,.2f}")
            logger.info(f"   Commission: ₹{commission:.2f}")
            logger.info(f"   Greeks: Delta={contract.delta:.3f}, Gamma={contract.gamma:.3f}, Theta={contract.theta:.3f}")

            return position_id

        except Exception as e:
            logger.error(f"❌ Error opening option position: {e}")
            return None

    def check_option_position_exits(self, current_price: float, symbol: str) -> List[Dict]:
        """Check if any open option positions should be closed."""
        closed_positions = []
        positions_to_close = []

        for position_id, position in self.open_positions.items():
            if position['underlying'] != symbol:
                continue

            # Time-based exit
            if self._should_force_time_exit(position['entry_time']):
                positions_to_close.append((position_id, current_price, 'Time Exit'))
                continue

            # Premium-based exit logic (more realistic for options)
            current_premium = position.get('current_premium', position['entry_price'])
            entry_premium = position['entry_price']
            
            # Calculate premium change percentage
            premium_change_pct = (current_premium - entry_premium) / entry_premium if entry_premium > 0 else 0
            
            # Exit rules based on premium P&L
            if premium_change_pct <= -0.5:  # Premium halved (50% loss)
                positions_to_close.append((position_id, current_premium, 'Stop Loss - Premium -50%'))
            elif premium_change_pct >= 1.0:  # Premium doubled (100% gain)
                positions_to_close.append((position_id, current_premium, 'Target Hit - Premium +100%'))
            elif premium_change_pct <= -0.3:  # 30% loss (conservative stop)
                positions_to_close.append((position_id, current_premium, 'Stop Loss - Premium -30%'))
            elif premium_change_pct >= 0.5:  # 50% gain (take profit)
                positions_to_close.append((position_id, current_premium, 'Target Hit - Premium +50%'))

        # Process closures
        for position_id, exit_price, exit_reason in positions_to_close:
            closed_position = self.close_option_position(position_id, exit_price, exit_reason)
            if closed_position:
                closed_positions.append(closed_position)

        # Daily loss check
        self._update_daily_loss_and_check()

        return closed_positions

    def _should_force_time_exit(self, entry_time: datetime) -> bool:
        """Check if position should be force-exited due to time."""
        if self.max_holding_minutes is None:
            return False
        delta_min = (self._now_tz() - entry_time).total_seconds() / 60.0
        return delta_min >= float(self.max_holding_minutes)

    def close_option_position(self, position_id: str, exit_price: float, exit_reason: str) -> Optional[Dict]:
        """Close an option position."""
        if position_id not in self.open_positions:
            return None

        position = self.open_positions[position_id]

        # Calculate P&L with proper commission accounting
        entry_value = position['entry_price'] * position['total_quantity']
        exit_value = exit_price * position['total_quantity']
        entry_commission = position.get('entry_commission', 0)
        exit_commission = self._commission_amount(exit_value)
        
        # P&L = (Exit value - Entry value) - (Entry commission + Exit commission)
        pnl = (exit_value - entry_value) - (entry_commission + exit_commission)

        # Add capital back
        self.current_capital += exit_value - exit_commission

        # Calculate returns
        returns = (pnl / entry_value) * 100 if entry_value > 0 else 0.0

        # Create closed position record
        closed_position = {
            **position,
            'exit_price': exit_price,
            'exit_time': self._now_tz(),
            'exit_reason': exit_reason,
            'pnl': pnl,
            'returns': returns,
            'duration': (self._now_tz() - position['entry_time']).total_seconds() / 60.0,
            'exit_commission': exit_commission,
            'status': 'CLOSED'
        }

        # Remove from open positions
        del self.open_positions[position_id]
        self.closed_positions.append(closed_position)
        self.trade_history.append(closed_position)

        # Update database
        try:
            self.db.update_option_position_status(position_id, 'CLOSED', {
                'contract_symbol': position['contract_symbol'],
                'underlying': position['underlying'],
                'strategy': position['strategy'],
                'entry_time': position['entry_time'].isoformat(),
                'exit_time': closed_position['exit_time'].isoformat(),
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'premium_risk': position['premium_risk'],
                'pnl': pnl,
                'returns': returns,
                'exit_reason': exit_reason
            })
        except Exception as e:
            logger.warning(f"Failed to update position status in DB: {e}")

        # Update peak / drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        else:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, drawdown)

        # Accumulate daily pnl
        self.daily_pnl += pnl

        logger.info(f"🔒 Closed option position: {position_id}")
        logger.info(f"   {position['signal_type']} {position['strategy']}")
        logger.info(f"   Entry: ₹{position['entry_price']:.2f} → Exit: ₹{exit_price:.2f}")
        logger.info(f"   P&L: ₹{pnl:+.2f} ({returns:+.2f}%)")
        logger.info(f"   Reason: {exit_reason}")
        logger.info(f"   Duration: {closed_position['duration']:.1f} minutes")

        return closed_position

    def reprice_open_positions(self, underlying_price: float):
        """Update LTP/Greeks & mark-to-market P&L for open positions."""
        for pos_id, pos in list(self.open_positions.items()):
            try:
                # Get current option price and Greeks
                new_premium, greeks = self._estimate_option_price_and_greeks(
                    pos['contract_symbol'], underlying_price, pos['strike'], 
                    pos['option_type'], pos['expiry']
                )
                
                # Update position with current market data
                pos['current_premium'] = new_premium
                pos['current_delta'] = greeks.get('delta', pos.get('delta', 0))
                pos['current_gamma'] = greeks.get('gamma', pos.get('gamma', 0))
                pos['current_theta'] = greeks.get('theta', pos.get('theta', 0))
                pos['current_vega'] = greeks.get('vega', pos.get('vega', 0))
                
                # Calculate unrealized P&L
                entry_value = pos['entry_price'] * pos['total_quantity']
                current_value = new_premium * pos['total_quantity']
                pos['unrealized_pnl'] = current_value - entry_value - pos.get('entry_commission', 0)
                
                # Calculate unrealized returns
                pos['unrealized_returns'] = (pos['unrealized_pnl'] / entry_value * 100) if entry_value > 0 else 0.0
                
            except Exception as e:
                logger.debug(f"Failed to reprice {pos_id}: {e}")
                continue

    def _estimate_option_price_and_greeks(self, contract_symbol: str, underlying_price: float, 
                                        strike: float, option_type: str, expiry: datetime) -> Tuple[float, Dict]:
        """Estimate option price and Greeks using simplified Black-Scholes approximation."""
        try:
            # Calculate time to expiry in years
            days_to_expiry = (expiry - self._now_tz()).days
            time_to_expiry = max(days_to_expiry / 365.0, 0.001)  # Minimum 1 day
            
            # Simplified IV estimation (in real implementation, use actual IV)
            implied_volatility = 0.25  # 25% default IV
            
            # Simplified option pricing (for now - can be enhanced with proper Black-Scholes)
            moneyness = strike / underlying_price
            
            if option_type == 'CE':  # Call option
                if moneyness < 0.98:  # ITM
                    intrinsic = underlying_price - strike
                    time_value = max(50, intrinsic * 0.1)  # Minimum time value
                    premium = intrinsic + time_value
                elif moneyness > 1.02:  # OTM
                    premium = max(10, 100 * (1 - moneyness))  # Small premium for OTM
                else:  # ATM
                    premium = underlying_price * 0.01  # 1% of underlying
            else:  # Put option
                if moneyness > 1.02:  # ITM
                    intrinsic = strike - underlying_price
                    time_value = max(50, intrinsic * 0.1)
                    premium = intrinsic + time_value
                elif moneyness < 0.98:  # OTM
                    premium = max(10, 100 * (moneyness - 1))
                else:  # ATM
                    premium = underlying_price * 0.01
            
            # Simplified Greeks
            if option_type == 'CE':
                delta = 0.5 if abs(moneyness - 1) < 0.02 else (0.8 if moneyness < 1 else 0.2)
            else:
                delta = -0.5 if abs(moneyness - 1) < 0.02 else (-0.8 if moneyness > 1 else -0.2)
            
            gamma = 0.01 if abs(moneyness - 1) < 0.02 else 0.005
            theta = -premium * 0.1  # Daily theta decay
            vega = premium * 0.5  # Vega exposure
            
            return premium, {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
            
        except Exception as e:
            logger.error(f"Error estimating option price: {e}")
            return 0.0, {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

    def _update_daily_loss_and_check(self):
        """Reset daily counters and check daily loss cap."""
        now = self._now_tz()
        if now.date() != self.last_day:
            self.last_day = now.date()
            self.nav_at_start_of_day = self.current_capital
            self.daily_pnl = 0.0
            self.daily_loss_limit_hit = False

        if self.daily_pnl < -abs(self.max_daily_loss_pct * self.nav_at_start_of_day):
            logger.warning(f"🚫 Daily loss limit breached: PnL={self.daily_pnl:.2f} <= limit {-self.max_daily_loss_pct*self.nav_at_start_of_day:.2f}")
            self.daily_loss_limit_hit = True

    # -------------------------
    # reporting / running
    # -------------------------
    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades else 0.0
        total_pnl = sum(p['pnl'] for p in self.closed_positions) if total_trades else 0.0
        avg_pnl = (total_pnl / total_trades) if total_trades else 0.0
        returns = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_drawdown': self.max_drawdown * 100,
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'returns': returns,
            'open_positions': len(self.open_positions),
            'session_duration_hours': (self._now_tz() - self.session_start).total_seconds() / 3600.0
        }

    def print_performance_summary(self):
        """Print current performance summary."""
        perf = self.get_performance_summary()

        print("\n" + "=" * 60)
        print("📊 OPTIONS TRADING PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"💰 Capital: ₹{perf['current_capital']:,.2f} / ₹{perf['initial_capital']:,.2f}")
        print(f"📈 Returns: {perf['returns']:+.2f}%")
        print(f"📊 Total Trades: {perf['total_trades']}")
        print(f"✅ Wins: {perf['winning_trades']} | ❌ Losses: {perf['losing_trades']}")
        print(f"🎯 Win Rate: {perf['win_rate']:.1f}%")
        print(f"💵 Total P&L: ₹{perf['total_pnl']:+,.2f}")
        print(f"📊 Avg P&L: ₹{perf['avg_pnl']:+,.2f}")
        print(f"📉 Max Drawdown: {perf['max_drawdown']:.2f}%")
        print(f"🔓 Open Positions: {perf['open_positions']}")
        print(f"⏱️ Session Duration: {perf['session_duration_hours']:.1f} hours")
        print("=" * 60)

    def is_market_open(self, current_time: datetime) -> bool:
        """Check if market is open (NSE trading hours)."""
        now = current_time.astimezone(self.tzinfo) if current_time.tzinfo else current_time.replace(tzinfo=self.tzinfo)
        if now.weekday() >= 5:
            return False
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_start <= now <= market_end

    def close_all_positions(self, current_price: float):
        """Close all open option positions."""
        logger.info("🔒 Closing all open option positions...")
        for position_id in list(self.open_positions.keys()):
            self.close_option_position(position_id, current_price, "Session End")

    def save_session_data(self):
        """Save session data to database."""
        try:
            for position in self.closed_positions:
                try:
                    self.db.log_live_signal(
                        timestamp=position['entry_time'],
                        symbol=position['underlying'],
                        strategy=position['strategy'],
                        signal=position['signal_type'],
                        price=position['entry_price'],
                        confidence=position['confidence'],
                        reasoning=position['reasoning'],
                        stop_loss=None,  # Options don't have traditional stop loss
                        target1=None,
                        target2=None,
                        target3=None,
                        position_multiplier=position['quantity']
                    )
                except Exception:
                    logger.exception("Failed to save closed option position to DB")

            logger.info(f"💾 Saved {len(self.closed_positions)} option trades to database")

        except Exception as e:
            logger.error(f"❌ Error saving session data: {e}")

    def run_options_trading(self, symbol: str, timeframe: str, check_interval: int = 60):
        """Run options trading session."""
        logger.info(f"🚀 Starting options trading session for {symbol} {timeframe}")
        logger.info(f"⏱️ Check interval: {check_interval} seconds")

        self.is_running = True

        try:
            while self.is_running:
                current_time = datetime.now(tz=self.tzinfo)
                if not self.is_market_open(current_time):
                    logger.info("🏛️ Market closed, waiting...")
                    time.sleep(300)
                    continue

                # Reset daily counters when date shifts
                self._update_daily_loss_and_check()
                if self.daily_loss_limit_hit:
                    logger.warning("🚫 Daily loss limit hit — sleeping until next day.")
                    time.sleep(300)
                    continue

                df = self.get_latest_data(symbol, timeframe)
                if df.empty:
                    logger.warning("⚠️ No data available, waiting...")
                    time.sleep(check_interval)
                    continue

                current_price = float(df['close'].iloc[-1])

                # Reprice all open positions first
                self.reprice_open_positions(current_price)

                # Check exits first
                closed_positions = self.check_option_position_exits(current_price, symbol)

                # Generate new option signals
                option_signals = self.generate_option_signals(df, symbol)

                # Process new signals
                for option_signal in option_signals:
                    if self.should_open_option_position(option_signal):
                        position_id = self.open_option_position(option_signal)
                        if position_id:
                            time.sleep(0.5)

                # Print periodic summary
                if len(self.closed_positions) and (len(self.closed_positions) % 5 == 0):
                    self.print_performance_summary()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Options trading session stopped by user")
        except Exception as e:
            logger.exception(f"❌ Error in options trading session: {e}")
        finally:
            self.is_running = False
            try:
                last_price = float(df['close'].iloc[-1]) if 'df' in locals() and not df.empty else 0.0
            except Exception:
                last_price = 0.0
            if last_price > 0:
                self.close_all_positions(last_price)
            self.print_performance_summary()
            self.save_session_data()


def main():
    parser = argparse.ArgumentParser(description='Options Trading Bot')
    parser.add_argument('--symbol', default='NSE:NIFTY50-INDEX', help='Underlying symbol')
    parser.add_argument('--timeframe', default='5min', help='Timeframe')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.02, help='Max risk per trade')
    parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    parser.add_argument('--confidence', type=float, default=40.0, help='Min confidence to open trades')
    parser.add_argument('--exposure', type=float, default=0.6, help='Max portfolio exposure (0-1)')
    parser.add_argument('--daily_loss', type=float, default=0.03, help='Max daily loss percent (0-1)')
    parser.add_argument('--max_hold', type=int, default=24*60, help='Max holding minutes before forced exit')
    parser.add_argument('--commission_bps', type=float, default=1.0, help='Commission in bps')
    parser.add_argument('--slippage_bps', type=float, default=5.0, help='Slippage in bps')
    parser.add_argument('--tz', type=str, default='Asia/Kolkata', help='Timezone name')
    parser.add_argument('--expiry', type=str, default='weekly', help='Expiry type (weekly/monthly)')
    parser.add_argument('--strike', type=str, default='atm', help='Strike selection (atm/otm/itm/delta)')
    parser.add_argument('--delta', type=float, default=0.30, help='Target delta for delta-based selection')

    args = parser.parse_args()

    # Convert strike selection string to enum
    strike_selection_map = {
        'atm': StrikeSelection.ATM,
        'otm': StrikeSelection.OTM,
        'itm': StrikeSelection.ITM,
        'delta': StrikeSelection.DELTA
    }
    strike_selection = strike_selection_map.get(args.strike.lower(), StrikeSelection.ATM)

    bot = OptionsTradingBot(
        initial_capital=args.capital,
        max_risk_per_trade=args.risk,
        confidence_cutoff=args.confidence,
        exposure_limit=args.exposure,
        max_daily_loss_pct=args.daily_loss,
        max_holding_minutes=args.max_hold,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        timezone=args.tz,
        expiry_type=args.expiry,
        strike_selection=strike_selection,
        delta_target=args.delta
    )

    try:
        bot.run_options_trading(
            symbol=args.symbol,
            timeframe=args.timeframe,
            check_interval=args.interval
        )
    except KeyboardInterrupt:
        logger.info("🛑 Options trading stopped by user")
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    main() 