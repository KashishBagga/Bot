#!/usr/bin/env python3
"""
Live Paper Trading Bot
Simulates real trading conditions with live data feeds and realistic execution
(Updated: adds confidence cutoff, exposure cap, daily loss cap, max holding time,
 trailing stop, commission & slippage, and timezone-safe market hours.)
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
from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
from src.strategies.supertrend_ema import SupertrendEma
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from simple_backtest import OptimizedBacktester
from src.data.local_data_loader import LocalDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingBot:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_risk_per_trade: float = 0.02,
        confidence_cutoff: float = 40.0,
        exposure_limit: float = 0.6,            # max portfolio exposure as % of NAV (0-1)
        max_daily_loss_pct: float = 0.03,       # stop trading for day if loss > this % of equity
        max_holding_minutes: Optional[int] = 24 * 60,  # force exit after this many minutes
        trailing_stop_pct: Optional[float] = 0.01,     # trailing stop width as fraction of entry (1% default)
        commission_bps: float = 1.0,            # basis points (0.01% default)
        slippage_bps: float = 5.0,              # basis points (0.05% default)
        timezone: str = "Asia/Kolkata"
    ):
        """Initialize paper trading bot with improved risk & safety features."""
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

        # Initialize database, data loader, and backtester
        self.db = UnifiedDatabase()
        self.data_loader = LocalDataLoader()
        self.backtester = OptimizedBacktester()

        # Trading session state
        self.session_start = self._now_tz()
        self.is_running = False

        logger.info(f"🚀 Paper Trading Bot initialized with ₹{initial_capital:,.2f} capital")
        logger.info(f"📊 Max risk per trade: {max_risk_per_trade*100:.1f}% | Confidence cutoff: {self.confidence_cutoff}")
        logger.info(f"📛 Exposure limit: {self.exposure_limit*100:.1f}%, Daily loss cap: {self.max_daily_loss_pct*100:.1f}%")
        logger.info(f"🕰️ Timezone: {self.tz_name} | Trailing stop: {self.trailing_stop_pct}")

    # -------------------------
    # small helpers
    # -------------------------
    def _now_tz(self) -> datetime:
        """Return timezone-aware now in configured timezone."""
        now = datetime.now().astimezone(self.tzinfo) if hasattr(datetime.now(), "astimezone") else datetime.now()
        try:
            return now.astimezone(self.tzinfo)
        except Exception:
            # fallback naive
            return now

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage in bps to a price (positive for buy, negative for sell)."""
        factor = 1.0 + (self.slippage_bps / 10000.0) if is_buy else 1.0 - (self.slippage_bps / 10000.0)
        return float(price) * factor

    def _commission_amount(self, notional: float) -> float:
        """Return commission amount from notional based on bps."""
        return abs(notional) * (self.commission_bps / 10000.0)

    def _normalize_signal_prices(self, signal: Dict, current_price: float) -> Dict:
        """
        Signals may carry stop_loss/target as EITHER amounts (ATR multiple) or absolute prices.
        Heuristic:
          - if stop_loss <= 0.5 * current_price it's likely an absolute price?  (not reliable)
          - simpler/safer heuristic: if stop_loss < current_price * 0.15 treat as amount (distance).
        We'll treat small values as amounts and convert to absolute price.
        """
        s = signal.copy()
        entry = float(signal.get('price', current_price))

        def conv_amt_to_price(amount: float, side_str: str):
            if amount is None or amount == 0:
                return None
            # treat as amount (distance)
            if side_str.upper().startswith("BUY"):
                return entry - float(amount)  # stop price
            else:
                return entry + float(amount)

        # stop_loss
        sl = float(signal.get('stop_loss', 0) or 0)
        if sl > 0 and sl < (current_price * 0.15):
            # small => amount
            s['stop_loss'] = conv_amt_to_price(sl, signal['signal'])
        else:
            # treat as absolute price
            s['stop_loss'] = sl if sl > 0 else None

        # targets
        for t in ['target1', 'target2', 'target3']:
            val = float(signal.get(t, 0) or 0)
            if val > 0 and val < (current_price * 0.15):
                # amount -> convert
                if signal['signal'].upper().startswith("BUY"):
                    s[t] = entry + val
                else:
                    s[t] = entry - val
            else:
                s[t] = val if val > 0 else None

        return s

    def _current_total_exposure(self) -> float:
        """Return total notional exposure across open positions in same currency."""
        tot = 0.0
        for pos in self.open_positions.values():
            tot += pos['position_size'] * pos['entry_price']
        # estimate NAV as cash + invested
        nav = self.current_capital + tot
        if nav <= 0:
            return 1.0
        return tot / nav

    def calculate_position_size(self, entry_price: float, stop_loss: float, confidence: float) -> float:
        """Calculate position size based on risk management rules."""
        risk_amount = self.current_capital * self.max_risk_per_trade
        # confidence multiplier: normalized around 50, capped
        confidence_multiplier = min(max(confidence / 50.0, 0.5), 1.5)
        adjusted_risk = risk_amount * confidence_multiplier

        price_risk = abs(entry_price - stop_loss) if stop_loss is not None else 0.0
        if price_risk <= 0:
            return 0.0

        position_size = adjusted_risk / price_risk
        lot_size = 50  # default; consider symbol metadata in future
        # floor to lot size
        position_size = (position_size // lot_size) * lot_size

        if position_size < lot_size:
            position_size = lot_size if adjusted_risk >= price_risk * lot_size else 0.0

        return float(position_size)

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

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Generate trading signals using all strategies (vectorized)."""
        signals: List[Dict] = []

        for strategy_name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'analyze_vectorized'):
                    signals_df = strategy.analyze_vectorized(df)
                    if not signals_df.empty:
                        # iterate over signals but only the last few (live systems only need fresh signals)
                        # keep up to last 10 signals per strategy to avoid floods
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
                                'position_multiplier': float(row.get('position_multiplier', 1.0))
                            }
                            signals.append(signal)
            except Exception as e:
                logger.error(f"❌ Error in {strategy_name}: {e}")
                continue

        return signals

    # -------------------------
    # execution / risk checks
    # -------------------------
    def should_open_position(self, signal: Dict, current_price: float) -> bool:
        """Check if we should open a position based on signal and current conditions."""

        # respect daily-stop
        if self.daily_loss_limit_hit:
            logger.warning("🚫 Daily loss limit hit — no new positions today.")
            return False

        # unique symbol-direction constraint (avoid duplicates)
        for pos in self.open_positions.values():
            if pos['symbol'] == signal.get('symbol') and pos['direction'] == signal['signal']:
                return False

        # confidence threshold configurable
        if float(signal.get('confidence', 0.0)) < self.confidence_cutoff:
            return False

        # price movement check (within 1% default)
        price_diff = abs(current_price - float(signal['price'])) / float(signal['price'])
        if price_diff > 0.01:
            logger.info(f"⚠️ Price moved too much: signal={signal['price']:.2f}, current={current_price:.2f}")
            return False

        # exposure check
        est_exposure_after = self._current_total_exposure()
        # this is pre-check; after opening the new position exposure will increase — approximate with notional
        # We will perform a stronger check in open_position before committing
        if est_exposure_after >= self.exposure_limit:
            logger.warning(f"🚫 Exposure limit reached: {est_exposure_after:.2%} >= {self.exposure_limit:.2%}")
            return False

        return True

    def open_position(self, signal: Dict, current_price: float, symbol: str) -> Optional[str]:
        """Open a new trading position with commission & slippage and exposure checks."""
        try:
            # normalize stop/targets (handle amounts vs prices)
            sig = self._normalize_signal_prices(signal, current_price)

            # compute an entry price with slippage applied
            is_buy = sig['signal'].upper().startswith('BUY')
            exec_price = self._apply_slippage(current_price, is_buy=is_buy)

            # compute position size (uses normalized stop price)
            stop_price = sig.get('stop_loss') or None
            # if stop_price is None we cannot size reliably
            if stop_price is None:
                logger.info("⚠️ No valid stop price -> cannot size position, rejecting")
                return None

            position_size = self.calculate_position_size(exec_price, float(stop_price), float(sig.get('confidence', 0)))
            if position_size <= 0:
                logger.info(f"⚠️ Position size too small for {sig['strategy']}")
                return None

            # notional and exposure checks
            notional = position_size * exec_price
            est_total_notional = sum([p['position_size'] * p['entry_price'] for p in self.open_positions.values()]) + notional
            nav_est = self.current_capital + sum([p['position_size'] * p['entry_price'] for p in self.open_positions.values()])
            exposure_after = est_total_notional / nav_est if nav_est > 0 else 1.0
            if exposure_after > self.exposure_limit:
                logger.warning(f"🚫 Opening this position would exceed exposure limit ({exposure_after:.2%} > {self.exposure_limit:.2%})")
                return None

            # required capital check (cap at 80% of available)
            required_capital = notional
            if required_capital > self.current_capital * 0.8:
                logger.info(f"⚠️ Insufficient capital for {sig['strategy']}: required ₹{required_capital:,.2f}")
                return None

            # commission & fees subtracted immediately (simple model)
            commission = self._commission_amount(notional)
            self.current_capital -= commission  # pay commission now
            logger.debug(f"Commission paid on entry: ₹{commission:.2f}")

            # reserve capital for the position by reducing current_capital by required capital
            # note: we treat this as blocked capital; in a more realistic margin model you'd handle margin differently
            self.current_capital -= required_capital

            position_id = f"{symbol}_{sig['strategy']}_{int(time.time())}"
            position = {
                'id': position_id,
                'symbol': symbol,
                'strategy': sig['strategy'],
                'direction': sig['signal'],
                'entry_price': exec_price,
                'position_size': position_size,
                'stop_loss': float(stop_price),
                'target1': sig.get('target1'),
                'target2': sig.get('target2'),
                'target3': sig.get('target3'),
                'entry_time': self._now_tz(),
                'confidence': sig.get('confidence', 0.0),
                'reasoning': sig.get('reasoning', ''),
                'status': 'OPEN'
            }

            # store an internal field to help trailing logic
            position['_initial_stop'] = float(stop_price)
            position['_notional'] = notional

            self.open_positions[position_id] = position

            logger.info(f"✅ Opened {sig['signal']} position: {position_id}")
            logger.info(f"   Size: {position_size} units at ₹{exec_price:.2f} | Notional: ₹{notional:,.2f}")
            logger.info(f"   Stop Loss: ₹{position['stop_loss']:.2f}, Target1: {position.get('target1')}")
            logger.info(f"   Commission paid: ₹{commission:.2f}")

            return position_id

        except Exception as e:
            logger.error(f"❌ Error opening position: {e}")
            return None

    def _should_force_time_exit(self, entry_time: datetime) -> bool:
        if self.max_holding_minutes is None:
            return False
        delta_min = (self._now_tz() - entry_time).total_seconds() / 60.0
        return delta_min >= float(self.max_holding_minutes)

    def _apply_trailing(self, pos: Dict, current_price: float):
        """Update position stop_loss in-place using trailing_stop_pct if configured."""
        if self.trailing_stop_pct is None:
            return
        if pos['direction'].upper().startswith('BUY'):
            # move stop up if price has moved enough
            new_stop = current_price - (pos['entry_price'] * self.trailing_stop_pct)
            if new_stop > pos['stop_loss']:
                logger.debug(f"Trailing stop updated for {pos['id']}: {pos['stop_loss']:.2f} -> {new_stop:.2f}")
                pos['stop_loss'] = new_stop
        else:
            new_stop = current_price + (pos['entry_price'] * self.trailing_stop_pct)
            if new_stop < pos['stop_loss']:
                logger.debug(f"Trailing stop updated for {pos['id']}: {pos['stop_loss']:.2f} -> {new_stop:.2f}")
                pos['stop_loss'] = new_stop

    def check_position_exits(self, current_price: float, symbol: str) -> List[Dict]:
        """Check if any open positions should be closed. Returns list of closed positions."""
        closed_positions = []
        positions_to_close = []

        # Update trailing stops first
        for position in self.open_positions.values():
            if position['symbol'] != symbol:
                continue
            self._apply_trailing(position, current_price)

            # time-based exit
            if self._should_force_time_exit(position['entry_time']):
                positions_to_close.append((position['id'], current_price, 'Time Exit'))
                continue

            # check classical SL/TP
            if position['direction'] == 'BUY CALL':
                if current_price <= position['stop_loss']:
                    positions_to_close.append((position['id'], position['stop_loss'], 'Stop Loss'))
                elif position.get('target3') and current_price >= position['target3']:
                    positions_to_close.append((position['id'], position['target3'], 'Target 3'))
                elif position.get('target2') and current_price >= position['target2']:
                    positions_to_close.append((position['id'], position['target2'], 'Target 2'))
                elif position.get('target1') and current_price >= position['target1']:
                    positions_to_close.append((position['id'], position['target1'], 'Target 1'))

            elif position['direction'] == 'BUY PUT':
                if current_price >= position['stop_loss']:
                    positions_to_close.append((position['id'], position['stop_loss'], 'Stop Loss'))
                elif position.get('target3') and current_price <= position['target3']:
                    positions_to_close.append((position['id'], position['target3'], 'Target 3'))
                elif position.get('target2') and current_price <= position['target2']:
                    positions_to_close.append((position['id'], position['target2'], 'Target 2'))
                elif position.get('target1') and current_price <= position['target1']:
                    positions_to_close.append((position['id'], position['target1'], 'Target 1'))

        # process closures
        for position_id, exit_price, exit_reason in positions_to_close:
            closed_position = self.close_position(position_id, exit_price, exit_reason)
            if closed_position:
                closed_positions.append(closed_position)

        # daily loss check
        self._update_daily_loss_and_check()

        return closed_positions

    def close_position(self, position_id: str, exit_price: float, exit_reason: str) -> Optional[Dict]:
        """Close a trading position and apply commission & slippage at exit."""
        if position_id not in self.open_positions:
            return None

        position = self.open_positions[position_id]
        is_buy = position['direction'].upper().startswith('BUY')

        # apply slippage to exit
        exec_price = self._apply_slippage(exit_price, is_buy=not is_buy)  # closing a long is a sell (no slippage sign flip)
        notional = position['position_size'] * exec_price
        commission = self._commission_amount(notional)

        # compute pnl
        if position['direction'] == 'BUY CALL':
            pnl = (exec_price - position['entry_price']) * position['position_size'] - commission
        else:
            pnl = (position['entry_price'] - exec_price) * position['position_size'] - commission

        # free reserved capital: add position notional back to capital, then subtract commission (already done)
        self.current_capital += position['position_size'] * exec_price
        self.current_capital -= commission  # pay exit commission

        returns_pct = (pnl / (position['entry_price'] * position['position_size'])) * 100 if (position['entry_price'] * position['position_size']) else 0.0

        closed_position = {
            **position,
            'exit_price': float(exec_price),
            'exit_time': self._now_tz(),
            'exit_reason': exit_reason,
            'pnl': float(pnl),
            'returns': float(returns_pct),
            'duration': (self._now_tz() - position['entry_time']).total_seconds() / 60.0,
            'status': 'CLOSED'
        }

        # remove from open positions
        del self.open_positions[position_id]
        self.closed_positions.append(closed_position)
        self.trade_history.append(closed_position)

        # update peak / drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        else:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, drawdown)

        # accumulate daily pnl and check daily limit
        self.daily_pnl += closed_position['pnl']

        logger.info(f"🔒 Closed position: {position_id}")
        logger.info(f"   {position['direction']} {position['strategy']}")
        logger.info(f"   Entry: ₹{position['entry_price']:.2f} → Exit: ₹{exec_price:.2f}")
        logger.info(f"   P&L: ₹{pnl:+.2f} ({returns_pct:+.2f}%)")
        logger.info(f"   Reason: {exit_reason}")
        logger.info(f"   Duration: {closed_position['duration']:.1f} minutes")

        return closed_position

    def _update_daily_loss_and_check(self):
        """Reset daily counters at midnight IST and check daily loss cap."""
        now = self._now_tz()
        if now.date() != self.last_day:
            # reset at start of new day
            self.last_day = now.date()
            self.nav_at_start_of_day = self.current_capital
            self.daily_pnl = 0.0
            self.daily_loss_limit_hit = False

        # check daily loss cap
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
        print("📊 PAPER TRADING PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"💰 Capital: ₹{perf['current_capital']:,.2f} / ₹{perf['initial_capital']:,.2f}")
        print(f"📈 Returns: {perf['returns']:+.2f}%")
        print(f"📊 Total Trades: {perf['total_trades']}")
        print(f"✅ Wins: {perf['winning_trades']} | ❌ Losses: {perf['losing_trades']}")
        print(f"🎯 Win Rate: {perf['win_rate']:.1f}%")
        print(f"💵 Total P&L: ₹{perf['avg_pnl'] * perf['total_trades']:+,.2f}")
        print(f"📊 Avg P&L: ₹{perf['avg_pnl']:+,.2f}")
        print(f"📉 Max Drawdown: {perf['max_drawdown']:.2f}%")
        print(f"🔓 Open Positions: {perf['open_positions']}")
        print(f"⏱️ Session Duration: {perf['session_duration_hours']:.1f} hours")
        print("=" * 60)

    def is_market_open(self, current_time: datetime) -> bool:
        """Check if market is open (NSE trading hours) using timezone-aware time."""
        now = current_time.astimezone(self.tzinfo) if current_time.tzinfo else current_time.replace(tzinfo=self.tzinfo)
        if now.weekday() >= 5:
            return False
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_start <= now <= market_end

    def close_all_positions(self, current_price: float):
        """Close all open positions at current price."""
        logger.info("🔒 Closing all open positions...")
        
        for position_id in list(self.open_positions.keys()):
            self.close_position(position_id, current_price, "Session End")

    def save_session_data(self):
        """Save session data to database (both open & closed)."""
        try:
            # save closed trades
            for position in self.closed_positions:
                try:
                    self.db.log_live_signal(
                        timestamp=position['entry_time'],
                        symbol=position['symbol'],
                        strategy=position['strategy'],
                        signal=position['direction'],
                        price=position['entry_price'],
                        confidence=position['confidence'],
                        reasoning=position['reasoning'],
                        stop_loss=position['stop_loss'],
                        target1=position.get('target1'),
                        target2=position.get('target2'),
                        target3=position.get('target3'),
                        position_multiplier=position.get('position_multiplier', 1.0)
                    )
                except Exception:
                    logger.exception("Failed to save closed position to DB")

            # save open positions snapshot
            for pos in self.open_positions.values():
                try:
                    self.db.log_live_signal(
                        timestamp=pos['entry_time'],
                        symbol=pos['symbol'],
                        strategy=pos['strategy'],
                        signal=pos['direction'],
                        price=pos['entry_price'],
                        confidence=pos.get('confidence', 0),
                        reasoning=pos.get('reasoning', ''),
                        stop_loss=pos.get('stop_loss'),
                        target1=pos.get('target1'),
                        target2=pos.get('target2'),
                        target3=pos.get('target3'),
                        position_multiplier=pos.get('position_multiplier', 1.0)
                    )
                except Exception:
                    logger.exception("Failed to save open position to DB")

            logger.info(f"💾 Saved {len(self.closed_positions)} closed trades and {len(self.open_positions)} open positions to database")

        except Exception as e:
            logger.error(f"❌ Error saving session data: {e}")

    def run_paper_trading(self, symbol: str, timeframe: str, check_interval: int = 60):
        """Run paper trading session."""
        logger.info(f"🚀 Starting paper trading session for {symbol} {timeframe}")
        logger.info(f"⏱️ Check interval: {check_interval} seconds")

        self.is_running = True

        try:
            while self.is_running:
                current_time = datetime.now(tz=self.tzinfo)
                if not self.is_market_open(current_time):
                    logger.info("🏛️ Market closed, waiting...")
                    time.sleep(300)
                    continue

                # reset daily counters when date shifts
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

                # 1) Check exits first using latest price
                closed_positions = self.check_position_exits(current_price, symbol)

                # 2) Get signals and try to open
                signals = self.generate_signals(df, symbol)

                for signal in signals:
                    # attach symbol
                    signal['symbol'] = symbol
                    if self.should_open_position(signal, current_price):
                        position_id = self.open_position(signal, current_price, symbol)
                        if position_id:
                            # small throttle so DB / logs are in order
                            time.sleep(0.5)

                # occasional performance print
                if len(self.closed_positions) and (len(self.closed_positions) % 5 == 0):
                    self.print_performance_summary()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Paper trading session stopped by user")
        except Exception as e:
            logger.exception(f"❌ Error in paper trading session: {e}")
        finally:
            self.is_running = False
            # close remaining positions at last known price
            try:
                last_price = float(df['close'].iloc[-1]) if 'df' in locals() and not df.empty else 0.0
            except Exception:
                last_price = 0.0
            if last_price > 0:
                self.close_all_positions(last_price)
            self.print_performance_summary()
            self.save_session_data()

def main():
    parser = argparse.ArgumentParser(description='Paper Trading Bot')
    parser.add_argument('--symbol', default='NSE:NIFTY50-INDEX', help='Trading symbol')
    parser.add_argument('--timeframe', default='5min', help='Timeframe')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.02, help='Max risk per trade')
    parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    parser.add_argument('--confidence', type=float, default=40.0, help='Min confidence to open trades')
    parser.add_argument('--exposure', type=float, default=0.6, help='Max portfolio exposure (0-1)')
    parser.add_argument('--daily_loss', type=float, default=0.03, help='Max daily loss percent (0-1)')
    parser.add_argument('--max_hold', type=int, default=24*60, help='Max holding minutes before forced exit')
    parser.add_argument('--trailing', type=float, default=0.01, help='Trailing stop pct (e.g. 0.01 for 1%)')
    parser.add_argument('--commission_bps', type=float, default=1.0, help='Commission in bps')
    parser.add_argument('--slippage_bps', type=float, default=5.0, help='Slippage in bps')
    parser.add_argument('--tz', type=str, default='Asia/Kolkata', help='Timezone name')

    args = parser.parse_args()

    bot = PaperTradingBot(
        initial_capital=args.capital,
        max_risk_per_trade=args.risk,
        confidence_cutoff=args.confidence,
        exposure_limit=args.exposure,
        max_daily_loss_pct=args.daily_loss,
        max_holding_minutes=args.max_hold,
        trailing_stop_pct=args.trailing,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        timezone=args.tz
    )

    try:
        bot.run_paper_trading(
            symbol=args.symbol,
            timeframe=args.timeframe,
            check_interval=args.interval
        )
    except KeyboardInterrupt:
        logger.info("🛑 Paper trading stopped by user")
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main() 