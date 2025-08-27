#!/usr/bin/env python3
"""
Live Paper Trading Bot
Simulates real trading conditions with live data feeds and realistic execution
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
import sqlite3
from pathlib import Path

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
    def __init__(self, initial_capital: float = 100000.0, max_risk_per_trade: float = 0.02):
        """Initialize paper trading bot."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.open_positions = {}
        self.closed_positions = []
        self.trade_history = []
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        
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
        self.session_start = datetime.now()
        self.is_running = False
        
        logger.info(f"🚀 Paper Trading Bot initialized with ₹{initial_capital:,.2f} capital")
        logger.info(f"📊 Max risk per trade: {max_risk_per_trade*100:.1f}%")

    def calculate_position_size(self, entry_price: float, stop_loss: float, confidence: float) -> float:
        """Calculate position size based on risk management rules."""
        risk_amount = self.current_capital * self.max_risk_per_trade
        confidence_multiplier = min(confidence / 50.0, 1.5)
        adjusted_risk = risk_amount * confidence_multiplier
        
        price_risk = abs(entry_price - stop_loss)
        if price_risk <= 0:
            return 0
        
        position_size = adjusted_risk / price_risk
        lot_size = 50  # NIFTY lot size
        position_size = (position_size // lot_size) * lot_size
        
        if position_size < lot_size:
            position_size = lot_size if adjusted_risk >= price_risk * lot_size else 0
            
        return position_size

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
        """Generate trading signals using all strategies."""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'analyze_vectorized'):
                    signals_df = strategy.analyze_vectorized(df)
                    
                    if not signals_df.empty:
                        for idx, row in signals_df.iterrows():
                            signal = {
                                'timestamp': df.loc[idx, 'timestamp'],
                                'strategy': strategy_name,
                                'signal': row['signal'],
                                'price': float(row['price']),
                                'confidence': float(row.get('confidence_score', 0)),
                                'reasoning': str(row.get('reasoning', ''))[:200],
                                'stop_loss': float(row.get('stop_loss', 0)),
                                'target1': float(row.get('target1', 0)),
                                'target2': float(row.get('target2', 0)),
                                'target3': float(row.get('target3', 0)),
                                'position_multiplier': float(row.get('position_multiplier', 1.0))
                            }
                            signals.append(signal)
                            
            except Exception as e:
                logger.error(f"❌ Error in {strategy_name}: {e}")
                continue
                
        return signals

    def should_open_position(self, signal: Dict, current_price: float) -> bool:
        """Check if we should open a position based on signal and current conditions."""
        # Check if we already have a position in this direction
        for pos_id, position in self.open_positions.items():
            if position['symbol'] == signal.get('symbol') and position['direction'] == signal['signal']:
                return False
        
        # Check confidence threshold
        if signal['confidence'] < 40:
            return False
            
        # Check if price is still valid (within 1% of signal price)
        price_diff = abs(current_price - signal['price']) / signal['price']
        if price_diff > 0.01:
            logger.info(f"⚠️ Price moved too much: signal={signal['price']:.2f}, current={current_price:.2f}")
            return False
            
        return True

    def open_position(self, signal: Dict, current_price: float, symbol: str) -> Optional[str]:
        """Open a new trading position."""
        try:
            position_size = self.calculate_position_size(
                current_price, 
                signal['stop_loss'], 
                signal['confidence']
            )
            
            if position_size <= 0:
                logger.info(f"⚠️ Position size too small for {signal['strategy']}")
                return None
            
            required_capital = position_size * current_price
            
            if required_capital > self.current_capital * 0.8:
                logger.info(f"⚠️ Insufficient capital for {signal['strategy']}: required ₹{required_capital:,.2f}")
                return None
            
            position_id = f"{symbol}_{signal['strategy']}_{int(time.time())}"
            
            position = {
                'id': position_id,
                'symbol': symbol,
                'strategy': signal['strategy'],
                'direction': signal['signal'],
                'entry_price': current_price,
                'position_size': position_size,
                'stop_loss': signal['stop_loss'],
                'target1': signal['target1'],
                'target2': signal['target2'],
                'target3': signal['target3'],
                'entry_time': datetime.now(),
                'confidence': signal['confidence'],
                'reasoning': signal['reasoning'],
                'status': 'OPEN'
            }
            
            self.current_capital -= required_capital
            self.open_positions[position_id] = position
            
            logger.info(f"✅ Opened {signal['signal']} position: {position_id}")
            logger.info(f"   Size: {position_size} shares at ₹{current_price:.2f}")
            logger.info(f"   Stop Loss: ₹{signal['stop_loss']:.2f}, Target: ₹{signal['target1']:.2f}")
            logger.info(f"   Confidence: {signal['confidence']:.1f}%")
            
            return position_id
            
        except Exception as e:
            logger.error(f"❌ Error opening position: {e}")
            return None

    def check_position_exits(self, current_price: float, symbol: str) -> List[Dict]:
        """Check if any open positions should be closed."""
        closed_positions = []
        positions_to_close = []
        
        for position_id, position in self.open_positions.items():
            if position['symbol'] != symbol:
                continue
                
            exit_price = None
            exit_reason = None
            should_close = False
            
            if position['direction'] == 'BUY CALL':
                if current_price <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'Stop Loss'
                    should_close = True
                elif current_price >= position['target3']:
                    exit_price = position['target3']
                    exit_reason = 'Target 3'
                    should_close = True
                elif current_price >= position['target2']:
                    exit_price = position['target2']
                    exit_reason = 'Target 2'
                    should_close = True
                elif current_price >= position['target1']:
                    exit_price = position['target1']
                    exit_reason = 'Target 1'
                    should_close = True
                    
            elif position['direction'] == 'BUY PUT':
                if current_price >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'Stop Loss'
                    should_close = True
                elif current_price <= position['target3']:
                    exit_price = position['target3']
                    exit_reason = 'Target 3'
                    should_close = True
                elif current_price <= position['target2']:
                    exit_price = position['target2']
                    exit_reason = 'Target 2'
                    should_close = True
                elif current_price <= position['target1']:
                    exit_price = position['target1']
                    exit_reason = 'Target 1'
                    should_close = True
            
            if should_close:
                positions_to_close.append((position_id, exit_price, exit_reason))
        
        for position_id, exit_price, exit_reason in positions_to_close:
            closed_position = self.close_position(position_id, exit_price, exit_reason)
            if closed_position:
                closed_positions.append(closed_position)
        
        return closed_positions

    def close_position(self, position_id: str, exit_price: float, exit_reason: str) -> Optional[Dict]:
        """Close a trading position."""
        if position_id not in self.open_positions:
            return None
            
        position = self.open_positions[position_id]
        
        if position['direction'] == 'BUY CALL':
            pnl = (exit_price - position['entry_price']) * position['position_size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['position_size']
        
        position_value = position['position_size'] * exit_price
        self.current_capital += position_value
        
        returns = (pnl / (position['entry_price'] * position['position_size'])) * 100
        
        closed_position = {
            **position,
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'exit_reason': exit_reason,
            'pnl': pnl,
            'returns': returns,
            'duration': (datetime.now() - position['entry_time']).total_seconds() / 60,
            'status': 'CLOSED'
        }
        
        del self.open_positions[position_id]
        self.closed_positions.append(closed_position)
        self.trade_history.append(closed_position)
        self.daily_pnl += pnl
        
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        else:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        logger.info(f"🔒 Closed position: {position_id}")
        logger.info(f"   {position['direction']} {position['strategy']}")
        logger.info(f"   Entry: ₹{position['entry_price']:.2f} → Exit: ₹{exit_price:.2f}")
        logger.info(f"   P&L: ₹{pnl:+.2f} ({returns:+.2f}%)")
        logger.info(f"   Reason: {exit_reason}")
        logger.info(f"   Duration: {closed_position['duration']:.1f} minutes")
        
        return closed_position

    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_drawdown': 0,
                'current_capital': self.current_capital,
                'returns': 0
            }
        
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = sum(p['pnl'] for p in self.closed_positions)
        avg_pnl = total_pnl / total_trades
        
        returns = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_drawdown': self.max_drawdown * 100,
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'returns': returns,
            'open_positions': len(self.open_positions),
            'session_duration': (datetime.now() - self.session_start).total_seconds() / 3600
        }

    def print_performance_summary(self):
        """Print current performance summary."""
        perf = self.get_performance_summary()
        
        print("\n" + "="*60)
        print("📊 PAPER TRADING PERFORMANCE SUMMARY")
        print("="*60)
        print(f"💰 Capital: ₹{perf['current_capital']:,.2f} / ₹{perf['initial_capital']:,.2f}")
        print(f"📈 Returns: {perf['returns']:+.2f}%")
        print(f"📊 Total Trades: {perf['total_trades']}")
        print(f"✅ Wins: {perf['winning_trades']} | ❌ Losses: {perf['losing_trades']}")
        print(f"🎯 Win Rate: {perf['win_rate']:.1f}%")
        print(f"💵 Total P&L: ₹{perf['total_pnl']:+,.2f}")
        print(f"📊 Avg P&L: ₹{perf['avg_pnl']:+,.2f}")
        print(f"📉 Max Drawdown: {perf['max_drawdown']:.2f}%")
        print(f"🔓 Open Positions: {perf['open_positions']}")
        print(f"⏱️ Session Duration: {perf['session_duration']:.1f} hours")
        print("="*60)

    def is_market_open(self, current_time: datetime) -> bool:
        """Check if market is open (NSE trading hours)."""
        ist_time = current_time + timedelta(hours=5, minutes=30)
        
        if ist_time.weekday() >= 5:
            return False
        
        market_start = ist_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = ist_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= ist_time <= market_end

    def close_all_positions(self, current_price: float):
        """Close all open positions at current price."""
        logger.info("🔒 Closing all open positions...")
        
        for position_id in list(self.open_positions.keys()):
            self.close_position(position_id, current_price, "Session End")

    def save_session_data(self):
        """Save session data to database."""
        try:
            for position in self.closed_positions:
                self.db.log_live_signal(
                    timestamp=position['entry_time'],
                    symbol=position['symbol'],
                    strategy=position['strategy'],
                    signal=position['direction'],
                    price=position['entry_price'],
                    confidence=position['confidence'],
                    reasoning=position['reasoning'],
                    stop_loss=position['stop_loss'],
                    target1=position['target1'],
                    target2=position['target2'],
                    target3=position['target3'],
                    position_multiplier=position['position_multiplier']
                )
            
            logger.info(f"💾 Saved {len(self.closed_positions)} trades to database")
            
        except Exception as e:
            logger.error(f"❌ Error saving session data: {e}")

    def run_paper_trading(self, symbol: str, timeframe: str, check_interval: int = 60):
        """Run paper trading session."""
        logger.info(f"🚀 Starting paper trading session for {symbol} {timeframe}")
        logger.info(f"⏱️ Check interval: {check_interval} seconds")
        
        self.is_running = True
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                if not self.is_market_open(current_time):
                    logger.info("🏛️ Market closed, waiting...")
                    time.sleep(300)
                    continue
                
                df = self.get_latest_data(symbol, timeframe)
                if df.empty:
                    logger.warning("⚠️ No data available, waiting...")
                    time.sleep(check_interval)
                    continue
                
                current_price = df['close'].iloc[-1]
                
                closed_positions = self.check_position_exits(current_price, symbol)
                signals = self.generate_signals(df, symbol)
                
                for signal in signals:
                    signal['symbol'] = symbol
                    
                    if self.should_open_position(signal, current_price):
                        position_id = self.open_position(signal, current_price, symbol)
                        if position_id:
                            time.sleep(1)
                
                if len(self.closed_positions) % 5 == 0 and self.closed_positions:
                    self.print_performance_summary()
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Paper trading session stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in paper trading session: {e}")
        finally:
            self.is_running = False
            self.close_all_positions(current_price)
            self.print_performance_summary()
            self.save_session_data()

def main():
    parser = argparse.ArgumentParser(description='Paper Trading Bot')
    parser.add_argument('--symbol', default='NSE:NIFTY50-INDEX', help='Trading symbol')
    parser.add_argument('--timeframe', default='5min', help='Timeframe')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.02, help='Max risk per trade')
    parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    
    args = parser.parse_args()
    
    bot = PaperTradingBot(
        initial_capital=args.capital,
        max_risk_per_trade=args.risk
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
        logger.error(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main() 