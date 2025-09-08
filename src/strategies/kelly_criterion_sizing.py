"""
Kelly Criterion Position Sizing Strategy
Optimal position sizing based on win rate and average win/loss
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class KellyCriterionSizing:
    def __init__(self):
        self.name = "kelly_criterion_sizing"
        self.min_confidence_threshold = 25
        
        # Kelly Criterion parameters
        self.max_kelly_fraction = 0.25  # Maximum 25% of capital per trade
        self.min_kelly_fraction = 0.01  # Minimum 1% of capital per trade
        self.kelly_multiplier = 0.5  # Conservative Kelly (50% of optimal)
        
        # Performance tracking
        self.trade_history = []
        self.win_rate = 0.5  # Default 50% win rate
        self.avg_win = 0.02  # Default 2% average win
        self.avg_loss = 0.015  # Default 1.5% average loss
        
    def update_performance_metrics(self, trade_result: Dict[str, Any]):
        """Update performance metrics based on trade results"""
        try:
            self.trade_history.append(trade_result)
            
            # Keep only last 100 trades for calculation
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
                
            # Calculate win rate
            wins = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
            total_trades = len(self.trade_history)
            
            if total_trades > 0:
                self.win_rate = wins / total_trades
                
            # Calculate average win and loss
            winning_trades = [trade for trade in self.trade_history if trade.get('pnl', 0) > 0]
            losing_trades = [trade for trade in self.trade_history if trade.get('pnl', 0) < 0]
            
            if winning_trades:
                self.avg_win = np.mean([trade['pnl'] for trade in winning_trades])
            if losing_trades:
                self.avg_loss = abs(np.mean([trade['pnl'] for trade in losing_trades]))
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
            
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion fraction"""
        try:
            if avg_loss == 0:
                return self.min_kelly_fraction
                
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply conservative multiplier
            kelly_fraction *= self.kelly_multiplier
            
            # Clamp to reasonable bounds
            kelly_fraction = max(self.min_kelly_fraction, min(kelly_fraction, self.max_kelly_fraction))
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return self.min_kelly_fraction
            
    def calculate_position_size(self, signal: Dict[str, Any], available_capital: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            symbol = signal.get('symbol')
            confidence = signal.get('confidence', 0)
            entry_price = signal.get('price', 0)
            
            if entry_price <= 0 or available_capital <= 0:
                return 0.0
                
            # Calculate Kelly fraction
            kelly_fraction = self.calculate_kelly_fraction(
                self.win_rate, self.avg_win, self.avg_loss
            )
            
            # Adjust Kelly fraction based on confidence
            confidence_multiplier = max(0.5, min(1.5, confidence / 100.0))
            adjusted_kelly = kelly_fraction * confidence_multiplier
            
            # Calculate position size
            position_value = available_capital * adjusted_kelly
            position_size = position_value / entry_price
            
            # Ensure we don't exceed available capital
            max_affordable = available_capital * 0.9 / entry_price
            position_size = min(position_size, max_affordable)
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
            
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Generate signal with Kelly Criterion sizing information"""
        try:
            if len(df) < 20:
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': 0,
                    'strategy': self.name,
                    'reason': 'insufficient_data'
                }
            
            # Simple trend following signal
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Calculate trend strength
            trend_strength = 0
            if latest['ema_20'] > latest['ema_50']:
                trend_strength = 1
            elif latest['ema_20'] < latest['ema_50']:
                trend_strength = -1
                
            # Calculate momentum
            momentum = (latest['close'] - prev['close']) / prev['close']
            
            # Generate signal
            if trend_strength > 0 and momentum > 0:
                confidence = min(100, 30 + abs(momentum) * 1000)
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'confidence': confidence,
                    'strategy': self.name,
                    'reason': 'kelly_bullish_trend'
                }
            elif trend_strength < 0 and momentum < 0:
                confidence = min(100, 30 + abs(momentum) * 1000)
                return {
                    'symbol': symbol,
                    'action': 'SELL',
                    'confidence': confidence,
                    'strategy': self.name,
                    'reason': 'kelly_bearish_trend'
                }
            else:
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': 0,
                    'strategy': self.name,
                    'reason': 'no_clear_trend'
                }
                
        except Exception as e:
            logger.error(f"Error generating Kelly signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0,
                'strategy': self.name,
                'reason': f'error: {str(e)}'
            }
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for Kelly Criterion"""
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'win_rate': 0.5,
                    'avg_win': 0.02,
                    'avg_loss': 0.015,
                    'kelly_fraction': 0.01
                }
                
            total_trades = len(self.trade_history)
            wins = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
            win_rate = wins / total_trades if total_trades > 0 else 0.5
            
            winning_trades = [trade for trade in self.trade_history if trade.get('pnl', 0) > 0]
            losing_trades = [trade for trade in self.trade_history if trade.get('pnl', 0) < 0]
            
            avg_win = np.mean([trade['pnl'] for trade in winning_trades]) if winning_trades else 0.02
            avg_loss = abs(np.mean([trade['pnl'] for trade in losing_trades])) if losing_trades else 0.015
            
            kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'kelly_fraction': kelly_fraction
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0.5,
                'avg_win': 0.02,
                'avg_loss': 0.015,
                'kelly_fraction': 0.01
            }
