#!/usr/bin/env python3
"""
Unified Strategy Engine for consistent signal generation across backtest and live trading
"""

import logging
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Import strategies
from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
from src.strategies.supertrend_ema import SupertrendEma
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma

logger = logging.getLogger(__name__)

class UnifiedStrategyEngine:
    """Unified strategy engine for consistent signal generation"""
    
    def __init__(self, symbols: List[str], confidence_cutoff: float = 0.6):
        self.symbols = symbols
        self.confidence_cutoff = confidence_cutoff
        self.tz = ZoneInfo("Asia/Kolkata")
        
        # Initialize strategies
        self.strategies = {
            'ema_crossover_enhanced': EmaCrossoverEnhanced(),
            'supertrend_ema': SupertrendEma(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
        }
        
        logger.info(f"✅ Unified Strategy Engine initialized with {len(self.strategies)} strategies")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_prices: Dict[str, float]) -> List[Dict]:
        """
        Generate signals from all strategies for all symbols
        
        Args:
            data: Dictionary of {symbol: DataFrame} with OHLCV data
            current_prices: Dictionary of {symbol: current_price}
            
        Returns:
            List of signal dictionaries
        """
        all_signals = []
        
        for symbol in self.symbols:
            if symbol not in data or data[symbol] is None or data[symbol].empty:
                logger.debug(f"⚠️ No data available for {symbol}")
                continue
                
            if symbol not in current_prices:
                logger.debug(f"⚠️ No current price for {symbol}")
                continue
            
            symbol_data = data[symbol]
            current_price = current_prices[symbol]
            
            # Generate signals from each strategy
            for strategy_name, strategy in self.strategies.items():
                try:
                    # Use the analyze method instead of generate_signals
                    logger.debug(f"🔍 Analyzing {strategy_name} for {symbol} with {len(symbol_data)} candles")
                    analysis = strategy.analyze(symbol_data)
                    
                    if analysis and 'signals' in analysis and analysis['signals']:
                        logger.debug(f"📊 {strategy_name} generated {len(analysis['signals'])} signals for {symbol}")
                        # Add metadata to each signal
                        for signal in analysis['signals']:
                            signal.update({
                                'symbol': symbol,
                                'strategy': strategy_name,
                                'timestamp': self.now_kolkata(),
                                'current_price': current_price
                            })
                            
                            # Filter by confidence
                            if signal.get('confidence', 0) >= self.confidence_cutoff:
                                all_signals.append(signal)
                                logger.debug(f"✅ Signal passed confidence filter: {signal}")
                            else:
                                logger.debug(f"❌ Signal failed confidence filter: {signal.get('confidence', 0)} < {self.confidence_cutoff}")
                    else:
                        logger.debug(f"⚠️ {strategy_name} generated no signals for {symbol}")
                                
                except Exception as e:
                    logger.error(f"❌ Error generating signals for {strategy_name} on {symbol}: {e}")
                    continue
        
        # Sort by confidence and limit to top signals
        all_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Limit to top 5 signals per symbol to avoid overwhelming
        limited_signals = []
        symbol_counts = {}
        
        for signal in all_signals:
            symbol = signal['symbol']
            if symbol not in symbol_counts:
                symbol_counts[symbol] = 0
            
            if symbol_counts[symbol] < 5:
                limited_signals.append(signal)
                symbol_counts[symbol] += 1
        
        if limited_signals:
            logger.info(f"📊 Generated {len(limited_signals)} signals (limited to top 5 per symbol)")
        
        return limited_signals
    
    def get_strategy_performance(self, trades: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate performance metrics for each strategy
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of strategy performance metrics
        """
        strategy_stats = {}
        
        for trade in trades:
            strategy = trade.get('strategy', 'unknown')
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': 0.0,
                    'total_fees': 0.0,
                    'max_drawdown': 0.0
                }
            
            stats = strategy_stats[strategy]
            stats['total_trades'] += 1
            
            pnl = trade.get('pnl', 0)
            fees = trade.get('fees', 0)
            
            stats['total_pnl'] += pnl
            stats['total_fees'] += fees
            
            if pnl > 0:
                stats['winning_trades'] += 1
            elif pnl < 0:
                stats['losing_trades'] += 1
        
        # Calculate additional metrics
        for strategy, stats in strategy_stats.items():
            if stats['total_trades'] > 0:
                stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
                stats['avg_pnl'] = stats['total_pnl'] / stats['total_trades']
                stats['avg_fees'] = stats['total_fees'] / stats['total_trades']
                stats['net_pnl'] = stats['total_pnl'] - stats['total_fees']
            else:
                stats['win_rate'] = 0.0
                stats['avg_pnl'] = 0.0
                stats['avg_fees'] = 0.0
                stats['net_pnl'] = 0.0
        
        return strategy_stats
    
    def now_kolkata(self) -> datetime:
        """Get current time in Kolkata timezone"""
        return datetime.now(self.tz)
    
    def validate_signal(self, signal: Dict) -> bool:
        """
        Validate a signal for consistency
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if signal is valid, False otherwise
        """
        required_fields = ['symbol', 'strategy', 'signal', 'confidence', 'timestamp']
        
        for field in required_fields:
            if field not in signal:
                logger.warning(f"⚠️ Signal missing required field: {field}")
                return False
        
        if signal['confidence'] < 0 or signal['confidence'] > 1:
            logger.warning(f"⚠️ Invalid confidence value: {signal['confidence']}")
            return False
        
        if signal['signal'] not in ['BUY_CALL', 'BUY_PUT', 'SELL_CALL', 'SELL_PUT']:
            logger.warning(f"⚠️ Invalid signal type: {signal['signal']}")
            return False
        
        return True 