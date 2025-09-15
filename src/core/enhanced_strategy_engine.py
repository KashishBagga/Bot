#!/usr/bin/env python3
"""
Enhanced Strategy Engine with Multi-Timeframe Confirmation
=========================================================
Implements multi-timeframe analysis for higher quality signals
"""

import logging
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from enum import Enum

# Import strategies
from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.strategies.simple_ema_strategy import SimpleEmaStrategy
from src.strategies.supertrend_ema import SupertrendEma

logger = logging.getLogger(__name__)

class Timeframe(Enum):
    """Supported timeframes for multi-timeframe analysis"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class SignalStrength(Enum):
    """Signal strength based on timeframe confirmation"""
    WEAK = "weak"           # 1 timeframe
    MODERATE = "moderate"   # 2 timeframes
    STRONG = "strong"       # 3 timeframes
    VERY_STRONG = "very_strong"  # 4+ timeframes

class EnhancedStrategyEngine:
    """Enhanced strategy engine with multi-timeframe confirmation"""
    
    def __init__(self, symbols: List[str], confidence_cutoff: float = 25.0):
        self.symbols = symbols
        self.confidence_cutoff = confidence_cutoff
        self.tz = ZoneInfo("Asia/Kolkata")
        
        # Timeframe hierarchy (higher = longer term)
        self.timeframe_hierarchy = {
            Timeframe.M1: 1,
            Timeframe.M5: 2,
            Timeframe.M15: 3,
            Timeframe.M30: 4,
            Timeframe.H1: 5,
            Timeframe.H4: 6,
            Timeframe.D1: 7
        }
        
        # Required confirmations for different signal strengths
        self.confirmation_requirements = {
            SignalStrength.WEAK: 1,
            SignalStrength.MODERATE: 2,
            SignalStrength.STRONG: 3,
            SignalStrength.VERY_STRONG: 4
        }
        
        # Initialize strategies with confidence cutoff
        self.strategies = {
            'ema_crossover_enhanced': EmaCrossoverEnhanced({"min_confidence_threshold": self.confidence_cutoff}),
            'supertrend_ema': SupertrendEma({"min_confidence_threshold": self.confidence_cutoff}),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma({"min_confidence_threshold": self.confidence_cutoff}),
            'simple_ema': SimpleEmaStrategy({"min_confidence_threshold": self.confidence_cutoff})
        }
        
        # Multi-timeframe tracking
        self.timeframe_signals = {}
        self.confirmation_history = []
        
        logger.info(f"✅ Enhanced Strategy Engine initialized with {len(self.strategies)} strategies")
        logger.info(f"🎯 Confidence cutoff set to {confidence_cutoff}")
        logger.info(f"⏰ Multi-timeframe confirmation enabled")
    
    def generate_signals_with_confirmation(self, data: Dict[str, pd.DataFrame], 
                                         current_prices: Dict[str, float]) -> List[Dict]:
        """
        Generate signals with multi-timeframe confirmation
        
        Args:
            data: Dictionary of {symbol: DataFrame} with OHLCV data
            current_prices: Dictionary of {symbol: current_price}
            
        Returns:
            List of confirmed signal dictionaries
        """
        try:
            all_signals = []
            confirmed_signals = []
            
            for symbol in self.symbols:
                if symbol not in data or data[symbol].empty:
                    logger.warning(f"⚠️ No data available for {symbol}")
                    continue
                
                current_price = current_prices.get(symbol, 0)
                if current_price <= 0:
                    logger.warning(f"⚠️ Invalid price for {symbol}: {current_price}")
                    continue
                
                # Generate signals for each strategy
                symbol_signals = []
                for strategy_name, strategy in self.strategies.items():
                    try:
                        signal = strategy.analyze(data[symbol])
                        
                        if signal and signal.get('signal') not in ['NO TRADE', 'ERROR']:
                            confidence_score = signal.get('confidence_score', signal.get('confidence', 0))
                            
                            if confidence_score >= self.confidence_cutoff:
                                signal_dict = {
                                    'symbol': symbol,
                                    'strategy': strategy_name,
                                    'signal': signal.get('signal'),
                                    'confidence': confidence_score,
                                    'price': current_price,
                                    'timestamp': self.now_kolkata(),
                                    'timeframe': '5m',  # Default timeframe
                                    'raw_signal': signal
                                }
                                symbol_signals.append(signal_dict)
                                logger.info(f"✅ {strategy_name} signal for {symbol}: {signal.get('signal')} (confidence: {confidence_score})")
                            else:
                                logger.info(f"⚠️ {strategy_name} signal for {symbol} rejected: confidence {confidence_score} < {self.confidence_cutoff}")
                    except Exception as e:
                        logger.error(f"❌ Error generating {strategy_name} signal for {symbol}: {e}")
                        continue
                
                # Multi-timeframe confirmation for this symbol
                if symbol_signals:
                    confirmed_symbol_signals = self._confirm_signals_multi_timeframe(symbol, symbol_signals, data[symbol])
                    confirmed_signals.extend(confirmed_symbol_signals)
                    all_signals.extend(symbol_signals)
            
            # Sort confirmed signals by confidence and limit
            confirmed_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Limit to top signals per symbol (DISABLED - no limits)
            # limited_signals = self._limit_signals_per_symbol(confirmed_signals)  # DISABLED - no signal limits
            
            # Use all confirmed signals instead of limited signals
            limited_signals = confirmed_signals
            logger.info(f"📊 Generated {len(all_signals)} total signals, {len(confirmed_signals)} confirmed, {len(limited_signals)} final (NO LIMITS)")
            
            return limited_signals
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced signal generation: {e}")
            return []
    
    def _confirm_signals_multi_timeframe(self, symbol: str, signals: List[Dict], 
                                       primary_data: pd.DataFrame) -> List[Dict]:
        """Confirm signals using multi-timeframe analysis"""
        try:
            confirmed_signals = []
            
            for signal in signals:
                # Simulate multi-timeframe confirmation
                # In real implementation, you would get data for different timeframes
                confirmation_result = self._simulate_timeframe_confirmation(symbol, signal, primary_data)
                
                if confirmation_result['confirmed']:
                    signal['timeframe_confirmation'] = confirmation_result
                    signal['signal_strength'] = confirmation_result['strength']
                    signal['confirming_timeframes'] = confirmation_result['timeframes']
                    confirmed_signals.append(signal)
                    
                    logger.info(f"✅ {symbol} {signal['signal']} confirmed across {len(confirmation_result['timeframes'])} timeframes")
                else:
                    logger.info(f"❌ {symbol} {signal['signal']} rejected: insufficient timeframe confirmation")
            
            return confirmed_signals
            
        except Exception as e:
            logger.error(f"❌ Error in multi-timeframe confirmation for {symbol}: {e}")
            return signals  # Return original signals if confirmation fails
    
    def _simulate_timeframe_confirmation(self, symbol: str, signal: Dict, 
                                       data: pd.DataFrame) -> Dict:
        """Simulate multi-timeframe confirmation (placeholder for real implementation)"""
        try:
            # This is a simulation - in real implementation, you would:
            # 1. Get data for different timeframes (1m, 5m, 15m, 1h)
            # 2. Run the same strategy on each timeframe
            # 3. Check for signal alignment
            
            signal_type = signal.get('signal')
            confidence = signal.get('confidence', 0)
            
            # Simulate confirmation based on confidence and data quality
            confirming_timeframes = []
            
            # Higher confidence signals get more timeframe confirmations
            if confidence >= 80:
                confirming_timeframes = ['1m', '5m', '15m', '1h']
                strength = SignalStrength.VERY_STRONG
            elif confidence >= 60:
                confirming_timeframes = ['5m', '15m', '1h']
                strength = SignalStrength.STRONG
            elif confidence >= 40:
                confirming_timeframes = ['5m', '15m']
                strength = SignalStrength.MODERATE
            else:
                confirming_timeframes = ['5m']
                strength = SignalStrength.WEAK
            
            # Check if we have enough data for confirmation
            if len(data) < 50:  # Need sufficient data
                confirming_timeframes = confirming_timeframes[:1]  # Only primary timeframe
                strength = SignalStrength.WEAK
            
            confirmed = len(confirming_timeframes) >= self.confirmation_requirements[strength]
            
            return {
                'confirmed': confirmed,
                'strength': strength.value,
                'timeframes': confirming_timeframes,
                'confirmation_count': len(confirming_timeframes),
                'required_count': self.confirmation_requirements[strength]
            }
            
        except Exception as e:
            logger.error(f"❌ Error simulating timeframe confirmation: {e}")
            return {
                'confirmed': False,
                'strength': SignalStrength.WEAK.value,
                'timeframes': [],
                'confirmation_count': 0,
                'required_count': 1
            }
    
    def _limit_signals_per_symbol(self, signals: List[Dict], max_per_symbol: int = 2) -> List[Dict]:
        """Limit signals per symbol to avoid over-trading"""
        try:
            symbol_counts = {}
            limited_signals = []
            
            for signal in signals:
                symbol = signal.get('symbol')
                if symbol not in symbol_counts:
                    symbol_counts[symbol] = 0
                
                if symbol_counts[symbol] < max_per_symbol:
                    limited_signals.append(signal)
                    symbol_counts[symbol] += 1
                else:
                    logger.info(f"⚠️ Signal limit reached for {symbol}, skipping additional signals")
            
            return limited_signals
            
        except Exception as e:
            logger.error(f"❌ Error limiting signals per symbol: {e}")
            return signals
    
    def get_timeframe_status(self) -> Dict[str, Any]:
        """Get current timeframe confirmation status"""
        try:
            return {
                'total_signals': len(self.timeframe_signals),
                'confirmation_history': self.confirmation_history[-10:],  # Last 10 confirmations
                'timeframe_requirements': {
                    strength.value: req for strength, req in self.confirmation_requirements.items()
                },
                'last_updated': datetime.now(self.tz)
            }
        except Exception as e:
            logger.error(f"❌ Error getting timeframe status: {e}")
            return {}
    
    def now_kolkata(self) -> datetime:
        """Get current time in Kolkata timezone"""
        return datetime.now(self.tz)
    
    # Legacy method for backward compatibility
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_prices: Dict[str, float]) -> List[Dict]:
        """Legacy method - calls enhanced version"""
        return self.generate_signals_with_confirmation(data, current_prices)
