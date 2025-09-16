#!/usr/bin/env python3
"""
Fixed Strategy Engine with Proper Signal Generation and Execution
===============================================================
Fixes the signal generation and execution flow with proper data handling
"""

import logging
import pandas as pd
import numpy as np
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
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class MarketCondition(Enum):
    """Market condition based on volatility and trend"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"

class FixedStrategyEngine:
    """Fixed strategy engine with proper signal generation and execution"""
    
    def __init__(self, symbols: List[str], confidence_cutoff: float = 25.0):
        self.symbols = symbols
        self.confidence_cutoff = confidence_cutoff
        self.tz = ZoneInfo("Asia/Kolkata")
        
        # Initialize strategies
        self.strategies = {
            'ema_crossover_enhanced': EmaCrossoverEnhanced(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma(),
            'simple_ema': SimpleEmaStrategy(),
            'supertrend_ema': SupertrendEma()
        }
        
        # Position sizing parameters
        self.base_position_size = 1.0
        self.max_position_size = 3.0
        
        # Risk management parameters
        self.base_stop_loss = 0.02  # 2% base stop loss
        self.max_stop_loss = 0.05   # 5% maximum stop loss
        
        logger.info(f"🚀 Fixed Strategy Engine initialized for {len(symbols)} symbols")
    
    def generate_signals(self, historical_data: Dict[str, pd.DataFrame], 
                        current_prices: Dict[str, float]) -> List[Dict]:
        """Generate signals for all symbols with proper data handling."""
        all_signals = []
        
        for symbol in self.symbols:
            if symbol not in historical_data or symbol not in current_prices:
                logger.warning(f"⚠️ Missing data for {symbol}")
                continue
            
            data = historical_data[symbol]
            current_price = current_prices[symbol]
            
            if data is None or len(data) < 50:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(data) if data is not None else 0} candles")
                continue
            
            # Generate signals for this symbol
            symbol_signals = self._generate_symbol_signals(symbol, data, current_price)
            all_signals.extend(symbol_signals)
        
        # Sort by confidence
        all_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"📊 Generated {len(all_signals)} total signals")
        return all_signals
    
    def _generate_symbol_signals(self, symbol: str, data: pd.DataFrame, 
                               current_price: float) -> List[Dict]:
        """Generate signals for a single symbol."""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                # Generate signal using strategy
                signal_result = strategy.analyze(data)
                
                if not signal_result or signal_result.get('signal') in ['NO TRADE', 'ERROR']:
                    continue
                
                signal_type = signal_result.get('signal')
                confidence = signal_result.get('confidence', signal_result.get('confidence_score', 0))
                
                # Check confidence threshold
                if confidence < self.confidence_cutoff:
                    logger.debug(f"⚠️ {strategy_name} signal for {symbol} rejected: confidence {confidence} < {self.confidence_cutoff}")
                    continue
                
                # Calculate position size and risk parameters
                position_size = self._calculate_position_size(confidence)
                stop_loss_price = self._calculate_stop_loss(current_price, signal_type)
                take_profit_price = self._calculate_take_profit(current_price, signal_type)
                
                # Create signal
                signal = {
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'signal': signal_type,
                    'confidence': confidence,
                    'price': current_price,
                    'timestamp': datetime.now(self.tz).isoformat(),
                    'timeframe': '5m',  # Default timeframe
                    'strength': 'moderate',  # Default strength
                    'confirmed': True,
                    'position_size': position_size,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'indicator_values': {
                        'ema_12': signal_result.get('ema_12'),
                        'ema_26': signal_result.get('ema_26'),
                        'rsi': signal_result.get('rsi'),
                        'macd': signal_result.get('macd'),
                        'supertrend': signal_result.get('supertrend')
                    },
                    'market_condition': 'trending',  # Default
                    'volatility': 0.2  # Default
                }
                
                signals.append(signal)
                logger.info(f"✅ {strategy_name} signal for {symbol}: {signal_type} (confidence: {confidence})")
                
            except Exception as e:
                logger.error(f"❌ Error generating {strategy_name} signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence."""
        # Base position size
        base_size = 5000.0  # Base position size
        
        # Adjust for confidence
        confidence_multiplier = confidence / 100.0
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier
        
        # Ensure within bounds
        position_size = max(1000, min(position_size, 10000))
        
        return round(position_size, 2)
    
    def _calculate_stop_loss(self, entry_price: float, signal_type: str) -> float:
        """Calculate stop loss price."""
        if signal_type == 'BUY CALL':
            return entry_price * (1 - self.base_stop_loss)
        else:  # BUY PUT
            return entry_price * (1 + self.base_stop_loss)
    
    def _calculate_take_profit(self, entry_price: float, signal_type: str) -> float:
        """Calculate take profit price."""
        if signal_type == 'BUY CALL':
            return entry_price * (1 + 0.05)  # 5% take profit
        else:  # BUY PUT
            return entry_price * (1 - 0.05)  # 5% take profit
    
    def get_strategy_performance(self, market: str) -> List[Tuple]:
        """Get strategy performance statistics."""
        # This would integrate with the enhanced database
        # For now, return placeholder data
        return [
            ('simple_ema', 100, 5000.0, 50.0, 60, -100.0, 200.0, 30, 20),
            ('ema_crossover_enhanced', 80, 3000.0, 37.5, 45, -150.0, 180.0, 25, 15),
            ('supertrend_macd_rsi_ema', 60, 2000.0, 33.3, 40, -200.0, 150.0, 20, 10)
        ]
