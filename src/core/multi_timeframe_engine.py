#!/usr/bin/env python3
"""
Multi-Timeframe Confirmation Engine
==================================
Implements multi-timeframe analysis for higher quality signals
"""

import logging
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from enum import Enum

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

class MultiTimeframeEngine:
    """Multi-timeframe confirmation engine for trading signals"""
    
    def __init__(self, primary_timeframe: Timeframe = Timeframe.M5):
        self.primary_timeframe = primary_timeframe
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
        
        logger.info(f"✅ Multi-Timeframe Engine initialized with primary timeframe: {primary_timeframe.value}")
    
    def analyze_multi_timeframe(self, symbol: str, data_provider, strategies: Dict) -> Dict[str, Any]:
        """
        Analyze signals across multiple timeframes for a symbol
        
        Args:
            symbol: Trading symbol
            data_provider: Data provider instance
            strategies: Dictionary of strategy instances
            
        Returns:
            Dict containing multi-timeframe analysis results
        """
        try:
            timeframe_signals = {}
            confirmed_signals = []
            
            # Analyze each timeframe
            for timeframe in [Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1]:
                try:
                    # Get data for this timeframe
                    data = self._get_timeframe_data(symbol, timeframe, data_provider)
                    if data is None or data.empty:
                        logger.warning(f"⚠️ No data available for {symbol} on {timeframe.value}")
                        continue
                    
                    # Generate signals for this timeframe
                    timeframe_signal = self._generate_timeframe_signal(
                        symbol, timeframe, data, strategies
                    )
                    
                    if timeframe_signal:
                        timeframe_signals[timeframe] = timeframe_signal
                        logger.info(f"✅ {timeframe.value} signal for {symbol}: {timeframe_signal.get('signal')} (confidence: {timeframe_signal.get('confidence', 0)})")
                    
                except Exception as e:
                    logger.error(f"❌ Error analyzing {timeframe.value} for {symbol}: {e}")
                    continue
            
            # Analyze signal confirmation
            if timeframe_signals:
                confirmation_analysis = self._analyze_signal_confirmation(timeframe_signals)
                confirmed_signals = confirmation_analysis.get('confirmed_signals', [])
                
                # Calculate overall signal strength
                signal_strength = self._calculate_signal_strength(confirmation_analysis)
                
                return {
                    'symbol': symbol,
                    'timeframe_signals': timeframe_signals,
                    'confirmed_signals': confirmed_signals,
                    'signal_strength': signal_strength,
                    'confirmation_analysis': confirmation_analysis,
                    'timestamp': datetime.now(self.tz)
                }
            else:
                return {
                    'symbol': symbol,
                    'timeframe_signals': {},
                    'confirmed_signals': [],
                    'signal_strength': SignalStrength.WEAK,
                    'confirmation_analysis': {'total_signals': 0, 'confirmed_count': 0},
                    'timestamp': datetime.now(self.tz)
                }
                
        except Exception as e:
            logger.error(f"❌ Error in multi-timeframe analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timeframe_signals': {},
                'confirmed_signals': [],
                'signal_strength': SignalStrength.WEAK,
                'confirmation_analysis': {'total_signals': 0, 'confirmed_count': 0, 'error': str(e)},
                'timestamp': datetime.now(self.tz)
            }
    
    def _get_timeframe_data(self, symbol: str, timeframe: Timeframe, data_provider) -> Optional[pd.DataFrame]:
        """Get historical data for a specific timeframe"""
        try:
            # This would integrate with your data provider
            # For now, return None as placeholder
            # In real implementation, you'd call data_provider.get_historical_data(symbol, timeframe.value)
            return None
        except Exception as e:
            logger.error(f"❌ Error getting {timeframe.value} data for {symbol}: {e}")
            return None
    
    def _generate_timeframe_signal(self, symbol: str, timeframe: Timeframe, 
                                 data: pd.DataFrame, strategies: Dict) -> Optional[Dict]:
        """Generate signal for a specific timeframe"""
        try:
            # Use the primary strategy for signal generation
            # In real implementation, you might use different strategies for different timeframes
            primary_strategy = list(strategies.values())[0] if strategies else None
            
            if primary_strategy and not data.empty:
                # Generate signal using the strategy
                signal = primary_strategy.analyze(data)
                
                if signal and signal.get('signal') not in ['NO TRADE', 'ERROR']:
                    return {
                        'timeframe': timeframe.value,
                        'signal': signal.get('signal'),
                        'confidence': signal.get('confidence', 0),
                        'strategy': signal.get('strategy', 'unknown'),
                        'price': signal.get('price', 0),
                        'timestamp': datetime.now(self.tz)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating {timeframe.value} signal for {symbol}: {e}")
            return None
    
    def _analyze_signal_confirmation(self, timeframe_signals: Dict[Timeframe, Dict]) -> Dict[str, Any]:
        """Analyze signal confirmation across timeframes"""
        try:
            # Group signals by type
            signal_groups = {}
            for timeframe, signal_data in timeframe_signals.items():
                signal_type = signal_data.get('signal')
                if signal_type not in signal_groups:
                    signal_groups[signal_type] = []
                signal_groups[signal_type].append({
                    'timeframe': timeframe,
                    'confidence': signal_data.get('confidence', 0),
                    'data': signal_data
                })
            
            # Find confirmed signals
            confirmed_signals = []
            for signal_type, signals in signal_groups.items():
                if len(signals) >= 2:  # At least 2 timeframes confirm
                    # Calculate weighted confidence
                    total_confidence = sum(s['confidence'] for s in signals)
                    avg_confidence = total_confidence / len(signals)
                    
                    confirmed_signals.append({
                        'signal_type': signal_type,
                        'confirming_timeframes': [s['timeframe'].value for s in signals],
                        'confirmation_count': len(signals),
                        'weighted_confidence': avg_confidence,
                        'signals': signals
                    })
            
            return {
                'total_signals': len(timeframe_signals),
                'confirmed_count': len(confirmed_signals),
                'signal_groups': signal_groups,
                'confirmed_signals': confirmed_signals
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing signal confirmation: {e}")
            return {
                'total_signals': 0,
                'confirmed_count': 0,
                'error': str(e)
            }
    
    def _calculate_signal_strength(self, confirmation_analysis: Dict) -> SignalStrength:
        """Calculate overall signal strength based on confirmations"""
        try:
            confirmed_count = confirmation_analysis.get('confirmed_count', 0)
            
            if confirmed_count >= 4:
                return SignalStrength.VERY_STRONG
            elif confirmed_count >= 3:
                return SignalStrength.STRONG
            elif confirmed_count >= 2:
                return SignalStrength.MODERATE
            else:
                return SignalStrength.WEAK
                
        except Exception as e:
            logger.error(f"❌ Error calculating signal strength: {e}")
            return SignalStrength.WEAK
    
    def get_timeframe_status(self, symbol: str) -> Dict[str, Any]:
        """Get current timeframe status for a symbol"""
        # This would return real-time status
        # For now, return mock data
        return {
            'symbol': symbol,
            'timeframes': {
                '1m': {'signals': 0, 'confirmed': 0, 'rejected': 0},
                '5m': {'signals': 0, 'confirmed': 0, 'rejected': 0},
                '15m': {'signals': 0, 'confirmed': 0, 'rejected': 0},
                '1h': {'signals': 0, 'confirmed': 0, 'rejected': 0}
            },
            'last_updated': datetime.now(self.tz)
        }
