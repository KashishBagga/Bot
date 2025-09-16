#!/usr/bin/env python3
"""
Market Regime Detection and Strategy Adaptation
Advanced strategies for different market conditions
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    SIDEWAYS = "SIDEWAYS"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"

class StrategyType(Enum):
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    BREAKOUT = "BREAKOUT"
    VOLATILITY = "VOLATILITY"
    ARBITRAGE = "ARBITRAGE"
    HEDGING = "HEDGING"

@dataclass
class RegimeSignal:
    """Market regime signal data structure"""
    regime: MarketRegime
    confidence: float
    strength: float
    duration: int
    indicators: Dict[str, float]
    timestamp: datetime

@dataclass
class StrategySignal:
    """Strategy signal data structure"""
    strategy_type: StrategyType
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    timestamp: datetime

class MarketRegimeDetector:
    """Advanced market regime detection system"""
    
    def __init__(self):
        self.regime_history = []
        self.regime_models = {}
        
    def detect_regime(self, data: pd.DataFrame) -> RegimeSignal:
        """Detect current market regime using multiple indicators"""
        try:
            # Calculate technical indicators
            indicators = self._calculate_indicators(data)
            
            # Detect regime using ensemble method
            regime_scores = self._calculate_regime_scores(indicators)
            
            # Find best regime
            best_regime = max(regime_scores.keys(), key=lambda x: regime_scores[x])
            confidence = regime_scores[best_regime]
            
            # Calculate regime strength
            strength = self._calculate_regime_strength(indicators, best_regime)
            
            # Estimate duration
            duration = self._estimate_regime_duration(indicators, best_regime)
            
            signal = RegimeSignal(
                regime=best_regime,
                confidence=confidence,
                strength=strength,
                duration=duration,
                indicators=indicators,
                timestamp=datetime.now()
            )
            
            self.regime_history.append(signal)
            
            logger.info(f"✅ Market regime detected: {best_regime.value} (confidence: {confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Regime detection failed: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive technical indicators"""
        try:
            # Price-based indicators
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            
            # Volatility indicators
            data['returns'] = data['close'].pct_change()
            data['volatility'] = data['returns'].rolling(20).std()
            data['atr'] = self._calculate_atr(data)
            
            # Momentum indicators
            data['rsi'] = self._calculate_rsi(data['close'])
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['momentum'] = data['close'].pct_change(20)
            
            # Trend indicators
            data['adx'] = self._calculate_adx(data)
            data['trend_strength'] = abs(data['close'] - data['sma_20']) / data['sma_20']
            
            # Volume indicators
            if 'volume' in data.columns:
                data['volume_sma'] = data['volume'].rolling(20).mean()
                data['volume_ratio'] = data['volume'] / data['volume_sma']
            else:
                data['volume_ratio'] = 1.0
            
            # Get latest values
            indicators = {
                'sma_20': data['sma_20'].iloc[-1],
                'sma_50': data['sma_50'].iloc[-1],
                'ema_12': data['ema_12'].iloc[-1],
                'ema_26': data['ema_26'].iloc[-1],
                'volatility': data['volatility'].iloc[-1],
                'atr': data['atr'].iloc[-1],
                'rsi': data['rsi'].iloc[-1],
                'macd': data['macd'].iloc[-1],
                'macd_signal': data['macd_signal'].iloc[-1],
                'momentum': data['momentum'].iloc[-1],
                'adx': data['adx'].iloc[-1],
                'trend_strength': data['trend_strength'].iloc[-1],
                'volume_ratio': data['volume_ratio'].iloc[-1],
                'current_price': data['close'].iloc[-1]
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Indicator calculation failed: {e}")
            return {}
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(period).mean()
            
            return atr
        except:
            return pd.Series([0] * len(data), index=data.index)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)"""
        try:
            # Calculate True Range
            tr = self._calculate_atr(data, 1)
            
            # Calculate Directional Movement
            dm_plus = np.where(
                (data['high'].diff() > data['low'].diff().abs()) & (data['high'].diff() > 0),
                data['high'].diff(), 0
            )
            dm_minus = np.where(
                (data['low'].diff().abs() > data['high'].diff()) & (data['low'].diff() < 0),
                data['low'].diff().abs(), 0
            )
            
            # Calculate smoothed values
            tr_smooth = tr.rolling(period).mean()
            dm_plus_smooth = pd.Series(dm_plus, index=data.index).rolling(period).mean()
            dm_minus_smooth = pd.Series(dm_minus, index=data.index).rolling(period).mean()
            
            # Calculate DI+ and DI-
            di_plus = 100 * (dm_plus_smooth / tr_smooth)
            di_minus = 100 * (dm_minus_smooth / tr_smooth)
            
            # Calculate ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(period).mean()
            
            return adx
        except:
            return pd.Series([25] * len(data), index=data.index)
    
    def _calculate_regime_scores(self, indicators: Dict[str, float]) -> Dict[MarketRegime, float]:
        """Calculate scores for each market regime"""
        try:
            scores = {}
            
            # Bull Trend Score
            bull_score = 0
            if indicators['sma_20'] > indicators['sma_50']:
                bull_score += 0.3
            if indicators['ema_12'] > indicators['ema_26']:
                bull_score += 0.2
            if indicators['momentum'] > 0.02:
                bull_score += 0.2
            if indicators['rsi'] > 50 and indicators['rsi'] < 70:
                bull_score += 0.2
            if indicators['adx'] > 25:
                bull_score += 0.1
            scores[MarketRegime.BULL_TREND] = bull_score
            
            # Bear Trend Score
            bear_score = 0
            if indicators['sma_20'] < indicators['sma_50']:
                bear_score += 0.3
            if indicators['ema_12'] < indicators['ema_26']:
                bear_score += 0.2
            if indicators['momentum'] < -0.02:
                bear_score += 0.2
            if indicators['rsi'] < 50 and indicators['rsi'] > 30:
                bear_score += 0.2
            if indicators['adx'] > 25:
                bear_score += 0.1
            scores[MarketRegime.BEAR_TREND] = bear_score
            
            # High Volatility Score
            high_vol_score = 0
            if indicators['volatility'] > 0.03:
                high_vol_score += 0.4
            if indicators['atr'] > indicators['current_price'] * 0.02:
                high_vol_score += 0.3
            if indicators['volume_ratio'] > 1.5:
                high_vol_score += 0.3
            scores[MarketRegime.HIGH_VOLATILITY] = high_vol_score
            
            # Low Volatility Score
            low_vol_score = 0
            if indicators['volatility'] < 0.015:
                low_vol_score += 0.4
            if indicators['atr'] < indicators['current_price'] * 0.01:
                low_vol_score += 0.3
            if indicators['volume_ratio'] < 0.8:
                low_vol_score += 0.3
            scores[MarketRegime.LOW_VOLATILITY] = low_vol_score
            
            # Sideways Score
            sideways_score = 0
            if abs(indicators['trend_strength']) < 0.02:
                sideways_score += 0.4
            if indicators['adx'] < 20:
                sideways_score += 0.3
            if abs(indicators['momentum']) < 0.01:
                sideways_score += 0.3
            scores[MarketRegime.SIDEWAYS] = sideways_score
            
            # Breakout Score
            breakout_score = 0
            if indicators['volume_ratio'] > 2.0:
                breakout_score += 0.4
            if indicators['trend_strength'] > 0.03:
                breakout_score += 0.3
            if indicators['rsi'] > 70 or indicators['rsi'] < 30:
                breakout_score += 0.3
            scores[MarketRegime.BREAKOUT] = breakout_score
            
            # Reversal Score
            reversal_score = 0
            if indicators['rsi'] > 80 or indicators['rsi'] < 20:
                reversal_score += 0.4
            if indicators['macd'] * indicators['macd_signal'] < 0:
                reversal_score += 0.3
            if indicators['volume_ratio'] > 1.5:
                reversal_score += 0.3
            scores[MarketRegime.REVERSAL] = reversal_score
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ Regime score calculation failed: {e}")
            return {}
    
    def _calculate_regime_strength(self, indicators: Dict[str, float], regime: MarketRegime) -> float:
        """Calculate strength of the detected regime"""
        try:
            if regime == MarketRegime.BULL_TREND:
                return min(1.0, indicators['momentum'] * 10 + indicators['trend_strength'] * 5)
            elif regime == MarketRegime.BEAR_TREND:
                return min(1.0, abs(indicators['momentum']) * 10 + indicators['trend_strength'] * 5)
            elif regime == MarketRegime.HIGH_VOLATILITY:
                return min(1.0, indicators['volatility'] * 20)
            elif regime == MarketRegime.LOW_VOLATILITY:
                return min(1.0, (0.02 - indicators['volatility']) * 20)
            elif regime == MarketRegime.SIDEWAYS:
                return min(1.0, (0.02 - abs(indicators['trend_strength'])) * 25)
            elif regime == MarketRegime.BREAKOUT:
                return min(1.0, indicators['volume_ratio'] * 0.5)
            elif regime == MarketRegime.REVERSAL:
                return min(1.0, abs(indicators['rsi'] - 50) / 50)
            else:
                return 0.5
        except:
            return 0.5
    
    def _estimate_regime_duration(self, indicators: Dict[str, float], regime: MarketRegime) -> int:
        """Estimate duration of the current regime in days"""
        try:
            # Base duration estimates
            base_durations = {
                MarketRegime.BULL_TREND: 30,
                MarketRegime.BEAR_TREND: 25,
                MarketRegime.HIGH_VOLATILITY: 10,
                MarketRegime.LOW_VOLATILITY: 20,
                MarketRegime.SIDEWAYS: 15,
                MarketRegime.BREAKOUT: 5,
                MarketRegime.REVERSAL: 3
            }
            
            base_duration = base_durations.get(regime, 15)
            
            # Adjust based on strength
            strength = self._calculate_regime_strength(indicators, regime)
            adjusted_duration = int(base_duration * (0.5 + strength))
            
            return max(1, min(60, adjusted_duration))  # Between 1 and 60 days
            
        except:
            return 15

class RegimeAdaptiveStrategies:
    """Strategies that adapt to market regimes"""
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.strategies = {}
        
    def generate_regime_adaptive_signals(self, data: pd.DataFrame, 
                                       symbol: str) -> List[StrategySignal]:
        """Generate signals based on current market regime"""
        try:
            # Detect current regime
            regime_signal = self.regime_detector.detect_regime(data)
            
            if not regime_signal:
                return []
            
            # Generate strategy signals based on regime
            signals = []
            
            if regime_signal.regime == MarketRegime.BULL_TREND:
                signals.extend(self._generate_bull_trend_signals(data, symbol, regime_signal))
            elif regime_signal.regime == MarketRegime.BEAR_TREND:
                signals.extend(self._generate_bear_trend_signals(data, symbol, regime_signal))
            elif regime_signal.regime == MarketRegime.HIGH_VOLATILITY:
                signals.extend(self._generate_high_volatility_signals(data, symbol, regime_signal))
            elif regime_signal.regime == MarketRegime.LOW_VOLATILITY:
                signals.extend(self._generate_low_volatility_signals(data, symbol, regime_signal))
            elif regime_signal.regime == MarketRegime.SIDEWAYS:
                signals.extend(self._generate_sideways_signals(data, symbol, regime_signal))
            elif regime_signal.regime == MarketRegime.BREAKOUT:
                signals.extend(self._generate_breakout_signals(data, symbol, regime_signal))
            elif regime_signal.regime == MarketRegime.REVERSAL:
                signals.extend(self._generate_reversal_signals(data, symbol, regime_signal))
            
            logger.info(f"✅ Generated {len(signals)} regime-adaptive signals for {symbol}")
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Regime-adaptive signal generation failed: {e}")
            return []
    
    def _generate_bull_trend_signals(self, data: pd.DataFrame, symbol: str, 
                                   regime_signal: RegimeSignal) -> List[StrategySignal]:
        """Generate signals for bull trend regime"""
        signals = []
        
        # Momentum strategy
        if regime_signal.indicators['momentum'] > 0.03:
            signals.append(StrategySignal(
                strategy_type=StrategyType.MOMENTUM,
                symbol=symbol,
                signal='BUY',
                confidence=regime_signal.confidence * 0.8,
                entry_price=data['close'].iloc[-1],
                stop_loss=data['close'].iloc[-1] * 0.95,
                take_profit=data['close'].iloc[-1] * 1.10,
                position_size=0.1,
                reasoning=f"Bull trend momentum strategy - momentum: {regime_signal.indicators['momentum']:.3f}",
                timestamp=datetime.now()
            ))
        
        # Breakout strategy
        if regime_signal.indicators['volume_ratio'] > 1.5:
            signals.append(StrategySignal(
                strategy_type=StrategyType.BREAKOUT,
                symbol=symbol,
                signal='BUY',
                confidence=regime_signal.confidence * 0.7,
                entry_price=data['close'].iloc[-1],
                stop_loss=data['close'].iloc[-1] * 0.97,
                take_profit=data['close'].iloc[-1] * 1.08,
                position_size=0.08,
                reasoning=f"Bull trend breakout strategy - volume ratio: {regime_signal.indicators['volume_ratio']:.2f}",
                timestamp=datetime.now()
            ))
        
        return signals
    
    def _generate_bear_trend_signals(self, data: pd.DataFrame, symbol: str, 
                                   regime_signal: RegimeSignal) -> List[StrategySignal]:
        """Generate signals for bear trend regime"""
        signals = []
        
        # Momentum strategy (short)
        if regime_signal.indicators['momentum'] < -0.03:
            signals.append(StrategySignal(
                strategy_type=StrategyType.MOMENTUM,
                symbol=symbol,
                signal='SELL',
                confidence=regime_signal.confidence * 0.8,
                entry_price=data['close'].iloc[-1],
                stop_loss=data['close'].iloc[-1] * 1.05,
                take_profit=data['close'].iloc[-1] * 0.90,
                position_size=0.1,
                reasoning=f"Bear trend momentum strategy - momentum: {regime_signal.indicators['momentum']:.3f}",
                timestamp=datetime.now()
            ))
        
        # Mean reversion strategy
        if regime_signal.indicators['rsi'] < 30:
            signals.append(StrategySignal(
                strategy_type=StrategyType.MEAN_REVERSION,
                symbol=symbol,
                signal='BUY',
                confidence=regime_signal.confidence * 0.6,
                entry_price=data['close'].iloc[-1],
                stop_loss=data['close'].iloc[-1] * 0.95,
                take_profit=data['close'].iloc[-1] * 1.05,
                position_size=0.05,
                reasoning=f"Bear trend mean reversion - RSI: {regime_signal.indicators['rsi']:.1f}",
                timestamp=datetime.now()
            ))
        
        return signals
    
    def _generate_high_volatility_signals(self, data: pd.DataFrame, symbol: str, 
                                        regime_signal: RegimeSignal) -> List[StrategySignal]:
        """Generate signals for high volatility regime"""
        signals = []
        
        # Volatility strategy
        signals.append(StrategySignal(
            strategy_type=StrategyType.VOLATILITY,
            symbol=symbol,
            signal='BUY',
            confidence=regime_signal.confidence * 0.7,
            entry_price=data['close'].iloc[-1],
            stop_loss=data['close'].iloc[-1] * 0.90,
            take_profit=data['close'].iloc[-1] * 1.15,
            position_size=0.06,
            reasoning=f"High volatility strategy - volatility: {regime_signal.indicators['volatility']:.3f}",
            timestamp=datetime.now()
        ))
        
        return signals
    
    def _generate_low_volatility_signals(self, data: pd.DataFrame, symbol: str, 
                                       regime_signal: RegimeSignal) -> List[StrategySignal]:
        """Generate signals for low volatility regime"""
        signals = []
        
        # Mean reversion strategy
        if abs(regime_signal.indicators['momentum']) < 0.01:
            signals.append(StrategySignal(
                strategy_type=StrategyType.MEAN_REVERSION,
                symbol=symbol,
                signal='BUY',
                confidence=regime_signal.confidence * 0.6,
                entry_price=data['close'].iloc[-1],
                stop_loss=data['close'].iloc[-1] * 0.98,
                take_profit=data['close'].iloc[-1] * 1.03,
                position_size=0.08,
                reasoning=f"Low volatility mean reversion - momentum: {regime_signal.indicators['momentum']:.3f}",
                timestamp=datetime.now()
            ))
        
        return signals
    
    def _generate_sideways_signals(self, data: pd.DataFrame, symbol: str, 
                                 regime_signal: RegimeSignal) -> List[StrategySignal]:
        """Generate signals for sideways regime"""
        signals = []
        
        # Mean reversion strategy
        signals.append(StrategySignal(
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol=symbol,
            signal='BUY',
            confidence=regime_signal.confidence * 0.7,
            entry_price=data['close'].iloc[-1],
            stop_loss=data['close'].iloc[-1] * 0.97,
            take_profit=data['close'].iloc[-1] * 1.04,
            position_size=0.1,
            reasoning=f"Sideways mean reversion - trend strength: {regime_signal.indicators['trend_strength']:.3f}",
            timestamp=datetime.now()
        ))
        
        return signals
    
    def _generate_breakout_signals(self, data: pd.DataFrame, symbol: str, 
                                 regime_signal: RegimeSignal) -> List[StrategySignal]:
        """Generate signals for breakout regime"""
        signals = []
        
        # Breakout strategy
        signals.append(StrategySignal(
            strategy_type=StrategyType.BREAKOUT,
            symbol=symbol,
            signal='BUY',
            confidence=regime_signal.confidence * 0.8,
            entry_price=data['close'].iloc[-1],
            stop_loss=data['close'].iloc[-1] * 0.95,
            take_profit=data['close'].iloc[-1] * 1.12,
            position_size=0.12,
            reasoning=f"Breakout strategy - volume ratio: {regime_signal.indicators['volume_ratio']:.2f}",
            timestamp=datetime.now()
        ))
        
        return signals
    
    def _generate_reversal_signals(self, data: pd.DataFrame, symbol: str, 
                                 regime_signal: RegimeSignal) -> List[StrategySignal]:
        """Generate signals for reversal regime"""
        signals = []
        
        # Mean reversion strategy
        if regime_signal.indicators['rsi'] > 80:
            signals.append(StrategySignal(
                strategy_type=StrategyType.MEAN_REVERSION,
                symbol=symbol,
                signal='SELL',
                confidence=regime_signal.confidence * 0.7,
                entry_price=data['close'].iloc[-1],
                stop_loss=data['close'].iloc[-1] * 1.05,
                take_profit=data['close'].iloc[-1] * 0.95,
                position_size=0.08,
                reasoning=f"Reversal strategy - RSI: {regime_signal.indicators['rsi']:.1f}",
                timestamp=datetime.now()
            ))
        elif regime_signal.indicators['rsi'] < 20:
            signals.append(StrategySignal(
                strategy_type=StrategyType.MEAN_REVERSION,
                symbol=symbol,
                signal='BUY',
                confidence=regime_signal.confidence * 0.7,
                entry_price=data['close'].iloc[-1],
                stop_loss=data['close'].iloc[-1] * 0.95,
                take_profit=data['close'].iloc[-1] * 1.05,
                position_size=0.08,
                reasoning=f"Reversal strategy - RSI: {regime_signal.indicators['rsi']:.1f}",
                timestamp=datetime.now()
            ))
        
        return signals
    
    def run_regime_analysis(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Run comprehensive regime analysis"""
        try:
            logger.info(f"🚀 Starting regime analysis for {symbol}")
            
            # Detect regime
            regime_signal = self.regime_detector.detect_regime(data)
            
            if not regime_signal:
                return {'error': 'Failed to detect market regime'}
            
            # Generate adaptive signals
            signals = self.generate_regime_adaptive_signals(data, symbol)
            
            # Compile results
            results = {
                'symbol': symbol,
                'regime_detection': {
                    'regime': regime_signal.regime.value,
                    'confidence': regime_signal.confidence,
                    'strength': regime_signal.strength,
                    'duration': regime_signal.duration,
                    'indicators': regime_signal.indicators,
                    'timestamp': regime_signal.timestamp.isoformat()
                },
                'strategy_signals': [
                    {
                        'strategy_type': signal.strategy_type.value,
                        'signal': signal.signal,
                        'confidence': signal.confidence,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'position_size': signal.position_size,
                        'reasoning': signal.reasoning,
                        'timestamp': signal.timestamp.isoformat()
                    } for signal in signals
                ],
                'analysis_summary': {
                    'total_signals': len(signals),
                    'buy_signals': len([s for s in signals if s.signal == 'BUY']),
                    'sell_signals': len([s for s in signals if s.signal == 'SELL']),
                    'avg_confidence': np.mean([s.confidence for s in signals]) if signals else 0,
                    'regime_stability': regime_signal.strength
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Regime analysis completed for {symbol}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Regime analysis failed: {e}")
            return {'error': str(e)}

def main():
    """Run market regime detection and strategy adaptation"""
    regime_system = RegimeAdaptiveStrategies()
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate price data with different regimes
    returns = np.random.normal(0.001, 0.02, 100)  # Bullish trend
    returns[50:] = np.random.normal(-0.0005, 0.025, 50)  # Bearish trend
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Run analysis
    results = regime_system.run_regime_analysis(data, 'NSE:NIFTY50-INDEX')
    
    print("\n" + "="*80)
    print("📊 MARKET REGIME DETECTION AND STRATEGY ADAPTATION")
    print("="*80)
    
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return False
    
    print(f"\n🎯 REGIME DETECTION:")
    regime = results['regime_detection']
    print(f"   Regime: {regime['regime']}")
    print(f"   Confidence: {regime['confidence']:.2f}")
    print(f"   Strength: {regime['strength']:.2f}")
    print(f"   Duration: {regime['duration']} days")
    
    print(f"\n📊 KEY INDICATORS:")
    indicators = regime['indicators']
    for indicator, value in indicators.items():
        if isinstance(value, float):
            print(f"   {indicator}: {value:.4f}")
        else:
            print(f"   {indicator}: {value}")
    
    print(f"\n📋 STRATEGY SIGNALS:")
    signals = results['strategy_signals']
    for i, signal in enumerate(signals, 1):
        print(f"\n   {i}. {signal['strategy_type']} - {signal['signal']}")
        print(f"      Confidence: {signal['confidence']:.2f}")
        print(f"      Entry: {signal['entry_price']:.2f}")
        print(f"      Stop Loss: {signal['stop_loss']:.2f}")
        print(f"      Take Profit: {signal['take_profit']:.2f}")
        print(f"      Position Size: {signal['position_size']:.2f}")
        print(f"      Reasoning: {signal['reasoning']}")
    
    print(f"\n📈 ANALYSIS SUMMARY:")
    summary = results['analysis_summary']
    print(f"   Total Signals: {summary['total_signals']}")
    print(f"   Buy Signals: {summary['buy_signals']}")
    print(f"   Sell Signals: {summary['sell_signals']}")
    print(f"   Average Confidence: {summary['avg_confidence']:.2f}")
    print(f"   Regime Stability: {summary['regime_stability']:.2f}")
    
    print("\n" + "="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
