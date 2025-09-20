#!/usr/bin/env python3
"""
Advanced Multi-Strategy System with Dynamic Strategy Selection
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    SCALPING = "scalping"
    SWING = "swing"

@dataclass
class StrategyConfig:
    """Configuration for individual strategies."""
    name: str
    type: StrategyType
    timeframes: List[str]
    symbols: List[str]
    weight: float  # Portfolio weight (0.0 to 1.0)
    risk_per_trade: float  # Risk per trade (0.0 to 1.0)
    max_positions: int
    enabled: bool = True

class AdvancedMultiStrategy:
    """Advanced multi-strategy system with dynamic allocation."""
    
    def __init__(self):
        self.strategies = {}
        self.strategy_performance = {}
        self.dynamic_weights = {}
        self.market_regime = "NEUTRAL"
        self.volatility_threshold = 0.02
        
        # Initialize strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all available strategies."""
        
        # 1. Momentum Strategy
        self.strategies['momentum_ema'] = StrategyConfig(
            name="Momentum EMA",
            type=StrategyType.MOMENTUM,
            timeframes=["5m", "15m", "1h"],
            symbols=["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"],
            weight=0.25,
            risk_per_trade=0.02,
            max_positions=5
        )
        
        # 2. Mean Reversion Strategy
        self.strategies['mean_reversion_rsi'] = StrategyConfig(
            name="Mean Reversion RSI",
            type=StrategyType.MEAN_REVERSION,
            timeframes=["15m", "30m", "1h"],
            symbols=["NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ"],
            weight=0.20,
            risk_per_trade=0.015,
            max_positions=3
        )
        
        # 3. Breakout Strategy
        self.strategies['breakout_volume'] = StrategyConfig(
            name="Volume Breakout",
            type=StrategyType.BREAKOUT,
            timeframes=["15m", "1h", "4h"],
            symbols=["NSE:FINNIFTY-INDEX", "NSE:NIFTY50-INDEX"],
            weight=0.20,
            risk_per_trade=0.025,
            max_positions=4
        )
        
        # 4. Trend Following Strategy
        self.strategies['trend_supertrend'] = StrategyConfig(
            name="SuperTrend Following",
            type=StrategyType.TREND_FOLLOWING,
            timeframes=["1h", "4h", "1d"],
            symbols=["NSE:NIFTYBANK-INDEX", "NSE:RELIANCE-EQ"],
            weight=0.25,
            risk_per_trade=0.02,
            max_positions=4
        )
        
        # 5. Scalping Strategy
        self.strategies['scalping_macd'] = StrategyConfig(
            name="MACD Scalping",
            type=StrategyType.SCALPING,
            timeframes=["1m", "5m"],
            symbols=["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"],
            weight=0.10,
            risk_per_trade=0.01,
            max_positions=2
        )
        
        logger.info(f"✅ Initialized {len(self.strategies)} strategies")
    
    def analyze_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Analyze current market regime."""
        try:
            # Simple regime detection based on volatility and trend
            volatility = self._calculate_market_volatility(market_data)
            trend_strength = self._calculate_trend_strength(market_data)
            
            if volatility > self.volatility_threshold * 2:
                if trend_strength > 0.5:
                    regime = "HIGH_VOLATILITY_TRENDING"
                else:
                    regime = "HIGH_VOLATILITY_RANGING"
            elif trend_strength > 0.7:
                regime = "STRONG_TREND"
            elif trend_strength < 0.3:
                regime = "RANGING"
            else:
                regime = "NEUTRAL"
            
            if regime != self.market_regime:
                logger.info(f"🔄 Market regime changed: {self.market_regime} -> {regime}")
                self.market_regime = regime
                self._adjust_strategy_weights()
            
            return regime
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return "NEUTRAL"
    
    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility."""
        try:
            # Simple volatility calculation
            prices = []
            for symbol, data in market_data.items():
                if 'close' in data:
                    prices.extend(data['close'][-20:])  # Last 20 prices
            
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
                return volatility
            
            return 0.01  # Default low volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.01
    
    def _calculate_trend_strength(self, market_data: Dict[str, Any]) -> float:
        """Calculate overall trend strength."""
        try:
            trend_scores = []
            
            for symbol, data in market_data.items():
                if 'close' in data and len(data['close']) >= 20:
                    prices = data['close'][-20:]
                    
                    # Simple trend strength using linear regression
                    x = np.arange(len(prices))
                    slope = np.polyfit(x, prices, 1)[0]
                    
                    # Normalize slope
                    price_range = max(prices) - min(prices)
                    if price_range > 0:
                        trend_score = abs(slope) / price_range
                        trend_scores.append(min(trend_score, 1.0))
            
            if trend_scores:
                return np.mean(trend_scores)
            
            return 0.5  # Neutral
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.5
    
    def _adjust_strategy_weights(self):
        """Adjust strategy weights based on market regime."""
        
        regime_weights = {
            "HIGH_VOLATILITY_TRENDING": {
                "momentum_ema": 0.35,
                "trend_supertrend": 0.35,
                "breakout_volume": 0.20,
                "mean_reversion_rsi": 0.05,
                "scalping_macd": 0.05
            },
            "HIGH_VOLATILITY_RANGING": {
                "mean_reversion_rsi": 0.40,
                "scalping_macd": 0.30,
                "breakout_volume": 0.15,
                "momentum_ema": 0.10,
                "trend_supertrend": 0.05
            },
            "STRONG_TREND": {
                "trend_supertrend": 0.40,
                "momentum_ema": 0.30,
                "breakout_volume": 0.20,
                "mean_reversion_rsi": 0.05,
                "scalping_macd": 0.05
            },
            "RANGING": {
                "mean_reversion_rsi": 0.45,
                "scalping_macd": 0.25,
                "breakout_volume": 0.15,
                "momentum_ema": 0.10,
                "trend_supertrend": 0.05
            },
            "NEUTRAL": {
                "momentum_ema": 0.25,
                "mean_reversion_rsi": 0.20,
                "breakout_volume": 0.20,
                "trend_supertrend": 0.25,
                "scalping_macd": 0.10
            }
        }
        
        if self.market_regime in regime_weights:
            new_weights = regime_weights[self.market_regime]
            
            for strategy_name, weight in new_weights.items():
                if strategy_name in self.strategies:
                    self.strategies[strategy_name].weight = weight
                    
            logger.info(f"📊 Adjusted strategy weights for regime: {self.market_regime}")
    
    def generate_diversified_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate signals from all active strategies."""
        all_signals = []
        
        # Analyze market regime first
        self.analyze_market_regime(market_data)
        
        # Generate signals from each strategy
        for strategy_name, config in self.strategies.items():
            if not config.enabled:
                continue
            
            try:
                signals = self._generate_strategy_signals(strategy_name, config, market_data)
                
                # Add strategy metadata to signals
                for signal in signals:
                    signal['strategy_name'] = strategy_name
                    signal['strategy_type'] = config.type.value
                    signal['strategy_weight'] = config.weight
                    signal['max_risk'] = config.risk_per_trade
                
                all_signals.extend(signals)
                
            except Exception as e:
                logger.error(f"Error generating signals for {strategy_name}: {e}")
        
        # Rank and filter signals
        ranked_signals = self._rank_and_filter_signals(all_signals)
        
        logger.info(f"📈 Generated {len(ranked_signals)} diversified signals from {len(self.strategies)} strategies")
        return ranked_signals
    
    def _generate_strategy_signals(self, strategy_name: str, config: StrategyConfig, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate signals for a specific strategy."""
        signals = []
        
        for symbol in config.symbols:
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            
            # Generate signals based on strategy type
            if config.type == StrategyType.MOMENTUM:
                signals.extend(self._momentum_signals(symbol, data, config))
            elif config.type == StrategyType.MEAN_REVERSION:
                signals.extend(self._mean_reversion_signals(symbol, data, config))
            elif config.type == StrategyType.BREAKOUT:
                signals.extend(self._breakout_signals(symbol, data, config))
            elif config.type == StrategyType.TREND_FOLLOWING:
                signals.extend(self._trend_following_signals(symbol, data, config))
            elif config.type == StrategyType.SCALPING:
                signals.extend(self._scalping_signals(symbol, data, config))
        
        return signals
    
    def _momentum_signals(self, symbol: str, data: Dict[str, Any], config: StrategyConfig) -> List[Dict[str, Any]]:
        """Generate momentum-based signals."""
        signals = []
        
        try:
            if 'close' in data and len(data['close']) >= 20:
                prices = data['close']
                
                # EMA crossover
                ema_short = self._calculate_ema(prices, 9)
                ema_long = self._calculate_ema(prices, 21)
                
                if len(ema_short) >= 2 and len(ema_long) >= 2:
                    if ema_short[-1] > ema_long[-1] and ema_short[-2] <= ema_long[-2]:
                        signals.append({
                            'symbol': symbol,
                            'signal': 'BUY CALL',
                            'confidence': 75.0,
                            'price': prices[-1],
                            'timestamp': datetime.now(),
                            'timeframe': '15m',
                            'strategy': config.name
                        })
                    elif ema_short[-1] < ema_long[-1] and ema_short[-2] >= ema_long[-2]:
                        signals.append({
                            'symbol': symbol,
                            'signal': 'BUY PUT',
                            'confidence': 75.0,
                            'price': prices[-1],
                            'timestamp': datetime.now(),
                            'timeframe': '15m',
                            'strategy': config.name
                        })
        
        except Exception as e:
            logger.error(f"Error generating momentum signals for {symbol}: {e}")
        
        return signals
    
    def _mean_reversion_signals(self, symbol: str, data: Dict[str, Any], config: StrategyConfig) -> List[Dict[str, Any]]:
        """Generate mean reversion signals."""
        signals = []
        
        try:
            if 'close' in data and len(data['close']) >= 14:
                prices = data['close']
                rsi = self._calculate_rsi(prices, 14)
                
                if len(rsi) > 0:
                    current_rsi = rsi[-1]
                    
                    if current_rsi < 30:  # Oversold
                        signals.append({
                            'symbol': symbol,
                            'signal': 'BUY CALL',
                            'confidence': 70.0,
                            'price': prices[-1],
                            'timestamp': datetime.now(),
                            'timeframe': '30m',
                            'strategy': config.name
                        })
                    elif current_rsi > 70:  # Overbought
                        signals.append({
                            'symbol': symbol,
                            'signal': 'BUY PUT',
                            'confidence': 70.0,
                            'price': prices[-1],
                            'timestamp': datetime.now(),
                            'timeframe': '30m',
                            'strategy': config.name
                        })
        
        except Exception as e:
            logger.error(f"Error generating mean reversion signals for {symbol}: {e}")
        
        return signals
    
    def _breakout_signals(self, symbol: str, data: Dict[str, Any], config: StrategyConfig) -> List[Dict[str, Any]]:
        """Generate breakout signals."""
        signals = []
        
        try:
            if 'high' in data and 'low' in data and 'close' in data and len(data['close']) >= 20:
                highs = data['high']
                lows = data['low']
                closes = data['close']
                
                # Support and resistance levels
                resistance = max(highs[-10:])
                support = min(lows[-10:])
                current_price = closes[-1]
                
                if current_price > resistance * 1.001:  # Breakout above resistance
                    signals.append({
                        'symbol': symbol,
                        'signal': 'BUY CALL',
                        'confidence': 80.0,
                        'price': current_price,
                        'timestamp': datetime.now(),
                        'timeframe': '1h',
                        'strategy': config.name
                    })
                elif current_price < support * 0.999:  # Breakdown below support
                    signals.append({
                        'symbol': symbol,
                        'signal': 'BUY PUT',
                        'confidence': 80.0,
                        'price': current_price,
                        'timestamp': datetime.now(),
                        'timeframe': '1h',
                        'strategy': config.name
                    })
        
        except Exception as e:
            logger.error(f"Error generating breakout signals for {symbol}: {e}")
        
        return signals
    
    def _trend_following_signals(self, symbol: str, data: Dict[str, Any], config: StrategyConfig) -> List[Dict[str, Any]]:
        """Generate trend following signals."""
        signals = []
        
        try:
            if 'high' in data and 'low' in data and 'close' in data and len(data['close']) >= 10:
                highs = data['high']
                lows = data['low']
                closes = data['close']
                
                # SuperTrend calculation (simplified)
                atr = self._calculate_atr(highs, lows, closes, 10)
                if len(atr) > 0:
                    hl2 = [(h + l) / 2 for h, l in zip(highs, lows)]
                    multiplier = 3.0
                    
                    if len(hl2) >= 2 and len(atr) >= 2:
                        upper_band = hl2[-1] + (multiplier * atr[-1])
                        lower_band = hl2[-1] - (multiplier * atr[-1])
                        
                        if closes[-1] > upper_band:
                            signals.append({
                                'symbol': symbol,
                                'signal': 'BUY CALL',
                                'confidence': 85.0,
                                'price': closes[-1],
                                'timestamp': datetime.now(),
                                'timeframe': '4h',
                                'strategy': config.name
                            })
                        elif closes[-1] < lower_band:
                            signals.append({
                                'symbol': symbol,
                                'signal': 'BUY PUT',
                                'confidence': 85.0,
                                'price': closes[-1],
                                'timestamp': datetime.now(),
                                'timeframe': '4h',
                                'strategy': config.name
                            })
        
        except Exception as e:
            logger.error(f"Error generating trend following signals for {symbol}: {e}")
        
        return signals
    
    def _scalping_signals(self, symbol: str, data: Dict[str, Any], config: StrategyConfig) -> List[Dict[str, Any]]:
        """Generate scalping signals."""
        signals = []
        
        try:
            if 'close' in data and len(data['close']) >= 26:
                prices = data['close']
                
                # MACD calculation
                ema12 = self._calculate_ema(prices, 12)
                ema26 = self._calculate_ema(prices, 26)
                
                if len(ema12) >= 2 and len(ema26) >= 2:
                    macd = [e12 - e26 for e12, e26 in zip(ema12, ema26)]
                    signal_line = self._calculate_ema(macd, 9)
                    
                    if len(macd) >= 2 and len(signal_line) >= 2:
                        if macd[-1] > signal_line[-1] and macd[-2] <= signal_line[-2]:
                            signals.append({
                                'symbol': symbol,
                                'signal': 'BUY CALL',
                                'confidence': 65.0,
                                'price': prices[-1],
                                'timestamp': datetime.now(),
                                'timeframe': '5m',
                                'strategy': config.name
                            })
                        elif macd[-1] < signal_line[-1] and macd[-2] >= signal_line[-2]:
                            signals.append({
                                'symbol': symbol,
                                'signal': 'BUY PUT',
                                'confidence': 65.0,
                                'price': prices[-1],
                                'timestamp': datetime.now(),
                                'timeframe': '5m',
                                'strategy': config.name
                            })
        
        except Exception as e:
            logger.error(f"Error generating scalping signals for {symbol}: {e}")
        
        return signals
    
    def _rank_and_filter_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank and filter signals based on multiple criteria."""
        
        # Calculate composite score for each signal
        for signal in signals:
            score = 0.0
            
            # Base confidence score
            score += signal.get('confidence', 50) / 100.0
            
            # Strategy weight bonus
            score += signal.get('strategy_weight', 0.2) * 0.5
            
            # Market regime compatibility
            strategy_type = signal.get('strategy_type', '')
            if self.market_regime == "STRONG_TREND" and strategy_type in ['momentum', 'trend_following']:
                score += 0.3
            elif self.market_regime == "RANGING" and strategy_type in ['mean_reversion', 'scalping']:
                score += 0.3
            elif self.market_regime == "HIGH_VOLATILITY_TRENDING" and strategy_type in ['breakout', 'momentum']:
                score += 0.3
            
            signal['composite_score'] = score
        
        # Sort by composite score
        signals.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Filter top signals (max 10)
        return signals[:10]
    
    # Helper methods for technical indicators
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return []
        
        ema = []
        multiplier = 2 / (period + 1)
        
        # First EMA is SMA
        ema.append(sum(prices[:period]) / period)
        
        for i in range(period, len(prices)):
            ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(ema_value)
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int) -> List[float]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return []
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(delta, 0) for delta in deltas]
        losses = [-min(delta, 0) for delta in deltas]
        
        rsi = []
        
        # First RSI calculation
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))
        else:
            rsi.append(100)
        
        # Subsequent RSI calculations
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))
            else:
                rsi.append(100)
        
        return rsi
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> List[float]:
        """Calculate Average True Range."""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return []
        
        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        atr = []
        if len(true_ranges) >= period:
            # First ATR is simple average
            atr.append(sum(true_ranges[:period]) / period)
            
            # Subsequent ATRs use smoothing
            for i in range(period, len(true_ranges)):
                atr_value = (atr[-1] * (period - 1) + true_ranges[i]) / period
                atr.append(atr_value)
        
        return atr
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance statistics for all strategies."""
        return {
            'active_strategies': len([s for s in self.strategies.values() if s.enabled]),
            'total_strategies': len(self.strategies),
            'market_regime': self.market_regime,
            'strategy_weights': {name: config.weight for name, config in self.strategies.items()},
            'strategy_performance': self.strategy_performance
        }
