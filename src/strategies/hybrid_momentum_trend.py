#!/usr/bin/env python3
"""
Hybrid Momentum Trend Strategy
Combines SuperTrend, MACD, RSI, and EMA for high-confidence signals
"""

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from src.core.indicators import indicators

class HybridMomentumTrend(Strategy):
    """
    Hybrid strategy combining multiple indicators for high-confidence signals
    - SuperTrend for trend direction
    - MACD for momentum confirmation
    - RSI for overbought/oversold conditions
    - EMA for trend alignment
    - Volume for confirmation
    """
    
    def __init__(self, params: Dict[str, Any] = None, timeframe_data: Optional[Dict[str, pd.DataFrame]] = None):
        """Initialize the Hybrid Momentum Trend strategy."""
        default_params = {
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'volume_threshold': 1.3,
            'momentum_threshold': 0.8,
            'trend_strength_min': 0.5
        }
        
        if params:
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        else:
            params = default_params
            
        super().__init__("hybrid_momentum_trend", params)
        self.timeframe_data = timeframe_data or {}
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data."""
        # SuperTrend indicators
        data['supertrend'] = indicators.supertrend(data, period=10, multiplier=3)
        data['supertrend_direction'] = data['supertrend'].apply(lambda x: 1 if x > 0 else -1)
        
        # EMA indicators
        data['ema_21'] = indicators.ema(data, period=21)
        data['ema_50'] = indicators.ema(data, period=50)
        data['ema_200'] = indicators.ema(data, period=200)
        
        # MACD indicators
        data['macd'] = indicators.macd(data, fast=12, slow=26, signal=9)
        data['macd_signal'] = indicators.macd_signal(data, fast=12, slow=26, signal=9)
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # RSI
        data['rsi'] = indicators.rsi(data, period=14)
        
        # Volume analysis
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Momentum indicators
        data['price_momentum'] = data['close'].pct_change(5) * 100
        data['trend_strength'] = abs(data['ema_21'] - data['ema_50']) / data['ema_50'] * 100
        
        # ATR for volatility
        data['atr'] = indicators.atr(data, period=14)
        
        return data
    
    def analyze(self, candle: pd.Series, index: int, df: pd.DataFrame, future_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Analyze data and generate high-confidence hybrid signals."""
        if index < 50 or future_data is None or future_data.empty:
            return None
            
        # Get indicator values
        supertrend = candle.get('supertrend', 0)
        supertrend_direction = candle.get('supertrend_direction', 0)
        ema_21 = candle.get('ema_21', 0)
        ema_50 = candle.get('ema_50', 0)
        ema_200 = candle.get('ema_200', 0)
        macd = candle.get('macd', 0)
        macd_signal = candle.get('macd_signal', 0)
        macd_histogram = candle.get('macd_histogram', 0)
        rsi = candle.get('rsi', 50)
        volume_ratio = candle.get('volume_ratio', 1.0)
        price_momentum = candle.get('price_momentum', 0)
        trend_strength = candle.get('trend_strength', 0)
        atr = candle.get('atr', candle['close'] * 0.01)
        price = candle['close']
        
        # OPTIMIZATION: Multi-factor signal validation
        signal = "NO TRADE"
        confidence_score = 0
        
        # Check for valid indicator values
        if supertrend <= 0 or ema_21 <= 0 or ema_50 <= 0 or ema_200 <= 0:
            return None
            
        # OPTIMIZATION: Enhanced hybrid signal conditions
        if supertrend_direction > 0 and price > supertrend:  # Bullish SuperTrend
            # Multi-factor bullish validation
            trend_aligned = price > ema_21 > ema_50 > ema_200  # Strong uptrend
            macd_bullish = macd > macd_signal and macd_histogram > 0  # MACD bullish
            rsi_good = 40 < rsi < 70  # RSI not overbought
            volume_good = volume_ratio > self.params['volume_threshold']  # Strong volume
            momentum_good = price_momentum > self.params['momentum_threshold']  # Positive momentum
            trend_strong = trend_strength > self.params['trend_strength_min']  # Strong trend
            
            if (trend_aligned and macd_bullish and rsi_good and 
                volume_good and momentum_good and trend_strong):
                
                signal = "BUY CALL"
                # OPTIMIZATION: Comprehensive confidence calculation
                trend_score = min(25, trend_strength * 2)
                macd_score = min(20, abs(macd_histogram) * 10)
                rsi_score = min(15, abs(rsi - 50) / 2)
                volume_score = min(15, (volume_ratio - 1.0) * 10)
                momentum_score = min(15, price_momentum)
                supertrend_score = min(10, (price - supertrend) / supertrend * 100)
                
                confidence_score = 70 + trend_score + macd_score + rsi_score + volume_score + momentum_score + supertrend_score
                
        elif supertrend_direction < 0 and price < supertrend:  # Bearish SuperTrend
            # Multi-factor bearish validation
            trend_aligned = price < ema_21 < ema_50 < ema_200  # Strong downtrend
            macd_bearish = macd < macd_signal and macd_histogram < 0  # MACD bearish
            rsi_good = 30 < rsi < 60  # RSI not oversold
            volume_good = volume_ratio > self.params['volume_threshold']  # Strong volume
            momentum_good = price_momentum < -self.params['momentum_threshold']  # Negative momentum
            trend_strong = trend_strength > self.params['trend_strength_min']  # Strong trend
            
            if (trend_aligned and macd_bearish and rsi_good and 
                volume_good and momentum_good and trend_strong):
                
                signal = "BUY PUT"
                # OPTIMIZATION: Comprehensive confidence calculation
                trend_score = min(25, trend_strength * 2)
                macd_score = min(20, abs(macd_histogram) * 10)
                rsi_score = min(15, abs(rsi - 50) / 2)
                volume_score = min(15, (volume_ratio - 1.0) * 10)
                momentum_score = min(15, abs(price_momentum))
                supertrend_score = min(10, (supertrend - price) / supertrend * 100)
                
                confidence_score = 70 + trend_score + macd_score + rsi_score + volume_score + momentum_score + supertrend_score
        
        # OPTIMIZATION: Very high minimum confidence threshold for quality
        if confidence_score < 85:  # High threshold for hybrid strategy
            return None
            
        # Determine confidence level
        if confidence_score >= 95:
            confidence = "Very High"
        elif confidence_score >= 90:
            confidence = "High"
        elif confidence_score >= 85:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # OPTIMIZATION: Aggressive risk-reward ratios for high-confidence signals
        if confidence_score >= 95:
            stop_loss = int(round(0.8 * atr))  # Very tight stop loss
            target1 = int(round(2.4 * atr))    # 3:1 R:R
            target2 = int(round(4.0 * atr))    # 5:1 R:R
            target3 = int(round(5.6 * atr))    # 7:1 R:R
        elif confidence_score >= 90:
            stop_loss = int(round(1.0 * atr))
            target1 = int(round(2.5 * atr))    # 2.5:1 R:R
            target2 = int(round(4.0 * atr))    # 4:1 R:R
            target3 = int(round(5.5 * atr))    # 5.5:1 R:R
        else:  # 85-89
            stop_loss = int(round(1.2 * atr))
            target1 = int(round(2.4 * atr))    # 2:1 R:R
            target2 = int(round(3.6 * atr))    # 3:1 R:R
            target3 = int(round(4.8 * atr))    # 4:1 R:R
        
        # OPTIMIZATION: Full position sizing for high-confidence signals
        position_multiplier = 1.0  # Full position for hybrid strategy
        
        # Calculate performance if we have future data
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        
        if signal != "NO TRADE" and future_data is not None and not future_data.empty:
            # Check future prices to see if targets or stop loss were hit
            if signal == "BUY CALL":
                max_future_price = future_data['high'].max()
                min_future_price = future_data['low'].min()
                
                # Check if stop loss was hit
                if min_future_price <= (price - stop_loss):
                    outcome = "Loss"
                    pnl = -stop_loss * position_multiplier
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {price - stop_loss}"
                else:
                    outcome = "Win"
                    # Check which targets were hit
                    if max_future_price >= (price + target1):
                        targets_hit += 1
                        pnl += target1 * position_multiplier
                    if max_future_price >= (price + target2):
                        targets_hit += 1
                        pnl += (target2 - target1) * position_multiplier
                    if max_future_price >= (price + target3):
                        targets_hit += 1
                        pnl += (target3 - target2) * position_multiplier
                    
            elif signal == "BUY PUT":
                max_future_price = future_data['high'].max()
                min_future_price = future_data['low'].min()
                
                # Check if stop loss was hit
                if max_future_price >= (price + stop_loss):
                    outcome = "Loss"
                    pnl = -stop_loss * position_multiplier
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {price + stop_loss}"
                else:
                    outcome = "Win"
                    # Check which targets were hit
                    if min_future_price <= (price - target1):
                        targets_hit += 1
                        pnl += target1 * position_multiplier
                    if min_future_price <= (price - target2):
                        targets_hit += 1
                        pnl += (target2 - target1) * position_multiplier
                    if min_future_price <= (price - target3):
                        targets_hit += 1
                        pnl += (target3 - target2) * position_multiplier
        
        # Build comprehensive reasoning string
        price_reason = f"Hybrid: SuperTrend {supertrend_direction > 0 and 'BULLISH' or 'BEARISH'}"
        price_reason += f", Trend: {'STRONG UP' if price > ema_21 > ema_50 > ema_200 else 'STRONG DOWN' if price < ema_21 < ema_50 < ema_200 else 'MIXED'}"
        price_reason += f", MACD: {'BULLISH' if macd > macd_signal else 'BEARISH'} ({macd_histogram:.2f})"
        price_reason += f", RSI: {rsi:.1f}, Volume: {volume_ratio:.1f}x"
        price_reason += f", Momentum: {price_momentum:.1f}%, Trend Strength: {trend_strength:.1f}%"
        price_reason += f", Confidence: {confidence_score}"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "price": price,
            "stop_loss": stop_loss,
            "target": target1,
            "target2": target2,
            "target3": target3,
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "reasoning": price_reason
        } 