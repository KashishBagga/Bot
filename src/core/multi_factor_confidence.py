#!/usr/bin/env python3
"""
Multi-Factor Confidence Scoring System
Comprehensive evaluation of market conditions for optimal signal quality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class MultiFactorConfidence:
    """
    Advanced multi-factor confidence scoring system
    Evaluates 6 key factors: Trend, Momentum, Volatility, Volume, Structure, Safety
    """
    
    def __init__(self):
        self.factors = {
            'trend': {'weight': 0.25, 'max_score': 25},
            'momentum': {'weight': 0.20, 'max_score': 20},
            'volatility': {'weight': 0.15, 'max_score': 15},
            'volume': {'weight': 0.15, 'max_score': 15},
            'structure': {'weight': 0.15, 'max_score': 15},
            'safety': {'weight': 0.10, 'max_score': 10}
        }
        
    def calculate_confidence(self, candle: Dict, signal_type: str, 
                           timeframe_data: Dict = None) -> Dict:
        """
        Calculate comprehensive confidence score based on multiple factors
        
        Args:
            candle: Current candle data with indicators
            signal_type: 'BUY' or 'SELL'
            timeframe_data: Multi-timeframe data for confirmation
            
        Returns:
            Dict with confidence score, factors, and reasoning
        """
        
        scores = {}
        reasons = {}
        
        # 1. TREND FACTOR (25 points)
        trend_score, trend_reasons = self._evaluate_trend(candle, signal_type, timeframe_data)
        scores['trend'] = trend_score
        reasons['trend'] = trend_reasons
        
        # 2. MOMENTUM FACTOR (20 points)
        momentum_score, momentum_reasons = self._evaluate_momentum(candle, signal_type)
        scores['momentum'] = momentum_score
        reasons['momentum'] = momentum_reasons
        
        # 3. VOLATILITY FACTOR (15 points)
        volatility_score, volatility_reasons = self._evaluate_volatility(candle, signal_type)
        scores['volatility'] = volatility_score
        reasons['volatility'] = volatility_reasons
        
        # 4. VOLUME FACTOR (15 points)
        volume_score, volume_reasons = self._evaluate_volume(candle, signal_type)
        scores['volume'] = volume_score
        reasons['volume'] = volume_reasons
        
        # 5. STRUCTURE FACTOR (15 points)
        structure_score, structure_reasons = self._evaluate_structure(candle, signal_type)
        scores['structure'] = structure_score
        reasons['structure'] = structure_reasons
        
        # 6. SAFETY FACTOR (10 points)
        safety_score, safety_reasons = self._evaluate_safety(candle, signal_type)
        scores['safety'] = safety_score
        reasons['safety'] = safety_reasons
        
        # Calculate weighted total score
        total_score = sum(scores.values())
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(total_score)
        
        return {
            'total_score': total_score,
            'confidence_level': confidence_level,
            'factors': scores,
            'reasons': reasons,
            'max_possible': 100
        }
    
    def _evaluate_trend(self, candle: Dict, signal_type: str, 
                       timeframe_data: Dict = None) -> Tuple[int, List[str]]:
        """Evaluate trend strength and alignment (25 points)"""
        score = 0
        reasons = []
        
        # SuperTrend direction (0-10 points)
        supertrend_direction = candle.get('supertrend_direction', 0)
        if signal_type == 'BUY' and supertrend_direction > 0:
            score += 10
            reasons.append("SuperTrend bullish")
        elif signal_type == 'SELL' and supertrend_direction < 0:
            score += 10
            reasons.append("SuperTrend bearish")
        elif supertrend_direction == 0:
            score += 5
            reasons.append("SuperTrend neutral")
        
        # EMA alignment (0-8 points)
        ema_9 = candle.get('ema_9', 0)
        ema_21 = candle.get('ema_21', 0)
        ema_50 = candle.get('ema_50', 0)
        close = candle.get('close', 0)
        
        if ema_9 and ema_21 and ema_50 and close:
            # Check EMA alignment
            if signal_type == 'BUY':
                if close > ema_9 > ema_21 > ema_50:  # Perfect bullish alignment
                    score += 8
                    reasons.append("Perfect bullish EMA alignment")
                elif close > ema_9 and ema_9 > ema_21:  # Good bullish alignment
                    score += 6
                    reasons.append("Good bullish EMA alignment")
                elif close > ema_21:  # Basic bullish alignment
                    score += 4
                    reasons.append("Basic bullish EMA alignment")
            else:  # SELL
                if close < ema_9 < ema_21 < ema_50:  # Perfect bearish alignment
                    score += 8
                    reasons.append("Perfect bearish EMA alignment")
                elif close < ema_9 and ema_9 < ema_21:  # Good bearish alignment
                    score += 6
                    reasons.append("Good bearish EMA alignment")
                elif close < ema_21:  # Basic bearish alignment
                    score += 4
                    reasons.append("Basic bearish EMA alignment")
        
        # Multi-timeframe trend consensus (0-7 points)
        if timeframe_data:
            consensus_score = self._calculate_trend_consensus(timeframe_data, signal_type)
            score += consensus_score
            if consensus_score >= 5:
                reasons.append("Multi-timeframe trend consensus")
        
        return min(score, 25), reasons
    
    def _evaluate_momentum(self, candle: Dict, signal_type: str) -> Tuple[int, List[str]]:
        """Evaluate momentum indicators (20 points)"""
        score = 0
        reasons = []
        
        # RSI momentum (0-8 points)
        rsi = candle.get('rsi', 50)
        if signal_type == 'BUY':
            if 30 <= rsi <= 45:  # Oversold bounce
                score += 8
                reasons.append(f"RSI oversold bounce ({rsi:.1f})")
            elif 45 < rsi <= 60:  # Good momentum
                score += 6
                reasons.append(f"RSI good momentum ({rsi:.1f})")
            elif 60 < rsi <= 70:  # Acceptable momentum
                score += 4
                reasons.append(f"RSI acceptable ({rsi:.1f})")
        else:  # SELL
            if 55 <= rsi <= 70:  # Overbought reversal
                score += 8
                reasons.append(f"RSI overbought reversal ({rsi:.1f})")
            elif 40 <= rsi < 55:  # Good momentum
                score += 6
                reasons.append(f"RSI good momentum ({rsi:.1f})")
            elif 30 <= rsi < 40:  # Acceptable momentum
                score += 4
                reasons.append(f"RSI acceptable ({rsi:.1f})")
        
        # MACD momentum (0-7 points)
        macd = candle.get('macd', 0)
        macd_signal = candle.get('macd_signal', 0)
        macd_histogram = candle.get('macd_histogram', 0)
        
        if signal_type == 'BUY':
            if macd > macd_signal and macd > 0 and macd_histogram > 0:
                score += 7
                reasons.append("MACD bullish momentum")
            elif macd > macd_signal:
                score += 5
                reasons.append("MACD bullish crossover")
            elif macd > 0:
                score += 3
                reasons.append("MACD above zero")
        else:  # SELL
            if macd < macd_signal and macd < 0 and macd_histogram < 0:
                score += 7
                reasons.append("MACD bearish momentum")
            elif macd < macd_signal:
                score += 5
                reasons.append("MACD bearish crossover")
            elif macd < 0:
                score += 3
                reasons.append("MACD below zero")
        
        # Price momentum (0-5 points)
        price_change = candle.get('price_change_pct', 0)
        if abs(price_change) > 0.5:  # Strong momentum
            if (signal_type == 'BUY' and price_change > 0) or (signal_type == 'SELL' and price_change < 0):
                score += 5
                reasons.append(f"Strong price momentum ({price_change:.2f}%)")
        elif abs(price_change) > 0.2:  # Moderate momentum
            if (signal_type == 'BUY' and price_change > 0) or (signal_type == 'SELL' and price_change < 0):
                score += 3
                reasons.append(f"Moderate price momentum ({price_change:.2f}%)")
        
        return min(score, 20), reasons
    
    def _evaluate_volatility(self, candle: Dict, signal_type: str) -> Tuple[int, List[str]]:
        """Evaluate volatility conditions (15 points)"""
        score = 0
        reasons = []
        
        # ATR-based volatility (0-8 points)
        atr = candle.get('atr', 0)
        atr_percentile = candle.get('atr_percentile', 50)
        close = candle.get('close', 1)
        
        if atr and close:
            atr_pct = (atr / close) * 100
            
            # Optimal volatility for trading
            if 0.3 <= atr_pct <= 0.8:  # Good volatility range
                score += 8
                reasons.append(f"Optimal volatility (ATR: {atr_pct:.2f}%)")
            elif 0.2 <= atr_pct <= 1.0:  # Acceptable volatility
                score += 6
                reasons.append(f"Acceptable volatility (ATR: {atr_pct:.2f}%)")
            elif 0.1 <= atr_pct <= 1.5:  # Marginal volatility
                score += 4
                reasons.append(f"Marginal volatility (ATR: {atr_pct:.2f}%)")
            else:
                score += 2
                reasons.append(f"Low volatility (ATR: {atr_pct:.2f}%)")
        
        # Bollinger Bands position (0-7 points)
        bb_upper = candle.get('bb_upper', 0)
        bb_lower = candle.get('bb_lower', 0)
        bb_middle = candle.get('bb_middle', 0)
        
        if bb_upper and bb_lower and bb_middle:
            bb_position = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            if signal_type == 'BUY':
                if bb_position <= 0.2:  # Near lower band (oversold)
                    score += 7
                    reasons.append("Near BB lower band (oversold)")
                elif bb_position <= 0.4:  # Below middle
                    score += 5
                    reasons.append("Below BB middle")
                elif bb_position <= 0.6:  # Near middle
                    score += 3
                    reasons.append("Near BB middle")
            else:  # SELL
                if bb_position >= 0.8:  # Near upper band (overbought)
                    score += 7
                    reasons.append("Near BB upper band (overbought)")
                elif bb_position >= 0.6:  # Above middle
                    score += 5
                    reasons.append("Above BB middle")
                elif bb_position >= 0.4:  # Near middle
                    score += 3
                    reasons.append("Near BB middle")
        
        return min(score, 15), reasons
    
    def _evaluate_volume(self, candle: Dict, signal_type: str) -> Tuple[int, List[str]]:
        """Evaluate volume conditions (15 points)"""
        score = 0
        reasons = []
        
        # Volume ratio (0-8 points)
        volume_ratio = candle.get('volume_ratio', 1.0)
        
        if volume_ratio >= 2.0:  # Very high volume
            score += 8
            reasons.append(f"Very high volume ({volume_ratio:.1f}x)")
        elif volume_ratio >= 1.5:  # High volume
            score += 6
            reasons.append(f"High volume ({volume_ratio:.1f}x)")
        elif volume_ratio >= 1.2:  # Above average volume
            score += 4
            reasons.append(f"Above average volume ({volume_ratio:.1f}x)")
        elif volume_ratio >= 0.8:  # Normal volume
            score += 2
            reasons.append(f"Normal volume ({volume_ratio:.1f}x)")
        
        # Volume trend (0-7 points)
        volume_sma = candle.get('volume_sma', 0)
        current_volume = candle.get('volume', 0)
        
        if volume_sma and current_volume:
            if current_volume > volume_sma * 1.5:  # Strong volume trend
                score += 7
                reasons.append("Strong volume trend")
            elif current_volume > volume_sma * 1.2:  # Good volume trend
                score += 5
                reasons.append("Good volume trend")
            elif current_volume > volume_sma:  # Positive volume trend
                score += 3
                reasons.append("Positive volume trend")
        
        return min(score, 15), reasons
    
    def _evaluate_structure(self, candle: Dict, signal_type: str) -> Tuple[int, List[str]]:
        """Evaluate price structure and patterns (15 points)"""
        score = 0
        reasons = []
        
        # Candle structure (0-8 points)
        open_price = candle.get('open', 0)
        high = candle.get('high', 0)
        low = candle.get('low', 0)
        close = candle.get('close', 0)
        
        if all([open_price, high, low, close]):
            body_size = abs(close - open_price)
            total_range = high - low
            body_ratio = body_size / total_range if total_range > 0 else 0
            
            if signal_type == 'BUY':
                if close > open_price:  # Bullish candle
                    if body_ratio >= 0.7:  # Strong bullish
                        score += 8
                        reasons.append("Strong bullish candle")
                    elif body_ratio >= 0.5:  # Good bullish
                        score += 6
                        reasons.append("Good bullish candle")
                    elif body_ratio >= 0.3:  # Weak bullish
                        score += 4
                        reasons.append("Weak bullish candle")
            else:  # SELL
                if close < open_price:  # Bearish candle
                    if body_ratio >= 0.7:  # Strong bearish
                        score += 8
                        reasons.append("Strong bearish candle")
                    elif body_ratio >= 0.5:  # Good bearish
                        score += 6
                        reasons.append("Good bearish candle")
                    elif body_ratio >= 0.3:  # Weak bearish
                        score += 4
                        reasons.append("Weak bearish candle")
        
        # Support/Resistance levels (0-7 points)
        price_position = candle.get('price_position', 0.5)
        
        if signal_type == 'BUY':
            if price_position <= 0.2:  # Near support
                score += 7
                reasons.append("Near support level")
            elif price_position <= 0.4:  # Below middle
                score += 5
                reasons.append("Below middle range")
        else:  # SELL
            if price_position >= 0.8:  # Near resistance
                score += 7
                reasons.append("Near resistance level")
            elif price_position >= 0.6:  # Above middle
                score += 5
                reasons.append("Above middle range")
        
        return min(score, 15), reasons
    
    def _evaluate_safety(self, candle: Dict, signal_type: str) -> Tuple[int, List[str]]:
        """Evaluate safety and risk factors (10 points)"""
        score = 0
        reasons = []
        
        # Market regime safety (0-5 points)
        vix_proxy = candle.get('vix_proxy', 0)
        
        if vix_proxy:
            if 0.015 <= vix_proxy <= 0.035:  # Normal volatility regime
                score += 5
                reasons.append("Normal volatility regime")
            elif 0.010 <= vix_proxy <= 0.050:  # Acceptable regime
                score += 3
                reasons.append("Acceptable volatility regime")
            else:
                score += 1
                reasons.append("High volatility regime")
        
        # Trend strength safety (0-5 points)
        adx = candle.get('adx', 0)
        
        if adx:
            if adx >= 25:  # Strong trend
                score += 5
                reasons.append("Strong trend (ADX >= 25)")
            elif adx >= 20:  # Moderate trend
                score += 3
                reasons.append("Moderate trend (ADX >= 20)")
            elif adx >= 15:  # Weak trend
                score += 1
                reasons.append("Weak trend (ADX >= 15)")
        
        return min(score, 10), reasons
    
    def _calculate_trend_consensus(self, timeframe_data: Dict, signal_type: str) -> int:
        """Calculate multi-timeframe trend consensus"""
        consensus_count = 0
        total_timeframes = 0
        
        for timeframe, data in timeframe_data.items():
            if data is not None and not data.empty:
                total_timeframes += 1
                # Check if trend aligns with signal
                # This is a simplified check - can be enhanced
                if signal_type == 'BUY':
                    # Check for bullish trend indicators
                    pass
                else:
                    # Check for bearish trend indicators
                    pass
        
        if total_timeframes == 0:
            return 0
        
        consensus_ratio = consensus_count / total_timeframes
        
        if consensus_ratio >= 0.8:
            return 7
        elif consensus_ratio >= 0.6:
            return 5
        elif consensus_ratio >= 0.4:
            return 3
        else:
            return 1
    
    def _get_confidence_level(self, score: int) -> str:
        """Convert score to confidence level"""
        if score >= 85:
            return "Very High"
        elif score >= 70:
            return "High"
        elif score >= 55:
            return "Medium"
        elif score >= 40:
            return "Low"
        else:
            return "Very Low"
    
    def get_optimization_recommendations(self, scores: Dict) -> List[str]:
        """Get recommendations for improving confidence"""
        recommendations = []
        
        if scores.get('trend', 0) < 15:
            recommendations.append("Improve trend alignment - check EMA and SuperTrend")
        
        if scores.get('momentum', 0) < 12:
            recommendations.append("Enhance momentum - look for better RSI/MACD conditions")
        
        if scores.get('volatility', 0) < 10:
            recommendations.append("Wait for better volatility conditions")
        
        if scores.get('volume', 0) < 10:
            recommendations.append("Seek higher volume confirmation")
        
        if scores.get('structure', 0) < 10:
            recommendations.append("Look for better price structure")
        
        if scores.get('safety', 0) < 6:
            recommendations.append("Consider market regime safety")
        
        return recommendations 