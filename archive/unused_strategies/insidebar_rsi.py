"""
InsidebarRsi strategy.
Trading strategy implementation.
"""
import pandas as pd
from typing import Dict, Any, Optional
from src.core.strategy import Strategy
from src.core.indicators import indicators
from src.models.database import db
from datetime import datetime, timedelta

class InsidebarRsi(Strategy):
    """Trading strategy implementation for Inside Bar with RSI confirmation."""
    
    def __init__(self, params: Dict[str, Any] = None, timeframe_data: Dict[str, pd.DataFrame] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters
            timeframe_data: Dictionary of timeframes with their respective data
        """
        default_params = {
            'rsi_overbought': 60,
            'rsi_extreme_overbought': 70,
            'rsi_oversold': 40,
            'rsi_extreme_oversold': 30,
            'volatility_threshold': 1.5,  # ATR multiplier for volatility check
            'momentum_threshold': 0.5,    # Price momentum threshold
            'volume_factor': 1.2          # Volume confirmation factor
        }
        
        # Use provided params or defaults
        if params:
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        else:
            params = default_params
            
        super().__init__("insidebar_rsi", params)
        self.timeframe_data = timeframe_data or {}
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        """
        # Identify inside bars
        data['prev_high'] = data['high'].shift(1)
        data['prev_low'] = data['low'].shift(1)
        data['is_inside'] = (data['high'] < data['prev_high']) & (data['low'] > data['prev_low'])
        
        # Calculate RSI levels based on thresholds
        def get_rsi_level(rsi):
            if rsi < self.params['rsi_extreme_oversold']:
                return "Extreme Oversold"
            elif rsi < self.params['rsi_oversold']:
                return "Oversold"
            elif rsi < self.params['rsi_overbought']:
                return "Neutral"
            elif rsi < self.params['rsi_extreme_overbought']:
                return "Overbought"
            else:
                return "Extreme Overbought"
        
        # Apply the function to create an RSI level column
        data['rsi_level'] = data['rsi'].apply(get_rsi_level)
        
        # Advanced confidence indicators
        data['price_range'] = data['high'] - data['low']
        data['body_size'] = abs(data['close'] - data['open'])
        data['body_ratio'] = data['body_size'] / data['price_range'].replace(0, float('nan'))
        
        # Volume analysis
        data['avg_volume_5'] = data['volume'].rolling(5).mean()
        data['volume_ratio'] = data['volume'] / data['avg_volume_5'].replace(0, float('nan'))
        
        # Price momentum
        data['price_momentum'] = data['close'].pct_change(3) * 100
        
        # Volatility measure
        data['volatility_ratio'] = data['atr'] / data['close'] * 100
        
        return data
    
    def calculate_confidence_score(self, candle: pd.Series) -> tuple:
        """Calculate confidence score based on multiple market conditions.
        
        Returns:
            tuple: (confidence_level, confidence_score, detailed_reasons)
        """
        confidence_score = 0
        reasons = []
        
        # 1. RSI Strength (0-30 points)
        rsi = candle['rsi']
        if rsi <= 20:  # Extreme oversold
            confidence_score += 30
            reasons.append(f"Extreme RSI oversold ({rsi:.1f})")
        elif rsi <= 30:  # Strong oversold
            confidence_score += 25
            reasons.append(f"Strong RSI oversold ({rsi:.1f})")
        elif rsi <= 40:  # Mild oversold
            confidence_score += 15
            reasons.append(f"Mild RSI oversold ({rsi:.1f})")
        elif rsi >= 80:  # Extreme overbought
            confidence_score += 30
            reasons.append(f"Extreme RSI overbought ({rsi:.1f})")
        elif rsi >= 70:  # Strong overbought
            confidence_score += 25
            reasons.append(f"Strong RSI overbought ({rsi:.1f})")
        elif rsi >= 60:  # Mild overbought
            confidence_score += 15
            reasons.append(f"Mild RSI overbought ({rsi:.1f})")
        
        # 2. Inside Bar Quality (0-20 points)
        if candle['is_inside']:
            prev_range = candle['prev_high'] - candle['prev_low']
            current_range = candle['high'] - candle['low']
            compression_ratio = current_range / prev_range if prev_range > 0 else 0
            
            if compression_ratio <= 0.5:  # High compression
                confidence_score += 20
                reasons.append(f"High compression inside bar ({compression_ratio:.2f})")
            elif compression_ratio <= 0.7:  # Good compression
                confidence_score += 15
                reasons.append(f"Good compression inside bar ({compression_ratio:.2f})")
            else:  # Low compression
                confidence_score += 10
                reasons.append(f"Low compression inside bar ({compression_ratio:.2f})")
        
        # 3. Volume Confirmation (0-15 points)
        volume_ratio = candle.get('volume_ratio', 1)
        if volume_ratio >= 2.0:  # High volume
            confidence_score += 15
            reasons.append(f"High volume confirmation ({volume_ratio:.1f}x)")
        elif volume_ratio >= 1.5:  # Above average volume
            confidence_score += 10
            reasons.append(f"Above average volume ({volume_ratio:.1f}x)")
        elif volume_ratio >= 1.2:  # Decent volume
            confidence_score += 5
            reasons.append(f"Decent volume ({volume_ratio:.1f}x)")
        
        # 4. Market Volatility (0-15 points)
        volatility = candle.get('volatility_ratio', 1)
        atr = candle.get('atr', 50)
        if volatility >= 2.0 and atr > 30:  # High volatility, good range
            confidence_score += 15
            reasons.append(f"High volatility environment ({volatility:.1f}%)")
        elif volatility >= 1.5 and atr > 20:  # Moderate volatility
            confidence_score += 10
            reasons.append(f"Moderate volatility ({volatility:.1f}%)")
        elif atr > 15:  # Decent range
            confidence_score += 5
            reasons.append(f"Decent price range (ATR: {atr:.1f})")
        
        # 5. Price Momentum Alignment (0-10 points)
        momentum = candle.get('price_momentum', 0)
        if abs(momentum) >= 1.0:  # Strong momentum
            confidence_score += 10
            direction = "bullish" if momentum > 0 else "bearish"
            reasons.append(f"Strong {direction} momentum ({momentum:.1f}%)")
        elif abs(momentum) >= 0.5:  # Moderate momentum
            confidence_score += 5
            direction = "bullish" if momentum > 0 else "bearish"
            reasons.append(f"Moderate {direction} momentum ({momentum:.1f}%)")
        
        # 6. Candle Quality (0-10 points)
        body_ratio = candle.get('body_ratio', 0.5)
        if body_ratio >= 0.7:  # Strong directional candle
            confidence_score += 10
            reasons.append(f"Strong directional candle ({body_ratio:.2f})")
        elif body_ratio >= 0.5:  # Decent body
            confidence_score += 5
            reasons.append(f"Decent candle body ({body_ratio:.2f})")
        
        # Determine confidence level
        if confidence_score >= 70:
            confidence_level = "Very High"
        elif confidence_score >= 50:
            confidence_level = "High"
        elif confidence_score >= 30:
            confidence_level = "Medium"
        elif confidence_score >= 15:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        return confidence_level, confidence_score, reasons
    
    def safe_signal_time(self, val):
        return val if isinstance(val, (pd.Timestamp, datetime)) else datetime.now()
    
    def to_ist_str(self, val):
        if isinstance(val, (pd.Timestamp, datetime)):
            ist_dt = val + timedelta(hours=5, minutes=30)
            return ist_dt.strftime("%Y-%m-%d %H:%M:%S")
        return None
    
    def analyze(self, candle: pd.Series, index: int, df: pd.DataFrame, future_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Analyze data and generate trading signals with enhanced quality filters."""
        if index < 50 or future_data is None or future_data.empty:
            return None
            
        # Get indicator values
        is_inside = candle.get('is_inside', False)
        rsi = candle.get('rsi', 50)
        rsi_level = candle.get('rsi_level', 'Neutral')
        volume_ratio = candle.get('volume_ratio', 1.0)
        body_ratio = candle.get('body_ratio', 0.5)
        price_momentum = candle.get('price_momentum', 0)
        volatility_ratio = candle.get('volatility_ratio', 1.0)
        ema_21 = candle.get('ema_21', 0)
        ema_50 = candle.get('ema_50', 0)
        macd = candle.get('macd', 0)
        macd_signal = candle.get('macd_signal', 0)
        atr = candle.get('atr', candle['close'] * 0.01)
        price = candle['close']
        
        # OPTIMIZATION: Enhanced signal quality filters
        signal = "NO TRADE"
        confidence_score = 0
        
        # Check for valid indicator values
        if not is_inside or ema_21 <= 0 or ema_50 <= 0:
            return None
            
        # OPTIMIZATION: Enhanced inside bar signal conditions
        if rsi_level in ['Extreme Oversold', 'Oversold']:  # Bullish setup
            # Additional bullish filters for better win rate
            if (price > ema_21 and ema_21 > ema_50 and  # Trend alignment
                macd > macd_signal and                 # MACD bullish
                volume_ratio > 1.2 and                 # Strong volume
                body_ratio > 0.4 and                   # Decent body size
                price_momentum > 0.5):                 # Positive momentum
                
                signal = "BUY CALL"
                # OPTIMIZATION: Better confidence calculation
                rsi_strength = min(25, (40 - rsi) * 2)  # RSI oversold strength
                trend_strength = min(20, (price - ema_21) / ema_21 * 100)
                volume_strength = min(15, (volume_ratio - 1.0) * 10)
                momentum_strength = min(10, price_momentum)
                body_strength = min(10, body_ratio * 20)
                
                confidence_score = 60 + rsi_strength + trend_strength + volume_strength + momentum_strength + body_strength
                
        elif rsi_level in ['Extreme Overbought', 'Overbought']:  # Bearish setup
            # Additional bearish filters for better win rate
            if (price < ema_21 and ema_21 < ema_50 and  # Trend alignment
                macd < macd_signal and                 # MACD bearish
                volume_ratio > 1.2 and                 # Strong volume
                body_ratio > 0.4 and                   # Decent body size
                price_momentum < -0.5):                # Negative momentum
                
                signal = "BUY PUT"
                # OPTIMIZATION: Better confidence calculation
                rsi_strength = min(25, (rsi - 60) * 2)  # RSI overbought strength
                trend_strength = min(20, (ema_21 - price) / ema_21 * 100)
                volume_strength = min(15, (volume_ratio - 1.0) * 10)
                momentum_strength = min(10, abs(price_momentum))
                body_strength = min(10, body_ratio * 20)
                
                confidence_score = 60 + rsi_strength + trend_strength + volume_strength + momentum_strength + body_strength
        
        # OPTIMIZATION: Higher minimum confidence threshold for better quality
        if confidence_score < 75:  # Increased from implicit lower threshold
            return None
            
        # Determine confidence level
        if confidence_score >= 90:
            confidence = "Very High"
        elif confidence_score >= 80:
            confidence = "High"
        elif confidence_score >= 75:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # OPTIMIZATION: Improved risk-reward ratios based on confidence
        if confidence_score >= 85:
            stop_loss = int(round(1.0 * atr))  # Tighter stop loss
            target1 = int(round(2.5 * atr))    # 2.5:1 R:R
            target2 = int(round(4.0 * atr))    # 4:1 R:R
            target3 = int(round(5.5 * atr))    # 5.5:1 R:R
        elif confidence_score >= 80:
            stop_loss = int(round(1.2 * atr))
            target1 = int(round(3.0 * atr))    # 2.5:1 R:R
            target2 = int(round(4.5 * atr))    # 3.75:1 R:R
            target3 = int(round(6.0 * atr))    # 5:1 R:R
        else:  # 75-79
            stop_loss = int(round(1.5 * atr))
            target1 = int(round(3.5 * atr))    # 2.33:1 R:R
            target2 = int(round(5.0 * atr))    # 3.33:1 R:R
            target3 = int(round(6.5 * atr))    # 4.33:1 R:R
        
        # OPTIMIZATION: Position sizing based on confidence
        position_multiplier = 0.7 if confidence_score >= 80 else 0.5
        
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
        
        # Build reasoning string
        price_reason = f"Inside Bar: {rsi_level} RSI ({rsi:.1f})"
        price_reason += f", Trend: {'BULLISH' if price > ema_21 else 'BEARISH'}"
        price_reason += f", EMA21: {ema_21:.1f} vs EMA50: {ema_50:.1f}"
        price_reason += f", MACD: {macd:.2f} vs {macd_signal:.2f}"
        price_reason += f", Volume: {volume_ratio:.1f}x"
        price_reason += f", Body: {body_ratio:.2f}, Momentum: {price_momentum:.1f}%"
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
