"""
InsidebarRsi strategy.
Trading strategy implementation.
"""
import pandas as pd
from typing import Dict, Any
from src.core.strategy import Strategy
from src.core.indicators import indicators
from src.models.database import db
from datetime import datetime, timedelta

class InsidebarRsi(Strategy):
    """Trading strategy implementation for Inside Bar with RSI confirmation."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the strategy.
        
        Args:
            params: Strategy parameters
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
    
    def analyze(self, data: pd.DataFrame, index_name: str = None, future_data = None) -> Dict[str, Any]:
        """Analyze data and generate trading signals.
        
        Args:
            data: Market data with indicators
            index_name: Name of the index or symbol being analyzed
            future_data: Future candles for performance calculation
            
        Returns:
            Dict[str, Any]: Signal data
        """
        # Calculate indicators if they haven't been calculated yet
        if 'is_inside' not in data.columns:
            data = self.calculate_indicators(data)
        
        # Get the latest candle
        candle = data.iloc[-1]
        
        # Set default values
        signal = "NO TRADE"
        trade_type = "Intraday"
        rsi_reason = price_reason = ""
        
        # Performance tracking variables
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        
        # Check if we have an inside bar - mandatory requirement
        if not candle['is_inside']:
            return {
                "signal": "NO TRADE",
                "price": candle['close'],
                "rsi": candle['rsi'],
                "rsi_level": candle['rsi_level'],
                "ema_20": candle['ema'] if 'ema' in candle else 0,
                "atr": candle['atr'],
                "confidence": "Very Low",
                "confidence_score": 0,
                "rsi_reason": "No inside bar pattern detected",
                "price_reason": "Inside bar formation required for signal",
                "trade_type": trade_type,
                "stop_loss": 0,
                "target": 0,
                "target2": 0,
                "target3": 0,
                "outcome": "No Trade",
                "pnl": 0,
                "targets_hit": 0,
                "stoploss_count": 0,
                "failure_reason": "",
                "exit_time": None
            }
        
        # Calculate confidence score based on market conditions
        confidence_level, confidence_score, confidence_reasons = self.calculate_confidence_score(candle)
        
        # FIXED LOGIC: Correct RSI interpretation with enhanced confidence
        # When RSI is oversold (< 40), expect price to go UP -> BUY CALL
        # When RSI is overbought (> 60), expect price to go DOWN -> BUY PUT
        rsi_level = candle['rsi_level']
        
        if candle['rsi'] < self.params['rsi_oversold']:
            signal = "BUY CALL"
            rsi_reason = f"RSI {candle['rsi']:.2f} < {self.params['rsi_oversold']} (Oversold - expect bounce up)"
            price_reason = "Inside bar pattern + RSI oversold reversal"
        elif candle['rsi'] > self.params['rsi_overbought']:
            signal = "BUY PUT"
            rsi_reason = f"RSI {candle['rsi']:.2f} > {self.params['rsi_overbought']} (Overbought - expect pullback down)"
            price_reason = "Inside bar pattern + RSI overbought reversal"
        
        # Enhanced filtering: Only trade with Medium+ confidence (score >= 30)
        if signal != "NO TRADE" and confidence_score < 30:
            signal = "NO TRADE"
            rsi_reason += f" (Filtered: Confidence score {confidence_score} < 30)"
            confidence_level = "Very Low"
        
        # Additional quality checks for high confidence trades
        if signal != "NO TRADE" and confidence_score >= 50:
            # Check for divergence or momentum alignment
            momentum = candle.get('price_momentum', 0)
            if signal == "BUY CALL" and momentum < -1.0:  # Bearish momentum conflicts with bullish signal
                confidence_score -= 15
                confidence_level = "Medium" if confidence_score >= 30 else "Low"
                rsi_reason += " (Momentum divergence detected)"
            elif signal == "BUY PUT" and momentum > 1.0:  # Bullish momentum conflicts with bearish signal
                confidence_score -= 15
                confidence_level = "Medium" if confidence_score >= 30 else "Low"
                rsi_reason += " (Momentum divergence detected)"
        
        # Dynamic risk management based on confidence and market conditions
        atr = candle['atr']
        volatility_ratio = candle.get('volatility_ratio', 1)
        
        # Adjust stop loss based on confidence and volatility
        if confidence_score >= 70:  # Very high confidence
            stop_loss_multiplier = 0.6  # Tighter stop
        elif confidence_score >= 50:  # High confidence
            stop_loss_multiplier = 0.7
        elif confidence_score >= 30:  # Medium confidence
            stop_loss_multiplier = 0.8
        else:  # Low confidence
            stop_loss_multiplier = 1.0
        
        # Adjust for volatility
        if volatility_ratio >= 2.0:  # High volatility
            stop_loss_multiplier *= 1.2
        elif volatility_ratio <= 1.0:  # Low volatility
            stop_loss_multiplier *= 0.9
        
        stop_loss = int(round(stop_loss_multiplier * atr))
        
        # Dynamic targets based on confidence
        if confidence_score >= 70:
            target1 = int(round(1.5 * atr))  # Aggressive targets for high confidence
            target2 = int(round(2.5 * atr))
            target3 = int(round(3.5 * atr))
        elif confidence_score >= 50:
            target1 = int(round(1.2 * atr))  # Moderate targets
            target2 = int(round(2.0 * atr))
            target3 = int(round(3.0 * atr))
        else:
            target1 = int(round(1.0 * atr))  # Conservative targets
            target2 = int(round(1.5 * atr))
            target3 = int(round(2.0 * atr))
        
        # Calculate performance if we have a signal and future data
        if signal != "NO TRADE" and future_data is not None and not future_data.empty:
            # Calculate performance metrics with trailing stop logic
            if signal == "BUY CALL":
                stop_loss_price = candle['close'] - stop_loss
                target1_price = candle['close'] + target1
                target2_price = candle['close'] + target2
                target3_price = candle['close'] + target3
                
                highest_price = candle['close']
                trailing_sl = None
                target1_hit = target2_hit = target3_hit = False
                for i, future_candle in future_data.iterrows():
                    # Check if stop loss is hit first
                    if not target1_hit and future_candle['low'] <= stop_loss_price:
                        outcome = "Loss"
                        pnl = -stop_loss
                        stoploss_count = 1
                        failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        break
                    # Check which targets are hit
                    if not target1_hit and future_candle['high'] >= target1_price:
                        target1_hit = True
                        targets_hit = 1
                        pnl = target1
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = highest_price - stop_loss
                        outcome = "Win"
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        continue
                    if target1_hit and not target2_hit and future_candle['high'] >= target2_price:
                        target2_hit = True
                        targets_hit = 2
                        pnl += (target2 - target1)
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = max(trailing_sl, highest_price - stop_loss)
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    if target2_hit and not target3_hit and future_candle['high'] >= target3_price:
                        target3_hit = True
                        targets_hit = 3
                        pnl += (target3 - target2)
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = max(trailing_sl, highest_price - stop_loss)
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    # After target1, always trail SL at highest_price - stop_loss
                    if target1_hit:
                        highest_price = max(highest_price, future_candle['high'])
                        trailing_sl = max(trailing_sl, highest_price - stop_loss)
                        # If price hits trailing SL, exit
                        if future_candle['low'] <= trailing_sl:
                            outcome = "Win"
                            pnl = trailing_sl - candle['close']
                            failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                            if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                                exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                            elif 'time' in future_data.columns:
                                exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                            break
            elif signal == "BUY PUT":
                stop_loss_price = candle['close'] + stop_loss
                target1_price = candle['close'] - target1
                target2_price = candle['close'] - target2
                target3_price = candle['close'] - target3
                
                lowest_price = candle['close']
                trailing_sl = None
                target1_hit = target2_hit = target3_hit = False
                for i, future_candle in future_data.iterrows():
                    # Check if stop loss is hit first
                    if not target1_hit and future_candle['high'] >= stop_loss_price:
                        outcome = "Loss"
                        pnl = -stop_loss
                        stoploss_count = 1
                        failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        break
                    # Check which targets are hit
                    if not target1_hit and future_candle['low'] <= target1_price:
                        target1_hit = True
                        targets_hit = 1
                        pnl = target1
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = lowest_price + stop_loss
                        outcome = "Win"
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                        continue
                    if target1_hit and not target2_hit and future_candle['low'] <= target2_price:
                        target2_hit = True
                        targets_hit = 2
                        pnl += (target2 - target1)
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    if target2_hit and not target3_hit and future_candle['low'] <= target3_price:
                        target3_hit = True
                        targets_hit = 3
                        pnl += (target3 - target2)
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                        if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                            exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                        elif 'time' in future_data.columns:
                            exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                    # After target1, always trail SL at lowest_price + stop_loss
                    if target1_hit:
                        lowest_price = min(lowest_price, future_candle['low'])
                        trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                        # If price hits trailing SL, exit
                        if future_candle['high'] >= trailing_sl:
                            outcome = "Win"
                            pnl = candle['close'] - trailing_sl
                            failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                            if hasattr(future_candle, 'name') and isinstance(future_candle.name, pd.Timestamp):
                                exit_time = future_candle.name.strftime("%Y-%m-%d %H:%M:%S")
                            elif 'time' in future_data.columns:
                                exit_time = future_candle['time'].strftime("%Y-%m-%d %H:%M:%S")
                            break
        
        # Defensive IST conversion for exit_time in signal_data
        exit_time_val = exit_time
        exit_time_str = self.to_ist_str(exit_time) or (str(exit_time) if exit_time is not None else None)
        
        # Combine all confidence reasons
        detailed_reasons = "; ".join(confidence_reasons) if confidence_reasons else "Standard analysis"
        
        # Return the signal data
        signal_data = {
            "signal": signal,
            "price": candle['close'],
            "rsi": candle['rsi'],
            "rsi_level": rsi_level,
            "ema_20": candle['ema'] if 'ema' in candle else 0,
            "atr": candle['atr'],
            "confidence": confidence_level,
            "confidence_score": confidence_score,
            "rsi_reason": rsi_reason,
            "price_reason": f"{price_reason} | {detailed_reasons}",
            "trade_type": trade_type,
            "stop_loss": stop_loss,
            "target": target1,
            "target2": target2,
            "target3": target3,
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time_str
        }
        
        # If index_name is provided, log to database
        if index_name and signal != "NO TRADE":
            db_signal_data = signal_data.copy()
            signal_time = self.safe_signal_time(candle.name)
            # Only convert to IST if signal_time is a datetime
            if isinstance(signal_time, (pd.Timestamp, datetime)):
                db_signal_data["signal_time"] = self.to_ist_str(signal_time)
            else:
                db_signal_data["signal_time"] = str(signal_time) if signal_time is not None else None
            db_signal_data["index_name"] = index_name
            db.log_strategy('insidebar_rsi', db_signal_data)
        
        return signal_data
