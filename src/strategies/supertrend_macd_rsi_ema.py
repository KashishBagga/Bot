import pandas as pd
from typing import Dict, Any, Optional
from src.core.strategy import Strategy
from src.core.indicators import indicators
from src.models.database import db
from src.services.option_utils import get_nearest_expiry, construct_option_symbol, fetch_option_ohlcv
import math
from datetime import datetime
import pytz

class SupertrendMacdRsiEma(Strategy):
    def __init__(self, params: Dict[str, Any] = None, timeframe_data: Optional[Dict[str, pd.DataFrame]] = None):
        super().__init__("supertrend_macd_rsi_ema", params)
        self.params.setdefault('supertrend_period', 10)
        self.params.setdefault('supertrend_multiplier', 3.0)
        self.timeframe_data = timeframe_data or {}

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add Supertrend indicator
        period = self.params['supertrend_period']
        multiplier = self.params['supertrend_multiplier']
        supertrend_data = indicators.supertrend(data, period=period, multiplier=multiplier)
        data['supertrend'] = supertrend_data['supertrend']
        data['supertrend_direction'] = supertrend_data['direction']
        
        # Add MACD indicator
        macd_data = indicators.macd(data)
        data['macd'] = macd_data['macd']
        data['macd_signal'] = macd_data['signal']
        data['macd_histogram'] = macd_data['histogram']
        
        # Add RSI indicator
        data['rsi'] = indicators.rsi(data)
        
        # Add EMA indicator
        data['ema'] = indicators.ema(data, period=20)
        
        # Add body ratio calculations
        data['body'] = abs(data['close'] - data['open'])
        data['full_range'] = data['high'] - data['low']
        data['body_ratio'] = data['body'] / data['full_range'].replace(0, float('nan'))
        
        return data

    def calculate_performance(self, signal: str, entry_price: float, stop_loss: float, 
                             target: float, target2: float, target3: float,
                             future_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for a trade signal.
        
        Args:
            signal: Trade signal ("BUY CALL" or "BUY PUT")
            entry_price: Entry price for the trade
            stop_loss: Stop loss amount (not price)
            target: First target amount
            target2: Second target amount  
            target3: Third target amount
            future_data: Future candles for performance tracking
            
        Returns:
            Dict containing outcome, pnl, targets_hit, stoploss_count, failure_reason, exit_time
        """
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        
        if future_data is None or future_data.empty:
            return {
                "outcome": "Data Missing",
                "pnl": 0.0,
                "targets_hit": 0,
                "stoploss_count": 0,
                "failure_reason": "No future data available",
                "exit_time": None
            }
        
        if signal == "BUY CALL":
            stop_loss_price = entry_price - stop_loss
            target1_price = entry_price + target
            target2_price = entry_price + target2
            target3_price = entry_price + target3
            
            highest_price = entry_price
            trailing_sl = None
            target1_hit = target2_hit = target3_hit = False
            
            for i, future_candle in future_data.iterrows():
                # Check stop loss first
                if not target1_hit and future_candle['low'] <= stop_loss_price:
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    break
                
                # Check targets
                if not target1_hit and future_candle['high'] >= target1_price:
                    target1_hit = True
                    targets_hit = 1
                    pnl = target
                    highest_price = max(highest_price, future_candle['high'])
                    trailing_sl = highest_price - stop_loss
                    outcome = "Win"
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                    
                if target1_hit and not target2_hit and future_candle['high'] >= target2_price:
                    target2_hit = True
                    targets_hit = 2
                    pnl += (target2 - target)
                    highest_price = max(highest_price, future_candle['high'])
                    trailing_sl = max(trailing_sl, highest_price - stop_loss)
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                    
                if target2_hit and not target3_hit and future_candle['high'] >= target3_price:
                    target3_hit = True
                    targets_hit = 3
                    pnl += (target3 - target2)
                    highest_price = max(highest_price, future_candle['high'])
                    trailing_sl = max(trailing_sl, highest_price - stop_loss)
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                
                # Update trailing stop
                if target1_hit:
                    highest_price = max(highest_price, future_candle['high'])
                    trailing_sl = max(trailing_sl, highest_price - stop_loss)
                    if future_candle['low'] <= trailing_sl:
                        outcome = "Win"
                        pnl = trailing_sl - entry_price
                        failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                        exit_time = self.safe_signal_time(future_candle.get('time', i))
                        break
                        
        elif signal == "BUY PUT":
            stop_loss_price = entry_price + stop_loss
            target1_price = entry_price - target
            target2_price = entry_price - target2
            target3_price = entry_price - target3
            
            lowest_price = entry_price
            trailing_sl = None
            target1_hit = target2_hit = target3_hit = False
            
            for i, future_candle in future_data.iterrows():
                # Check stop loss first
                if not target1_hit and future_candle['high'] >= stop_loss_price:
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    break
                
                # Check targets
                if not target1_hit and future_candle['low'] <= target1_price:
                    target1_hit = True
                    targets_hit = 1
                    pnl = target
                    lowest_price = min(lowest_price, future_candle['low'])
                    trailing_sl = lowest_price + stop_loss
                    outcome = "Win"
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                    
                if target1_hit and not target2_hit and future_candle['low'] <= target2_price:
                    target2_hit = True
                    targets_hit = 2
                    pnl += (target2 - target)
                    lowest_price = min(lowest_price, future_candle['low'])
                    trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                    
                if target2_hit and not target3_hit and future_candle['low'] <= target3_price:
                    target3_hit = True
                    targets_hit = 3
                    pnl += (target3 - target2)
                    lowest_price = min(lowest_price, future_candle['low'])
                    trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                
                # Update trailing stop
                if target1_hit:
                    lowest_price = min(lowest_price, future_candle['low'])
                    trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                    if future_candle['high'] >= trailing_sl:
                        outcome = "Win"
                        pnl = entry_price - trailing_sl
                        failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                        exit_time = self.safe_signal_time(future_candle.get('time', i))
                        break
        
        return {
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time
        }

    def safe_signal_time(self, val):
        """Safely convert a value to a datetime, returning a valid datetime or None."""
        if val is None:
            return datetime.now()
        
        if isinstance(val, datetime):
            return val
        
        try:
            if isinstance(val, (int, float)):
                # If it's a number, treat it as a timestamp
                return datetime.fromtimestamp(val)
            else:
                # Try to parse as datetime
                return pd.to_datetime(val)
        except:
            # If all else fails, return current time
            return datetime.now()

    def to_ist_str(self, val):
        """Convert a datetime value to IST string format."""
        try:
            if val is None:
                return None
            
            if isinstance(val, datetime):
                # Convert to IST
                ist_tz = pytz.timezone('Asia/Kolkata')
                if val.tzinfo is None:
                    val = pytz.utc.localize(val)
                ist_dt = val.astimezone(ist_tz)
                return ist_dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
        return None

    def analyze(self, candle: pd.Series, index: int, df: pd.DataFrame, future_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Multi-timeframe analysis method."""
        # Ensure DataFrame has DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            else:
                raise ValueError("DataFrame must have a datetime index or 'time' column")

        if index >= len(df):
            return None

        ts = df.index[index]
        
        # If we have timeframe data, use multi-timeframe analysis
        if self.timeframe_data:
            results = []
            timeframes = {
                "3min": df, 
                "15min": self.timeframe_data.get("15min"), 
                "30min": self.timeframe_data.get("30min")
            }
            
            for tf, tf_df in timeframes.items():
                if tf_df is None or tf_df.empty:
                    # If any timeframe is missing, fall back to single timeframe
                    break
                tf_result = self._evaluate_timeframe(tf_df, tf, ts)
                if tf_result is None:
                    break
                results.append(tf_result)
            
            # If we have all timeframe results, use multi-timeframe logic
            if len(results) == 3:
                bullish_votes = sum(1 for r in results if r["signal_direction"] == 1)
                bearish_votes = sum(1 for r in results if r["signal_direction"] == -1)
                
                if bullish_votes >= 2:
                    signal = "BUY CALL"
                    confidence = "High" if bullish_votes == 3 else "Medium"
                elif bearish_votes >= 2:
                    signal = "BUY PUT"
                    confidence = "High" if bearish_votes == 3 else "Medium"
                else:
                    return None  # No clear signal across timeframes
                
                # Use base timeframe (3min) candle for calculations
                base_candle = results[0]["candle"]
                
                # Calculate performance if future data is available
                outcome = "Pending"
                pnl = 0.0
                targets_hit = 0
                stoploss_count = 0
                failure_reason = ""
                exit_time = ""
                
                if future_data is not None and not future_data.empty:
                    atr = base_candle['atr']
                    stop_loss = math.ceil(atr)
                    target = math.ceil(1.5 * atr)
                    target2 = math.ceil(2.0 * atr)
                    target3 = math.ceil(2.5 * atr)
                    
                    result = self.calculate_performance(
                        signal, base_candle['close'], stop_loss, target, target2, target3, future_data
                    )
                    outcome = result['outcome']
                    pnl = result['pnl']
                    targets_hit = result['targets_hit']
                    stoploss_count = result['stoploss_count']
                    failure_reason = result['failure_reason']
                    exit_time = self.to_ist_str(result['exit_time']) or ""
                
                return {
                    "signal": signal,
                    "price": base_candle['close'],
                    "rsi": base_candle['rsi'],
                    "macd": base_candle['macd'],
                    "macd_signal": base_candle['macd_signal'],
                    "ema_20": base_candle['ema'],
                    "atr": base_candle['atr'],
                    "supertrend": base_candle['supertrend'],
                    "supertrend_direction": base_candle['supertrend_direction'],
                    "stop_loss": math.ceil(base_candle['atr']),
                    "target": math.ceil(1.5 * base_candle['atr']),
                    "target2": math.ceil(2.0 * base_candle['atr']),
                    "target3": math.ceil(2.5 * base_candle['atr']),
                    "confidence": confidence,
                    "rsi_reason": f"RSI {base_candle['rsi']:.2f}",
                    "macd_reason": f"MACD {base_candle['macd']:.2f} vs Signal {base_candle['macd_signal']:.2f}",
                    "price_reason": f"Multi-timeframe confirmation ({bullish_votes if signal == 'BUY CALL' else bearish_votes}/3 timeframes)",
                    "trade_type": "Intraday",
                    "option_chain_confirmation": "Yes" if confidence == "High" else "No",
                    "outcome": outcome,
                    "pnl": pnl,
                    "targets_hit": targets_hit,
                    "stoploss_count": stoploss_count,
                    "failure_reason": failure_reason,
                    "exit_time": exit_time,
                    "option_symbol": "",
                    "option_expiry": "",
                    "option_strike": 0,
                    "option_type": "",
                    "option_entry_price": 0.0
                }
        
        # Fall back to single timeframe analysis if multi-timeframe data not available
        return self.analyze_single_timeframe(df.iloc[index:index+1], future_data)

    def analyze_single_timeframe(self, data: pd.DataFrame, future_data=None) -> Dict[str, Any]:
        """Original single timeframe analysis method."""
        if 'supertrend' not in data.columns:
            data = self.add_indicators(data)

        candle = data.iloc[-1]
        signal = "NO TRADE"
        rsi_reason = macd_reason = price_reason = ""
        option_type = outcome = failure_reason = exit_time = option_symbol = None
        option_expiry = option_strike = option_entry_price = 0
        pnl = 0.0
        targets_hit = stoploss_count = 0

        # Calculate comprehensive confidence score based on all indicators
        confidence_score = 0
        confidence_reasons = []
        
        # 1. RSI Analysis (0-25 points)
        rsi = candle['rsi']
        if rsi <= 25:  # Extreme oversold
            confidence_score += 25
            confidence_reasons.append(f"Extreme RSI oversold ({rsi:.1f})")
        elif rsi <= 35:  # Strong oversold
            confidence_score += 20
            confidence_reasons.append(f"Strong RSI oversold ({rsi:.1f})")
        elif rsi >= 75:  # Extreme overbought
            confidence_score += 25
            confidence_reasons.append(f"Extreme RSI overbought ({rsi:.1f})")
        elif rsi >= 65:  # Strong overbought
            confidence_score += 20
            confidence_reasons.append(f"Strong RSI overbought ({rsi:.1f})")
        elif 45 <= rsi <= 55:  # Neutral zone
            confidence_score += 10
            confidence_reasons.append(f"RSI in neutral zone ({rsi:.1f})")
        
        # 2. MACD Analysis (0-25 points)
        macd = candle['macd']
        macd_signal = candle['macd_signal']
        macd_histogram = candle['macd_histogram']
        
        macd_strength = abs(macd - macd_signal)
        macd_direction = "bullish" if macd > macd_signal else "bearish"
        
        if macd_strength > 5 and macd_histogram > 2:  # Strong bullish MACD
            confidence_score += 25
            confidence_reasons.append(f"Strong bullish MACD crossover ({macd:.2f} > {macd_signal:.2f})")
        elif macd_strength > 5 and macd_histogram < -2:  # Strong bearish MACD
            confidence_score += 25
            confidence_reasons.append(f"Strong bearish MACD crossover ({macd:.2f} < {macd_signal:.2f})")
        elif macd_strength > 2:  # Good MACD divergence
            confidence_score += 15
            confidence_reasons.append(f"Good MACD {macd_direction} signal")
        elif macd_strength > 1:  # Mild MACD signal
            confidence_score += 8
            confidence_reasons.append(f"Mild MACD {macd_direction} signal")
        
        # 3. SuperTrend Analysis (0-20 points)
        supertrend = candle['supertrend']
        supertrend_direction = candle['supertrend_direction']
        price_to_st_distance = abs(candle['close'] - supertrend) / candle['close'] * 100
        
        if supertrend_direction > 0 and price_to_st_distance > 1.0:  # Strong bullish SuperTrend
            confidence_score += 20
            confidence_reasons.append(f"Strong bullish SuperTrend (price {price_to_st_distance:.2f}% above)")
        elif supertrend_direction < 0 and price_to_st_distance > 1.0:  # Strong bearish SuperTrend
            confidence_score += 20
            confidence_reasons.append(f"Strong bearish SuperTrend (price {price_to_st_distance:.2f}% below)")
        elif supertrend_direction > 0 and price_to_st_distance > 0.5:  # Good bullish SuperTrend
            confidence_score += 15
            confidence_reasons.append(f"Good bullish SuperTrend")
        elif supertrend_direction < 0 and price_to_st_distance > 0.5:  # Good bearish SuperTrend
            confidence_score += 15
            confidence_reasons.append(f"Good bearish SuperTrend")
        elif supertrend_direction != 0:  # Basic SuperTrend signal
            confidence_score += 8
            direction = "bullish" if supertrend_direction > 0 else "bearish"
            confidence_reasons.append(f"Basic {direction} SuperTrend signal")
        
        # 4. EMA Analysis (0-15 points)
        ema = candle['ema']
        price_to_ema_distance = (candle['close'] - ema) / ema * 100 if ema != 0 else 0
        
        if abs(price_to_ema_distance) > 2.0:  # Strong price-EMA separation
            confidence_score += 15
            direction = "above" if price_to_ema_distance > 0 else "below"
            confidence_reasons.append(f"Price strongly {direction} EMA ({price_to_ema_distance:.2f}%)")
        elif abs(price_to_ema_distance) > 1.0:  # Good price-EMA separation
            confidence_score += 10
            direction = "above" if price_to_ema_distance > 0 else "below"
            confidence_reasons.append(f"Price {direction} EMA ({price_to_ema_distance:.2f}%)")
        elif abs(price_to_ema_distance) > 0.5:  # Mild separation
            confidence_score += 5
            direction = "above" if price_to_ema_distance > 0 else "below"
            confidence_reasons.append(f"Price slightly {direction} EMA")
        
        # 5. Volume and Volatility Analysis (0-15 points)
        atr = candle['atr']
        body_size = abs(candle['close'] - candle['open'])
        price_range = candle['high'] - candle['low']
        body_ratio = body_size / price_range if price_range > 0 else 0
        
        if atr > 60 and body_ratio > 0.7:  # High volatility with strong candle
            confidence_score += 15
            confidence_reasons.append(f"High volatility with strong directional candle (ATR: {atr:.1f}, Body: {body_ratio:.2f})")
        elif atr > 40 and body_ratio > 0.5:  # Good volatility with decent candle
            confidence_score += 10
            confidence_reasons.append(f"Good volatility environment (ATR: {atr:.1f})")
        elif atr > 25:  # Decent volatility
            confidence_score += 5
            confidence_reasons.append(f"Decent volatility (ATR: {atr:.1f})")
        
        # Determine confidence level
        if confidence_score >= 85:
            confidence = "Very High"
        elif confidence_score >= 65:
            confidence = "High"
        elif confidence_score >= 45:
            confidence = "Medium"
        elif confidence_score >= 25:
            confidence = "Low"
        else:
            confidence = "Very Low"

        # Dynamic risk management based on confidence and market conditions
        if confidence_score >= 85:  # Very high confidence
            stop_loss_multiplier = 0.5  # Very tight stop loss
            target_multipliers = [1.2, 2.0, 3.0]  # Aggressive targets
        elif confidence_score >= 65:  # High confidence
            stop_loss_multiplier = 0.6
            target_multipliers = [1.0, 1.5, 2.2]
        elif confidence_score >= 45:  # Medium confidence
            stop_loss_multiplier = 0.7
            target_multipliers = [0.8, 1.2, 1.8]
        else:  # Low confidence
            stop_loss_multiplier = 0.8
            target_multipliers = [0.6, 1.0, 1.5]

        stop_loss = math.ceil(stop_loss_multiplier * atr)
        target = math.ceil(target_multipliers[0] * atr)
        target2 = math.ceil(target_multipliers[1] * atr)
        target3 = math.ceil(target_multipliers[2] * atr)

        # Enhanced signal criteria based on multi-indicator alignment
        is_buy_call = (
            rsi >= 45 and rsi <= 65 and  # RSI in optimal range or oversold
            macd > macd_signal and
            macd > -5 and  # MACD not extremely negative
            candle['close'] > candle['ema'] * 1.002 and     # Price above EMA with buffer
            supertrend_direction > 0 and
            candle['close'] > candle['supertrend'] * 1.001  # Price above SuperTrend with buffer
        ) or (
            rsi <= 35 and  # Oversold conditions for reversal
            macd > macd_signal and
            candle['close'] > candle['ema'] and
            supertrend_direction > 0
        )

        is_buy_put = (
            rsi >= 35 and rsi <= 55 and  # RSI in optimal range or overbought
            macd < macd_signal and
            macd < 5 and  # MACD not extremely positive
            candle['close'] < candle['ema'] * 0.998 and     # Price below EMA with buffer
            supertrend_direction < 0 and
            candle['close'] < candle['supertrend'] * 0.999  # Price below SuperTrend with buffer
        ) or (
            rsi >= 65 and  # Overbought conditions for reversal
            macd < macd_signal and
            candle['close'] < candle['ema'] and
            supertrend_direction < 0
        )

        if is_buy_call:
            signal = "BUY CALL"
            rsi_reason = f"RSI {candle['rsi']:.2f} in bullish zone"
            macd_reason = f"MACD {candle['macd']:.2f} > Signal {candle['macd_signal']:.2f}"
            price_reason = f"Price {candle['close']:.2f} >> EMA {candle['ema']:.2f} and > SuperTrend {candle['supertrend']:.2f}"
            option_type = "CE"

        elif is_buy_put:
            signal = "BUY PUT"
            rsi_reason = f"RSI {candle['rsi']:.2f} in bearish zone"
            macd_reason = f"MACD {candle['macd']:.2f} < Signal {candle['macd_signal']:.2f}"
            price_reason = f"Price {candle['close']:.2f} << EMA {candle['ema']:.2f} and < SuperTrend {candle['supertrend']:.2f}"
            option_type = "PE"

        # Enhanced filtering: Only trade with Medium+ confidence (score >= 45)
        if signal != "NO TRADE" and confidence_score < 45:
            signal = "NO TRADE"
            confidence = "Low"
            rsi_reason += f" (Filtered: Confidence score {confidence_score} < 45)"

        # Additional filter for very high confidence trades
        if signal != "NO TRADE" and confidence_score >= 65:
            # Check for indicator alignment
            indicators_aligned = (
                (signal == "BUY CALL" and rsi <= 60 and macd > 0 and supertrend_direction > 0) or
                (signal == "BUY PUT" and rsi >= 40 and macd < 0 and supertrend_direction < 0)
            )
            if not indicators_aligned:
                confidence_score -= 20  # Reduce confidence for misaligned indicators
                if confidence_score < 45:
                    signal = "NO TRADE"
                    confidence = "Low"
                    rsi_reason += " (Indicator misalignment detected)"

        # Fallback: If option data is missing, simulate on underlying
        if signal.startswith("BUY") and future_data is not None and not future_data.empty:
            option_entry_price = candle['close']
            result = self.calculate_performance(
                signal, option_entry_price, stop_loss, target, target2, target3, future_data
            )
            outcome = result['outcome']
            pnl = result['pnl']
            targets_hit = result['targets_hit']
            stoploss_count = result['stoploss_count']
            failure_reason = "Simulated on underlying due to missing option data"
            exit_time = result['exit_time']

        # --- Outcome logic fix ---
        # If a trade was simulated and outcome is set by calculate_performance, keep it
        # Otherwise, set outcome to 'No Trade' or 'Data Missing' as appropriate
        if signal == "NO TRADE":
            outcome = "No Trade"
        elif (signal.startswith("BUY") and (not option_entry_price or future_data is None or future_data.empty)):
            outcome = "Data Missing"
        elif outcome not in ["Win", "Loss"]:
            outcome = "No Trade"
        # Defensive: ensure no output field is None or blank
        if option_symbol is None:
            option_symbol = ""
        if option_expiry is None:
            option_expiry = ""
        if option_type is None:
            option_type = ""
        if option_entry_price is None:
            option_entry_price = 0.0
        if exit_time is None:
            exit_time = ""
        else:
            # Convert exit_time to IST string format for database compatibility
            exit_time = self.to_ist_str(exit_time) or ""
        if failure_reason is None:
            failure_reason = ""
        if pnl is None:
            pnl = 0.0
        if targets_hit is None:
            targets_hit = 0
        if stoploss_count is None:
            stoploss_count = 0
        return {
            "signal": signal,
            "price": candle['close'],
            "rsi": candle['rsi'],
            "macd": candle['macd'],
            "macd_signal": candle['macd_signal'],
            "ema_20": candle['ema'],
            "atr": candle['atr'],
            "supertrend": candle['supertrend'],
            "supertrend_direction": candle['supertrend_direction'],
            "stop_loss": stop_loss,
            "target": target,
            "target2": target2,
            "target3": target3,
            "confidence": confidence,
            "rsi_reason": rsi_reason,
            "macd_reason": macd_reason,
            "price_reason": price_reason,
            "trade_type": "Intraday",
            "option_chain_confirmation": "Yes" if confidence == "High" else "No",
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time,
            "option_symbol": option_symbol,
            "option_expiry": str(option_expiry) if option_expiry else "",
            "option_strike": option_strike,
            "option_type": option_type,
            "option_entry_price": option_entry_price
        }

    def _evaluate_timeframe(self, df: pd.DataFrame, timeframe: str, ts: datetime) -> Optional[Dict[str, Any]]:
        """Evaluate a specific timeframe for signal confirmation."""
        df = df[df.index <= ts].copy()
        if df.empty or len(df) < 20:  # Need at least 20 candles for indicators
            return None

        # Add indicators to this timeframe data
        try:
            df = self.add_indicators(df)
        except (IndexError, ValueError):
            # Not enough data for indicators
            return None
            
        candle = df.iloc[-1]

        # Check conditions for this timeframe
        is_buy_call = (
            candle['rsi'] > 55 and
            candle['macd'] > candle['macd_signal'] and
            candle['close'] > candle['ema'] * 1.005 and
            candle['supertrend_direction'] > 0
        )

        is_buy_put = (
            candle['rsi'] < 45 and
            candle['macd'] < candle['macd_signal'] and
            candle['close'] < candle['ema'] * 0.995 and
            candle['supertrend_direction'] < 0
        )

        signal_direction = 0
        if is_buy_call:
            signal_direction = 1
        elif is_buy_put:
            signal_direction = -1

        return {
            "signal_direction": signal_direction,
            "rsi": candle['rsi'],
            "macd": candle['macd'],
            "macd_signal": candle['macd_signal'],
            "ema": candle['ema'],
            "supertrend_direction": candle['supertrend_direction'],
            "candle": candle
        }
