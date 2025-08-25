import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from indicators.supertrend import get_supertrend_instance


class SupertrendEma(Strategy):
    """
    Multi-timeframe Supertrend + EMA strategy with signal confirmation across 3min, 15min, and 30min charts.
    """

    def __init__(self, params: Dict[str, Any] = None, timeframe_data: Optional[Dict[str, pd.DataFrame]] = None):
        params = params or {}
        self.ema_period = params.get("ema_period", 20)
        self.supertrend_period = params.get("supertrend_period", 10)
        self.supertrend_multiplier = params.get("supertrend_multiplier", 3.0)
        self.timeframe_data = timeframe_data or {}
        self._supertrend_instances = {}
        super().__init__("supertrend_ema", params)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema"] = df["close"].ewm(span=self.ema_period).mean()
        df["price_to_ema_ratio"] = (df["close"] / df["ema"] - 1) * 100
        return df

    def _get_supertrend_instance(self, timeframe: str):
        key = f"{self.__class__.__name__}_{timeframe}"
        if key not in self._supertrend_instances:
            self._supertrend_instances[key] = get_supertrend_instance(
                key, period=self.supertrend_period, multiplier=self.supertrend_multiplier
            )
        return self._supertrend_instances[key]

    def _evaluate_timeframe(self, df: pd.DataFrame, timeframe: str, ts: datetime) -> Optional[Dict[str, Any]]:
        df = df[df.index <= ts].copy()
        if df.empty or len(df) < 20:  # Need at least 20 candles for indicators
            return None

        try:
            candle = df.iloc[-1]
            st_instance = self._get_supertrend_instance(timeframe)
            st_data = st_instance.update(candle)

            ema = df["close"].ewm(span=self.ema_period).mean().iloc[-1]
            price_above_ema = candle["close"] > ema
        except (IndexError, ValueError):
            # Not enough data for indicators
            return None

        return {
            "supertrend": st_data["direction"],
            "ema_trend": 1 if price_above_ema else -1,
            "ema": ema,
            "candle": candle,
            "st_data": st_data
        }

    def safe_signal_time(self, val):
        return val if isinstance(val, (pd.Timestamp, datetime)) else datetime.now()

    def to_ist_str(self, val):
        if isinstance(val, (pd.Timestamp, datetime)):
            ist_dt = val + timedelta(hours=5, minutes=30)
            return ist_dt.strftime("%Y-%m-%d %H:%M:%S")
        return None

    def calculate_performance(self, signal: str, entry_price: float, stop_loss: float, 
                             target: float, target2: float, target3: float,
                             future_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics based on future data."""
        if future_data is None or future_data.empty:
            return {
                "outcome": "Pending",
                "pnl": 0.0,
                "targets_hit": 0,
                "stoploss_count": 0,
                "failure_reason": "",
                "exit_time": None
            }
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        if signal == "BUY CALL":
            for idx, candle in future_data.iterrows():
                current_time = self.safe_signal_time(candle.get('time', None))
                if candle['low'] <= (entry_price - stop_loss):
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {entry_price - stop_loss:.2f}"
                    exit_time = current_time
                    break
                if targets_hit == 0 and candle['high'] >= (entry_price + target):
                    targets_hit = 1
                    pnl = target
                    outcome = "Win"
                    exit_time = current_time
                if targets_hit == 1 and candle['high'] >= (entry_price + target2):
                    targets_hit = 2
                    pnl += (target2 - target)
                if targets_hit == 2 and candle['high'] >= (entry_price + target3):
                    targets_hit = 3
                    pnl += (target3 - target2)
                    exit_time = current_time
                    break
        elif signal == "BUY PUT":
            for idx, candle in future_data.iterrows():
                current_time = self.safe_signal_time(candle.get('time', None))
                if candle['high'] >= (entry_price + stop_loss):
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {entry_price + stop_loss:.2f}"
                    exit_time = current_time
                    break
                if targets_hit == 0 and candle['low'] <= (entry_price - target):
                    targets_hit = 1
                    pnl = target
                    outcome = "Win"
                    exit_time = current_time
                if targets_hit == 1 and candle['low'] <= (entry_price - target2):
                    targets_hit = 2
                    pnl += (target2 - target)
                if targets_hit == 2 and candle['low'] <= (entry_price - target3):
                    targets_hit = 3
                    pnl += (target3 - target2)
                    exit_time = current_time
                    break
        # Defensive IST conversion for exit_time
        exit_time_str = self.to_ist_str(exit_time) or (str(exit_time) if exit_time is not None else None)
        return {
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time_str
        }

    def analyze(self, candle: pd.Series, index: int, df: pd.DataFrame, future_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)

        if index < self.ema_period or index >= len(df):
            return None

        ts = df.index[index]
        
        # Calculate confidence score based on multiple market conditions instead of time filter
        confidence_score = 0
        confidence_reasons = []
        
        # Get current candle data
        candle = df.iloc[index]
        
        # 1. SuperTrend Signal Strength (0-25 points)
        st_instance = self._get_supertrend_instance("3min")
        st_data = st_instance.update(candle)
        supertrend_direction = st_data["direction"]
        
        if supertrend_direction != 0:  # Valid SuperTrend signal
            confidence_score += 25
            direction = "bullish" if supertrend_direction > 0 else "bearish"
            confidence_reasons.append(f"Strong SuperTrend {direction} signal")
        else:
            confidence_score += 5
            confidence_reasons.append("Weak SuperTrend signal")
        
        # 2. EMA Alignment (0-20 points)
        ema = df["close"].ewm(span=self.ema_period).mean().iloc[index]
        price_above_ema = candle["close"] > ema
        ema_distance = ((candle["close"] - ema) / ema) * 100
        
        if abs(ema_distance) > 1.0:  # Strong price-EMA separation
            confidence_score += 20
            direction = "above" if price_above_ema else "below"
            confidence_reasons.append(f"Price strongly {direction} EMA ({ema_distance:.2f}%)")
        elif abs(ema_distance) > 0.5:  # Good price-EMA separation
            confidence_score += 15
            direction = "above" if price_above_ema else "below"
            confidence_reasons.append(f"Price {direction} EMA ({ema_distance:.2f}%)")
        elif abs(ema_distance) > 0.2:  # Mild separation
            confidence_score += 10
            direction = "above" if price_above_ema else "below"
            confidence_reasons.append(f"Price slightly {direction} EMA ({ema_distance:.2f}%)")
        
        # 3. Multi-timeframe Consensus (0-30 points)
        df = self.add_indicators(df)
        
        results = []
        timeframes = {"3min": df, "15min": self.timeframe_data.get("15min"), "30min": self.timeframe_data.get("30min")}
        
        valid_timeframes = 0
        consensus_score = 0
        
        for tf, tf_df in timeframes.items():
            if tf_df is None or tf_df.empty:
                continue
            tf_result = self._evaluate_timeframe(tf_df, tf, ts)
            if tf_result is None:
                continue
            results.append(tf_result)
            valid_timeframes += 1
            
            # Count consensus
            if tf_result["supertrend"] == 1 and tf_result["ema_trend"] == 1:  # Bullish
                consensus_score += 1
            elif tf_result["supertrend"] == -1 and tf_result["ema_trend"] == -1:  # Bearish
                consensus_score += 1
        
        if valid_timeframes == 0:
            return None
        
        consensus_percentage = (consensus_score / valid_timeframes) * 100
        
        if consensus_percentage >= 100:  # Full consensus (3/3)
            confidence_score += 30
            confidence_reasons.append("Full multi-timeframe consensus (3/3)")
        elif consensus_percentage >= 67:  # Strong consensus (2/3)
            confidence_score += 20
            confidence_reasons.append("Strong multi-timeframe consensus (2/3)")
        elif consensus_percentage >= 33:  # Weak consensus (1/3)
            confidence_score += 10
            confidence_reasons.append("Weak multi-timeframe consensus (1/3)")
        
        # 4. Volume and Volatility (0-15 points)
        atr = candle.get("atr", 0)
        if atr > 50:  # High volatility
            confidence_score += 15
            confidence_reasons.append(f"High volatility environment (ATR: {atr:.1f})")
        elif atr > 30:  # Moderate volatility
            confidence_score += 10
            confidence_reasons.append(f"Moderate volatility (ATR: {atr:.1f})")
        elif atr > 15:  # Low volatility
            confidence_score += 5
            confidence_reasons.append(f"Low volatility (ATR: {atr:.1f})")
        
        # 5. Price Action Quality (0-10 points)
        price_range = candle["high"] - candle["low"]
        body_size = abs(candle["close"] - candle["open"])
        body_ratio = body_size / price_range if price_range > 0 else 0
        
        if body_ratio >= 0.7:  # Strong directional candle
            confidence_score += 10
            confidence_reasons.append(f"Strong directional candle ({body_ratio:.2f})")
        elif body_ratio >= 0.5:  # Good directional candle
            confidence_score += 5
            confidence_reasons.append(f"Good directional candle ({body_ratio:.2f})")
        
        # Multi-Factor Confidence Scoring System
        from src.core.multi_factor_confidence import MultiFactorConfidence
        
        mfc = MultiFactorConfidence()
        confidence_result = mfc.calculate_confidence(candle, "BUY" if supertrend_direction > 0 else "SELL", self.timeframe_data)
        
        confidence_score = confidence_result['total_score']
        confidence_factors = []
        
        # Add factor details to reasoning
        for factor, score in confidence_result['factors'].items():
            if score > 0:
                confidence_factors.append(f"{factor}: {score}")
        
        # Add detailed reasoning from multi-factor system
        for factor, reasons in confidence_result['reasons'].items():
            if reasons:
                confidence_factors.extend(reasons)
        
        # Multi-factor confidence already calculated above
        # confidence_score and confidence_factors are now set by MultiFactorConfidence system
        
        # OPTIMIZATION: Multi-factor confidence threshold for optimal win rate
        min_confidence_threshold = 50  # Reduced threshold to enable trading activity
        
        if confidence_score < min_confidence_threshold:
            return {
                "signal": "NO TRADE",
                "confidence": "Low",
                "confidence_score": confidence_score,
                "reasoning": f"Confidence {confidence_score} below {min_confidence_threshold} threshold. Factors: {', '.join(confidence_factors)}"
            }
        
        # Determine confidence level
        if confidence_score >= 90:
            confidence = "Very High"
        elif confidence_score >= 80:
            confidence = "High"
        elif confidence_score >= 70:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Enhanced signal generation based on confidence
        bullish_votes = sum(1 for r in results if r["supertrend"] == 1 and r["ema_trend"] == 1)
        bearish_votes = sum(1 for r in results if r["supertrend"] == -1 and r["ema_trend"] == -1)

        # Dynamic consensus requirements based on confidence
        if confidence_score >= 70:
            required_votes = 2  # High confidence allows 2/3 consensus
        else:
            required_votes = 3  # Low confidence requires unanimous 3/3 consensus

        if bullish_votes >= required_votes:
            signal = "BUY CALL"
        elif bearish_votes >= required_votes:
            signal = "BUY PUT"
        else:
            return {
                "signal": "NO TRADE",
                "confidence": confidence,
                "ema_value": 0.0,
                "supertrend_value": 0.0,
                "supertrend_direction": "WEAK",
                "supertrend_upperband": 0.0,
                "supertrend_lowerband": 0.0,
                "candle": candle.to_dict(),
                "outcome": "No Trade",
                "pnl": 0.0,
                "targets_hit": 0,
                "stoploss_count": 0,
                "failure_reason": f"Insufficient consensus: {max(bullish_votes, bearish_votes)}/{len(results)} votes (need {required_votes})",
                "exit_time": None
            }
        
        # Enhanced filtering: Only trade with Medium+ confidence (score >= 40)
        if confidence_score < 40:
            return {
                "signal": "NO TRADE",
                "confidence": "Very Low",
                "ema_value": 0.0,
                "supertrend_value": 0.0,
                "supertrend_direction": "LOW_CONFIDENCE",
                "supertrend_upperband": 0.0,
                "supertrend_lowerband": 0.0,
                "candle": candle.to_dict(),
                "outcome": "No Trade",
                "pnl": 0.0,
                "targets_hit": 0,
                "stoploss_count": 0,
                "failure_reason": f"Low confidence score: {confidence_score} < 40",
                "exit_time": None
            }
        
        base = results[0]
        
        # Dynamic risk management based on confidence
        atr = base["candle"].get("atr", 0)
        entry_price = base["candle"]["close"]
        
        if confidence_score >= 80:  # Very high confidence
            stop_loss_multiplier = 0.6  # Tighter stop loss
            target_multipliers = [1.5, 2.5, 3.5]  # Aggressive targets
        elif confidence_score >= 60:  # High confidence
            stop_loss_multiplier = 0.7
            target_multipliers = [1.3, 2.0, 3.0]
        elif confidence_score >= 40:  # Medium confidence
            stop_loss_multiplier = 0.8
            target_multipliers = [1.2, 1.8, 2.2]
        else:  # Should not reach here due to filtering, but safety
            stop_loss_multiplier = 1.0
            target_multipliers = [1.0, 1.5, 2.0]
        
        # Default outcome values
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        
        # If future_data is provided, calculate performance
        if future_data is not None and not future_data.empty:
            stop_loss = round(atr * stop_loss_multiplier, 2) if atr > 0 else 0
            target = round(atr * target_multipliers[0], 2) if atr > 0 else 0
            target2 = round(atr * target_multipliers[1], 2) if atr > 0 else 0
            target3 = round(atr * target_multipliers[2], 2) if atr > 0 else 0
            
            perf = self.calculate_performance(signal, entry_price, stop_loss, target, target2, target3, future_data)
            outcome = perf["outcome"]
            pnl = perf["pnl"]
            targets_hit = perf["targets_hit"]
            stoploss_count = perf["stoploss_count"]
            failure_reason = perf["failure_reason"]
            exit_time = perf["exit_time"]
        
        exit_time_str = self.to_ist_str(exit_time) or (str(exit_time) if exit_time is not None else None)
        
        # Combine confidence reasons
        detailed_reasons = "; ".join(confidence_reasons) if confidence_reasons else "Standard analysis"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "ema_value": round(base["ema"], 2),
            "supertrend_value": round(base["st_data"]["value"], 2),
            "supertrend_direction": base["st_data"]["direction"],
            "supertrend_upperband": round(base["st_data"]["upperband"], 2),
            "supertrend_lowerband": round(base["st_data"]["lowerband"], 2),
            "candle": base["candle"].to_dict(),
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "confidence_reasons": detailed_reasons,
            "exit_time": exit_time_str
        }

# Optional legacy adapter
def strategy_supertrend_ema(candle, index_name, future_data=None, price_to_ema_ratio=None):
    df = pd.DataFrame([candle])
    strategy = SupertrendEma()
    df = strategy.add_indicators(df)
    return strategy.analyze(df.iloc[-1], len(df) - 1, df)
