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
        """Analyze data and generate trading signals with enhanced quality filters."""
        if index < 50 or future_data is None or future_data.empty:
            return None
            
        # Get indicator values
        supertrend = candle.get('supertrend', 0)
        supertrend_direction = candle.get('supertrend_direction', 0)
        ema_21 = candle.get('ema_21', 0)
        ema_50 = candle.get('ema_50', 0)
        rsi = candle.get('rsi', 50)
        macd = candle.get('macd', 0)
        macd_signal = candle.get('macd_signal', 0)
        volume_ratio = candle.get('volume_ratio', 1.0)
        atr = candle.get('atr', candle['close'] * 0.01)
        price = candle['close']
        
        # OPTIMIZATION: Enhanced signal quality filters
        signal = "NO TRADE"
        confidence_score = 0
        
        # Check for valid indicator values
        if supertrend <= 0 or ema_21 <= 0 or ema_50 <= 0:
            return None
            
        # OPTIMIZATION: Enhanced SuperTrend signal conditions
        if supertrend_direction > 0 and price > supertrend:  # Bullish SuperTrend
            # Additional bullish filters for better win rate
            if (price > ema_21 and ema_21 > ema_50 and  # Trend alignment
                rsi > 45 and rsi < 75 and              # RSI not overbought
                macd > macd_signal and                 # MACD bullish
                volume_ratio > 1.0):                   # Above average volume
                
                signal = "BUY CALL"
                # OPTIMIZATION: Better confidence calculation
                trend_strength = min(20, (price - ema_21) / ema_21 * 100)
                rsi_strength = min(15, (rsi - 45) / 2)
                macd_strength = min(15, (macd - macd_signal) * 10)
                volume_strength = min(10, (volume_ratio - 1.0) * 10)
                
                confidence_score = 60 + trend_strength + rsi_strength + macd_strength + volume_strength
                
        elif supertrend_direction < 0 and price < supertrend:  # Bearish SuperTrend
            # Additional bearish filters for better win rate
            if (price < ema_21 and ema_21 < ema_50 and  # Trend alignment
                rsi < 55 and rsi > 25 and              # RSI not oversold
                macd < macd_signal and                 # MACD bearish
                volume_ratio > 1.0):                   # Above average volume
                
                signal = "BUY PUT"
                # OPTIMIZATION: Better confidence calculation
                trend_strength = min(20, (ema_21 - price) / ema_21 * 100)
                rsi_strength = min(15, (55 - rsi) / 2)
                macd_strength = min(15, (macd_signal - macd) * 10)
                volume_strength = min(10, (volume_ratio - 1.0) * 10)
                
                confidence_score = 60 + trend_strength + rsi_strength + macd_strength + volume_strength
        
        # OPTIMIZATION: Higher minimum confidence threshold for better quality
        if confidence_score < 70:  # Increased from 50
            return None
            
        # Determine confidence level
        if confidence_score >= 90:
            confidence = "Very High"
        elif confidence_score >= 80:
            confidence = "High"
        elif confidence_score >= 70:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # OPTIMIZATION: Improved risk-reward ratios based on confidence
        if confidence_score >= 85:
            stop_loss = int(round(1.2 * atr))  # Tighter stop loss
            target1 = int(round(2.4 * atr))    # 2:1 R:R
            target2 = int(round(3.6 * atr))    # 3:1 R:R
            target3 = int(round(4.8 * atr))    # 4:1 R:R
        elif confidence_score >= 75:
            stop_loss = int(round(1.5 * atr))
            target1 = int(round(3.0 * atr))    # 2:1 R:R
            target2 = int(round(4.5 * atr))    # 3:1 R:R
            target3 = int(round(6.0 * atr))    # 4:1 R:R
        else:  # 70-74
            stop_loss = int(round(1.8 * atr))
            target1 = int(round(3.6 * atr))    # 2:1 R:R
            target2 = int(round(5.4 * atr))    # 3:1 R:R
            target3 = int(round(7.2 * atr))    # 4:1 R:R
        
        # OPTIMIZATION: Position sizing based on confidence
        position_multiplier = 0.8 if confidence_score >= 80 else 0.6
        
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
        price_reason = f"SuperTrend: {supertrend_direction > 0 and 'BULLISH' or 'BEARISH'}"
        price_reason += f", Price: {price:.1f} vs SuperTrend: {supertrend:.1f}"
        price_reason += f", EMA21: {ema_21:.1f} vs EMA50: {ema_50:.1f}"
        price_reason += f", RSI: {rsi:.1f}, MACD: {macd:.2f} vs {macd_signal:.2f}"
        price_reason += f", Volume: {volume_ratio:.1f}x"
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

# Optional legacy adapter
def strategy_supertrend_ema(candle, index_name, future_data=None, price_to_ema_ratio=None):
    df = pd.DataFrame([candle])
    strategy = SupertrendEma()
    df = strategy.add_indicators(df)
    return strategy.analyze(df.iloc[-1], len(df) - 1, df)
