import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from indicators.supertrend import get_supertrend_instance


class SupertrendEma(Strategy):
    """
    Multi-timeframe Supertrend + EMA strategy with signal confirmation across 3min, 15min, and 30min charts.
    """

    # Minimum candles required for analysis
    min_candles = 60

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
            # Use closed candle for analysis
            candle = self.get_closed_candle(df)
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
                    if candle['high'] >= (entry_price + target2):
                        targets_hit = 2
                        pnl = target2
                        if candle['high'] >= (entry_price + target3):
                            targets_hit = 3
                            pnl = target3
                            outcome = "Win"
                            exit_time = current_time
                            break
                    if targets_hit == 1:
                        outcome = "Win"
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
                    if candle['low'] <= (entry_price - target2):
                        targets_hit = 2
                        pnl = target2
                        if candle['low'] <= (entry_price - target3):
                            targets_hit = 3
                            pnl = target3
                            outcome = "Win"
                            exit_time = current_time
                            break
                    if targets_hit == 1:
                        outcome = "Win"
                        exit_time = current_time
                        break
        return {
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time
        }

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and generate trading signals using closed candles."""
        if not self.validate_data(data):
            return {'signal': 'NO TRADE', 'reason': 'insufficient data'}

        try:
            # Use closed candle for analysis
            candle = self.get_closed_candle(data)
            
            # Add indicators
            df = self.calculate_indicators(data)
            
            # Get SuperTrend data
            st_instance = self._get_supertrend_instance("5min")
            st_data = st_instance.update(candle)
            
            if st_data is None or st_data[1] is None:
                return {'signal': 'NO TRADE', 'reason': 'insufficient supertrend data'}
            
            supertrend_direction = st_data[1]
            
            # Get EMAs
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            
            # Check for NaN values
            if pd.isna(ema_9) or pd.isna(ema_21) or pd.isna(rsi) or pd.isna(macd):
                return {'signal': 'NO TRADE', 'reason': 'indicator data unavailable'}
            
            # BUY CALL conditions
            if (supertrend_direction == 1 and  # SuperTrend uptrend
                ema_9 > ema_21 and  # EMA crossover
                30 < rsi < 80 and  # RSI in healthy range
                macd > macd_signal and  # MACD bullish
                volume_ratio > 0.5):  # Volume confirmation
                
                # Calculate confidence score
                trend_strength = 25 if ema_9 > ema_21 * 1.01 else 15
                rsi_strength = 20 if 40 < rsi < 70 else 10
                macd_strength = 15 if macd > macd_signal * 1.1 else 8
                volume_strength = 10 if volume_ratio > 1.0 else 5
                
                confidence_score = trend_strength + rsi_strength + macd_strength + volume_strength
                
                if confidence_score >= 70:  # High confidence threshold
                    atr = df['atr'].iloc[-1]
                    stop_loss = 1.5 * atr
                    target1 = 2.0 * atr
                    target2 = 3.0 * atr
                    target3 = 4.0 * atr
                    
                    # Dynamic position sizing based on confidence
                    position_multiplier = 1.0 if confidence_score >= 80 else 0.8
                    
                    return {
                        'signal': 'BUY CALL',
                        'price': candle['close'],
                        'confidence_score': confidence_score,
                        'stop_loss': stop_loss,
                        'target1': target1,
                        'target2': target2,
                        'target3': target3,
                        'position_multiplier': position_multiplier,
                        'timestamp': candle.name if hasattr(candle, 'name') else datetime.now(),
                        'reasoning': f"SuperTrend uptrend, EMA crossover, RSI {rsi:.1f}, MACD bullish, Volume {volume_ratio:.2f}"
                    }
            
            # BUY PUT conditions
            elif (supertrend_direction == -1 and  # SuperTrend downtrend
                  ema_9 < ema_21 and  # EMA crossover
                  20 < rsi < 70 and  # RSI in healthy range
                  macd < macd_signal and  # MACD bearish
                  volume_ratio > 0.5):  # Volume confirmation
                
                # Calculate confidence score
                trend_strength = 25 if ema_9 < ema_21 * 0.99 else 15
                rsi_strength = 20 if 30 < rsi < 60 else 10
                macd_strength = 15 if macd < macd_signal * 0.9 else 8
                volume_strength = 10 if volume_ratio > 1.0 else 5
                
                confidence_score = trend_strength + rsi_strength + macd_strength + volume_strength
                
                if confidence_score >= 70:  # High confidence threshold
                    atr = df['atr'].iloc[-1]
                    stop_loss = 1.5 * atr
                    target1 = 2.0 * atr
                    target2 = 3.0 * atr
                    target3 = 4.0 * atr
                    
                    # Dynamic position sizing based on confidence
                    position_multiplier = 1.0 if confidence_score >= 80 else 0.8
                    
                    return {
                        'signal': 'BUY PUT',
                        'price': candle['close'],
                        'confidence_score': confidence_score,
                        'stop_loss': stop_loss,
                        'target1': target1,
                        'target2': target2,
                        'target3': target3,
                        'position_multiplier': position_multiplier,
                        'timestamp': candle.name if hasattr(candle, 'name') else datetime.now(),
                        'reasoning': f"SuperTrend downtrend, EMA crossover, RSI {rsi:.1f}, MACD bearish, Volume {volume_ratio:.2f}"
                    }
            
            return {'signal': 'NO TRADE', 'reason': 'no signal conditions met'}
            
        except Exception as e:
            logging.error(f"Error in SupertrendEma analysis: {e}")
            return {'signal': 'ERROR', 'reason': str(e)}

# Optional legacy adapter
def strategy_supertrend_ema(candle, index_name, future_data=None, price_to_ema_ratio=None):
    df = pd.DataFrame([candle])
    strategy = SupertrendEma()
    df = strategy.add_indicators(df)
    return strategy.analyze(df.iloc[-1], len(df) - 1, df)
