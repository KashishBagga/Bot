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
        df = self.add_indicators(df)

        results = []
        timeframes = {"3min": df, "15min": self.timeframe_data.get("15min"), "30min": self.timeframe_data.get("30min")}
        for tf, tf_df in timeframes.items():
            if tf_df is None or tf_df.empty:
                return None
            tf_result = self._evaluate_timeframe(tf_df, tf, ts)
            if tf_result is None:
                return None
            results.append(tf_result)

        bullish_votes = sum(1 for r in results if r["supertrend"] == 1 and r["ema_trend"] == 1)
        bearish_votes = sum(1 for r in results if r["supertrend"] == -1 and r["ema_trend"] == -1)

        if bullish_votes >= 2:
            signal = "BUY CALL"
            confidence = bullish_votes / 3.0
        elif bearish_votes >= 2:
            signal = "BUY PUT"
            confidence = bearish_votes / 3.0
        else:
            return None

        base = results[0]
        # Default outcome values
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        # If future_data is provided, calculate performance
        if future_data is not None and not future_data.empty:
            entry_price = base["candle"]["close"]
            atr = base["candle"].get("atr", 0)
            stop_loss = round(atr * 1.0, 2) if atr > 0 else 0
            target = round(atr * 1.5, 2) if atr > 0 else 0
            target2 = round(atr * 2.0, 2) if atr > 0 else 0
            target3 = round(atr * 2.5, 2) if atr > 0 else 0
            perf = self.calculate_performance(signal, entry_price, stop_loss, target, target2, target3, future_data)
            outcome = perf["outcome"]
            pnl = perf["pnl"]
            targets_hit = perf["targets_hit"]
            stoploss_count = perf["stoploss_count"]
            failure_reason = perf["failure_reason"]
            exit_time = perf["exit_time"]
        exit_time_str = self.to_ist_str(exit_time) or (str(exit_time) if exit_time is not None else None)
        return {
            "signal": signal,
            "confidence": round(confidence, 2),
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
            "exit_time": exit_time_str
        }

# Optional legacy adapter
def strategy_supertrend_ema(candle, index_name, future_data=None, price_to_ema_ratio=None):
    df = pd.DataFrame([candle])
    strategy = SupertrendEma()
    df = strategy.add_indicators(df)
    return strategy.analyze(df.iloc[-1], len(df) - 1, df)
