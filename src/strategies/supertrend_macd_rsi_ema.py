import pandas as pd
from typing import Dict, Any
from src.core.strategy import Strategy
from src.core.indicators import indicators
from src.models.database import db
from src.services.option_utils import get_nearest_expiry, construct_option_symbol, fetch_option_ohlcv
import math

class SupertrendMacdRsiEma(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("supertrend_macd_rsi_ema", params)
        self.params.setdefault('supertrend_period', 10)
        self.params.setdefault('supertrend_multiplier', 3.0)

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params['supertrend_period']
        multiplier = self.params['supertrend_multiplier']
        supertrend_data = indicators.supertrend(data, period=period, multiplier=multiplier)
        data['supertrend'] = supertrend_data['supertrend']
        data['supertrend_direction'] = supertrend_data['direction']
        data['body'] = abs(data['close'] - data['open'])
        data['full_range'] = data['high'] - data['low']
        data['body_ratio'] = data['body'] / data['full_range'].replace(0, float('nan'))
        return data

    def analyze(self, data: pd.DataFrame, index_name: str = None, future_data=None) -> Dict[str, Any]:
        if 'supertrend' not in data.columns:
            data = self.calculate_indicators(data)

        candle = data.iloc[-1]
        signal = "NO TRADE"
        confidence = "Low"
        rsi_reason = macd_reason = price_reason = ""
        option_type = outcome = failure_reason = exit_time = option_symbol = None
        option_expiry = option_strike = option_entry_price = 0
        pnl = 0.0
        targets_hit = stoploss_count = 0

        atr = candle['atr']
        stop_loss = math.ceil(atr)
        target = math.ceil(1.5 * atr)
        target2 = math.ceil(2.0 * atr)
        target3 = math.ceil(2.5 * atr)

        is_buy_call = (
            candle['rsi'] > 65 and
            candle['macd'] > candle['macd_signal'] and
            candle['close'] > candle['ema'] * 0.1 and
            candle['supertrend_direction'] > 0
        )

        is_buy_put = (
            candle['rsi'] < 35 and
            candle['macd'] < candle['macd_signal'] and
            candle['close'] < candle['ema'] * 1.01 and
            candle['supertrend_direction'] < 0
        )

        if is_buy_call:
            signal = "BUY CALL"
            confidence = "High" if candle['rsi'] > 70 else "Medium"
            rsi_reason = f"RSI {candle['rsi']:.2f} > 65"
            macd_reason = f"MACD {candle['macd']:.2f} > Signal {candle['macd_signal']:.2f}"
            price_reason = f"Price {candle['close']:.2f} > EMA {candle['ema']:.2f}, Supertrend bullish"
            option_type = "CE"

        elif is_buy_put:
            signal = "BUY PUT"
            confidence = "High" if candle['rsi'] < 30 else "Medium"
            rsi_reason = f"RSI {candle['rsi']:.2f} < 45"
            macd_reason = f"MACD {candle['macd']:.2f} < Signal {candle['macd_signal']:.2f}"
            price_reason = f"Price {candle['close']:.2f} < EMA {candle['ema']:.2f}, Supertrend bearish"
            option_type = "PE"

        if signal.startswith("BUY") and index_name:
            option_expiry = get_nearest_expiry(candle.get('time'))
            option_strike = int(round(candle['close'] / 50) * 50)
            option_symbol = construct_option_symbol(index_name, option_expiry, option_strike, option_type)
            entry_time = candle.get('time')

            if entry_time:
                entry_date = entry_time.strftime('%Y-%m-%d')
                option_ohlcv = fetch_option_ohlcv(option_symbol, entry_date, entry_date, resolution="5")
                if not option_ohlcv.empty:
                    option_ohlcv['timedelta'] = (option_ohlcv['time'] - entry_time).abs()
                    entry_row = option_ohlcv.loc[option_ohlcv['timedelta'].idxmin()]
                    option_entry_price = entry_row['close']

            if option_entry_price and future_data is not None and not future_data.empty:
                option_future_ohlcv = fetch_option_ohlcv(
                    option_symbol,
                    future_data['time'].iloc[0].strftime('%Y-%m-%d'),
                    future_data['time'].iloc[-1].strftime('%Y-%m-%d'),
                    resolution="5"
                )
                if not option_future_ohlcv.empty:
                    option_future_ohlcv = option_future_ohlcv[option_future_ohlcv['time'] > entry_time]
                    result = self.calculate_performance(
                        signal, option_entry_price, stop_loss, target, target2, target3, option_future_ohlcv
                    )
                    outcome = result['outcome']
                    pnl = result['pnl']
                    targets_hit = result['targets_hit']
                    stoploss_count = result['stoploss_count']
                    failure_reason = result['failure_reason']
                    exit_time = result['exit_time']

        return {
            "signal": signal,
            "price": candle['close'],
            "rsi": candle['rsi'],
            "macd": candle['macd'],
            "macd_signal": candle['macd_signal'],
            "ema_20": candle['ema'],
            "atr": atr,
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
            "option_expiry": str(option_expiry) if option_expiry else None,
            "option_strike": option_strike,
            "option_type": option_type,
            "option_entry_price": option_entry_price
        }
