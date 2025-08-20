#!/usr/bin/env python3
from typing import Optional
import pandas as pd

class MacdCrossRsiFilter:
	def __init__(self, timeframe_data=None):
		self.timeframe_data = timeframe_data or {}

	def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
		# Assume indicators exist; add ATR if missing
		if 'atr' not in df.columns and len(df) >= 20:
			prev_close = df['close'].shift(1)
			tr1 = df['high'] - df['low']
			tr2 = (df['high'] - prev_close).abs()
			tr3 = (df['low'] - prev_close).abs()
			true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
			df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
		return df

	def analyze(self, candle, index, df: pd.DataFrame, future_data: Optional[pd.DataFrame] = None):
		if index < 50 or future_data is None or future_data.empty:
			return None
		
		price = float(candle['close'])
		macd = float(candle.get('macd', 0))
		macd_signal = float(candle.get('macd_signal', 0))
		macd_hist = float(candle.get('macd_histogram', candle.get('macd_diff', 0)))
		rsi = float(candle.get('rsi', 50))
		atr = float(candle.get('atr', 0))
		if atr <= 0:
			return None
		
		# Long on MACD cross up with RSI regime; Short on cross down with RSI regime
		long_setup = macd > macd_signal and macd_hist > 0 and rsi >= 45
		short_setup = macd < macd_signal and macd_hist < 0 and rsi <= 55
		if not (long_setup or short_setup):
			return None
		
		# Confidence
		conf = 50 + min(20, abs(macd - macd_signal) * 10) + min(10, abs(macd_hist) * 10)
		bias = 'BUY' if long_setup else 'SELL'
		
		# Risk/target
		stop_loss = 1.8 * atr
		target = 2.4 * atr
		slippage = price * 0.0003
		
		# Walk forward
		entry = price
		pnl = 0.0
		outcome = 'Pending'
		targets_hit = 0
		stoploss_count = 0
		for _, frow in future_data.iterrows():
			fhigh = float(frow['high'])
			flow = float(frow['low'])
			if bias == 'BUY':
				if fhigh >= entry + target:
					pnl = max(0.0, target - slippage)
					outcome = 'Win'
					targets_hit = 1
					break
				if flow <= entry - stop_loss:
					pnl = -max(0.0, stop_loss + slippage)
					outcome = 'Loss'
					stoploss_count = 1
					break
			else:
				if flow <= entry - target:
					pnl = max(0.0, target - slippage)
					outcome = 'Win'
					targets_hit = 1
					break
				if fhigh >= entry + stop_loss:
					pnl = -max(0.0, stop_loss + slippage)
					outcome = 'Loss'
					stoploss_count = 1
					break
		
		return {
			'signal': bias,
			'price': entry,
			'confidence': 'High' if conf >= 70 else ('Medium' if conf >= 60 else 'Low'),
			'confidence_score': int(conf),
			'stop_loss': stop_loss,
			'target': target,
			'outcome': outcome,
			'pnl': pnl,
			'targets_hit': targets_hit,
			'stoploss_count': stoploss_count,
			'reasoning': 'MACD cross with RSI regime'
		} 