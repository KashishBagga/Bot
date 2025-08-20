#!/usr/bin/env python3
from typing import Optional
import pandas as pd

class RsiMeanReversionBb:
	def __init__(self, timeframe_data=None):
		self.timeframe_data = timeframe_data or {}

	def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
		# Ensure Bollinger present
		if 'bb_upper' not in df.columns and len(df) >= 20:
			sma_20 = df['close'].rolling(window=20).mean()
			std_20 = df['close'].rolling(window=20).std()
			df['bb_upper'] = sma_20 + (std_20 * 2)
			df['bb_lower'] = sma_20 - (std_20 * 2)
			df['bb_middle'] = sma_20
		# ATR
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
		rsi = float(candle.get('rsi', 50))
		bb_lower = float(candle.get('bb_lower', 0))
		bb_upper = float(candle.get('bb_upper', 0))
		atr = float(candle.get('atr', 0))
		if atr <= 0 or bb_lower == 0 or bb_upper == 0:
			return None
		
		# Mean reversion: RSI <= 30 near/below lower band -> long; RSI >= 70 near/above upper band -> short
		long_setup = (rsi <= 30) and (price <= bb_lower)
		short_setup = (rsi >= 70) and (price >= bb_upper)
		if not (long_setup or short_setup):
			return None
		
		conf = 60 + min(10, abs(50 - rsi) / 5)
		bias = 'BUY' if long_setup else 'SELL'
		stop_loss = 1.5 * atr
		target = 1.5 * atr
		slippage = price * 0.0003
		
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
			'reasoning': 'RSI mean reversion with Bollinger Bands'
		} 