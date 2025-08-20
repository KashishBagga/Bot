#!/usr/bin/env python3
from typing import Optional
import pandas as pd

class Supertrend_ema_experimental:
	def __init__(self, timeframe_data=None):
		self.timeframe_data = timeframe_data or {}

	def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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
		atr = float(candle.get('atr', 0))
		ema21 = float(candle.get('ema_21', 0))
		ema50 = float(candle.get('ema_50', 0))
		st = float(candle.get('supertrend', candle.get('supertrend_upper', 0)))
		st_dir = int(candle.get('supertrend_direction', 0))
		vol_ratio = float(candle.get('volume_ratio', 1))
		
		if atr <= 0 or vol_ratio < 0.6:
			return None
		
		# Flip + EMA alignment
		long_setup = (st_dir > 0) and (price > st) and (price > ema21 > ema50)
		short_setup = (st_dir < 0) and (price < st) and (price < ema21 < ema50)
		if not (long_setup or short_setup):
			return None
		
		conf = 55
		conf += min(15, abs((price - ema21) / max(1e-6, ema21)) * 100)
		conf += min(10, max(0, (vol_ratio - 1) * 10))
		conf = max(0, min(100, conf))
		
		# MTF confirmation (15m)
		mtf_ok = True
		mtf_15 = None
		for tf_key in self.timeframe_data or {}:
			if '15' in tf_key:
				mtf_15 = self.timeframe_data[tf_key]
				break
		if isinstance(mtf_15, pd.DataFrame) and len(mtf_15) > 50:
			ref_time = df.index[index]
			mtf_slice = mtf_15[mtf_15.index <= ref_time]
			if not mtf_slice.empty:
				mtf_row = mtf_slice.iloc[-1]
				mtf_ok = (mtf_row.get('ema_21', 0) > mtf_row.get('ema_50', 0)) if long_setup else (mtf_row.get('ema_21', 0) < mtf_row.get('ema_50', 0))
		
		if not mtf_ok:
			return None
		
		# Risk/targets
		atr_sl = 2.0 * atr
		atr_tp = 2.8 * atr
		if long_setup:
			signal = 'BUY'
			stop_loss = atr_sl
			target = atr_tp
		else:
			signal = 'SELL'
			stop_loss = atr_sl
			target = atr_tp
		
		# Walk forward
		pnl = 0.0
		outcome = 'Pending'
		targets_hit = 0
		stoploss_count = 0
		entry = price
		for _, frow in future_data.iterrows():
			fhigh = float(frow['high'])
			flow = float(frow['low'])
			if signal == 'BUY':
				if fhigh >= entry + target:
					pnl = target
					outcome = 'Win'
					targets_hit = 1
					break
				if flow <= entry - stop_loss:
					pnl = -stop_loss
					outcome = 'Loss'
					stoploss_count = 1
					break
			else:
				if flow <= entry - target:
					pnl = target
					outcome = 'Win'
					targets_hit = 1
					break
				if fhigh >= entry + stop_loss:
					pnl = -stop_loss
					outcome = 'Loss'
					stoploss_count = 1
					break
		
		return {
			'signal': signal,
			'price': entry,
			'confidence': 'High' if conf >= 70 else ('Medium' if conf >= 60 else 'Low'),
			'confidence_score': int(conf),
			'stop_loss': stop_loss,
			'target': target,
			'outcome': outcome,
			'pnl': pnl,
			'targets_hit': targets_hit,
			'stoploss_count': stoploss_count,
			'reasoning': f"ST flip + EMA align; MTF={mtf_ok}"
		} 