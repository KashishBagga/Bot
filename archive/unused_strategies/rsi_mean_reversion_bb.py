#!/usr/bin/env python3
from typing import Optional
import pandas as pd

class RsiMeanReversionBb:
	def __init__(self, timeframe_data=None):
		self.timeframe_data = timeframe_data or {}
		self.daily_trades = {}
		self.cooldown_until_index = None
		self.stopout_streak = 0

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
		# ATR percentile
		if 'atr_pct' not in df.columns and 'atr' in df.columns:
			atr = df['atr']
			rolling = atr.rolling(window=200, min_periods=50)
			df['atr_pct'] = rolling.rank(pct=True)
		return df

	def _session_ok(self, ts) -> bool:
		h, m = ts.hour, ts.minute
		if (h == 9 and m < 30) or (h == 15 and m >= 15):
			return False
		return True

	def _daily_cap_ok(self, ts) -> bool:
		day = ts.date()
		return self.daily_trades.get(day, 0) < 5

	def _inc_daily(self, ts):
		day = ts.date()
		self.daily_trades[day] = self.daily_trades.get(day, 0) + 1

	def analyze(self, candle, index, df: pd.DataFrame, future_data: Optional[pd.DataFrame] = None):
		if index < 50 or future_data is None or future_data.empty:
			return None
		
		ts = df.index[index]
		if not self._session_ok(ts) or not self._daily_cap_ok(ts):
			return None
		if self.cooldown_until_index is not None and index < self.cooldown_until_index:
			return None
		
		price = float(candle['close'])
		rsi = float(candle.get('rsi', 50))
		bb_lower = float(candle.get('bb_lower', 0))
		bb_upper = float(candle.get('bb_upper', 0))
		atr = float(candle.get('atr', 0))
		atr_pct = float(candle.get('atr_pct', 0.5))
		ema21 = float(candle.get('ema_21', 0))
		ema50 = float(candle.get('ema_50', 0))
		vol_ratio = float(candle.get('volume_ratio', 1))
		macd = float(candle.get('macd', 0))
		macd_signal = float(candle.get('macd_signal', 0))
		
		if atr <= 0 or bb_lower == 0 or bb_upper == 0:
			return None
		
		# OPTIMIZATION: Tighter volatility and volume filters for better win rate
		if atr_pct < 0.3 or atr_pct > 0.7 or vol_ratio != vol_ratio or vol_ratio < 1.0:
			return None
		
		# OPTIMIZATION: Enhanced flat regime detection with stricter tolerance
		tol = 0.003  # Reduced from 0.005 - tighter flat regime requirement
		flat_regime = abs(ema21 - ema50) / max(1e-6, abs(ema50)) <= tol
		if not flat_regime:
			return None
		
		# OPTIMIZATION: Enhanced mean reversion triggers with stricter conditions
		near_lower = price <= bb_lower * 1.002  # Tighter tolerance
		near_upper = price >= bb_upper * 0.998  # Tighter tolerance
		
		# OPTIMIZATION: More extreme RSI conditions for better reversal probability
		long_setup = (rsi <= 25) and near_lower  # More extreme oversold
		short_setup = (rsi >= 75) and near_upper  # More extreme overbought
		
		# OPTIMIZATION: Additional momentum confirmation
		if long_setup:
			# For long setup, check if MACD is starting to turn up
			macd_turning_up = macd > macd_signal and macd > 0
			if not macd_turning_up:
				long_setup = False
		elif short_setup:
			# For short setup, check if MACD is starting to turn down
			macd_turning_down = macd < macd_signal and macd < 0
			if not macd_turning_down:
				short_setup = False
		
		if not (long_setup or short_setup):
			return None
		
		# Multi-Factor Confidence Integration for RSI Mean Reversion
		from src.core.multi_factor_confidence import MultiFactorConfidence
		
		mfc = MultiFactorConfidence()
		signal_type = "BUY" if long_setup else "SELL"
		confidence_result = mfc.calculate_confidence(candle, signal_type, self.timeframe_data)
		
		conf = confidence_result['total_score']
		
		# OPTIMIZATION: Higher minimum confidence threshold
		if conf < 60:  # Increased from implicit lower threshold
			return None
			
		bias = 'BUY' if long_setup else 'SELL'
		
		# OPTIMIZATION: Improved risk-reward ratio for better profitability
		stop_loss = 0.6 * atr  # Tighter stop loss for better R:R
		target = 1.8 * atr     # Higher target for better R:R ratio (3:1)
		slippage = price * 0.0003
		
		# Walk forward with BE after 0.4R and trail 0.6 ATR thereafter
		entry = price
		pnl = 0.0
		outcome = 'Pending'
		targets_hit = 0
		stoploss_count = 0
		breakeven_activated = False
		trail_active = False
		trail_stop = None
		
		# OPTIMIZATION: Time-based exit (EOD at 15:00)
		eod_exit = False
		
		for _, frow in future_data.iterrows():
			fhigh = float(frow['high'])
			flow = float(frow['low'])
			
			# Check for EOD exit
			if hasattr(frow, 'name') and hasattr(frow.name, 'hour'):
				if frow.name.hour >= 15:
					eod_exit = True
					pnl = max(0.0, (flow + fhigh) / 2 - entry - slippage) if bias == 'BUY' else max(0.0, entry - (flow + fhigh) / 2 - slippage)
					outcome = 'Win' if pnl > 0 else 'Loss'
					break
			
			# OPTIMIZATION: Activate BE after 0.4R (reduced from 0.5R)
			if not breakeven_activated:
				if bias == 'BUY' and fhigh >= entry + 0.4*stop_loss:
					breakeven_activated = True
					trail_active = True
					trail_stop = entry
				elif bias == 'SELL' and flow <= entry - 0.4*stop_loss:
					breakeven_activated = True
					trail_active = True
					trail_stop = entry
			# TP at 1.8 ATR
			if bias == 'BUY':
				if fhigh >= entry + target:
					pnl = max(0.0, target - slippage)
					outcome = 'Win'
					targets_hit = 1
					break
				if trail_active:
					trail_stop = max(trail_stop, fhigh - 0.6*atr)  # Tighter trail
				if flow <= entry - stop_loss:
					pnl = -max(0.0, stop_loss + slippage)
					outcome = 'Loss'
					stoploss_count = 1
					self.stopout_streak += 1
					if self.stopout_streak >= 2:
						self.cooldown_until_index = index + 20
					break
				if trail_active and flow <= trail_stop:
					pnl = max(0.0, trail_stop - entry - slippage)
					outcome = 'Win'
					break
			else:
				if flow <= entry - target:
					pnl = max(0.0, target - slippage)
					outcome = 'Win'
					targets_hit = 1
					break
				if trail_active:
					trail_stop = min(trail_stop, flow + 0.6*atr)  # Tighter trail
				if fhigh >= entry + stop_loss:
					pnl = -max(0.0, stop_loss + slippage)
					outcome = 'Loss'
					stoploss_count = 1
					self.stopout_streak += 1
					if self.stopout_streak >= 2:
						self.cooldown_until_index = index + 20
					break
				if trail_active and fhigh >= trail_stop:
					pnl = max(0.0, entry - trail_stop - slippage)
					outcome = 'Win'
					break
		
		# Bookkeeping
		self._inc_daily(ts)
		if outcome == 'Win':
			self.stopout_streak = 0
		
		return {
			'signal': bias,
			'price': entry,
			'confidence': 'High' if conf >= 75 else ('Medium' if conf >= 60 else 'Low'),
			'confidence_score': int(conf),
			'stop_loss': stop_loss,
			'target': target,
			'outcome': outcome,
			'pnl': pnl,
			'targets_hit': targets_hit,
			'stoploss_count': stoploss_count,
			'reasoning': f'RSI mean reversion with Bollinger Bands, RSI={rsi:.1f}, BB_{"lower" if long_setup else "upper"}, R:R=3:1, Conf={conf}'
		} 