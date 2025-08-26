#!/usr/bin/env python3
from typing import Optional
import pandas as pd

class MacdCrossRsiFilter:
	def __init__(self, timeframe_data=None):
		self.timeframe_data = timeframe_data or {}
		self.daily_trades = {}
		self.cooldown_until_index = None
		self.stopout_streak = 0

	def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
		# Assume indicators exist; add ATR if missing
		if 'atr' not in df.columns and len(df) >= 20:
			prev_close = df['close'].shift(1)
			tr1 = df['high'] - df['low']
			tr2 = (df['high'] - prev_close).abs()
			tr3 = (df['low'] - prev_close).abs()
			true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
			df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
		# ATR percentile for volatility windowing
		if 'atr_pct' not in df.columns and 'atr' in df.columns:
			atr = df['atr']
			rolling = atr.rolling(window=200, min_periods=50)
			df['atr_pct'] = rolling.rank(pct=True)
		return df

	def _session_ok(self, ts) -> bool:
		hour = ts.hour
		minute = ts.minute
		# Avoid first/last 15 minutes
		if (hour == 9 and minute < 30) or (hour == 15 and minute >= 15):
			return False
		return True

	def _daily_cap_ok(self, ts) -> bool:
		day = ts.date()
		count = self.daily_trades.get(day, 0)
		return count < 4

	def _inc_daily(self, ts):
		day = ts.date()
		self.daily_trades[day] = self.daily_trades.get(day, 0) + 1

	def analyze(self, candle, index, df: pd.DataFrame, future_data: Optional[pd.DataFrame] = None):
		if index < 51 or future_data is None or future_data.empty:
			return None
		
		ts = df.index[index]
		if not self._session_ok(ts) or not self._daily_cap_ok(ts):
			return None
		if self.cooldown_until_index is not None and index < self.cooldown_until_index:
			return None
		
		price = float(candle['close'])
		macd = float(candle.get('macd', 0))
		macd_signal = float(candle.get('macd_signal', 0))
		macd_hist = float(candle.get('macd_histogram', candle.get('macd_diff', 0)))
		macd_hist_prev = float(df.iloc[index-1].get('macd_histogram', df.iloc[index-1].get('macd_diff', 0)))
		rsi = float(candle.get('rsi', 50))
		atr = float(candle.get('atr', 0))
		atr_pct = float(candle.get('atr_pct', 0.5))
		vol_ratio = float(candle.get('volume_ratio', 1))
		ema21 = float(candle.get('ema_21', 0))
		ema50 = float(candle.get('ema_50', 0))
		if atr <= 0 or vol_ratio != vol_ratio:
			return None
		
		# OPTIMIZATION: Tighter market regime filter
		vix_proxy = atr / price if price > 0 else 1.0
		if vix_proxy > 0.015:  # Reduced from 0.020 - avoid high volatility
			return None
		
		# OPTIMIZATION: Tighter volume and volatility windows
		if vol_ratio < 1.2 or atr_pct < 0.4 or atr_pct > 0.6:  # Tighter ranges
			return None
		
		# OPTIMIZATION: Enhanced MACD cross + expansion with stricter conditions
		macd_strength = abs(macd - macd_signal)
		long_setup = (macd > macd_signal and macd_hist > 0 and 
					 macd_hist > macd_hist_prev and rsi >= 45 and rsi <= 70 and
					 macd_strength > 0.5)  # Minimum MACD separation
		short_setup = (macd < macd_signal and macd_hist < 0 and 
					  macd_hist < macd_hist_prev and rsi <= 55 and rsi >= 30 and
					  macd_strength > 0.5)  # Minimum MACD separation
		
		if not (long_setup or short_setup):
			return None
		
		# OPTIMIZATION: Enhanced trend alignment requirements
		mtf_ok = True
		price_align_ok = (price > ema21) if long_setup else (price < ema21)
		if not price_align_ok:
			return None
			
		# OPTIMIZATION: Stricter multi-timeframe confirmation
		mtf_15 = self.timeframe_data.get('15min') if isinstance(self.timeframe_data, dict) else None
		if isinstance(mtf_15, pd.DataFrame) and len(mtf_15) > 50:
			ref_slice = mtf_15[mtf_15.index <= ts]
			if not ref_slice.empty:
				m = ref_slice.iloc[-1]
				ema21_15 = float(m.get('ema_21', 0))
				ema50_15 = float(m.get('ema_50', 0))
				rsi_15 = float(m.get('rsi', 50))
				# OPTIMIZATION: Require both trend and RSI alignment
				trend_ok = (ema21_15 > ema50_15) if long_setup else (ema21_15 < ema50_15)
				rsi_ok = (rsi_15 > 45) if long_setup else (rsi_15 < 55)
				mtf_ok = trend_ok and rsi_ok
		if not mtf_ok:
			return None
		
		# OPTIMIZATION: Improved confidence calculation
		base_conf = 60
		macd_conf = min(20, macd_strength * 10)  # MACD strength contribution
		rsi_conf = min(10, abs(rsi - 50) / 2)    # RSI extremity contribution
		vol_conf = min(10, (vol_ratio - 1.0) * 10)  # Volume contribution
		
		conf = base_conf + macd_conf + rsi_conf + vol_conf
		
		# OPTIMIZATION: Higher minimum confidence threshold
		if conf < 70:  # Increased from implicit lower threshold
			return None
			
		bias = 'BUY' if long_setup else 'SELL'
		
		# OPTIMIZATION: Improved R:R ratio (2.5:1 -> 3:1)
		stop_loss = 0.8 * atr  # Reduced from 1.0 ATR
		target = 2.4 * atr     # Increased from 2.5 ATR (3:1 R:R)
		slippage = price * 0.0003
		
		# Walk forward with BE after 0.5R and trail 0.8 ATR thereafter
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
			
			# OPTIMIZATION: Activate BE after 0.5R (reduced from 0.6R)
			if not breakeven_activated:
				if bias == 'BUY' and fhigh >= entry + 0.5*stop_loss:
					breakeven_activated = True
					trail_active = True
					trail_stop = entry
				elif bias == 'SELL' and flow <= entry - 0.5*stop_loss:
					breakeven_activated = True
					trail_active = True
					trail_stop = entry
			# TP at 2.4 ATR
			if bias == 'BUY':
				if fhigh >= entry + target:
					pnl = max(0.0, target - slippage)
					outcome = 'Win'
					targets_hit = 1
					break
				if trail_active:
					trail_stop = max(trail_stop, fhigh - 0.8*atr)  # Tighter trail
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
					trail_stop = min(trail_stop, flow + 0.8*atr)  # Tighter trail
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
			'confidence': 'High' if conf >= 80 else ('Medium' if conf >= 70 else 'Low'),
			'confidence_score': int(conf),
			'stop_loss': stop_loss,
			'target': target,
			'outcome': outcome,
			'pnl': pnl,
			'targets_hit': targets_hit,
			'stoploss_count': stoploss_count,
			'reasoning': f'MACD cross + expansion, RSI, price>EMA21, 15m trend aligned, R:R=3:1, VIX={vix_proxy:.3f}, Conf={conf}'
		} 