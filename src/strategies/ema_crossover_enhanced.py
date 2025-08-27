import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from src.core.indicators import indicators
import logging

class EmaCrossoverEnhanced(Strategy):
    """
    Enhanced EMA Crossover Strategy with:
    - Higher timeframe trend filter (15m EMA direction)
    - Wider EMAs (20/50, 50/200) to reduce noise
    - ATR-based filters and stop-loss
    - Trailing stop-loss
    """

    # Minimum candles required for analysis
    min_candles = 100

    def __init__(self, params: Dict[str, Any] = None):
        params = params or {}
        self.ema_short = params.get("ema_short", 20)  # Wider EMAs
        self.ema_long = params.get("ema_long", 50)
        self.ema_trend = params.get("ema_trend", 200)  # Trend filter
        self.atr_period = params.get("atr_period", 14)
        self.atr_threshold = params.get("atr_threshold", 0.3)  # Reduced ATR threshold for more signals
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 15)  # Reduced ADX threshold for more signals
        self.min_confidence_threshold = params.get("min_confidence_threshold", 40)  # Reduced confidence threshold
        self.volume_threshold = params.get("volume_threshold", 0.3)  # Volume confirmation
        self.rsi_period = params.get("rsi_period", 14)
        self.ema_slope_period = params.get("ema_slope_period", 5)  # EMA slope for momentum
        super().__init__("ema_crossover_enhanced", params)

    def calculate_confidence_score(self, df: pd.DataFrame, mask: pd.Series, signal_type: str = 'bullish') -> pd.Series:
        """
        Unified confidence scoring system for EMA crossover strategy.
        Returns confidence scores for the given mask.
        """
        confidence_scores = pd.Series(0, index=df.index)
        
        if signal_type == 'bullish':
            # Trend alignment (35% weight)
            trend_aligned = df['close'] > df['ema_trend']
            confidence_scores.loc[mask & trend_aligned] += 35
            
            moderate_trend = (df['close'] > df['ema_long']) & ~trend_aligned
            confidence_scores.loc[mask & moderate_trend] += 20
            
            # ATR filter (15% weight)
            confidence_scores.loc[mask] += 15
            
            # ADX filter (15% weight)
            confidence_scores.loc[mask] += 15
            
            # RSI confirmation (10% weight)
            rsi_confirm = df['rsi'] > 45
            confidence_scores.loc[mask & rsi_confirm] += 10
            
            # Volume confirmation (10% weight)
            volume_spike = df['volume_ratio_ma'] > 1.2
            volume_above_avg = df['volume_ratio_ma'] > 1.0
            confidence_scores.loc[mask & volume_spike] += 10
            confidence_scores.loc[mask & volume_above_avg & ~volume_spike] += 5
            
            # EMA slope momentum (5% weight)
            momentum_confirm = df['ema_slope'] > 0
            confidence_scores.loc[mask & momentum_confirm] += 5
            
        elif signal_type == 'bearish':
            # Trend alignment (35% weight)
            trend_aligned = df['close'] < df['ema_trend']
            confidence_scores.loc[mask & trend_aligned] += 35
            
            moderate_trend = (df['close'] < df['ema_long']) & ~trend_aligned
            confidence_scores.loc[mask & moderate_trend] += 20
            
            # ATR filter (15% weight)
            confidence_scores.loc[mask] += 15
            
            # ADX filter (15% weight)
            confidence_scores.loc[mask] += 15
            
            # RSI confirmation (10% weight)
            rsi_confirm = df['rsi'] < 55
            confidence_scores.loc[mask & rsi_confirm] += 10
            
            # Volume confirmation (10% weight)
            volume_spike = df['volume_ratio_ma'] > 1.2
            volume_above_avg = df['volume_ratio_ma'] > 1.0
            confidence_scores.loc[mask & volume_spike] += 10
            confidence_scores.loc[mask & volume_above_avg & ~volume_spike] += 5
            
            # EMA slope momentum (5% weight)
            momentum_confirm = df['ema_slope'] < 0
            confidence_scores.loc[mask & momentum_confirm] += 5
        
        return confidence_scores

    def _ensure_emas(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Ensure EMA columns exist. Returns a DataFrame with 'ema_short' & 'ema_long'.
        If inplace=False, returns a copy (safe). If inplace=True, modifies provided df.
        """
        need_copy = not inplace
        if 'ema_short' in df.columns and 'ema_long' in df.columns:
            return df if not need_copy else df.copy()
        out = df if inplace else df.copy()
        out['ema_short'] = out['close'].ewm(span=self.ema_short, adjust=False).mean()
        out['ema_long'] = out['close'].ewm(span=self.ema_long, adjust=False).mean()
        return out

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators."""
        df = df.copy()
        
        # Add EMAs
        df['ema_short'] = indicators.ema(df, period=self.ema_short)
        df['ema_long'] = indicators.ema(df, period=self.ema_long)
        df['ema_trend'] = indicators.ema(df, period=self.ema_trend)
        
        # Add ATR for volatility filter
        df['atr'] = indicators.atr(df, period=self.atr_period)
        
        # Add ADX for trend strength
        df['adx'] = indicators.adx(df, period=self.adx_period)
        
        # Add RSI for additional confirmation
        df['rsi'] = indicators.rsi(df, period=14)
        
        # Add volume ratio
        df['volume_ratio'] = df['volume'].pct_change().fillna(0)
        
        # Add EMA slope for momentum
        df['ema_slope'] = df['ema_short'].diff(self.ema_slope_period).fillna(0)
        
        # Add volume moving average
        df['volume_ma'] = df['volume'].rolling(window=20).mean().fillna(0)
        df['volume_ratio_ma'] = (df['volume'] / df['volume_ma']).fillna(1)
        
        return df

    def detect_crossover(self, df: pd.DataFrame, short_col: str, long_col: str) -> Dict[str, Any]:
        """Detect EMA crossover using closed candles."""
        if len(df) < 3:
            return {'crossover': False, 'direction': 0}
        
        # Use closed candles for crossover detection
        prev = df.iloc[-2]  # Last closed candle
        prev_prev = df.iloc[-3]  # Previous closed candle
        
        # Check for crossover
        cross_up = (prev[short_col] > prev[long_col] and 
                   prev_prev[short_col] <= prev_prev[long_col])
        
        cross_down = (prev[short_col] < prev[long_col] and 
                     prev_prev[short_col] >= prev_prev[long_col])
        
        if cross_up:
            return {'crossover': True, 'direction': 1}
        elif cross_down:
            return {'crossover': True, 'direction': -1}
        else:
            return {'crossover': False, 'direction': 0}

    # -----------------------
    # FAST: vectorized API
    # -----------------------
    def analyze_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized detection of EMA crossovers with enhanced filters.
        Returns a DataFrame of signals (one row per signal) with columns:
        ['timestamp', 'signal', 'price', 'confidence_score', 'stop_loss', 'target1', 'target2', 'target3', 'position_multiplier', 'reasoning']
        Signal values: 'BUY CALL' (bullish crossover), 'BUY PUT' (bearish crossover)
        """
        # Ensure we have enough data
        if len(df) < self.min_candles:
            return pd.DataFrame()
        
        # Ensure indicators are present
        if 'ema_short' not in df.columns:
            df = self.add_indicators(df)
        
        # look-ahead safe crossover detection
        buy_mask = (df['ema_short'] > df['ema_long']) & (
            df['ema_short'].shift(1) <= df['ema_long'].shift(1)
        )
        sell_mask = (df['ema_short'] < df['ema_long']) & (
            df['ema_short'].shift(1) >= df['ema_long'].shift(1)
        )

        # Apply filters
        atr_filter = df['atr'] > self.atr_threshold
        adx_filter = df['adx'] > self.adx_threshold
        volume_filter = df['volume_ratio_ma'] > 0.8  # Reduced volume threshold for more signals
        
        # RSI filters - more flexible
        bullish_rsi = df['rsi'] > 40  # Reduced from 45
        bearish_rsi = df['rsi'] < 60  # Increased from 55
        
        # EMA slope momentum
        bullish_momentum = df['ema_slope'] > -0.1  # More flexible momentum
        bearish_momentum = df['ema_slope'] < 0.1   # More flexible momentum
        
        # Trend alignment - more flexible
        bullish_trend = df['close'] > df['ema_trend'] * 0.995  # 0.5% tolerance
        bearish_trend = df['close'] < df['ema_trend'] * 1.005  # 0.5% tolerance
        
        # Calculate confidence scores using unified system
        confidence_scores = pd.Series(0, index=df.index)
        
        # For bullish signals
        bullish_mask = buy_mask & atr_filter & adx_filter & bullish_rsi & volume_filter & bullish_momentum
        bullish_confidence = self.calculate_confidence_score(df, bullish_mask, 'bullish')
        confidence_scores += bullish_confidence
        
        # For bearish signals
        bearish_mask = sell_mask & atr_filter & adx_filter & bearish_rsi & volume_filter & bearish_momentum
        bearish_confidence = self.calculate_confidence_score(df, bearish_mask, 'bearish')
        confidence_scores += bearish_confidence
        
        # Apply confidence threshold
        valid_signals = confidence_scores >= self.min_confidence_threshold
        
        # Generate final signal mask
        signal_mask = (bullish_mask | bearish_mask) & valid_signals
        
        if not signal_mask.any():
            # no signals
            return pd.DataFrame()

        signals_df = df.loc[signal_mask, ['timestamp', 'close', 'ema_short', 'ema_long', 'atr', 'adx', 'rsi']].copy()
        
        # label signals to match your backtester expectations
        signals_df['signal'] = np.where(bullish_mask.loc[signals_df.index], 'BUY CALL', 'BUY PUT')
        signals_df['price'] = signals_df['close']
        signals_df['confidence_score'] = confidence_scores[signal_mask]
        
        # Calculate stop loss and targets (ATR-based)
        signals_df['stop_loss'] = 1.5 * signals_df['atr']
        signals_df['target1'] = 2.0 * signals_df['atr']
        signals_df['target2'] = 3.0 * signals_df['atr']
        signals_df['target3'] = 4.0 * signals_df['atr']
        
                # Dynamic position sizing
        high_confidence = signals_df['confidence_score'] >= 80
        signals_df['position_multiplier'] = np.where(high_confidence, 1.0, 0.8)
        
        # Add reasoning - properly formatted for each row
        signals_df['reasoning'] = signals_df.apply(
            lambda row: f"EMA crossover, ATR {row['atr']:.2f}, ADX {row['adx']:.2f}, RSI {row['rsi']:.1f}",
            axis=1
        )

        return signals_df[['timestamp', 'signal', 'price', 'confidence_score', 'stop_loss', 'target1', 'target2', 'target3', 'position_multiplier', 'reasoning']]

    # -----------------------
    # Compatibility: single-row API (uses precomputed EMAs)
    # -----------------------
    def analyze_row(self, idx: int, row: pd.Series, df: pd.DataFrame) -> Optional[dict]:
        """
        Backwards-compatible single-row analyzer.
        Assumes EMAs are precomputed in df (or will compute them once).
        Returns a dict like {'signal': 'BUY CALL', 'price': ..., 'confidence_score': ...} or None.
        Important: For performance, call _ensure_emas(df) once outside when using this in a loop.
        """
        # ensure EMAs exist (compute once per run ideally)
        df_ema = self._ensure_emas(df, inplace=False)

        if idx <= 0 or idx >= len(df_ema):
            return None

        cur_s = float(df_ema['ema_short'].iat[idx])
        cur_l = float(df_ema['ema_long'].iat[idx])
        prev_s = float(df_ema['ema_short'].iat[idx-1])
        prev_l = float(df_ema['ema_long'].iat[idx-1])

        # same conditions as vectorized masks
        if (cur_s > cur_l) and (prev_s <= prev_l):
            confidence = abs((cur_s - cur_l) / (abs(cur_l) + 1e-9))
            return {'signal': 'BUY CALL', 'price': float(df_ema['close'].iat[idx]), 'confidence_score': float(confidence)}
        if (cur_s < cur_l) and (prev_s >= prev_l):
            confidence = abs((cur_s - cur_l) / (abs(cur_l) + 1e-9))
            return {'signal': 'BUY PUT', 'price': float(df_ema['close'].iat[idx]), 'confidence_score': float(confidence)}
        return None

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and generate trading signals using enhanced filters."""
        if not self.validate_data(data):
            return {'signal': 'NO TRADE', 'reason': 'insufficient data'}

        try:
            # Use closed candle for analysis
            candle = self.get_closed_candle(data)
            
            # Add indicators
            df = self.calculate_indicators(data)
            
            # Get indicator values from closed candle
            ema_short = df['ema_short'].iloc[-1]
            ema_long = df['ema_long'].iloc[-1]
            ema_trend = df['ema_trend'].iloc[-1]
            atr = df['atr'].iloc[-1]
            adx = df['adx'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            
            # Check for NaN values
            if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(atr):
                return {'signal': 'NO TRADE', 'reason': 'indicator data unavailable'}
            
            # Detect crossover
            crossover = self.detect_crossover(df, 'ema_short', 'ema_long')
            
            if not crossover['crossover']:
                return {'signal': 'NO TRADE', 'reason': 'no EMA crossover detected'}
            
            # Calculate confidence score
            confidence_score = 0
            
            # Trend alignment (40% weight)
            if crossover['direction'] == 1:  # Bullish crossover
                if candle['close'] > ema_trend:
                    confidence_score += 40  # Strong trend alignment
                elif candle['close'] > ema_long:
                    confidence_score += 25  # Moderate trend alignment
                else:
                    confidence_score += 10  # Weak trend alignment
            else:  # Bearish crossover
                if candle['close'] < ema_trend:
                    confidence_score += 40
                elif candle['close'] < ema_long:
                    confidence_score += 25
                else:
                    confidence_score += 10
            
            # ATR filter (20% weight)
            if atr > self.atr_threshold:
                confidence_score += 20
            else:
                return {'signal': 'NO TRADE', 'reason': f'ATR too low: {atr:.2f} < {self.atr_threshold}'}
            
            # ADX filter (20% weight)
            if adx > self.adx_threshold:
                confidence_score += 20
            else:
                return {'signal': 'NO TRADE', 'reason': f'ADX too low: {adx:.2f} < {self.adx_threshold}'}
            
            # RSI confirmation (10% weight)
            if crossover['direction'] == 1 and rsi > 45:  # Bullish with RSI support
                confidence_score += 10
            elif crossover['direction'] == -1 and rsi < 55:  # Bearish with RSI support
                confidence_score += 10
            
            # Volume confirmation (10% weight)
            volume_ma = df['volume_ma'].iloc[-1]
            volume_ratio_ma = df['volume_ratio_ma'].iloc[-1]
            ema_slope = df['ema_slope'].iloc[-1]
            
            if volume_ratio_ma > 1.2:  # Volume spike
                confidence_score += 10
            elif volume_ratio_ma > 1.0:  # Above average volume
                confidence_score += 5
            
            # EMA slope momentum (5% weight)
            if crossover['direction'] == 1 and ema_slope > 0:  # Bullish momentum
                confidence_score += 5
            elif crossover['direction'] == -1 and ema_slope < 0:  # Bearish momentum
                confidence_score += 5
            
            # Generate signals based on crossover direction
            if crossover['direction'] == 1 and confidence_score >= self.min_confidence_threshold:
                # BUY CALL signal
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
                    'reasoning': f"EMA crossover bullish, Trend aligned, ATR {atr:.2f}, ADX {adx:.2f}, RSI {rsi:.1f}"
                }
            
            elif crossover['direction'] == -1 and confidence_score >= self.min_confidence_threshold:
                # BUY PUT signal
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
                    'reasoning': f"EMA crossover bearish, Trend aligned, ATR {atr:.2f}, ADX {adx:.2f}, RSI {rsi:.1f}"
                }
            
            return {'signal': 'NO TRADE', 'reason': f'confidence too low: {confidence_score} < {self.min_confidence_threshold}'}
            
        except Exception as e:
            logging.error(f"Error in EmaCrossoverEnhanced analysis: {e}")
            return {'signal': 'ERROR', 'reason': str(e)}

# -----------------------
# Test harness to validate parity
# -----------------------
def compare_row_vs_vectorized(strategy: EmaCrossoverEnhanced, df: pd.DataFrame, verbose: bool = True):
    """
    Compare signals produced by analyze_row (row-by-row) vs analyze_vectorized.
    Returns (identical:bool, details:dict)
    """
    # Precompute EMAs once
    df_ema = strategy._ensure_emas(df, inplace=False)

    # collect row-wise signals using analyze_row (fast now since EMAs are precomputed)
    row_signals = []
    for i in range(1, len(df_ema)):
        res = strategy.analyze_row(i, df_ema.iloc[i], df_ema)
        if res and res.get('signal'):
            row_signals.append({
                'timestamp': pd.to_datetime(df_ema['timestamp'].iat[i]),
                'signal': res['signal'],
                'price': res.get('price')
            })
    row_df = pd.DataFrame(row_signals)

    # collect vectorized signals
    vec_df = strategy.analyze_vectorized(df)

    # normalize timestamps and compare sets
    if row_df.empty and vec_df.empty:
        if verbose:
            print("Both methods produced zero signals — identical.")
        return True, {'row_count': 0, 'vec_count': 0, 'diff': 0, 'mismatches': []}

    # ensure timestamps are datetime64
    if not row_df.empty:
        row_df['timestamp'] = pd.to_datetime(row_df['timestamp'])
    if not vec_df.empty:
        vec_df['timestamp'] = pd.to_datetime(vec_df['timestamp'])

    # Merge on timestamp to find differences
    if row_df.empty:
        merged = vec_df.merge(pd.DataFrame(columns=['timestamp','signal','price']), on='timestamp', how='left', indicator=True)
    elif vec_df.empty:
        merged = row_df.merge(pd.DataFrame(columns=['timestamp','signal','price']), on='timestamp', how='left', indicator=True)
    else:
        merged = row_df.merge(vec_df[['timestamp','signal','price']], on='timestamp', suffixes=('_row', '_vec'), how='outer', indicator=True)

    # find mismatches where signals differ or recording differs
    mismatches = []
    if not merged.empty:
        for _, r in merged.iterrows():
            ts = r['timestamp']
            row_sig = r.get('signal_row') if 'signal_row' in r else r.get('signal') if '_row' not in r else None
            vec_sig = r.get('signal_vec') if 'signal_vec' in r else None
            if row_sig != vec_sig:
                mismatches.append({'timestamp': ts, 'row': row_sig, 'vec': vec_sig})

    diff_count = len(mismatches)
    if verbose:
        print(f"Row signals: {len(row_df)} | Vectorized signals: {len(vec_df)} | Differences: {diff_count}")
        if diff_count:
            print("Sample mismatches (first 10):")
            for m in mismatches[:10]:
                print(m)

    details = {
        'row_count': len(row_df),
        'vec_count': len(vec_df),
        'diff': diff_count,
        'mismatches': mismatches
    }
    return diff_count == 0, details

# -----------------------
# Test harness to validate parity
# -----------------------
def compare_row_vs_vectorized(strategy: EmaCrossoverEnhanced, df: pd.DataFrame, verbose: bool = True):
    """
    Compare signals produced by analyze_row (row-by-row) vs analyze_vectorized.
    Returns (identical:bool, details:dict)
    """
    # Precompute EMAs once
    df_ema = strategy._ensure_emas(df, inplace=False)

    # collect row-wise signals using analyze_row (fast now since EMAs are precomputed)
    row_signals = []
    for i in range(1, len(df_ema)):
        res = strategy.analyze_row(i, df_ema.iloc[i], df_ema)
        if res and res.get('signal'):
            row_signals.append({
                'timestamp': pd.to_datetime(df_ema['timestamp'].iat[i]),
                'signal': res['signal'],
                'price': res.get('price')
            })
    row_df = pd.DataFrame(row_signals)

    # collect vectorized signals
    vec_df = strategy.analyze_vectorized(df)

    # normalize timestamps and compare sets
    if row_df.empty and vec_df.empty:
        if verbose:
            print("Both methods produced zero signals — identical.")
        return True, {'row_count': 0, 'vec_count': 0, 'diff': 0, 'mismatches': []}

    # ensure timestamps are datetime64
    if not row_df.empty:
        row_df['timestamp'] = pd.to_datetime(row_df['timestamp'])
    if not vec_df.empty:
        vec_df['timestamp'] = pd.to_datetime(vec_df['timestamp'])

    # Merge on timestamp to find differences
    if row_df.empty:
        merged = vec_df.merge(pd.DataFrame(columns=['timestamp','signal','price']), on='timestamp', how='left', indicator=True)
    elif vec_df.empty:
        merged = row_df.merge(pd.DataFrame(columns=['timestamp','signal','price']), on='timestamp', how='left', indicator=True)
    else:
        merged = row_df.merge(vec_df[['timestamp','signal','price']], on='timestamp', suffixes=('_row', '_vec'), how='outer', indicator=True)

    # find mismatches where signals differ or recording differs
    mismatches = []
    if not merged.empty:
        for _, r in merged.iterrows():
            ts = r['timestamp']
            row_sig = r.get('signal_row') if 'signal_row' in r else r.get('signal') if '_row' not in r else None
            vec_sig = r.get('signal_vec') if 'signal_vec' in r else None
            if row_sig != vec_sig:
                mismatches.append({'timestamp': ts, 'row': row_sig, 'vec': vec_sig})

    diff_count = len(mismatches)
    if verbose:
        print(f"Row signals: {len(row_df)} | Vectorized signals: {len(vec_df)} | Differences: {diff_count}")
        if diff_count:
            print("Sample mismatches (first 10):")
            for m in mismatches[:10]:
                print(m)

    details = {
        'row_count': len(row_df),
        'vec_count': len(vec_df),
        'diff': diff_count,
        'mismatches': mismatches
    }
    return diff_count == 0, details 