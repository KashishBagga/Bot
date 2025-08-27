import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from src.core.indicators import indicators
from src.models.unified_database import UnifiedDatabase
import math
import logging
import pytz

class SupertrendMacdRsiEma(Strategy):
    """
    Multi-timeframe Supertrend + MACD + RSI + EMA strategy with signal confirmation.
    """

    # Minimum candles required for analysis
    min_candles = 60

    def __init__(self, params: Dict[str, Any] = None):
        params = params or {}
        self.supertrend_period = params.get("supertrend_period", 10)
        self.supertrend_multiplier = params.get("supertrend_multiplier", 3.0)
        self.rsi_period = params.get("rsi_period", 14)
        self.ema_period = params.get("ema_period", 20)
        self.macd_fast = params.get("macd_fast", 12)
        self.macd_slow = params.get("macd_slow", 26)
        self.macd_signal = params.get("macd_signal", 9)
        self.min_confidence_threshold = params.get("min_confidence_threshold", 45)  # Slightly reduced from 50
        super().__init__("supertrend_macd_rsi_ema", params)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators."""
        df = df.copy()
        
        # Add SuperTrend indicator
        period = self.supertrend_period
        multiplier = self.supertrend_multiplier
        supertrend_data = indicators.supertrend(df, period=period, multiplier=multiplier)
        df['supertrend'] = supertrend_data['supertrend']
        df['supertrend_direction'] = supertrend_data['direction']
        
        # Add MACD indicator
        macd_data = indicators.macd(df, fast_period=self.macd_fast, slow_period=self.macd_slow, signal_period=self.macd_signal)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Add RSI indicator
        df['rsi'] = indicators.rsi(df, period=self.rsi_period)
        
        # Add EMA indicator
        df['ema'] = indicators.ema(df, period=self.ema_period)
        
        # Add body ratio calculations
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        df['body_ratio'] = df['body_ratio'].fillna(0)
        
        return df

    def analyze_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized analysis of the entire dataframe - much faster than row-by-row processing.
        Returns a DataFrame with signals for all candles that meet the criteria.
        """
        try:
            # Ensure we have enough data
            if len(df) < self.min_candles:
                return pd.DataFrame()
            
            # Create a copy to avoid modifying original
            df = df.copy()
            
            # Ensure indicators are present
            if 'supertrend' not in df.columns:
                df = self.add_indicators(df)
            
            # Initialize signals DataFrame
            signals = pd.DataFrame(index=df.index)
            signals['signal'] = 'NO TRADE'
            signals['price'] = df['close']
            signals['confidence_score'] = 0
            signals['stop_loss'] = 0.0
            signals['target1'] = 0.0
            signals['target2'] = 0.0
            signals['target3'] = 0.0
            signals['position_multiplier'] = 1.0
            signals['reasoning'] = ''
            
            # Detect Supertrend crossovers (look-ahead safe with shift)
            # Bullish: price crosses above Supertrend
            bullish_cross = (df['close'] > df['supertrend']) & (df['close'].shift(1) <= df['supertrend'].shift(1))
            
            # Bearish: price crosses below Supertrend
            bearish_cross = (df['close'] < df['supertrend']) & (df['close'].shift(1) >= df['supertrend'].shift(1))
            
            # MACD crossovers
            macd_bullish = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            macd_bearish = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            
            # RSI conditions - slightly more flexible
            rsi_bullish = (df['rsi'] > 35) & (df['rsi'] < 75)  # Widened from 30-70
            rsi_bearish = (df['rsi'] > 35) & (df['rsi'] < 75)  # Widened from 30-70
            
            # EMA trend alignment
            bullish_ema = df['close'] > df['ema']
            bearish_ema = df['close'] < df['ema']
            
            # Body ratio filter (strong candles) - slightly more flexible
            body_filter = df['body_ratio'] > 0.25  # Reduced from 0.3
            
            # Calculate confidence scores vectorized
            confidence_scores = pd.Series(0, index=df.index)
            
            # For bullish signals - slightly adjusted scoring
            bullish_mask = bullish_cross & macd_bullish & rsi_bullish & bullish_ema & body_filter
            confidence_scores.loc[bullish_mask] += 30  # Supertrend crossover
            confidence_scores.loc[bullish_mask] += 25  # MACD confirmation
            confidence_scores.loc[bullish_mask] += 20  # RSI confirmation
            confidence_scores.loc[bullish_mask] += 15  # EMA alignment
            confidence_scores.loc[bullish_mask] += 10  # Body ratio filter
            
            # For bearish signals - slightly adjusted scoring
            bearish_mask = bearish_cross & macd_bearish & rsi_bearish & bearish_ema & body_filter
            confidence_scores.loc[bearish_mask] += 30  # Supertrend crossover
            confidence_scores.loc[bearish_mask] += 25  # MACD confirmation
            confidence_scores.loc[bearish_mask] += 20  # RSI confirmation
            confidence_scores.loc[bearish_mask] += 15  # EMA alignment
            confidence_scores.loc[bearish_mask] += 10  # Body ratio filter
            
            # Apply confidence threshold
            valid_signals = confidence_scores >= self.min_confidence_threshold
            
            # Generate signals
            signals.loc[bullish_mask & valid_signals, 'signal'] = 'BUY CALL'
            signals.loc[bearish_mask & valid_signals, 'signal'] = 'BUY PUT'
            
            # Set confidence scores
            signals.loc[valid_signals, 'confidence_score'] = confidence_scores[valid_signals]
            
            # Calculate stop loss and targets (ATR-based, using 1% as default)
            atr_default = df['close'] * 0.01  # 1% of close price
            signals.loc[valid_signals, 'stop_loss'] = 1.5 * atr_default.loc[valid_signals]
            signals.loc[valid_signals, 'target1'] = 2.0 * atr_default.loc[valid_signals]
            signals.loc[valid_signals, 'target2'] = 3.0 * atr_default.loc[valid_signals]
            signals.loc[valid_signals, 'target3'] = 4.0 * atr_default.loc[valid_signals]
            
            # Dynamic position sizing
            high_confidence = confidence_scores >= 80
            signals.loc[valid_signals & high_confidence, 'position_multiplier'] = 1.0
            signals.loc[valid_signals & ~high_confidence, 'position_multiplier'] = 0.8
            
            # Add reasoning
            signals.loc[valid_signals, 'reasoning'] = (
                f"Supertrend+MACD+RSI+EMA, RSI {df.loc[valid_signals, 'rsi'].round(1)}, "
                f"MACD {df.loc[valid_signals, 'macd'].round(3)}"
            )
            
            # Return only valid signals
            valid_signals_df = signals[signals['signal'] != 'NO TRADE'].copy()
            
            return valid_signals_df
            
        except Exception as e:
            logging.error(f"Error in SupertrendMacdRsiEma vectorized analysis: {e}")
            return pd.DataFrame()

    def calculate_performance(self, signal: str, entry_price: float, stop_loss: float, 
                             target: float, target2: float, target3: float,
                             future_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for a trade signal.
        
        Args:
            signal: Trade signal ("BUY CALL" or "BUY PUT")
            entry_price: Entry price for the trade
            stop_loss: Stop loss amount (not price)
            target: First target amount
            target2: Second target amount  
            target3: Third target amount
            future_data: Future candles for performance tracking
            
        Returns:
            Dict containing outcome, pnl, targets_hit, stoploss_count, failure_reason, exit_time
        """
        outcome = "Pending"
        pnl = 0.0
        targets_hit = 0
        stoploss_count = 0
        failure_reason = ""
        exit_time = None
        
        if future_data is None or future_data.empty:
            return {
                "outcome": "Data Missing",
                "pnl": 0.0,
                "targets_hit": 0,
                "stoploss_count": 0,
                "failure_reason": "No future data available",
                "exit_time": None
            }
        
        if signal == "BUY CALL":
            stop_loss_price = entry_price - stop_loss
            target1_price = entry_price + target
            target2_price = entry_price + target2
            target3_price = entry_price + target3
            
            highest_price = entry_price
            trailing_sl = None
            target1_hit = target2_hit = target3_hit = False
            
            for i, future_candle in future_data.iterrows():
                # Check stop loss first
                if not target1_hit and future_candle['low'] <= stop_loss_price:
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    break
                
                # Check targets
                if not target1_hit and future_candle['high'] >= target1_price:
                    target1_hit = True
                    targets_hit = 1
                    pnl = target
                    highest_price = max(highest_price, future_candle['high'])
                    trailing_sl = highest_price - stop_loss
                    outcome = "Win"
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                    
                if target1_hit and not target2_hit and future_candle['high'] >= target2_price:
                    target2_hit = True
                    targets_hit = 2
                    pnl += (target2 - target)
                    highest_price = max(highest_price, future_candle['high'])
                    trailing_sl = max(trailing_sl, highest_price - stop_loss)
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                    
                if target2_hit and not target3_hit and future_candle['high'] >= target3_price:
                    target3_hit = True
                    targets_hit = 3
                    pnl += (target3 - target2)
                    highest_price = max(highest_price, future_candle['high'])
                    trailing_sl = max(trailing_sl, highest_price - stop_loss)
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                
                # Update trailing stop
                if target1_hit:
                    highest_price = max(highest_price, future_candle['high'])
                    trailing_sl = max(trailing_sl, highest_price - stop_loss)
                    if future_candle['low'] <= trailing_sl:
                        outcome = "Win"
                        pnl = trailing_sl - entry_price
                        failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                        exit_time = self.safe_signal_time(future_candle.get('time', i))
                        break
                        
        elif signal == "BUY PUT":
            stop_loss_price = entry_price + stop_loss
            target1_price = entry_price - target
            target2_price = entry_price - target2
            target3_price = entry_price - target3
            
            lowest_price = entry_price
            trailing_sl = None
            target1_hit = target2_hit = target3_hit = False
            
            for i, future_candle in future_data.iterrows():
                # Check stop loss first
                if not target1_hit and future_candle['high'] >= stop_loss_price:
                    outcome = "Loss"
                    pnl = -stop_loss
                    stoploss_count = 1
                    failure_reason = f"Stop loss hit at {stop_loss_price:.2f}"
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    break
                
                # Check targets
                if not target1_hit and future_candle['low'] <= target1_price:
                    target1_hit = True
                    targets_hit = 1
                    pnl = target
                    lowest_price = min(lowest_price, future_candle['low'])
                    trailing_sl = lowest_price + stop_loss
                    outcome = "Win"
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                    
                if target1_hit and not target2_hit and future_candle['low'] <= target2_price:
                    target2_hit = True
                    targets_hit = 2
                    pnl += (target2 - target)
                    lowest_price = min(lowest_price, future_candle['low'])
                    trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                    
                if target2_hit and not target3_hit and future_candle['low'] <= target3_price:
                    target3_hit = True
                    targets_hit = 3
                    pnl += (target3 - target2)
                    lowest_price = min(lowest_price, future_candle['low'])
                    trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                    exit_time = self.safe_signal_time(future_candle.get('time', i))
                    continue
                
                # Update trailing stop
                if target1_hit:
                    lowest_price = min(lowest_price, future_candle['low'])
                    trailing_sl = min(trailing_sl, lowest_price + stop_loss)
                    if future_candle['high'] >= trailing_sl:
                        outcome = "Win"
                        pnl = entry_price - trailing_sl
                        failure_reason = f"Trailing SL hit at {trailing_sl:.2f} after targets"
                        exit_time = self.safe_signal_time(future_candle.get('time', i))
                        break
        
        return {
            "outcome": outcome,
            "pnl": pnl,
            "targets_hit": targets_hit,
            "stoploss_count": stoploss_count,
            "failure_reason": failure_reason,
            "exit_time": exit_time
        }

    def safe_signal_time(self, val):
        """Safely convert a value to a datetime, returning a valid datetime or None."""
        if val is None:
            return datetime.now()
        
        if isinstance(val, datetime):
            return val
        
        try:
            if isinstance(val, (int, float)):
                # If it's a number, treat it as a timestamp
                return datetime.fromtimestamp(val)
            else:
                # Try to parse as datetime
                return pd.to_datetime(val)
        except:
            # If all else fails, return current time
            return datetime.now()

    def to_ist_str(self, val):
        """Convert a datetime value to IST string format."""
        try:
            if val is None:
                return None
            
            if isinstance(val, datetime):
                # Convert to IST
                ist_tz = pytz.timezone('Asia/Kolkata')
                if val.tzinfo is None:
                    val = pytz.utc.localize(val)
                ist_dt = val.astimezone(ist_tz)
                return ist_dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
        return None

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and generate trading signals using closed candles."""
        if not self.validate_data(data):
            return {'signal': 'NO TRADE', 'reason': 'insufficient data'}

        try:
            # Use closed candle for analysis
            candle = self.get_closed_candle(data)
            
            # Add indicators
            df = self.calculate_indicators(data)
            
            # Get indicator values from closed candle
            supertrend_direction = df['supertrend_direction'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_histogram = df['macd_histogram'].iloc[-1]
            ema = df['ema'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            body_ratio = df['body_ratio'].iloc[-1]
            
            # Check for NaN values
            if pd.isna(supertrend_direction) or pd.isna(rsi) or pd.isna(macd) or pd.isna(ema):
                return {'signal': 'NO TRADE', 'reason': 'indicator data unavailable'}
            
            # ENHANCED FILTERS FOR BEST PERFORMER
            
            # 1. Time-based filter (avoid lunch hour 11:30-1:30)
            current_time = datetime.now().time()
            lunch_start = datetime.strptime('11:30:00', '%H:%M:%S').time()
            lunch_end = datetime.strptime('13:30:00', '%H:%M:%S').time()
            
            if lunch_start <= current_time <= lunch_end:
                return {'signal': 'NO TRADE', 'reason': 'avoiding lunch hour (low volume)'}
            
            # 2. Volume spike filter
            if volume_ratio < 1.0:
                return {'signal': 'NO TRADE', 'reason': f'volume too low: {volume_ratio:.2f} < 1.0'}
            
            # 3. Body ratio filter (avoid doji candles)
            if body_ratio < 0.3:
                return {'signal': 'NO TRADE', 'reason': f'body ratio too low: {body_ratio:.2f} < 0.3'}
            
            # Calculate enhanced confidence score with risk scaling
            confidence_score = 0
            filter_alignment = 0  # Count of aligned filters
            
            # Trend strength (40% weight)
            if supertrend_direction == 1:
                confidence_score += 40
                filter_alignment += 1
            elif supertrend_direction == -1:
                confidence_score += 40
                filter_alignment += 1
            
            # RSI strength (25% weight)
            if 30 <= rsi <= 70:
                confidence_score += 25
                filter_alignment += 1
            elif 25 <= rsi <= 75:
                confidence_score += 15
            else:
                confidence_score += 5
            
            # MACD strength (25% weight)
            if (macd > macd_signal and macd_histogram > 0) or (macd < macd_signal and macd_histogram < 0):
                confidence_score += 25
                filter_alignment += 1
            elif abs(macd - macd_signal) > 0.1:
                confidence_score += 15
            else:
                confidence_score += 5
            
            # Price vs EMA alignment (10% weight)
            if ((supertrend_direction == 1 and candle['close'] > ema) or 
                (supertrend_direction == -1 and candle['close'] < ema)):
                confidence_score += 10
                filter_alignment += 1
            
            # Volume confirmation (bonus)
            if volume_ratio > 1.5:
                confidence_score += 10
            
            # RISK SCALING BASED ON FILTER ALIGNMENT
            if filter_alignment >= 4:  # All major filters aligned
                risk_multiplier = 1.0  # Full position size
                confidence_threshold = 70
            elif filter_alignment >= 3:  # Most filters aligned
                risk_multiplier = 0.8  # 80% position size
                confidence_threshold = 75
            else:  # Weak alignment
                risk_multiplier = 0.5  # 50% position size
                confidence_threshold = 80
            
            # Generate signals based on SuperTrend direction
            if supertrend_direction == 1 and confidence_score >= confidence_threshold:
                # BUY CALL signal
                atr = df['atr'].iloc[-1]
                stop_loss = 1.5 * atr
                target1 = 2.0 * atr
                target2 = 3.0 * atr
                target3 = 4.0 * atr
                
                # Dynamic position sizing based on filter alignment
                position_multiplier = risk_multiplier
                
                return {
                    'signal': 'BUY CALL',
                    'price': candle['close'],
                    'confidence_score': confidence_score,
                    'stop_loss': stop_loss,
                    'target1': target1,
                    'target2': target2,
                    'target3': target3,
                    'position_multiplier': position_multiplier,
                    'filter_alignment': filter_alignment,
                    'risk_multiplier': risk_multiplier,
                    'timestamp': candle.name if hasattr(candle, 'name') else datetime.now(),
                    'reasoning': f"SuperTrend uptrend, RSI {rsi:.1f}, MACD bullish, Price above EMA, Volume {volume_ratio:.2f}, Filters: {filter_alignment}/4"
                }
            
            elif supertrend_direction == -1 and confidence_score >= confidence_threshold:
                # BUY PUT signal
                atr = df['atr'].iloc[-1]
                stop_loss = 1.5 * atr
                target1 = 2.0 * atr
                target2 = 3.0 * atr
                target3 = 4.0 * atr
                
                # Dynamic position sizing based on filter alignment
                position_multiplier = risk_multiplier
                
                return {
                    'signal': 'BUY PUT',
                    'price': candle['close'],
                    'confidence_score': confidence_score,
                    'stop_loss': stop_loss,
                    'target1': target1,
                    'target2': target2,
                    'target3': target3,
                    'position_multiplier': position_multiplier,
                    'filter_alignment': filter_alignment,
                    'risk_multiplier': risk_multiplier,
                    'timestamp': candle.name if hasattr(candle, 'name') else datetime.now(),
                    'reasoning': f"SuperTrend downtrend, RSI {rsi:.1f}, MACD bearish, Price below EMA, Volume {volume_ratio:.2f}, Filters: {filter_alignment}/4"
                }
            
            return {'signal': 'NO TRADE', 'reason': f'confidence too low: {confidence_score} < {confidence_threshold}'}
            
        except Exception as e:
            logging.error(f"Error in SupertrendMacdRsiEma analysis: {e}")
            return {'signal': 'ERROR', 'reason': str(e)}

    def analyze_strategy(self, df: pd.DataFrame, candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Strategy-specific analysis with pre-calculated indicators and market conditions - OPTIMIZED VERSION"""
        try:
            # Get previous candle for SuperTrend calculation
            prev_candle = None
            if len(df) > 1:
                prev_candle = df.iloc[-2].to_dict()
            
            # Get SuperTrend data
            st_instance = self._get_supertrend_instance("5min")
            st_data = st_instance.update(candle, prev_candle)
            
            if st_data is None or st_data[1] is None:
                return {'signal': 'NO TRADE', 'reason': 'insufficient supertrend data'}
            
            supertrend_direction = st_data[1]
            
            # Get indicators from pre-calculated data
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1] if 'ema_50' in df.columns else ema_21
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # Check for NaN values
            if pd.isna(ema_9) or pd.isna(ema_21) or pd.isna(rsi) or pd.isna(macd) or pd.isna(atr):
                return {'signal': 'NO TRADE', 'reason': 'indicator data unavailable'}
            
            # MARKET CONDITION ANALYSIS - CRITICAL FILTER
            if 'market_tradeable' in df.columns and 'market_condition' in df.columns:
                market_tradeable = df['market_tradeable'].iloc[-1]
                market_condition = df['market_condition'].iloc[-1]
                adx = df['adx'].iloc[-1] if 'adx' in df.columns else 20
                trend_direction = df['trend_direction'].iloc[-1] if 'trend_direction' in df.columns else 0
                
                # DEBUG: Print market condition info for first few signals
                if len(df) % 100 == 0:  # Print every 100th candle
                    print(f"DEBUG: Market condition: {market_condition}, Tradeable: {market_tradeable}, ADX: {adx}")
                
                # CRITICAL: Check if market is tradeable
                if not market_tradeable:
                    return {'signal': 'NO TRADE', 'reason': f"market not tradeable: {df['market_reason'].iloc[-1] if 'market_reason' in df.columns else 'unknown'}"}
                
                # CRITICAL: Only trade in trending markets - REJECT RANGING MARKETS
                if market_condition == 'RANGING':
                    return {'signal': 'NO TRADE', 'reason': f"market ranging: ADX {adx:.1f} < 25"}
                
                # CRITICAL: Only trade in trending markets - REJECT INSUFFICIENT DATA
                if market_condition == 'INSUFFICIENT_DATA':
                    return {'signal': 'NO TRADE', 'reason': f"insufficient data for market analysis"}
            else:
                # If market condition data is not available, don't trade
                return {'signal': 'NO TRADE', 'reason': 'market condition data not available'}
            
            # ENHANCED FILTERS - MORE RESTRICTIVE
            if atr < 0.5:  # Higher volatility requirement
                return {'signal': 'NO TRADE', 'reason': f'ATR too low: {atr:.2f} < 0.5'}
            
            if volume_ratio < 1.2:  # Higher volume requirement
                return {'signal': 'NO TRADE', 'reason': f'volume too low: {volume_ratio:.2f} < 1.2'}
            
            # Body ratio filter - stronger candles only
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            body_ratio = body_size / total_range if total_range > 0 else 0
            
            if body_ratio < 0.4:  # Stronger candle body requirement
                return {'signal': 'NO TRADE', 'reason': f'weak candle body: {body_ratio:.2f} < 0.4'}
            
            # IMPROVEMENT 1: REDUCE CONDITIONS - Pick only the 3 best signals
            # Core signals: SuperTrend + EMA + MACD (RSI is optional)
            core_signals = 0
            optional_signals = 0
            
            # Core signal 1: SuperTrend direction
            if supertrend_direction > 0:
                core_signals += 1
            elif supertrend_direction < 0:
                core_signals -= 1
            
            # Core signal 2: EMA alignment (9 > 21 > 50 for uptrend)
            if ema_9 > ema_21 > ema_50:
                core_signals += 1
            elif ema_9 < ema_21 < ema_50:
                core_signals -= 1
            
            # Core signal 3: MACD alignment
            if macd > macd_signal:
                core_signals += 1
            elif macd < macd_signal:
                core_signals -= 1
            
            # Optional signal: RSI (only if extreme)
            if rsi < 30:
                optional_signals += 1
            elif rsi > 70:
                optional_signals -= 1
            
            # IMPROVEMENT 2: DYNAMIC POSITION SIZING BASED ON SIGNAL STRENGTH
            if core_signals >= 2:  # Strong bullish alignment
                position_size = 1.0  # Full position
                confidence_threshold = 70
            elif core_signals == 1 and optional_signals >= 0:  # Moderate bullish
                position_size = 0.7  # 70% position
                confidence_threshold = 75
            elif core_signals == 0 and optional_signals >= 1:  # Weak bullish
                position_size = 0.5  # 50% position
                confidence_threshold = 80
            elif core_signals <= -2:  # Strong bearish alignment
                position_size = 1.0  # Full position
                confidence_threshold = 70
            elif core_signals == -1 and optional_signals <= 0:  # Moderate bearish
                position_size = 0.7  # 70% position
                confidence_threshold = 75
            elif core_signals == 0 and optional_signals <= -1:  # Weak bearish
                position_size = 0.5  # 50% position
                confidence_threshold = 80
            else:
                return {'signal': 'NO TRADE', 'reason': 'insufficient signal alignment'}
            
            # Multi-Factor Confidence Integration
            from src.core.multi_factor_confidence import MultiFactorConfidence
            mfc = MultiFactorConfidence()
            signal_type = "BUY" if core_signals > 0 else "SELL"
            confidence_result = mfc.calculate_confidence(candle, signal_type, self.timeframe_data)
            confidence_score = confidence_result['total_score']
            
            # Market condition bonus
            market_bonus = 0
            if market_condition in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
                market_bonus = 25
            elif market_condition in ['WEAK_UPTREND', 'WEAK_DOWNTREND']:
                market_bonus = 15
            
            confidence_score += market_bonus
            
            # BUY CALL conditions - OPTIMIZED
            if (core_signals >= 1 and  # At least one bullish core signal
                confidence_score >= confidence_threshold):
                
                # IMPROVEMENT 3: BETTER RISK MANAGEMENT
                stop_loss = 2.0 * atr  # ATR-based stop-loss
                target1 = 3.0 * atr   # First target (1:1.5)
                target2 = 5.0 * atr   # Second target (1:2.5)
                
                return {
                    'signal': 'BUY CALL',
                    'price': candle['close'],
                    'confidence': confidence_score,
                    'stop_loss': stop_loss,
                    'target': target1,
                    'target2': target2,
                    'position_size': position_size,
                    'timestamp': candle.get('timestamp', datetime.now()),
                    'reason': f"Core signals: {core_signals}, Optional: {optional_signals}, Market: {market_condition}, Confidence: {confidence_score:.1f}, Size: {position_size:.1f}"
                }
            
            # BUY PUT conditions - OPTIMIZED
            elif (core_signals <= -1 and  # At least one bearish core signal
                  confidence_score >= confidence_threshold):
                
                # IMPROVEMENT 3: BETTER RISK MANAGEMENT
                stop_loss = 2.0 * atr  # ATR-based stop-loss
                target1 = 3.0 * atr   # First target (1:1.5)
                target2 = 5.0 * atr   # Second target (1:2.5)
                
                return {
                    'signal': 'BUY PUT',
                    'price': candle['close'],
                    'confidence': confidence_score,
                    'stop_loss': stop_loss,
                    'target': target1,
                    'target2': target2,
                    'position_size': position_size,
                    'timestamp': candle.get('timestamp', datetime.now()),
                    'reason': f"Core signals: {core_signals}, Optional: {optional_signals}, Market: {market_condition}, Confidence: {confidence_score:.1f}, Size: {position_size:.1f}"
                }
            
            return {'signal': 'NO TRADE', 'reason': 'no conditions met'}
            
        except Exception as e:
            return {'signal': 'ERROR', 'reason': f'strategy error: {str(e)}'}
