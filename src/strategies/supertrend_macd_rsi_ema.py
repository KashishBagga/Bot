import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.core.strategy import Strategy
from src.core.indicators import indicators
from src.models.unified_database import UnifiedDatabase
import math

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
        self.min_confidence_threshold = params.get("min_confidence_threshold", 50)
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
            
            # Calculate confidence score based on multiple factors
            confidence_score = 0
            
            # Trend strength (40% weight)
            if supertrend_direction == 1:
                confidence_score += 40
            elif supertrend_direction == -1:
                confidence_score += 0
            else:
                confidence_score += 20
            
            # RSI strength (30% weight)
            if 30 <= rsi <= 70:
                confidence_score += 30
            elif 20 <= rsi <= 80:
                confidence_score += 20
            else:
                confidence_score += 10
            
            # MACD strength (30% weight)
            if macd > macd_signal and macd_histogram > 0:
                confidence_score += 30
            elif macd < macd_signal and macd_histogram < 0:
                confidence_score += 30
            else:
                confidence_score += 15
            
            # Volume confirmation (bonus)
            if volume_ratio > 1.0:
                confidence_score += 10
            
            # Body ratio confirmation (bonus)
            if body_ratio > 0.6:
                confidence_score += 10
            
            # BUY CALL conditions
            if (supertrend_direction == 1 and  # SuperTrend uptrend
                rsi > 30 and rsi < 80 and  # RSI in healthy range
                macd > macd_signal and  # MACD bullish
                candle['close'] > ema and  # Price above EMA
                volume_ratio > 0.5):  # Volume confirmation
                
                if confidence_score >= self.min_confidence_threshold:
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
                        'reasoning': f"SuperTrend uptrend, RSI {rsi:.1f}, MACD bullish, Price above EMA, Volume {volume_ratio:.2f}"
                    }
            
            # BUY PUT conditions
            elif (supertrend_direction == -1 and  # SuperTrend downtrend
                  rsi > 20 and rsi < 70 and  # RSI in healthy range
                  macd < macd_signal and  # MACD bearish
                  candle['close'] < ema and  # Price below EMA
                  volume_ratio > 0.5):  # Volume confirmation
                
                if confidence_score >= self.min_confidence_threshold:
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
                        'reasoning': f"SuperTrend downtrend, RSI {rsi:.1f}, MACD bearish, Price below EMA, Volume {volume_ratio:.2f}"
                    }
            
            return {'signal': 'NO TRADE', 'reason': 'no signal conditions met'}
            
        except Exception as e:
            logging.error(f"Error in SupertrendMacdRsiEma analysis: {e}")
            return {'signal': 'ERROR', 'reason': str(e)}
