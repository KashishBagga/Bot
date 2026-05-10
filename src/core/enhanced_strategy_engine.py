#!/usr/bin/env python3
"""
Enhanced Strategy Engine with Multi-Timeframe Confirmation
=========================================================
Implements multi-timeframe analysis for higher quality signals
with position sizing, dynamic stop-loss, and market condition filters
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from enum import Enum

# Import strategies
from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.strategies.simple_ema_strategy import SimpleEmaStrategy
from src.strategies.supertrend_ema import SupertrendEma

# Import regime detection (market_regime_strategies.py wired in)
from src.strategies.market_regime_strategies import MarketRegimeDetector

logger = logging.getLogger(__name__)

class Timeframe(Enum):
    """Supported timeframes for multi-timeframe analysis"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class SignalStrength(Enum):
    """Signal strength based on timeframe confirmation"""
    WEAK = "weak"           # 1 timeframe
    MODERATE = "moderate"   # 2 timeframes
    STRONG = "strong"       # 3 timeframes
    VERY_STRONG = "very_strong"  # 4+ timeframes

class MarketCondition(Enum):
    """Market condition based on volatility and trend"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"

class EnhancedStrategyEngine:
    def _filter_low_quality_signals(self, signals):
        """Filter out low-quality signals."""
        filtered_signals = []
        
        for signal in signals:
            # Only accept signals with confidence >= 50
            if signal.get('confidence', 0) >= 50:
                filtered_signals.append(signal)
            else:
                logger.debug(f"Filtered low-confidence signal: {signal['strategy']} {signal['signal']} (confidence: {signal.get('confidence', 0)})")
        
        return filtered_signals
    """Enhanced strategy engine with multi-timeframe confirmation"""
    
    def __init__(self, symbols: List[str], confidence_cutoff: float = 25.0):
        self.symbols = symbols
        self.confidence_cutoff = confidence_cutoff
        self.tz = ZoneInfo("Asia/Kolkata")
        
        # Timeframe hierarchy (higher = longer term)
        self.timeframe_hierarchy = {
            Timeframe.M1: 1,
            Timeframe.M5: 2,
            Timeframe.M15: 3,
            Timeframe.M30: 4,
            Timeframe.H1: 5,
            Timeframe.H4: 6,
            Timeframe.D1: 7
        }
        
        # Initialize strategies
        self.strategies = {
            'ema_crossover_enhanced': EmaCrossoverEnhanced(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma(),
            'simple_ema': SimpleEmaStrategy(),
            'supertrend_ema': SupertrendEma()
        }

        # Regime detector — wires MarketRegimeDetector into every signal cycle
        self.regime_detector = MarketRegimeDetector()

        # Position sizing parameters
        self.base_position_size = 1.0
        self.max_position_size = 3.0
        self.volatility_adjustment = True

        # Risk management parameters
        self.base_stop_loss = 0.02  # 2% base stop loss
        self.max_stop_loss = 0.05   # 5% maximum stop loss
        self.trailing_stop_enabled = True

        # Regime-to-Strategy Mapping (High-Probability Pairing)
        self.regime_strategy_map = {
            "BULL_TREND":      ['ema_crossover_enhanced', 'supertrend_macd_rsi_ema', 'supertrend_ema'],
            "BEAR_TREND":      ['ema_crossover_enhanced', 'supertrend_macd_rsi_ema', 'supertrend_ema'],
            "BREAKOUT":        ['ema_crossover_enhanced', 'supertrend_ema'],
            "REVERSAL":        ['supertrend_macd_rsi_ema'],
            "LOW_VOLATILITY":  ['simple_ema'],
            "HIGH_VOLATILITY": ['supertrend_macd_rsi_ema'], # More filters needed here
            "SIDEWAYS":        [], # Avoid trading in sideways
            "UNKNOWN":         ['simple_ema']
        }

        # Anti-Chop Thresholds
        self.CHOP_THRESHOLD = 61.8 # > 61.8 = high chop (Don't Trade)
        self.TREND_MIN_CHOP = 38.2 # < 38.2 = strong trend

        logger.info(f"🚀 Enhanced Strategy Engine initialized for {len(symbols)} symbols | regime-first ON")
    
    def calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(data) < period:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        if len(returns) < period:
            return 0.0
        
        return returns.tail(period).std() * np.sqrt(252)  # Annualized volatility
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for volatility measurement."""
        if len(data) < period:
            return 0.0
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.0
    
    def detect_market_condition(self, data: pd.DataFrame) -> MarketCondition:
        """Detect market condition based on volatility and trend."""
        if len(data) < 50:
            return MarketCondition.CALM
        
        # Calculate volatility
        volatility = self.calculate_volatility(data)
        
        # Calculate trend strength (ADX-like)
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Simple trend strength calculation
        price_change = abs(close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
        
        if volatility > 0.3:  # High volatility
            return MarketCondition.VOLATILE
        elif price_change > 0.05:  # Strong trend
            return MarketCondition.TRENDING
        elif price_change < 0.02:  # Weak movement
            return MarketCondition.RANGING
        else:
            return MarketCondition.CALM
    
    def calculate_position_size(self, symbol: str, signal_confidence: float, 
                              volatility: float, market_condition: MarketCondition) -> float:
        """Calculate position size based on signal confidence, volatility, and market condition."""
        # Base position size
        position_size = self.base_position_size
        
        # Adjust for signal confidence
        confidence_multiplier = signal_confidence / 100.0
        position_size *= confidence_multiplier
        
        # Adjust for volatility (lower volatility = larger position)
        if self.volatility_adjustment and volatility > 0:
            volatility_multiplier = max(0.5, 1.0 - (volatility * 2))
            position_size *= volatility_multiplier
        
        # Adjust for market condition
        if market_condition == MarketCondition.TRENDING:
            position_size *= 1.2  # Increase size in trending markets
        elif market_condition == MarketCondition.VOLATILE:
            position_size *= 0.7  # Reduce size in volatile markets
        elif market_condition == MarketCondition.RANGING:
            position_size *= 0.8  # Reduce size in ranging markets
        
        # Ensure within bounds
        position_size = max(0.1, min(position_size, self.max_position_size))
        
        return round(position_size, 2)
    
    def calculate_dynamic_stop_loss(self, entry_price: float, signal_type: str, 
                                  volatility: float, market_condition: MarketCondition) -> float:
        """Calculate dynamic stop loss based on volatility and market conditions."""
        # Base stop loss
        stop_loss = self.base_stop_loss
        
        # Adjust for volatility
        if volatility > 0.2:  # High volatility
            stop_loss *= 1.5
        elif volatility < 0.1:  # Low volatility
            stop_loss *= 0.8
        
        # Adjust for market condition
        if market_condition == MarketCondition.VOLATILE:
            stop_loss *= 1.3
        elif market_condition == MarketCondition.CALM:
            stop_loss *= 0.9
        
        # Ensure within bounds
        stop_loss = max(0.01, min(stop_loss, self.max_stop_loss))
        
        # Calculate stop loss price
        if signal_type == 'BUY CALL':
            stop_price = entry_price * (1 - stop_loss)
        else:  # BUY PUT
            stop_price = entry_price * (1 + stop_loss)
        
        return stop_price
    
    def filter_signals_by_market_condition(self, signals: List[Dict], 
                                         market_conditions: Dict[str, MarketCondition]) -> List[Dict]:
        """Filter signals based on market conditions."""
        filtered_signals = []
        
        for signal in signals:
            symbol = signal['symbol']
            market_condition = market_conditions.get(symbol, MarketCondition.CALM)
            
            # Skip signals in unfavorable market conditions
            if market_condition == MarketCondition.VOLATILE and signal['confidence'] < 60:
                logger.debug(f"⚠️ Skipping {symbol} signal due to high volatility and low confidence")
                continue
            
            if market_condition == MarketCondition.RANGING and signal['confidence'] < 50:
                logger.debug(f"⚠️ Skipping {symbol} signal due to ranging market and low confidence")
                continue
            
            filtered_signals.append(signal)
        
        return filtered_signals
    

    def generate_signals_for_all_symbols(
        self,
        historical_data: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
    ) -> List[Dict]:
        """
        Generate signals for all symbols.
        Now includes per-symbol:
          • Regime detection  (MarketRegimeDetector)
          • Regime-based confidence floor
          • ATR extraction passed into each signal
          • Multi-TF confirmation via 4h / 1D resampling
          • Market Context (Trend alignment across indices)
        """
        all_signals = []
        
        # ── 0. Calculate Global Market Context ───────────────────────
        market_sentiment = self._calculate_market_sentiment(historical_data)
        logger.info(f"🌍 Market Sentiment: {market_sentiment:.1f}% (+ve = Bullish, -ve = Bearish)")

        for symbol in self.symbols:
            if symbol not in historical_data or symbol not in current_prices:
                continue
            data = historical_data[symbol]
            current_price = current_prices[symbol]

            if data is None or len(data) < 50:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(data) if data is not None else 0} candles")
                continue

            # ── 1. Detect market regime ──────────────────────────────────
            regime = "UNKNOWN"
            try:
                rs = self.regime_detector.detect_regime(data)
                if rs:
                    regime = rs.regime.value   # e.g. 'BULL_TREND'
            except Exception as e:
                logger.debug(f"Regime detection failed for {symbol}: {e}")

            # ── 2. Regime-based minimum confidence floor ─────────────────
            regime_confidence_floor: float = {
                "SIDEWAYS":        70.0,
                "HIGH_VOLATILITY": 65.0,
                "LOW_VOLATILITY":  50.0,
                "REVERSAL":        65.0,
            }.get(regime, 50.0)

            # ── 3. Generate signals with regime context ──────────────────
            symbol_signals = self._generate_symbol_signals(
                symbol, data, current_price,
                regime=regime,
                regime_confidence_floor=regime_confidence_floor,
                market_sentiment=market_sentiment
            )
            all_signals.extend(symbol_signals)

        # ── 4. Strategy Confluence (Fewer, Better Setups) ──────────────
        # If multiple strategies signal same symbol/direction, merge and boost
        confluence_signals = self._apply_strategy_confluence(all_signals)

        confluence_signals.sort(key=lambda x: x['confidence'], reverse=True)
        logger.info(f"📊 Generated {len(confluence_signals)} signals (after confluence) across {len(self.symbols)} symbols")
        return confluence_signals

    def _apply_strategy_confluence(self, signals: List[Dict]) -> List[Dict]:
        """Merge signals for same symbol/direction and apply bonus."""
        grouped = {} # (symbol, direction) -> List[Signal]
        for s in signals:
            key = (s['symbol'], s['signal'])
            if key not in grouped: grouped[key] = []
            grouped[key].append(s)
        
        final_signals = []
        for (symbol, direction), group in grouped.items():
            if len(group) == 1:
                final_signals.append(group[0])
                continue
            
            # Confluence detected!
            best_signal = max(group, key=lambda x: x['confidence'])
            strategies = [s['strategy'] for s in group]
            
            # Apply confluence bonus
            bonus = 1.15 if len(group) >= 2 else 1.0
            best_signal['confidence'] = min(100.0, best_signal['confidence'] * bonus)
            best_signal['strategy'] = f"confluence({','.join(strategies)})"
            best_signal['reasoning'] = f"CONFLUENCE: {len(group)} strategies agree. " + best_signal.get('reasoning', '')
            
            final_signals.append(best_signal)
            logger.info(f"💎 CONFLUENCE for {symbol} {direction}: {len(group)} strategies aligned. Confidence boosted to {best_signal['confidence']:.1f}")
            
        return final_signals

    def _calculate_market_sentiment(self, historical_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate overall market sentiment based on EMAs of all tracked indices."""
        total_bias = 0
        count = 0
        for symbol, data in historical_data.items():
            if data is None or 'ema_50' not in data.columns:
                continue
            # +1 if price > EMA50, -1 if below
            bias = 1 if data['close'].iloc[-1] > data['ema_50'].iloc[-1] else -1
            total_bias += bias
            count += 1
        
        return (total_bias / count) * 100 if count > 0 else 0


    def _generate_symbol_signals(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_price: float,
        regime: str = "UNKNOWN",
        regime_confidence_floor: float = 50.0,
        market_sentiment: float = 0.0
    ) -> List[Dict]:
        """
        Generate signals for a single symbol.
        Enforces:
          1. Chop Filter (CHOP > 61.8 = NO TRADE)
          2. Regime-Strategy Mapping (Only allow context-aware strategies)
          3. Index Alignment (Market Sentiment trend check)
        """
        signals = []

        # ── 1. Chop Filter (The NO_TRADE engine) ───────────────────────
        chop_val = data['chop'].iloc[-1] if 'chop' in data.columns else 50
        if chop_val >= self.CHOP_THRESHOLD:
            logger.debug(f"🛑 Blocking {symbol} signals — Market is too choppy (CHOP: {chop_val:.1f})")
            return []

        # ── 2. Market Sentiment Alignment (Index Context) ──────────────
        # Reject if signal direction fights the overall market bias
        # (e.g. BUY signal when market_sentiment is significantly bearish)
        # We allow trades if sentiment is neutral (-30 to 30)

        # ── 3. Strategy-Regime Enforcement ────────────────────────────
        allowed_strategies = self.regime_strategy_map.get(regime, [])
        if not allowed_strategies:
            logger.debug(f"⏸️ No active strategies for regime: {regime}")
            return []

        # Pre-compute ATR once for all strategies on this symbol
        atr = self._compute_atr_from_data(data)

        for strategy_name, strategy in self.strategies.items():
            try:
                # Only run strategy if it fits the current market regime
                if strategy_name not in allowed_strategies:
                    continue

                signal_result = strategy.analyze(data)

                if not signal_result or signal_result.get('signal') in ['NO TRADE', 'ERROR']:
                    continue

                signal_type = signal_result.get('signal')
                confidence  = float(signal_result.get(
                    'confidence', signal_result.get('confidence_score', 0)
                ))

                # ── 2b. Context Check (Continued) ────────────────────────
                if signal_type == 'BUY CALL' and market_sentiment < -30:
                    logger.debug(f"⚠️ {symbol} BUY CALL rejected: Market sentiment is too bearish ({market_sentiment:.1f}%)")
                    continue
                if signal_type == 'BUY PUT' and market_sentiment > 30:
                    logger.debug(f"⚠️ {symbol} BUY PUT rejected: Market sentiment is too bullish ({market_sentiment:.1f}%)")
                    continue

                # Base confidence cutoff
                if confidence < self.confidence_cutoff:
                    continue

                # ── Multi-TF confirmation (4h + 1D resamples) ─────────────
                tf_count, tf_strength = self._check_higher_tf_alignment(data, signal_type)
                # tf_count: 0=weak, 1=moderate, 2=strong
                if tf_count == 0:
                    # Weak: penalise confidence
                    confidence = confidence * 0.85
                elif tf_count == 2:
                    # Both higher TFs agree: small bonus
                    confidence = min(100.0, confidence * 1.05)

                # Apply regime confidence floor
                if confidence < regime_confidence_floor:
                    logger.debug(
                        f"⚠️ {strategy_name}/{symbol} rejected by regime floor "
                        f"(conf={confidence:.1f} < {regime_confidence_floor}, regime={regime})"
                    )
                    continue

                # ── Build signal dict (with ATR + regime propagated) ───────
                signal = {
                    'symbol':         symbol,
                    'strategy':       strategy_name,
                    'signal':         signal_type,
                    'confidence':     round(confidence, 2),
                    'price':          current_price,
                    'timestamp':      datetime.now(self.tz).isoformat(),
                    'timeframe':      '1h',
                    'strength':       tf_strength,
                    'confirmed':      tf_count >= 1,
                    'tf_confirmations': tf_count,
                    # ATR forwarded so _open_trade can build ATR-based SL/TP
                    'atr':            atr,
                    # Regime forwarded for position sizing
                    'regime':         regime,
                    # Legacy fields kept for DB compatibility
                    'position_size':  self._calculate_position_size(confidence),
                    'stop_loss_price':  self._calculate_stop_loss(current_price, signal_type),
                    'take_profit_price': self._calculate_take_profit(current_price, signal_type),
                    'indicator_values': {
                        'rsi':        signal_result.get('rsi'),
                        'macd':       signal_result.get('macd'),
                        'supertrend': signal_result.get('supertrend'),
                        'atr':        atr,
                    },
                    'market_condition': regime,
                    'volatility':     signal_result.get('volatility', 0.2),
                }

                signals.append(signal)
                logger.info(
                    f"✅ {strategy_name} → {symbol}: {signal_type} "
                    f"conf={confidence:.1f} | regime={regime} | tf={tf_count}/2"
                )

            except Exception as e:
                logger.error(f"❌ Error generating {strategy_name} signal for {symbol}: {e}")
                continue

        return self._filter_low_quality_signals(signals)

    def generate_signals(self, symbol: str, historical_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Generate signals for a symbol using all strategies."""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                # Get data for the strategy's preferred timeframe
                timeframe = getattr(strategy, 'timeframe', '1h')
                if timeframe in historical_data:
                    data = historical_data[timeframe]
                    
                    if len(data) > 0:
                        signal = strategy.generate_signal(data, symbol)
                        if signal:
                            signal['strategy'] = strategy_name
                            signal['timeframe'] = timeframe
                            signals.append(signal)
                            
            except Exception as e:
                logger.error(f"❌ Error generating signal for {symbol} with {strategy_name}: {e}")
        
        return self._filter_low_quality_signals(signals)
    
    def confirm_signal_across_timeframes(self, symbol: str, signal: Dict, 
                                       historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """Confirm signal across multiple timeframes."""
        confirmation_count = 0
        confirming_timeframes = []
        
        # Check signal in different timeframes
        for timeframe_str, data in historical_data.items():
            if len(data) < 10:  # Need minimum data
                continue
                
            try:
                # Get the strategy that generated the original signal
                strategy_name = signal.get('strategy', 'simple_ema')
                strategy = self.strategies.get(strategy_name)
                
                if strategy:
                    # Generate signal in this timeframe
                    timeframe_signal = strategy.generate_signal(data, symbol)
                    
                    if timeframe_signal and timeframe_signal['signal'] == signal['signal']:
                        confirmation_count += 1
                        confirming_timeframes.append(timeframe_str)
                        
            except Exception as e:
                logger.debug(f"Error confirming signal in {timeframe_str}: {e}")
        
        # Calculate confirmation strength
        if confirmation_count >= 3:
            strength = SignalStrength.VERY_STRONG
        elif confirmation_count == 2:
            strength = SignalStrength.STRONG
        elif confirmation_count == 1:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        return {
            'confirmed': confirmation_count > 0,
            'confirmation_count': confirmation_count,
            'timeframes': confirming_timeframes,
            'strength': strength.value
        }
    
    def process_symbol(self, symbol: str, historical_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Process a single symbol and generate enhanced signals."""
        try:
            # Generate signals
            signals = self.generate_signals(symbol, historical_data)
            
            if not signals:
                return []
            
            # Get market condition for this symbol
            primary_timeframe = '1h'  # Use 1h as primary timeframe
            if primary_timeframe in historical_data:
                market_condition = self.detect_market_condition(historical_data[primary_timeframe])
            else:
                market_condition = MarketCondition.CALM
            
            # Calculate volatility
            volatility = 0.0
            if primary_timeframe in historical_data:
                volatility = self.calculate_volatility(historical_data[primary_timeframe])
            
            enhanced_signals = []
            
            for signal in signals:
                # Confirm signal across timeframes
                confirmation_result = self.confirm_signal_across_timeframes(symbol, signal, historical_data)
                
                if confirmation_result['confirmed']:
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        symbol, signal['confidence'], volatility, market_condition
                    )
                    
                    # Calculate dynamic stop loss
                    stop_loss_price = self.calculate_dynamic_stop_loss(
                        signal['price'], signal['signal'], volatility, market_condition
                    )
                    
                    # Create enhanced signal
                    enhanced_signal = {
                        'symbol': symbol,
                        'signal': signal['signal'],
                        'price': signal['price'],
                        'confidence': signal['confidence'],
                        'strategy': signal['strategy'],
                        'timeframe': signal['timeframe'],
                        'strength': confirmation_result['strength'],
                        'confirmation_count': confirmation_result['confirmation_count'],
                        'confirming_timeframes': confirmation_result['timeframes'],
                        'position_size': position_size,
                        'stop_loss_price': stop_loss_price,
                        'market_condition': market_condition.value,
                        'volatility': volatility,
                        'timestamp': datetime.now(self.tz)
                    }
                    
                    enhanced_signals.append(enhanced_signal)
                    
                    logger.debug(f"✅ {symbol} {signal['signal']} confirmed across {len(confirmation_result['timeframes'])} timeframes")
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"❌ Error processing symbol {symbol}: {e}")
            return []
    

    # ── Higher-timeframe helpers (added for multi-TF confirmation) ──────────

    @staticmethod
    def _resample_data(data: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV DataFrame to a coarser timeframe (e.g. '4h', '1D')."""
        try:
            df = data.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
            if 'volume' in df.columns:
                agg['volume'] = 'sum'
            return df.resample(rule).agg(agg).dropna()
        except Exception:
            return pd.DataFrame()

    def _check_higher_tf_alignment(self, data: pd.DataFrame, signal_type: str):
        """
        Resample 1h data to 4h and 1D and check EMA trend direction.
        Returns (confirmations: int, strength_label: str).
          0 → 'weak', 1 → 'moderate', 2 → 'strong'
        """
        confirmations = 0

        def _ema(series, span):
            return series.ewm(span=span, adjust=False).mean()

        # ── 4h alignment ─────────────────────────────────────────────────
        try:
            tf4h = self._resample_data(data, '4h')
            if len(tf4h) >= 20:
                ema20 = _ema(tf4h['close'], 20).iloc[-1]
                ema50 = _ema(tf4h['close'], min(50, len(tf4h))).iloc[-1]
                p4h   = tf4h['close'].iloc[-1]
                if signal_type == 'BUY CALL' and p4h > ema20 and ema20 >= ema50:
                    confirmations += 1
                elif signal_type == 'BUY PUT' and p4h < ema20 and ema20 <= ema50:
                    confirmations += 1
        except Exception:
            pass

        # ── 1D alignment ─────────────────────────────────────────────────
        try:
            tf1d = self._resample_data(data, '1D')
            if len(tf1d) >= 10:
                ema10 = _ema(tf1d['close'], 10).iloc[-1]
                p1d   = tf1d['close'].iloc[-1]
                if signal_type == 'BUY CALL' and p1d > ema10:
                    confirmations += 1
                elif signal_type == 'BUY PUT' and p1d < ema10:
                    confirmations += 1
        except Exception:
            pass

        strength = {0: 'weak', 1: 'moderate', 2: 'strong'}.get(confirmations, 'weak')
        return confirmations, strength

    @staticmethod
    def _compute_atr_from_data(data: pd.DataFrame, period: int = 14) -> float:
        """Compute latest ATR value from OHLCV DataFrame."""
        try:
            if data is None or len(data) < period + 1:
                return 0.0
            hi = data['high']
            lo = data['low']
            pc = data['close'].shift(1)
            tr = pd.concat([(hi - lo), (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            return float(atr) if not pd.isna(atr) else 0.0
        except Exception:
            return 0.0

    # ── Legacy position-size / SL / TP helpers (kept for DB compat) ──────────

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence."""
        # Base position size
        base_size = 5000.0  # Base position size
        
        # Adjust for confidence
        confidence_multiplier = confidence / 100.0
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier
        
        # Ensure within bounds
        position_size = max(1000, min(position_size, 10000))
        
        return round(position_size, 2)
    
    def _calculate_stop_loss(self, entry_price: float, signal_type: str) -> float:
        """Calculate stop loss price."""
        if signal_type == 'BUY CALL':
            return entry_price * (1 - self.base_stop_loss)
        else:  # BUY PUT
            return entry_price * (1 + self.base_stop_loss)
    
    def _calculate_take_profit(self, entry_price: float, signal_type: str) -> float:
        """Calculate take profit price."""
        if signal_type == 'BUY CALL':
            return entry_price * (1 + 0.05)  # 5% take profit
        else:  # BUY PUT
            return entry_price * (1 - 0.05)  # 5% take profit

    def run_analysis(self, historical_data: Dict[str, Dict[str, pd.DataFrame]]) -> List[Dict]:
        """Run enhanced analysis on all symbols."""
        all_signals = []
        
        # Detect market conditions for all symbols
        market_conditions = {}
        for symbol in self.symbols:
            if symbol in historical_data:
                primary_timeframe = '1h'
                if primary_timeframe in historical_data[symbol]:
                    market_conditions[symbol] = self.detect_market_condition(historical_data[symbol][primary_timeframe])
        
        # Process each symbol
        for symbol in self.symbols:
            if symbol in historical_data:
                signals = self.process_symbol(symbol, historical_data[symbol])
                all_signals.extend(signals)
        
        # Filter signals by market conditions
        filtered_signals = self.filter_signals_by_market_condition(all_signals, market_conditions)
        
        # Sort by confidence and strength
        filtered_signals.sort(key=lambda x: (x['confidence'], x['confirmation_count']), reverse=True)
        
        logger.info(f"📊 Generated {len(all_signals)} total signals, {len(filtered_signals)} after filtering")
        
        return filtered_signals
