#!/usr/bin/env python3
"""
IndicatorPipeline — Compute all indicators once per symbol per candle.
======================================================================
Receives raw OHLCV DataFrames from the data provider and produces a
fully decorated MarketSnapshot for all strategies to consume.

No API calls. No strategy logic. Pure computation.
"""

import logging
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd

from src.core.feature_store import FeatureStore
from src.core.market_snapshot import MarketSnapshot
from src.core.structure_engine import StructureEngine
from src.core.zone_engine import ZoneEngine
from src.core.volume_engine import VolumeEngine
from src.core.regime_engine import RegimeEngine
from src.core.quant_utils import QuantUtils

# MKE Stage 1 Context Imports
from src.core.market_knowledge import MarketContext, HTFStructure, StructureState, SwingStatus, ResearchEvent
from src.core.time_of_day_engine import TimeOfDayEngine
from src.core.pivot_engine import PivotEngine
from src.core.cluster_engine import ClusterEngine
from src.models.postgres_database import PostgresDatabase

logger = logging.getLogger(__name__)


class LegacyStructureReportAdapter:
    """Adapts StructureState and ResearchEvents to the legacy StructureReport interface for backward compatibility."""
    def __init__(self, state: StructureState, events: List[ResearchEvent]):
        self.state = state
        self.events = events
        
        # 1. Compute trend
        self.trend = "NEUTRAL"
        highs = [s for s in state.swings if s.type == "HIGH"]
        lows = [s for s in state.swings if s.type == "LOW"]
        if len(highs) >= 2 and len(lows) >= 2:
            last_h, prev_h = highs[-1].price, highs[-2].price
            last_l, prev_l = lows[-1].price, lows[-2].price
            if last_h > prev_h and last_l > prev_l:
                self.trend = "BULLISH"
            elif last_h < prev_h and last_l < prev_l:
                self.trend = "BEARISH"
                
        # 2. Compute phase
        self.market_phase = "NEUTRAL"
        if len(state.swings) > 0 and len(highs) > 0 and len(lows) > 0:
            current_close = state.developing_leg.current_price if state.developing_leg else state.swings[-1].price
            if self.trend == "BULLISH":
                self.market_phase = "EXPANSION" if current_close > highs[-1].price else "RETRACEMENT"
            elif self.trend == "BEARISH":
                self.market_phase = "EXPANSION" if current_close < lows[-1].price else "RETRACEMENT"

        # 3. Compute bos_count
        self.bos_count = 0
        if self.trend == "BULLISH":
            self.bos_count = sum(1 for s in state.swings if s.type == "HIGH" and s.status in [SwingStatus.BREACHED, SwingStatus.RETESTED])
        elif self.trend == "BEARISH":
            self.bos_count = sum(1 for s in state.swings if s.type == "LOW" and s.status in [SwingStatus.BREACHED, SwingStatus.RETESTED])

        # 4. Compute choch_detected
        self.choch_detected = any(e.event_type == "CHOCH_CONFIRMED" for e in events)

        # 5. Swing extremes
        self.last_swing_high = highs[-1].price if len(highs) > 0 else 0.0
        self.last_swing_low = lows[-1].price if len(lows) > 0 else 0.0
        
        # 6. Compression
        self.is_compressed = state.is_compressed

        # 7. Quality score
        score = 50.0
        if self.trend != "NEUTRAL":
            score += min(self.bos_count * 10, 30)
            if self.choch_detected:
                score -= 40
            if self.is_compressed:
                score += 20
            self.quality_score = max(0.0, min(100.0, score))
        else:
            self.quality_score = 0.0


class IndicatorPipeline:
    """
    Single-pass indicator computation. Instantiate once, call compute() each candle.
    All engine instances are reused — no re-initialisation per symbol.
    """

    def __init__(
        self,
        pivot_window: int = 3,
        zone_cluster_pct: float = 0.002,
        min_zone_score: float = 50.0,
        historical_days: int = 20,
    ):
        self.tod_engine = TimeOfDayEngine()
        self.pivot_engine = PivotEngine(pivot_window=pivot_window)
        self.cluster_engine = ClusterEngine()
        self.structure_engine = StructureEngine(pivot_window=pivot_window)
        self.zone_engine = ZoneEngine(cluster_pct=zone_cluster_pct)
        self.zone_engine.MIN_ZONE_SCORE = min_zone_score
        self.volume_engine = VolumeEngine(historical_days=historical_days)
        self.regime_engine = RegimeEngine()
        
        # Instantiate PostgresDatabase for persisting research events
        self.db = PostgresDatabase()

    @property
    def required_history(self) -> int:
        """Dynamically resolve the maximum required history length across all MKE/Indicator engines."""
        return max(
            self.structure_engine.required_history,
            self.pivot_engine.required_history,
            self.cluster_engine.required_history,
            self.tod_engine.required_history
        )

    # ── Public interface ───────────────────────────────────────────────────

    def compute(
        self,
        symbol: str,
        price: float,
        d1: Optional[pd.DataFrame],
        h1: Optional[pd.DataFrame],
        m5: Optional[pd.DataFrame],
        timestamp: datetime,
    ) -> Optional[MarketSnapshot]:
        """
        Run all indicator stages and return a fully decorated MarketSnapshot.
        Returns None if data is insufficient (e.g. API returned empty frames).
        """
        if h1 is None or m5 is None:
            logger.warning(f"[Pipeline] Insufficient data for {symbol} — skipping snapshot.")
            return None

        try:
            # Stage 1: Market structure (structural bias, zones)
            daily_bias, h1_structure, h1_zones = self._stage_structure(d1, h1, symbol)

            # Stage 2: Volume participation
            volume_report = self._stage_volume(m5, symbol)

            # Stage 3: Market regime
            market_regime = self._stage_regime(m5)

            # Stage 4: Feature store (ATR, EMAs, derived metrics)
            features = self._stage_features(m5, d1)

            # ── Stage 5: Market Context Engine (MKE) ──
            market_ctx = MarketContext()
            
            # Compute m5 structure (primary execution timeframe)
            m5_state, m5_events = self._compute_mke_structure(m5, symbol, "m5", confirmed_only_bars=False)
            market_ctx.structure = m5_state
            
            # Save m5 confirmed events
            for event in m5_events:
                self.db.save_market_event(event.to_dict())
                
            # Compute h1 HTF structures
            h1_conf_state, h1_conf_events = self._compute_mke_structure(h1, symbol, "h1", confirmed_only_bars=True)
            h1_dev_state, _ = self._compute_mke_structure(h1, symbol, "h1", confirmed_only_bars=False)
            market_ctx.htf_structure["h1"] = HTFStructure(confirmed=h1_conf_state, developing=h1_dev_state)
            
            # Save h1 confirmed events
            for event in h1_conf_events:
                self.db.save_market_event(event.to_dict())
                
            # Compute d1 HTF structures if available
            if d1 is not None and len(d1) >= 20:
                d1_conf_state, d1_conf_events = self._compute_mke_structure(d1, symbol, "d1", confirmed_only_bars=True)
                d1_dev_state, _ = self._compute_mke_structure(d1, symbol, "d1", confirmed_only_bars=False)
                market_ctx.htf_structure["d1"] = HTFStructure(confirmed=d1_conf_state, developing=d1_dev_state)
                
                # Save d1 confirmed events
                for event in d1_conf_events:
                    self.db.save_market_event(event.to_dict())

            return MarketSnapshot(
                symbol=symbol,
                current_price=price,
                timestamp=timestamp,
                d1=d1,
                h1=h1,
                m5=m5,
                daily_bias=daily_bias,
                h1_structure=h1_structure,
                h1_zones=h1_zones,
                market_regime=market_regime,
                volume_report=volume_report,
                features=features,
                market=market_ctx
            )

        except Exception as e:
            logger.error(f"[Pipeline] Snapshot computation failed for {symbol}: {e}", exc_info=True)
            return None

    # ── MKE Helper ─────────────────────────────────────────────────────────

    def _compute_mke_structure(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        confirmed_only_bars: bool = False
    ) -> Tuple[StructureState, List[ResearchEvent]]:
        if df is None or len(df) < 20:
            return StructureState(), []

        # If confirmed_only_bars, exclude the last bar (which is active/developing)
        df_to_analyze = df.iloc[:-1] if (confirmed_only_bars and len(df) > 1) else df

        # 1. Detect raw pivots
        swings = self.pivot_engine.detect_pivots(df_to_analyze, symbol, timeframe=timeframe)

        # 2. Detect clusters
        clusters = self.cluster_engine.detect_clusters(swings, symbol, timeframe=timeframe)

        # 3. Analyze structure
        state, events = self.structure_engine.analyze(df_to_analyze, swings, clusters, symbol, timeframe=timeframe)

        return state, events

    # ── Stage 1: Structure ────────────────────────────────────────────────

    def _stage_structure(self, d1, h1, symbol: str):
        """Daily bias, H1 structure report, H1 supply/demand zones."""
        daily_bias = QuantUtils.get_structural_bias(d1) if d1 is not None else "NEUTRAL"
        
        # Compute H1 developing state and wrap it in the legacy adapter for backward compatibility
        h1_state, h1_events = self._compute_mke_structure(h1, symbol, "h1", confirmed_only_bars=False)
        h1_structure = LegacyStructureReportAdapter(h1_state, h1_events)
        
        h1_zones = self.zone_engine.detect_zones(h1)
        return daily_bias, h1_structure, h1_zones

    # ── Stage 2: Volume ───────────────────────────────────────────────────

    def _stage_volume(self, m5, symbol: str):
        """Time-of-Day normalized RVOL."""
        return self.volume_engine.analyze(m5, symbol)

    # ── Stage 3: Regime ───────────────────────────────────────────────────

    def _stage_regime(self, m5) -> str:
        """Market regime classification (TREND_UP, RANGE, etc.)."""
        return self.regime_engine.detect_regime(m5)

    # ── Stage 4: Features ─────────────────────────────────────────────────

    def _stage_features(self, m5: pd.DataFrame, d1: Optional[pd.DataFrame]) -> FeatureStore:
        """
        Compute all indicator values into a FeatureStore.
        ─────────────────────────────────────────────────
        To add a new indicator (e.g. RSI, VWAP, ADX):
            1. Compute it here.
            2. Add it to the `data` dict with a clear key name.
            3. Nothing else changes — strategies read via features.get_float("rsi14").
        """
        # ATR (simple range-based approximation)
        atr = float(m5["high"].tail(14).mean() - m5["low"].tail(14).mean())

        # EMA values on close
        ema20 = float(m5["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(m5["close"].ewm(span=50, adjust=False).mean().iloc[-1])

        # Move quality metrics (used by StructuralStrategy filters)
        move_efficiency = float(QuantUtils.calculate_move_efficiency(m5, lookback=10))
        wickiness = float(QuantUtils.calculate_wickiness(m5, lookback=5))

        # EMA cross direction (convenience boolean for EMA strategy)
        ema_bullish = ema20 > ema50

        # ── VWAP Distance ──
        distance_to_vwap = 0.0
        try:
            last_date = m5.index[-1].date()
            day_mask = m5.index.date == last_date
            day_m5 = m5[day_mask]
            if len(day_m5) > 0:
                typical_price = (day_m5["high"] + day_m5["low"] + day_m5["close"]) / 3.0
                pv = typical_price * day_m5["volume"]
                cum_pv = pv.cumsum()
                cum_vol = day_m5["volume"].cumsum()
                latest_cum_vol = cum_vol.iloc[-1]
                vwap = float(cum_pv.iloc[-1] / latest_cum_vol) if latest_cum_vol > 0 else 0.0
                close = float(m5["close"].iloc[-1])
                distance_to_vwap = (close - vwap) / vwap if vwap > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error calculating VWAP distance: {e}")

        # ── ATR Percentile ──
        atr_percentile = 0.5
        try:
            high = m5["high"]
            low = m5["low"]
            close_prev = m5["close"].shift(1)
            tr1 = high - low
            tr2 = (high - close_prev).abs()
            tr3 = (low - close_prev).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_series = tr.rolling(window=14).mean()
            current_atr = atr_series.iloc[-1]
            atr_lookback = atr_series.tail(250).dropna()
            if len(atr_lookback) > 1:
                atr_percentile = float((atr_lookback < current_atr).sum() / len(atr_lookback))
        except Exception as e:
            logger.warning(f"Error calculating ATR percentile: {e}")

        # ── Previous Day Range Distances ──
        dist_prev_high = 0.0
        dist_prev_low = 0.0
        try:
            if d1 is not None and len(d1) >= 2:
                last_daily_date = d1.index[-1].date()
                current_date = m5.index[-1].date()
                if last_daily_date == current_date:
                    prev_day_row = d1.iloc[-2]
                else:
                    prev_day_row = d1.iloc[-1]
                prev_high = float(prev_day_row["high"])
                prev_low = float(prev_day_row["low"])
                close = float(m5["close"].iloc[-1])
                dist_prev_high = (close - prev_high) / prev_high if prev_high > 0 else 0.0
                dist_prev_low = (close - prev_low) / prev_low if prev_low > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error calculating previous day distances: {e}")

        data = {
            "atr":                    round(atr, 2),
            "ema20":                  round(ema20, 2),
            "ema50":                  round(ema50, 2),
            "move_efficiency":        round(move_efficiency, 4),
            "wickiness":              round(wickiness, 4),
            "ema_bullish":            ema_bullish,
            "distance_to_vwap":       round(distance_to_vwap, 4),
            "atr_percentile":         round(atr_percentile, 4),
            "dist_prev_high":         round(dist_prev_high, 4),
            "dist_prev_low":          round(dist_prev_low, 4),
        }

        return FeatureStore(data)
