#!/usr/bin/env python3
"""
IndicatorPipeline — Compute all indicators once per symbol per candle.
======================================================================
Receives raw OHLCV DataFrames from the data provider and produces a
fully decorated MarketSnapshot for all strategies to consume.

No API calls. No strategy logic. Pure computation.
"""

import logging
from datetime import datetime, time
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd

from src.core.pattern_engine import PatternEngine
from src.core.liquidity_engine import LiquidityEngine
from src.core.market_facts import MarketFacts
from src.core.trading_clock import TradingClock


from src.core.feature_store import FeatureStore
from src.core.market_snapshot import MarketSnapshot, OptionSnapshotRow
from src.core.structure_engine import StructureEngine
from src.core.zone_engine import ZoneEngine, confirm_zones_with_geometry
from src.core.volume_engine import VolumeEngine
from src.core.regime_engine import RegimeEngine
from src.core.quant_utils import QuantUtils
from src.core.level_engine import LevelEngine
from src.core.trendline_engine import TrendlineEngine
from src.core.fusion_engine import FusionEngine
from src.core.confluence_engine import ConfluenceEngine
from src.core.narrative_engine import NarrativeEngine
from src.core.market_geometry import GeometryContext, CompositeLevel, Trendline, ConfluenceZone, Channel
from src.core.options_intelligence_engine import OptionsIntelligenceEngine

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
        db=None,
    ):
        self.tod_engine = TimeOfDayEngine()
        self.pivot_engine = PivotEngine(pivot_window=pivot_window)
        self.cluster_engine = ClusterEngine()
        self.structure_engine = StructureEngine(pivot_window=pivot_window)
        self.zone_engine = ZoneEngine(cluster_pct=zone_cluster_pct)
        self.zone_engine.MIN_ZONE_SCORE = min_zone_score
        self.volume_engine = VolumeEngine(historical_days=historical_days)
        self.regime_engine = RegimeEngine()
        
        self.level_engine = LevelEngine()
        self.trendline_engine = TrendlineEngine()
        self.fusion_engine = FusionEngine()
        self.confluence_engine = ConfluenceEngine()
        self.narrative_engine = NarrativeEngine()
        self.pattern_engine = PatternEngine()
        self.liquidity_engine = LiquidityEngine()
        self.options_engine = OptionsIntelligenceEngine()

        # Persists research events (market_events) and reads live option-chain
        # data (get_option_chain_snapshot/get_atm_oi_series). Real PostgresDatabase
        # by default — callers replaying HISTORICAL candles (the backtester) should
        # inject a stub here instead: writing real-time-shaped events for a
        # replayed 2026-06-01 candle would pollute the live market_events table,
        # and get_option_chain_snapshot/get_atm_oi_series would return TODAY's
        # option data, not point-in-time-correct data for the replayed candle.
        # See src/backtesting/backtest_db_stub.py.
        self.db = db if db is not None else PostgresDatabase()


    @property
    def required_history(self) -> int:
        """Dynamically resolve the maximum required history length across all MKE/Indicator engines."""
        return max(
            self.structure_engine.required_history,
            self.pivot_engine.required_history,
            self.cluster_engine.required_history,
            self.tod_engine.required_history,
            self.trendline_engine.required_history
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
            # ── Stage 5: Market Context Engine (MKE) ──
            market_ctx = MarketContext()

            # Compute m5 structure (primary execution timeframe)
            m5_state, m5_events = self._compute_mke_structure(m5, symbol, "m5", confirmed_only_bars=False)
            market_ctx.structure = m5_state

            # Save m5 confirmed events
            for event in m5_events:
                self.db.save_market_event(event.to_dict())

            # Compute h1/d1 structure once, up front. Reused for: htf_structure
            # below, _stage_structure's legacy adapter/h1_zones, AND the MTF
            # geometry merge (levels/trendlines) a few lines down — previously
            # h1's "developing" state was computed twice per candle (identical
            # params, here and again later) purely because htf_structure was
            # assembled after _stage_structure had already run its own copy.
            h1_conf_state, h1_conf_events = self._compute_mke_structure(h1, symbol, "h1", confirmed_only_bars=True)
            h1_dev_state, h1_dev_events = self._compute_mke_structure(h1, symbol, "h1", confirmed_only_bars=False)

            d1_conf_state, d1_dev_state, d1_conf_events = None, None, []
            if d1 is not None and len(d1) >= 20:
                d1_conf_state, d1_conf_events = self._compute_mke_structure(d1, symbol, "d1", confirmed_only_bars=True)
                d1_dev_state, _ = self._compute_mke_structure(d1, symbol, "d1", confirmed_only_bars=False)

            # Stage 1: Market structure (structural bias, zones)
            daily_bias, h1_structure, h1_zones = self._stage_structure(d1, h1, symbol, h1_dev_state, h1_dev_events)

            # Multi-timeframe zone detection. h1_zones above is UNCHANGED — every
            # existing strategy reading snapshot.h1_zones keeps its exact current
            # behavior. m5_zones/d1_zones are new, additive fields (see
            # MarketSnapshot below); nothing reads them yet except _persist_sr_zones.
            m5_zones = self.zone_engine.detect_zones(m5, timeframe="m5", atr_pct_of_price=self._atr_pct_of_price(m5))
            d1_zones = self.zone_engine.detect_zones(d1, timeframe="d1", atr_pct_of_price=self._atr_pct_of_price(d1)) if d1 is not None else []

            # Stage 2: Volume participation
            volume_report = self._stage_volume(m5, symbol)

            # Stage 3: Market regime (enhanced — ADX + gap + volatility state + MTF context)
            regime_label  = self._stage_regime(m5, d1=d1, h1_structure=h1_structure, now=timestamp)
            market_regime = regime_label.label

            # Stage 4: Feature store (ATR, EMAs, derived metrics)
            features = self._stage_features(m5, d1)

            # Build MarketFacts shared context
            t_time = timestamp.time()
            session = "MID"
            if t_time >= time(9, 15) and t_time < time(10, 0):
                session = "OPEN"
            elif t_time >= time(14, 45) and t_time <= time(15, 30):
                session = "CLOSE"
                
            is_open_blackout = (t_time >= time(9, 15) and t_time < time(9, 30))
            is_close_blackout = (t_time >= time(15, 15) and t_time <= time(15, 30))
            
            facts = MarketFacts(
                symbol=symbol,
                current_price=price,
                current_bar=TradingClock.trading_bar_id(timestamp),
                atr=features.get_float("atr"),
                tick_size=0.05,
                timestamp=timestamp,
                session=session,
                is_open_blackout=is_open_blackout,
                is_close_blackout=is_close_blackout,
                swings=tuple(m5_state.swings),
                completed_legs=tuple(m5_state.legs),
                developing_leg=m5_state.developing_leg,
                clusters=tuple(m5_state.clusters),
                relationships=m5_state.relationships,
                rvol_tod=volume_report.rvol_tod if hasattr(volume_report, 'rvol_tod') else 1.0,
                atr_percentile=features.get_float("atr_percentile"),
                last_swing_high=m5_state.last_swing_high,
                last_swing_low=m5_state.last_swing_low,
                is_compressed=m5_state.is_compressed
            )
            
            # Compute Stage 5 Geometry raw parts — MTF-aware: m5 structural
            # levels/trendlines remain the primary confluence input, with h1/d1
            # structural levels and trendlines merged in before fusion so a Daily
            # swing overlapping a 1H swing overlapping a 5M trendline scores
            # higher in confluence than any single-timeframe hit alone.
            composites, trendlines, support_confluence, resistance_confluence, geo_events, channels = self._stage_geometry(
                m5=m5, h1=h1, d1=d1, structure=m5_state, features=features,
                current_price=price, symbol=symbol, now=timestamp,
                h1_structure=h1_conf_state, d1_structure=d1_conf_state,
            )
            
            # Stage 6: PatternEngine
            patterns_ctx = self.pattern_engine.detect(
                facts=facts,
                composites=composites,
                trendlines=trendlines,
                support_confluence=support_confluence,
                resistance_confluence=resistance_confluence
            )
            market_ctx.patterns = patterns_ctx
            
            # Stage 7: LiquidityEngine
            liquidity_ctx = self.liquidity_engine.analyze(
                facts=facts,
                composites=composites,
                trendlines=trendlines,
                patterns=patterns_ctx,
                m5=m5,
                d1=d1
            )
            market_ctx.liquidity = liquidity_ctx

            # Stage 8: OptionsIntelligenceEngine — real PCR/max-pain/OI walls, fed
            # from option_snapshots (populated by the OptionWarehouse background
            # service). Never lets missing/stale option data break the snapshot —
            # on any failure this stays None and strategies must treat that the
            # same as "no options data", not fabricate a neutral reading.
            try:
                chain_rows = self.db.get_option_chain_snapshot(symbol)
                market_ctx.options = self.options_engine.analyze(symbol, chain_rows, now=timestamp)
            except Exception as e:
                logger.debug(f"⚠️ OptionsIntelligenceEngine failed for {symbol}: {e}")
                market_ctx.options = None

            # Stage 8b: ATM ± depth option chain rows for OptionsScalpingStrategy.
            # Same "never break the snapshot on missing options data" contract as
            # Stage 8 — on any failure this stays None/empty.
            try:
                atm_rows = self.db.get_atm_oi_series(symbol, minutes=10)
                atm_chain = [OptionSnapshotRow(**row) for row in atm_rows]
            except Exception as e:
                logger.debug(f"⚠️ ATM OI series fetch failed for {symbol}: {e}")
                atm_chain = None

            # Stage 9: NarrativeEngine
            geo_ctx_temp = GeometryContext(
                composites=composites,
                trendlines=trendlines,
                current_price=price,
                support_confluence=support_confluence,
                resistance_confluence=resistance_confluence
            )
            
            narrative = self.narrative_engine.synthesize(
                structure=m5_state,
                levels_view=geo_ctx_temp.levels,
                trendlines_view=geo_ctx_temp.trendlines,
                support_confluence=support_confluence,
                resistance_confluence=resistance_confluence,
                regime={"volatility_state": market_regime},
                current_price=price,
                patterns=patterns_ctx,
                liquidity=liquidity_ctx
            )
            
            # Stage 10: GeometryContext built
            geometry_ctx = GeometryContext(
                composites=composites,
                trendlines=trendlines,
                current_price=price,
                support_confluence=support_confluence,
                resistance_confluence=resistance_confluence,
                narrative=narrative,
                pending_events=geo_events,
                channels=channels,
            )
            market_ctx.geometry = geometry_ctx

            # Cross-reference zone_engine's independently-computed swing/RVOL
            # zones against this same geometry pass's confluence levels — a
            # zone confirmed by both is a materially stronger S/R read than
            # either alone. Boosts scores only, never rejects a zone, so
            # existing h1_zones/m5_zones/d1_zones consumers see no regression.
            h1_zones = confirm_zones_with_geometry(h1_zones, geometry_ctx.levels)
            m5_zones = confirm_zones_with_geometry(m5_zones, geometry_ctx.levels)
            d1_zones = confirm_zones_with_geometry(d1_zones, geometry_ctx.levels)

            # Save geometry pending events
            for event in geo_events:
                self.db.save_market_event(event.to_dict())
                
            # Save pattern transition events
            for event in patterns_ctx.transition_events:
                research_event = event.to_research_event(timestamp)
                self.db.save_market_event(research_event.to_dict())

            # Save liquidity transition events
            for event in liquidity_ctx.transition_events:
                research_event = event.to_research_event(timestamp)
                self.db.save_market_event(research_event.to_dict())


            # HTF structures — h1_conf_state/h1_dev_state/d1_conf_state/d1_dev_state
            # were already computed once, earlier in this method (also reused for
            # the MTF geometry merge above); assign here rather than recomputing.
            market_ctx.htf_structure["h1"] = HTFStructure(confirmed=h1_conf_state, developing=h1_dev_state)

            # Save h1 confirmed events
            for event in h1_conf_events:
                self.db.save_market_event(event.to_dict())

            # Save d1 HTF structures if available
            if d1_conf_state is not None:
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
                m5_zones=m5_zones,
                d1_zones=d1_zones,
                market_regime=market_regime,
                volume_report=volume_report,
                regime_detail=regime_label,
                features=features,
                market=market_ctx,
                atm_chain=atm_chain
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

    def _stage_structure(self, d1, h1, symbol: str, h1_state: StructureState, h1_events: List[ResearchEvent]):
        """Daily bias, H1 structure report, H1 supply/demand zones.

        h1_state/h1_events are precomputed by compute() (confirmed_only_bars=False)
        and passed in rather than recomputed here — this used to run the identical
        _compute_mke_structure(h1, ..., confirmed_only_bars=False) call twice per
        candle (once here, once again for market_ctx.htf_structure["h1"].developing).
        """
        daily_bias = QuantUtils.get_structural_bias(d1) if d1 is not None else "NEUTRAL"
        h1_structure = LegacyStructureReportAdapter(h1_state, h1_events)
        h1_atr_pct = self._atr_pct_of_price(h1)
        h1_zones = self.zone_engine.detect_zones(h1, timeframe="h1", atr_pct_of_price=h1_atr_pct)
        return daily_bias, h1_structure, h1_zones

    # ── Stage 2: Volume ───────────────────────────────────────────────────

    def _stage_volume(self, m5, symbol: str):
        """Time-of-Day normalized RVOL."""
        return self.volume_engine.analyze(m5, symbol)

    # ── Stage 3: Regime ───────────────────────────────────────────────────

    def _stage_regime(self, m5, d1=None, h1_structure=None, now=None):
        """Market regime classification (enhanced: ADX, gap, ATR percentile,
        h1/d1 MTF context).

        Returns a RegimeLabel dataclass. Callers that only need the string
        should call regime_label.label for backward compatibility.
        """
        return self.regime_engine.detect_regime(m5, d1=d1, h1_structure=h1_structure, now=now)

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
        # ── ATR (True Range, 14-period rolling mean) ──────────────────────
        # BUG FIX: Previous formula was mean(highs) - mean(lows) which ignores
        # overnight gaps and is NOT a valid ATR. True Range = max(H-L, |H-PC|, |L-PC|)
        close_prev = m5["close"].shift(1)
        tr_series = pd.concat([
            m5["high"] - m5["low"],
            (m5["high"] - close_prev).abs(),
            (m5["low"]  - close_prev).abs(),
        ], axis=1).max(axis=1)
        atr_rolling = tr_series.rolling(window=14).mean()
        atr = float(atr_rolling.iloc[-1]) if not pd.isna(atr_rolling.iloc[-1]) else float(m5["high"].tail(14).mean() - m5["low"].tail(14).mean())

        # EMA values on close
        ema20 = float(m5["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(m5["close"].ewm(span=50, adjust=False).mean().iloc[-1])

        # Move quality metrics (used by StructuralStrategy filters)
        move_efficiency = float(QuantUtils.calculate_move_efficiency(m5, lookback=10))
        # Longer-lookback efficiency for H1-timeframe strategies (2h = 24 bars of 5m)
        move_efficiency_h1 = float(QuantUtils.calculate_move_efficiency(m5, lookback=24))
        wickiness = float(QuantUtils.calculate_wickiness(m5, lookback=5))

        # EMA cross direction (convenience boolean for EMA strategy)
        ema_bullish = ema20 > ema50

        # ── RSI-14 ─────────────────────────────────────────────────────────
        rsi14 = 50.0
        try:
            delta = m5["close"].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, float('nan'))
            rsi_series = 100 - (100 / (1 + rs))
            val = rsi_series.iloc[-1]
            rsi14 = float(val) if not pd.isna(val) else 50.0
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")

        # ── RSI-2 ──────────────────────────────────────────────────────────
        # Short-period RSI (Connors-style) — swings to genuine extremes
        # (<10 / >90) on ordinary bars, unlike RSI-14 which rarely gets past
        # 20/80 on 5m index data. Used for short-term mean-reversion, not
        # trend/momentum reads — that's what rsi14 is for.
        rsi2 = 50.0
        try:
            delta2 = m5["close"].diff()
            gain2 = delta2.clip(lower=0).rolling(window=2).mean()
            loss2 = (-delta2.clip(upper=0)).rolling(window=2).mean()
            rs2 = gain2 / loss2.replace(0, float('nan'))
            rsi2_series = 100 - (100 / (1 + rs2))
            val2 = rsi2_series.iloc[-1]
            rsi2 = float(val2) if not pd.isna(val2) else 50.0
        except Exception as e:
            logger.warning(f"Error calculating RSI-2: {e}")

        # ── VWAP Distance ──────────────────────────────────────────────────
        distance_to_vwap = 0.0
        try:
            last_date = m5.index[-1].date()
            day_mask = m5.index.date == last_date
            day_m5 = m5[day_mask]
            # BUG FIX: Need at least 8 bars (~40 min) for a meaningful intraday VWAP
            if len(day_m5) >= 8:
                typical_price = (day_m5["high"] + day_m5["low"] + day_m5["close"]) / 3.0
                pv = typical_price * day_m5["volume"]
                cum_pv = pv.cumsum()
                cum_vol = day_m5["volume"].cumsum()
                latest_cum_vol = cum_vol.iloc[-1]
                vwap = float(cum_pv.iloc[-1] / latest_cum_vol) if latest_cum_vol > 0 else 0.0
                close = float(m5["close"].iloc[-1])
                distance_to_vwap = (close - vwap) / vwap if vwap > 0 else 0.0
            # else: distance_to_vwap stays 0.0 (neutral) — VWAP not yet meaningful
        except Exception as e:
            logger.warning(f"Error calculating VWAP distance: {e}")

        # ── ATR Percentile ── reuse the True Range series already computed above
        atr_percentile = 0.5
        try:
            atr_lookback = atr_rolling.tail(250).dropna()
            current_atr = atr_rolling.iloc[-1]
            if len(atr_lookback) > 1 and not pd.isna(current_atr):
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

        # ── ADX (14-period) — reuse RegimeEngine static helper ─────────────
        adx_val   = 0.0
        adx_slope = 0.0
        try:
            from src.core.regime_engine import RegimeEngine as _RE
            adx_val, adx_slope = _RE._compute_adx(m5, period=14)
        except Exception as _e:
            logger.warning(f"FeatureStore ADX failed: {_e}")

        # ── Opening gap (requires d1) ─────────────────────────────────────
        gap_pct = 0.0
        try:
            if d1 is not None:
                from src.core.regime_engine import RegimeEngine as _RE2
                _, gap_pct = _RE2._compute_gap(m5, d1)
        except Exception as _e:
            logger.warning(f"FeatureStore gap_pct failed: {_e}")

        data = {
            "atr":                    round(atr, 2),
            "ema20":                  round(ema20, 2),
            "ema50":                  round(ema50, 2),
            "move_efficiency":        round(move_efficiency, 4),
            "move_efficiency_h1":     round(move_efficiency_h1, 4),
            "wickiness":              round(wickiness, 4),
            "ema_bullish":            ema_bullish,
            "rsi14":                  round(rsi14, 2),
            "rsi2":                   round(rsi2, 2),
            "distance_to_vwap":       round(distance_to_vwap, 4),
            "atr_percentile":         round(atr_percentile, 4),
            "dist_prev_high":         round(dist_prev_high, 4),
            "dist_prev_low":          round(dist_prev_low, 4),
            # New: regime-related features for ablation & regime gates
            "adx":                    round(adx_val, 2),
            "adx_slope":              round(adx_slope, 4),
            "gap_pct":                round(gap_pct, 4),
        }

        return FeatureStore(data)

    @staticmethod
    def _compute_atr(df: Optional[pd.DataFrame], window: int = 14) -> float:
        """Generic True-Range ATR for any OHLCV dataframe (any timeframe).

        Same formula as the m5 ATR in _stage_features, generalized so h1/d1
        trendline lifecycle simulation (breach/touch tolerance) uses a tolerance
        scaled to THAT timeframe's bar size — an m5-scale ATR is far too tight
        to sensibly judge whether an hourly or daily candle "broke" a trendline.
        """
        if df is None or len(df) < 2:
            return 0.0
        close_prev = df["close"].shift(1)
        tr_series = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close_prev).abs(),
            (df["low"] - close_prev).abs(),
        ], axis=1).max(axis=1)
        atr_rolling = tr_series.rolling(window=window).mean()
        val = atr_rolling.iloc[-1]
        if not pd.isna(val):
            return float(val)
        # Not enough bars for a full rolling window yet — fall back to a simple range mean.
        return float(df["high"].tail(window).mean() - df["low"].tail(window).mean())

    @classmethod
    def _atr_pct_of_price(cls, df: Optional[pd.DataFrame]) -> Optional[float]:
        """ATR as a fraction of the last close — feeds ZoneEngine's
        volatility-adaptive clustering tolerance so zone width scales with
        how quiet/volatile the market actually is, instead of a fixed % band."""
        if df is None or len(df) < 2:
            return None
        last_close = df["close"].iloc[-1]
        if not last_close or last_close <= 0:
            return None
        return cls._compute_atr(df) / float(last_close)

    def _stage_geometry(
        self,
        m5: pd.DataFrame,
        h1: Optional[pd.DataFrame],
        d1: Optional[pd.DataFrame],
        structure: StructureState,
        features: FeatureStore,
        current_price: float,
        symbol: str,
        now: datetime,
        h1_structure: Optional[StructureState] = None,
        d1_structure: Optional[StructureState] = None,
    ) -> Tuple[List[CompositeLevel], List[Trendline], Optional[ConfluenceZone], Optional[ConfluenceZone], List[ResearchEvent], List["Channel"]]:
        """Runs the complete geometry pipeline and returns raw levels, trendlines, and confluence.

        MTF-aware: m5 structural levels/trendlines are always computed. If
        h1_structure/d1_structure are supplied, their swing-derived structural
        levels and trendlines are additionally detected and merged into the SAME
        levels/trendlines lists before a single fusion pass — genuine multi-
        timeframe confluence (FusionEngine clusters by price proximity regardless
        of which timeframe produced each item), not three independent passes.
        """
        atr = features.get_float("atr")
        if atr <= 0:
            atr = 1.0

        ema20 = features.get_float("ema20")
        ema50 = features.get_float("ema50")

        # Reconstruct VWAP price from distance_to_vwap if possible
        vwap = current_price
        try:
            dist_vwap = features.get_float("distance_to_vwap")
            vwap = current_price / (1.0 + dist_vwap)
        except Exception:
            pass

        # 1. Level detection (m5) — includes the singleton institutional/
        #    open-of-day/round-number/technical levels, detected exactly once.
        levels, level_events = self.level_engine.detect_levels(
            m5=m5, h1=h1, d1=d1, structure=structure, atr=atr,
            vwap=vwap, ema20=ema20, ema50=ema50, current_price=current_price,
            symbol=symbol, now=now
        )

        # 2. Trendline detection (m5)
        trendlines, tl_events = self.trendline_engine.detect_trendlines(
            m5=m5, structure=structure, atr=atr, current_price=current_price,
            symbol=symbol, now=now, timeframe="m5"
        )

        # 2b. MTF structural levels + trendlines (h1/d1) — additive merge, no
        # duplication of the m5-only singleton levels above.
        for tf_df, tf_structure, tf_name in ((h1, h1_structure, "h1"), (d1, d1_structure, "d1")):
            if tf_df is None or tf_structure is None or len(tf_structure.swings) < 3:
                continue
            tf_atr = self._compute_atr(tf_df)
            if tf_atr <= 0:
                continue

            tf_levels, tf_level_events = self.level_engine.detect_structural_levels(
                tf_structure, tf_df, current_price, symbol, now, timeframe=tf_name
            )
            levels.extend(tf_levels)
            level_events.extend(tf_level_events)

            tf_trendlines, tf_tl_events = self.trendline_engine.detect_trendlines(
                m5=tf_df, structure=tf_structure, atr=tf_atr, current_price=current_price,
                symbol=symbol, now=now, timeframe=tf_name
            )
            trendlines.extend(tf_trendlines)
            tl_events.extend(tf_tl_events)

        # 3. Fusion (clustering) — one pass over the merged MTF set. Clustering
        #    tolerance stays keyed off the m5 ATR (unchanged) — this sizes the
        #    final confluence bands to what's actionable on the execution
        #    timeframe, not a deliberate oversight.
        composites = self.fusion_engine.fuse(
            levels=levels, trendlines=trendlines, atr=atr
        )

        # 4. Confluence zones
        support_confluence, resistance_confluence = self.confluence_engine.calculate_confluence(
            composites=composites, trendlines=trendlines, current_price=current_price, atr=atr
        )

        # 5. Channels — parallel opposite-role trendline pairs, per timeframe.
        channels = self.trendline_engine.detect_channels(trendlines, current_price)

        return composites, trendlines, support_confluence, resistance_confluence, level_events + tl_events, channels

