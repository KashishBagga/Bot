#!/usr/bin/env python3
"""
Narrative Engine (MKE Stage 5 — Geometry Layer)
================================================
Synthesizes enums, confluence scores, and structure states into a unified,
traceable MarketNarrative description.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from src.core.market_geometry import (
    MarketNarrative, NarrativeEvidence, TrendBias, StructurePhase,
    VolatilityState, LiquidityState, NarrativeBias, ConfluenceZone,
    GeometryContext
)
from src.core.market_knowledge import StructureState

logger = logging.getLogger(__name__)


class NarrativeEngine:
    """Derives MarketNarrative with fully traceable sources."""

    required_history = 1

    def synthesize(
        self,
        structure: StructureState,
        levels_view: Any,          # LevelsView
        trendlines_view: Any,      # TrendlinesView
        support_confluence: Optional[ConfluenceZone],
        resistance_confluence: Optional[ConfluenceZone],
        regime: Optional[Dict[str, Any]],
        current_price: float,
        patterns: Optional[Any] = None, # PatternsContext
        liquidity: Optional[Any] = None, # LiquidityContext (M2C)
    ) -> MarketNarrative:
        """
        Synthesize the narrative for the current tick.
        """
        # 1. Derive trend
        trend, trend_ev = self._derive_trend(structure)

        # 2. Derive phase
        phase, phase_ev = self._derive_phase(structure, trend)

        # 3. Derive volatility
        vol, vol_ev = self._derive_volatility(regime)

        # 4. Derive liquidity
        liq, liq_ev = self._derive_liquidity(structure, current_price, liquidity)

        # 5. Derive bias
        bias, confidence, bias_ev = self._derive_bias(
            trend=trend,
            support_confluence=support_confluence,
            resistance_confluence=resistance_confluence,
            structure=structure
        )

        # 6. Extract nearest support/resistance labels
        nearest_sup = levels_view.nearest_support()
        nearest_res = levels_view.nearest_resistance()
        
        nearest_support_label = nearest_sup.label if nearest_sup else "None"
        nearest_resistance_label = nearest_res.label if nearest_res else "None"

        sup_score = support_confluence.total_score if support_confluence else 0.0
        res_score = resistance_confluence.total_score if resistance_confluence else 0.0

        dominant_pattern = None
        if patterns:
            most_conf = patterns.most_confident()
            if most_conf:
                dominant_pattern = most_conf.type.value

        return MarketNarrative(
            primary_trend=trend,
            secondary_structure=phase,
            volatility_state=vol,
            liquidity_state=liq,
            nearest_support_label=nearest_support_label,
            nearest_resistance_label=nearest_resistance_label,
            support_confluence_score=sup_score,
            resistance_confluence_score=res_score,
            dominant_pattern=dominant_pattern,
            bias=bias,
            bias_confidence=round(confidence, 3),
            evidence={
                "primary_trend": trend_ev,
                "secondary_structure": phase_ev,
                "volatility_state": vol_ev,
                "liquidity_state": liq_ev,
                "bias": bias_ev,
            }
        )


    def _derive_trend(self, structure: StructureState) -> Tuple[TrendBias, NarrativeEvidence]:
        last_high_label = None
        last_low_label = None
        
        for swing in reversed(structure.swings):
            rel = structure.relationships.get(swing.id)
            if rel:
                if swing.type == "HIGH" and last_high_label is None:
                    last_high_label = rel.label
                elif swing.type == "LOW" and last_low_label is None:
                    last_low_label = rel.label

        sources = []
        if last_high_label:
            sources.append(f"last_high_relation={last_high_label}")
        if last_low_label:
            sources.append(f"last_low_relation={last_low_label}")

        if not sources:
            sources.append("no_swings_in_history")
            return TrendBias.NEUTRAL, NarrativeEvidence(sources=tuple(sources))

        if last_high_label == "HH" and last_low_label == "HL":
            trend = TrendBias.BULLISH
        elif last_high_label == "LH" and last_low_label == "LL":
            trend = TrendBias.BEARISH
        else:
            # Check swing ratio counts to break tie
            hh_count = sum(1 for rel in structure.relationships.values() if rel.label == "HH")
            lh_count = sum(1 for rel in structure.relationships.values() if rel.label == "LH")
            hl_count = sum(1 for rel in structure.relationships.values() if rel.label == "HL")
            ll_count = sum(1 for rel in structure.relationships.values() if rel.label == "LL")
            
            sources.append(f"hh={hh_count}, lh={lh_count}, hl={hl_count}, ll={ll_count}")
            
            if (hh_count + hl_count) > (lh_count + ll_count):
                trend = TrendBias.BULLISH
            elif (lh_count + ll_count) > (hh_count + hl_count):
                trend = TrendBias.BEARISH
            else:
                trend = TrendBias.NEUTRAL

        return trend, NarrativeEvidence(sources=tuple(sources))

    def _derive_phase(self, structure: StructureState, trend: TrendBias) -> Tuple[StructurePhase, NarrativeEvidence]:
        sources = []
        if structure.is_compressed:
            sources.append("structure_is_compressed=True")
            return StructurePhase.COMPRESSION, NarrativeEvidence(sources=tuple(sources))

        phase = StructurePhase.CONTINUATION
        if structure.developing_leg:
            leg_type = "UP_LEG" if structure.developing_leg.current_price > structure.developing_leg.start_anchor.price else "DOWN_LEG"
            sources.append(f"developing_leg={leg_type}")
            sources.append(f"trend={trend.value}")
            
            if trend == TrendBias.BULLISH:
                phase = StructurePhase.CONTINUATION if leg_type == "UP_LEG" else StructurePhase.PULLBACK
            elif trend == TrendBias.BEARISH:
                phase = StructurePhase.CONTINUATION if leg_type == "DOWN_LEG" else StructurePhase.PULLBACK
            else:
                phase = StructurePhase.COMPRESSION
        else:
            sources.append("no_active_developing_leg")
            phase = StructurePhase.COMPRESSION

        return phase, NarrativeEvidence(sources=tuple(sources))

    def _derive_volatility(self, regime: Optional[Dict[str, Any]]) -> Tuple[VolatilityState, NarrativeEvidence]:
        sources = []
        vol_state = VolatilityState.NORMAL
        
        if regime and isinstance(regime, dict):
            reg_val = regime.get("volatility_state", "NORMAL").upper()
            sources.append(f"regime_volatility={reg_val}")
            if "COMPRESSED" in reg_val or "LOW" in reg_val:
                vol_state = VolatilityState.COMPRESSED
            elif "EXPANDING" in reg_val or "HIGH" in reg_val:
                vol_state = VolatilityState.EXPANDING
        else:
            sources.append("no_regime_context_provided")
            
        return vol_state, NarrativeEvidence(sources=tuple(sources))

    def _derive_liquidity(
        self,
        structure: StructureState,
        current_price: float,
        liquidity: Optional[Any] = None
    ) -> Tuple[LiquidityState, NarrativeEvidence]:
        sources = []
        liq = LiquidityState.CLEAN
        
        for cluster in structure.clusters:
            if cluster.type == "EQH" and cluster.price > current_price:
                liq = LiquidityState.EQH_ABOVE
                sources.append(f"EQH_above={cluster.price}")
                break
            elif cluster.type == "EQL" and cluster.price < current_price:
                liq = LiquidityState.EQL_BELOW
                sources.append(f"EQL_below={cluster.price}")
                break

        # Enrich with Stage 7 Liquidity Engine order-flow context
        if liquidity and liquidity.liq_map:
            liq_map = liquidity.liq_map
            
            # Check for active sweep (occurring on the current bar)
            if liq_map.active_sweep:
                from src.core.market_liquidity import SweepType
                if liq_map.active_sweep.type == SweepType.BULLISH:
                    liq = LiquidityState.SWEPT_DOWN
                    sources.append(f"active_bullish_sweep_of_{liq_map.active_sweep.level_swept}")
                else:
                    liq = LiquidityState.SWEPT_UP
                    sources.append(f"active_bearish_sweep_of_{liq_map.active_sweep.level_swept}")
            
            # Append nearest imbalances and pressure state
            if liq_map.nearest_bullish_imbalance:
                sources.append(f"nearest_bullish_imbalance={liq_map.nearest_bullish_imbalance.bottom}-{liq_map.nearest_bullish_imbalance.top}")
            if liq_map.nearest_bearish_imbalance:
                sources.append(f"nearest_bearish_imbalance={liq_map.nearest_bearish_imbalance.bottom}-{liq_map.nearest_bearish_imbalance.top}")
            
            sources.append(f"pressure_state={liq_map.pressure_state.value}")

        if liq == LiquidityState.CLEAN and not sources:
            sources.append("no_untested_liquidity_clusters_nearby")

        return liq, NarrativeEvidence(sources=tuple(sources))

    def _derive_bias(
        self,
        trend: TrendBias,
        support_confluence: Optional[ConfluenceZone],
        resistance_confluence: Optional[ConfluenceZone],
        structure: StructureState
    ) -> Tuple[NarrativeBias, float, NarrativeEvidence]:
        sup_score = support_confluence.total_score if support_confluence else 0.0
        res_score = resistance_confluence.total_score if resistance_confluence else 0.0

        sources = [
            f"trend={trend.value}",
            f"support_confluence={sup_score}",
            f"resistance_confluence={res_score}"
        ]

        bias = NarrativeBias.NEUTRAL
        confidence = 0.5

        # Rule 1: High support confluence + low resistance confluence
        if sup_score >= 50 and res_score < 40:
            confidence = min(0.95, 0.5 + 0.5 * (sup_score - res_score) / 100.0)
            if trend == TrendBias.BULLISH:
                bias = NarrativeBias.CONTINUATION
            elif trend == TrendBias.BEARISH:
                bias = NarrativeBias.REVERSAL

        # Rule 2: High resistance confluence + low support confluence
        elif res_score >= 50 and sup_score < 40:
            confidence = min(0.95, 0.5 + 0.5 * (res_score - sup_score) / 100.0)
            if trend == TrendBias.BEARISH:
                bias = NarrativeBias.CONTINUATION
            elif trend == TrendBias.BULLISH:
                bias = NarrativeBias.REVERSAL

        # Rule 3: Trend alignment fallback
        else:
            if trend == TrendBias.BULLISH:
                bias = NarrativeBias.CONTINUATION
                confidence = 0.60
                sources.append("trend_continuation_fallback")
            elif trend == TrendBias.BEARISH:
                bias = NarrativeBias.CONTINUATION
                confidence = 0.60
                sources.append("trend_continuation_fallback")
            else:
                bias = NarrativeBias.NEUTRAL
                confidence = 0.50
                sources.append("neutral_no_clear_bias")

        return bias, confidence, NarrativeEvidence(sources=tuple(sources))
