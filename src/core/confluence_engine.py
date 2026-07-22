#!/usr/bin/env python3
"""
Confluence Engine (MKE Stage 5 — Geometry Layer)
=================================================
Calculates support and resistance ConfluenceZones near the current price.
Uses a tanh-normalized score to handle multiple inputs without scaling issues.
"""

import logging
import math
from typing import List, Tuple, Optional

from src.core.market_geometry import (
    CompositeLevel, Trendline, ConfluenceZone, ConfluenceComponent,
    LevelDirection, LevelPriority, GeometryStatus
)

logger = logging.getLogger(__name__)


class ConfluenceEngine:
    """Calculates ConfluenceZones from composites and standalone trendlines."""

    CONFLUENCE_MIDPOINT = 1.36   # Two high-confidence overlapping objects
    CONFLUENCE_THRESHOLD = 35.0  # Minimum score to expose a zone

    def calculate_confluence(
        self,
        composites: List[CompositeLevel],
        trendlines: List[Trendline],
        current_price: float,
        atr: float,
    ) -> Tuple[Optional[ConfluenceZone], Optional[ConfluenceZone]]:
        """
        Calculate one support ConfluenceZone (below current_price) and
        one resistance ConfluenceZone (above current_price).
        """
        if current_price <= 0 or atr <= 0:
            return None, None

        # Identify standalone trendlines (not member of any composite)
        fused_trendline_ids = set()
        for c in composites:
            for tl in c.raw_trendlines:
                fused_trendline_ids.add(tl.id)

        standalone_tls = [t for t in trendlines if t.id not in fused_trendline_ids and t.status != GeometryStatus.BROKEN]

        support_zone = self._calculate_zone(
            composites=[c for c in composites if c.direction == LevelDirection.SUPPORT],
            trendlines=[t for t in standalone_tls if t.role.value == "SUPPORT"],
            current_price=current_price,
            atr=atr,
            direction=LevelDirection.SUPPORT
        )

        resistance_zone = self._calculate_zone(
            composites=[c for c in composites if c.direction == LevelDirection.RESISTANCE],
            trendlines=[t for t in standalone_tls if t.role.value == "RESISTANCE"],
            current_price=current_price,
            atr=atr,
            direction=LevelDirection.RESISTANCE
        )

        return support_zone, resistance_zone

    def _calculate_zone(
        self,
        composites: List[CompositeLevel],
        trendlines: List[Trendline],
        current_price: float,
        atr: float,
        direction: LevelDirection,
    ) -> Optional[ConfluenceZone]:
        proximity_thresholds = {
            LevelPriority.INSTITUTIONAL: 0.35 * atr,
            LevelPriority.STRUCTURAL: 0.25 * atr,
            LevelPriority.TECHNICAL: 0.20 * atr,
        }
        standalone_tl_threshold = 0.25 * atr

        components: List[ConfluenceComponent] = []

        # 1. Process composites
        for comp in composites:
            dist = abs(comp.price - current_price)
            threshold = proximity_thresholds[comp.priority]

            if dist <= threshold:
                decay = 1.0 - (dist / threshold)
                contribution = comp.confidence * decay
                dist_pct = dist / current_price if current_price > 0 else 0.0

                label = comp.label
                explanation = f"{label} @ {comp.price} ({dist_pct * 100:.2f}% away, conf={comp.confidence:.2f})"

                components.append(ConfluenceComponent(
                    source_id=comp.id,
                    source_type=f"COMPOSITE_{comp.label.replace(' + ', '_')}",
                    contribution=round(contribution, 3),
                    distance_pct=round(dist_pct, 6),
                    explanation=explanation
                ))

        # 2. Process standalone trendlines
        for tl in trendlines:
            dist = abs(tl.price_at_now - current_price)
            threshold = standalone_tl_threshold

            if dist <= threshold:
                decay = 1.0 - (dist / threshold)
                contribution = tl.confidence * decay
                dist_pct = dist / current_price if current_price > 0 else 0.0

                explanation = f"Trendline @ {tl.price_at_now} ({dist_pct * 100:.2f}% away, conf={tl.confidence:.2f})"

                components.append(ConfluenceComponent(
                    source_id=tl.id,
                    source_type="TRENDLINE_" + tl.role.value,
                    contribution=round(contribution, 3),
                    distance_pct=round(dist_pct, 6),
                    explanation=explanation
                ))

        if not components:
            return None

        # Compute scores
        raw_sum = sum(comp.contribution for comp in components)
        total_score = 100.0 * math.tanh(raw_sum / self.CONFLUENCE_MIDPOINT)

        if total_score < self.CONFLUENCE_THRESHOLD:
            return None

        # Calculate weighted average price
        sum_contribution = sum(comp.contribution for comp in components)
        
        # Get prices/bands from actual objects to construct ConfluenceZone bands
        band_lows = []
        band_highs = []
        weighted_price_sum = 0.0

        for comp in components:
            # Find the corresponding composite or trendline
            matching_comp = next((c for c in composites if c.id == comp.source_id), None)
            if matching_comp:
                band_lows.append(matching_comp.band_low)
                band_highs.append(matching_comp.band_high)
                weighted_price_sum += matching_comp.price * comp.contribution
            else:
                matching_tl = next((t for t in trendlines if t.id == comp.source_id), None)
                if matching_tl:
                    band_lows.append(matching_tl.price_at_now)
                    band_highs.append(matching_tl.price_at_now)
                    weighted_price_sum += matching_tl.price_at_now * comp.contribution

        if sum_contribution > 0:
            zone_price = weighted_price_sum / sum_contribution
        else:
            zone_price = sum(band_lows) / len(band_lows)

        band_low = min(band_lows)
        band_high = max(band_highs)
        width = band_high - band_low

        # Build Explanation
        labels = []
        for comp in components:
            label = comp.source_type.replace("COMPOSITE_", "").replace("_", " ")
            if label not in labels:
                labels.append(label)
        
        exp_summary = " + ".join(labels)
        explanation = f"{exp_summary} @ {round(zone_price, 2)} (±{round(width/2.0, 1)}pts, score={int(total_score)})"

        return ConfluenceZone(
            price=round(zone_price, 2),
            band_low=round(band_low, 2),
            band_high=round(band_high, 2),
            width=round(width, 2),
            total_score=round(total_score, 1),
            direction=direction,
            components=tuple(components),
            explanation=explanation
        )
