#!/usr/bin/env python3
"""
Cluster Engine (MKE Stage 1 Context)
====================================
Groups adjacent equal highs/lows into LiquidityCluster entities.
Helps strategies identify major horizontal liquidity pools.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np

from src.core.market_knowledge import SwingPoint, LiquidityCluster

logger = logging.getLogger(__name__)


class ClusterEngine:
    """Groups swing points into liquidity clusters based on price proximity."""
    required_history = 120

    def __init__(self, tolerance_pct: float = 0.0005):
        self.tolerance = tolerance_pct

    def detect_clusters(
        self,
        swings: List[SwingPoint],
        symbol: str,
        timeframe: str = "m5"
    ) -> List[LiquidityCluster]:
        """
        Groups SwingPoints of the same type (HIGH -> EQH, LOW -> EQL) 
        whose prices are within tolerance_pct.
        """
        if len(swings) < 2:
            return []

        clean_symbol = symbol.replace(":", "_").replace("-", "_")
        high_swings = [s for s in swings if s.type == "HIGH"]
        low_swings = [s for s in swings if s.type == "LOW"]

        clusters: List[LiquidityCluster] = []

        # Process High Swings (EQH)
        clusters.extend(self._find_clusters_for_type(high_swings, "EQH", clean_symbol, timeframe))

        # Process Low Swings (EQL)
        clusters.extend(self._find_clusters_for_type(low_swings, "EQL", clean_symbol, timeframe))

        return clusters

    def _find_clusters_for_type(
        self,
        pivots: List[SwingPoint],
        cluster_type: str,
        clean_symbol: str,
        timeframe: str
    ) -> List[LiquidityCluster]:
        """Core clustering algorithm."""
        if len(pivots) < 2:
            return []

        # Sort pivots by price to group close ones
        sorted_pivots = sorted(pivots, key=lambda p: p.price)
        grouped_pivots: List[List[SwingPoint]] = []
        
        current_group: List[SwingPoint] = [sorted_pivots[0]]
        for p in sorted_pivots[1:]:
            # Check price distance relative to the group's first element price
            base_price = current_group[0].price
            if (p.price - base_price) / base_price <= self.tolerance:
                current_group.append(p)
            else:
                if len(current_group) >= 2:
                    grouped_pivots.append(current_group)
                current_group = [p]
        
        if len(current_group) >= 2:
            grouped_pivots.append(current_group)

        clusters: List[LiquidityCluster] = []
        for group in grouped_pivots:
            # Sort group pivots by timestamp to find the latest
            group_sorted_by_time = sorted(group, key=lambda p: p.timestamp)
            last_ts = group_sorted_by_time[-1].timestamp
            ts_str = last_ts.strftime("%Y%m%d_%H%M%S")

            avg_price = float(np.mean([p.price for p in group]))
            member_ids = [p.id for p in group_sorted_by_time]
            
            # Strength increases with the number of touch points and average strength of member pivots
            avg_member_strength = float(np.mean([p.strength for p in group]))
            touches_factor = min(1.0, 0.5 + 0.25 * (len(group) - 2))
            strength = round(0.6 * avg_member_strength + 0.4 * touches_factor, 3)

            cluster_id = f"cluster_{clean_symbol}_{timeframe}_{cluster_type.lower()}_{int(avg_price)}_{ts_str}"
            clusters.append(LiquidityCluster(
                id=cluster_id,
                price=round(avg_price, 2),
                type=cluster_type,
                member_swing_ids=member_ids,
                strength=strength,
                last_touched=last_ts
            ))

        return clusters
