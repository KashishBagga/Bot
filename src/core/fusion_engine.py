#!/usr/bin/env python3
"""
Fusion Engine (MKE Stage 5 — Geometry Layer)
=============================================
Combines raw HorizontalLevels and Trendlines (at price_at_now) into CompositeLevels
using role-aware hierarchical clustering and cross-tier attraction.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union, Set, FrozenSet

from src.core.market_geometry import (
    HorizontalLevel, Trendline, CompositeLevel, LevelDirection,
    LevelPriority, GeometryStatus, FormationReason, LevelType
)

logger = logging.getLogger(__name__)


@dataclass
class ClusterMember:
    price: float
    priority: LevelPriority
    direction: LevelDirection
    confidence: float
    item: Union[HorizontalLevel, Trendline]


class FusionEngine:
    """Combines levels and trendlines into CompositeLevels."""

    required_history = 1

    def fuse(
        self,
        levels: List[HorizontalLevel],
        trendlines: List[Trendline],
        atr: float,
    ) -> List[CompositeLevel]:
        """
        Cluster nearby levels and trendlines into CompositeLevels.

        Args:
            levels: Raw HorizontalLevel list
            trendlines: Projected Trendline list
            atr: Average True Range (for merge tolerance)

        Returns:
            List[CompositeLevel] — fuzzed zones.
        """
        # Step 1: Convert trendlines and levels to ClusterMembers
        members: List[ClusterMember] = []

        for hl in levels:
            members.append(ClusterMember(
                price=hl.price,
                priority=hl.priority,
                direction=LevelDirection.SUPPORT if hl.direction == LevelDirection.SUPPORT else LevelDirection.RESISTANCE,
                confidence=hl.confidence,
                item=hl
            ))

        for tl in trendlines:
            # Enforce that trendline status is not broken
            if tl.status == GeometryStatus.BROKEN:
                continue
            direction = LevelDirection.SUPPORT if tl.role.value == "SUPPORT" else LevelDirection.RESISTANCE
            members.append(ClusterMember(
                price=tl.price_at_now,
                priority=LevelPriority.TECHNICAL,
                direction=direction,
                confidence=tl.confidence,
                item=tl
            ))

        # Step 2: Split by direction (SUPPORT / RESISTANCE)
        support_members = [m for m in members if m.direction == LevelDirection.SUPPORT]
        resistance_members = [m for m in members if m.direction == LevelDirection.RESISTANCE]

        composites: List[CompositeLevel] = []

        # Run hierarchical clustering for each direction
        composites.extend(self._cluster_direction(support_members, atr, LevelDirection.SUPPORT))
        composites.extend(self._cluster_direction(resistance_members, atr, LevelDirection.RESISTANCE))

        return composites

    def _cluster_direction(
        self,
        members: List[ClusterMember],
        atr: float,
        direction: LevelDirection,
    ) -> List[CompositeLevel]:
        if not members:
            return []

        # Tolerances per priority tier
        tolerances = {
            LevelPriority.INSTITUTIONAL: 0.15 * atr,
            LevelPriority.STRUCTURAL: 0.25 * atr,
            LevelPriority.TECHNICAL: 0.35 * atr,
        }

        # Step A: Split members by priority tier
        inst_pool = [m for m in members if m.priority == LevelPriority.INSTITUTIONAL]
        struct_pool = [m for m in members if m.priority == LevelPriority.STRUCTURAL]
        tech_pool = [m for m in members if m.priority == LevelPriority.TECHNICAL]

        # We will form clusters hierarchically
        clusters: List[List[ClusterMember]] = []

        # Helper to check if a member is absorbed by any existing cluster
        def try_absorb(m: ClusterMember, existing_clusters: List[List[ClusterMember]]) -> bool:
            for cluster in existing_clusters:
                # Get the highest priority of any member in this cluster
                highest_priority = min(c.priority for c in cluster)
                tol = tolerances[highest_priority]
                
                # Check price range of the cluster
                prices = [c.price for c in cluster]
                min_p, max_p = min(prices), max(prices)
                
                # Absorb if within the band or within the tolerance of the cluster average price
                avg_p = sum(prices) / len(prices)
                if (min_p - tol <= m.price <= max_p + tol) or (abs(m.price - avg_p) <= tol):
                    cluster.append(m)
                    return True
            return False

        # Greedy cluster helper within a pool
        def greedy_cluster(pool: List[ClusterMember], tol: float) -> List[List[ClusterMember]]:
            if not pool:
                return []
            sorted_pool = sorted(pool, key=lambda m: m.price)
            res = [[sorted_pool[0]]]
            for m in sorted_pool[1:]:
                # If close to the last cluster's average/extreme
                last_cluster = res[-1]
                avg_p = sum(c.price for c in last_cluster) / len(last_cluster)
                if m.price - avg_p <= tol:
                    last_cluster.append(m)
                else:
                    res.append([m])
            return res

        # ── 1. Cluster INSTITUTIONAL pool ──────────────────────────────────────
        inst_clusters = greedy_cluster(inst_pool, tolerances[LevelPriority.INSTITUTIONAL])
        clusters.extend(inst_clusters)

        # ── 2. Absorb STRUCTURAL members or cluster remaining ──────────────────
        struct_remaining: List[ClusterMember] = []
        for m in struct_pool:
            if not try_absorb(m, clusters):
                struct_remaining.append(m)

        struct_clusters = greedy_cluster(struct_remaining, tolerances[LevelPriority.STRUCTURAL])
        clusters.extend(struct_clusters)

        # ── 3. Absorb TECHNICAL members or cluster remaining ───────────────────
        tech_remaining: List[ClusterMember] = []
        for m in tech_pool:
            if not try_absorb(m, clusters):
                tech_remaining.append(m)

        tech_clusters = greedy_cluster(tech_remaining, tolerances[LevelPriority.TECHNICAL])
        clusters.extend(tech_clusters)

        # ── 4. Build CompositeLevels from clusters ──────────────────────────────
        composite_levels: List[CompositeLevel] = []

        for cluster in clusters:
            if not cluster:
                continue

            # Filter: max confidence >= 0.4 and at least one raw level (as per plan step 7)
            # Wait, let's unpack levels and trendlines
            raw_levels = tuple(c.item for c in cluster if isinstance(c.item, HorizontalLevel))
            raw_trendlines = tuple(c.item for c in cluster if isinstance(c.item, Trendline))

            # Filter: keep only composites where: len(raw_levels) >= 1 AND max(member.confidence) >= 0.4
            max_conf = max(c.confidence for c in cluster)
            if len(raw_levels) < 1 or max_conf < 0.4:
                continue

            # Compute confidence-weighted average price
            total_weight = sum(c.confidence for c in cluster)
            if total_weight > 0:
                price = sum(c.price * c.confidence for c in cluster) / total_weight
            else:
                price = sum(c.price for c in cluster) / len(cluster)

            prices = [c.price for c in cluster]
            band_low = min(prices)
            band_high = max(prices)
            width = band_high - band_low

            # Priority is the highest priority (lowest IntEnum value)
            priority = min(c.priority for c in cluster)

            # Gather member types
            member_types: List[str] = []
            for c in cluster:
                if isinstance(c.item, HorizontalLevel):
                    member_types.append(c.item.type.value)
                elif isinstance(c.item, Trendline):
                    member_types.append("TRENDLINE")

            # Determine formation reasons
            reasons = set()
            if len(cluster) > 1:
                reasons.add(FormationReason.PRICE_PROXIMITY)
            if any(isinstance(c.item, HorizontalLevel) and c.item.type == LevelType.ROUND_NUMBER for c in cluster):
                reasons.add(FormationReason.ROUND_NUMBER_CLUSTER)
            if len(raw_trendlines) > 0 and len(raw_levels) > 0:
                reasons.add(FormationReason.TRENDLINE_INTERSECTION)
            if any(isinstance(c.item, HorizontalLevel) and c.item.role_reversal for c in cluster):
                reasons.add(FormationReason.ROLE_REVERSAL)
            if any(isinstance(c.item, HorizontalLevel) and "LIQUIDITY" in c.item.provenance.get("source", "") for c in cluster):
                reasons.add(FormationReason.LIQUIDITY_CLUSTER)

            # Build geometry relations graph: member_id -> [other_member_ids]
            member_ids = []
            for c in cluster:
                member_ids.append(c.item.id)

            geometry_relations: Dict[str, List[str]] = {}
            for i, m_id in enumerate(member_ids):
                geometry_relations[m_id] = [other_id for j, other_id in enumerate(member_ids) if i != j]

            # Generate stable ID: cl_ + sha256(sorted member IDs)[:12]
            cl_id = CompositeLevel.make_id(member_ids)

            composite_levels.append(CompositeLevel(
                id=cl_id,
                price=round(price, 2),
                band_low=round(band_low, 2),
                band_high=round(band_high, 2),
                width=round(width, 2),
                direction=direction,
                priority=priority,
                status=GeometryStatus.ACTIVE,
                confidence=round(max_conf, 3),
                raw_levels=raw_levels,
                raw_trendlines=raw_trendlines,
                member_types=tuple(member_types),
                formation_reasons=frozenset(reasons),
                geometry_relations=geometry_relations,
                provenance={"source": "FUSION_ENGINE"}
            ))

        return composite_levels
