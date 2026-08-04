#!/usr/bin/env python3
"""
Options Intelligence Engine
============================
Turns real option-chain OI (captured by src.warehouse.option_warehouse into
the option_snapshots table) into decision-usable signals: Put-Call Ratio,
max pain, and the nearest call/put OI walls (de-facto strike-based S/R).

This is the same math already used in the Option Intelligence dashboard page
(src/trading/pages/5_📊_Option_Intelligence.py — compute_pcr/compute_max_pain/
top-OI-strike walls), generalized into a reusable engine so strategies can see
it too, not just the UI.

PCR interpretation follows the existing dashboard convention (option sellers
are the informed side): heavy put OI below spot = sellers comfortable being
short puts there = they don't expect a fall past that level = bullish signal.
Heavy call OI above spot = symmetric bearish signal.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Same thresholds already used in the dashboard's pcr_interpretation().
PCR_BULLISH_THRESHOLD = 1.3
PCR_BEARISH_THRESHOLD = 0.7

# How many top-OI strikes per side count as "the wall" (matches dashboard's top_ce/top_pe).
TOP_OI_STRIKE_COUNT = 3

# A snapshot older than this is not fresh enough to trade on.
STALE_AFTER_SECONDS = 5 * 60


@dataclass(frozen=True)
class OiWall:
    strike: float
    oi: int


@dataclass(frozen=True)
class OptionsIntelligence:
    """Per-symbol options-derived market read, attached as snapshot.market.options."""
    underlying: str
    pcr: Optional[float]
    pcr_bias: str  # "BULLISH" | "BEARISH" | "NEUTRAL" | "UNKNOWN"
    max_pain_strike: Optional[float]
    call_oi_walls: List[OiWall] = field(default_factory=list)  # sorted by OI desc — nearest-strong = walls[0]
    put_oi_walls: List[OiWall] = field(default_factory=list)
    as_of: Optional[datetime] = None
    is_stale: bool = True

    @property
    def call_oi_wall(self) -> Optional[OiWall]:
        """Strongest call-OI wall (resistance) — None if no chain data."""
        return self.call_oi_walls[0] if self.call_oi_walls else None

    @property
    def put_oi_wall(self) -> Optional[OiWall]:
        """Strongest put-OI wall (support) — None if no chain data."""
        return self.put_oi_walls[0] if self.put_oi_walls else None


def compute_pcr(chain_rows: List[Dict[str, Any]]) -> Optional[float]:
    """Put-Call Ratio from OI (total put OI / total call OI)."""
    call_oi = sum(r["oi"] for r in chain_rows if r["option_type"] == "CE" and r["oi"])
    put_oi = sum(r["oi"] for r in chain_rows if r["option_type"] == "PE" and r["oi"])
    return round(put_oi / call_oi, 3) if call_oi else None


def compute_max_pain(chain_rows: List[Dict[str, Any]]) -> Optional[float]:
    """Strike that minimises total OI-weighted payout (seller max pain)."""
    strikes = sorted(set(r["strike"] for r in chain_rows))
    if not strikes:
        return None
    ce_rows = {r["strike"]: r["oi"] for r in chain_rows if r["option_type"] == "CE"}
    pe_rows = {r["strike"]: r["oi"] for r in chain_rows if r["option_type"] == "PE"}
    pain_map = {}
    for s in strikes:
        call_pain = sum(max(s - k, 0) * (ce_rows.get(k, 0) or 0) for k in strikes)
        put_pain = sum(max(k - s, 0) * (pe_rows.get(k, 0) or 0) for k in strikes)
        pain_map[s] = call_pain + put_pain
    return min(pain_map, key=pain_map.get)


def pcr_bias(pcr: Optional[float]) -> str:
    if pcr is None:
        return "UNKNOWN"
    if pcr > PCR_BULLISH_THRESHOLD:
        return "BULLISH"
    if pcr < PCR_BEARISH_THRESHOLD:
        return "BEARISH"
    return "NEUTRAL"


def top_oi_walls(chain_rows: List[Dict[str, Any]], option_type: str) -> List[OiWall]:
    by_strike = [
        (r["strike"], r.get("oi") or 0)
        for r in chain_rows
        if r["option_type"] == option_type
    ]
    ranked = sorted(by_strike, key=lambda x: x[1], reverse=True)[:TOP_OI_STRIKE_COUNT]
    return [OiWall(strike=float(strike), oi=int(oi)) for strike, oi in ranked if oi > 0]


class OptionsIntelligenceEngine:
    """Stateless — call analyze() with a fresh chain snapshot each candle."""

    def analyze(
        self,
        underlying: str,
        chain_rows: List[Dict[str, Any]],
        now: Optional[datetime] = None,
    ) -> OptionsIntelligence:
        now = now or datetime.now(timezone.utc)

        if not chain_rows:
            return OptionsIntelligence(
                underlying=underlying,
                pcr=None,
                pcr_bias="UNKNOWN",
                max_pain_strike=None,
                as_of=None,
                is_stale=True,
            )

        latest_row_time = max(
            (r["time"] for r in chain_rows if r.get("time") is not None),
            default=None,
        )
        is_stale = True
        if latest_row_time is not None:
            # Normalize both sides to tz-aware UTC before subtracting — comparing
            # a naive and aware datetime raises TypeError, and DB timestamps come
            # back tz-aware while `timestamp` from the trading loop may not.
            ref_now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
            ref_row_time = latest_row_time if latest_row_time.tzinfo else latest_row_time.replace(tzinfo=timezone.utc)
            age = (ref_now - ref_row_time).total_seconds()
            is_stale = age > STALE_AFTER_SECONDS

        pcr = compute_pcr(chain_rows)
        return OptionsIntelligence(
            underlying=underlying,
            pcr=pcr,
            pcr_bias=pcr_bias(pcr),
            max_pain_strike=compute_max_pain(chain_rows),
            call_oi_walls=top_oi_walls(chain_rows, "CE"),
            put_oi_walls=top_oi_walls(chain_rows, "PE"),
            as_of=latest_row_time,
            is_stale=is_stale,
        )
