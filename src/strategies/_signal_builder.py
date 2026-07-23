#!/usr/bin/env python3
"""
Shared signal-construction helpers for the pattern strategies.
==============================================================
Keeps every strategy emitting the SAME signal schema the trader/DB expect
(symbol, signal, strategy, price, stop_loss, take_profit, tp1, rr_ratio,
accepted, rejection_reasons, features, candidate_id, confidence, diagnostics)
and applies the common risk plumbing once:

  * take-profit = nearest opposing H1 zone, floored at a 2R projection,
    capped at 5×ATR (matches the ORB / structural conventions)
  * RR gate, zero-risk gate
  * confluence boost from snapshot.market_view (composability)

Directional strategies call ``build_directional_signal``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.market_snapshot import MarketSnapshot

MIN_RR = 1.5
TP_ATR_CAP = 5.0


def _clean_symbol(symbol: str) -> str:
    return symbol.replace(":", "_").replace("-", "_")


def build_directional_signal(
    snapshot: MarketSnapshot,
    side: str,                      # "BUY CALL" | "BUY PUT"
    setup_type: str,                # e.g. "DOUBLE_BOTTOM", "TRIANGLE_ASC"
    entry: float,
    stop_loss: float,
    base_confidence: float,         # 0..1 before confluence boost
    rejection_reasons: Optional[List[str]] = None,
    diagnostics: Optional[Dict] = None,
    respect_bias: str = "soft",     # "hard" → reject on mismatch; "soft" → only dampen confidence; "off"
    min_rr: float = MIN_RR,
) -> Optional[Dict]:
    """Return a fully-formed signal dict, or None if inputs are degenerate."""
    reasons: List[str] = list(rejection_reasons or [])
    atr = snapshot.features.get_float("atr")
    if atr <= 0:
        atr = max(1.0, entry * 0.001)

    if stop_loss is None or entry is None or entry == stop_loss:
        return None
    risk_dist = abs(entry - stop_loss)
    if risk_dist <= 0:
        return None

    # ── Take-profit: nearest opposing zone, floored at 2R, capped at 5×ATR ──
    tp_floor = entry + 2.0 * risk_dist if side == "BUY CALL" else entry - 2.0 * risk_dist
    tp_from_zone = None
    for z in (snapshot.h1_zones or []):
        if side == "BUY CALL" and getattr(z, "zone_type", "") == "SUPPLY" and z.level > entry:
            tp_from_zone = z.level
            break
        if side == "BUY PUT" and getattr(z, "zone_type", "") == "DEMAND" and z.level < entry:
            tp_from_zone = z.level
            break
    if tp_from_zone is not None:
        take_profit = max(tp_floor, tp_from_zone) if side == "BUY CALL" else min(tp_floor, tp_from_zone)
    else:
        take_profit = tp_floor

    max_tp_dist = atr * TP_ATR_CAP
    if abs(take_profit - entry) > max_tp_dist:
        take_profit = entry + max_tp_dist if side == "BUY CALL" else entry - max_tp_dist
        reasons.append("TP_CAPPED")

    rr = round(abs(take_profit - entry) / risk_dist, 2)
    if rr < min_rr:
        reasons.append("LOW_RR")

    # ── Bias handling ──
    bias = snapshot.daily_bias
    bias_mismatch = (side == "BUY CALL" and bias == "BEARISH") or (side == "BUY PUT" and bias == "BULLISH")
    if bias_mismatch and respect_bias == "hard":
        reasons.append("BIAS_MISMATCH")

    # ── Confluence boost from the shared MarketView ──
    confidence = float(base_confidence)
    boost = 1.0
    view = snapshot.market_view
    if view is not None:
        boost = view.confluence_boost(side)
        confidence *= boost
    if bias_mismatch and respect_bias == "soft":
        confidence *= 0.85
    confidence = round(min(0.99, max(0.0, confidence)), 3)

    accepted = len(reasons) == 0

    diag = dict(diagnostics or {})
    diag.update({
        "atr": round(atr, 2),
        "rr": rr,
        "confluence_boost": round(boost, 3),
        "market_view": view.summary if view is not None else "n/a",
    })

    ts = snapshot.timestamp
    candidate_id = (
        f"cand_{_clean_symbol(snapshot.symbol)}_{setup_type}_{entry:.2f}_"
        f"{ts.strftime('%Y%m%d_%H%M%S')}"
    )

    return {
        "symbol": snapshot.symbol,
        "signal": side,
        "strategy": setup_type,
        "price": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "tp1": entry + risk_dist * 1.5 if side == "BUY CALL" else entry - risk_dist * 1.5,
        "rr_ratio": rr,
        "timestamp": ts.isoformat(),
        "accepted": accepted,
        "rejection_reasons": reasons,
        "features": snapshot.features.to_dict(),
        "candidate_id": candidate_id,
        "confidence": round(confidence * 100, 1),  # trader/sizer expect 0..100 scale
        "diagnostics": diag,
    }
