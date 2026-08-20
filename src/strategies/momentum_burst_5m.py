#!/usr/bin/env python3
"""
Momentum Burst (5m-only) Strategy
==================================
Hypothesis: a genuine outlier move announces itself on the 5-minute chart
alone — a range-expansion candle on a real volume burst, immediately
confirmed by a same-direction follow-through candle. This deliberately does
NOT gate on Daily/1H bias or structure: MTF-gated strategies (Structural_v3.2)
are inherently conservative/lagging (Daily -> 1H -> 5M bias gating), and
historical shadow-trade data shows that lag costs real edge once the market
turns choppy faster than Daily/1H can confirm (Structural_v3.2's BREAKOUT
setups flipped from +0.8..+2.1R/day to -0.1..-0.5R/day once trend_quality
dropped below 0.35 in late July 2026). This strategy exists to catch exactly
the fast reversals that gap leaves open — it must be free to fire against a
stale or contradicting Daily bias.
"""

import logging
from typing import List, Dict, Any

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class MomentumBurst5mStrategy(BaseStrategy):
    """Pure 5m momentum-burst strategy — no HTF gating by design."""

    metadata = StrategyMetadata(
        id="momentum_burst_5m",
        name="Momentum Burst (5m)",
        hypothesis_id="momentum_burst_5m_no_mtf",
        hypothesis_family="Momentum",
        hypothesis_text=(
            "A range-expansion candle on an RVOL burst, confirmed by same-direction "
            "follow-through, catches fast moves that MTF-gated strategies arrive late to."
        ),
        version="v1.0",
        archetype="Momentum",
        exit_profile="INDEX_TP_EXPANSION",
        maturity="PAPER",
        tags=["momentum", "5m_only", "no_mtf_gating"],
    )

    def __init__(
        self,
        range_atr_mult: float = 1.8,
        min_body_fraction: float = 0.55,
        rvol_burst_threshold: float = 2.0,
        follow_through_giveback_pct: float = 0.35,
        target_rr: float = 2.2,
        sl_atr_floor_mult: float = 0.6,
    ):
        self.range_atr_mult = range_atr_mult
        self.min_body_fraction = min_body_fraction
        self.rvol_burst_threshold = rvol_burst_threshold
        self.follow_through_giveback_pct = follow_through_giveback_pct
        self.target_rr = target_rr
        self.sl_atr_floor_mult = sl_atr_floor_mult

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            # Deliberately reads ONLY snapshot.m5 / features / volume_report.
            # No snapshot.daily_bias, snapshot.h1_structure, snapshot.h1_zones,
            # or snapshot.market_regime anywhere in this method — see module
            # docstring for why.
            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < 20:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0
            move_efficiency = snapshot.features.get_float("move_efficiency")
            current_time = snapshot.timestamp

            if atr <= 0:
                return self._empty_result(experiment_name, errors=["INVALID_ATR"])

            trigger = m5_df.iloc[-2]  # last CLOSED candle
            confirm = m5_df.iloc[-1]  # already closed too — evaluate() runs once per closed bar

            trigger_high, trigger_low = float(trigger["high"]), float(trigger["low"])
            trigger_open, trigger_close = float(trigger["open"]), float(trigger["close"])
            trigger_range = trigger_high - trigger_low
            trigger_body = abs(trigger_close - trigger_open)
            body_fraction = trigger_body / trigger_range if trigger_range > 0 else 0.0

            is_expansion_candle = trigger_range >= (self.range_atr_mult * atr)
            is_directional_body = body_fraction >= self.min_body_fraction
            is_rvol_burst = rvol >= self.rvol_burst_threshold

            if not (is_expansion_candle and is_directional_body and is_rvol_burst):
                return self._empty_result(experiment_name)

            trigger_is_bullish = trigger_close > trigger_open
            side = "BUY CALL" if trigger_is_bullish else "BUY PUT"
            setup_type = "MOMENTUM_BURST"

            confirm_close = float(confirm["close"])
            confirm_low, confirm_high = float(confirm["low"]), float(confirm["high"])
            if side == "BUY CALL":
                confirm_body_dir_ok = confirm_close > trigger_close
                giveback = trigger_close - confirm_low
            else:
                confirm_body_dir_ok = confirm_close < trigger_close
                giveback = confirm_high - trigger_close
            giveback_ok = giveback <= (self.follow_through_giveback_pct * trigger_range)

            rejection_reasons: List[str] = []
            if not (confirm_body_dir_ok and giveback_ok):
                rejection_reasons.append("NO_FOLLOW_THROUGH")

            risk_dist = max(trigger_range * 0.5, atr * self.sl_atr_floor_mult)
            if side == "BUY CALL":
                sl = price - risk_dist
                take_profit = price + (risk_dist * self.target_rr)
            else:
                sl = price + risk_dist
                take_profit = price - (risk_dist * self.target_rr)

            if risk_dist == 0.0:
                rejection_reasons.append("ZERO_RISK")
            if current_time.hour >= 15:
                rejection_reasons.append("LATE_SESSION")

            rr = round(abs(take_profit - price) / risk_dist, 2) if risk_dist > 0 else 0.0

            confidence = round(
                0.5
                + 0.25 * min(rvol / 3.0, 1.0)
                + 0.25 * min(trigger_range / (atr * 3.0), 1.0),
                2,
            )

            diagnostics = {
                "trigger_range": round(trigger_range, 2),
                "atr": round(atr, 2),
                "rvol": round(rvol, 2),
                "body_fraction": round(body_fraction, 3),
                "giveback_pct": round(giveback / trigger_range, 3) if trigger_range > 0 else None,
                "move_efficiency": round(move_efficiency, 3),
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}"
                f"_MOMBURST_{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            )

            sig = {
                'symbol': snapshot.symbol,
                'signal': side,
                'strategy': setup_type,
                'price': price,
                'stop_loss': sl,
                'take_profit': take_profit,
                'tp1': price + (risk_dist * 1.5) if side == "BUY CALL" else price - (risk_dist * 1.5),
                'rr_ratio': rr,
                'timestamp': current_time.isoformat(),
                'accepted': accepted,
                'rejection_reasons': rejection_reasons,
                'features': snapshot.features.to_dict(),
                'candidate_id': candidate_id,
                'confidence': confidence,
                'diagnostics': diagnostics,
            }

            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[MomentumBurst5mStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
