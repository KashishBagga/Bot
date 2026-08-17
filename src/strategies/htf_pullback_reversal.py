#!/usr/bin/env python3
"""
HTF Pullback Reversal Strategy
===============================
Hypothesis: a Daily-trend continuation pulling back to the 1H EMA20 ("1H as
support/resistance"), confirmed by a 5M rejection candle in the trend
direction, is a genuinely different signal from StructuralStrategy's
zone/fractal SWEEP-BREAKOUT-TRAP logic — a moving-average pullback-in-trend
play, not a structural-break play. Touches no zones and no HH/HL fractal
structure of its own; Daily conviction is required (unlike MomentumBurst5m,
which requires none) — that's the defining contrast between the two new
archetypes.
"""

import logging
from typing import List, Dict, Any

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.quant_utils import QuantUtils

logger = logging.getLogger(__name__)


class HtfPullbackReversalStrategy(BaseStrategy):
    """Daily bias + 1H EMA20 pullback + 5M rejection-candle trigger."""

    metadata = StrategyMetadata(
        id="htf_pullback_reversal",
        name="HTF Pullback Reversal",
        hypothesis_id="htf_pullback_ema20",
        hypothesis_family="Trend Continuation",
        hypothesis_text=(
            "Daily trend pulling back to the 1H EMA20, confirmed by a 5M rejection "
            "candle in the trend direction, is a swing-continuation entry distinct "
            "from structural zone/fractal breaks."
        ),
        version="v1.0",
        maturity="PAPER",
        tags=["trend_continuation", "mtf", "ema_pullback"],
    )

    def __init__(
        self,
        pullback_tolerance_pct: float = 0.006,
        sl_buffer_atr_mult: float = 0.3,
        target_rr_floor: float = 1.8,
        rvol_threshold: float = 0.9,
    ):
        self.pullback_tolerance_pct = pullback_tolerance_pct
        self.sl_buffer_atr_mult = sl_buffer_atr_mult
        self.target_rr_floor = target_rr_floor
        self.rvol_threshold = rvol_threshold

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            h1_df = snapshot.h1
            m5_df = snapshot.m5
            if h1_df is None or len(h1_df) < 30 or m5_df is None or len(m5_df) < 20:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            current_time = snapshot.timestamp
            atr = snapshot.features.get_float("atr")
            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0

            # Daily leg — reuses the already-computed daily_bias (no duplicate
            # daily-trend metric); requires conviction, unlike MomentumBurst5m.
            daily_bias = snapshot.daily_bias
            if daily_bias not in ("BULLISH", "BEARISH"):
                return self._empty_result(experiment_name)

            # 1H leg — "1H as support/resistance" made mechanical via a locally
            # computed EMA20 (features.ema20/ema50 are m5-frame, not 1H-frame).
            h1_ema20_series = h1_df["close"].ewm(span=20, adjust=False).mean()
            h1_ema20_now = float(h1_ema20_series.iloc[-1])
            if h1_ema20_now <= 0:
                return self._empty_result(experiment_name, errors=["INVALID_EMA"])

            dist_to_h1_ema_pct = abs(price - h1_ema20_now) / h1_ema20_now
            is_near_h1_ema = dist_to_h1_ema_pct <= self.pullback_tolerance_pct

            h1_recent_high = float(h1_df["high"].iloc[-6:].max())
            h1_recent_low = float(h1_df["low"].iloc[-6:].min())

            if daily_bias == "BULLISH":
                pulled_back_from_above = h1_recent_high > h1_ema20_now * (1 + self.pullback_tolerance_pct)
                valid_pullback = (
                    is_near_h1_ema
                    and pulled_back_from_above
                    and price >= h1_ema20_now * (1 - self.pullback_tolerance_pct)
                )
            else:  # BEARISH
                pulled_back_from_below = h1_recent_low < h1_ema20_now * (1 - self.pullback_tolerance_pct)
                valid_pullback = (
                    is_near_h1_ema
                    and pulled_back_from_below
                    and price <= h1_ema20_now * (1 + self.pullback_tolerance_pct)
                )

            if not valid_pullback:
                return self._empty_result(experiment_name)

            # 5M leg — same rejection primitive already trusted by other strategies.
            if not QuantUtils.is_strong_rejection(m5_df, candle_idx=-2):
                return self._empty_result(experiment_name)

            last_closed = m5_df.iloc[-2]
            last_open, last_close = float(last_closed["open"]), float(last_closed["close"])

            setup_type = "HTF_PULLBACK"
            if daily_bias == "BULLISH" and last_close > last_open:
                side = "BUY CALL"
            elif daily_bias == "BEARISH" and last_close < last_open:
                side = "BUY PUT"
            else:
                return self._empty_result(experiment_name)

            if side == "BUY CALL":
                sl = h1_ema20_now - (atr * self.sl_buffer_atr_mult)
            else:
                sl = h1_ema20_now + (atr * self.sl_buffer_atr_mult)
            risk_dist = abs(price - sl)

            # TP may use HTF zones as a target floor — appropriate here since
            # this strategy's whole premise is HTF context (unlike
            # MomentumBurst5m, which deliberately avoids HTF zone targets).
            tp_floor = price + (risk_dist * self.target_rr_floor) if side == "BUY CALL" else price - (risk_dist * self.target_rr_floor)
            take_profit = tp_floor
            d1_zones = getattr(snapshot, "d1_zones", None) or []
            if side == "BUY CALL":
                candidate_zones = [z.level for z in d1_zones if getattr(z, "level", None) and z.level > price]
                if candidate_zones:
                    take_profit = max(tp_floor, min(candidate_zones))
            else:
                candidate_zones = [z.level for z in d1_zones if getattr(z, "level", None) and z.level < price]
                if candidate_zones:
                    take_profit = min(tp_floor, max(candidate_zones))

            rejection_reasons: List[str] = []
            if rvol < self.rvol_threshold:
                rejection_reasons.append("LOW_RVOL")
            if risk_dist == 0.0:
                rejection_reasons.append("ZERO_RISK")
            rr = round(abs(take_profit - price) / risk_dist, 2) if risk_dist > 0 else 0.0
            if rr < 1.5:
                rejection_reasons.append("LOW_RR")
            if current_time.hour >= 15:
                rejection_reasons.append("LATE_SESSION")

            confidence = round(
                0.5
                + 0.3 * (1 - min(dist_to_h1_ema_pct / self.pullback_tolerance_pct, 1.0))
                + 0.2 * min(rvol / 1.5, 1.0),
                2,
            )

            diagnostics = {
                "h1_ema20": round(h1_ema20_now, 2),
                "dist_to_h1_ema_pct": round(dist_to_h1_ema_pct, 5),
                "daily_bias": daily_bias,
                "rvol": round(rvol, 2),
                "atr": round(atr, 2),
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}"
                f"_HTFPULLBACK_{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
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
            logger.error(f"[HtfPullbackReversalStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
