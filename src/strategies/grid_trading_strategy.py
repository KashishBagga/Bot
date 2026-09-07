#!/usr/bin/env python3
"""
Grid Trading Strategy
========================
Hypothesis: in a genuinely range-bound/low-ADX regime, price oscillates
around a session anchor rather than trending — a fixed ladder of buy/sell
levels spaced at regular ATR-scaled intervals above and below that anchor
captures the oscillation without needing to predict which way price turns
next. Buy CALL when price is sitting at a grid level BELOW the anchor
(fading the dip back toward it), buy PUT when price is at a level ABOVE it
(fading the rip back toward it) — classic grid trading, adapted to a
single-instrument index-options book (no short-selling the index itself,
so both "legs" of the grid are expressed as directional option buys).

Distinct from every existing RANGE-bucket strategy (RSI-2, VWAP reversion,
ATR squeeze): those all key off an oscillator/VWAP-distance/volatility
percentile. This one is pure price-ladder distance from a fixed anchor,
gated to only fire when ADX confirms an actual range (not a trend the grid
would otherwise fight). Self-contained — computes everything from
snapshot.m5/snapshot.features, anchor is the session's first 5m candle open
(deterministic per day, no stored state needed across candles).
"""

import logging
from typing import List, Dict, Any

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class GridTradingStrategy(BaseStrategy):
    """Fixed ATR-spaced grid of buy/sell levels around the session anchor, active only in range regimes."""

    metadata = StrategyMetadata(
        id="grid_trading",
        name="Grid Trading",
        hypothesis_id="grid_trading_range",
        hypothesis_family="Mean Reversion",
        hypothesis_text=(
            "In a low-ADX range regime, price oscillates around a session "
            "anchor — a fixed ladder of ATR-spaced levels above/below it can "
            "be faded back toward the anchor without predicting direction."
        ),
        version="v1.0",
        archetype="Mean-Reversion",
        exit_profile="INDEX_TP_EXPANSION",
        maturity="RESEARCH",
        tags=["grid", "range", "mean_reversion"],
    )

    def __init__(
        self,
        grid_spacing_atr_mult: float = 0.5,
        max_grid_levels: int = 3,
        adx_ceiling: float = 20.0,
        min_rr: float = 1.2,
    ):
        self.grid_spacing_atr_mult = grid_spacing_atr_mult
        self.max_grid_levels = max_grid_levels
        self.adx_ceiling = adx_ceiling
        self.min_rr = min_rr

    def thesis_key(self, signal: dict) -> tuple:
        # One active thesis PER GRID LEVEL, not just per direction — distinct
        # levels are genuinely independent bets in a real grid.
        return (
            signal.get("symbol", ""), "GRID",
            signal.get("signal", ""), signal.get("diagnostics", {}).get("level_index"),
        )

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < 20:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_DATA"])

            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            adx = snapshot.features.get_float("adx")
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            # Session anchor: first 5m candle's open of the current trading day.
            current_time = snapshot.timestamp
            today = current_time.date()
            todays_bars = m5_df[m5_df.index.date == today] if hasattr(m5_df.index, "date") else m5_df.iloc[-1:]
            if len(todays_bars) == 0:
                return self._empty_result(experiment_name, errors=["NO_SESSION_BARS"])
            anchor = float(todays_bars.iloc[0]["open"])

            spacing = atr * self.grid_spacing_atr_mult
            if spacing <= 0:
                return self._empty_result(experiment_name, errors=["ZERO_GRID_SPACING"])

            level_index = int(round((price - anchor) / spacing))

            if level_index == 0 or abs(level_index) > self.max_grid_levels:
                return self._empty_result(experiment_name)

            setup_type = "GRID_LEVEL"
            if level_index < 0:
                # Price sitting below the anchor by |level_index| grid steps — fade the dip.
                # Target the ANCHOR itself, not just the next rung: risk is a
                # fixed one grid-step (the next level further out), but reward
                # scales with distance from the anchor, same as real grid
                # economics (outer levels carry better R:R, not the same ~1:1
                # every rung would give if the target were just the next rung).
                side = "BUY CALL"
                sl = price - spacing
                take_profit = anchor
            else:
                side = "BUY PUT"
                sl = price + spacing
                take_profit = anchor

            rejection_reasons: List[str] = []

            # Grid trading fights trends — only active when ADX confirms an
            # actual range, otherwise every level just bleeds against a trend.
            if adx > self.adx_ceiling:
                rejection_reasons.append("TRENDING_NOT_RANGE")

            risk_dist = abs(price - sl)
            if risk_dist == 0.0:
                rejection_reasons.append("ZERO_RISK")

            reward = abs(take_profit - price)
            rr = round(reward / risk_dist, 2) if risk_dist > 0 else 0.0
            if rr < self.min_rr:
                rejection_reasons.append("LOW_RR")

            confidence = 0.5 if rejection_reasons else round(min(0.5 + 0.05 * abs(level_index), 0.75), 2)

            diagnostics = {
                "anchor": round(anchor, 2),
                "level_index": level_index,
                "grid_spacing": round(spacing, 2),
                "adx": round(adx, 2),
                "rr_ratio": rr,
            }

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_GRID{level_index}_"
                f"{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            )

            sig = {
                "symbol": snapshot.symbol,
                "signal": side,
                "strategy": setup_type,
                "price": price,
                "stop_loss": sl,
                "take_profit": take_profit,
                "tp1": take_profit,
                "rr_ratio": rr,
                "timestamp": current_time.isoformat() if hasattr(current_time, "isoformat") else str(current_time),
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
                "features": snapshot.features.to_dict(),
                "candidate_id": candidate_id,
                "confidence": confidence,
                "diagnostics": diagnostics,
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[GridTradingStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
