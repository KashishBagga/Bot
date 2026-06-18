#!/usr/bin/env python3
"""
EMAStrategy — EMA 20/50 Crossover Strategy.
=============================================
First non-structural strategy. Validates the multi-experiment framework
end-to-end. High signal frequency makes framework bugs surface quickly.

Logic:
    EMA20 > EMA50 → BUY CALL (bullish crossover regime)
    EMA20 < EMA50 → BUY PUT  (bearish crossover regime)

SL/TP computed using ATR (standard for momentum strategies):
    SL = 1.0 × ATR from entry
    TP = 2.0 × ATR from entry (default 2:1 RR)

Filters:
    RR >= 1.5  (rejects if ATR is too compressed)

This strategy intentionally uses require_float() for EMAs — it genuinely
cannot evaluate without them. If the IndicatorPipeline didn't compute them,
that's a bug worth surfacing loudly, not silently defaulting to 0.0.
"""

import logging
from typing import List

from src.core.base_strategy import BaseStrategy, StrategyResult
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class EMAStrategy(BaseStrategy):
    """
    EMA 20/50 crossover strategy.
    First companion strategy in the multi-experiment framework.
    """

    id = "ema_crossover"
    name = "EMA 20/50 Crossover"
    version = "v1.0"

    def __init__(self, fast: int = 20, slow: int = 50, min_rr: float = 1.5):
        """
        Parameters baked in at construction. Immutable after that.

        Args:
            fast:   Fast EMA period (default 20)
            slow:   Slow EMA period (default 50)
            min_rr: Minimum risk/reward ratio to accept a signal (default 1.5)
        """
        self.fast = fast
        self.slow = slow
        self.min_rr = min_rr
        self.fast_key = f"ema{fast}"
        self.slow_key = f"ema{slow}"

        logger.info(
            f"📈 EMAStrategy initialized "
            f"[EMA{fast}/EMA{slow}, min_rr={min_rr}]"
        )

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        experiment_name: str,
    ) -> StrategyResult:
        """
        Evaluate EMA crossover state and produce a signal if clear regime exists.

        Contract:
            - Never raises.
            - Uses require_float() for EMAs — surfaces missing feature loudly via errors.
            - Returns empty signals if EMAs unavailable or regime is flat.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # ── Required features — fail loud if missing ──────────────────────
        try:
            ema_fast = snapshot.features.require_float(self.fast_key)
            ema_slow = snapshot.features.require_float(self.slow_key)
        except KeyError as e:
            errors.append(str(e))
            return self._empty_result(experiment_name, errors=errors)

        atr = snapshot.features.get_float("atr", default=50.0)
        price = snapshot.current_price

        # ── Regime check ──────────────────────────────────────────────────
        if ema_fast > ema_slow:
            direction = "BUY CALL"
            sl = round(price - atr, 2)
            tp = round(price + atr * 2.0, 2)
        elif ema_fast < ema_slow:
            direction = "BUY PUT"
            sl = round(price + atr, 2)
            tp = round(price - atr * 2.0, 2)
        else:
            # EMAs exactly equal — no clear regime
            return self._empty_result(
                experiment_name,
                warnings=["EMA_FLAT"],
                diagnostics={"ema_fast": ema_fast, "ema_slow": ema_slow}
            )

        # ── Cross strength warning ─────────────────────────────────────────
        cross_pct = abs(ema_fast - ema_slow) / ema_slow if ema_slow != 0 else 0.0
        if cross_pct < 0.001:
            warnings.append(f"WEAK_CROSS:{cross_pct:.4f}")

        # ── Risk/Reward ───────────────────────────────────────────────────
        risk = abs(price - sl)
        reward = abs(tp - price)
        rr = round(reward / risk, 2) if risk > 0 else 0.0

        rejection_reasons = []
        if rr < self.min_rr:
            rejection_reasons.append("LOW_RR")
        accepted = len(rejection_reasons) == 0

        # ── Candidate ID ──────────────────────────────────────────────────
        candidate_id = (
            f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}"
            f"_ema{self.fast}_{self.slow}"
            f"_{snapshot.timestamp.strftime('%Y%m%d_%H%M')}"
        )

        signal = {
            # Core signal fields (must match schema expected by PositionManager)
            "symbol":            snapshot.symbol,
            "signal":            direction,
            "strategy":          "EMA_CROSS",
            "price":             price,
            "stop_loss":         sl,
            "take_profit":       tp,
            "tp1":               round(price + (risk * 1.5), 2) if direction == "BUY CALL" else round(price - (risk * 1.5), 2),
            "rr_ratio":          rr,
            "timestamp":         snapshot.timestamp.isoformat(),
            "accepted":          accepted,
            "rejection_reasons": rejection_reasons,
            "candidate_id":      candidate_id,
            # Framework metadata
            "experiment_name":   experiment_name,
            "strategy_id":       self.id,
            "version":           self.version,
            # Feature snapshot for audit/DB
            "features": {
                self.fast_key: ema_fast,
                self.slow_key: ema_slow,
                "atr":         atr,
                "cross_pct":   round(cross_pct, 4),
            },
        }

        diagnostics = {
            "ema_fast":   ema_fast,
            "ema_slow":   ema_slow,
            "atr":        atr,
            "cross_pct":  round(cross_pct, 4),
            "direction":  direction,
            "rr":         rr,
        }

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=[signal],
            diagnostics=diagnostics,
            errors=errors,
            warnings=warnings,
        )

    def __repr__(self) -> str:
        return f"EMAStrategy(EMA{self.fast}/EMA{self.slow}, min_rr={self.min_rr})"
