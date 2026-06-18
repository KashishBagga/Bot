#!/usr/bin/env python3
"""
StructuralStrategy — Thin adapter wrapping EnhancedStrategyEngine.
==================================================================
Implements BaseStrategy without modifying EnhancedStrategyEngine at all.
The existing engine is the source of truth — this adapter just calls it
and packages the output into a StrategyResult.

SL/TP stay INSIDE this strategy because structural stops ARE the thesis:
    SWEEP:    stop 1 tick beyond the sweep wick
    BREAKOUT: stop 0.3×ATR below the broken structure level

These are not risk management decisions. They define when the thesis is
invalidated. Pulling them into a generic RiskEngine would be incorrect.

Regression guarantee:
    Every signal produced by StructuralStrategy must be byte-for-byte
    identical to what EnhancedStrategyEngine._evaluate_structural_setups()
    produced before this wrapper existed.
"""

import logging
from typing import List

from src.core.base_strategy import BaseStrategy, StrategyResult
from src.core.market_snapshot import MarketSnapshot
from src.core.enhanced_strategy_engine import EnhancedStrategyEngine

logger = logging.getLogger(__name__)


class StructuralStrategy(BaseStrategy):
    """
    Institutional Structural Strategy (Sweep / Breakout / FFT Trap).
    Phase 3.2 — Code frozen.

    Wraps EnhancedStrategyEngine._evaluate_structural_setups() with
    the BaseStrategy interface. Zero logic changes.
    """

    id = "structural"
    name = "Structural Breakout"
    version = "v3.2"

    def __init__(
        self,
        min_zone_score: float = 50.0,
        rvol_threshold: float = 1.0,
    ):
        """
        Parameters are baked in at construction. Immutable after that.
        Each Experiment that uses StructuralStrategy must create its OWN instance.
        Never pass the same instance to multiple Experiments.
        """
        self.min_zone_score = min_zone_score
        self.rvol_threshold = rvol_threshold

        # Internal engine — NOT shared, NOT modified. This instance belongs to this strategy.
        self._engine = EnhancedStrategyEngine(
            symbols=[],   # Symbols not needed at init — injected per evaluate() call
            min_zone_score=min_zone_score,
            rvol_threshold=rvol_threshold,
        )

        logger.info(
            f"🏛️ StructuralStrategy initialized "
            f"[rvol>={rvol_threshold}, zone_score>={min_zone_score}]"
        )

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        experiment_name: str,
    ) -> StrategyResult:
        """
        Delegate to EnhancedStrategyEngine._evaluate_structural_setups().
        Package results into a StrategyResult.

        Contract:
            - Never raises.
            - All exceptions go into result.errors.
            - Every signal dict is tagged with experiment_name, strategy_id, version.
        """
        errors: List[str] = []
        warnings: List[str] = []
        raw_signals: List[dict] = []

        try:
            raw_signals = self._engine._evaluate_structural_setups(
                symbol=snapshot.symbol,
                m5_df=snapshot.m5,
                zones=snapshot.h1_zones,
                daily_bias=snapshot.daily_bias,
                h1_struct=snapshot.h1_structure,
                regime=snapshot.market_regime,
                price=snapshot.current_price,
            )
        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(
                f"[StructuralStrategy] Engine error for {snapshot.symbol}: {e}",
                exc_info=True
            )

        # Tag every signal with experiment metadata
        for sig in raw_signals:
            sig["experiment_name"] = experiment_name
            sig["strategy_id"] = self.id
            sig["version"] = self.version

        # Diagnostic snapshot for logging / debugging
        diagnostics = {
            "rvol_threshold": self.rvol_threshold,
            "min_zone_score": self.min_zone_score,
            "rvol_actual": snapshot.volume_report.rvol_tod if snapshot.volume_report else None,
            "daily_bias": snapshot.daily_bias,
            "market_regime": snapshot.market_regime,
            "zones_detected": len(snapshot.h1_zones),
        }

        # Warn if RVOL is very low (legitimate signal, useful diagnostic)
        if snapshot.volume_report and snapshot.volume_report.rvol_tod < 0.5:
            warnings.append(f"VERY_LOW_RVOL:{snapshot.volume_report.rvol_tod:.2f}")

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=raw_signals,
            diagnostics=diagnostics,
            errors=errors,
            warnings=warnings,
        )

    def __repr__(self) -> str:
        return (
            f"StructuralStrategy("
            f"rvol>={self.rvol_threshold}, "
            f"zone_score>={self.min_zone_score})"
        )
