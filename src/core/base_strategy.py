#!/usr/bin/env python3
"""
BaseStrategy + StrategyResult — Core abstractions for the strategy framework.
=============================================================================

BaseStrategy:
    Abstract base class every strategy implements.
    One method: evaluate(snapshot, experiment_name) -> StrategyResult

StrategyResult:
    Fully resolved output from one strategy evaluation on one snapshot.
    Contains:
      - signals:     Fully resolved signal dicts (entry, sl, tp, rr, accepted, ...)
      - errors:      Strategy couldn't evaluate (FEATURE_MISSING, ENGINE_ERROR, ...)
      - warnings:    Strategy evaluated but flagged something notable (WEAK_CROSS, ATR_SPIKE)
      - diagnostics: Per-evaluation metadata for debugging
      - runtime_ms:  Set by ExperimentRegistry after evaluate() returns

Design rules:
    - evaluate() NEVER raises. All failures go into result.errors.
    - errors   = "I couldn't produce a valid signal"
    - warnings = "I produced a signal but something looked unusual"
    - Neither errors nor warnings stop other experiments from running.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

from src.core.market_snapshot import MarketSnapshot


@dataclass(frozen=True)
class StrategyMetadata:
    id: str
    name: str
    hypothesis_id: str
    hypothesis_family: str
    hypothesis_text: str
    version: str
    author: str = "Kashish"
    expected_holding: Tuple[int, int] = (5, 15)
    preferred_regimes: List[str] = field(default_factory=list)
    preferred_sessions: List[str] = field(default_factory=list)
    risk_profile: str = "medium"
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    maturity: str = "RESEARCH"  # RESEARCH, PAPER, SHADOW_LIVE, LIVE, RETIRED
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """
    Output of one strategy evaluation on one MarketSnapshot.
    Created by BaseStrategy.evaluate() and returned to ExperimentRegistry.run().
    """
    experiment_name: str
    strategy_id: str        # Stable DB key — never changes. e.g. "structural"
    version: str            # Algorithm version — e.g. "v3.2"

    signals: List[Dict[str, Any]]   # Fully resolved signal dicts, ready for PositionManager
    diagnostics: Dict[str, Any]     # Internal metadata for debugging / logging

    runtime_ms: float = 0.0         # Set by ExperimentRegistry after evaluate() returns

    errors: List[str] = field(default_factory=list)
    # Non-fatal. Strategy couldn't evaluate at all.
    # Examples: "FEATURE_MISSING:ema50", "ENGINE_ERROR:divide by zero"
    # Action: log warning, skip this result, continue other experiments.

    warnings: List[str] = field(default_factory=list)
    # Non-fatal. Strategy evaluated but noticed something unusual.
    # Examples: "WEAK_CROSS", "ATR_SPIKE", "LOW_ZONE_SCORE"
    # Action: log info, still process signals normally.

    @property
    def has_signals(self) -> bool:
        return len(self.signals) > 0

    @property
    def accepted_signals(self) -> List[Dict[str, Any]]:
        return [s for s in self.signals if s.get("accepted", False)]

    @property
    def rejected_signals(self) -> List[Dict[str, Any]]:
        return [s for s in self.signals if not s.get("accepted", True)]

    def __repr__(self) -> str:
        return (
            f"StrategyResult("
            f"experiment={self.experiment_name}, "
            f"signals={len(self.signals)}, "
            f"accepted={len(self.accepted_signals)}, "
            f"errors={self.errors}, "
            f"warnings={self.warnings}, "
            f"runtime={self.runtime_ms:.1f}ms)"
        )


class BaseStrategy:
    """
    Abstract base class for all strategies.

    Every strategy must implement:
        metadata — StrategyMetadata object
        evaluate(snapshot, experiment_name) -> StrategyResult
    """

    metadata: StrategyMetadata = StrategyMetadata(
        id="base",
        name="Base Strategy",
        hypothesis_id="base_hypothesis",
        hypothesis_family="Base",
        hypothesis_text="Base strategy class",
        version="v0.0"
    )

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def version(self) -> str:
        return self.metadata.version

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        experiment_name: str,
    ) -> StrategyResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate()"
        )

    def thesis_key(self, signal: dict) -> tuple:
        """
        Define the deduplication key for a rejected signal (CF deduplication).
        The trader prepends experiment_name, so this should return the
        strategy-specific dimensions only.

        One active CF per thesis_key — subsequent candles with the same thesis
        are skipped until the existing CF exits (SL/TP/SESSION_END).

        Default: (symbol, setup_type, direction)
        Override per strategy to match the actual uniqueness of a thesis:
          - StructuralStrategy: (symbol, setup_type, direction)  — same breakout direction
          - EMAStrategy:        (symbol, direction)               — one per crossover direction
          - Custom strategy:    (symbol, swing_high_timestamp)    — per structural level
        """
        return (
            signal.get('symbol', ''),
            signal.get('strategy', ''),  # setup_type key in signal dict
            signal.get('signal', ''),    # 'BUY CALL' or 'BUY PUT'
        )

    def _empty_result(
        self,
        experiment_name: str,
        errors: List[str] = None,
        warnings: List[str] = None,
        diagnostics: Dict[str, Any] = None,
    ) -> StrategyResult:
        """Convenience helper — return a no-signal result with optional errors/warnings."""
        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=[],
            diagnostics=diagnostics or {},
            errors=errors or [],
            warnings=warnings or [],
        )

    def _tag_signal(self, sig: dict, experiment_name: str) -> dict:
        """Add experiment metadata to a signal dict in-place and return it."""
        sig["experiment_name"] = experiment_name
        sig["strategy_id"] = self.id
        sig["version"] = self.version
        return sig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, version={self.version})"
