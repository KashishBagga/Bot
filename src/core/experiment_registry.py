#!/usr/bin/env python3
"""
ExperimentRegistry — Top-level runtime orchestrator.
====================================================
Holds a list of active Experiments and runs all of them against
each MarketSnapshot. One experiment crash never affects others.

Usage:
    registry = ExperimentRegistry()
    registry.register(Experiment("Structural_v3.2_RVOL1.0", ...))
    registry.register(Experiment("EMA_20_50", ...))

    results = registry.run(snapshot)
    for result in results:
        if result.errors:
            logger.warning(...)
        for sig in result.signals:
            ...

Litmus test for extensibility:
    Adding a new strategy requires ONLY:
        class MyStrategy(BaseStrategy): ...
        registry.register(Experiment("MyExp", MyStrategy(), {}))
    If any other file needs editing, there is hidden coupling.
"""

import logging
import time
from typing import List, Optional

from src.core.base_strategy import BaseStrategy, StrategyResult
from src.core.experiment import Experiment
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class ExperimentRegistry:
    """
    Holds and runs all registered experiments.
    The top-level abstraction in the strategy layer.
    """

    def __init__(self):
        self.experiments: List[Experiment] = []

    def register(self, experiment: Experiment) -> None:
        """
        Register an experiment. Logs a warning if an experiment with the same
        name is already registered (duplicate names → ambiguous DB records).
        """
        existing_names = {e.name for e in self.experiments}
        if experiment.name in existing_names:
            logger.warning(
                f"⚠️ Experiment '{experiment.name}' is already registered. "
                f"Duplicate names will cause ambiguous DB records."
            )
        self.experiments.append(experiment)
        logger.info(
            f"📋 Registered experiment: {experiment.name} "
            f"[{experiment.strategy.id}@{experiment.strategy.version}] "
            f"hash={experiment.config_hash} git={experiment.git_commit}"
        )

    def run(self, snapshot: MarketSnapshot) -> List[StrategyResult]:
        """
        Evaluate all active experiments against this snapshot.
        One experiment crash never stops the others.
        Returns results in registration order.
        """
        results = []
        active = [e for e in self.experiments if e.status == "active"]

        for exp in active:
            t0 = time.perf_counter()
            try:
                result = exp.strategy.evaluate(snapshot, experiment_name=exp.name)
            except Exception as e:
                logger.error(
                    f"💥 Experiment '{exp.name}' raised an unhandled exception: {e}",
                    exc_info=True
                )
                result = StrategyResult(
                    experiment_name=exp.name,
                    strategy_id=exp.strategy.id,
                    version=exp.strategy.version,
                    signals=[],
                    diagnostics={},
                    errors=[f"UNHANDLED_EXCEPTION:{e}"],
                    warnings=[],
                )

            result.runtime_ms = (time.perf_counter() - t0) * 1000

            if result.errors:
                logger.warning(
                    f"⚡ [{exp.name}] errors: {result.errors} "
                    f"({result.runtime_ms:.1f}ms)"
                )
            if result.warnings:
                logger.info(
                    f"⚡ [{exp.name}] warnings: {result.warnings}"
                )
            if result.has_signals:
                logger.debug(
                    f"✅ [{exp.name}] {len(result.signals)} signal(s) "
                    f"({len(result.accepted_signals)} accepted) "
                    f"in {result.runtime_ms:.1f}ms"
                )

            results.append(result)

        return results

    def pause(self, name: str) -> bool:
        """Pause an experiment by name. Returns True if found."""
        for exp in self.experiments:
            if exp.name == name:
                exp.status = "paused"
                logger.info(f"⏸️  Paused experiment: {name}")
                return True
        logger.warning(f"Experiment '{name}' not found.")
        return False

    def resume(self, name: str) -> bool:
        """Resume a paused experiment. Returns True if found."""
        for exp in self.experiments:
            if exp.name == name:
                exp.status = "active"
                logger.info(f"▶️  Resumed experiment: {name}")
                return True
        logger.warning(f"Experiment '{name}' not found.")
        return False

    def get(self, name: str) -> Optional[Experiment]:
        """Look up an experiment by name."""
        return next((e for e in self.experiments if e.name == name), None)

    @property
    def active_experiments(self) -> List[Experiment]:
        return [e for e in self.experiments if e.status == "active"]

    def summary(self) -> str:
        lines = [f"ExperimentRegistry ({len(self.experiments)} total):"]
        for exp in self.experiments:
            lines.append(
                f"  [{exp.status.upper():8}] {exp.name} "
                f"({exp.strategy.id}@{exp.strategy.version}) "
                f"hash={exp.config_hash}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ExperimentRegistry("
            f"total={len(self.experiments)}, "
            f"active={len(self.active_experiments)})"
        )
