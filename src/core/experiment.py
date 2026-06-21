#!/usr/bin/env python3
"""
Experiment — Algorithm + Parameters + Metadata.
================================================
An Experiment binds:
  - A fully configured, immutable strategy instance
  - The parameters used to configure it
  - A config_hash proving identical configuration across time
  - A git_commit capturing the exact code version
  - Rich metadata for the experiments DB table

Design rules:
  - Each Experiment owns its OWN strategy instance. Never share instances.
  - Parameters are baked into the strategy at __init__ time. Not mutated after.
  - config_hash enables reproducibility: same hash = same config, guaranteed.
  - git_commit enables code reproducibility: which exact code produced this result?

Example:
    Experiment(
        name="Structural_v3.2_RVOL1.0",
        strategy=StructuralStrategy(rvol_threshold=1.0, min_zone_score=50.0),
        params={"rvol_threshold": 1.0, "min_zone_score": 50.0},
        description="Production structural strategy — RVOL threshold 1.0x"
    )
"""

import hashlib
import json
import subprocess
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional

from src.core.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """
    Binds a strategy algorithm to a specific parameter configuration.
    The name is the unique key used in all DB tables (experiment_name column).
    """

    name: str
    """
    Unique identifier used as DB key across all tables.
    Convention: "{StrategyName}_{Version}_{KeyParam}"
    Examples: "Structural_v3.2_RVOL1.0", "EMA_20_50", "ORB_15min"
    """

    strategy: BaseStrategy
    """
    Pre-configured, immutable strategy instance.
    Parameters must be baked into the strategy at construction time.
    Never pass the same instance to multiple Experiments.
    """

    params: dict
    """
    Parameter dict stored for reference and DB audit.
    Must match what was passed to strategy.__init__().
    Used only for documentation — the strategy itself is the source of truth.
    """

    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"   # active | paused | completed | failed
    notes: str = ""

    config_hash: str = field(init=False)
    """
    SHA-256 of params (first 16 chars). Computed automatically.
    Proves this experiment uses exactly these parameters.
    Compare hashes to verify two runs used identical configs.
    """

    git_commit: Optional[str] = field(init=False)
    """
    Git SHA of the current HEAD at Experiment construction time.
    Proves which exact code version generated these results.
    None if not in a git repo or git is unavailable.
    """

    def __post_init__(self):
        self.config_hash = Experiment.make_config_hash(self.params)
        self.git_commit = Experiment._get_git_commit()

    @staticmethod
    def make_config_hash(params: dict) -> str:
        """
        Returns a short SHA-256 of the params dict (sorted keys).
        16 characters — readable in logs, unique enough for research use.
        """
        serialised = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """Capture current git HEAD SHA. Returns None if git unavailable."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def to_db_dict(self) -> Dict[str, Any]:
        """Serialise to a dict for insertion into the experiments table."""
        return {
            "name":              self.name,
            "strategy_id":       self.strategy.id,
            "version":           self.strategy.version,
            "config_hash":       self.config_hash,
            "git_commit":        self.git_commit,
            "params":            json.dumps(self.params, default=str),
            "description":       self.description,
            "created_at":        self.created_at,
            "status":            self.status,
            "notes":             self.notes,
            "strategy_metadata": json.dumps(asdict(self.strategy.metadata), default=str) if hasattr(self.strategy, 'metadata') and self.strategy.metadata else None,
        }

    def __repr__(self) -> str:
        return (
            f"Experiment("
            f"name={self.name!r}, "
            f"strategy={self.strategy.id}@{self.strategy.version}, "
            f"config_hash={self.config_hash}, "
            f"git={self.git_commit}, "
            f"status={self.status})"
        )
