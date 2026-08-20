#!/usr/bin/env python3
"""
Strategy Registry Report
=========================
Prints the "identity" of every actively-registered experiment: archetype,
exit profile, maturity, regime affinity, and key params — sourced directly
from StrategyMetadata (src/core/base_strategy.py) and
EXPERIMENT_REGIME_AFFINITY (src/core/regime_router.py).

Pure and DB-free — build_registry() has no side effects, so this is safe to
run any time, including outside market hours or without Postgres/Fyers up.
"""

from src.core.experiment_factory import build_registry
from src.core.regime_router import EXPERIMENT_REGIME_AFFINITY

COLS = ["Experiment", "Archetype", "ExitProfile", "Maturity", "RegimeAffinity", "ConfigHash"]
WIDTHS = [34, 20, 20, 10, 26, 12]


def _fmt_affinity(name: str) -> str:
    affinity = EXPERIMENT_REGIME_AFFINITY.get(name, "ANY")
    if affinity == "ANY":
        return "ANY"
    if not affinity:
        return "NONE (never real)"
    return ",".join(sorted(affinity))


def _row(*cells) -> str:
    return " | ".join(str(c).ljust(w)[:w] for c, w in zip(cells, WIDTHS))


def run():
    registry = build_registry()
    print(_row(*COLS))
    print("-+-".join("-" * w for w in WIDTHS))

    by_archetype = {}
    unwired = []

    for exp in registry.experiments:
        meta = exp.strategy.metadata
        print(_row(
            exp.name,
            meta.archetype,
            meta.exit_profile,
            meta.maturity,
            _fmt_affinity(exp.name),
            exp.config_hash,
        ))
        by_archetype.setdefault(meta.archetype, []).append(exp.name)
        if meta.exit_profile == "PREMIUM_UNWIRED":
            unwired.append(exp.name)

    print(f"\n{len(registry.experiments)} experiments across {len(by_archetype)} archetypes:")
    for archetype, names in sorted(by_archetype.items(), key=lambda kv: -len(kv[1])):
        print(f"  {archetype:20} {len(names):3}  {', '.join(names)}")

    if unwired:
        print(f"\n⚠ PREMIUM_UNWIRED (no confirmed exit-management consumer): {', '.join(unwired)}")


if __name__ == "__main__":
    run()
