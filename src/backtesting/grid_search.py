#!/usr/bin/env python3
"""
Grid-Search Parameter Sweep
=============================
Runs one strategy class through TransparentBacktester.simulate_trades() once
per parameter combination in a grid, ranks the results, and prints/persists
a leaderboard. This is the "auto backtesting engine" gap in the strategy
research framework — CLAUDE.md's Auto Backtesting Engine idea — built as a
thin layer on top of the existing single-registry backtester rather than a
parallel simulation path, so every combo goes through the exact same
MarketSnapshot -> IndicatorPipeline -> Experiment replay as production.

Each combo becomes its own single-Experiment ExperimentRegistry (built
directly, not via experiment_factory.build_registry()) so the sweep never
touches or duplicates the real production experiment set.

Usage:
    python3 src/backtesting/grid_search.py

Edit STRATEGY_CLASS / PARAM_GRID / SYMBOLS / DAYS below for a different
strategy or grid — this is a research script, not a general CLI, so the
grid is defined in code rather than parsed from arbitrary arguments.
"""

import os
import sys
import json
import logging
import itertools
from datetime import datetime
from typing import Dict, List, Any, Type

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.core.base_strategy import BaseStrategy
from src.core.experiment import Experiment
from src.core.experiment_registry import ExperimentRegistry
from src.backtesting.advanced_backtester import TransparentBacktester

logger = logging.getLogger("GridSearch")
logger.setLevel(logging.INFO)
logger.handlers = []
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

os.makedirs("backtest_runs", exist_ok=True)


def _expand_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of a {param_name: [values]} grid into a flat list
    of {param_name: value} kwargs dicts, one per combination."""
    keys = list(param_grid.keys())
    combos = []
    for values in itertools.product(*param_grid.values()):
        combos.append(dict(zip(keys, values)))
    return combos


def run_grid_search(
    strategy_class: Type[BaseStrategy],
    param_grid: Dict[str, List[Any]],
    symbols: List[str],
    days: int,
    rank_by: str = "expectancy",
) -> List[Dict[str, Any]]:
    """Fetches historical data ONCE, then replays it once per param combo
    (each combo gets its own single-Experiment registry) — avoids redundant
    Fyers/Postgres candle fetches across the sweep.

    rank_by: one of 'expectancy', 'total_r', 'win_rate' (higher is better),
    with max_drawdown_r reported alongside as a tiebreaker signal, never
    silently optimized for — a high-expectancy/high-drawdown combo is still
    surfaced, just not auto-preferred.
    """
    tester = TransparentBacktester(symbols, days=days)
    tester.fetch_data()
    if not tester.historical_data:
        logger.error("No historical data loaded for any symbol — aborting grid search")
        return []

    combos = _expand_grid(param_grid)
    logger.info(f"Sweeping {strategy_class.__name__} over {len(combos)} combination(s), "
                f"{days}d, symbols={symbols}")

    leaderboard: List[Dict[str, Any]] = []
    for i, kwargs in enumerate(combos, start=1):
        exp_name = f"{strategy_class.__name__}_grid_{i}"
        registry = ExperimentRegistry()
        registry.register(Experiment(
            name=exp_name,
            strategy=strategy_class(**kwargs),
            params=kwargs,
            description=f"Grid-search combo {i}/{len(combos)}",
        ))

        result = tester.simulate_trades(verbose=False, registry=registry)
        metrics = result['per_experiment'].get(exp_name, {
            'expectancy': 0.0, 'win_rate': 0.0, 'total_r': 0.0, 'trades': 0, 'max_drawdown_r': 0.0,
        })
        row = {'params': kwargs, **metrics}
        leaderboard.append(row)
        logger.info(f"[{i}/{len(combos)}] {kwargs} -> trades={metrics['trades']} "
                    f"win%={metrics['win_rate']*100:.1f} total_r={metrics['total_r']:.2f} "
                    f"exp={metrics['expectancy']:.2f}R max_dd={metrics['max_drawdown_r']:.2f}R")

    leaderboard.sort(key=lambda r: r.get(rank_by, 0.0), reverse=True)

    logger.info("\n" + "=" * 70)
    logger.info(f"LEADERBOARD (ranked by {rank_by}, {len(leaderboard)} combos, {days}d)")
    logger.info("=" * 70)
    for rank, row in enumerate(leaderboard, start=1):
        logger.info(f"#{rank:3d} {row['params']} trades={row['trades']:4d} "
                    f"win%={row['win_rate']*100:5.1f} total_r={row['total_r']:8.2f} "
                    f"exp={row['expectancy']:6.2f}R max_dd={row['max_drawdown_r']:6.2f}R")

    run_id = f"grid_{strategy_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_path = f"backtest_runs/{run_id}.json"
    with open(out_path, "w") as f:
        json.dump({
            'run_id': run_id,
            'strategy': strategy_class.__name__,
            'param_grid': param_grid,
            'days': days,
            'symbols': symbols,
            'rank_by': rank_by,
            'leaderboard': leaderboard,
        }, f, indent=2, default=str)
    logger.info(f"\nSaved full leaderboard to {out_path}")

    return leaderboard


if __name__ == "__main__":
    # Example sweep: ADX/DMI+CCI trend strategy over ADX threshold, CCI
    # period and CCI entry level. Swap STRATEGY_CLASS/PARAM_GRID for any
    # other single-leg BaseStrategy subclass to sweep a different one.
    from src.strategies.adx_dmi_cci_strategy import AdxDmiCciStrategy

    STRATEGY_CLASS = AdxDmiCciStrategy
    PARAM_GRID = {
        "adx_threshold": [15.0, 20.0, 25.0],
        "cci_period": [14, 20],
        "cci_entry_level": [80.0, 100.0],
    }
    SYMBOLS = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
    DAYS = 30
    if len(sys.argv) > 1:
        try:
            DAYS = int(sys.argv[1])
        except ValueError:
            pass

    run_grid_search(STRATEGY_CLASS, PARAM_GRID, SYMBOLS, DAYS, rank_by="expectancy")
