#!/usr/bin/env python3
"""
Feature Ablation Engine
========================
Replays historical trade signals with modified filter thresholds to measure
the marginal impact of each filter on Profit Factor, Win Rate, and Expectancy.

Usage:
    from src.analytics.ablation_engine import AblationEngine
    engine = AblationEngine()

    # Test RVOL threshold impact
    df = engine.run_ablation("rvol_tod", [0.0, 0.5, 0.8, 1.0, 1.2, 1.5], days=30)
    print(df.to_string())

Filters supported via `features` JSONB column:
    rvol_tod           - All RVOL-gated strategies
    move_efficiency    - EMA Pullback, VWAP Reclaim, CPR
    atr_percentile     - ATR Squeeze, Straddle (compression gate)
    adx                - New: regime strength gate (ADX > threshold)
    gap_pct            - New: gap magnitude filter

Filters supported via `diagnostics` JSONB column:
    zone_score         - Geometry strategies (min_confluence_score)
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Map filter_name -> JSONB column that contains it
FILTER_SOURCES = {
    "rvol_tod":        "features",
    "move_efficiency": "features",
    "atr_percentile":  "features",
    "adx":             "features",
    "gap_pct":         "features",
    "zone_score":      "diagnostics",
}

# For each filter: the gate direction (>= means "trade allowed if value >= threshold")
FILTER_DIRECTION = {
    "rvol_tod":        ">=",   # higher RVOL = stronger signal
    "move_efficiency": ">=",
    "atr_percentile":  "<=",   # lower ATR percentile = more compression = straddle trigger
    "adx":             ">=",
    "gap_pct":         "ANY",  # unsigned: abs(gap_pct) >= threshold
    "zone_score":      ">=",
}


class AblationEngine:
    """
    Replay historical trade signals with modified filter thresholds.

    For each threshold value, the engine re-applies the filter condition to
    the `features` or `diagnostics` JSONB column of trade_performance records
    and recomputes performance metrics for the surviving subset.

    This answers: "If we had required rvol >= X, how would our results change?"
    """

    def __init__(self, db=None):
        if db is None:
            from src.models.postgres_database import PostgresDatabase
            db = PostgresDatabase()
        self.db = db

    def run_ablation(
        self,
        filter_name: str,
        values: List[float],
        strategy: Optional[str] = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Run a threshold sweep for one filter.

        Parameters
        ----------
        filter_name : Feature key to test (e.g. "rvol_tod", "move_efficiency").
        values      : List of threshold values to test (e.g. [0.0, 0.5, 1.0, 1.5]).
        strategy    : Optional strategy name to restrict the analysis.
        days        : Lookback window in days.

        Returns
        -------
        pd.DataFrame with columns:
            threshold | trade_count | win_rate | expectancy | profit_factor |
            gross_profit | gross_loss | avg_hold_minutes | notes
        """
        if filter_name not in FILTER_SOURCES:
            raise ValueError(
                f"Unknown filter '{filter_name}'. "
                f"Supported: {list(FILTER_SOURCES.keys())}"
            )

        # Fetch all trades for the window (with the relevant JSONB column)
        trades = self._fetch_trades(strategy=strategy, days=days, filter_name=filter_name)

        if not trades:
            logger.warning(f"[Ablation] No trades found for strategy={strategy}, days={days}")
            return pd.DataFrame()

        df_all = pd.DataFrame(trades)
        df_all["_filter_val"] = df_all.apply(
            lambda row: self._extract_filter_value(row, filter_name), axis=1
        )
        direction = FILTER_DIRECTION.get(filter_name, ">=")

        results = []
        for thresh in values:
            subset = self._apply_threshold(df_all, "_filter_val", thresh, direction)
            metrics = self._compute_metrics(subset)
            metrics["threshold"] = thresh
            metrics["filter_name"] = filter_name
            metrics["notes"] = (
                f"{len(subset)}/{len(df_all)} trades pass"
                f" ({len(subset)/max(len(df_all),1):.0%})"
            )
            results.append(metrics)

        result_df = pd.DataFrame(results)
        cols = ["threshold", "trade_count", "win_rate", "expectancy",
                "profit_factor", "sharpe", "kelly_pct",
                "gross_profit", "gross_loss", "avg_hold_minutes", "notes"]
        existing = [c for c in cols if c in result_df.columns]
        return result_df[existing]

    def run_full_ablation(
        self,
        strategy: Optional[str] = None,
        days: int = 30,
    ) -> dict[str, pd.DataFrame]:
        """
        Run ablation for all supported filters.

        Returns a dict mapping filter_name -> ablation DataFrame.
        Useful for generating a complete ablation report.
        """
        default_grids = {
            "rvol_tod":        [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
            "move_efficiency": [0.0, 0.30, 0.40, 0.45, 0.55, 0.65],
            "atr_percentile":  [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
            "adx":             [0.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            "gap_pct":         [0.0, 0.002, 0.004, 0.006, 0.010],
            "zone_score":      [0.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        }
        results = {}
        for fname, grid in default_grids.items():
            try:
                df = self.run_ablation(fname, grid, strategy=strategy, days=days)
                results[fname] = df
                logger.info(f"[Ablation] {fname}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"[Ablation] {fname} failed: {e}")
                results[fname] = pd.DataFrame()
        return results

    # ── Private helpers ─────────────────────────────────────────────────────

    def _fetch_trades(self, strategy, days, filter_name) -> list:
        """Fetch closed trades with the relevant JSONB column."""
        source_col = FILTER_SOURCES.get(filter_name, "features")
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            with self.db._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    exp_clause = "AND strategy = %(strat)s" if strategy else ""
                    cursor.execute(f"""
                        SELECT
                            trade_id, strategy, setup_type, market_regime,
                            final_pnl_r, duration_minutes, entry_time, exit_time,
                            {source_col}
                        FROM trade_performance
                        WHERE exit_time IS NOT NULL
                          AND final_pnl_r IS NOT NULL
                          AND valid = TRUE
                          AND entry_time > NOW() - INTERVAL '{days} days'
                          {exp_clause}
                        ORDER BY entry_time
                    """, {"strat": strategy} if strategy else {})
                    return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            logger.error(f"[Ablation] fetch_trades failed: {e}", exc_info=True)
            return []

    @staticmethod
    def _extract_filter_value(row: dict, filter_name: str) -> Optional[float]:
        """Extract the numeric value for the given filter from features/diagnostics JSONB."""
        source_col = FILTER_SOURCES.get(filter_name, "features")
        blob = row.get(source_col)
        if not isinstance(blob, dict):
            return None
        val = blob.get(filter_name)
        if val is None and filter_name == "zone_score":
            # Diagnostics may use different keys
            val = blob.get("confluence_score") or blob.get("zone_confluence_score")
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _apply_threshold(df: pd.DataFrame, col: str, thresh: float, direction: str) -> pd.DataFrame:
        """Filter DataFrame rows by threshold condition."""
        vals = df[col]
        if direction == ">=":
            mask = vals >= thresh
        elif direction == "<=":
            mask = vals <= thresh
        elif direction == "ANY":
            mask = vals.abs() >= thresh
        else:
            mask = vals >= thresh
        # Rows with None/NaN filter value are excluded by default
        return df[mask.fillna(False)]

    @staticmethod
    def _compute_metrics(df: pd.DataFrame) -> dict:
        """Compute performance metrics for a subset of trades."""
        if df.empty:
            return {
                "trade_count": 0, "win_rate": 0.0, "expectancy": 0.0,
                "profit_factor": 0.0, "sharpe": 0.0, "kelly_pct": 0.0,
                "gross_profit": 0.0, "gross_loss": 0.0, "avg_hold_minutes": 0.0,
            }

        pnl = pd.to_numeric(df["final_pnl_r"], errors="coerce").dropna()
        total   = len(pnl)
        wins    = (pnl > 0).sum()
        losses  = total - wins
        gp      = pnl[pnl > 0].sum()
        gl      = pnl[pnl <= 0].abs().sum()
        mr      = pnl.mean()
        sr      = pnl.std()
        aw      = pnl[pnl > 0].mean() if wins > 0 else 0.0
        al      = pnl[pnl <= 0].mean() if losses > 0 else 0.0

        win_r   = wins / total if total > 0 else 0.0
        loss_r  = losses / total if total > 0 else 0.0
        pf      = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
        sharpe  = (mr / sr * math.sqrt(252)) if sr > 0 else 0.0
        wlr     = abs(aw / al) if al != 0 else 0.0
        kelly   = (win_r - (loss_r / wlr)) if wlr > 0 else 0.0
        hold    = pd.to_numeric(df.get("duration_minutes", pd.Series()), errors="coerce").mean()

        return {
            "trade_count":      total,
            "win_rate":         round(win_r, 4),
            "expectancy":       round(float(mr), 4),
            "profit_factor":    round(min(float(pf), 99.0), 3),
            "sharpe":           round(float(sharpe), 3),
            "kelly_pct":        round(max(float(kelly), 0.0), 4),
            "gross_profit":     round(float(gp), 3),
            "gross_loss":       round(float(gl), 3),
            "avg_hold_minutes": round(float(hold) if not pd.isna(hold) else 0.0, 1),
        }
