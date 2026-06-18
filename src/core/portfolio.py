#!/usr/bin/env python3
"""
Portfolio — Passive per-experiment analytics collector.
=======================================================
NOT in the execution path. The trader loop still enters/exits positions
directly through PositionManager. Portfolio simply observes and accumulates
per-experiment metrics as a side-effect.

This abstraction is intentionally thin in v1. When capital management
arrives (position sizing, drawdown circuit breakers, correlation limits),
Portfolio naturally becomes the gatekeeper between signals and execution.
For now: it watches and measures.

Usage (in trader loop):
    portfolio.on_entry()
    portfolio.on_exit(pnl_r=1.4)
    print(portfolio.metrics.win_rate)
    print(portfolio.metrics.expectancy)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """
    Per-experiment analytics. Updated on every entry and exit event.
    All PnL is in R-multiples (units of initial risk per trade).
    """
    experiment_name: str

    # Trade counts
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    open_positions: int = 0

    # R-multiple PnL
    total_pnl_r: float = 0.0
    daily_pnl_r: float = 0.0
    peak_pnl_r: float = 0.0       # Highest total_pnl_r ever reached
    max_drawdown_r: float = 0.0   # Largest peak-to-trough drawdown in R

    # Session tracking
    session_date: Optional[date] = None
    last_updated: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def expectancy(self) -> float:
        """Average R per trade."""
        return self.total_pnl_r / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        if self.losses == 0:
            return float("inf") if self.wins > 0 else 0.0
        return self.wins / self.losses  # Simplified — improve when trade-level P&L tracked

    def summary_line(self) -> str:
        return (
            f"[{self.experiment_name}] "
            f"Trades={self.total_trades} "
            f"WinRate={self.win_rate:.1%} "
            f"Expectancy={self.expectancy:+.2f}R "
            f"TotalPnL={self.total_pnl_r:+.2f}R "
            f"MaxDD={self.max_drawdown_r:.2f}R "
            f"Open={self.open_positions}"
        )


class Portfolio:
    """
    Passive analytics observer for one experiment.
    Does NOT enter or exit positions. Does NOT gate signals.
    Trader calls on_entry() / on_exit() as lightweight side-effects.

    One Portfolio instance per Experiment in the ExperimentRegistry.
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.metrics = PortfolioMetrics(experiment_name=experiment_name)
        self._pnl_history: List[float] = []  # Rolling trade PnL for future Sharpe etc.

    # ── Event hooks ────────────────────────────────────────────────────────

    def on_entry(self, timestamp: Optional[datetime] = None) -> None:
        """Called when a real position is opened."""
        self.metrics.open_positions += 1
        self.metrics.last_updated = timestamp or datetime.now()

    def on_exit(self, pnl_r: float, timestamp: Optional[datetime] = None) -> None:
        """Called when a real position is closed."""
        self.metrics.total_trades += 1
        self.metrics.open_positions = max(0, self.metrics.open_positions - 1)
        self.metrics.total_pnl_r += pnl_r
        self.metrics.daily_pnl_r += pnl_r
        self.metrics.last_updated = timestamp or datetime.now()

        if pnl_r > 0:
            self.metrics.wins += 1
        else:
            self.metrics.losses += 1

        self._pnl_history.append(pnl_r)

        # Update peak and drawdown
        if self.metrics.total_pnl_r > self.metrics.peak_pnl_r:
            self.metrics.peak_pnl_r = self.metrics.total_pnl_r

        drawdown = self.metrics.peak_pnl_r - self.metrics.total_pnl_r
        if drawdown > self.metrics.max_drawdown_r:
            self.metrics.max_drawdown_r = drawdown

        logger.debug(f"📊 {self.metrics.summary_line()}")

    def reset_daily(self, session_date: Optional[date] = None) -> None:
        """Call at session start to reset daily PnL counter."""
        self.metrics.daily_pnl_r = 0.0
        self.metrics.session_date = session_date or date.today()
        logger.info(f"🔄 [{self.experiment_name}] Daily PnL reset for {self.metrics.session_date}")

    # ── Reporting ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        return self.metrics.summary_line()

    def __repr__(self) -> str:
        return f"Portfolio({self.metrics.summary_line()})"


class PortfolioManager:
    """
    Manages one Portfolio per experiment. Convenience wrapper used by the trader.
    """

    def __init__(self):
        self._portfolios: Dict[str, Portfolio] = {}

    def register(self, experiment_name: str) -> Portfolio:
        p = Portfolio(experiment_name)
        self._portfolios[experiment_name] = p
        return p

    def get(self, experiment_name: str) -> Optional[Portfolio]:
        return self._portfolios.get(experiment_name)

    def on_entry(self, experiment_name: str, timestamp: Optional[datetime] = None) -> None:
        p = self._portfolios.get(experiment_name)
        if p:
            p.on_entry(timestamp)

    def on_exit(self, experiment_name: str, pnl_r: float, timestamp: Optional[datetime] = None) -> None:
        p = self._portfolios.get(experiment_name)
        if p:
            p.on_exit(pnl_r, timestamp)

    def reset_daily(self) -> None:
        today = date.today()
        for p in self._portfolios.values():
            p.reset_daily(today)

    def print_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("📊 Portfolio Summary")
        logger.info("=" * 60)
        for p in self._portfolios.values():
            logger.info(p.summary())
        logger.info("=" * 60)
