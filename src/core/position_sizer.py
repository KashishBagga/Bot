#!/usr/bin/env python3
"""
Risk-Based Position Sizer with Kelly Criterion
==============================================
Calculates position sizes using:
  1. Fixed-fraction risk model  (risk 1% of capital per trade)
  2. Half-Kelly Criterion       (adjusts fraction by edge / historical win-rate)
  3. Regime-based multiplier    (reduce size in volatile / sideways regimes)
  4. Per-strategy capital allocation (more capital to better performers)

Usage:
    from src.core.position_sizer import PositionSizer
    sizer = PositionSizer(capital=50000)
    size = sizer.get_position_size(
        entry_price=19500, stop_loss_price=19350, strategy='supertrend_macd_rsi_ema',
        confidence=72, regime='BULL_TREND'
    )
"""

import logging
import math
from typing import Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Configuration (Tighter for Phase 2/3) ──────────────────────────────
RISK_FRACTION        = 0.0075 # Risk 0.75% per trade
KELLY_HALF           = True   # Use half-Kelly for safety
MIN_POSITION_AMOUNT  = 500    # Minimum position size in INR
MAX_POSITION_AMOUNT  = 5_000  # Hard cap ₹5,000 per position (reduced from 10k)
MAX_PORTFOLIO_EXPOSURE = 0.40 # Max 40% capital deployment (reduced from 60%)


@dataclass
class StrategyStats:
    """Running performance stats used for Kelly + capital allocation."""
    total_trades:   int   = 0
    winning_trades: int   = 0
    gross_wins:     float = 0.0   # sum of profitable P&L
    gross_losses:   float = 0.0   # sum of (abs) losing P&L
    kelly_fraction: float = RISK_FRACTION  # dynamically updated

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.5  # assume 50% until we have data
        return self.winning_trades / self.total_trades

    @property
    def avg_win(self) -> float:
        return self.gross_wins / max(1, self.winning_trades)

    @property
    def avg_loss(self) -> float:
        losses = self.total_trades - self.winning_trades
        return self.gross_losses / max(1, losses)

    def update(self, pnl: float):
        """Call after each closed trade to update stats."""
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            self.gross_wins     += pnl
        else:
            self.gross_losses   += abs(pnl)
        self._update_kelly()

    def _update_kelly(self):
        """
        Kelly fraction = W - (1-W)/R
         W = win rate
         R = avg_win / avg_loss  (reward-to-risk ratio)
        We use half-Kelly for conservatism.
        """
        w = self.win_rate
        r = self.avg_win / max(0.01, self.avg_loss)
        kelly = w - (1 - w) / r
        kelly = max(0.0, kelly)
        self.kelly_fraction = kelly / 2 if KELLY_HALF else kelly


# ── Regime multipliers ────────────────────────────────────────────────────────
REGIME_MULTIPLIERS: Dict[str, float] = {
    "BULL_TREND":       1.0,
    "BEAR_TREND":       1.0,
    "BREAKOUT":         1.1,   # slightly larger on breakout
    "REVERSAL":         0.7,
    "HIGH_VOLATILITY":  0.6,
    "LOW_VOLATILITY":   0.8,
    "SIDEWAYS":         0.5,   # range markets: size down
    "UNKNOWN":          0.8,
}

# Confidence multiplier: scale 0.5→1.0 linearly from conf 50→100
def _confidence_multiplier(confidence: float) -> float:
    clamped = max(50.0, min(100.0, confidence))
    return 0.5 + 0.5 * (clamped - 50) / 50.0


class PositionSizer:
    """
    Calculates lot/notional position sizes for each trade.
    Thread-safe for a single-threaded trading loop.
    """

    def __init__(self, capital: float):
        self.capital = capital
        self._strategy_stats: Dict[str, StrategyStats] = {}

    def update_capital(self, new_capital: float):
        """Call daily or after significant P&L changes."""
        self.capital = new_capital

    def record_trade_result(self, strategy: str, pnl: float):
        """
        Feed closed trade P&L so Kelly fraction stays calibrated.
        Call this from _close_trade() in the Trader.
        """
        stats = self._get_or_create_stats(strategy)
        stats.update(pnl)
        logger.debug(
            f"📊 [{strategy}] win_rate={stats.win_rate:.1%} | "
            f"kelly_fraction={stats.kelly_fraction:.3f}"
        )

    def get_position_size(
        self,
        entry_price:     float,
        stop_loss_price: float,
        strategy:        str,
        confidence:      float = 70.0,
        regime:          str   = "UNKNOWN",
        deployed_capital: float = 0.0,   # already used capital across open trades
    ) -> float:
        """
        Calculate position size (notional capital to deploy) for a new trade.

        Formula:
            risk_amount  = capital × effective_fraction
            position_size = risk_amount / risk_per_unit
            risk_per_unit = |entry - stop_loss| / entry  (as fraction)

        Returns: position size in INR (notional)
        """
        try:
            # ── 0. High Water Mark & Drawdown Check ──────────────────
            if not hasattr(self, 'high_water_mark'):
                self.high_water_mark = self.capital
            
            if self.capital > self.high_water_mark:
                self.high_water_mark = self.capital
            
            drawdown = (self.high_water_mark - self.capital) / self.high_water_mark
            drawdown_multiplier = 1.0
            if drawdown > 0.10:
                drawdown_multiplier = 0.5
                logger.warning(f"📉 Drawdown protection active ({drawdown*100:.1f}%): Scaling size by 50%")

            risk_per_unit = abs(entry_price - stop_loss_price) / entry_price
            if risk_per_unit <= 0:
                logger.warning("SL == entry price, using minimum position size")
                return MIN_POSITION_AMOUNT

            # --- Capital availability check ---
            available_capital = self.capital - deployed_capital
            max_allowed       = self.capital * MAX_PORTFOLIO_EXPOSURE - deployed_capital
            if max_allowed <= 0:
                logger.warning("Portfolio exposure limit reached")
                return 0.0

            # --- Effective risk fraction ---
            stats             = self._get_or_create_stats(strategy)
            
            # Regime multiplier
            regime_mult = REGIME_MULTIPLIERS.get(regime.upper(), REGIME_MULTIPLIERS["UNKNOWN"])

            # Confidence multiplier
            conf_mult = _confidence_multiplier(confidence)

            # Final fraction
            final_fraction = effective_fraction * regime_mult * conf_mult

            # Notional to deploy
            risk_amount   = self.capital * final_fraction
            position_size = risk_amount / risk_per_unit

            # Apply bounds
            position_size = max(MIN_POSITION_AMOUNT, position_size)
            position_size = min(MAX_POSITION_AMOUNT, position_size)
            position_size = min(position_size, max_allowed)

            logger.info(
                f"💰 Position size [{strategy}|{regime}|conf={confidence:.0f}]: "
                f"frac={final_fraction:.3f} → ₹{position_size:,.0f} "
                f"(risk/unit={risk_per_unit:.2%})"
            )
            return round(position_size, 2)

        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return MIN_POSITION_AMOUNT

    def get_strategy_allocation_weight(self, strategy: str) -> float:
        """
        Return a capital-weight multiplier for a strategy based on its Sharpe-like score.
        Used to decide how much capital each strategy bucket gets.
        Range: 0.5 (poor) → 1.5 (excellent)
        """
        stats = self._get_or_create_stats(strategy)
        if stats.total_trades < 5:
            return 1.0  # neutral weight until enough data

        # Profit factor = gross_wins / gross_losses
        if stats.gross_losses == 0:
            pf = 2.0
        else:
            pf = stats.gross_wins / stats.gross_losses

        # Map profit factor to weight: PF<0.8→0.5, PF=1.0→1.0, PF≥2.0→1.5
        weight = 0.5 + min(1.0, (pf - 0.5) / 1.5)
        weight = round(max(0.5, min(1.5, weight)), 2)
        return weight

    def get_kelly_report(self) -> dict:
        """Return a summary of Kelly fractions per strategy for monitoring."""
        report = {}
        for name, stats in self._strategy_stats.items():
            report[name] = {
                "trades":         stats.total_trades,
                "win_rate":       round(stats.win_rate, 3),
                "avg_win":        round(stats.avg_win, 2),
                "avg_loss":       round(stats.avg_loss, 2),
                "kelly_fraction": round(stats.kelly_fraction, 4),
                "profit_factor":  round(
                    stats.gross_wins / max(0.01, stats.gross_losses), 2
                ),
            }
        return report

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_or_create_stats(self, strategy: str) -> StrategyStats:
        if strategy not in self._strategy_stats:
            self._strategy_stats[strategy] = StrategyStats()
        return self._strategy_stats[strategy]
