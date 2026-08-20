#!/usr/bin/env python3
"""
NIFTY-BANKNIFTY Relative Value Strategy
=========================================
Hypothesis: every other strategy in this system trades NIFTY and BANKNIFTY
completely independently — nothing trades the relationship BETWEEN them.
When one index decouples from the other intraday (the NIFTY/BANKNIFTY ratio
stretches away from its own recent rolling mean), that divergence tends to
close back up, independent of either index's own absolute direction. This is
a genuinely different edge from every directional/structural/OI strategy
already in this system: it doesn't care whether the market is trending or
ranging, only whether the two indices are moving together or not.

Execution note: this system has no native "spread" position — each active
trade is keyed by (symbol, experiment_name) and exists on one symbol. A real
pairs trade is two legs (fade the rich one, buy the cheap one). This strategy
gets both legs for "free" from the framework's own calling convention: the
registry calls evaluate() once per symbol per candle on the SAME strategy
instance (exactly how every symbol-agnostic strategy here already works,
e.g. StructuralStrategy). This strategy just remembers each symbol's latest
snapshot and, once it's seen both for the same candle, emits one leg on each
call — the NIFTY call emits the NIFTY leg, the BANKNIFTY call emits the
BANKNIFTY leg, both keyed under this one experiment_name. No framework
changes needed.
"""

import logging
from typing import List, Dict, Any, Optional

import pandas as pd

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot

logger = logging.getLogger(__name__)


class RelativeValueStrategy(BaseStrategy):
    """NIFTY/BANKNIFTY ratio mean-reversion — fades the richer index, buys the cheaper one."""

    metadata = StrategyMetadata(
        id="relative_value_nifty_banknifty",
        name="NIFTY-BANKNIFTY Relative Value",
        hypothesis_id="index_ratio_mean_reversion",
        hypothesis_family="Relative Value",
        hypothesis_text=(
            "When the NIFTY/BANKNIFTY price ratio stretches significantly "
            "away from its own recent rolling mean, the divergence tends to "
            "close — trade the laggard toward the leader, independent of "
            "either index's own trend or range state."
        ),
        version="v1.0",
        archetype="RelativeValue",
        exit_profile="INDEX_TP_EXPANSION",
        maturity="RESEARCH",
        tags=["relative_value", "pairs", "mean_reversion", "cross_symbol"],
    )

    def __init__(
        self,
        nifty_symbol: str = "NSE:NIFTY50-INDEX",
        banknifty_symbol: str = "NSE:NIFTYBANK-INDEX",
        lookback_bars: int = 60,
        z_entry: float = 2.0,
        min_rr: float = 1.2,
        tp_ratio_reversion_fraction: float = 0.6,
    ):
        self.nifty_symbol = nifty_symbol
        self.banknifty_symbol = banknifty_symbol
        self.lookback_bars = lookback_bars
        self.z_entry = z_entry
        self.min_rr = min_rr
        # Target only partial reversion to the rolling mean (not the full
        # distance) — same logic as every other mean-reversion strategy here
        # capping its target rather than assuming a full round-trip.
        self.tp_ratio_reversion_fraction = tp_ratio_reversion_fraction
        self._snapshots: Dict[str, MarketSnapshot] = {}

    def _other_symbol(self, symbol: str) -> Optional[str]:
        if symbol == self.nifty_symbol:
            return self.banknifty_symbol
        if symbol == self.banknifty_symbol:
            return self.nifty_symbol
        return None

    def _compute_ratio_zscore(self, nifty_snap: MarketSnapshot, bank_snap: MarketSnapshot):
        """Returns (z, ratio_now, rolling_mean, rolling_std) or None if there
        isn't enough aligned history yet."""
        n_close = nifty_snap.m5["close"]
        b_close = bank_snap.m5["close"]
        # Align on shared timestamps — the two symbols' candle histories can
        # have slightly different available bars (late data, gaps).
        aligned = pd.concat([n_close, b_close], axis=1, join="inner", keys=["nifty", "bank"]).dropna()
        if len(aligned) < self.lookback_bars + 1:
            return None

        ratio = aligned["nifty"] / aligned["bank"]
        window = ratio.tail(self.lookback_bars)
        rolling_mean = float(window.mean())
        rolling_std = float(window.std())
        if rolling_std <= 1e-9:
            return None

        ratio_now = float(ratio.iloc[-1])
        z = (ratio_now - rolling_mean) / rolling_std
        return z, ratio_now, rolling_mean, rolling_std

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            other_symbol = self._other_symbol(snapshot.symbol)
            if other_symbol is None:
                # Not one of the two symbols this strategy knows about.
                return self._empty_result(experiment_name)

            self._snapshots[snapshot.symbol] = snapshot
            other_snap = self._snapshots.get(other_symbol)
            if other_snap is None or other_snap.timestamp != snapshot.timestamp:
                # Haven't seen the other leg's snapshot for this candle yet —
                # nothing to do on THIS call; the other symbol's call this
                # same candle (or the next one, once both are fresh) does the
                # actual work.
                return self._empty_result(experiment_name)

            nifty_snap = snapshot if snapshot.symbol == self.nifty_symbol else other_snap
            bank_snap = snapshot if snapshot.symbol == self.banknifty_symbol else other_snap

            result = self._compute_ratio_zscore(nifty_snap, bank_snap)
            if result is None:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_ALIGNED_HISTORY"])
            z, ratio_now, rolling_mean, rolling_std = result

            if abs(z) < self.z_entry:
                return self._empty_result(experiment_name)

            # z > 0: NIFTY is relatively rich vs BankNifty (ratio stretched high)
            #   -> fade NIFTY (BUY PUT), buy the laggard BankNifty (BUY CALL)
            # z < 0: NIFTY is relatively cheap vs BankNifty
            #   -> buy the laggard NIFTY (BUY CALL), fade BankNifty (BUY PUT)
            nifty_side = "BUY PUT" if z > 0 else "BUY CALL"
            bank_side = "BUY CALL" if z > 0 else "BUY PUT"

            side = nifty_side if snapshot.symbol == self.nifty_symbol else bank_side
            price = snapshot.current_price
            atr = snapshot.features.get_float("atr")
            if atr <= 0:
                return self._empty_result(experiment_name, errors=["FEATURE_MISSING:atr"])

            # Target: this symbol's price if the ratio reverted PART of the
            # way back to its rolling mean, holding the other symbol's price
            # fixed (the standard simplifying assumption for a ratio target —
            # the actual convergence can come from either leg moving).
            target_ratio = rolling_mean + (ratio_now - rolling_mean) * (1.0 - self.tp_ratio_reversion_fraction)
            if snapshot.symbol == self.nifty_symbol:
                take_profit = target_ratio * bank_snap.current_price
            else:
                take_profit = nifty_snap.current_price / target_ratio if target_ratio > 0 else price

            min_sl_dist = atr * 0.5
            sl = (price - min_sl_dist) if side == "BUY CALL" else (price + min_sl_dist)

            risk_dist = abs(price - sl)
            reward = abs(take_profit - price)
            max_tp_dist = atr * 5.0
            if reward > max_tp_dist:
                reward = max_tp_dist
                take_profit = (price + max_tp_dist) if side == "BUY CALL" else (price - max_tp_dist)

            rejection_reasons: List[str] = []
            rr = round(reward / risk_dist, 2) if risk_dist > 0 else 0.0
            if rr < self.min_rr:
                rejection_reasons.append("LOW_RR")
            if reward <= 0:
                rejection_reasons.append("NO_REVERSION_ROOM")

            confidence = round(min(0.5 + 0.1 * (abs(z) - self.z_entry), 0.9), 2)
            accepted = len(rejection_reasons) == 0
            current_time = snapshot.timestamp
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}_RELVAL_"
                f"{price:.2f}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            )

            diagnostics = {
                "ratio_now": round(ratio_now, 4),
                "ratio_rolling_mean": round(rolling_mean, 4),
                "ratio_rolling_std": round(rolling_std, 6),
                "z_score": round(z, 2),
                "atr": round(atr, 2),
                "rr_ratio": rr,
                "other_symbol": other_symbol,
            }

            sig = {
                "symbol": snapshot.symbol,
                "signal": side,
                "strategy": "RATIO_DIVERGENCE",
                "price": price,
                "stop_loss": sl,
                "take_profit": take_profit,
                "tp1": price + (risk_dist * 1.5) if side == "BUY CALL" else price - (risk_dist * 1.5),
                "rr_ratio": rr,
                "timestamp": current_time.isoformat() if hasattr(current_time, "isoformat") else str(current_time),
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
                "features": snapshot.features.to_dict(),
                "candidate_id": candidate_id,
                "confidence": confidence,
                "diagnostics": diagnostics,
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[RelativeValueStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
