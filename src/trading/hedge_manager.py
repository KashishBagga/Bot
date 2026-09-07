#!/usr/bin/env python3
"""
HedgeManager — protective hedge-on-loss for open REAL positions.
===================================================================
Hypothesis: once a real position's unrealized loss crosses a threshold, the
original structural/indicator thesis is probably wrong for the rest of the
session — rather than only trailing/stopping the original leg, buy an
opposite-side option sized to cap further downside on that specific position.

Self-contained by design (same rule the strategy archetype recipe in
CLAUDE.md already applies to strategies): computes unrealized loss purely
from the position dict + current price, duplicating the same index_pnl_r
formula indian_trader.py's _update_position() uses, rather than reading any
shared mutable state. One hedge per original trade_id — tracked by the
caller (indian_trader.py keeps the "already hedged" set, this class is
otherwise stateless across calls) so a position sitting past threshold for
multiple ticks doesn't spawn a hedge every candle.

The emitted signal is a normal signal dict — same contract every strategy's
evaluate() produces (symbol, signal, price, stop_loss, take_profit, tp1,
strategy, accepted, rejection_reasons, features, candidate_id, confidence,
diagnostics) — so it flows through the EXACT SAME _enter_position() /
_update_position() engine every real trade uses. No parallel hedge-specific
position lifecycle to maintain.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class HedgeManager:
    """Decides whether an open real position needs a protective hedge leg."""

    EXPERIMENT_NAME = "Hedge_Protective"

    def __init__(
        self,
        loss_r_threshold: float = 1.5,
        hedge_target_r: float = 1.0,
        hedge_stop_r: float = -0.5,
    ):
        self.loss_r_threshold = loss_r_threshold
        self.hedge_target_r = hedge_target_r
        self.hedge_stop_r = hedge_stop_r

    def check_position(self, pos: Dict, current_price: float, timestamp) -> Optional[Dict]:
        """Returns a hedge signal dict if `pos` has crossed the loss
        threshold and hasn't been hedged yet (caller's responsibility to
        check/set that), else None.

        Only ever hedges real, still-open, single-leg directional positions
        (signal in {BUY CALL, BUY PUT}) — combo/multi-leg positions carry
        their own combined-premium risk profile and aren't in scope here.
        """
        if pos.get('is_counterfactual'):
            return None
        side = pos.get('signal')
        if side not in ('BUY CALL', 'BUY PUT'):
            return None

        entry_price = pos.get('entry_price')
        stop_loss_distance = pos.get('stop_loss_distance') or 0.0
        if not entry_price or stop_loss_distance <= 0:
            return None

        if side == 'BUY CALL':
            loss_r = (entry_price - current_price) / stop_loss_distance
        else:
            loss_r = (current_price - entry_price) / stop_loss_distance

        if loss_r < self.loss_r_threshold:
            return None

        # Hedge in the OPPOSITE direction of the original thesis — the leg
        # that gains if the original one keeps losing.
        hedge_side = 'BUY PUT' if side == 'BUY CALL' else 'BUY CALL'
        risk_dist = stop_loss_distance
        sl = current_price - risk_dist if hedge_side == 'BUY CALL' else current_price + risk_dist
        reward_dist = risk_dist * (self.hedge_target_r / abs(self.hedge_stop_r))
        tp = current_price + reward_dist if hedge_side == 'BUY CALL' else current_price - reward_dist

        original_trade_id = pos.get('trade_id') or pos.get('candidate_id') or 'unknown'
        candidate_id = f"cand_hedge_{original_trade_id}"

        logger.warning(
            f"🛡️ HEDGE triggered for {pos.get('symbol')} [{pos.get('experiment_name')}] "
            f"loss={loss_r:.2f}R (threshold={self.loss_r_threshold}) -> {hedge_side} protective leg"
        )

        sig = {
            'symbol': pos['symbol'],
            'signal': hedge_side,
            'strategy': 'HEDGE_PROTECTIVE',
            'price': current_price,
            'stop_loss': sl,
            'take_profit': tp,
            'tp1': tp,
            'rr_ratio': round(self.hedge_target_r / abs(self.hedge_stop_r), 2),
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'accepted': True,
            'rejection_reasons': [],
            'features': pos.get('features', {}),
            'candidate_id': candidate_id,
            'confidence': 0.5,
            'diagnostics': {
                'hedged_trade_id': original_trade_id,
                'hedged_experiment': pos.get('experiment_name'),
                'loss_r_at_trigger': round(loss_r, 2),
            },
            'experiment_name': self.EXPERIMENT_NAME,
            'strategy_id': 'hedge_protective',
            'version': 'v1.0',
        }
        return sig
