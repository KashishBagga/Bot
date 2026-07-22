#!/usr/bin/env python3
"""
Live ↔ Replay Parity Framework (P0)
===================================
Verifies the system behaves identical in Live vs Backtest.
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Any

from src.core.enhanced_strategy_engine import EnhancedStrategyEngine
from src.models.postgres_database import PostgresDatabase

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ParityEngine")

class ParityEngine:
    """Audits live signals against hypothetical replay results."""
    
    def __init__(self, symbols: List[str]):
        self.db = PostgresDatabase()
        self.engine = EnhancedStrategyEngine(symbols)

    def run_parity_test(self, historical_data: Dict[str, Dict[str, pd.DataFrame]], current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Verify Signal, Entry, Exit, and PnL parity between Live and Replay.

        HONESTY CONTRACT: this only reports metrics it actually computed.
        - `replay_determinism` is a real check (engine run twice must match).
        - `signal_match_pct` is computed ONLY when live signals for `parity_date`
          exist in the DB; otherwise it is None (NOT a fabricated 100%).
        - entry/exit/pnl parity are NOT yet implemented and are reported as None
          with status NOT_IMPLEMENTED. They must never be presented as passing
          until a real live-fill-vs-replay-fill comparison exists.
        """
        logger.info("🧪 Starting Deep Parity Audit...")

        # 1. Determinism Test: Run Replay twice — this IS real.
        replay_1 = self.engine.generate_signals_for_all_symbols(historical_data, current_prices)
        replay_2 = self.engine.generate_signals_for_all_symbols(historical_data, current_prices)

        def to_list(result):
            if isinstance(result, dict):
                return list(result.values())
            return result if isinstance(result, list) else []

        r1 = to_list(replay_1)
        r2 = to_list(replay_2)

        determinism_pass = (r1 == r2)
        if not determinism_pass:
            logger.error("🚨 DETERMINISM FAILURE: Replay runs do not match!")

        # 2. Real signal-overlap vs live, only if live data is available.
        signal_match_pct = None
        signal_match_status = "NO_LIVE_DATA"
        try:
            replay_keys = {
                (s.get('symbol'), s.get('strategy'), s.get('signal'))
                for s in r1 if s.get('accepted')
            }
            live_keys = self._fetch_live_signal_keys()
            if live_keys:
                overlap = replay_keys & live_keys
                denom = len(replay_keys | live_keys)
                signal_match_pct = round(100.0 * len(overlap) / denom, 2) if denom else None
                signal_match_status = "COMPUTED"
        except Exception as e:
            logger.error(f"Could not compute live signal overlap: {e}")
            signal_match_status = "ERROR"

        scorecard = {
            'replay_determinism': "PASS" if determinism_pass else "FAIL",
            'signal_match_pct': signal_match_pct,          # None if no live data — NOT fabricated
            'signal_match_status': signal_match_status,
            'entry_match_pct': None,                        # NOT_IMPLEMENTED — do not fabricate
            'exit_match_pct': None,
            'pnl_match_pct': None,
            'fill_parity_status': "NOT_IMPLEMENTED",
            # Alert whenever anything is not verifiably passing.
            'parity_alert': (
                not determinism_pass
                or signal_match_pct is None
                or signal_match_pct < 95.0
            ),
        }

        logger.info(
            f"✅ Parity Audit Complete. Determinism: {scorecard['replay_determinism']} | "
            f"signal_match={signal_match_pct} ({signal_match_status}) | "
            f"fill_parity=NOT_IMPLEMENTED"
        )
        return scorecard

    def _fetch_live_signal_keys(self) -> set:
        """Return {(symbol, setup_type, signal_side)} of accepted live signals.

        Returns an empty set when no live signals are available, so callers can
        distinguish "no data" from a real 0% match. Concrete implementation is
        left as a follow-up (requires a live-signals query keyed by date); until
        then this honestly returns nothing rather than pretending to match.
        """
        return set()

if __name__ == "__main__":
    # Test stub
    engine = ParityEngine(["NSE:NIFTY50-INDEX"])
    print(f"Parity Scorecard: {engine.run_parity_test({}, {})}")
