#!/usr/bin/env python3
"""
No-op DB stub for backtest-mode IndicatorPipeline construction.
=================================================================
IndicatorPipeline.compute() calls three self.db methods while it's replaying
historical candles:

- save_market_event() — would write real-time-shaped research events for a
  historical (e.g. 2026-06-01) candle into the live `market_events` table,
  indistinguishable from live-trading events.
- get_option_chain_snapshot() / get_atm_oi_series() — read whatever option
  data is CURRENTLY in the DB (there's no historical option-chain store,
  see CLAUDE.md), which is today's data, not point-in-time-correct data for
  the replayed candle. Options-dependent strategies (OIWallReaction,
  PCRExtremeReversal, the spread/condor/butterfly/scalping family) would
  silently trade on the wrong day's option chain if this weren't stubbed out.

Passing this into `IndicatorPipeline(db=BacktestDBStub())` makes both no-ops:
zero market_events writes, and options-dependent strategies correctly see
"no options data" (empty chain / None) for every replayed candle instead of
a wrong one. Everything else in the pipeline (structure, zones, volume,
regime, geometry) needs no options data and is unaffected.
"""

from typing import Any, Dict, List


class BacktestDBStub:
    """Duck-typed no-op replacement for PostgresDatabase's IndicatorPipeline-facing surface."""

    def save_market_event(self, event: Dict[str, Any]) -> None:
        pass

    def get_option_chain_snapshot(self, symbol: str) -> List[Dict[str, Any]]:
        return []

    def get_atm_oi_series(self, symbol: str, minutes: int = 10) -> List[Dict[str, Any]]:
        return []
