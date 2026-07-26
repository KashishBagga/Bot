#!/usr/bin/env python3
"""
Multi-Leg Options Execution Engine
====================================
Resolves N option legs (vertical spreads, straddles, strangles) atomically
against real quotes, reusing the same ExpiryResolver / StrikeSelector /
PremiumResolver machinery as the single-leg OptionExecutionEngine — no
fabricated premiums here either; a leg that can't get a real quote fails the
whole combo the same way a single-leg entry already refuses to trade on an
unresolved premium.

This does NOT invent multi-leg risk semantics for arbitrary combos — it only
computes net_premium_paid/max_loss/max_profit for the two combo shapes this
system currently builds (both always a net debit / defined-risk):
    - vertical spread   (buy one leg, sell a further OTM leg, same direction)
    - long straddle/strangle (buy both a CE and a PE)
Selling naked premium (iron condor/butterfly, uncovered short legs) is out of
scope until that combo type gets its own margin/max-loss model — deliberately
not guessed at here.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any

from src.models.postgres_database import PostgresDatabase
from src.core.options_execution_engine import ExpiryResolver, StrikeSelector, PremiumResolver, OptionContract

logger = logging.getLogger("MultiLegExecution")


@dataclass
class ComboLeg:
    option_type: str      # 'CE' or 'PE'
    side: str             # 'BUY' or 'SELL'
    strikes_away: int      # 0 = ATM, signed offset in strike intervals
    contract: OptionContract


@dataclass
class ResolvedCombo:
    combo_type: str
    legs: List[ComboLeg]
    net_premium_paid: float   # per lot; positive = net debit
    max_loss: float           # per lot
    max_profit: Any           # per lot; None = theoretically unbounded (long vol)


class MultiLegExecutionEngine:
    """Facade resolving a list of leg specs into real, tradable ComboLegs."""

    def __init__(self, db: PostgresDatabase, data_provider):
        self.db = db
        self.data_provider = data_provider
        self.expiry_resolver = ExpiryResolver(db)

    def resolve(
        self, index_symbol: str, index_ltp: float, combo_type: str, leg_specs: List[Dict[str, Any]],
    ) -> ResolvedCombo:
        """leg_specs: [{'option_type': 'CE'|'PE', 'side': 'BUY'|'SELL', 'strikes_away': int}, ...]

        Raises ValueError if any leg's premium can't be resolved from a real
        quote — the caller must skip the entry entirely, same contract as
        OptionExecutionEngine.resolve() for single-leg trades.
        """
        expiry = self.expiry_resolver.get_active_expiry(index_symbol, self.data_provider)
        selector = StrikeSelector(index_symbol)
        premium_resolver = PremiumResolver(self.db, self.data_provider)
        base = "BANKNIFTY" if "BANK" in index_symbol else "NIFTY"

        legs: List[ComboLeg] = []
        for spec in leg_specs:
            option_type = spec["option_type"]
            side = spec["side"]
            strikes_away = spec.get("strikes_away", 0)

            strike = selector.select_strike_offset(index_ltp, option_type, strikes_away)
            option_symbol = f"NSE:{base}{expiry}{strike}{option_type}"

            premium, bid, ask, volume = premium_resolver.resolve_premium(
                index_symbol, strike, option_type, expiry, option_symbol
            )

            contract = OptionContract(
                symbol=option_symbol, strike=float(strike), expiry=expiry,
                option_type=option_type, premium=premium, bid=bid, ask=ask,
                volume=volume, resolved_at=datetime.now(),
            )
            legs.append(ComboLeg(option_type=option_type, side=side, strikes_away=strikes_away, contract=contract))

        net_premium_paid = sum(
            leg.contract.premium if leg.side == "BUY" else -leg.contract.premium for leg in legs
        )

        max_loss, max_profit = self._risk_profile(combo_type, legs, net_premium_paid, selector.interval)

        logger.info(
            f"Resolved {combo_type} for {index_symbol}: "
            f"{[(l.option_type, l.side, l.contract.strike) for l in legs]} "
            f"net_premium={net_premium_paid:.2f} max_loss={max_loss:.2f}"
        )
        return ResolvedCombo(
            combo_type=combo_type, legs=legs, net_premium_paid=net_premium_paid,
            max_loss=max_loss, max_profit=max_profit,
        )

    @staticmethod
    def _risk_profile(combo_type: str, legs: List[ComboLeg], net_premium_paid: float, interval: float):
        """Max loss / max profit per lot for the combo shapes this system builds.

        Both are net-debit, defined-risk structures:
          - LONG_STRADDLE / LONG_STRANGLE: max_loss = premium paid (both legs
            expire worthless). max_profit is theoretically unbounded on the
            underlying, so left as None — R-multiple tracking still works
            fine off max_loss as the risk unit.
          - BULL_CALL_SPREAD / BEAR_PUT_SPREAD: max_loss = premium paid,
            max_profit = spread width (the strike distance between the two
            legs) minus premium paid.
        """
        if combo_type in ("LONG_STRADDLE", "LONG_STRANGLE"):
            return net_premium_paid, None

        if combo_type in ("BULL_CALL_SPREAD", "BEAR_PUT_SPREAD"):
            strikes_away_values = [leg.strikes_away for leg in legs]
            spread_width = (max(strikes_away_values) - min(strikes_away_values)) * interval
            max_profit = max(spread_width - net_premium_paid, 0.0)
            return net_premium_paid, max_profit

        raise ValueError(f"Unknown combo_type for risk profiling: {combo_type}")
