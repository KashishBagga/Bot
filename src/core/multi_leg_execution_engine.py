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
computes net_premium_paid/max_loss/max_profit for the combo shapes this
system currently builds, all defined-risk (every short leg has a protective
long leg further out — no naked/uncovered premium):
    - debit vertical spread  (buy one leg, sell a further OTM leg, same
      direction — max loss = premium paid)
    - long straddle/strangle (buy both a CE and a PE — max loss = premium paid)
    - credit vertical spread (sell the near leg, buy a further OTM leg as
      protection — max loss = spread width minus credit received)
    - iron condor / iron butterfly (sell a near-the-money CE and PE, buy
      further-out wings on both sides for protection — max loss = wing
      width minus credit received; iron butterfly is the same shape with
      both short legs at strikes_away=0 instead of 1 apart)
A combo type with no branch in _risk_profile below is genuinely unsupported —
adding a new combo means adding its risk model here first, not guessing.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any

from src.models.postgres_database import PostgresDatabase
from src.core.options_execution_engine import (
    ExpiryResolver, StrikeSelector, PremiumResolver, OptionContract, realistic_fill_price,
)

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

        # Real fills buy at ask and sell at bid, not at LTP — with 2-4 legs per
        # combo this compounds fast; see realistic_fill_price().
        net_premium_paid = sum(
            realistic_fill_price(leg.contract.premium, leg.contract.bid, leg.contract.ask, leg.side)
            if leg.side == "BUY" else
            -realistic_fill_price(leg.contract.premium, leg.contract.bid, leg.contract.ask, leg.side)
            for leg in legs
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

          - LONG_STRADDLE / LONG_STRANGLE: net debit. max_loss = premium paid
            (both legs expire worthless). max_profit is theoretically
            unbounded on the underlying, so left as None — R-multiple
            tracking still works fine off max_loss as the risk unit.
          - BULL_CALL_SPREAD / BEAR_PUT_SPREAD: net debit. max_loss = premium
            paid, max_profit = spread width (the strike distance between the
            two legs) minus premium paid.
          - BULL_PUT_SPREAD / BEAR_CALL_SPREAD: net credit — net_premium_paid
            is negative here (credit received, not paid). max_loss = spread
            width minus credit received; max_profit = credit received.
        """
        if combo_type in ("LONG_STRADDLE", "LONG_STRANGLE"):
            return net_premium_paid, None

        if combo_type in ("BULL_CALL_SPREAD", "BEAR_PUT_SPREAD"):
            strikes_away_values = [leg.strikes_away for leg in legs]
            spread_width = (max(strikes_away_values) - min(strikes_away_values)) * interval
            max_profit = max(spread_width - net_premium_paid, 0.0)
            return net_premium_paid, max_profit

        if combo_type in ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"):
            strikes_away_values = [leg.strikes_away for leg in legs]
            spread_width = (max(strikes_away_values) - min(strikes_away_values)) * interval
            credit_received = -net_premium_paid
            max_loss = max(spread_width - credit_received, 0.01)
            max_profit = credit_received
            return max_loss, max_profit

        if combo_type in ("IRON_CONDOR", "IRON_BUTTERFLY"):
            # Same defined-risk shape as Iron Condor — short strikes at the
            # short-leg strikes_away distance apart from the long wings on
            # each side (0 apart for Iron Butterfly's ATM shorts, 2 apart for
            # Iron Condor's OTM shorts) — the width/credit math is identical.
            put_legs = [l.strikes_away for l in legs if l.option_type == "PE"]
            call_legs = [l.strikes_away for l in legs if l.option_type == "CE"]
            put_width = (max(put_legs) - min(put_legs)) * interval if len(put_legs) >= 2 else 0.0
            call_width = (max(call_legs) - min(call_legs)) * interval if len(call_legs) >= 2 else 0.0
            spread_width = max(put_width, call_width)
            credit_received = -net_premium_paid
            max_loss = max(spread_width - credit_received, 0.01)
            max_profit = credit_received
            return max_loss, max_profit

        if combo_type == "BUTTERFLY_SPREAD":
            strikes_away_values = sorted(list(set(leg.strikes_away for leg in legs)))
            if len(strikes_away_values) >= 2:
                spread_width = (strikes_away_values[1] - strikes_away_values[0]) * interval
            else:
                spread_width = 0.0
            max_profit = max(spread_width - net_premium_paid, 0.0)
            return net_premium_paid, max_profit

        raise ValueError(f"Unknown combo_type for risk profiling: {combo_type}")

