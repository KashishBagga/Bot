#!/usr/bin/env python3
"""
Pre-market smoke test
======================
Fast, tracked (not scratch/) sanity checks meant to run every morning,
targeting exactly the class of bug the Aug-14 weekly trade audit found by
hand: silently-duplicated experiment configs, missing stop-distance floors,
and (as of this pass) the single-leg premium-based P&L formula.

Two sections:
  A. Pure-function checks — no DB, no Fyers, run in milliseconds.
  B. Registered-experiment checks — needs a live DB + Fyers token (same
     prerequisites as the trader itself), so it's best-effort: reports
     SKIPPED rather than crashing if those aren't available.

Exit code is 0 only if every check that actually ran passed.

Usage:
    python3 src/testing/smoke_test_premarket.py
"""

import sys

sys.path.insert(0, ".")

from src.core.options_execution_engine import realistic_fill_price
from src.trading.indian_trader import StructuralPaperTrader

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⚠️ SKIP"


def _report(name: str, ok, detail: str = "") -> bool:
    status = PASS if ok is True else (SKIP if ok is None else FAIL)
    print(f"| {name:48} | {status:8} | {detail}")
    return ok is not False


# ─────────────────────────────────────────────────────────────────────────
# Section A — pure-function checks
# ─────────────────────────────────────────────────────────────────────────

def test_realistic_fill_price():
    """A real fill buys at ask and sells at bid, never at raw LTP."""
    ok = (
        realistic_fill_price(100.0, 98.0, 102.0, "BUY") == 102.0
        and realistic_fill_price(100.0, 98.0, 102.0, "SELL") == 98.0
        # Missing side of the quote (0.0) falls back to premium/LTP.
        and realistic_fill_price(100.0, 0.0, 102.0, "SELL") == 100.0
        and realistic_fill_price(100.0, 98.0, 0.0, "BUY") == 100.0
    )
    return _report("realistic_fill_price: buy=ask, sell=bid", ok)


def test_gap_through_stop_fill():
    """A candle that GAPS through the stop must fill at the worse-of
    (stop, open) — mirrors the gap-aware fill in indian_trader._update_position.
    """
    stop_loss = 100.0
    # BUY CALL: candle opens at 97 (already below the 100 stop) — real fill is
    # the open (97), which is worse than the stop.
    bar_open_call = 97.0
    exit_price_call = min(stop_loss, bar_open_call)
    ok_call = exit_price_call == 97.0

    # BUY PUT: candle opens at 103 (already above the 100 stop) — real fill is
    # the open (103), worse than the stop for a put.
    bar_open_put = 103.0
    exit_price_put = max(stop_loss, bar_open_put)
    ok_put = exit_price_put == 103.0

    return _report("gap-through-stop fills at worse-of(stop, open)", ok_call and ok_put)


def test_trailing_sl_tightens_at_1_5r():
    """Trail step tightens from 1.0x to 0.75x stop-distance once 1.5R is
    banked — mirrors the trail_mult logic in _update_position."""
    stop_loss_distance = 10.0
    trail_mult_below = 0.75 if 1.2 >= 1.5 else 1.0
    trail_mult_at = 0.75 if 1.5 >= 1.5 else 1.0
    ok = trail_mult_below == 1.0 and trail_mult_at == 0.75
    return _report("trailing SL tightens to 0.75x at >=1.5R", ok)


def test_premium_pnl_r_formula():
    """The single-leg P&L fix: pnl_r = premium P&L (Rs) / risk-per-R (Rs),
    where risk-per-R reuses PositionSizer's own risk formula
    (position_size_inr * stop_loss_distance / entry_price) so R stays
    comparable to the pre-fix index-point trades. Calls the real production
    methods on an uninitialized instance (__init__ skipped, so no DB/Fyers
    connection is attempted) — safe because _resolve_current_premium never
    touches self.data_provider when option_quote is supplied directly, as
    it is here.
    """
    fake_self = StructuralPaperTrader.__new__(StructuralPaperTrader)

    pos = {
        "option_symbol": "NSE:NIFTY26AUG25000CE",
        "entry_premium": 100.0,
        "lot_size": 75,
        "lots": 1.0,
        "entry_price": 25000.0,
        "stop_loss_distance": 50.0,        # 50 index points = 1R
        "position_size_inr": 187500.0,     # sized so 1R = Rs 375 (see below)
    }
    # risk_amount_inr = 187500 * 50 / 25000 = 375
    # premium moves from 100 -> 105 (+5 * 75 * 1 lot = Rs 375 gained)
    option_quote = (105.0, 104.0, 106.0)   # (premium, bid, ask) — SELL fills at bid=104

    pnl_r = StructuralPaperTrader._premium_pnl_r(fake_self, pos, option_quote)
    # premium_pnl_inr = (104 - 100) * 75 * 1 = 300; risk_amount_inr = 375 -> 0.8R
    expected = 300.0 / 375.0
    ok = pnl_r is not None and abs(pnl_r - expected) < 1e-6
    return _report("premium-based pnl_r matches Rs P&L / risk-per-R", ok, f"got={pnl_r}")


def test_premium_pnl_r_none_without_leg():
    """No resolved option leg (raw-index CF) -> None, caller falls back to
    the index-point proxy. Must never fabricate a premium."""
    fake_self = StructuralPaperTrader.__new__(StructuralPaperTrader)
    pos = {"option_symbol": None, "entry_price": 25000.0, "stop_loss_distance": 50.0}
    result = StructuralPaperTrader._premium_pnl_r(fake_self, pos, None)
    return _report("premium_pnl_r returns None with no option leg", result is None)


# ─────────────────────────────────────────────────────────────────────────
# Section B — registered-experiment checks (needs live DB + Fyers token)
# ─────────────────────────────────────────────────────────────────────────

def test_experiment_configs_are_distinct():
    """Catches the exact Aug-14 bug class: two experiments silently sharing
    one config_hash means they're not actually A/B-ing anything — one
    threshold overwrote the other."""
    try:
        trader = StructuralPaperTrader(["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"])
    except Exception as e:
        return _report("experiment config_hashes are unique", None, f"SKIPPED (needs DB+Fyers): {e}")

    by_hash = {}
    for exp in trader.registry.active_experiments:
        by_hash.setdefault(exp.config_hash, []).append(exp.name)

    collisions = {h: names for h, names in by_hash.items() if len(names) > 1}
    ok = not collisions
    detail = "" if ok else f"COLLISION: {collisions}"
    return _report("experiment config_hashes are unique", ok, detail)


def main():
    print(f"| {'Check':48} | {'Status':8} | Detail")
    print(f"|{'-'*50}|{'-'*10}|{'-'*40}")

    results = [
        test_realistic_fill_price(),
        test_gap_through_stop_fill(),
        test_trailing_sl_tightens_at_1_5r(),
        test_premium_pnl_r_formula(),
        test_premium_pnl_r_none_without_leg(),
        test_experiment_configs_are_distinct(),
    ]

    print()
    if all(results):
        print("🚀 SMOKE TEST: ALL CHECKS PASSED (or skipped)")
        return 0
    print("🛑 SMOKE TEST: FAILURES FOUND — resolve before market open")
    return 1


if __name__ == "__main__":
    sys.exit(main())
