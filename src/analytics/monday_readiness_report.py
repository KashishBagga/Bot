#!/usr/bin/env python3
"""
Monday Readiness Report (The Go/No-Go Gate)
===========================================
Binary status check for production readiness.
"""

import logging
import sys
from datetime import datetime, timedelta

from src.analytics.parity_engine import ParityEngine
from src.warehouse.option_warehouse import OptionWarehouse
from src.adapters.data.fyers_data_provider import FyersDataProvider

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReadinessGate")

SYMBOLS = ["NSE:NIFTY50-INDEX"]


def check_token_health() -> bool:
    """Fast pre-market check that the Fyers access token is actually usable.

    The token expires every morning and refresh requires an interactive
    browser login (authenticate_fyers.py) — there's no way to automate that
    step. What CAN be automated is catching a stale/missing token loudly
    *before* market open instead of the trader silently degrading to
    'no live prices' at 09:15 with only a log warning. Runs first and fast —
    no point running the slower parity/warehouse checks below against a dead
    token, they'll all fail for the same root cause.
    """
    try:
        provider = FyersDataProvider()
        prices = provider.get_current_prices_batch(SYMBOLS)
        if any(v is not None and v > 0 for v in prices.values()):
            logger.info(f"✅ Fyers token OK — live quote received: {prices}")
            return True
        print("\n" + "!" * 70)
        print("❌ FYERS TOKEN INVALID/EXPIRED OR MARKET DATA UNAVAILABLE")
        print("   Got no usable quote for", SYMBOLS)
        print("   Run: python3 authenticate_fyers.py   (before 09:15 IST)")
        print("!" * 70 + "\n")
        return False
    except Exception as e:
        print("\n" + "!" * 70)
        print(f"❌ FYERS TOKEN CHECK FAILED: {e}")
        print("   Run: python3 authenticate_fyers.py   (before 09:15 IST)")
        print("!" * 70 + "\n")
        return False


def _previous_trading_day():
    """Yesterday, rolled back to Friday if that lands on a weekend — this
    check runs pre-market, so "today" has no live trades yet; the most
    recent CLOSED session is what a parity check can actually verify."""
    d = datetime.now().date() - timedelta(days=1)
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d -= timedelta(days=1)
    return d


def run_readiness_check():
    logger.info("🚦 Running Sunday Night Go/No-Go Readiness Check...")

    # 0. Token health — fail fast and loud rather than let a dead token
    # silently fail every check below for the same underlying reason.
    if not check_token_health():
        print("🛑 SYSTEM STATUS: NOT READY (token check failed — see above)")
        return False

    # 1. Parity Check — real backtest replay of the last CLOSED session
    # compared against that session's actual live + counterfactual trades.
    prev_day = _previous_trading_day()
    parity = ParityEngine(SYMBOLS)
    parity_stats = parity.run_fill_parity_test(target_date=prev_day)

    # 2. Warehouse Health
    warehouse = OptionWarehouse(SYMBOLS)
    health = warehouse.get_health_report()

    checks = {
        "Replay Determinism": parity_stats.get('replay_determinism') == "PASS",
        "Missing Data < 1%": health.get('missing_pct', 100) < 1.0,
        "Latency < 1000ms": health.get('avg_latency_ms', 5000) < 1000.0,
        "Zero LTP < 1%": health.get('zero_ltp_pct', 100) < 1.0,
    }

    # Fill-parity fields can legitimately be None — e.g. no live trades fired
    # last session, or the backtester found nothing to replay — that's "no
    # data yet", not a failure, so it's surfaced separately rather than
    # folded into the hard go/no-go boolean.
    fill_parity = {
        "Entry Match > 80%": (parity_stats.get('entry_match_pct'), 80.0),
        "Exit Match > 70%": (parity_stats.get('exit_match_pct'), 70.0),
        "PnL Match > 70%": (parity_stats.get('pnl_match_pct'), 70.0),
    }

    print("\n| Monday Readiness Check | Status |")
    print("| ---------------------- | ------ |")
    all_pass = True
    for check, status in checks.items():
        pass_str = "✅ PASS" if status else "❌ FAIL"
        if not status:
            all_pass = False
        print(f"| {check:22} | {pass_str:6} |")
    for check, (value, threshold) in fill_parity.items():
        label = "⚠️ NO DATA" if value is None else ("✅ PASS" if value >= threshold else "❌ FAIL")
        print(f"| {check:22} | {label:18} |")

    if parity_stats.get('fill_parity_status') in ("NO_LIVE_DATA", "NO_REPLAY_TRADES", "NO_BACKTEST_DATA"):
        logger.warning(
            f"⚠️ Fill-parity check for {prev_day} returned "
            f"{parity_stats.get('fill_parity_status')} — no comparison was possible "
            f"(e.g. no trades fired that session). Not treated as a failure."
        )

    print("\n" + "=" * 30)
    if all_pass:
        print("🚀 SYSTEM STATUS: READY")
    else:
        print("🛑 SYSTEM STATUS: NOT READY")
        print("⚠️ Resolve critical failures before market open.")
    print("=" * 30)

    return all_pass


if __name__ == "__main__":
    sys.exit(0 if run_readiness_check() else 1)
