#!/usr/bin/env python3
"""
Monday Readiness Report (The Go/No-Go Gate)
===========================================
Binary status check for production readiness.
"""

import logging
from datetime import datetime, timedelta

from src.analytics.parity_engine import ParityEngine
from src.warehouse.option_warehouse import OptionWarehouse
from src.adapters.data.fyers_data_provider import FyersDataProvider

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReadinessGate")

SYMBOLS = ["NSE:NIFTY50-INDEX"]


def _fetch_parity_inputs(symbols):
    """Pull real recent MTF data + current prices so the parity check has
    something to actually compare, instead of the {}/{} mock inputs that made
    determinism/signal-match checks vacuously trivial."""
    provider = FyersDataProvider()
    end_date = datetime.now()
    historical_data = {}
    for symbol in symbols:
        try:
            d1 = provider.get_historical_data(symbol, end_date - timedelta(days=40), end_date, "1D")
            h1 = provider.get_historical_data(symbol, end_date - timedelta(days=10), end_date, "60")
            m5 = provider.get_historical_data(symbol, end_date - timedelta(days=5), end_date, "5")
            if d1 is not None and h1 is not None and m5 is not None:
                historical_data[symbol] = {"1d": d1, "1h": h1, "5m": m5}
        except Exception as e:
            logger.error(f"❌ Could not fetch historical data for {symbol}: {e}")

    current_prices = provider.get_current_prices_batch(symbols)
    return historical_data, current_prices


def run_readiness_check():
    logger.info("🚦 Running Sunday Night Go/No-Go Readiness Check...")

    # 1. Parity Check — fed real recent data, not mock inputs.
    historical_data, current_prices = _fetch_parity_inputs(SYMBOLS)
    parity = ParityEngine(SYMBOLS)
    parity_stats = parity.run_parity_test(historical_data, current_prices)

    # 2. Warehouse Health
    warehouse = OptionWarehouse(SYMBOLS)
    health = warehouse.get_health_report()

    # Checks that are actually implemented and safe to gate on. entry/exit/pnl
    # match are reported by ParityEngine as None/NOT_IMPLEMENTED by design (no
    # live-fill-vs-replay-fill comparison exists yet) — treating that as a hard
    # FAIL would block every run forever on a feature gap, and treating it as a
    # PASS would fabricate confidence. So they're surfaced separately below,
    # not folded into the go/no-go boolean.
    checks = {
        "Replay Determinism": parity_stats.get('replay_determinism') == "PASS",
        "Signal Match > 95%": (parity_stats.get('signal_match_pct') or 0) >= 95.0,
        "Missing Data < 1%": health.get('missing_pct', 100) < 1.0,
        "Latency < 1000ms": health.get('avg_latency_ms', 5000) < 1000.0,
        "Zero LTP < 1%": health.get('zero_ltp_pct', 100) < 1.0,
    }

    not_implemented = {
        "Entry Match > 95%": parity_stats.get('entry_match_pct'),
        "Exit Match > 95%": parity_stats.get('exit_match_pct'),
        "PnL Match > 90%": parity_stats.get('pnl_match_pct'),
    }

    print("\n| Monday Readiness Check | Status |")
    print("| ---------------------- | ------ |")
    all_pass = True
    for check, status in checks.items():
        pass_str = "✅ PASS" if status else "❌ FAIL"
        if not status:
            all_pass = False
        print(f"| {check:22} | {pass_str:6} |")
    for check, value in not_implemented.items():
        label = "⚠️ NOT IMPLEMENTED" if value is None else ("✅ PASS" if value >= 90.0 else "❌ FAIL")
        print(f"| {check:22} | {label:18} |")

    if parity_stats.get('signal_match_status') == "NO_LIVE_DATA":
        logger.warning(
            "⚠️ No live signals found for parity comparison — signal_match_pct "
            "defaulted to 0 and will FAIL until live signal history exists for "
            "today. This is expected the first time this runs each session."
        )

    print("\n" + "=" * 30)
    if all_pass:
        print("🚀 SYSTEM STATUS: READY")
    else:
        print("🛑 SYSTEM STATUS: NOT READY")
        print("⚠️ Resolve critical failures before market open.")
    if any(v is None for v in not_implemented.values()):
        print("⚠️ NOTE: entry/exit/pnl fill-parity checks are not implemented yet — "
              "READY status does not cover fill-level parity.")
    print("=" * 30)

    return all_pass


if __name__ == "__main__":
    run_readiness_check()
