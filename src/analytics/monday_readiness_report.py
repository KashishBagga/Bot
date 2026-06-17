#!/usr/bin/env python3
"""
Monday Readiness Report (The Go/No-Go Gate)
===========================================
Binary status check for production readiness.
"""

import logging
from src.analytics.parity_engine import ParityEngine
from src.warehouse.option_warehouse import OptionWarehouse
from src.analytics.trade_auditor import TradeAuditor

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReadinessGate")

def run_readiness_check():
    logger.info("🚦 Running Sunday Night Go/No-Go Readiness Check...")
    
    # 1. Parity Check
    parity = ParityEngine(["NSE:NIFTY50-INDEX"])
    # Mock data for readiness check
    parity_stats = parity.run_parity_test({}, {})
    
    # 2. Warehouse Health
    warehouse = OptionWarehouse(["NSE:NIFTY50-INDEX"])
    health = warehouse.get_health_report()
    
    # Check criteria
    checks = {
        "Replay Determinism": parity_stats.get('replay_determinism') == "PASS",
        "Signal Match > 95%": parity_stats.get('signal_match_pct', 0) >= 95.0,
        "Entry Match > 95%": parity_stats.get('entry_match_pct', 0) >= 95.0,
        "Exit Match > 95%": parity_stats.get('exit_match_pct', 0) >= 95.0,
        "PnL Match > 90%": parity_stats.get('pnl_match_pct', 0) >= 90.0,
        "Missing Data < 1%": health.get('missing_pct', 100) < 1.0,
        "Latency < 1000ms": health.get('avg_latency_ms', 5000) < 1000.0,
        "Zero LTP < 1%": health.get('zero_ltp_pct', 100) < 1.0
    }
    
    print("\n| Monday Readiness Check | Status |")
    print("| ---------------------- | ------ |")
    all_pass = True
    for check, status in checks.items():
        pass_str = "✅ PASS" if status else "❌ FAIL"
        if not status: all_pass = False
        print(f"| {check:22} | {pass_str:6} |")
        
    print("\n" + "=" * 30)
    if all_pass:
        print("🚀 SYSTEM STATUS: READY")
        print("=" * 30)
    else:
        print("🛑 SYSTEM STATUS: NOT READY")
        print("⚠️ Resolve critical failures before market open.")
        print("=" * 30)

if __name__ == "__main__":
    run_readiness_check()
