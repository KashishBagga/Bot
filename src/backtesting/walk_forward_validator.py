#!/usr/bin/env python3
"""
Walk-Forward Validation Engine (Phase 3 - Truth)
================================================
Splits data into In-Sample (IS) and Out-of-Sample (OOS).
The OOS data is "Invisible" to the system until the very end.
This is the only way to prove a genuine edge exists.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.backtesting.advanced_backtester import TransparentBacktester

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("TruthDetector")

class WalkForwardValidator:
    def __init__(self, symbols: List[str], total_days: int = 60):
        self.symbols = symbols
        self.total_days = total_days
        self.tester = TransparentBacktester(symbols, days=total_days)
        
    def run_validation(self):
        logger.info(f"🔍 Initializing Walk-Forward Validation for {self.total_days} days...")
        self.tester.fetch_data()
        
        # Split point: Last 30% of data is Out-of-Sample
        split_idx = int(self.total_days * 0.7)
        logger.info(f"📅 In-Sample: First {split_idx} days | Out-of-Sample: Last {self.total_days - split_idx} days")
        
        params = {'confidence_cutoff': 70.0}
        
        # ── 1. Run In-Sample (IS) Audit ────────────────────────────
        logger.info("\n" + "="*40)
        logger.info("🧪 PHASE 1: IN-SAMPLE AUDIT (Known Data)")
        logger.info("="*40)
        
        original_data = self.tester.historical_data
        is_data = {}
        oos_data = {}
        
        for symbol, bundle in original_data.items():
            m5 = bundle['5m']
            # Find the split date based on calendar days
            start_dt = m5.index[0]
            split_dt = start_dt + timedelta(days=split_idx)
            
            is_data[symbol] = {
                '1d': bundle['1d'][bundle['1d'].index < split_dt] if bundle['1d'] is not None else None,
                '1h': bundle['1h'][bundle['1h'].index < split_dt],
                '5m': m5[m5.index < split_dt]
            }
            oos_data[symbol] = {
                '1d': bundle['1d'][bundle['1d'].index >= split_dt] if bundle['1d'] is not None else None,
                '1h': bundle['1h'][bundle['1h'].index >= split_dt],
                '5m': m5[m5.index >= split_dt]
            }

        # Run IS Test
        self.tester.historical_data = is_data
        is_metrics, _ = self.tester.simulate_trades(params, verbose=False)
        
        # ── 2. Run Out-of-Sample (OOS) Audit ───────────────────────
        logger.info("\n" + "="*40)
        logger.info("💀 PHASE 2: OUT-OF-SAMPLE AUDIT (Invisible Data)")
        logger.info("="*40)
        
        self.tester.historical_data = oos_data
        oos_metrics, _ = self.tester.simulate_trades(params, verbose=False)
        
        # ── 3. Final Comparison (The Truth) ────────────────────────
        logger.info("\n" + "📊 TRUTH REPORT")
        logger.info("-" * 40)
        logger.info(f"In-Sample Expectancy:      {is_metrics['expectancy']:.2f}R")
        logger.info(f"Out-of-Sample Expectancy:  {oos_metrics['expectancy']:.2f}R")
        
        if is_metrics['expectancy'] <= 0:
            logger.error("❌ FAILED: Strategy has no edge even in known data.")
            return

        stability = oos_metrics['expectancy'] / is_metrics['expectancy'] if is_metrics['expectancy'] > 0 else 0
        
        logger.info(f"Expectancy Stability:      {stability*100:.1f}%")
        
        if stability >= 0.7:
            logger.info("✅ SUCCESS: Strategy is robust. Moving to Paper Trading.")
        elif stability > 0.4:
            logger.warning("⚠️ WARNING: Performance degraded in OOS. Optimization likely.")
        else:
            logger.error("❌ FAILED: Strategy is curve-fitted. Do NOT trade.")

if __name__ == "__main__":
    validator = WalkForwardValidator(["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"], total_days=60)
    validator.run_validation()
