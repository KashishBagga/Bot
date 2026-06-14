#!/usr/bin/env python3
"""
Sensitivity Analyzer (Phase 3 - Optimization)
==============================================
Runs a grid search across strategy parameters to find the "Optimal Frontier".
Tests Zone Scores, RVOL thresholds, and HTF Bias modes.
"""

import os
import sys
import logging
from itertools import product
from typing import List, Dict, Any

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.backtesting.advanced_backtester import TransparentBacktester

# Setup logging
logging.basicConfig(level=logging.ERROR) # Suppress normal logs during grid search
logger = logging.getLogger("Analyzer")

def run_sensitivity_analysis(symbols: List[str], days: int = 45):
    tester = TransparentBacktester(symbols, days=days)
    tester.fetch_data()
    
    # ── Grid Parameters ──────────────────────────────────────────
    zone_scores = [40.0, 50.0, 60.0]
    rvol_thresholds = [1.2, 1.5, 2.0]
    htf_modes = ['STRICT', 'RELAXED']
    
    results = []
    
    print("\n" + "="*85)
    print(f"{'ZONE':<6} | {'RVOL':<5} | {'HTF':<8} | {'TRADES':<7} | {'WIN%':<6} | {'PNL (R)':<8} | {'EXPECTANCY':<10}")
    print("-" * 85)
    
    combinations = list(product(zone_scores, rvol_thresholds, htf_modes))
    
    for zone, rvol, htf in combinations:
        params = {
            'min_zone_score': zone,
            'rvol_threshold': rvol,
            'htf_mode': htf,
            'confidence_cutoff': 60.0
        }
        
        metrics, _ = tester.simulate_trades(params, verbose=False)
        
        results.append({
            'zone': zone, 'rvol': rvol, 'htf': htf,
            'trades': metrics['trades'], 'win_rate': metrics['win_rate'],
            'total_r': metrics['total_r'], 'expectancy': metrics['expectancy']
        })
        
        print(f"{zone:<6} | {rvol:<5} | {htf:<8} | {metrics['trades']:<7} | {metrics['win_rate']*100:>5.1f}% | {metrics['total_r']:>7.2f} | {metrics['expectancy']:>9.2f}R")

    # ── Analysis Summary ─────────────────────────────────────────
    print("="*85)
    
    profitable = [r for r in results if r['expectancy'] > 0 and r['trades'] >= 3]
    if profitable:
        best = max(profitable, key=lambda x: x['expectancy'])
        print(f"\n✅ OPTIMAL CONFIGURATION FOUND:")
        print(f"   Zone Score: {best['zone']} | RVOL: {best['rvol']} | HTF: {best['htf']}")
        print(f"   Expectancy: {best['expectancy']:.2f}R per trade")
    else:
        print("\n❌ NO PROFITABLE CONFIGURATION FOUND in this regime.")

if __name__ == "__main__":
    symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
    run_sensitivity_analysis(symbols, days=45)
