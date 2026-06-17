#!/usr/bin/env python3
"""
Data Quality & Discovery CLI
============================
Nightly job to verify feature population rates.
"""

import logging
from src.models.enhanced_database import EnhancedTradingDatabase

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataQuality")

def run_quality_check():
    db = EnhancedTradingDatabase()
    scores = db.get_data_completeness_score()
    
    if not scores:
        print("⚠️ No trade data found to analyze.")
        return
        
    print("\n📊 Trade Features Coverage Report")
    print("=" * 30)
    
    all_clear = True
    for feature, score in scores.items():
        status = "✅" if score >= 95 else "🚨"
        if score < 95: all_clear = False
        print(f"{feature:20}: {score:6.1f}% {status}")
        
    if not all_clear:
        print("\n⚠️ WARNING: Some features fall below the 95% population threshold!")
    else:
        print("\n✅ DATA QUALITY EXCELLENT: All key features are being captured.")

if __name__ == "__main__":
    run_quality_check()
