#!/usr/bin/env python3
"""
Filter Attribution Dashboard (Week 2 Preview - Deliverable 4)
============================================================
Analyze the impact of risk filters on profitability.
"""

import logging
from src.models.postgres_database import PostgresDatabase

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FilterAttribution")

class FilterAttribution:
    def __init__(self):
        self.db = PostgresDatabase()

    def run_attribution(self):
        """Analyze how many signals each filter removed and the hypothetical impact."""
        logger.info("📊 Analyzing Filter Attribution...")
        
        try:
            with self.db._get_connection() as conn:
                with conn.cursor() as cursor:
                    # 1. Total Signals
                    cursor.execute("SELECT COUNT(*) FROM signals")
                    total = cursor.fetchone()[0] or 1
                    
                    # 2. Attribution by Rejection Reason
                    cursor.execute("""
                        SELECT rejected_reason, COUNT(*) 
                        FROM signals 
                        WHERE accepted = FALSE 
                        GROUP BY rejected_reason
                    """)
                    rejections = cursor.fetchall()
                    
                    print("\n🎯 Filter Attribution Report")
                    print("=" * 40)
                    print(f"Total Signals Generated: {total}")
                    print("-" * 40)
                    print(f"{'Filter/Reason':20} | {'Signals Removed':15} | {'Match %':10}")
                    print("-" * 40)
                    
                    for reason, count in rejections:
                        pct = (count / total) * 100
                        print(f"{reason:20} | {count:15} | {pct:8.1f}%")
                        
                    print("-" * 40)
                    print("💡 Insight: High 'BIAS_OPPOSED' rejections indicate strong directionality filters.")
                    
        except Exception as e:
            logger.error(f"❌ Failed to run attribution: {e}")

if __name__ == "__main__":
    attr = FilterAttribution()
    attr.run_attribution()
