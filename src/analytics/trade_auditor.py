#!/usr/bin/env python3
"""
End-of-Day Trade Auditor (P0)
============================
Generates detailed JSON reports for every trade to verify system logic.
"""

import json
import logging
from datetime import datetime
from src.models.postgres_database import PostgresDatabase

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TradeAuditor")

class TradeAuditor:
    def __init__(self):
        self.db = PostgresDatabase()

    def generate_audit_report(self, date_str: str):
        """Audit every trade for the given date."""
        logger.info(f"🧐 Auditing trades for {date_str}...")
        
        try:
            with self.db._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM trade_performance")
                    trades = cursor.fetchall()
                    audit_log = []
                    for trade in trades:
                        # Map postgres row to dict (simplified)
                        features = json.loads(trade[11]) if trade[11] else {}
                        audit_log.append({
                            'trade_id': trade[0],
                            'strategy': trade[3],
                            'strategy_version': features.get('strategy_version', 'v3.2'),
                            'experiment_id': features.get('experiment_id', 'CORE_PAPER'),
                            'feature_version': features.get('feature_version', 'v1.1'),
                            'symbol': trade[4],
                            'pnl': trade[9],
                            'exit_reason': trade[10],
                            'mfe': trade[7],
                            'mae': trade[8]
                        })
                    
                    filename = f"data/audit_{date_str}.json"
                    with open(filename, 'w') as f:
                        json.dump(audit_log, f, indent=4)
                    
                    logger.info(f"✅ Audit report generated: {filename}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to generate audit report: {e}")

if __name__ == "__main__":
    auditor = TradeAuditor()
    auditor.generate_audit_report(datetime.now().strftime("%Y-%m-%d"))
