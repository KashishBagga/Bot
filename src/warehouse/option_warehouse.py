#!/usr/bin/env python3
"""
Option Warehouse Background Service (Priority 6)
==============================================
Captures raw option chain snapshots (ATM-5 to ATM+5) every few seconds.
"""

import time
import logging
import asyncio
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Any

# Path Injection
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.adapters.data.fyers_data_provider import FyersDataProvider
from src.models.postgres_database import PostgresDatabase

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OptionWarehouse")

class OptionWarehouse:
    """Service to capture and store raw option chain data."""
    
    def __init__(self, symbols: List[str]):
        self.underlyings = symbols
        self.data_provider = FyersDataProvider()
        self.db = PostgresDatabase()
        self.tz = ZoneInfo("Asia/Kolkata")
        self.interval = 3 
        
        # Health Metrics (P0)
        self.stats = {
            'expected': 0,
            'received': 0,
            'zeros': 0,
            'latency_ms': []
        }

    async def run(self):
        """Main loop to capture snapshots during market hours."""
        logger.info(f"🚀 Option Warehouse started for: {self.underlyings}")
        
        while True:
            try:
                now = datetime.now(self.tz)
                # Check market hours
                if (9 <= now.hour <= 15) and (now.minute >= 15 or now.hour > 9) and (now.minute <= 30 or now.hour < 15):
                    if now.weekday() < 5: # Monday to Friday
                        for underlying in self.underlyings:
                            start_time = time.time()
                            self.stats['expected'] += 1
                            chain = self.data_provider.get_option_chain(underlying)
                            latency = (time.time() - start_time) * 1000
                            self.stats['latency_ms'].append(latency)
                            
                            if chain and 'snapshots' in chain:
                                self.stats['received'] += 1
                                self.db.save_option_snapshots(chain['snapshots'])
                                
                                # Quality Check
                                if any(s['ltp'] == 0 for s in chain['snapshots']):
                                    self.stats['zeros'] += 1
                                    
                                logger.debug(f"📊 Captured {len(chain['snapshots'])} option snapshots for {underlying}")
                
                await asyncio.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"❌ Error in Option Warehouse loop: {e}")
                await asyncio.sleep(10)

    def get_health_report(self) -> Dict[str, Any]:
        """Return data quality metrics."""
        expected = self.stats['expected']
        received = self.stats['received']
        missing_pct = ((expected - received) / expected * 100) if expected else 0
        avg_latency = (sum(self.stats['latency_ms']) / len(self.stats['latency_ms'])) if self.stats['latency_ms'] else 0
        
        return {
            'missing_pct': round(missing_pct, 2),
            'zero_ltp_pct': round((self.stats['zeros'] / received * 100), 2) if received else 0,
            'avg_latency_ms': round(avg_latency, 2),
            'health_status': "HEALTHY" if missing_pct < 1 and avg_latency < 1000 else "DEGRADED"
        }

if __name__ == "__main__":
    symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
    warehouse = OptionWarehouse(symbols)
    asyncio.run(warehouse.run())
