#!/usr/bin/env python3
"""
Institutional Structural Trader (Live Paper Mode)
=================================================
Version: 3.1 (Code Freeze)
- Uses Fractal Structural Bias (Daily HH/HL)
- Uses ToD Normalized RVOL
- Uses Structural Invalidation Stops
- Logs all "False Negatives" for regime analysis
"""

import os
import sys
import time
import logging
import schedule
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict

# Path Injection
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.adapters.data.fyers_data_provider import FyersDataProvider
from src.core.enhanced_strategy_engine import EnhancedStrategyEngine

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("paper_trading.log"), logging.StreamHandler()]
)
logger = logging.getLogger("LiveTrader")

class StructuralPaperTrader:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data_provider = FyersDataProvider()
        # Initialize with Frozen Phase 3.1 Parameters
        self.engine = EnhancedStrategyEngine(
            symbols, 
            min_zone_score=60.0, 
            rvol_threshold=2.0
        )
        self.tz = ZoneInfo("Asia/Kolkata")
        logger.info("🏛️ Structural Paper Trader Initialized | Architecture Frozen")

    def market_loop(self):
        """Main loop to be run every 5 minutes during market hours."""
        now = datetime.now(self.tz)
        if not (9 <= now.hour <= 15):
            return

        print(f"\n--- {now.strftime('%H:%M:%S')} Market Pulse ---")
        
        try:
            # 1. Fetch Multi-Timeframe Data
            mtf_data = {}
            current_prices = {}
            
            end_date = datetime.now(self.tz)
            start_date_d1 = end_date - timedelta(days=40)
            start_date_h1 = end_date - timedelta(days=10)
            start_date_m5 = end_date - timedelta(days=5)
            
            for symbol in self.symbols:
                # Fetch data using explicit date ranges
                d1 = self.data_provider.get_historical_data(symbol, start_date_d1, end_date, "1D")
                h1 = self.data_provider.get_historical_data(symbol, start_date_h1, end_date, "60")
                m5 = self.data_provider.get_historical_data(symbol, start_date_m5, end_date, "5")
                
                if d1 is not None and h1 is not None and m5 is not None:
                    mtf_data[symbol] = {'1d': d1, '1h': h1, '5m': m5}
                    current_prices[symbol] = m5['close'].iloc[-1]
                else:
                    logger.warning(f"⚠️ Could not fetch complete MTF data for {symbol}")

            # 2. Generate Structural Signals
            signals = self.engine.generate_signals_for_all_symbols(mtf_data, current_prices)

            # 3. Handle Signals & Diagnostic Logging
            if not signals:
                print("🧘 Status: Sidelined (No Institutional Alignment)")
            
            for sig in signals:
                logger.info(f"🚀 SIGNAL DETECTED: {sig['symbol']} {sig['signal']} | {sig['strategy']}")
                logger.info(f"   Entry: {sig['price']} | SL: {sig['stop_loss']} | TP: {sig['take_profit']} (RR: {sig['rr_ratio']})")
                self._execute_paper_trade(sig)

        except Exception as e:
            logger.error(f"❌ Error in market loop: {e}")

    def _execute_paper_trade(self, sig: Dict):
        """Simulate a trade execution for paper trading."""
        # In a real setup, this would call fyers.place_order()
        # For now, we log it as a 'Live Paper Entry'
        with open("trade_journal.csv", "a") as f:
            f.write(f"{sig['timestamp']},{sig['symbol']},{sig['signal']},{sig['price']},{sig['stop_loss']},{sig['take_profit']},{sig['strategy']}\n")
        print(f"✅ Paper Trade Logged: {sig['symbol']} {sig['signal']} at {sig['price']}")

def main():
    trader = StructuralPaperTrader(["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"])
    
    # Schedule to run every 5 minutes
    schedule.every(5).minutes.do(trader.market_loop)
    
    logger.info("⏱️ Scheduler started. Waiting for next 5-minute candle...")
    
    # Run once immediately for testing
    trader.market_loop()
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
