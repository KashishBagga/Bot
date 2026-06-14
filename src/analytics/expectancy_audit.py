#!/usr/bin/env python3
"""
Expectancy Audit Tool (Tier 3, Item 11)
=======================================
Calculates statistical edge across different:
- Market Regimes
- Trading Sessions (Opening, Midday, Afternoon)
- Symbols
- Strategy Types
"""

import sqlite3
import pandas as pd
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpectancyAuditor:
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path

    def run_audit(self):
        """Generates a full audit report from closed trades."""
        conn = sqlite3.connect(self.db_path)
        
        # Load trades and research logs
        query = """
        SELECT t.*, r.indicators, r.regime_data
        FROM trades t
        LEFT JOIN research_logs r ON t.trade_id = r.trade_id
        WHERE t.status = 'closed'
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            logger.warning("No closed trades found for audit.")
            return

        # 1. Overall Expectancy
        # E = (Win% * AvgWin) - (Loss% * AvgLoss)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        win_rate = len(wins) / len(df)
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        print("\n" + "="*50)
        print("📊 TRADING SYSTEM AUDIT REPORT")
        print("="*50)
        print(f"Total Trades:    {len(df)}")
        print(f"Win Rate:        {win_rate:.1%}")
        print(f"Avg Win/Loss:    ₹{avg_win:.2f} / ₹{avg_loss:.2f}")
        print(f"Expectancy/Trade: ₹{expectancy:.2f}")
        
        # 2. Expectancy by Regime
        print("\n📈 EXPECTANCY BY REGIME")
        regime_stats = df.groupby('regime')['pnl'].agg(['count', 'mean', 'sum'])
        print(regime_stats.to_string())

        # 3. Strategy Success Rate
        print("\n🎯 STRATEGY PERFORMANCE")
        strat_stats = df.groupby('strategy')['pnl'].agg(['count', 'mean', 'sum'])
        print(strat_stats.to_string())

        # 4. Hourly Performance
        # Assuming entry_time is ISO string
        df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
        print("\n🕒 PERFORMANCE BY HOUR (IST)")
        hour_stats = df.groupby('hour')['pnl'].agg(['count', 'mean'])
        print(hour_stats.to_string())

if __name__ == "__main__":
    auditor = ExpectancyAuditor()
    auditor.run_audit()
