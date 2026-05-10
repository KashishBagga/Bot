#!/usr/bin/env python3
import sqlite3
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExpectancyTool")

def calculate_expectancy(db_path="trading.db"):
    try:
        conn = sqlite3.connect(db_path)
        
        # Load closed trades
        query = "SELECT strategy, regime, pnl, commission, confidence, market_sentiment FROM closed_trades WHERE pnl IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            logger.info("No closed trades found to analyze.")
            return

        df['net_pnl'] = df['pnl'] - df['commission']
        
        # Global Metrics
        total_trades = len(df)
        win_rate = len(df[df['net_pnl'] > 0]) / total_trades
        avg_win = df[df['net_pnl'] > 0]['net_pnl'].mean() if not df[df['net_pnl'] > 0].empty else 0
        avg_loss = abs(df[df['net_pnl'] <= 0]['net_pnl'].mean()) if not df[df['net_pnl'] <= 0].empty else 1
        
        # Expectancy = (WinProb * AvgWin) - (LossProb * AvgLoss)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        print("\n" + "="*50)
        print("📊 GLOBAL TRADING EXPECTANCY")
        print("="*50)
        print(f"Total Trades:  {total_trades}")
        print(f"Win Rate:      {win_rate*100:.1f}%")
        print(f"Avg Win:       ₹{avg_win:.2f}")
        print(f"Avg Loss:      ₹{avg_loss:.2f}")
        print(f"RR Ratio:      {avg_win/avg_loss:.2f}")
        print(f"EXPECTANCY:    ₹{expectancy:.2f} per trade")
        print("="*50)

        # Analysis by Regime
        print("\n📈 PERFORMANCE BY REGIME")
        regime_stats = df.groupby('regime').agg({
            'net_pnl': ['count', 'sum', 'mean'],
        })
        regime_stats.columns = ['Count', 'Total PnL', 'Avg PnL']
        print(regime_stats)

        # Analysis by Strategy
        print("\n🎯 PERFORMANCE BY STRATEGY")
        strat_stats = df.groupby('strategy').agg({
            'net_pnl': ['count', 'sum', 'mean'],
        })
        strat_stats.columns = ['Count', 'Total PnL', 'Avg PnL']
        print(strat_stats)

        conn.close()
    except Exception as e:
        logger.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    calculate_expectancy()
