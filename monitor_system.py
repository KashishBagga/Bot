#!/usr/bin/env python3
"""
Enhanced Trading System Monitor
Shows real-time status with live data updates
"""

import time
import os
import sys
from datetime import datetime
from live_paper_trading import LivePaperTradingSystem

def monitor_system():
    """Monitor the trading system in real-time."""
    
    try:
        # Initialize system
        system = LivePaperTradingSystem(initial_capital=30000, verbose=False)
        
        # Track previous values for change detection
        prev_cash = system.cash
        prev_equity = system._equity({})
        prev_open_trades = len(system.open_trades)
        prev_closed_trades = len(system.closed_trades)
        
        while True:
            # Clear screen (works on Unix/Mac)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Get current status
            status = system.get_simple_status()
            
            # Show real-time market data
            try:
                # Get current prices for symbols
                for symbol in system.symbols:
                    try:
                        price = system._get_price_cached(symbol)
                        if price and price > 0:
                        else:
                    except Exception as e:
            logger.error(f"Unexpected error: {e}")
            except Exception as e:
            
            # Show options data status
            try:
                import sqlite3
                conn = sqlite3.connect("unified_trading.db")
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM options_data")
                total_options = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM live_trading_signals")
                total_signals = cursor.fetchone()[0]
                
                cursor.execute("SELECT MAX(timestamp) FROM options_data")
                latest_update = cursor.fetchone()[0]
                
                
                conn.close()
            except Exception as e:
            
            # Show recent activity with change indicators
            
            # Check for changes
            current_cash = system.cash
            current_equity = system._equity({})
            current_open_trades = len(system.open_trades)
            current_closed_trades = len(system.closed_trades)
            
            if current_cash != prev_cash:
                change = current_cash - prev_cash
                prev_cash = current_cash
            
            if current_equity != prev_equity:
                change = current_equity - prev_equity
                prev_equity = current_equity
            
            if current_open_trades != prev_open_trades:
                change = current_open_trades - prev_open_trades
                if change > 0:
                else:
                prev_open_trades = current_open_trades
            
            if current_closed_trades != prev_closed_trades:
                change = current_closed_trades - prev_closed_trades
                prev_closed_trades = current_closed_trades
            
            # Show recent trades
            if system.closed_trades:
                for trade in system.closed_trades[-3:]:  # Last 3 trades
                    pnl_str = f"₹{trade.pnl:+,.2f}" if trade.pnl else "N/A"
            
            # Show system health
            
            # Wait and refresh
            time.sleep(5)
            
    except KeyboardInterrupt:
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    monitor_system() 