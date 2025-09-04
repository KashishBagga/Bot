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
    print("Starting Enhanced Trading System Monitor...")
    print("Press Ctrl+C to stop\n")
    
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
            print(status)
            
            # Show real-time market data
            print("\n=== LIVE MARKET DATA ===")
            try:
                # Get current prices for symbols
                for symbol in system.symbols:
                    try:
                        price = system._get_price_cached(symbol)
                        if price and price > 0:
                            print(f"  {symbol}: ₹{price:,.2f}")
                        else:
                            print(f"  {symbol}: No data")
                    except:
                        print(f"  {symbol}: Error")
            except Exception as e:
                print(f"  Market data error: {e}")
            
            # Show options data status
            print("\n=== OPTIONS DATA STATUS ===")
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
                
                print(f"  Total Options: {total_options:,} contracts")
                print(f"  Live Signals: {total_signals} signals")
                print(f"  Latest Update: {latest_update if latest_update else 'Never'}")
                
                conn.close()
            except Exception as e:
                print(f"  Database error: {e}")
            
            # Show recent activity with change indicators
            print("\n=== RECENT ACTIVITY ===")
            
            # Check for changes
            current_cash = system.cash
            current_equity = system._equity({})
            current_open_trades = len(system.open_trades)
            current_closed_trades = len(system.closed_trades)
            
            if current_cash != prev_cash:
                change = current_cash - prev_cash
                print(f"  💰 Cash changed: ₹{change:+,.2f}")
                prev_cash = current_cash
            
            if current_equity != prev_equity:
                change = current_equity - prev_equity
                print(f"  📈 Equity changed: ₹{change:+,.2f}")
                prev_equity = current_equity
            
            if current_open_trades != prev_open_trades:
                change = current_open_trades - prev_open_trades
                if change > 0:
                    print(f"  🚀 New trade opened: +{change}")
                else:
                    print(f"  🔒 Trade closed: {change}")
                prev_open_trades = current_open_trades
            
            if current_closed_trades != prev_closed_trades:
                change = current_closed_trades - prev_closed_trades
                print(f"  📊 Trade completed: +{change}")
                prev_closed_trades = current_closed_trades
            
            # Show recent trades
            if system.closed_trades:
                print(f"\n  Recent Trades:")
                for trade in system.closed_trades[-3:]:  # Last 3 trades
                    pnl_str = f"₹{trade.pnl:+,.2f}" if trade.pnl else "N/A"
                    print(f"    {trade.strategy} {trade.signal_type} | {trade.underlying} | P&L: {pnl_str}")
            
            # Show system health
            print(f"\n=== SYSTEM HEALTH ===")
            print(f"  Trading Thread: {'Active' if system._trading_thread and system._trading_thread.is_alive() else 'Inactive'}")
            print(f"  Stop Event: {'Set' if system._stop_event.is_set() else 'Not Set'}")
            print(f"  Last Update: {datetime.now().strftime('%H:%M:%S')}")
            
            # Wait and refresh
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    monitor_system() 