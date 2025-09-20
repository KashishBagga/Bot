#!/usr/bin/env python3
"""
Enhanced Actionable Trading Dashboard
Provides clear, actionable insights for trading decisions
"""

import os
import sys
import time
import signal
import sqlite3
from datetime import datetime
import pytz
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.models.consolidated_database import ConsolidatedTradingDatabase

class ActionableTradingDashboard:
    def __init__(self):
        self.tz = pytz.timezone('Asia/Kolkata')
        self.db = ConsolidatedTradingDatabase()
        self.running = True
        self.confidence_threshold = 30.0  # Only show signals above this threshold
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        print("\n🛑 Shutting down dashboard...")
        self.running = False
        
    def get_confidence_color(self, confidence: float) -> str:
        """Get color code for confidence level."""
        if confidence >= 70:
            return "��"  # Strong
        elif confidence >= 50:
            return "🟡"  # Moderate
        else:
            return "🔴"  # Weak
            
    def get_confidence_status(self, confidence: float) -> str:
        """Get status text for confidence level."""
        if confidence >= 70:
            return "STRONG"
        elif confidence >= 50:
            return "MODERATE"
        else:
            return "WEAK"
            
    def format_currency(self, amount: float) -> str:
        """Format currency with proper symbols."""
        if amount >= 0:
            return f"+₹{amount:,.2f}"
        else:
            return f"-₹{abs(amount):,.2f}"
            
    def display_header(self):
        """Display dashboard header."""
        current_time = datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S IST")
        print("\n" + "="*80)
        print(f"🚀 ACTIONABLE TRADING DASHBOARD - {current_time}")
        print("="*80)
        
    def display_system_status(self):
        """Display system status with actionable insights."""
        print("\n🔌 SYSTEM STATUS")
        print("-" * 50)
        
        # Check if traders are running
        crypto_running = self._check_trader_running("crypto_trader")
        indian_running = self._check_trader_running("indian_trader")
        
        print(f"Crypto Trader:  {'🟢 Running' if crypto_running else '🔴 Stopped'}")
        print(f"Indian Trader:  {'🟢 Running' if indian_running else '🔴 Stopped'}")
        
        if not crypto_running and not indian_running:
            print("⚠️  WARNING: No traders running!")
        elif not crypto_running:
            print("⚠️  WARNING: Crypto trader not running")
        elif not indian_running:
            print("⚠️  WARNING: Indian trader not running")
            
    def _check_trader_running(self, trader_name: str) -> bool:
        """Check if a trader is running."""
        try:
            import subprocess
            result = subprocess.run(['pgrep', '-f', trader_name], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
            
    def display_pnl_summary(self, market: str, market_name: str):
        """Display comprehensive P&L summary."""
        print(f"\n💰 {market_name.upper()} P&L SUMMARY")
        print("-" * 50)
        
        # Get realized P&L from closed trades
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT SUM(pnl) FROM closed_trades 
                    WHERE market = ? AND pnl IS NOT NULL
                ''', (market,))
                result = cursor.fetchone()
                realized_pnl = result[0] or 0.0
        except:
            realized_pnl = 0.0
        
        # Get unrealized P&L (simplified for now)
        unrealized_pnl = 0.0  # Will be calculated with real prices
        
        total_pnl = realized_pnl + unrealized_pnl
        
        print(f"Realized P&L:   {self.format_currency(realized_pnl)}")
        print(f"Unrealized P&L: {self.format_currency(unrealized_pnl)}")
        print(f"Total P&L:      {self.format_currency(total_pnl)}")
        
        # P&L status
        if total_pnl > 0:
            print("Status: 🟢 PROFITABLE")
        elif total_pnl < 0:
            print("Status: 🔴 LOSING")
        else:
            print("Status: ⚪ BREAKEVEN")
            
    def display_strategy_performance(self, market: str, market_name: str):
        """Display detailed strategy performance."""
        print(f"\n🎯 {market_name.upper()} STRATEGY PERFORMANCE")
        print("-" * 50)
        
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        strategy,
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl
                    FROM closed_trades 
                    WHERE market = ? AND pnl IS NOT NULL
                    GROUP BY strategy
                ''', (market,))
                
                strategies = []
                for row in cursor.fetchall():
                    strategy, total_trades, winning_trades, total_pnl, avg_pnl = row
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    strategies.append((strategy, total_trades, win_rate, total_pnl or 0.0))
                
                # Sort by total P&L
                strategies.sort(key=lambda x: x[3], reverse=True)
                
                if not strategies:
                    print("No strategy performance data available")
                    return
                
                for strategy, total_trades, win_rate, total_pnl in strategies:
                    # Color code based on performance
                    if total_pnl > 0:
                        pnl_color = "🟢"
                    elif total_pnl < 0:
                        pnl_color = "🔴"
                    else:
                        pnl_color = "⚪"
                        
                    print(f"{pnl_color} {strategy:<25} | {total_trades:>3} trades | Win: {win_rate:>5.1f}% | P&L: {self.format_currency(total_pnl)}")
                    
        except Exception as e:
            print(f"❌ Error loading strategy performance: {e}")
            
    def display_risk_metrics(self, market: str, market_name: str, total_capital: float = 100000):
        """Display risk metrics."""
        print(f"\n⚠️  {market_name.upper()} RISK STATUS")
        print("-" * 50)
        
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Get open positions
                cursor.execute('''
                    SELECT symbol, signal, entry_price, quantity
                    FROM open_trades 
                    WHERE market = ? AND status = 'OPEN'
                ''', (market,))
                
                total_exposure = 0.0
                position_count = 0
                
                for trade in cursor.fetchall():
                    symbol, signal, entry_price, quantity = trade
                    exposure = entry_price * quantity
                    total_exposure += exposure
                    position_count += 1
                
                exposure_percentage = (total_exposure / total_capital * 100) if total_capital > 0 else 0
                max_allowed = 20.0
                is_safe = exposure_percentage <= max_allowed
                
                print(f"Open Positions: {position_count}")
                print(f"Total Exposure: ₹{total_exposure:,.2f} ({exposure_percentage:.1f}% of capital)")
                print(f"Max Allowed: {max_allowed}%")
                
                if is_safe:
                    print("Circuit Breaker: ✅ SAFE")
                else:
                    print("Circuit Breaker: 🔴 OVER-EXPOSED")
                    
        except Exception as e:
            print(f"❌ Error loading risk metrics: {e}")
            
    def display_actionable_signals(self, market: str, market_name: str):
        """Display only actionable signals."""
        print(f"\n📡 {market_name.upper()} ACTIONABLE SIGNALS")
        print("-" * 50)
        
        try:
            # Get recent signals
            recent_signals = self.db.get_recent_signals_with_execution(market, limit=10)
            
            if not recent_signals:
                print("No recent signals")
                return
                
            actionable_count = 0
            
            for signal in recent_signals:
                if len(signal) >= 13:
                    # Parse the actual signal format: (id, market, symbol, strategy, signal, confidence, price, timestamp, timeframe, strength, confirmed, executed, rejection_reason)
                    signal_id, market, symbol, strategy, signal_type, confidence, price, timestamp, timeframe, strength, confirmed, executed, rejection_reason = signal
                    
                    # Convert confidence to percentage if it's a decimal
                    if confidence <= 1.0:
                        confidence_percent = confidence * 100
                    else:
                        confidence_percent = confidence
                    
                    # Only show actionable signals
                    if confidence_percent >= self.confidence_threshold:
                        actionable_count += 1
                        
                        confidence_color = self.get_confidence_color(confidence_percent)
                        confidence_status = self.get_confidence_status(confidence_percent)
                        
                        status_icon = "✅" if executed else "⏳"
                        status_text = "EXECUTED" if executed else "PENDING"
                        
                        print(f"{confidence_color} {timestamp} | {symbol} | {strategy}")
                        print(f"   {signal_type} @ ₹{price} | Conf: {confidence_percent:.1f}% ({confidence_status}) | {status_icon} {status_text}")
                        if rejection_reason:
                            print(f"   Rejection: {rejection_reason}")
                        print()
                        
            if actionable_count == 0:
                print(f"⚠️  No signals above {self.confidence_threshold}% confidence threshold")
                print("Consider lowering threshold or checking strategy performance")
                
        except Exception as e:
            print(f"❌ Error loading signals: {e}")
            
    def display_data_source_status(self):
        """Display data source status."""
        print("\n📊 DATA SOURCE STATUS")
        print("-" * 50)
        
        # Check WebSocket status
        try:
            from src.core.fyers_websocket_manager import FyersWebSocketManager
            ws_manager = FyersWebSocketManager(['NSE:NIFTY50-INDEX'])
            ws_connected = ws_manager.is_connected
        except:
            ws_connected = False
            
        print(f"Primary: WebSocket {'✅' if ws_connected else '❌'}")
        print(f"Fallback: REST API {'✅' if not ws_connected else '❌'}")
        
        if not ws_connected:
            print("⚠️  WARNING: Running on degraded mode (REST API)")
            print("   WebSocket auto-reconnect in progress...")
            
    def run(self):
        """Run the dashboard."""
        print("🚀 Starting Actionable Trading Dashboard...")
        
        while self.running:
            try:
                self.display_header()
                self.display_system_status()
                self.display_data_source_status()
                
                # Display market-specific information
                for market, market_name in [('crypto', 'Crypto'), ('indian', 'Indian')]:
                    self.display_pnl_summary(market, market_name)
                    self.display_strategy_performance(market, market_name)
                    self.display_risk_metrics(market, market_name)
                    self.display_actionable_signals(market, market_name)
                
                print("\n" + "="*80)
                print("🔄 Auto-refresh every 15 seconds | Press Ctrl+C to exit")
                print("💡 Focus on actionable signals above 30% confidence")
                
                if self.running:
                    time.sleep(15)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Dashboard error: {e}")
                time.sleep(5)
                
        print("\n🏁 Dashboard shutdown complete")

if __name__ == "__main__":
    dashboard = ActionableTradingDashboard()
    dashboard.run()
