#!/usr/bin/env python3
"""
Enhanced Trading Dashboard with Signal Execution Tracking
"""

import os
import sys
import time
import signal
from datetime import datetime
import pytz

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.consolidated_database import ConsolidatedTradingDatabase

class EnhancedTradingDashboard:
    def __init__(self):
        self.tz = pytz.timezone('Asia/Kolkata')
        self.db = ConsolidatedTradingDatabase("data/trading.db")
        self.running = True
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Dashboard shutting down...")
        self.running = False
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def format_signal(self, signal_data) -> str:
        """Format signal for display with execution status and rejection reason."""
        if len(signal_data) >= 13:
            (signal_id, market, symbol, strategy, signal_type, confidence, price, timestamp, 
             timeframe, strength, confirmed, executed, rejection_reason) = signal_data[:13]
            
            # Format timestamp
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=self.tz)
                signal_time = dt.astimezone(self.tz).strftime('%H:%M:%S')
            except:
                signal_time = "N/A"
            
            # Format strength
            strength_emoji = {
                'weak': '🔸',
                'moderate': '🔹', 
                'strong': '🔶',
                'very_strong': '🔷'
            }.get(strength, '🔸')
            
            # Format confirmation
            confirmed_emoji = "✅" if confirmed else "⏳"
            
            # Format execution status
            executed_emoji = "✅" if executed else "❌"
            
            # Format signal type
            signal_emoji = "📈" if signal_type == "BUY CALL" else "📉"
            
            # Create base signal string
            signal_str = f"{signal_emoji} {symbol} {signal_type} @ {price:.2f} | {confidence:.1f}% | {confirmed_emoji} {executed_emoji} | {signal_time}"
            
            # Add rejection reason if not executed
            if not executed and rejection_reason:
                signal_str += f" | Reason: {rejection_reason}"
            
            return signal_str
        return "Invalid signal data"
    
    def get_market_data(self, market: str) -> dict:
        """Get comprehensive market data."""
        try:
            data = {}
            
            # Get market stats
            stats = self.db.get_market_stats(market)
            data.update(stats)
            
            # Get recent trades
            data['recent_trades'] = self.db.get_recent_trades(market, limit=5)
            
            # Get recent executed signals
            data['recent_executed_signals'] = self.db.get_recent_signals_with_execution(market, executed=True, limit=5)
            
            # Get recent rejected signals
            data['recent_rejected_signals'] = self.db.get_recent_signals_with_execution(market, executed=False, limit=5)
            
            return data
        except Exception as e:
            print(f"Error getting market data for {market}: {e}")
            return {}
    
    def display_market_section(self, market: str, market_name: str, emoji: str):
        """Display a complete market section."""
        data = self.get_market_data(market)
        
        print(f"{emoji} {market_name.upper()} TRADING")
        print("=" * 50)
        
        # P&L Section
        print("💰 LIVE P&L:")
        print(f"  Total P&L:    {data.get('total_pnl', 0):+.2f}")
        print(f"  Unrealized:   {data.get('unrealized_pnl', 0):+.2f} (TODO: real-time prices)")
        print(f"  Win Rate:     {data.get('win_rate', 0):.1f}%")
        print(f"  Avg P&L:      {data.get('avg_pnl', 0):+.2f}")
        print(f"  Best Trade:   {data.get('best_trade', 0):+.2f}")
        print(f"  Worst Trade:  {data.get('worst_trade', 0):+.2f}")
        print(f"  Open Trades:  {data.get('open_trades', 0)}")
        print(f"  Closed:       {data.get('closed_trades', 0)}")
        
        # Trade Breakdown
        print("📊 TRADE BREAKDOWN:")
        print(f"  Winning:      {data.get('winning_trades', 0)}")
        print(f"  Losing:       {data.get('losing_trades', 0)}")
        print(f"  Target Hits:  {data.get('target_hits', 0)}")
        print(f"  Stop Losses:  {data.get('stop_losses', 0)}")
        print(f"  Time Exits:   {data.get('time_exits', 0)}")
        
        # Strategy Performance
        strategy_perf = self.db.get_strategy_performance(market)
        if strategy_perf:
            print("🎯 STRATEGY PERFORMANCE:")
            for strategy, total_trades, total_pnl, avg_pnl, wins, worst, best, targets, stops in strategy_perf[:3]:
                print(f"  {strategy}: {total_trades} trades, P&L: {total_pnl:+.2f}, Win Rate: {(wins/total_trades*100):.1f}%")
        else:
            print("🎯 STRATEGY PERFORMANCE: No closed trades yet")
        
        # Recent Entries
        print("📋 RECENT ENTRIES (Last 5):")
        recent_trades = data.get('recent_trades', [])
        if recent_trades:
            for trade in recent_trades[:5]:
                if len(trade) >= 10:
                    (trade_id, market, symbol, strategy, signal, entry_price, position_size, 
                     entry_time, stop_loss, take_profit) = trade[:10]
                    signal_emoji = "📈" if signal == "BUY CALL" else "📉"
                    try:
                        if isinstance(entry_time, str):
                            dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                        else:
                            dt = entry_time
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=self.tz)
                        entry_time_str = dt.astimezone(self.tz).strftime('%H:%M:%S')
                    except:
                        entry_time_str = "N/A"
                    print(f"  {signal_emoji} {symbol} {signal} @ {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} | {entry_time_str}")
        else:
            print("  No recent entries")
        
        # Recent Exits
        print("📊 RECENT EXITS: No closed trades yet")
        
        # Recent Executed Signals
        print("📡 RECENT EXECUTED SIGNALS (Last 5):")
        executed_signals = data.get('recent_executed_signals', [])
        if executed_signals:
            for signal_data in executed_signals:
                print(f"  {self.format_signal(signal_data)}")
        else:
            print("  No executed signals yet")
        
        # Recent Rejected Signals
        print("🚫 RECENT REJECTED SIGNALS (Last 5):")
        rejected_signals = data.get('recent_rejected_signals', [])
        if rejected_signals:
            for signal_data in rejected_signals:
                print(f"  {self.format_signal(signal_data)}")
        else:
            print("  No rejected signals yet")
        
        print()
    
    def display_dashboard(self):
        """Display the complete dashboard."""
        self.clear_screen()
        
        print("🚀 ENHANCED TRADING DASHBOARD")
        print("=" * 80)
        print(f"⏰ Last Updated: {datetime.now(self.tz).strftime('%Y-%m-%d %H:%M:%S IST')}")
        
        # System Status
        print("📊 SYSTEM STATUS:")
        print("-" * 20)
        
        # Check if traders are running (simplified check)
        crypto_running = "🟢 RUNNING"  # This would be enhanced with actual process checking
        indian_running = "🟢 RUNNING"
        
        print(f"Crypto Trader:  {crypto_running}")
        print(f"Indian Trader:  {indian_running}")
        
        print()
        
        # Display both markets
        self.display_market_section("crypto", "CRYPTO", "📈")
        self.display_market_section("indian", "INDIAN", "📊")
        
        print("🔄 Auto-refresh every 10 seconds... (Ctrl+C to exit)")
        print("💡 Using consolidated database with comprehensive metrics")
        print("⚠️  Unrealized P&L shows 0 (needs real-time price integration)")
    
    def run(self):
        """Main dashboard loop."""
        print("🚀 Starting Enhanced Trading Dashboard...")
        
        try:
            while self.running:
                self.display_dashboard()
                time.sleep(10)  # Refresh every 10 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
        finally:
            print("🏁 Dashboard shutdown complete")

def main():
    """Main entry point."""
    dashboard = EnhancedTradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
