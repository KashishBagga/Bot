#!/usr/bin/env python3
"""
Enhanced Trading Dashboard with Real-Time P&L and WebSocket Integration
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
                signal_time = str(timestamp)[:8]
            
            # Status indicators
            status_icon = "✅" if executed else "⏳" if confirmed else "❌"
            execution_status = "EXECUTED" if executed else "PENDING" if confirmed else "REJECTED"
            
            # Rejection reason
            rejection_info = ""
            if rejection_reason and not executed:
                rejection_info = f" ({rejection_reason})"
            
            return (f"{status_icon} {signal_time} | {symbol} | {strategy} | "
                   f"{signal_type} @ {price} | Conf: {confidence}% | {execution_status}{rejection_info}")
        else:
            return str(signal_data)
    
    def get_websocket_status(self) -> dict:
        """Get WebSocket connection status."""
        try:
            from src.core.fyers_websocket_manager import get_websocket_manager
            
            # Check if WebSocket manager exists
            symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"]
            ws_manager = get_websocket_manager(symbols)
            
            return {
                'connected': ws_manager.is_connected,
                'running': ws_manager.is_running,
                'live_data_count': len(ws_manager.get_all_live_data()),
                'symbols': len(symbols)
            }
        except Exception as e:
            return {
                'connected': False,
                'running': False,
                'live_data_count': 0,
                'symbols': 0,
                'error': str(e)
            }
    
    def get_real_time_pnl(self, market: str) -> dict:
        """Get real-time P&L calculation."""
        try:
            from src.core.fyers_websocket_manager import get_websocket_manager
            
            # Get current prices from WebSocket
            symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"]
            ws_manager = get_websocket_manager(symbols)
            
            current_prices = {}
            if ws_manager.is_connected:
                all_live_data = ws_manager.get_all_live_data()
                for symbol, market_data in all_live_data.items():
                    current_prices[symbol] = market_data.ltp
            
            # Calculate unrealized P&L
            unrealized_pnl = self.db.calculate_unrealized_pnl(market, current_prices)
            
            return {
                'unrealized_pnl': unrealized_pnl,
                'current_prices': current_prices,
                'websocket_connected': ws_manager.is_connected,
                'price_count': len(current_prices)
            }
        except Exception as e:
            return {
                'unrealized_pnl': 0.0,
                'current_prices': {},
                'websocket_connected': False,
                'price_count': 0,
                'error': str(e)
            }
    
    def display_websocket_status(self):
        """Display WebSocket connection status."""
        ws_status = self.get_websocket_status()
        
        print("🔌 WebSocket Status:")
        if ws_status.get('error'):
            print(f"   ❌ Error: {ws_status['error']}")
        else:
            connection_status = "🟢 Connected" if ws_status['connected'] else "🔴 Disconnected"
            running_status = "🟢 Running" if ws_status['running'] else "🔴 Stopped"
            
            print(f"   Connection: {connection_status}")
            print(f"   Status: {running_status}")
            print(f"   Live Data: {ws_status['live_data_count']} symbols")
            print(f"   Subscribed: {ws_status['symbols']} symbols")
    
    def display_real_time_pnl(self, market: str, market_name: str):
        """Display real-time P&L for a market."""
        pnl_data = self.get_real_time_pnl(market)
        
        print(f"💰 {market_name} Real-Time P&L:")
        if pnl_data.get('error'):
            print(f"   ❌ Error: {pnl_data['error']}")
        else:
            ws_status = "🟢 WebSocket" if pnl_data['websocket_connected'] else "🔴 REST API"
            pnl_value = pnl_data['unrealized_pnl']
            pnl_icon = "📈" if pnl_value > 0 else "📉" if pnl_value < 0 else "➖"
            
            print(f"   {pnl_icon} Unrealized P&L: {pnl_value:.2f}")
            print(f"   Data Source: {ws_status}")
            print(f"   Live Prices: {pnl_data['price_count']} symbols")
            
            # Show current prices if available
            if pnl_data['current_prices']:
                print("   Current Prices:")
                for symbol, price in pnl_data['current_prices'].items():
                    print(f"     {symbol}: {price}")
    
    def display_market_section(self, market: str, market_name: str, icon: str):
        """Display comprehensive market information."""
        print(f"\n{icon} {market_name} MARKET")
        print("=" * 50)
        
        # Get market statistics
        stats = self.db.get_market_statistics(market)
        if stats:
            total_trades, open_trades, closed_trades, total_pnl, win_rate = stats
            
            print(f"📊 Statistics:")
            print(f"   Total Trades: {total_trades}")
            print(f"   Open Trades: {open_trades}")
            print(f"   Closed Trades: {closed_trades}")
            print(f"   Total P&L: {total_pnl:.2f}")
            print(f"   Win Rate: {win_rate:.1f}%")
        
        # Display real-time P&L
        self.display_real_time_pnl(market, market_name)
        
        # Get recent signals
        recent_signals = self.db.get_recent_signals(market, limit=5)
        if recent_signals:
            print(f"\n📡 Recent Signals:")
            for signal_data in recent_signals:
                print(f"   {self.format_signal(signal_data)}")
        
        # Get strategy performance
        strategy_perf = self.db.get_strategy_performance(market)
        if strategy_perf:
            print(f"\n🎯 Strategy Performance:")
            for strategy_data in strategy_perf[:3]:  # Top 3 strategies
                strategy, total_trades, total_pnl, avg_pnl, wins, worst, best, targets, stops = strategy_data
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                print(f"   {strategy}: {total_trades} trades, P&L: {total_pnl:.2f}, Win: {win_rate:.1f}%")
    
    def display_dashboard(self):
        """Display the main dashboard."""
        self.clear_screen()
        
        # Header
        current_time = datetime.now(self.tz).strftime('%Y-%m-%d %H:%M:%S IST')
        print(f"🚀 Enhanced Trading Dashboard - {current_time}")
        print("=" * 80)
        
        # WebSocket Status
        self.display_websocket_status()
        print()
        
        # Check if traders are running
        crypto_running = self._is_trader_running("crypto_trader.py")
        indian_running = self._is_trader_running("indian_trader.py")
        
        print("🤖 Trader Status:")
        print(f"Crypto Trader:  {'🟢 Running' if crypto_running else '🔴 Stopped'}")
        print(f"Indian Trader:  {'🟢 Running' if indian_running else '🔴 Stopped'}")
        print()
        
        # Display both markets
        self.display_market_section("crypto", "CRYPTO", "📈")
        self.display_market_section("indian", "INDIAN", "📊")
        
        print("\n🔄 Auto-refresh every 10 seconds... (Ctrl+C to exit)")
        print("💡 Real-time P&L with WebSocket integration")
        print("📡 Live market data streaming enabled")
    
    def _is_trader_running(self, trader_name: str) -> bool:
        """Check if a trader process is running."""
        try:
            import subprocess
            result = subprocess.run(['pgrep', '-f', trader_name], 
                                  capture_output=True, text=True)
            return bool(result.stdout.strip())
        except:
            return False
    
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
