#!/usr/bin/env python3
"""
Enhanced Performance Dashboard with Advanced Analytics
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPerformanceDashboard:
    """Enhanced performance dashboard with comprehensive analytics"""
    
    def __init__(self):
        self.database = None
        self.real_time_data = None
        self.system_monitor = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize dashboard components"""
        try:
            from src.models.enhanced_database import EnhancedTradingDatabase
            from src.core.enhanced_real_time_manager import EnhancedRealTimeDataManager
            from src.api.fyers import FyersClient
            from src.monitoring.system_monitor import SystemMonitor
            
            self.database = EnhancedTradingDatabase("data/enhanced_trading.db")
            data_provider = FyersClient()
            symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"]
            self.real_time_data = EnhancedRealTimeDataManager(data_provider, symbols)
            self.system_monitor = SystemMonitor()
            
            logger.info("✅ Enhanced performance dashboard initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize dashboard: {e}")
            raise
    
    def display_dashboard(self):
        """Display comprehensive performance dashboard"""
        print("\n" + "="*80)
        print("🚀 ENHANCED TRADING PERFORMANCE DASHBOARD")
        print("="*80)
        
        # Real-time data section
        self._display_real_time_data()
        
        # Database statistics section
        self._display_database_statistics()
        
        # Signal performance section
        self._display_signal_performance()
        
        # Trade performance section
        self._display_trade_performance()
        
        # System health section
        self._display_system_health()
        
        # Market conditions section
        self._display_market_conditions()
        
        print("="*80)
        print(f"📊 Dashboard Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def _display_real_time_data(self):
        """Display real-time market data"""
        print("\n📡 REAL-TIME MARKET DATA")
        print("-" * 40)
        
        try:
            # Get current prices
            current_prices = self.real_time_data.get_current_prices()
            
            if current_prices:
                for symbol, price in current_prices.items():
                    print(f"  {symbol}: ₹{price:,.2f}")
            else:
                print("  ⏳ No real-time data available")
                
        except Exception as e:
            print(f"  ❌ Error fetching real-time data: {e}")
    
    def _display_database_statistics(self):
        """Display database statistics"""
        print("\n🗄️ DATABASE STATISTICS")
        print("-" * 40)
        
        try:
            # Get market statistics
            stats = self.database.get_market_statistics("indian")
            
            print(f"  📊 Total Signals: {stats.get('total_signals', 0)}")
            print(f"  ✅ Executed Signals: {stats.get('executed_signals', 0)}")
            print(f"  ❌ Rejected Signals: {stats.get('rejected_signals', 0)}")
            print(f"  📈 Execution Rate: {stats.get('execution_rate', 0):.1f}%")
            print(f"  🔄 Open Trades: {stats.get('open_trades', 0)}")
            print(f"  ✅ Closed Trades: {stats.get('closed_trades', 0)}")
            print(f"  💰 Total P&L: ₹{stats.get('total_pnl', 0):,.2f}")
            
        except Exception as e:
            print(f"  ❌ Error fetching database statistics: {e}")
    
    def _display_signal_performance(self):
        """Display signal performance analytics"""
        print("\n🎯 SIGNAL PERFORMANCE")
        print("-" * 40)
        
        try:
            # Get entry signals
            entry_signals = self.database.get_entry_signals("indian", limit=10)
            
            if entry_signals:
                print(f"  📊 Recent Signals ({len(entry_signals)}):")
                for signal in entry_signals[:5]:  # Show last 5
                    symbol = signal['symbol']
                    strategy = signal['strategy']
                    signal_type = signal['signal_type']
                    confidence = signal['confidence']
                    price = signal['price']
                    timestamp = signal['timestamp']
                    
                    print(f"    {timestamp} | {symbol} | {strategy} | {signal_type} | {confidence:.1f}% | ₹{price:,.2f}")
            else:
                print("  ⏳ No signals available")
                
        except Exception as e:
            print(f"  ❌ Error fetching signal performance: {e}")
    
    def _display_trade_performance(self):
        """Display trade performance analytics"""
        print("\n📈 TRADE PERFORMANCE")
        print("-" * 40)
        
        try:
            # Get daily summary
            today = datetime.now().strftime("%Y-%m-%d")
            daily_summary = self.database.get_daily_summary(today, "indian")
            
            if daily_summary:
                print(f"  📅 Today's Performance:")
                print(f"    💰 Total P&L: ₹{daily_summary['total_pnl']:,.2f}")
                print(f"    ✅ Win Rate: {daily_summary['win_rate']*100:.1f}%")
                print(f"    ⏱️ Avg Trade Duration: {daily_summary['avg_trade_duration']:.1f} min")
                print(f"    📊 Max Drawdown: {daily_summary['max_drawdown']*100:.1f}%")
                print(f"    📈 Volatility: {daily_summary['volatility']*100:.1f}%")
            else:
                print("  ⏳ No daily summary available")
                
        except Exception as e:
            print(f"  ❌ Error fetching trade performance: {e}")
    
    def _display_system_health(self):
        """Display system health metrics"""
        print("\n🖥️ SYSTEM HEALTH")
        print("-" * 40)
        
        try:
            # Get system metrics
            metrics = self.system_monitor.get_system_metrics()
            
            print(f"  �� CPU Usage: {metrics.cpu_percent:.1f}%")
            print(f"  🧠 Memory Usage: {metrics.memory_percent:.1f}%")
            print(f"  💾 Disk Usage: {metrics.disk_percent:.1f}%")
            
            # WebSocket status
            ws_status = self.real_time_data.get_connection_status()
            if ws_status.get('connected', False):
                print(f"  📡 WebSocket: ✅ Connected")
            else:
                print(f"  📡 WebSocket: ❌ Disconnected")
                
        except Exception as e:
            print(f"  ❌ Error fetching system health: {e}")
    
    def _display_market_conditions(self):
        """Display market conditions"""
        print("\n🌊 MARKET CONDITIONS")
        print("-" * 40)
        
        try:
            # Get current market conditions (mock for now)
            conditions = {
                "NSE:NIFTY50-INDEX": "TRENDING_UP",
                "NSE:NIFTYBANK-INDEX": "SIDEWAYS",
                "NSE:FINNIFTY-INDEX": "TRENDING_DOWN"
            }
            
            for symbol, condition in conditions.items():
                print(f"  {symbol}: {condition}")
                
        except Exception as e:
            print(f"  ❌ Error fetching market conditions: {e}")
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring"""
        print(f"🚀 Starting enhanced performance monitoring (every {interval_seconds}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")

def main():
    """Main function"""
    dashboard = EnhancedPerformanceDashboard()
    
    try:
        # Display initial dashboard
        dashboard.display_dashboard()
        
        # Start continuous monitoring
        dashboard.start_monitoring(interval_seconds=30)
        
    except Exception as e:
        logger.error(f"❌ Dashboard error: {e}")

if __name__ == "__main__":
    main()
