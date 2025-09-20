#!/usr/bin/env python3
"""
Enhanced Trading Dashboard with All Improvements
Real-time monitoring, performance analytics, and system management
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pytz

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.models.consolidated_database import ConsolidatedTradingDatabase
from src.monitoring.enhanced_performance_monitor import get_performance_monitor
from src.automation.trading_automation import automation_system
from src.strategies.advanced_multi_strategy import AdvancedMultiStrategy

logger = logging.getLogger(__name__)

class EnhancedTradingDashboard:
    """Enhanced trading dashboard with comprehensive monitoring."""
    
    def __init__(self):
        self.tz = pytz.timezone('Asia/Kolkata')
        self.db = ConsolidatedTradingDatabase("data/trading.db")
        self.performance_monitor = get_performance_monitor()
        self.multi_strategy = AdvancedMultiStrategy()
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
    def display_dashboard(self):
        """Display the enhanced dashboard."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        current_time = datetime.now(self.tz)
        print(f"🚀 ENHANCED TRADING DASHBOARD - {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print("=" * 100)
        
        # System Status
        self._display_system_status()
        
        # Performance Metrics
        self._display_performance_metrics()
        
        # Market Statistics
        self._display_market_statistics()
        
        # Strategy Performance
        self._display_strategy_performance()
        
        # Recent Alerts
        self._display_recent_alerts()
        
        # Automation Status
        self._display_automation_status()
        
        print("=" * 100)
        print("🔄 Dashboard auto-refreshes every 30 seconds | Press Ctrl+C to exit")
    
    def _display_system_status(self):
        """Display system status section."""
        print("\n🔌 SYSTEM STATUS")
        print("-" * 50)
        
        # Trading systems status
        crypto_running = self._check_process_running("crypto_trader")
        indian_running = self._check_process_running("indian_trader")
        
        print(f"Crypto Trader:  {'🟢 Running' if crypto_running else '🔴 Stopped'}")
        print(f"Indian Trader:  {'🟢 Running' if indian_running else '🔴 Stopped'}")
        print(f"Performance Monitor: {'🟢 Active' if self.performance_monitor.is_running else '🔴 Inactive'}")
        print(f"Automation System: {'🟢 Active' if automation_system.is_running else '🔴 Inactive'}")
    
    def _display_performance_metrics(self):
        """Display performance metrics section."""
        print("\n📊 PERFORMANCE METRICS")
        print("-" * 50)
        
        try:
            metrics = self.performance_monitor.get_real_time_metrics()
            
            print(f"Total P&L:      ${metrics['total_pnl']:>10.2f}")
            print(f"Daily P&L:      ${metrics['daily_pnl']:>10.2f}")
            print(f"Win Rate:       {metrics['win_rate']:>10.1f}%")
            print(f"Profit Factor:  {metrics['profit_factor']:>10.2f}")
            print(f"Total Trades:   {metrics['total_trades']:>10}")
            print(f"Open Positions: {metrics['open_positions']:>10}")
            print(f"Max Drawdown:   ${metrics['max_drawdown']:>10.2f}")
            
        except Exception as e:
            print(f"❌ Error loading performance metrics: {e}")
    
    def _display_market_statistics(self):
        """Display market statistics section."""
        print("\n📈 MARKET STATISTICS")
        print("-" * 50)
        
        for market in ['crypto', 'indian']:
            try:
                stats = self.db.get_market_stats(market)
                
                print(f"\n{market.upper()} MARKET:")
                print(f"  Open Trades:    {stats.get('open_trades', 0):>6}")
                print(f"  Closed Trades:  {stats.get('closed_trades', 0):>6}")
                print(f"  Avg P&L:        ${stats.get('avg_pnl', 0):>8.2f}")
                print(f"  Total P&L:      ${stats.get('total_pnl', 0):>8.2f}")
                print(f"  Win Rate:       {stats.get('win_rate', 0):>6.1f}%")
                print(f"  Best Trade:     ${stats.get('best_trade', 0):>8.2f}")
                
            except Exception as e:
                print(f"❌ Error loading {market} statistics: {e}")
    
    def _display_strategy_performance(self):
        """Display strategy performance section."""
        print("\n🎯 STRATEGY PERFORMANCE")
        print("-" * 50)
        
        try:
            strategy_perf = self.multi_strategy.get_strategy_performance()
            
            print(f"Market Regime:     {strategy_perf['market_regime']}")
            print(f"Active Strategies: {strategy_perf['active_strategies']}/{strategy_perf['total_strategies']}")
            
            print("\nStrategy Weights:")
            for name, weight in strategy_perf['strategy_weights'].items():
                print(f"  {name:<20}: {weight:>6.1%}")
                
        except Exception as e:
            print(f"❌ Error loading strategy performance: {e}")
    
    def _display_recent_alerts(self):
        """Display recent alerts section."""
        print("\n🚨 RECENT ALERTS")
        print("-" * 50)
        
        try:
            summary = self.performance_monitor.get_performance_summary(1)  # Last 1 day
            alerts = summary['recent_alerts'][:5]  # Show last 5 alerts
            
            if alerts:
                for alert in alerts:
                    timestamp = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                    level_icon = {'INFO': '🔵', 'WARNING': '🟡', 'ERROR': '🔴', 'CRITICAL': '🚨'}.get(alert['level'], '⚪')
                    print(f"  {level_icon} {timestamp} - {alert['title']}")
            else:
                print("  ✅ No recent alerts")
                
        except Exception as e:
            print(f"❌ Error loading alerts: {e}")
    
    def _display_automation_status(self):
        """Display automation status section."""
        print("\n🤖 AUTOMATION STATUS")
        print("-" * 50)
        
        try:
            status = automation_system.get_status()
            
            print(f"Automation Running: {'✅ Yes' if status['running'] else '❌ No'}")
            
            print("\nProcess Status:")
            for name, proc_status in status['processes'].items():
                status_icon = '🟢' if proc_status['running'] else '🔴'
                enabled_text = 'Enabled' if proc_status['enabled'] else 'Disabled'
                pid_text = f"PID: {proc_status['pid']}" if proc_status['pid'] else "Not Running"
                print(f"  {status_icon} {name:<15}: {enabled_text:<8} | {pid_text}")
                
        except Exception as e:
            print(f"❌ Error loading automation status: {e}")
    
    def _check_process_running(self, process_name: str) -> bool:
        """Check if a trading process is running."""
        try:
            import subprocess
            result = subprocess.run(['pgrep', '-f', process_name], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def run_dashboard(self):
        """Run the dashboard with auto-refresh."""
        try:
            while True:
                self.display_dashboard()
                time.sleep(30)  # Refresh every 30 seconds
                
        except KeyboardInterrupt:
            print("\n🏁 Dashboard shutdown")
            self.performance_monitor.stop_monitoring()

def main():
    """Main dashboard entry point."""
    dashboard = EnhancedTradingDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
