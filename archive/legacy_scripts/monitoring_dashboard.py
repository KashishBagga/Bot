#!/usr/bin/env python3
"""
LIVE TRADING MONITORING DASHBOARD
Real-time performance tracking and alerts for profitable trading system
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import json

class TradingMonitor:
    """Monitor live trading performance and generate alerts"""
    
    def __init__(self, db_path: str = "trading_signals.db"):
        self.db_path = db_path
        self.risk_limits = {
            "daily_loss_limit": -2000,
            "max_daily_trades": 20,
            "max_consecutive_losses": 3,
            "min_confidence_threshold": 75,
            "max_drawdown_warning": -1000
        }
    
    def get_daily_performance(self, date: str = None) -> Dict:
        """Get performance metrics for a specific date"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        conn = sqlite3.connect(self.db_path)
        
        # Get live signals for the date
        signals_query = """
        SELECT * FROM trading_signals 
        WHERE DATE(timestamp) = ? 
        ORDER BY timestamp DESC
        """
        signals_df = pd.read_sql_query(signals_query, conn, params=[date])
        
        # Get rejected signals for the date
        rejected_query = """
        SELECT * FROM rejected_signals_live 
        WHERE DATE(timestamp) = ?
        ORDER BY timestamp DESC
        """
        rejected_df = pd.read_sql_query(rejected_query, conn, params=[date])
        
        conn.close()
        
        # Calculate metrics
        total_signals = len(signals_df)
        total_rejected = len(rejected_df)
        total_pnl = 0  # Live trading P&L calculation would need to be implemented separately
        
        # Win/Loss analysis - placeholder for live trading
        wins = losses = 0
        win_rate = 0
        
        # Confidence analysis
        avg_confidence = signals_df['confidence_score'].mean() if not signals_df.empty else 0
        
        return {
            "date": date,
            "total_signals": total_signals,
            "total_rejected": total_rejected,
            "total_pnl": round(total_pnl, 2),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "avg_confidence": round(avg_confidence, 1),
            "signals_df": signals_df,
            "rejected_df": rejected_df
        }
    
    def check_risk_alerts(self, performance: Dict) -> List[str]:
        """Check for risk limit violations and generate alerts"""
        alerts = []
        
        # Daily loss limit
        if performance["total_pnl"] < self.risk_limits["daily_loss_limit"]:
            alerts.append(f"🚨 DAILY LOSS LIMIT EXCEEDED: ₹{performance['total_pnl']} < ₹{self.risk_limits['daily_loss_limit']}")
        
        # Too many trades
        if performance["total_signals"] > self.risk_limits["max_daily_trades"]:
            alerts.append(f"⚠️ HIGH TRADE COUNT: {performance['total_signals']} > {self.risk_limits['max_daily_trades']} daily limit")
        
        # Low win rate warning
        if performance["total_signals"] >= 5 and performance["win_rate"] < 20:
            alerts.append(f"⚠️ LOW WIN RATE: {performance['win_rate']}% < 20% (min threshold)")
        
        # Consecutive losses check - would need live P&L tracking
        # Placeholder for when live trading outcome tracking is implemented
        
        # Unusual rejection rate
        total_attempts = performance["total_signals"] + performance["total_rejected"]
        if total_attempts > 10:
            rejection_rate = (performance["total_rejected"] / total_attempts) * 100
            if rejection_rate > 95:
                alerts.append(f"📊 HIGH REJECTION RATE: {rejection_rate:.1f}% - confidence threshold may be too high")
        
        return alerts
    
    def generate_daily_report(self, date: str = None) -> str:
        """Generate comprehensive daily performance report"""
        performance = self.get_daily_performance(date)
        alerts = self.check_risk_alerts(performance)
        
        date_str = performance["date"]
        
        report = f"""
📊 DAILY TRADING REPORT - {date_str}
{'=' * 50}

💰 FINANCIAL PERFORMANCE:
   • Total P&L: ₹{performance['total_pnl']}
   • Total Trades: {performance['total_signals']}
   • Win/Loss: {performance['wins']}/{performance['losses']}
   • Win Rate: {performance['win_rate']}%
   
🎯 SIGNAL QUALITY:
   • Average Confidence: {performance['avg_confidence']}
   • Signals Generated: {performance['total_signals']}
   • Signals Rejected: {performance['total_rejected']}
   • Rejection Rate: {((performance['total_rejected'] / (performance['total_signals'] + performance['total_rejected'])) * 100) if (performance['total_signals'] + performance['total_rejected']) > 0 else 0:.1f}%

"""
        
        # Add alerts section
        if alerts:
            report += "🚨 RISK ALERTS:\n"
            for alert in alerts:
                report += f"   {alert}\n"
            report += "\n"
        else:
            report += "✅ All risk parameters within limits\n\n"
        
        # Add recent trades
        if not performance["signals_df"].empty:
            report += "📈 RECENT TRADES:\n"
            recent_trades = performance["signals_df"].head(5)
            for _, trade in recent_trades.iterrows():
                time_str = pd.to_datetime(trade['timestamp']).strftime('%H:%M')
                report += f"   {time_str} | {trade['symbol']} | {trade['signal']} | Conf: {trade.get('confidence_score', 0)}\n"
            report += "\n"
        
        # Performance vs targets
        report += "🎯 TARGET PROGRESS:\n"
        daily_target = 100  # ₹100/day target
        if performance['total_pnl'] >= daily_target:
            report += f"   ✅ Daily target achieved: ₹{performance['total_pnl']} ≥ ₹{daily_target}\n"
        else:
            remaining = daily_target - performance['total_pnl']
            report += f"   📊 Progress: ₹{performance['total_pnl']}/₹{daily_target} (₹{remaining} remaining)\n"
        
        return report
    
    def get_weekly_summary(self) -> str:
        """Generate weekly performance summary"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        conn = sqlite3.connect(self.db_path)
        
        # Get week's data (simplified since live trading doesn't have outcome/pnl yet)
        weekly_query = """
        SELECT DATE(timestamp) as date, 
               COUNT(*) as trades,
               AVG(confidence_score) as avg_confidence
        FROM trading_signals 
        WHERE DATE(timestamp) BETWEEN ? AND ?
        GROUP BY DATE(timestamp)
        ORDER BY date
        """
        
        weekly_df = pd.read_sql_query(
            weekly_query, 
            conn, 
            params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
        )
        
        conn.close()
        
        if weekly_df.empty:
            return "📊 No trading data available for the past week"
        
        total_trades = weekly_df['trades'].sum()
        avg_confidence = weekly_df['avg_confidence'].mean()
        
        summary = f"""
📅 WEEKLY PERFORMANCE SUMMARY
{'=' * 40}

📊 Total Trades: {total_trades}
🎯 Avg Confidence: {avg_confidence:.1f}
📈 Avg Daily Trades: {total_trades/7:.1f}

📆 DAILY BREAKDOWN:
"""
        
        for _, day in weekly_df.iterrows():
            summary += f"   {day['date']}: {day['trades']} trades | Conf: {day.get('avg_confidence', 0):.1f}\n"
        
        return summary
    
    def save_report(self, report: str, filename: str = None):
        """Save report to file with timestamp"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"reports/daily_report_{timestamp}.txt"
        
        os.makedirs("reports", exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"📁 Report saved to {filename}")

def generate_dashboard():
    """Generate and display current trading dashboard"""
    monitor = TradingMonitor()
    
    print("🚀 LIVE TRADING DASHBOARD")
    print("=" * 60)
    
    # Today's performance
    daily_report = monitor.generate_daily_report()
    print(daily_report)
    
    # Weekly summary
    weekly_summary = monitor.get_weekly_summary()
    print(weekly_summary)
    
    # Save today's report
    monitor.save_report(daily_report)
    
    return monitor

def setup_alerts():
    """Setup automated alert system (placeholder for cron/scheduler)"""
    print("\n🔔 ALERT SYSTEM SETUP")
    print("-" * 30)
    print("To setup automated alerts, add to crontab:")
    print("# Daily report at 4 PM")
    print("0 16 * * 1-5 cd /Users/kashishbaggafeast/Desktop/Bot && python3 monitoring_dashboard.py")
    print("\n# Hourly alerts during trading hours")  
    print("0 9-15 * * 1-5 cd /Users/kashishbaggafeast/Desktop/Bot && python3 -c 'from monitoring_dashboard import check_alerts; check_alerts()'")

def check_alerts():
    """Quick alert check for automated monitoring"""
    monitor = TradingMonitor()
    performance = monitor.get_daily_performance()
    alerts = monitor.check_risk_alerts(performance)
    
    if alerts:
        print("🚨 TRADING ALERTS:")
        for alert in alerts:
            print(f"  {alert}")
        
        # In production, send to Telegram/SMS/Email here
        return True
    return False

if __name__ == "__main__":
    # Generate dashboard
    monitor = generate_dashboard()
    
    # Show alert setup instructions
    setup_alerts()
    
    print(f"\n✅ Monitoring dashboard ready!")
    print(f"📊 Database: {monitor.db_path}")
    print(f"⚠️ Risk limits: {monitor.risk_limits}") 