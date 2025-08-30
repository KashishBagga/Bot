#!/usr/bin/env python3
"""
Daily Auto-Backtest Scheduler
Runs backtest automatically at EOD and generates reports
"""

import os
import sys
import time
import logging
import schedule
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import subprocess
import json

# Add src to path
sys.path.append('src')

from enhanced_backtest_system import EnhancedBacktestSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_backtest_scheduler.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DailyBacktestScheduler:
    """Daily backtest scheduler"""
    
    def __init__(self, 
                 symbols: list = None,
                 capital: float = 20000.0,
                 confidence: float = 0.6,
                 reports_dir: str = "daily_reports"):
        
        self.symbols = symbols or ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
        self.capital = capital
        self.confidence = confidence
        self.reports_dir = reports_dir
        self.tz = ZoneInfo("Asia/Kolkata")
        
        # Create reports directory
        os.makedirs(self.reports_dir, exist_ok=True)
        
        logger.info(f"✅ Daily Backtest Scheduler initialized")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(f"💰 Capital: ₹{capital:,.2f}")
        logger.info(f"📁 Reports Directory: {reports_dir}")
    
    def run_daily_backtest(self):
        """Run daily backtest for today's data"""
        
        today = datetime.now(self.tz).date()
        today_str = today.strftime('%Y-%m-%d')
        
        logger.info(f"🚀 Starting daily backtest for {today_str}")
        
        try:
            # Initialize backtest system
            backtest = EnhancedBacktestSystem(
                symbols=self.symbols,
                initial_capital=self.capital,
                confidence_cutoff=self.confidence
            )
            
            # Fetch today's data
            data = backtest.fetch_historical_data(today_str, today_str)
            
            if not data:
                logger.warning(f"⚠️ No data available for {today_str}")
                return
            
            # Run backtest
            results = backtest.run_backtest(data)
            
            # Generate report
            report_filename = f"daily_backtest_{today_str}.html"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            report_file = backtest.generate_report(results, report_path)
            
            # Save results as JSON for programmatic access
            json_filename = f"daily_backtest_{today_str}.json"
            json_path = os.path.join(self.reports_dir, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Print summary
            summary = results['summary']
            logger.info(f"✅ Daily backtest completed for {today_str}")
            logger.info(f"📊 Total Return: {summary['total_return_pct']:+.2f}%")
            logger.info(f"💰 Total P&L: ₹{summary['total_pnl']:+,.2f}")
            logger.info(f"🎯 Total Trades: {summary['total_trades']}")
            logger.info(f"📈 Win Rate: {summary['win_rate_pct']:.1f}%")
            logger.info(f"📄 Report: {report_file}")
            logger.info(f"📄 JSON: {json_path}")
            
            # Generate weekly summary if it's Friday
            if today.weekday() == 4:  # Friday
                self.generate_weekly_summary()
            
        except Exception as e:
            logger.error(f"❌ Error running daily backtest: {e}")
    
    def generate_weekly_summary(self):
        """Generate weekly summary report"""
        
        today = datetime.now(self.tz).date()
        week_start = today - timedelta(days=today.weekday())
        week_end = today
        
        logger.info(f"📊 Generating weekly summary for {week_start} to {week_end}")
        
        weekly_results = {
            'period': f"{week_start} to {week_end}",
            'daily_results': [],
            'summary': {
                'total_days': 0,
                'profitable_days': 0,
                'total_pnl': 0.0,
                'total_return_pct': 0.0,
                'total_trades': 0,
                'avg_win_rate': 0.0
            }
        }
        
        # Collect daily results
        for i in range(5):  # Monday to Friday
            date = week_start + timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            
            json_file = os.path.join(self.reports_dir, f"daily_backtest_{date_str}.json")
            
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        daily_result = json.load(f)
                    
                    weekly_results['daily_results'].append({
                        'date': date_str,
                        'summary': daily_result['summary']
                    })
                    
                    # Update summary
                    weekly_results['summary']['total_days'] += 1
                    weekly_results['summary']['total_pnl'] += daily_result['summary']['total_pnl']
                    weekly_results['summary']['total_trades'] += daily_result['summary']['total_trades']
                    
                    if daily_result['summary']['total_pnl'] > 0:
                        weekly_results['summary']['profitable_days'] += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error reading {json_file}: {e}")
        
        # Calculate averages
        if weekly_results['summary']['total_days'] > 0:
            weekly_results['summary']['total_return_pct'] = (
                weekly_results['summary']['total_pnl'] / self.capital * 100
            )
            weekly_results['summary']['avg_win_rate'] = (
                sum(r['summary']['win_rate_pct'] for r in weekly_results['daily_results']) / 
                len(weekly_results['daily_results'])
            )
        
        # Save weekly summary
        weekly_filename = f"weekly_summary_{week_start}_{week_end}.json"
        weekly_path = os.path.join(self.reports_dir, weekly_filename)
        
        with open(weekly_path, 'w') as f:
            json.dump(weekly_results, f, indent=2)
        
        logger.info(f"📊 Weekly summary saved: {weekly_path}")
        
        # Generate weekly HTML report
        self.generate_weekly_html_report(weekly_results, weekly_path.replace('.json', '.html'))
    
    def generate_weekly_html_report(self, weekly_results: dict, output_file: str):
        """Generate weekly HTML report"""
        
        summary = weekly_results['summary']
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Weekly Trading Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .daily {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Weekly Trading Summary</h1>
                <p>Period: {weekly_results['period']}</p>
                <p>Generated on: {datetime.now(self.tz).strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>Weekly Summary</h2>
                <table>
                    <tr><td>Trading Days</td><td>{summary['total_days']}</td></tr>
                    <tr><td>Profitable Days</td><td>{summary['profitable_days']}</td></tr>
                    <tr><td>Total P&L</td><td class="{'positive' if summary['total_pnl'] > 0 else 'negative'}">₹{summary['total_pnl']:+,.2f}</td></tr>
                    <tr><td>Total Return</td><td class="{'positive' if summary['total_return_pct'] > 0 else 'negative'}">{summary['total_return_pct']:+.2f}%</td></tr>
                    <tr><td>Total Trades</td><td>{summary['total_trades']}</td></tr>
                    <tr><td>Average Win Rate</td><td>{summary['avg_win_rate']:.1f}%</td></tr>
                </table>
            </div>
            
            <div class="daily">
                <h2>Daily Breakdown</h2>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>P&L</th>
                        <th>Return</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                    </tr>
        """
        
        for daily in weekly_results['daily_results']:
            daily_summary = daily['summary']
            html_content += f"""
                    <tr>
                        <td>{daily['date']}</td>
                        <td class="{'positive' if daily_summary['total_pnl'] > 0 else 'negative'}">₹{daily_summary['total_pnl']:+,.2f}</td>
                        <td class="{'positive' if daily_summary['total_return_pct'] > 0 else 'negative'}">{daily_summary['total_return_pct']:+.2f}%</td>
                        <td>{daily_summary['total_trades']}</td>
                        <td>{daily_summary['win_rate_pct']:.1f}%</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📄 Weekly HTML report generated: {output_file}")
    
    def schedule_backtest(self, time_str: str = "15:45"):
        """Schedule daily backtest"""
        
        logger.info(f"⏰ Scheduling daily backtest at {time_str}")
        
        # Schedule daily backtest
        schedule.every().day.at(time_str).do(self.run_daily_backtest)
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def run_manual_backtest(self, date_str: str = None):
        """Run manual backtest for a specific date"""
        
        if date_str is None:
            date_str = datetime.now(self.tz).strftime('%Y-%m-%d')
        
        logger.info(f"🔧 Running manual backtest for {date_str}")
        self.run_daily_backtest()

def main():
    parser = argparse.ArgumentParser(description='Daily Backtest Scheduler')
    parser.add_argument('--symbols', nargs='+', default=['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX'],
                       help='Trading symbols')
    parser.add_argument('--capital', type=float, default=20000.0, help='Initial capital')
    parser.add_argument('--confidence', type=float, default=0.6, help='Confidence cutoff')
    parser.add_argument('--reports-dir', default='daily_reports', help='Reports directory')
    parser.add_argument('--schedule', action='store_true', help='Run as scheduler')
    parser.add_argument('--time', default='15:45', help='Schedule time (HH:MM)')
    parser.add_argument('--manual', help='Run manual backtest for specific date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = DailyBacktestScheduler(
        symbols=args.symbols,
        capital=args.capital,
        confidence=args.confidence,
        reports_dir=args.reports_dir
    )
    
    if args.manual:
        # Run manual backtest
        scheduler.run_manual_backtest(args.manual)
    elif args.schedule:
        # Run as scheduler
        scheduler.schedule_backtest(args.time)
    else:
        # Run today's backtest
        scheduler.run_daily_backtest()

if __name__ == "__main__":
    main() 