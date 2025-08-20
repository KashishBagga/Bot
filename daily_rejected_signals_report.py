#!/usr/bin/env python3
"""
Daily Rejected Signals Report
Generates daily analysis of rejected signals with P&L impact
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.enhanced_rejected_signals import EnhancedRejectedSignals

def generate_daily_report():
    """Generate daily rejected signals report"""
    print(f"📊 DAILY REJECTED SIGNALS REPORT - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    enhanced_system = EnhancedRejectedSignals()
    
    # Get missed opportunities for today
    missed_report = enhanced_system.get_missed_opportunities_report(days=1, min_pnl=25.0)
    
    print(f"🎯 TODAY'S MISSED OPPORTUNITIES:")
    print(f"  Total Missed: {missed_report.get('total_missed_opportunities', 0)}")
    print(f"  Missed P&L: ₹{missed_report.get('total_missed_pnl', 0):.2f}")
    print(f"  Average P&L: ₹{missed_report.get('avg_missed_pnl', 0):.2f}")
    
    # Get rejection analysis
    rejection_analysis = enhanced_system.get_rejection_analysis(days=1)
    
    if rejection_analysis and 'overall_stats' in rejection_analysis:
        stats = rejection_analysis['overall_stats']
        print(f"\n📈 TODAY'S REJECTION ANALYSIS:")
        print(f"  Total Rejected: {stats.get('total_rejected', 0)}")
        print(f"  Would Have Won: {stats.get('would_have_won', 0)}")
        print(f"  Would Have Lost: {stats.get('would_have_lost', 0)}")
        print(f"  Rejection Efficiency: {stats.get('rejection_efficiency', 0):.1f}%")
        
        # Category breakdown
        print(f"\n🔍 REJECTION CATEGORIES:")
        for category in rejection_analysis.get('category_analysis', []):
            print(f"  • {category['category']}: {category['count']} signals, ₹{category['total_pnl']:.2f}")
    
    # Get top missed opportunities
    if missed_report.get('top_missed_trades'):
        print(f"\n🏆 TOP MISSED OPPORTUNITIES:")
        for i, trade in enumerate(missed_report['top_missed_trades'][:3], 1):
            strategy, symbol, signal, reason, category, price, confidence, pnl, targets, outcome, timestamp = trade
            print(f"  {i}. {strategy} - {symbol} - {signal}")
            print(f"     💰 Missed P&L: ₹{pnl:.2f} | Reason: {reason}")
    
    print(f"\n📋 RECOMMENDATIONS:")
    if missed_report.get('total_missed_pnl', 0) > 200:
        print(f"  🔴 High missed profits today - consider reviewing rejection criteria")
    elif missed_report.get('total_missed_pnl', 0) > 100:
        print(f"  🟡 Moderate missed profits - monitor patterns")
    else:
        print(f"  🟢 Low missed profits - rejection criteria working well")
    
    print(f"\n✅ Daily report completed at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    generate_daily_report()
