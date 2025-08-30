#!/usr/bin/env python3
"""
Options Analytics Dashboard
Comprehensive dashboard for unified options data analytics and monitoring.
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.unified_database_enhanced import UnifiedDatabaseEnhanced

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('options_analytics_dashboard')

class OptionsAnalyticsDashboard:
    """Comprehensive analytics dashboard for options data."""
    
    def __init__(self):
        self.database = UnifiedDatabaseEnhanced()
    
    def get_system_overview(self) -> Dict:
        """Get comprehensive system overview."""
        try:
            summary = self.database.get_database_summary()
            
            # Calculate additional metrics
            total_records = sum([
                summary.get('raw_options_chain_count', 0),
                summary.get('options_data_count', 0),
                summary.get('market_summary_count', 0),
                summary.get('ohlc_candles_count', 0),
                summary.get('alerts_count', 0),
                summary.get('data_quality_log_count', 0),
                summary.get('greeks_analysis_count', 0),
                summary.get('volatility_surface_count', 0),
                summary.get('strategy_signals_count', 0),
                summary.get('performance_metrics_count', 0)
            ])
            
            # Calculate data freshness
            date_range = summary.get('date_range', {})
            if date_range.get('end'):
                try:
                    last_update = datetime.fromisoformat(date_range['end'].replace('Z', '+00:00'))
                    time_diff = datetime.now(timezone.utc) - last_update
                    data_freshness_minutes = int(time_diff.total_seconds() / 60)
                except:
                    data_freshness_minutes = None
            else:
                data_freshness_minutes = None
            
            overview = {
                'total_records': total_records,
                'symbols': summary.get('symbols', []),
                'avg_quality_score': summary.get('avg_quality_score', 0),
                'market_open_records': summary.get('market_open_records', 0),
                'unacknowledged_alerts': summary.get('unacknowledged_alerts', 0),
                'data_freshness_minutes': data_freshness_minutes,
                'date_range': date_range,
                'table_stats': {
                    'raw_options_chain': summary.get('raw_options_chain_count', 0),
                    'options_data': summary.get('options_data_count', 0),
                    'market_summary': summary.get('market_summary_count', 0),
                    'ohlc_candles': summary.get('ohlc_candles_count', 0),
                    'alerts': summary.get('alerts_count', 0),
                    'quality_log': summary.get('data_quality_log_count', 0),
                    'greeks_analysis': summary.get('greeks_analysis_count', 0),
                    'volatility_surface': summary.get('volatility_surface_count', 0),
                    'strategy_signals': summary.get('strategy_signals_count', 0),
                    'performance_metrics': summary.get('performance_metrics_count', 0)
                }
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting system overview: {e}")
            return {}
    
    def print_dashboard(self, detailed: bool = False):
        """Print comprehensive dashboard."""
        print("\n" + "="*80)
        print("🚀 ENHANCED UNIFIED OPTIONS ANALYTICS DASHBOARD")
        print("="*80)
        
        # System Overview
        overview = self.get_system_overview()
        if overview:
            print(f"\n📊 SYSTEM OVERVIEW")
            print("-" * 50)
            print(f"Total Records: {overview['total_records']:,}")
            print(f"Symbols: {', '.join(overview['symbols'])}")
            print(f"Average Quality Score: {overview['avg_quality_score']:.2f}")
            print(f"Market Open Records: {overview['market_open_records']:,}")
            print(f"Unacknowledged Alerts: {overview['unacknowledged_alerts']:,}")
            
            if overview['data_freshness_minutes'] is not None:
                print(f"Data Freshness: {overview['data_freshness_minutes']} minutes ago")
            
            print(f"\n📋 TABLE STATISTICS")
            print("-" * 30)
            for table, count in overview['table_stats'].items():
                print(f"  {table.replace('_', ' ').title()}: {count:,}")
        
        print("\n" + "="*80)
        print("✅ Dashboard generated successfully")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Options Analytics Dashboard')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed analytics for each symbol')
    
    args = parser.parse_args()
    
    try:
        dashboard = OptionsAnalyticsDashboard()
        dashboard.print_dashboard(detailed=args.detailed)
        
    except Exception as e:
        logger.error(f"❌ Dashboard error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 