#!/usr/bin/env python3
"""
Database Views for Trading Analytics
Comprehensive views for analyzing trades, P&L, strategy performance, and optimization
"""

import sqlite3
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class DatabaseViews:
    """Database views for comprehensive trading analytics"""
    
    def __init__(self, db_path: str = "trading_signals.db"):
        self.db_path = db_path
    
    def create_all_views(self):
        """Create all database views for analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("🔧 Creating comprehensive database views...")
        
        # 1. Strategy Performance Summary View
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS strategy_performance_summary AS
        SELECT 
            strategy,
            symbol,
            COUNT(*) as total_trades,
            SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'Loss' THEN 1 ELSE 0 END) as losses,
            ROUND(AVG(CASE WHEN outcome = 'Win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(SUM(pnl), 2) as total_pnl,
            ROUND(AVG(pnl), 2) as avg_pnl,
            ROUND(AVG(confidence_score), 1) as avg_confidence,
            ROUND(MAX(pnl), 2) as max_profit,
            ROUND(MIN(pnl), 2) as max_loss,
            ROUND(AVG(targets_hit), 1) as avg_targets_hit,
            ROUND(AVG(stoploss_count), 1) as avg_stoploss_count,
            MIN(timestamp) as first_trade,
            MAX(timestamp) as last_trade
        FROM trades_backtest 
        GROUP BY strategy, symbol
        ORDER BY total_pnl DESC
        """)
        
        # 2. Daily Performance View
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS daily_performance AS
        SELECT 
            DATE(timestamp) as trade_date,
            strategy,
            symbol,
            COUNT(*) as trades,
            SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'Loss' THEN 1 ELSE 0 END) as losses,
            ROUND(AVG(CASE WHEN outcome = 'Win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(SUM(pnl), 2) as daily_pnl,
            ROUND(AVG(pnl), 2) as avg_pnl,
            ROUND(AVG(confidence_score), 1) as avg_confidence
        FROM trades_backtest 
        GROUP BY DATE(timestamp), strategy, symbol
        ORDER BY trade_date DESC, daily_pnl DESC
        """)
        
        # 3. Monthly Performance View
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS monthly_performance AS
        SELECT 
            strftime('%Y-%m', timestamp) as month,
            strategy,
            symbol,
            COUNT(*) as trades,
            SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'Loss' THEN 1 ELSE 0 END) as losses,
            ROUND(AVG(CASE WHEN outcome = 'Win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(SUM(pnl), 2) as monthly_pnl,
            ROUND(AVG(pnl), 2) as avg_pnl,
            ROUND(AVG(confidence_score), 1) as avg_confidence
        FROM trades_backtest 
        GROUP BY strftime('%Y-%m', timestamp), strategy, symbol
        ORDER BY month DESC, monthly_pnl DESC
        """)
        
        # 4. Confidence Analysis View
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS confidence_analysis AS
        SELECT 
            strategy,
            symbol,
            CASE 
                WHEN confidence_score >= 80 THEN 'Very High (80+)'
                WHEN confidence_score >= 70 THEN 'High (70-79)'
                WHEN confidence_score >= 60 THEN 'Medium (60-69)'
                WHEN confidence_score >= 50 THEN 'Low (50-59)'
                ELSE 'Very Low (<50)'
            END as confidence_level,
            COUNT(*) as trades,
            SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) as wins,
            ROUND(AVG(CASE WHEN outcome = 'Win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(SUM(pnl), 2) as total_pnl,
            ROUND(AVG(pnl), 2) as avg_pnl
        FROM trades_backtest 
        GROUP BY strategy, symbol, confidence_level
        ORDER BY strategy, symbol, total_pnl DESC
        """)
        
        # 5. Rejection Analysis View
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS rejection_analysis AS
        SELECT 
            strategy,
            symbol,
            rejection_reason,
            rejection_category,
            COUNT(*) as rejection_count,
            ROUND(AVG(confidence_score), 1) as avg_confidence,
            ROUND(AVG(rsi), 1) as avg_rsi,
            ROUND(AVG(macd), 2) as avg_macd,
            ROUND(AVG(atr), 2) as avg_atr,
            ROUND(AVG(ema_21), 2) as avg_ema21,
            ROUND(AVG(supertrend_direction), 1) as avg_supertrend
        FROM rejected_signals_backtest 
        GROUP BY strategy, symbol, rejection_reason, rejection_category
        ORDER BY rejection_count DESC
        """)
        
        # 6. Recent Trades View
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS recent_trades AS
        SELECT 
            timestamp,
            strategy,
            symbol,
            signal,
            price,
            stop_loss,
            target,
            target2,
            target3,
            reasoning,
            confidence,
            confidence_score,
            outcome,
            pnl,
            targets_hit,
            stoploss_count,
            exit_time,
            market_condition
        FROM trades_backtest 
        ORDER BY timestamp DESC
        """)
        
        # 7. Strategy Comparison View
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS strategy_comparison AS
        SELECT 
            strategy,
            COUNT(DISTINCT symbol) as symbols_traded,
            COUNT(*) as total_trades,
            ROUND(AVG(CASE WHEN outcome = 'Win' THEN 1.0 ELSE 0.0 END) * 100, 2) as overall_win_rate,
            ROUND(SUM(pnl), 2) as total_pnl,
            ROUND(AVG(pnl), 2) as avg_pnl,
            ROUND(AVG(confidence_score), 1) as avg_confidence,
            ROUND(MAX(pnl), 2) as max_profit,
            ROUND(MIN(pnl), 2) as max_loss,
            ROUND(AVG(targets_hit), 1) as avg_targets_hit,
            ROUND(AVG(stoploss_count), 1) as avg_stoploss_count
        FROM trades_backtest 
        GROUP BY strategy
        ORDER BY total_pnl DESC
        """)
        
        # 8. Symbol Performance View
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS symbol_performance AS
        SELECT 
            symbol,
            COUNT(DISTINCT strategy) as strategies_used,
            COUNT(*) as total_trades,
            ROUND(AVG(CASE WHEN outcome = 'Win' THEN 1.0 ELSE 0.0 END) * 100, 2) as overall_win_rate,
            ROUND(SUM(pnl), 2) as total_pnl,
            ROUND(AVG(pnl), 2) as avg_pnl,
            ROUND(AVG(confidence_score), 1) as avg_confidence,
            ROUND(MAX(pnl), 2) as max_profit,
            ROUND(MIN(pnl), 2) as max_loss
        FROM trades_backtest 
        GROUP BY symbol
        ORDER BY total_pnl DESC
        """)
        
        # 9. Market Condition Analysis View
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS market_condition_analysis AS
        SELECT 
            market_condition,
            strategy,
            symbol,
            COUNT(*) as trades,
            SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) as wins,
            ROUND(AVG(CASE WHEN outcome = 'Win' THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
            ROUND(SUM(pnl), 2) as total_pnl,
            ROUND(AVG(pnl), 2) as avg_pnl,
            ROUND(AVG(confidence_score), 1) as avg_confidence
        FROM trades_backtest 
        WHERE market_condition IS NOT NULL AND market_condition != 'Unknown'
        GROUP BY market_condition, strategy, symbol
        ORDER BY total_pnl DESC
        """)
        
        # 10. Risk Analysis View
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS risk_analysis AS
        SELECT 
            strategy,
            symbol,
            ROUND(AVG(stop_loss), 2) as avg_stop_loss,
            ROUND(AVG(target), 2) as avg_target,
            ROUND(AVG(target2), 2) as avg_target2,
            ROUND(AVG(target3), 2) as avg_target3,
            ROUND(AVG(targets_hit), 1) as avg_targets_hit,
            ROUND(AVG(stoploss_count), 1) as avg_stoploss_count,
            ROUND(COUNT(CASE WHEN stoploss_count > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as stoploss_rate,
            ROUND(COUNT(CASE WHEN targets_hit > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as target_hit_rate
        FROM trades_backtest 
        GROUP BY strategy, symbol
        ORDER BY avg_targets_hit DESC
        """)
        
        conn.commit()
        conn.close()
        
        print("✅ All database views created successfully!")
        print("📊 Available views:")
        print("  • strategy_performance_summary")
        print("  • daily_performance")
        print("  • monthly_performance")
        print("  • confidence_analysis")
        print("  • rejection_analysis")
        print("  • recent_trades")
        print("  • strategy_comparison")
        print("  • symbol_performance")
        print("  • market_condition_analysis")
        print("  • risk_analysis")
    
    def get_view_data(self, view_name: str, limit: int = 10) -> List[Dict]:
        """Get data from a specific view"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SELECT * FROM {view_name} LIMIT {limit}")
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                result.append(dict(zip(columns, row)))
            
            return result
        except Exception as e:
            print(f"Error fetching from view {view_name}: {e}")
            return []
        finally:
            conn.close()
    
    def print_view_summary(self, view_name: str, limit: int = 5):
        """Print summary of a specific view"""
        data = self.get_view_data(view_name, limit)
        
        if not data:
            print(f"❌ No data found in view: {view_name}")
            return
        
        print(f"\n📊 {view_name.upper()} SUMMARY")
        print("-" * 50)
        
        for i, row in enumerate(data, 1):
            print(f"{i}. {row}")
        
        print(f"\n📈 Total records in {view_name}: {len(data)}")

def main():
    """Create all database views"""
    views = DatabaseViews()
    views.create_all_views()
    
    # Print sample data from key views
    print("\n🔍 SAMPLE DATA FROM KEY VIEWS")
    print("=" * 60)
    
    views.print_view_summary("strategy_performance_summary", 3)
    views.print_view_summary("daily_performance", 3)
    views.print_view_summary("confidence_analysis", 3)
    views.print_view_summary("rejection_analysis", 3)

if __name__ == "__main__":
    main() 